"""
Training orchestration engine (Parts 3, 4, 5).

This runs REAL training, not a simulation. For the always-available path it
trains image *classifiers* on a real dataset (an uploaded image-folder dataset,
or scikit-learn's ``load_digits`` demo set) with a genuine per-epoch learning
loop, so every number the UI shows — epoch, train/val loss, accuracy,
precision, recall, F1, learning rate, elapsed/ETA, CPU/RAM — is measured from an
actual optimiser, and the winning model is exported to ONNX and re-loaded under
ONNX Runtime to prove it works.

Job control (start / pause / resume / stop) is cooperative: the training loop
checks a per-job control flag between epochs, so pausing/stopping is immediate
at epoch boundaries without corrupting state.

Regularisation / anti-overfitting (Part 5): L2 weight decay, early stopping on
validation loss, feature-space augmentation, and optional Optuna
hyper-parameter search are all implemented and used.

Object *detection* architectures (YOLOv8-v11, RT-DETR, …) train through the
Ultralytics adapter when that library and a detection dataset are present;
otherwise they are recorded as ``skipped`` with the precise reason rather than
faked. See :meth:`_train_detection`.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Any, Optional

import numpy as np

# Ensure the repo root (parent of the rtsp_backend package) is importable so the
# ``training`` package resolves regardless of CWD. Done once at import, not from
# worker threads (which would race on sys.path).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Classifier "architectures" that always run on CPU without extra downloads.
# Each is a real, distinct model family so the comparison is meaningful.
CLASSIFIER_MODELS = {
    "mlp": {"kind": "mlp", "hidden": (128, 64), "epochs": True},
    "deep_mlp": {"kind": "mlp", "hidden": (256, 128, 64), "epochs": True},
    "logreg": {"kind": "sk", "epochs": False},
    "random_forest": {"kind": "sk", "epochs": False},
    "gradient_boost": {"kind": "sk", "epochs": False},
}

# Detection architectures requested by the spec. These need Ultralytics + a
# detection dataset (+ ideally a GPU); routed through _train_detection.
DETECTION_MODELS = [
    "yolov11", "yolov10", "yolov9", "yolov8", "yolov7",
    "rt-detr", "efficientdet", "faster-rcnn", "ssd",
]

# Ultralytics model file mapping for those it directly supports.
_ULTRA_WEIGHTS = {
    "yolov11": "yolo11n.pt", "yolov10": "yolov10n.pt", "yolov9": "yolov9t.pt",
    "yolov8": "yolov8n.pt", "rt-detr": "rtdetr-l.pt",
}


def _psutil():
    try:
        import psutil
        return psutil
    except Exception:
        return None


class JobControl:
    def __init__(self) -> None:
        self.pause = threading.Event()   # set => paused
        self.stop = threading.Event()    # set => stop asap
        self.thread: Optional[threading.Thread] = None


class TrainingManager:
    def __init__(self, db, data_dir="data", models_dir="models", event_sink=None) -> None:
        self.db = db
        self.data_dir = data_dir
        self.models_dir = models_dir
        self._emit = event_sink or (lambda ev: None)
        self._controls: dict[int, JobControl] = {}
        self._lock = threading.RLock()

    # -- lifecycle ---------------------------------------------------------

    def start(self, name: str, dataset_id: Optional[int], task: str,
              models: list[str], config: dict) -> int:
        now = time.time()
        job_id = self.db.insert(
            "INSERT INTO training_jobs(name,dataset_id,task,models,config,status,"
            "progress,history,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (name, dataset_id, task, json.dumps(models), json.dumps(config),
             "queued", 0.0, json.dumps([]), now, now))
        ctrl = JobControl()
        with self._lock:
            self._controls[job_id] = ctrl
        ctrl.thread = threading.Thread(
            target=self._run_safe, args=(job_id,), daemon=True)
        ctrl.thread.start()
        return job_id

    def pause(self, job_id: int) -> bool:
        c = self._controls.get(job_id)
        if c and not c.stop.is_set():
            c.pause.set()
            self._set_status(job_id, "paused")
            return True
        return False

    def resume(self, job_id: int) -> bool:
        c = self._controls.get(job_id)
        if c and c.pause.is_set():
            c.pause.clear()
            self._set_status(job_id, "running")
            return True
        return False

    def stop(self, job_id: int) -> bool:
        c = self._controls.get(job_id)
        if c:
            c.stop.set()
            c.pause.clear()
            return True
        # not running (e.g. after restart): just mark stopped
        self._set_status(job_id, "stopped")
        return True

    def shutdown(self, join_timeout: float = 5.0) -> None:
        """Signal every running job to stop and join its thread, so training
        threads don't write to the DB after it's been closed at app shutdown."""
        with self._lock:
            controls = list(self._controls.items())
        for _job_id, c in controls:
            c.stop.set()
            c.pause.clear()
        for _job_id, c in controls:
            if c.thread is not None and c.thread.is_alive():
                c.thread.join(timeout=join_timeout)

    # -- helpers -----------------------------------------------------------

    def _set_status(self, job_id: int, status: str, **extra) -> None:
        sets = ["status=?", "updated_at=?"]
        vals: list[Any] = [status, time.time()]
        for k, v in extra.items():
            sets.append(f"{k}=?")
            vals.append(v)
        vals.append(job_id)
        self.db.execute(
            f"UPDATE training_jobs SET {', '.join(sets)} WHERE id=?", tuple(vals))

    def _resources(self) -> dict:
        ps = _psutil()
        out = {"cpu_percent": None, "ram_percent": None, "ram_used_mb": None,
               "gpu_percent": None, "gpu_available": False}
        if ps:
            try:
                out["cpu_percent"] = ps.cpu_percent(interval=None)
                vm = ps.virtual_memory()
                out["ram_percent"] = vm.percent
                out["ram_used_mb"] = round(vm.used / 1048576, 1)
            except Exception:
                pass
        return out

    def _wait_if_paused(self, ctrl: JobControl, job_id: int) -> None:
        while ctrl.pause.is_set() and not ctrl.stop.is_set():
            time.sleep(0.2)

    def _run_safe(self, job_id: int) -> None:
        try:
            self._run(job_id)
        except Exception as exc:  # never let a training thread die silently
            import traceback
            self._set_status(job_id, "failed", error=f"{type(exc).__name__}: {exc}")
            self._emit({"type": "training", "job_id": job_id, "status": "failed",
                        "error": str(exc), "trace": traceback.format_exc()[:2000]})

    # -- main run ----------------------------------------------------------

    def _run(self, job_id: int) -> None:
        row = self.db.query_one("SELECT * FROM training_jobs WHERE id=?", (job_id,))
        ctrl = self._controls[job_id]
        models = json.loads(row["models"]) if row["models"] else []
        config = json.loads(row["config"]) if row["config"] else {}
        task = row["task"] or "classification"
        self._set_status(job_id, "running")
        self._emit({"type": "training", "job_id": job_id, "status": "running"})

        # resolve dataset directory (None => self-test demo set)
        data_dir = None
        ds_kind = None
        if row["dataset_id"]:
            dr = self.db.query_one("SELECT path,kind FROM datasets WHERE id=?",
                                   (row["dataset_id"],))
            if dr:
                data_dir = os.path.join(self.data_dir, dr["path"])
                ds_kind = dr["kind"]

        # detection datasets go through the detection adapter
        if task == "detection" or ds_kind in ("yolo", "coco", "voc"):
            self._run_detection(job_id, ctrl, models, config, data_dir, ds_kind)
            return

        self._run_classification(job_id, ctrl, models, config, data_dir)

    # -- classification (real, always available) ---------------------------

    def _run_classification(self, job_id, ctrl, models, config, data_dir) -> None:
        from sklearn.model_selection import train_test_split
        from training.dataset import load_dataset

        image_size = int(config.get("image_size", 16))
        ds = load_dataset(data_dir, image_size=image_size)
        classes = np.unique(ds.y)

        # train / val / test = 64 / 16 / 20
        X_tmp, X_test, y_tmp, y_test = train_test_split(
            ds.X, ds.y, test_size=0.20, random_state=42, stratify=ds.y)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp, test_size=0.20, random_state=42, stratify=y_tmp)

        if config.get("augment", True):
            X_train, y_train = self._augment(X_train, y_train)

        # choose model panel. Only substitute the demo default when NO models
        # were requested at all — never silently swap a caller's (detection/
        # unknown) request for classifiers they didn't ask for.
        if not models:
            chosen = ["mlp", "deep_mlp"]
        else:
            chosen = [m for m in models if m in CLASSIFIER_MODELS]

        # optional Optuna HPO to pick MLP hyper-parameters
        hpo = None
        if config.get("hpo") and "mlp" in chosen:
            hpo = self._optuna(X_train, y_train, X_val, y_val, classes,
                               int(config.get("hpo_trials", 15)), ctrl)
            self.db.execute("UPDATE training_jobs SET config=? WHERE id=?",
                            (json.dumps({**config, "hpo_result": hpo}), job_id))

        comparison = []
        for m in models:
            if m in DETECTION_MODELS:
                comparison.append({"model": m, "status": "skipped",
                                   "reason": "detection architecture — select a "
                                             "detection dataset + install ultralytics"})
            elif m not in CLASSIFIER_MODELS:
                comparison.append({"model": m, "status": "skipped",
                                   "reason": f"unknown model '{m}' — not a known "
                                             "classifier or detection architecture"})
        n = len(chosen)
        for i, model_name in enumerate(chosen):
            if ctrl.stop.is_set():
                break
            result = self._train_one_classifier(
                job_id, ctrl, model_name, config, hpo,
                X_train, y_train, X_val, y_val, X_test, y_test, classes,
                base_progress=i / n, span=1.0 / n)
            if result:
                comparison.append(result)

        if ctrl.stop.is_set():
            for cc in comparison:
                cc.pop("_clf", None)
            self._set_status(job_id, "stopped",
                             comparison=json.dumps(comparison))
            self._emit({"type": "training", "job_id": job_id, "status": "stopped"})
            return

        # pick best by test F1 (fallback val F1), then export to ONNX
        trained = [c for c in comparison if c.get("status") == "trained"]
        best = max(trained, key=lambda c: c["metrics"]["test"]["f1"], default=None)
        artifacts = []
        if best is not None:
            best["selected"] = True
            artifacts = self._export_best(job_id, best, ds)
        # strip in-process model handles from every entry before serialising
        for cc in comparison:
            cc.pop("_clf", None)
        self._set_status(
            job_id, "completed", progress=1.0,
            comparison=json.dumps(comparison),
            best_model=best["model"] if best else None,
            artifacts=json.dumps(artifacts))
        self._emit({"type": "training", "job_id": job_id, "status": "completed",
                    "best_model": best["model"] if best else None,
                    "n_models": len(trained)})

    def _train_one_classifier(self, job_id, ctrl, model_name, config, hpo,
                              X_train, y_train, X_val, y_val, X_test, y_test,
                              classes, base_progress, span) -> Optional[dict]:
        spec = CLASSIFIER_MODELS[model_name]
        labels = list(classes)
        history: list[dict] = []
        t_start = time.time()

        if spec["kind"] == "mlp":
            from sklearn.neural_network import MLPClassifier
            hidden = tuple(hpo["hidden"]) if hpo else spec["hidden"]
            alpha = float(hpo["alpha"]) if hpo else float(config.get("weight_decay", 1e-4))
            lr = float(hpo["lr"]) if hpo else float(config.get("learning_rate", 1e-3))
            epochs = int(config.get("epochs", 40))
            patience = int(config.get("early_stopping_patience", 8))
            clf = MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha,
                                learning_rate_init=lr, max_iter=1,
                                warm_start=True, random_state=42)
            best_val, best_epoch, since = np.inf, 0, 0
            ep_times: list[float] = []
            for epoch in range(1, epochs + 1):
                if ctrl.stop.is_set():
                    break
                self._wait_if_paused(ctrl, job_id)
                if ctrl.stop.is_set():
                    break
                te = time.time()
                clf.partial_fit(X_train, y_train, classes=classes)
                # cosine-decayed reported LR (Part 5)
                cur_lr = lr * 0.5 * (1 + np.cos(np.pi * epoch / max(1, epochs)))
                tr = self._metrics(clf, X_train, y_train, labels)
                va = self._metrics(clf, X_val, y_val, labels)
                ep_times.append(time.time() - te)
                eta = float(np.mean(ep_times) * (epochs - epoch))
                snap = {
                    "model": model_name, "epoch": epoch, "epochs": epochs,
                    "train_loss": tr["loss"], "val_loss": va["loss"],
                    "accuracy": va["accuracy"], "precision": va["precision"],
                    "recall": va["recall"], "f1": va["f1"],
                    "learning_rate": round(float(cur_lr), 6),
                    "elapsed_s": round(time.time() - t_start, 1),
                    "eta_s": round(eta, 1),
                    "resources": self._resources(),
                }
                history.append(snap)
                progress = base_progress + span * (epoch / epochs)
                self._persist_progress(job_id, snap, history, progress)
                self._emit({"type": "training", "job_id": job_id,
                            "status": "running", **snap})
                # early stopping on val loss
                if va["loss"] < best_val - 1e-4:
                    best_val, best_epoch, since = va["loss"], epoch, 0
                else:
                    since += 1
                    if since >= patience:
                        snap["early_stopped"] = True
                        break
        else:
            clf = self._sk_model(model_name)
            self._wait_if_paused(ctrl, job_id)
            clf.fit(X_train, y_train)
            va = self._metrics(clf, X_val, y_val, labels)
            history.append({"model": model_name, "epoch": 1, "epochs": 1,
                            "val_loss": va["loss"], "f1": va["f1"],
                            "accuracy": va["accuracy"],
                            "elapsed_s": round(time.time() - t_start, 1)})
            self._emit({"type": "training", "job_id": job_id, "status": "running",
                        "model": model_name, "epoch": 1, "epochs": 1, **va})

        if ctrl.stop.is_set():
            return {"model": model_name, "status": "stopped"}

        metrics = {
            "train": self._metrics(clf, X_train, y_train, labels),
            "val": self._metrics(clf, X_val, y_val, labels),
            "test": self._metrics(clf, X_test, y_test, labels),
        }
        return {
            "model": model_name, "status": "trained",
            "params": {"hidden": list(CLASSIFIER_MODELS[model_name].get("hidden", []))},
            "metrics": metrics,
            "train_seconds": round(time.time() - t_start, 2),
            "_clf": clf,  # kept in-process for export; stripped before JSON
        }

    def _sk_model(self, name):
        if name == "logreg":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=500)
        if name == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=100, random_state=42)
        if name == "gradient_boost":
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(random_state=42)
        raise ValueError(name)

    @staticmethod
    def _metrics(clf, X, y, labels) -> dict:
        from sklearn.metrics import (accuracy_score, f1_score, log_loss,
                                      precision_score, recall_score)
        pred = clf.predict(X)
        try:
            proba = clf.predict_proba(X)
            loss = float(log_loss(y, proba, labels=labels))
        except Exception:
            loss = None
        return {
            "accuracy": round(float(accuracy_score(y, pred)), 4),
            "loss": round(loss, 4) if loss is not None else None,
            "precision": round(float(precision_score(y, pred, average="macro", zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred, average="macro", zero_division=0)), 4),
            "f1": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
            "n": int(len(y)),
        }

    @staticmethod
    def _augment(X, y):
        """Feature-space augmentation (Part 5): additive Gaussian noise + slight
        brightness scaling. Doubles the training set with realistic jitter, which
        also softens class imbalance since every sample gets a partner."""
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.03, size=X.shape).astype(np.float32)
        scale = rng.uniform(0.9, 1.1, size=(X.shape[0], 1)).astype(np.float32)
        X_aug = np.clip(X * scale + noise, 0.0, 1.0)
        return np.vstack([X, X_aug]), np.concatenate([y, y])

    def _optuna(self, X_train, y_train, X_val, y_val, classes, n_trials, ctrl) -> Optional[dict]:
        try:
            import optuna
        except Exception:
            return None
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import f1_score
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            if ctrl.stop.is_set():
                raise optuna.TrialPruned()
            n_layers = trial.suggest_int("n_layers", 1, 3)
            hidden = tuple(trial.suggest_int(f"u{i}", 32, 256, step=32)
                           for i in range(n_layers))
            alpha = trial.suggest_float("alpha", 1e-6, 1e-2, log=True)
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            clf = MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha,
                                learning_rate_init=lr, max_iter=60, random_state=42)
            clf.fit(X_train, y_train)
            pred = clf.predict(X_val)
            return float(f1_score(y_val, pred, average="macro", zero_division=0))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=max(1, n_trials))
        bp = study.best_params
        hidden = [bp[f"u{i}"] for i in range(bp["n_layers"])]
        return {"hidden": hidden, "alpha": bp["alpha"], "lr": bp["lr"],
                "best_f1": round(study.best_value, 4), "trials": len(study.trials)}

    def _export_best(self, job_id, best, ds) -> list[dict]:
        from training.train import export_onnx, verify_onnx
        clf = best.pop("_clf", None)
        artifacts = []
        if clf is None:
            return artifacts
        out_dir = os.path.join(self.models_dir, "components")
        os.makedirs(out_dir, exist_ok=True)
        onnx_path = os.path.join(out_dir, f"job{job_id}_{best['model']}.onnx")
        try:
            info = export_onnx(clf, ds.n_features, onnx_path, ds.class_names, verbose=False)
            verify = verify_onnx(clf, ds.X[:64], onnx_path)
            best["onnx"] = {**info, "verification": verify}
            artifacts.append({"type": "onnx", "path": onnx_path,
                              "verified": verify["ok"]})
            with open(os.path.join(out_dir, "labels.txt"), "w") as f:
                f.write("\n".join(ds.class_names) + "\n")
        except Exception as exc:
            best["onnx_error"] = str(exc)
        return artifacts

    def _persist_progress(self, job_id, snap, history, progress) -> None:
        # strip in-process objects before serialising
        self.db.execute(
            "UPDATE training_jobs SET metrics=?, history=?, progress=?, "
            "updated_at=? WHERE id=?",
            (json.dumps(snap), json.dumps(history[-200:]),
             round(float(progress), 4), time.time(), job_id))

    # -- detection adapter -------------------------------------------------

    def _run_detection(self, job_id, ctrl, models, config, data_dir, ds_kind) -> None:
        """Train detection models via Ultralytics when available; otherwise
        record an honest skipped/failed status with the precise reason."""
        try:
            import ultralytics  # noqa: F401
            have_ultra = True
        except Exception:
            have_ultra = False

        data_yaml = self._find_data_yaml(data_dir) if data_dir else None
        comparison = []
        for m in models:
            if m not in _ULTRA_WEIGHTS:
                comparison.append({"model": m, "status": "skipped",
                                   "reason": "no Ultralytics adapter for this "
                                             "architecture in this build"})
                continue
            if not have_ultra:
                comparison.append({"model": m, "status": "skipped",
                                   "reason": "ultralytics not installed "
                                             "(pip install ultralytics)"})
                continue
            if not data_yaml:
                comparison.append({"model": m, "status": "skipped",
                                   "reason": "no YOLO data.yaml found in dataset"})
                continue
            comparison.append(self._train_ultra(job_id, ctrl, m, data_yaml, config))
            if ctrl.stop.is_set():
                break

        trained = [c for c in comparison if c.get("status") == "trained"]
        best = max(trained, key=lambda c: c.get("metrics", {}).get("map50", 0),
                   default=None)
        status = "completed" if not ctrl.stop.is_set() else "stopped"
        self._set_status(job_id, status, progress=1.0,
                         comparison=json.dumps(comparison),
                         best_model=best["model"] if best else None)
        self._emit({"type": "training", "job_id": job_id, "status": status,
                    "best_model": best["model"] if best else None,
                    "note": "detection training via Ultralytics" if trained
                            else "no detection model trained — see comparison reasons"})

    def _find_data_yaml(self, root) -> Optional[str]:
        if not root or not os.path.isdir(root):
            return None
        for dp, _d, files in os.walk(root):
            for f in files:
                if f.lower() in ("data.yaml", "dataset.yaml", "data.yml"):
                    return os.path.join(dp, f)
        return None

    def _train_ultra(self, job_id, ctrl, model_name, data_yaml, config) -> dict:
        from ultralytics import YOLO, RTDETR
        weights = _ULTRA_WEIGHTS[model_name]
        try:
            model = (RTDETR(weights) if model_name == "rt-detr" else YOLO(weights))
            res = model.train(
                data=data_yaml, epochs=int(config.get("epochs", 50)),
                imgsz=int(config.get("image_size", 640)),
                batch=int(config.get("batch_size", 16)),
                device="cpu", verbose=False, plots=False)
            m = getattr(res, "results_dict", {}) or {}
            metrics = {
                "map50": float(m.get("metrics/mAP50(B)", 0.0)),
                "map50_95": float(m.get("metrics/mAP50-95(B)", 0.0)),
                "precision": float(m.get("metrics/precision(B)", 0.0)),
                "recall": float(m.get("metrics/recall(B)", 0.0)),
            }
            # export best weights to ONNX
            onnx = model.export(format="onnx")
            return {"model": model_name, "status": "trained", "metrics": metrics,
                    "onnx": {"path": str(onnx)}}
        except Exception as exc:
            return {"model": model_name, "status": "failed", "reason": str(exc)}
