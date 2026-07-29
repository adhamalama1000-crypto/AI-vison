"""
Training and benchmarking for the industrial component detector.

Model selection is a measurement, not an opinion. :func:`benchmark` trains every
requested architecture on the same split with the same budget and returns a
ranked table from :mod:`rtsp_backend.electrical.metrics`, so "we chose YOLOv11"
is a statement with numbers behind it.

Architectures
-------------
``yolo11{n,s,m,l,x}``
    Current Ultralytics detector generation. Best accuracy-per-millisecond for
    this problem and the default recommendation.
``yolov8{n,s,m,l,x}``
    Previous generation. Included because it is the most widely reproduced
    baseline; useful as a control.
``rtdetr-{l,x}``
    Transformer detector, NMS-free. Tends to win on densely packed scenes, which
    a DIN rail full of adjacent modular devices very much is — worth measuring
    rather than assuming.
``yolo12{n,s,m}``
    Attempted only if the installed Ultralytics build exposes it; reported as
    ``skipped`` otherwise rather than silently substituted. (The prompt asks for
    YOLOv12 "if stable" — this is that condition, enforced in code.)

Open-vocabulary models (OWLv2, Grounding DINO, Florence-2) are **not** trained
here; they are used zero-shot at inference through
:mod:`rtsp_backend.electrical.recognizer`. :func:`evaluate_zero_shot` scores them
on the same validation split so a trained model can be compared against the
zero-shot baseline on equal terms — which is the only way to know whether
training was worth it.

Nothing in this module fabricates a result. Without ``ultralytics`` installed,
every training call reports ``skipped`` with the reason.
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

from rtsp_backend.electrical import metrics as em
from rtsp_backend.electrical import taxonomy as tax

#: Architectures with a real adapter here. Anything else is reported as skipped.
SUPPORTED_ARCHS: tuple[str, ...] = (
    "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
    "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
    "rtdetr-l", "rtdetr-x",
    "yolo12n", "yolo12s", "yolo12m",
)

#: Architectures that only exist in newer Ultralytics builds.
CONDITIONAL_ARCHS: frozenset[str] = frozenset({"yolo12n", "yolo12s", "yolo12m"})

DEFAULT_ARCH = "yolo11s"


@contextlib.contextmanager
def quiet_stdout():
    """Route a library's output to stderr for the duration of a block.

    Ultralytics writes its banner, per-epoch table and export summary to stdout.
    Every CLI subcommand here contracts to emit **JSON on stdout** so it can be piped
    (``cli eval > eval.json``, then ``cli export --eval-json``), and that chatter
    silently corrupts the JSON — which is exactly how the documented export pipe was
    found to be broken.

    Two mechanisms are needed, and the second is the one that actually matters:

    1. ``redirect_stdout`` catches plain ``print`` calls.
    2. Ultralytics emits most of its output through ``ultralytics.utils.LOGGER``,
       a :mod:`logging` logger whose ``StreamHandler`` captured the *real*
       ``sys.stdout`` at import time. ``redirect_stdout`` cannot reach that, because
       the handler holds a direct reference to the original stream. Those handlers
       are repointed at stderr and restored afterwards.

    Nothing is suppressed — progress stays visible on stderr, where every other
    human-facing message in this package already goes.
    """
    import logging

    restore: list[tuple] = []
    try:
        from ultralytics.utils import LOGGER  # type: ignore

        loggers = [LOGGER]
    except Exception:
        loggers = []
    # The root logger can also carry a stdout handler installed by a dependency.
    loggers.append(logging.getLogger())

    for logger in loggers:
        for handler in list(getattr(logger, "handlers", [])):
            stream = getattr(handler, "stream", None)
            if stream is not None and stream is not sys.stderr:
                restore.append((handler, stream))
                handler.stream = sys.stderr
    try:
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        for handler, stream in restore:
            handler.stream = stream


class TrainingAborted(RuntimeError):
    """Raised by a training callback to stop a run early, on purpose.

    :func:`train` catches every other exception and reports it as a failed result,
    because one broken architecture must not abort a benchmark of six. This
    exception is the documented exception to that rule: it propagates, so a caller
    that deliberately stops a run — hyperparameter pruning, a user cancel — is not
    told its run "failed".
    """

    def __init__(self, reason: str, value: Optional[float] = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.value = value


def ultralytics_available() -> tuple[bool, Optional[str]]:
    try:
        import ultralytics  # noqa: F401
        return True, getattr(ultralytics, "__version__", None)
    except Exception:
        return False, None


def arch_available(arch: str) -> bool:
    """Whether Ultralytics can actually construct this architecture."""
    ok, _ = ultralytics_available()
    if not ok or arch not in SUPPORTED_ARCHS:
        return False
    if arch not in CONDITIONAL_ARCHS:
        return True
    try:
        from ultralytics.cfg import get_cfg  # noqa: F401
        from ultralytics import YOLO
        YOLO(f"{arch}.yaml")     # config-only construction, no weight download
        return True
    except Exception:
        return False


@dataclass
class TrainConfig:
    """Industrial-detection training recipe.

    Defaults are tuned for panel imagery rather than copied from COCO:

    * ``imgsz=960`` — modular devices are small relative to a cabinet photograph;
      at 640 an MCB in a wide shot is a handful of pixels.
    * ``degrees=8`` / ``fliplr=0.0`` — panels are gravity-oriented and devices
      carry directional markings, so mild rotation helps and horizontal mirroring
      teaches the model wrong geometry (a mirrored nameplate is not a real thing).
    * ``mosaic`` on, ``close_mosaic=10`` — mosaic helps small-object recall but
      destroys the row/layout context the model should learn, so it is turned off
      for the final epochs.
    * strong ``hsv_v`` / moderate ``hsv_s`` — lighting varies enormously in the
      field; device colour is a real class signal and must not be destroyed.
    """

    data: str
    arch: str = DEFAULT_ARCH
    epochs: int = 120
    imgsz: int = 960
    batch: int = 8
    device: str = "cpu"
    workers: int = 4
    patience: int = 25
    seed: int = 0
    project: str = "runs/electrical"
    name: Optional[str] = None
    pretrained: bool = True
    #: Fine-tune from this checkpoint instead of the architecture's COCO weights.
    #: Used for synthetic → real domain transfer and for staged class expansion.
    #: When the checkpoint's head has a different class count from ``data``,
    #: Ultralytics keeps the backbone and reinitialises the head — which is the
    #: intended behaviour for expanding a profile, and is reported as such.
    init_from: Optional[str] = None
    #: Freeze the first N layers. Standard staged transfer learning: freeze the
    #: backbone while the head adapts to the new domain, then unfreeze and train
    #: end to end at a lower learning rate. 10 freezes the YOLO backbone; 0 is
    #: full fine-tuning.
    freeze: Optional[int] = None
    optimizer: str = "auto"
    lr0: float = 0.01
    cos_lr: bool = True
    warmup_epochs: float = 3.0
    # augmentation
    degrees: float = 8.0
    translate: float = 0.10
    scale: float = 0.45
    shear: float = 3.0
    perspective: float = 0.0008
    fliplr: float = 0.0
    flipud: float = 0.0
    hsv_h: float = 0.012
    hsv_s: float = 0.55
    hsv_v: float = 0.50
    mosaic: float = 0.8
    mixup: float = 0.10
    copy_paste: float = 0.10
    close_mosaic: int = 10
    extra: dict[str, Any] = field(default_factory=dict)

    def to_kwargs(self) -> dict:
        kw = {
            "data": self.data, "epochs": self.epochs, "imgsz": self.imgsz,
            "batch": self.batch, "device": self.device, "workers": self.workers,
            "patience": self.patience, "seed": self.seed,
            # Absolute, deliberately. Ultralytics resolves a RELATIVE project under
            # its own settings' runs_dir/<task>, so "runs/electrical" became
            # "runs/detect/runs/electrical" — which broke the documented artifact
            # path and put training output somewhere other than where hpo.py keeps
            # its study database. An absolute path is used verbatim.
            "project": os.path.abspath(self.project),
            "name": self.name or self.arch,
            "pretrained": self.pretrained, "optimizer": self.optimizer,
            **({"freeze": self.freeze} if self.freeze is not None else {}),
            "lr0": self.lr0, "cos_lr": self.cos_lr,
            "warmup_epochs": self.warmup_epochs,
            "degrees": self.degrees, "translate": self.translate,
            "scale": self.scale, "shear": self.shear,
            "perspective": self.perspective, "fliplr": self.fliplr,
            "flipud": self.flipud, "hsv_h": self.hsv_h, "hsv_s": self.hsv_s,
            "hsv_v": self.hsv_v, "mosaic": self.mosaic, "mixup": self.mixup,
            "copy_paste": self.copy_paste, "close_mosaic": self.close_mosaic,
            "exist_ok": True, "plots": True, "val": True,
        }
        kw.update(self.extra)
        return kw


@dataclass
class TrainResult:
    arch: str
    status: str                     # trained | skipped | failed
    reason: Optional[str] = None
    weights: Optional[str] = None
    onnx: Optional[str] = None
    duration_s: Optional[float] = None
    ultralytics_metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "arch": self.arch, "status": self.status, "reason": self.reason,
            "weights": self.weights, "onnx": self.onnx,
            "duration_s": (round(self.duration_s, 1) if self.duration_s else None),
            "ultralytics_metrics": self.ultralytics_metrics,
        }


def train(cfg: TrainConfig, export_onnx: bool = True,
          log: Optional[Callable[[str], None]] = None,
          callbacks: Optional[dict] = None) -> TrainResult:
    """Train one architecture. Returns a result even on failure — never raises.

    ``callbacks`` maps an Ultralytics callback event name to a callable, e.g.
    ``{"on_fit_epoch_end": fn}``. This is what makes per-epoch hyperparameter
    pruning possible (:mod:`training.electrical.hpo`): without an intermediate
    signal a pruner has nothing to prune on, because training is one blocking call.
    A callback that raises propagates — which is deliberate, since that is how
    Optuna's ``TrialPruned`` escapes a training run.
    """
    say = log or (lambda m: None)
    ok, version = ultralytics_available()
    if not ok:
        return TrainResult(cfg.arch, "skipped",
                           "ultralytics is not installed (pip install ultralytics)")
    if cfg.arch not in SUPPORTED_ARCHS:
        return TrainResult(cfg.arch, "skipped",
                           f"no adapter for '{cfg.arch}'; supported: "
                           f"{', '.join(SUPPORTED_ARCHS)}")
    if not arch_available(cfg.arch):
        return TrainResult(cfg.arch, "skipped",
                           f"'{cfg.arch}' is not available in the installed "
                           f"ultralytics {version}; upgrade or choose another "
                           f"architecture")
    if not os.path.exists(cfg.data):
        return TrainResult(cfg.arch, "skipped",
                           f"dataset yaml not found: {cfg.data}")

    started = time.perf_counter()
    try:
        from ultralytics import RTDETR, YOLO  # type: ignore
        Model = RTDETR if cfg.arch.startswith("rtdetr") else YOLO
        if cfg.init_from:
            if not os.path.exists(cfg.init_from):
                return TrainResult(
                    cfg.arch, "skipped",
                    f"init_from checkpoint not found: {cfg.init_from}")
            stem = cfg.init_from
            say(f"[{cfg.arch}] fine-tuning from {cfg.init_from}")
        else:
            stem = f"{cfg.arch}.pt" if cfg.pretrained else f"{cfg.arch}.yaml"
        say(f"[{cfg.arch}] loading {stem}")
        with quiet_stdout():
            model = Model(stem)
        for event, fn in (callbacks or {}).items():
            model.add_callback(event, fn)
        say(f"[{cfg.arch}] training for {cfg.epochs} epoch(s) at {cfg.imgsz}px")
        with quiet_stdout():
            results = model.train(**cfg.to_kwargs())

        save_dir = getattr(getattr(results, "save_dir", None), "__str__",
                           lambda: None)()
        weights = None
        if save_dir:
            cand = os.path.join(str(save_dir), "weights", "best.pt")
            weights = cand if os.path.exists(cand) else None

        onnx_path = None
        if export_onnx and weights:
            say(f"[{cfg.arch}] exporting ONNX")
            try:
                with quiet_stdout():
                    exported = Model(weights).export(
                        format="onnx", imgsz=cfg.imgsz, opset=12, simplify=True)
                onnx_path = str(exported) if exported else None
                if onnx_path and os.path.exists(onnx_path):
                    _write_classes_json(os.path.dirname(onnx_path))
            except Exception as exc:
                say(f"[{cfg.arch}] ONNX export failed: {exc}")

        metrics_dict: dict = {}
        rd = getattr(results, "results_dict", None)
        if isinstance(rd, dict):
            metrics_dict = {str(k): float(v) for k, v in rd.items()
                            if isinstance(v, (int, float))}
        return TrainResult(cfg.arch, "trained", None, weights, onnx_path,
                           time.perf_counter() - started, metrics_dict)
    except TrainingAborted:
        # Deliberate early stop (hyperparameter pruning, user cancel). Not a
        # failure, and the caller needs to distinguish the two.
        raise
    except Exception as exc:
        return TrainResult(cfg.arch, "failed", f"{type(exc).__name__}: {exc}",
                           None, None, time.perf_counter() - started)


def _write_classes_json(directory: str) -> None:
    """Pin the label order next to an exported checkpoint.

    The old pipeline shipped a ``labels.txt`` of bare integers, so labels came
    out as ``"0"``…``"9"``. Writing the authoritative class list beside every
    export makes that failure structurally impossible.

    Delegates to :func:`training.electrical.export.write_classes_json` so there is
    exactly one writer of this file. Two writers is how the taxonomy version in one
    of them silently goes stale.
    """
    try:
        from . import export as _export

        _export.write_classes_json(directory)
        _export.write_labels(directory)
    except OSError:
        pass


# --------------------------------------------------------------------------
# prediction collection + evaluation
# --------------------------------------------------------------------------

def collect_predictions(recognizer, image_dir: str,
                        limit: Optional[int] = None) -> list[dict]:
    """Run a recogniser over a directory and return metric-ready predictions."""
    import cv2

    out: list[dict] = []
    if not os.path.isdir(image_dir):
        return out
    files = [f for f in sorted(os.listdir(image_dir))
             if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]
    if limit:
        files = files[:limit]
    for fn in files:
        img = cv2.imread(os.path.join(image_dir, fn), cv2.IMREAD_COLOR)
        if img is None:
            continue
        image_id = os.path.splitext(fn)[0]
        if hasattr(recognizer, "recognize"):
            res = recognizer.recognize(img)
            for c in res.accepted:
                out.append({"image_id": image_id, "class_id": c.class_id,
                            "box": tuple(c.box), "score": float(c.score)})
        else:
            for d in recognizer.infer(img) or []:
                cid = (d.extra or {}).get("class_id") or tax.resolve(d.label) \
                    or tax.UNKNOWN_COMPONENT_ID
                out.append({"image_id": image_id, "class_id": cid,
                            "box": tuple(d.bbox.as_list()),
                            "score": float(d.confidence)})
    return out


def evaluated_image_ids(image_dir: str, limit: Optional[int] = None) -> set[str]:
    """The image ids :func:`collect_predictions` will actually run over.

    Mirrors its file selection so ground truth can be restricted to the same set.
    An image that produced no detections contributes nothing to ``preds``, so the
    evaluated set cannot be recovered from the predictions afterwards.
    """
    if not os.path.isdir(image_dir):
        return set()
    files = [f for f in sorted(os.listdir(image_dir))
             if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]
    if limit:
        files = files[:limit]
    return {os.path.splitext(f)[0] for f in files}


def load_ground_truth(dataset_root: str, split: str = "val",
                      names: Optional[Sequence[str]] = None) -> list[dict]:
    """Read YOLO labels into metric-ready ground truth (absolute pixel boxes).

    Label indices are read in the dataset's **own** label space, which for a
    profile-scoped dataset is not the taxonomy's: ``profiles.apply`` remaps
    indices to 0..N-1. Reading an 8-class core8 split through the 54-class
    taxonomy index yields ground truth labelled ``rccb``/``fuse`` for boxes that
    are actually ``contactor``/``relay``, and every metric computed against it is
    meaningless — precision and recall collapse for a reason that has nothing to
    do with the model.
    """
    import cv2

    from . import datasets as ds

    inv, _ = ds.label_index(dataset_root, names)
    img_dir = os.path.join(dataset_root, "images", split)
    lbl_dir = os.path.join(dataset_root, "labels", split)
    gts: list[dict] = []
    if not os.path.isdir(lbl_dir):
        return gts
    for fn in sorted(os.listdir(lbl_dir)):
        if not fn.endswith(".txt"):
            continue
        stem = os.path.splitext(fn)[0]
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            cand = os.path.join(img_dir, stem + ext)
            if os.path.exists(cand):
                img_path = cand
                break
        if img_path is None:
            continue
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        with open(os.path.join(lbl_dir, fn), "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    cid = inv[int(float(parts[0]))]
                    cx, cy, bw, bh = (float(parts[1]) * w, float(parts[2]) * h,
                                      float(parts[3]) * w, float(parts[4]) * h)
                except (ValueError, KeyError):
                    continue
                gts.append({"image_id": stem, "class_id": cid,
                            "box": (cx - bw / 2, cy - bh / 2,
                                    cx + bw / 2, cy + bh / 2)})
    return gts


def evaluate_backend(backend_id: str, dataset_root: str, split: str = "val",
                     params: Optional[dict] = None,
                     limit: Optional[int] = None) -> dict:
    """Evaluate any registered component backend on a dataset split."""
    from rtsp_backend.ai import registry
    from rtsp_backend import electrical  # noqa: F401  (registers backends)

    gts = load_ground_truth(dataset_root, split)
    if not gts:
        return {"status": "skipped",
                "reason": f"no ground truth under {dataset_root}/labels/{split}"}
    try:
        cls = registry.get("components", backend_id)
        inst = cls(**(params or {}))
        inst.load()
    except Exception as exc:
        return {"status": "skipped", "reason": f"{backend_id}: {exc}"}

    image_dir = os.path.join(dataset_root, "images", split)
    if limit:
        # Restrict the ground truth to the images that were inferred. Otherwise
        # every box in the un-inferred remainder of the split counts as a false
        # negative and the reported recall describes the limit, not the model.
        keep = evaluated_image_ids(image_dir, limit)
        gts = [g for g in gts if g.get("image_id") in keep]
    preds = collect_predictions(inst, image_dir, limit=limit)
    report = em.evaluate(gts, preds)
    report["status"] = "evaluated"
    report["backend_id"] = backend_id
    report["confusion_matrix"] = em.confusion_matrix(gts, preds)
    report["threshold_recommendations"] = {
        cid: v["recommended_threshold"]
        for cid, v in em.optimise_thresholds(gts, preds).items()
    }
    return report


def evaluate_zero_shot(dataset_root: str, split: str = "val",
                       backends: Sequence[str] = ("openvocab_owlv2",),
                       limit: Optional[int] = 25) -> dict:
    """Score the zero-shot detectors on the same split as a trained model.

    Kept separate from :func:`benchmark` because zero-shot inference is orders of
    magnitude slower per image, so it is normally run on a subset.
    """
    return {bid: evaluate_backend(bid, dataset_root, split, limit=limit)
            for bid in backends}


# --------------------------------------------------------------------------
# benchmark
# --------------------------------------------------------------------------

def profile_trained(weights: str, dataset_root: str, label: str,
                    split: str = "val", imgsz: int = 960,
                    device: str = "cpu", warmup: Optional[int] = None,
                    runs: Optional[int] = None,
                    log: Optional[Callable[[str], None]] = None):
    """Measure runtime characteristics of a trained checkpoint.

    Kept here rather than in :mod:`training.electrical.bench` so that module stays
    free of backend-loading concerns and can profile any already-loaded backend.
    """
    from . import bench as bm
    from rtsp_backend.ai import registry
    from rtsp_backend import electrical  # noqa: F401  (registers backends)

    try:
        cls = registry.get("components", "industrial_ultralytics")
        inst = cls(weights=weights, imgsz=imgsz, device=device)
        inst.load()
    except Exception as exc:
        return bm.RuntimeProfile(label=label, status="skipped",
                                 reason=f"{type(exc).__name__}: {exc}",
                                 device=device)
    return bm.profile_backend(
        inst, os.path.join(dataset_root, "images", split), label,
        warmup=warmup if warmup is not None else bm.DEFAULT_WARMUP,
        runs=runs if runs is not None else bm.DEFAULT_RUNS,
        device=device, log=log)


def benchmark(dataset_yaml: str, dataset_root: str,
              archs: Sequence[str] = ("yolo11s", "yolov8s", "rtdetr-l"),
              epochs: int = 60, imgsz: int = 960, batch: int = 8,
              device: str = "cpu", split: str = "val",
              measure_runtime: bool = True,
              runs: Optional[int] = None,
              latency_budget_ms: Optional[float] = None,
              weights: Optional[dict] = None,
              min_map_50_95: float = 0.0,
              log: Optional[Callable[[str], None]] = None) -> dict:
    """Train each architecture on one split, then rank on accuracy AND speed.

    Ranking on mAP alone reliably picks the largest model, which for a platform
    whose default deployment is CPU ONNX Runtime is usually the wrong call — a few
    points of mAP for several times the latency is a bad trade on a 4-core box.
    So each trained checkpoint is also profiled for latency, throughput and memory,
    and :func:`training.electrical.bench.select_best` makes one decision that states
    what it traded away.

    ``measure_runtime=False`` reverts to accuracy-only ranking.
    """
    say = log or (lambda m: None)
    from . import bench as bm

    trained: dict[str, TrainResult] = {}
    reports: dict[str, dict] = {}
    profiles: dict[str, dict] = {}

    for arch in archs:
        cfg = TrainConfig(data=dataset_yaml, arch=arch, epochs=epochs,
                          imgsz=imgsz, batch=batch, device=device,
                          name=f"bench_{arch}")
        say(f"=== {arch} ===")
        res = train(cfg, export_onnx=True, log=say)
        trained[arch] = res
        if res.status != "trained" or not res.weights:
            say(f"[{arch}] {res.status}: {res.reason}")
            continue
        rep = evaluate_backend("industrial_ultralytics", dataset_root, split,
                               params={"weights": res.weights, "imgsz": imgsz,
                                       "device": device})
        if rep.get("status") == "evaluated":
            reports[arch] = rep
            say(f"[{arch}] mAP@50 {rep['map_50']:.3f}  "
                f"mAP@50-95 {rep['map_50_95']:.3f}  F1 {rep['overall']['f1']:.3f}")
        if measure_runtime:
            profiles[arch] = profile_trained(
                res.weights, dataset_root, arch, split=split, imgsz=imgsz,
                device=device, runs=runs, log=say).to_dict()

    comparison = em.compare_models(reports) if reports else {
        "ranking": [], "winner": None,
        "criterion": "mAP@0.5:0.95, then F1"}

    selection = bm.select_best(
        reports, profiles, weights=weights,
        latency_budget_ms=(latency_budget_ms
                           if latency_budget_ms is not None
                           else bm.DEFAULT_LATENCY_BUDGET_MS),
        min_map_50_95=min_map_50_95)

    return {
        "training": {a: r.to_dict() for a, r in trained.items()},
        "evaluation": reports,
        "runtime": profiles,
        # Accuracy-only ranking, kept so the accuracy view is still available.
        "comparison": comparison,
        "table": em.format_table(comparison),
        # The combined decision, and the one to act on.
        "selection": selection,
        "runtime_table": (bm.format_profile_table(profiles) if profiles
                          else "runtime not measured"),
        "selection_table": bm.format_selection_table(selection),
        "recommended_arch": selection.get("winner"),
        "note": ("Architectures reported as 'skipped' were not silently "
                 "substituted — they are genuinely unavailable in this "
                 "environment. 'comparison' ranks on accuracy alone; "
                 "'selection' is the deployment decision and accounts for "
                 "latency and memory too."),
    }


__all__ = [
    "SUPPORTED_ARCHS", "CONDITIONAL_ARCHS", "DEFAULT_ARCH", "TrainingAborted",
    "TrainConfig",
    "TrainResult", "ultralytics_available", "arch_available", "train",
    "collect_predictions", "load_ground_truth", "evaluate_backend",
    "evaluate_zero_shot", "profile_trained", "benchmark",
]
