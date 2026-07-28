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

import json
import os
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
            "project": self.project, "name": self.name or self.arch,
            "pretrained": self.pretrained, "optimizer": self.optimizer,
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
          log: Optional[Callable[[str], None]] = None) -> TrainResult:
    """Train one architecture. Returns a result even on failure — never raises."""
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
        stem = f"{cfg.arch}.pt" if cfg.pretrained else f"{cfg.arch}.yaml"
        say(f"[{cfg.arch}] loading {stem}")
        model = Model(stem)
        say(f"[{cfg.arch}] training for {cfg.epochs} epoch(s) at {cfg.imgsz}px")
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
                exported = Model(weights).export(format="onnx", imgsz=cfg.imgsz,
                                                 opset=12, simplify=True)
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
    except Exception as exc:
        return TrainResult(cfg.arch, "failed", f"{type(exc).__name__}: {exc}",
                           None, None, time.perf_counter() - started)


def _write_classes_json(directory: str) -> None:
    """Pin the label order next to an exported checkpoint.

    The old pipeline shipped a ``labels.txt`` of bare integers, so labels came
    out as ``"0"``…``"9"``. Writing the authoritative class list beside every
    export makes that failure structurally impossible.
    """
    try:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "classes.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"classes": list(tax.CLASS_ORDER),
                       "taxonomy_version": "5.0"}, fh, indent=2)
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


def load_ground_truth(dataset_root: str, split: str = "val") -> list[dict]:
    """Read YOLO labels into metric-ready ground truth (absolute pixel boxes)."""
    import cv2

    inv = {v: k for k, v in tax.class_index().items()}
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

    preds = collect_predictions(inst, os.path.join(dataset_root, "images", split),
                               limit=limit)
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

def benchmark(dataset_yaml: str, dataset_root: str,
              archs: Sequence[str] = ("yolo11s", "yolov8s", "rtdetr-l"),
              epochs: int = 60, imgsz: int = 960, batch: int = 8,
              device: str = "cpu", split: str = "val",
              log: Optional[Callable[[str], None]] = None) -> dict:
    """Train each architecture on the same data and rank them by measured mAP."""
    say = log or (lambda m: None)
    trained: dict[str, TrainResult] = {}
    reports: dict[str, dict] = {}

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

    comparison = em.compare_models(reports) if reports else {
        "ranking": [], "winner": None,
        "criterion": "mAP@0.5:0.95, then F1"}
    return {
        "training": {a: r.to_dict() for a, r in trained.items()},
        "evaluation": reports,
        "comparison": comparison,
        "table": em.format_table(comparison),
        "note": ("Architectures reported as 'skipped' were not silently "
                 "substituted — they are genuinely unavailable in this "
                 "environment."),
    }


__all__ = [
    "SUPPORTED_ARCHS", "CONDITIONAL_ARCHS", "DEFAULT_ARCH", "TrainConfig",
    "TrainResult", "ultralytics_available", "arch_available", "train",
    "collect_predictions", "load_ground_truth", "evaluate_backend",
    "evaluate_zero_shot", "benchmark",
]
