"""
Auto-annotation — pre-labelling to make human labelling faster, not to replace it.

Labelling is the bottleneck. A panel photograph at row framing carries ~12
devices, so the 6600-annotation shortfall that
:func:`training.electrical.datasets.requirements_report` reports is several
hundred hours of drawing boxes by hand. Pre-labelling with a model and having a
human *correct* boxes instead of drawing them is a 3–5× speed-up on that work.

What this module will and will not do
------------------------------------
It runs a detector — a zero-shot open-vocabulary model (OWLv2 / Grounding DINO)
when there is no trained checkpoint yet, or an earlier trained checkpoint once
there is one — over unlabelled images and writes YOLO label files.

It will **not** pretend those labels are ground truth. Every image gets a verdict
in the review manifest:

``auto``
    Every box cleared the accept threshold. Still needs a human glance, but a
    fast one.
``review``
    At least one box landed between the review and accept thresholds. A human
    must look at this image properly.
``uncertain``
    Boxes were found but the model was not confident about their classes; they are
    written as ``unknown_industrial_component`` rather than guessed into a class.
``empty``
    Nothing was detected. An empty label file is written, because for a photograph
    that genuinely contains no target devices that is the correct label — but the
    manifest flags it so a batch of empties is not mistaken for a completed batch.

Boxes below the review threshold are discarded entirely: a label file full of
junk boxes is slower to fix than an empty one.

The output is deliberately importable back into a labelling tool (Roboflow, CVAT,
Label Studio all read YOLO), so the workflow is: auto-label → import → human
corrects → export → :func:`training.electrical.split.split_dataset` → train.

Nothing in here fabricates a detection. With no usable backend installed it
reports ``skipped`` with the reason and writes no labels at all.
"""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from rtsp_backend.electrical import taxonomy as tax

from . import datasets as ds

#: Boxes at or above this confidence are written with their predicted class.
DEFAULT_ACCEPT = 0.35
#: Boxes between this and ``accept`` are written but the image is flagged for a
#: proper human pass. Below it, boxes are discarded.
DEFAULT_REVIEW = 0.15

#: Backends to try, in order of preference. The trained detectors come first —
#: once a checkpoint exists it is both faster and more accurate on this taxonomy
#: than any zero-shot model. Open-vocabulary models are the bootstrap for round
#: one, when no checkpoint exists yet.
DEFAULT_BACKENDS: tuple[str, ...] = (
    "industrial_onnx",
    "industrial_ultralytics",
    "openvocab_owlv2",
    "openvocab_grounding_dino",
)


@dataclass
class ImageVerdict:
    filename: str
    verdict: str                    # auto | review | uncertain | empty
    boxes: int = 0
    boxes_discarded: int = 0
    min_confidence: Optional[float] = None
    max_confidence: Optional[float] = None
    classes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "filename": self.filename, "verdict": self.verdict,
            "boxes": self.boxes, "boxes_discarded": self.boxes_discarded,
            "min_confidence": self.min_confidence,
            "max_confidence": self.max_confidence,
            "classes": self.classes,
        }


def load_backend(backends: Sequence[str] = DEFAULT_BACKENDS,
                 params: Optional[dict] = None,
                 log: Optional[Callable[[str], None]] = None):
    """Load the first available detection backend.

    Returns ``(instance, backend_id)`` or ``(None, reasons)`` where ``reasons``
    is a per-backend explanation. Backends are never substituted silently.
    """
    say = log or (lambda m: None)
    from rtsp_backend import electrical  # noqa: F401  (registers backends)
    from rtsp_backend.ai import registry

    reasons: dict[str, str] = {}
    for bid in backends:
        try:
            cls = registry.get("components", bid)
        except Exception as exc:
            reasons[bid] = f"not registered: {exc}"
            continue
        try:
            inst = cls(**(params or {}))
            inst.load()
        except Exception as exc:
            reasons[bid] = f"{type(exc).__name__}: {exc}"
            continue
        if getattr(inst, "ready", False):
            say(f"using backend '{bid}'")
            return inst, bid
        reasons[bid] = (getattr(inst, "_reason", None)
                        or getattr(inst, "_error", None)
                        or getattr(inst, "_status", "not ready"))
    return None, reasons


def _detections(backend, image) -> list[tuple[str, tuple[float, float, float, float], float]]:
    """Normalise either backend interface to ``(class_id, xyxy, score)``."""
    out: list[tuple[str, tuple, float]] = []
    if hasattr(backend, "recognize"):
        res = backend.recognize(image)
        for c in getattr(res, "accepted", []) or []:
            out.append((c.class_id, tuple(c.box), float(c.score)))
        return out
    for d in backend.infer(image) or []:
        cid = ((d.extra or {}).get("class_id")
               or tax.resolve(d.label)
               or tax.UNKNOWN_COMPONENT_ID)
        out.append((cid, tuple(d.bbox.as_list()), float(d.confidence)))
    return out


def _to_yolo_line(class_index: int, box: Sequence[float],
                  width: int, height: int) -> Optional[str]:
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    # Clip to the frame: a detector can return a box that overhangs the edge, and
    # a YOLO label outside [0,1] is silently dropped by most trainers.
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(width), x2), min(float(height), y2)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 1.0 or bh <= 1.0:
        return None
    cx, cy = (x1 + x2) / 2.0 / width, (y1 + y2) / 2.0 / height
    return (f"{class_index} {cx:.6f} {cy:.6f} "
            f"{bw / width:.6f} {bh / height:.6f}")


def autolabel_directory(image_dir: str, out_root: str,
                        backends: Sequence[str] = DEFAULT_BACKENDS,
                        params: Optional[dict] = None,
                        accept: float = DEFAULT_ACCEPT,
                        review: float = DEFAULT_REVIEW,
                        split: str = "train",
                        limit: Optional[int] = None,
                        copy_images: bool = True,
                        log: Optional[Callable[[str], None]] = None) -> dict:
    """Pre-label every image in ``image_dir`` into a YOLO dataset at ``out_root``.

    Writes ``images/<split>`` + ``labels/<split>`` + ``dataset.yaml`` so the
    result imports straight into a labelling tool, plus
    ``autolabel_manifest.json`` holding the per-image verdicts and the review
    queue ordered worst-first.
    """
    say = log or (lambda m: None)
    if review > accept:
        raise ValueError(f"review threshold ({review}) must be <= accept "
                         f"({accept})")
    if not os.path.isdir(image_dir):
        return {"status": "skipped", "reason": f"not a directory: {image_dir}"}

    try:
        import cv2
    except ImportError as exc:
        return {"status": "skipped",
                "reason": f"opencv is required for auto-labelling: {exc}"}

    backend, info = load_backend(backends, params, log=say)
    if backend is None:
        return {
            "status": "skipped",
            "reason": ("no detection backend is available, so no labels were "
                       "written. Install a zero-shot backend "
                       "(pip install -r requirements-openvocab.txt) to bootstrap "
                       "labelling without a trained model, or drop a trained "
                       "checkpoint into models/components/."),
            "backend_errors": info,
        }
    backend_id = info if isinstance(info, str) else "unknown"

    files = [f for f in sorted(os.listdir(image_dir))
             if f.lower().endswith(ds.IMAGE_EXTS)]
    if limit:
        files = files[:limit]
    if not files:
        return {"status": "skipped",
                "reason": f"no images found in {image_dir}"}

    d_img = os.path.join(out_root, "images", split)
    d_lbl = os.path.join(out_root, "labels", split)
    os.makedirs(d_img, exist_ok=True)
    os.makedirs(d_lbl, exist_ok=True)

    idx = tax.class_index()
    unknown_index = idx.get(tax.UNKNOWN_COMPONENT_ID)
    verdicts: list[ImageVerdict] = []
    totals: Counter = Counter()
    unreadable: list[str] = []

    for n, fn in enumerate(files, 1):
        path = os.path.join(image_dir, fn)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            unreadable.append(fn)
            continue
        h, w = img.shape[:2]
        try:
            dets = _detections(backend, img)
        except Exception as exc:
            say(f"  {fn}: inference failed ({type(exc).__name__}: {exc})")
            unreadable.append(fn)
            continue

        lines: list[str] = []
        scores: list[float] = []
        classes: Counter = Counter()
        discarded = 0
        has_low = False
        has_unknown = False

        for cid, box, score in dets:
            if score < review:
                discarded += 1
                continue
            if score < accept:
                has_low = True
            # An honest unknown stays unknown. The whole point of this pass is to
            # save the labeller time, and a confidently-wrong class label costs
            # more time than an unlabelled box.
            if cid == tax.UNKNOWN_COMPONENT_ID or cid not in idx:
                has_unknown = True
                if unknown_index is None:
                    discarded += 1
                    continue
                class_index = unknown_index
                cid = tax.UNKNOWN_COMPONENT_ID
            else:
                class_index = idx[cid]
            line = _to_yolo_line(class_index, box, w, h)
            if line is None:
                discarded += 1
                continue
            lines.append(line)
            scores.append(score)
            classes[cid] += 1

        with open(os.path.join(d_lbl, os.path.splitext(fn)[0] + ".txt"), "w",
                  encoding="utf-8") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))
        if copy_images:
            shutil.copy2(path, os.path.join(d_img, fn))
        else:
            dst = os.path.join(d_img, fn)
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(path), dst)

        if not lines:
            verdict = "empty"
        elif has_unknown:
            verdict = "uncertain"
        elif has_low:
            verdict = "review"
        else:
            verdict = "auto"

        verdicts.append(ImageVerdict(
            filename=fn, verdict=verdict, boxes=len(lines),
            boxes_discarded=discarded,
            min_confidence=round(min(scores), 4) if scores else None,
            max_confidence=round(max(scores), 4) if scores else None,
            classes=dict(classes)))
        totals.update(classes)
        if n % 25 == 0 or n == len(files):
            say(f"  {n}/{len(files)} image(s)")

    ds.write_dataset_yaml(out_root)

    by_verdict = Counter(v.verdict for v in verdicts)
    # Review queue: uncertain first, then lowest-confidence review images. This
    # is the order a human should work in — the worst predictions are both the
    # ones most likely to be wrong and the ones the model learns most from.
    queue = sorted(
        (v for v in verdicts if v.verdict in ("uncertain", "review")),
        key=lambda v: (0 if v.verdict == "uncertain" else 1,
                       v.min_confidence if v.min_confidence is not None else 0.0))

    manifest = {
        "status": "labelled",
        "backend": backend_id,
        "image_dir": image_dir,
        "out_root": out_root,
        "split": split,
        "thresholds": {"accept": accept, "review": review},
        "images_processed": len(verdicts),
        "images_unreadable": unreadable,
        "boxes_written": int(sum(v.boxes for v in verdicts)),
        "boxes_discarded": int(sum(v.boxes_discarded for v in verdicts)),
        "by_verdict": dict(by_verdict),
        "instances_per_class": dict(totals.most_common()),
        "review_queue": [v.filename for v in queue],
        "per_image": [v.to_dict() for v in verdicts],
        "human_review_required": True,
        "note": (
            f"{by_verdict.get('auto', 0)} image(s) were labelled confidently, "
            f"{by_verdict.get('review', 0)} need a review pass, "
            f"{by_verdict.get('uncertain', 0)} contain boxes the model could not "
            f"classify (written as '{tax.UNKNOWN_COMPONENT_ID}'), and "
            f"{by_verdict.get('empty', 0)} produced no detections. These are "
            f"PRE-LABELS, not ground truth: import them into a labelling tool, "
            f"correct them, and export before training. Training directly on "
            f"un-reviewed output teaches the model its own mistakes."),
        "next_step": (
            f"Import {out_root} into Roboflow/CVAT/Label Studio as YOLO, work "
            f"the review_queue first, then export and run: python -m "
            f"training.electrical.cli split --src <exported> --dst data/final"),
    }
    with open(os.path.join(out_root, "autolabel_manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    say(manifest["note"])
    return manifest


def annotation_instructions() -> dict:
    """The labelling guide, as a document a labeller can be handed.

    The brief asks for annotation instructions where classes are missing from
    public data — which, per :func:`training.electrical.datasets.plan`, is most
    of them. These rules are what keep a hand-labelled set internally consistent;
    inconsistent labelling caps achievable mAP no matter how long you train.
    """
    return {
        "scope": (
            "One box per physically separate, separately-replaceable device "
            "visible inside the panel. If you would order it as its own line "
            "item, it is its own box."),
        "box_tightness": [
            "Tight to the device housing, including its own terminals and "
            "terminal shrouds, excluding the wires leaving them.",
            "Include the front lever/actuator; exclude the DIN rail underneath.",
            "If a cable bundle crosses the device, box the device's full extent "
            "as if the cable were transparent — do not box only the visible part. "
            "Occlusion consistency matters more than pixel purity.",
        ],
        "class_specific_rules": [
            "CONTACTOR + OVERLOAD RELAY: two boxes. The overload bolted beneath "
            "a contactor is a separate device with its own part number.",
            "MOTOR STARTER: one box only when it is a single integrated unit "
            "(MPCB / manual motor starter). A contactor+overload assembly is not "
            "a motor starter — it is two boxes.",
            "TERMINAL BLOCK: one box per contiguous STRIP, not per pole. A "
            "40-way strip is one box. Per-pole boxes generate hundreds of "
            "instances per image and destroy the class balance.",
            "DIN RAIL: one box per continuous rail run, including the populated "
            "part. Also label visibly empty rail sections — otherwise the class "
            "is learned as 'gap between devices'.",
            "CABLE DUCT: one box per continuous duct run. Label lid-on and "
            "lid-off examples as the same class.",
            "BUSBAR: one box per continuous bar run. Bare copper and "
            "insulated/sleeved bar are the same class.",
            "MCB: one box per device, not per pole. A 3-pole MCB with a common "
            "toggle is ONE box; three separate 1-pole MCBs side by side are "
            "THREE boxes, even when linked by a busbar comb.",
            "CIRCUIT BREAKER (generic): use only when the family genuinely "
            "cannot be determined. If you can see it is an MCB, MCCB or ACB, "
            "label that instead — the generic class exists for ambiguity, not "
            "for convenience.",
            "INDICATOR LAMP vs PUSH BUTTON: an illuminated push button is a "
            "PUSH BUTTON. A lamp has no actuator travel — look for the bezel.",
            "EMERGENCY STOP: red mushroom head. Label it as EMERGENCY STOP, "
            "never as a push button, whether latched or released.",
            "PLC: one box for the CPU, and a separate box per I/O module. A "
            "compact all-in-one PLC is one box.",
        ],
        "unknown_policy": (
            f"If you cannot identify a device with confidence, label it "
            f"'{tax.UNKNOWN_COMPONENT_ID}'. Never guess. Unknown labels are "
            f"useful — they become the model's honest fallback and they form the "
            f"next capture list. A wrong confident label is actively harmful and "
            f"is very hard to find again later."),
        "do_not_label": [
            "Wires and cable bundles (wiring analysis is out of scope by design).",
            "Labels, legend plates, warning stickers and cable markers.",
            "The enclosure, door, mounting plate and hinges.",
            "Devices outside the panel that happen to be in frame.",
            "Anything you can only see reflected in the door glass.",
        ],
        "quality_control": [
            "Two labellers independently label the same 10% sample. Measure "
            "box-level IoU agreement and class agreement. Below ~0.85 IoU or "
            "~0.90 class agreement, fix the guide before labelling more — you "
            "are otherwise buying noise at scale.",
            "Re-review every image whose auto-label verdict was 'uncertain'. "
            "Those are the model's blind spots and the highest-value corrections.",
            "Never review your own auto-labels immediately after generating "
            "them; anchoring makes wrong boxes look right.",
        ],
        "workflow": [
            "1. Capture per training.electrical.datasets.custom_collection_plan().",
            "2. python -m training.electrical.cli autolabel --images <dir> "
            "--out data/prelabelled",
            "3. Import data/prelabelled into a labelling tool as YOLO.",
            "4. Correct, working the review_queue from autolabel_manifest.json "
            "first.",
            "5. Export as YOLO, then python -m training.electrical.cli split.",
            "6. python -m training.electrical.cli analyse — confirm the class "
            "counts moved before training.",
            "7. Train, then feed every 'unknown' detection from production back "
            "into step 2. That loop is what closes the long tail.",
        ],
    }


__all__ = ["DEFAULT_ACCEPT", "DEFAULT_REVIEW", "DEFAULT_BACKENDS",
           "ImageVerdict", "load_backend", "autolabel_directory",
           "annotation_instructions"]
