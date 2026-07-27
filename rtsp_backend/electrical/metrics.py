"""
Detection evaluation metrics — the evidence layer.

"The model is better now" is not a claim this project is allowed to make without
numbers, so the numbers live here: precision, recall, F1, mAP@0.5,
mAP@0.5:0.95, a per-class breakdown, a confusion matrix (with background rows
and columns so false positives and missed detections are visible), a
false-positive cause analysis, and a confidence-threshold optimiser.

Implemented from first principles on numpy — no dependency on the training stack,
so evaluation runs anywhere the backend runs and is unit-testable against hand
computable cases.

Conventions
-----------
Ground truth and predictions are plain dicts::

    gt   = {"image_id": str, "class_id": str, "box": (x1, y1, x2, y2)}
    pred = {"image_id": str, "class_id": str, "box": (...), "score": float}

Matching follows the standard COCO/VOC greedy rule: within an image and class,
predictions are considered in descending score order and matched to the
highest-IoU unmatched ground truth above the IoU threshold.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from . import postprocess as pp
from . import taxonomy as tax

DEFAULT_IOU = 0.5
COCO_IOUS: tuple[float, ...] = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))


# --------------------------------------------------------------------------
# matching
# --------------------------------------------------------------------------

@dataclass
class MatchResult:
    """Per-prediction match outcome at one IoU threshold."""

    #: 1 for a true positive, 0 for a false positive — index-aligned to `scores`
    tp: np.ndarray
    scores: np.ndarray
    classes: list[str]
    #: ground truths never matched: (image_id, class_id, box)
    false_negatives: list[tuple[str, str, tuple[float, float, float, float]]]
    #: predictions that matched nothing, with the best-overlapping GT class (or
    #: None) so we can say *why* it was wrong
    false_positives: list[dict] = field(default_factory=list)
    gt_count_per_class: dict[str, int] = field(default_factory=dict)


def match(gts: Sequence[Mapping], preds: Sequence[Mapping],
          iou_thr: float = DEFAULT_IOU) -> MatchResult:
    """Greedily match predictions to ground truth per image and class."""
    gt_by_img: dict[str, list[dict]] = defaultdict(list)
    for g in gts:
        gt_by_img[str(g["image_id"])].append(
            {"class_id": str(g["class_id"]),
             "box": tuple(float(v) for v in g["box"]), "used": False})

    gt_per_class: dict[str, int] = defaultdict(int)
    for g in gts:
        gt_per_class[str(g["class_id"])] += 1

    ordered = sorted(preds, key=lambda p: -float(p.get("score", 0.0)))
    tp_flags: list[int] = []
    scores: list[float] = []
    classes: list[str] = []
    fps: list[dict] = []

    for p in ordered:
        img = str(p["image_id"])
        cid = str(p["class_id"])
        box = tuple(float(v) for v in p["box"])
        scores.append(float(p.get("score", 0.0)))
        classes.append(cid)

        best, best_iou = None, iou_thr
        for g in gt_by_img.get(img, ()):
            if g["used"] or g["class_id"] != cid:
                continue
            v = pp.iou(box, g["box"])
            if v >= best_iou:
                best, best_iou = g, v
        if best is not None:
            best["used"] = True
            tp_flags.append(1)
            continue

        tp_flags.append(0)
        # Explain the false positive: did it land on a device of another class
        # (a confusion), or on nothing at all (a hallucination)?
        overlap_cls, overlap_iou = None, 0.0
        for g in gt_by_img.get(img, ()):
            v = pp.iou(box, g["box"])
            if v > overlap_iou:
                overlap_cls, overlap_iou = g["class_id"], v
        fps.append({
            "image_id": img, "class_id": cid, "box": box,
            "score": float(p.get("score", 0.0)),
            "best_overlap_class": overlap_cls,
            "best_overlap_iou": round(float(overlap_iou), 4),
            "cause": ("class_confusion" if overlap_iou >= iou_thr
                      else ("localisation" if overlap_iou >= 0.1
                            else "spurious_detection")),
        })

    fns = [(img, g["class_id"], g["box"])
           for img, gl in gt_by_img.items() for g in gl if not g["used"]]

    return MatchResult(
        tp=np.array(tp_flags, dtype=np.int32),
        scores=np.array(scores, dtype=np.float64),
        classes=classes, false_negatives=fns, false_positives=fps,
        gt_count_per_class=dict(gt_per_class),
    )


# --------------------------------------------------------------------------
# precision / recall / AP
# --------------------------------------------------------------------------

def precision_recall_f1(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4)}


def average_precision(tp_flags: Sequence[int], scores: Sequence[float],
                      n_gt: int) -> float:
    """All-point interpolated AP (the COCO/VOC-2010 definition)."""
    if n_gt <= 0:
        return 0.0
    if len(tp_flags) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    tp = np.asarray(tp_flags, dtype=np.float64)[order]
    fp = 1.0 - tp
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    recall = ctp / float(n_gt)
    precision = ctp / np.maximum(ctp + cfp, 1e-12)

    # monotone-decreasing precision envelope
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate(gts: Sequence[Mapping], preds: Sequence[Mapping],
             iou_thr: float = DEFAULT_IOU,
             ious: Sequence[float] = COCO_IOUS) -> dict:
    """Full evaluation report at ``iou_thr`` plus mAP averaged over ``ious``."""
    m = match(gts, preds, iou_thr)
    classes = sorted(set(m.gt_count_per_class) | set(m.classes))

    per_class: dict[str, dict] = {}
    aps: list[float] = []
    for cid in classes:
        sel = [i for i, c in enumerate(m.classes) if c == cid]
        tp = int(m.tp[sel].sum()) if sel else 0
        fp = len(sel) - tp
        n_gt = int(m.gt_count_per_class.get(cid, 0))
        fn = max(0, n_gt - tp)
        stats = precision_recall_f1(tp, fp, fn)
        ap = average_precision(m.tp[sel].tolist() if sel else [],
                               m.scores[sel].tolist() if sel else [], n_gt)
        stats["ap"] = round(ap, 4)
        stats["support"] = n_gt
        stats["name"] = tax.display_name(cid)
        per_class[cid] = stats
        if n_gt > 0:
            aps.append(ap)

    total_tp = int(m.tp.sum())
    total_fp = int(len(m.tp) - total_tp)
    total_fn = len(m.false_negatives)
    overall = precision_recall_f1(total_tp, total_fp, total_fn)

    # mAP averaged over IoU thresholds (the COCO primary metric)
    map_by_iou: dict[str, float] = {}
    for thr in ious:
        mm = match(gts, preds, thr)
        thr_aps: list[float] = []
        for cid in sorted(mm.gt_count_per_class):
            sel = [i for i, c in enumerate(mm.classes) if c == cid]
            thr_aps.append(average_precision(
                mm.tp[sel].tolist() if sel else [],
                mm.scores[sel].tolist() if sel else [],
                int(mm.gt_count_per_class[cid])))
        map_by_iou[f"{thr:.2f}"] = round(
            float(np.mean(thr_aps)) if thr_aps else 0.0, 4)

    fp_causes: dict[str, int] = defaultdict(int)
    for fp_item in m.false_positives:
        fp_causes[fp_item["cause"]] += 1

    fn_by_class: dict[str, int] = defaultdict(int)
    for _, cid, _ in m.false_negatives:
        fn_by_class[cid] += 1

    return {
        "iou_threshold": iou_thr,
        "overall": overall,
        "map_50": round(float(np.mean(aps)) if aps else 0.0, 4),
        "map_50_95": round(float(np.mean(list(map_by_iou.values())))
                           if map_by_iou else 0.0, 4),
        "map_by_iou": map_by_iou,
        "per_class": per_class,
        "macro_precision": round(float(np.mean(
            [v["precision"] for v in per_class.values() if v["support"]]))
            if any(v["support"] for v in per_class.values()) else 0.0, 4),
        "macro_recall": round(float(np.mean(
            [v["recall"] for v in per_class.values() if v["support"]]))
            if any(v["support"] for v in per_class.values()) else 0.0, 4),
        "false_positive_analysis": {
            "total": len(m.false_positives),
            "by_cause": dict(fp_causes),
            "worst_offenders": _worst_fp_classes(m.false_positives),
            "examples": m.false_positives[:25],
        },
        "false_negative_analysis": {
            "total": total_fn,
            "by_class": {tax.display_name(k): v
                         for k, v in sorted(fn_by_class.items(),
                                            key=lambda kv: -kv[1])},
            "examples": [{"image_id": i, "class_id": c,
                          "name": tax.display_name(c), "box": list(b)}
                         for i, c, b in m.false_negatives[:25]],
        },
        "counts": {"ground_truth": len(gts), "predictions": len(preds)},
    }


def _worst_fp_classes(fps: Sequence[Mapping], top: int = 5) -> list[dict]:
    tally: dict[str, int] = defaultdict(int)
    for f in fps:
        tally[str(f["class_id"])] += 1
    ranked = sorted(tally.items(), key=lambda kv: -kv[1])[:top]
    return [{"class_id": c, "name": tax.display_name(c), "false_positives": n}
            for c, n in ranked]


# --------------------------------------------------------------------------
# confusion matrix
# --------------------------------------------------------------------------

def confusion_matrix(gts: Sequence[Mapping], preds: Sequence[Mapping],
                     iou_thr: float = DEFAULT_IOU,
                     classes: Optional[Sequence[str]] = None) -> dict:
    """Confusion matrix with an explicit ``__background__`` row and column.

    Row = ground-truth class, column = predicted class. The background column
    counts missed detections; the background row counts detections on nothing.
    Without those two, a detection confusion matrix hides exactly the failures
    that matter.
    """
    labels = list(classes) if classes else sorted(
        {str(g["class_id"]) for g in gts} | {str(p["class_id"]) for p in preds})
    bg = "__background__"
    index = {c: i for i, c in enumerate(labels + [bg])}
    n = len(labels) + 1
    mat = np.zeros((n, n), dtype=np.int64)

    gt_by_img: dict[str, list[dict]] = defaultdict(list)
    for g in gts:
        gt_by_img[str(g["image_id"])].append(
            {"class_id": str(g["class_id"]),
             "box": tuple(float(v) for v in g["box"]), "used": False})

    for p in sorted(preds, key=lambda x: -float(x.get("score", 0.0))):
        img, cid = str(p["image_id"]), str(p["class_id"])
        box = tuple(float(v) for v in p["box"])
        best, best_iou = None, iou_thr
        for g in gt_by_img.get(img, ()):
            if g["used"]:
                continue
            v = pp.iou(box, g["box"])
            if v >= best_iou:
                best, best_iou = g, v
        if best is None:
            mat[index[bg], index.get(cid, index[bg])] += 1
        else:
            best["used"] = True
            mat[index.get(best["class_id"], index[bg]),
                index.get(cid, index[bg])] += 1

    for gl in gt_by_img.values():
        for g in gl:
            if not g["used"]:
                mat[index.get(g["class_id"], index[bg]), index[bg]] += 1

    per_class_acc: dict[str, float] = {}
    for c in labels:
        i = index[c]
        row = mat[i].sum()
        per_class_acc[c] = round(float(mat[i, i] / row), 4) if row else 0.0

    return {
        "labels": labels + [bg],
        "names": [tax.display_name(c) for c in labels] + ["(background)"],
        "matrix": mat.tolist(),
        "per_class_accuracy": per_class_acc,
        "per_class_accuracy_named": {
            tax.display_name(c): v for c, v in per_class_acc.items()},
    }


# --------------------------------------------------------------------------
# threshold optimisation
# --------------------------------------------------------------------------

def optimise_thresholds(gts: Sequence[Mapping], preds: Sequence[Mapping],
                        iou_thr: float = DEFAULT_IOU,
                        grid: Optional[Sequence[float]] = None,
                        objective: str = "f1",
                        min_precision: float = 0.0) -> dict:
    """Per-class confidence threshold that maximises F1 (or precision-at-recall).

    This is how the acceptance thresholds in
    :func:`~.taxonomy.default_thresholds` should be *derived* once a validation
    set exists, rather than guessed. ``min_precision`` lets an operator demand a
    precision floor — the industrial trade-off, where a false alarm costs more
    than a miss.
    """
    grid = list(grid) if grid else [round(0.05 * i, 2) for i in range(1, 20)]
    by_class_preds: dict[str, list[Mapping]] = defaultdict(list)
    for p in preds:
        by_class_preds[str(p["class_id"])].append(p)

    out: dict[str, dict] = {}
    for cid, cpreds in by_class_preds.items():
        cgts = [g for g in gts if str(g["class_id"]) == cid]
        best: Optional[dict] = None
        curve: list[dict] = []
        for thr in grid:
            kept = [p for p in cpreds if float(p.get("score", 0.0)) >= thr]
            m = match(cgts, kept, iou_thr)
            tp = int(m.tp.sum())
            fp = int(len(m.tp) - tp)
            fn = len(m.false_negatives)
            stats = precision_recall_f1(tp, fp, fn)
            stats["threshold"] = thr
            curve.append(stats)
            if stats["precision"] < min_precision:
                continue
            score = stats[objective] if objective in stats else stats["f1"]
            if best is None or score > best[objective]:
                best = stats
        out[cid] = {
            "name": tax.display_name(cid),
            "recommended_threshold": (best or {}).get(
                "threshold", tax.spec(cid).min_conf),
            "at_recommended": best,
            "current_default": tax.spec(cid).min_conf,
            "curve": curve,
        }
    return out


# --------------------------------------------------------------------------
# comparison table
# --------------------------------------------------------------------------

def compare_models(reports: Mapping[str, Mapping]) -> dict:
    """Side-by-side table of several :func:`evaluate` reports.

    Used by the benchmark CLI to pick a model on measurable performance rather
    than on reputation.
    """
    rows: list[dict] = []
    for name, rep in reports.items():
        overall = rep.get("overall") or {}
        rows.append({
            "model": name,
            "precision": overall.get("precision"),
            "recall": overall.get("recall"),
            "f1": overall.get("f1"),
            "map_50": rep.get("map_50"),
            "map_50_95": rep.get("map_50_95"),
            "false_positives": (rep.get("false_positive_analysis") or {}).get("total"),
            "false_negatives": (rep.get("false_negative_analysis") or {}).get("total"),
        })
    rows.sort(key=lambda r: (-(r.get("map_50_95") or 0), -(r.get("f1") or 0)))
    return {"ranking": rows,
            "winner": rows[0]["model"] if rows else None,
            "criterion": "mAP@0.5:0.95, then F1"}


def format_table(comparison: Mapping) -> str:
    rows = comparison.get("ranking") or []
    if not rows:
        return "(no models evaluated)"
    head = f"{'model':<28}{'P':>8}{'R':>8}{'F1':>8}{'mAP50':>9}{'mAP50-95':>10}{'FP':>7}{'FN':>7}"
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{str(r['model'])[:27]:<28}"
            f"{(r.get('precision') or 0):>8.3f}{(r.get('recall') or 0):>8.3f}"
            f"{(r.get('f1') or 0):>8.3f}{(r.get('map_50') or 0):>9.3f}"
            f"{(r.get('map_50_95') or 0):>10.3f}"
            f"{(r.get('false_positives') or 0):>7}{(r.get('false_negatives') or 0):>7}")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_IOU", "COCO_IOUS", "MatchResult", "match", "precision_recall_f1",
    "average_precision", "evaluate", "confusion_matrix", "optimise_thresholds",
    "compare_models", "format_table",
]
