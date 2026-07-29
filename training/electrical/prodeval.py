"""Production-path evaluation and the acceptance sweep.

Ultralytics' own ``val`` numbers describe the *checkpoint*. They do not describe
what the deployed system returns, because between the model head and the API
response sits the whole gate cascade in
:mod:`rtsp_backend.electrical.postprocess`: per-class NMS, cross-class dedupe,
the geometric plausibility gate, per-class acceptance thresholds, the
"demote to unknown rather than guess" rule, and the detection cap. A checkpoint
with a flattering mAP can still ship badly if the gate is tuned wrong.

This module therefore evaluates **only** through the production path — the same
:meth:`~rtsp_backend.electrical.recognizer.IndustrialRecognizer.recognize` call
the runtime makes — and reports the numbers an operator actually feels:
false positives per image, false negatives per image, how often the system
abstains, and how many boxes the cascade accepted versus threw away.

The cheap-sweep trick
---------------------
The acceptance sweep explores ``decode_floor`` × ``unknown_floor`` ×
``strictness``, which is 280 configurations in the default grid. Re-running
inference for each would mean 280 full passes over the validation split. It is
not necessary:

* ``decode_floor`` is a **pure score cutoff** applied inside
  :func:`~rtsp_backend.electrical.recognizer.decode_yolo` (``keep = scores >=
  conf_thr``) with no top-k truncation, so the candidate set at floor *f* is
  exactly the subset of the candidate set at any lower floor with
  ``score >= f``.
* ``unknown_floor`` and the per-class thresholds are consumed entirely by
  :func:`~rtsp_backend.electrical.postprocess.confidence_gate`, downstream of
  inference.

So we run inference **once** at the lowest floor in the grid, cache the raw
candidates per image, and then replay the gate over the cache. Every point in
the sweep is bit-identical to what a fresh inference pass at that floor would
have produced, at 1/280th of the cost. Measured: 280 points over a 160-image
split in about a minute, against roughly an hour of repeated CPU inference.

Note that ``strictness`` — the global multiplier on the per-class acceptance
thresholds — is in the grid for a specific measured reason. See :func:`sweep`:
neither floor can change which boxes clear their class threshold, so sweeping the
two floors alone reports a completely flat precision/recall surface.

Asserted versus localised
-------------------------
A detection demoted to :data:`~rtsp_backend.electrical.taxonomy.UNKNOWN_COMPONENT_ID`
is not a misclassification — it is a deliberate abstention. Counting it as a
false positive against a typed ground-truth box would punish the honesty rule
the specification asks for. So the primary metrics are computed over
**asserted** detections only (what the system actually claims), and abstentions
are reported separately as ``unknown_rate``.

Because that alone cannot distinguish "never saw the device" from "saw it but
would not name it", every report also carries a **class-agnostic** recall
(``recall_localised``) computed over *all* accepted boxes including unknowns.
The gap between ``recall`` and ``recall_localised`` is the share of the misses
that are classification failures rather than detection failures — which is what
makes the per-class failure analysis diagnostic instead of merely descriptive.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

from rtsp_backend.electrical import metrics as em
from rtsp_backend.electrical import postprocess as pp
from rtsp_backend.electrical import taxonomy as tax

#: Decode floors explored by the default sweep.
DECODE_FLOORS: tuple[float, ...] = (0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20)

#: Unknown floors explored by the default sweep.
UNKNOWN_FLOORS: tuple[float, ...] = (0.10, 0.15, 0.18, 0.20, 0.25)

#: Global multipliers on every per-class acceptance threshold. This is the
#: dimension that actually trades precision against recall — see :func:`sweep`.
#: 1.0 is the taxonomy default; the range spans roughly 0.12–0.60 effective
#: threshold for a class whose default is 0.40.
STRICTNESS_VALUES: tuple[float, ...] = (0.30, 0.50, 0.70, 0.85, 1.00, 1.15,
                                        1.30, 1.50)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class ImageCandidates:
    """Raw, pre-gate model output for one image, plus what it takes to gate it."""

    image_id: str
    shape: tuple[int, int]                  # (h, w) of the original image
    candidates: list[pp.Candidate] = field(default_factory=list)


@dataclass
class CandidateCache:
    """The single inference pass every point in a sweep is replayed against."""

    images: list[ImageCandidates] = field(default_factory=list)
    backend_id: str = ""
    split: str = ""
    #: The decode floor inference was actually run at. Replaying the gate at a
    #: floor *below* this would be a lie, so :func:`production_report` refuses.
    base_decode_floor: float = 0.0
    #: Backends that cannot report a raw score floor (zero-shot prompt models)
    #: are still usable, but the sweep cannot go below what they emitted.
    notes: list[str] = field(default_factory=list)

    @property
    def image_count(self) -> int:
        return len(self.images)

    @property
    def raw_count(self) -> int:
        return sum(len(i.candidates) for i in self.images)


def _image_files(image_dir: str, limit: Optional[int] = None) -> list[str]:
    if not os.path.isdir(image_dir):
        return []
    files = [f for f in sorted(os.listdir(image_dir))
             if f.lower().endswith(IMAGE_EXTS)]
    return files[:limit] if limit else files


def cache_candidates(backend_id: str, dataset_root: str, split: str = "val",
                     params: Optional[Mapping] = None,
                     limit: Optional[int] = None,
                     base_decode_floor: float = min(DECODE_FLOORS),
                     log=None) -> CandidateCache:
    """Run inference once and keep every raw candidate above ``base_decode_floor``.

    The backend is instantiated with ``decode_floor`` forced to
    ``base_decode_floor`` so the cache is a superset of every point the sweep
    will ask for. Any other caller-supplied param is passed through untouched.
    """
    import cv2

    from rtsp_backend import electrical  # noqa: F401  (registers the backends)
    from rtsp_backend.ai import registry

    merged = dict(params or {})
    merged["decode_floor"] = float(base_decode_floor)
    inst = registry.get("components", backend_id)(**merged)
    inst.load()

    cache = CandidateCache(backend_id=backend_id, split=split,
                           base_decode_floor=float(base_decode_floor))
    image_dir = os.path.join(dataset_root, "images", split)
    files = _image_files(image_dir, limit)
    if not files:
        cache.notes.append(f"no images under {image_dir}")
        return cache

    for n, fn in enumerate(files, 1):
        img = cv2.imread(os.path.join(image_dir, fn), cv2.IMREAD_COLOR)
        if img is None:
            cache.notes.append(f"unreadable: {fn}")
            continue
        cands = inst.raw_candidates(img)
        cache.images.append(ImageCandidates(
            image_id=os.path.splitext(fn)[0],
            shape=(int(img.shape[0]), int(img.shape[1])),
            candidates=list(cands)))
        if log and n % 25 == 0:
            log(f"  cached {n}/{len(files)} images "
                f"({cache.raw_count} raw candidates)")
    return cache


def gate_config(unknown_floor: float,
                thresholds: Optional[Mapping[str, float]] = None,
                strictness: float = 1.0,
                check_plausibility: bool = True) -> pp.GateConfig:
    """Build the production gate config for one operating point."""
    cfg = pp.GateConfig()
    cfg.unknown_floor = float(unknown_floor)
    cfg.strictness = float(strictness)
    cfg.check_plausibility = bool(check_plausibility)
    if thresholds:
        cfg.thresholds = {**cfg.thresholds,
                          **{str(k): float(v) for k, v in thresholds.items()}}
    return cfg


def _as_pred(image_id: str, c: pp.Candidate) -> dict:
    return {"image_id": image_id, "class_id": c.class_id,
            "box": tuple(float(v) for v in c.box), "score": float(c.score)}


def _class_agnostic(items: Sequence[Mapping]) -> list[dict]:
    """Relabel everything to one class, so matching measures localisation only."""
    return [{**dict(i), "class_id": "any"} for i in items]


def replay_gate(cache: CandidateCache, decode_floor: float,
                cfg: pp.GateConfig) -> dict:
    """Replay the production gate over a cached inference pass.

    Returns the asserted predictions, the abstentions, and the accounting of
    what the cascade dropped and why.
    """
    asserted: list[dict] = []
    unknown: list[dict] = []
    dropped_by_reason: dict[str, int] = {}
    below_decode = 0
    kept_raw = 0
    truncated_images = 0

    for item in cache.images:
        kept = [c for c in item.candidates if float(c.score) >= decode_floor]
        below_decode += len(item.candidates) - len(kept)
        kept_raw += len(kept)
        res = pp.run(kept, item.shape, cfg)
        for c in res.accepted:
            if c.class_id == tax.UNKNOWN_COMPONENT_ID:
                unknown.append(_as_pred(item.image_id, c))
            else:
                asserted.append(_as_pred(item.image_id, c))
        for reason, n in res.diagnostics.dropped.items():
            dropped_by_reason[reason] = dropped_by_reason.get(reason, 0) + int(n)
        if res.truncated:
            truncated_images += 1

    accepted_total = len(asserted) + len(unknown)
    gate_dropped = sum(dropped_by_reason.values())
    return {
        "asserted": asserted,
        "unknown": unknown,
        "accepted_total": accepted_total,
        "raw_candidates": cache.raw_count,
        "kept_after_decode_floor": kept_raw,
        "dropped_below_decode_floor": below_decode,
        "dropped_by_gate": gate_dropped,
        "dropped_by_reason": dict(sorted(dropped_by_reason.items(),
                                         key=lambda kv: -kv[1])),
        "rejected_total": below_decode + gate_dropped,
        "truncated_images": truncated_images,
    }


def operating_point(gts: Sequence[Mapping], cache: CandidateCache,
                    decode_floor: float, unknown_floor: float,
                    thresholds: Optional[Mapping[str, float]] = None,
                    strictness: float = 1.0,
                    check_plausibility: bool = True) -> tuple[list, list, dict]:
    """Aligned ground truth, asserted detections, and the gate accounting.

    Shared by :func:`production_report` and the gallery writer so that what a
    gallery displays is exactly what the reported metrics counted, rather than a
    second derivation that can drift from it.
    """
    if decode_floor < cache.base_decode_floor - 1e-9:
        raise ValueError(
            f"decode_floor {decode_floor} is below the {cache.base_decode_floor} "
            "floor inference was cached at; re-cache before evaluating it")
    ids = {i.image_id for i in cache.images}
    aligned = [g for g in gts if g.get("image_id") in ids]
    cfg = gate_config(unknown_floor, thresholds, strictness, check_plausibility)
    gated = replay_gate(cache, decode_floor, cfg)
    return aligned, gated["asserted"], gated


def production_report(gts: Sequence[Mapping], cache: CandidateCache,
                      decode_floor: float, unknown_floor: float,
                      thresholds: Optional[Mapping[str, float]] = None,
                      strictness: float = 1.0,
                      check_plausibility: bool = True,
                      iou_thr: float = em.DEFAULT_IOU,
                      full: bool = True) -> dict:
    """Evaluate one operating point through the production path.

    ``full=False`` skips the mAP@0.5:0.95 sweep over IoU thresholds, which is
    the expensive part; the sweep uses it for the grid and then recomputes the
    winner in full.
    """
    # The ground truth is aligned to the images actually inferred. With --limit,
    # predictions cover a subset while load_ground_truth reads the whole split;
    # scoring the two against each other turns every unevaluated image's boxes
    # into false negatives and reports a recall that has nothing to do with the
    # model. Aligning inside here rather than in the caller means no caller can
    # forget to do it.
    gt_all = len(gts)
    gts, asserted, gated = operating_point(
        gts, cache, decode_floor, unknown_floor, thresholds, strictness,
        check_plausibility)
    unknown = gated["unknown"]
    images = max(1, cache.image_count)

    ious = em.COCO_IOUS if full else (iou_thr,)
    rep = em.evaluate(gts, asserted, iou_thr=iou_thr, ious=ious)

    # Class-agnostic recall over *every* accepted box, abstentions included.
    # This is the localisation ceiling: recall can never exceed it, and the gap
    # is misses that are classification failures rather than blindness.
    agnostic = em.evaluate(_class_agnostic(gts),
                           _class_agnostic(asserted + unknown),
                           iou_thr=iou_thr, ious=(iou_thr,))
    recall_localised = agnostic["overall"]["recall"]

    fp = rep["false_positive_analysis"]["total"]
    fn = rep["false_negative_analysis"]["total"]
    accepted_total = gated["accepted_total"]

    production = {
        "decode_floor": round(float(decode_floor), 4),
        "unknown_floor": round(float(unknown_floor), 4),
        "strictness": round(float(strictness), 4),
        "images": cache.image_count,
        "ground_truth": len(gts),
        #: Boxes belonging to images outside the evaluated set, excluded above.
        #: Non-zero whenever --limit is in play; reported so a smaller
        #: ground-truth count than the split's total is explained, not mysterious.
        "ground_truth_outside_evaluated_images": gt_all - len(gts),
        "precision": rep["overall"]["precision"],
        "recall": rep["overall"]["recall"],
        "f1": rep["overall"]["f1"],
        "map_50": rep["map_50"],
        "map_50_95": rep["map_50_95"] if full else None,
        "macro_precision": rep["macro_precision"],
        "macro_recall": rep["macro_recall"],
        # Per-image rates use the number of images actually evaluated as the
        # denominator. Dividing by the ground-truth or prediction count instead
        # makes the figure move when the gate changes and is meaningless.
        "fp_per_image": round(fp / images, 4),
        "fn_per_image": round(fn / images, 4),
        "false_positives": fp,
        "false_negatives": fn,
        "accepted_detections": accepted_total,
        "accepted_asserted": len(asserted),
        "accepted_unknown": len(unknown),
        "rejected_detections": gated["rejected_total"],
        "unknown_rate": round(len(unknown) / accepted_total, 4) if accepted_total
                        else 0.0,
        "detections_per_image": round(accepted_total / images, 4),
        "recall_localised": recall_localised,
        "classification_shortfall": round(
            max(0.0, recall_localised - rep["overall"]["recall"]), 4),
        "raw_candidates": gated["raw_candidates"],
        "kept_after_decode_floor": gated["kept_after_decode_floor"],
        "dropped_below_decode_floor": gated["dropped_below_decode_floor"],
        "dropped_by_gate": gated["dropped_by_gate"],
        "dropped_by_reason": gated["dropped_by_reason"],
        "truncated_images": gated["truncated_images"],
    }
    return {"production": production, "detail": rep,
            "backend_id": cache.backend_id, "split": cache.split,
            "iou_threshold": iou_thr,
            "evaluated_via": "production inference path "
                             "(recognize -> postprocess.run)"}


# --------------------------------------------------------------------------
# the acceptance sweep
# --------------------------------------------------------------------------

#: Objectives the sweep can rank an operating point by.
OBJECTIVES = ("f1", "map_50", "precision", "recall", "production_score")


def score_point(prod: Mapping, objective: str = "f1",
                max_fp_per_image: Optional[float] = None,
                min_precision: float = 0.0,
                min_recall: float = 0.0) -> Optional[float]:
    """Rank one operating point, or ``None`` if it violates a constraint.

    ``production_score`` is a deliberately opinionated blend for a panel
    inspector: F1 carries the weight, mAP@0.5 keeps localisation honest, and
    false positives are penalised because an operator who is shown junk stops
    trusting the tool faster than one who is shown too little.

    The abstention penalty is small on purpose. An "unknown industrial
    component" box is far cheaper than a confident wrong label, so it must not be
    scored like a false positive — but it cannot be free either. Left free, two
    configurations with identical asserted detections tie, and the tie breaks
    toward whichever one demotes the most junk to unknown: the measured effect on
    an early checkpoint was 32 boxes per image at a 62% unknown rate scoring
    exactly the same as a quiet configuration. The operator still has to look at
    those boxes.
    """
    if prod["precision"] < min_precision or prod["recall"] < min_recall:
        return None
    if max_fp_per_image is not None and prod["fp_per_image"] > max_fp_per_image:
        return None
    if objective == "production_score":
        return round(0.60 * prod["f1"] + 0.40 * (prod["map_50"] or 0.0)
                     - 0.02 * prod["fp_per_image"]
                     - 0.05 * float(prod.get("unknown_rate") or 0.0), 6)
    value = prod.get(objective)
    return None if value is None else float(value)


def sweep(gts: Sequence[Mapping], cache: CandidateCache,
          decode_floors: Sequence[float] = DECODE_FLOORS,
          unknown_floors: Sequence[float] = UNKNOWN_FLOORS,
          strictness_values: Sequence[float] = STRICTNESS_VALUES,
          objective: str = "production_score",
          max_fp_per_image: Optional[float] = None,
          min_precision: float = 0.0,
          min_recall: float = 0.0,
          thresholds: Optional[Mapping[str, float]] = None,
          check_plausibility: bool = True,
          iou_thr: float = em.DEFAULT_IOU,
          log=None) -> dict:
    """Evaluate the ``decode_floor`` × ``unknown_floor`` × ``strictness`` grid.

    Why ``strictness`` is in here
    ----------------------------
    Sweeping only the two floors cannot find a precision/recall operating point,
    and this is a property of the cascade rather than of any particular
    checkpoint. :func:`~rtsp_backend.electrical.postprocess.confidence_gate`
    asserts a class when ``score >= threshold_for(class)``; the taxonomy's
    thresholds sit around 0.38–0.40. Every floor in the default grid is at or
    below 0.25, so no choice of either floor can change which boxes clear their
    class threshold — the asserted set, and therefore precision, recall and mAP,
    is invariant across the whole 35-point grid. Measured on an early checkpoint:
    35 points, exactly one distinct (precision, recall, mAP@0.5) triple. The
    floors move only the abstention rate and the accept/reject counts.

    ``strictness`` is the global multiplier on every per-class threshold, so it is
    the dimension that actually trades precision against recall. Without it the
    sweep reports a flat surface and "the best operating point" is decided by
    tie-break.

    The grid is scored with a single-IoU pass (cheap); the winner is then
    recomputed in full so the reported mAP@0.5:0.95 is real and not an
    extrapolation.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    floors = sorted({round(float(d), 4) for d in decode_floors})
    too_low = [d for d in floors if d < cache.base_decode_floor - 1e-9]
    if too_low:
        raise ValueError(
            f"decode floors {too_low} are below the cached {cache.base_decode_floor}")

    rows: list[dict] = []
    for d in floors:
        for u in sorted({round(float(x), 4) for x in unknown_floors}):
            for st in sorted({round(float(x), 4) for x in strictness_values}):
                rep = production_report(gts, cache, d, u, thresholds=thresholds,
                                        strictness=st,
                                        check_plausibility=check_plausibility,
                                        iou_thr=iou_thr, full=False)
                prod = rep["production"]
                prod["objective"] = objective
                prod["score"] = score_point(prod, objective, max_fp_per_image,
                                            min_precision, min_recall)
                rows.append(prod)
                if log:
                    s = prod["score"]
                    log(f"  decode={d:<5} unknown={u:<5} strict={st:<5} "
                        f"P={prod['precision']:.3f} R={prod['recall']:.3f} "
                        f"F1={prod['f1']:.3f} mAP50={prod['map_50']:.3f} "
                        f"FP/img={prod['fp_per_image']:.2f} "
                        f"unk={prod['unknown_rate']:.2f} "
                        f"score={'rejected' if s is None else f'{s:.4f}'}")

    eligible = [r for r in rows if r["score"] is not None]
    result = {
        "grid": rows,
        "objective": objective,
        "constraints": {"max_fp_per_image": max_fp_per_image,
                        "min_precision": min_precision,
                        "min_recall": min_recall},
        "points_evaluated": len(rows),
        "points_eligible": len(eligible),
        "backend_id": cache.backend_id,
        "split": cache.split,
        "images": cache.image_count,
    }
    if not eligible:
        result["status"] = "no_eligible_operating_point"
        result["reason"] = ("every point in the grid violated the constraints; "
                           "relax max_fp_per_image / min_precision / min_recall")
        return result

    # Deterministic tie-break, because a flat region of the surface is normal
    # rather than exceptional: prefer the higher floors (a quieter, cheaper
    # configuration that discards more before the gate) and, among those, the
    # strictness closest to the documented taxonomy default — do not deviate from
    # the defaults without a measured gain.
    best = max(eligible, key=lambda r: (r["score"], r["decode_floor"],
                                        r["unknown_floor"],
                                        -abs(r["strictness"] - 1.0)))
    if log:
        log(f"  best: decode={best['decode_floor']} "
            f"unknown={best['unknown_floor']} strictness={best['strictness']} "
            f"score={best['score']:.4f}")
    full_rep = production_report(gts, cache, best["decode_floor"],
                                 best["unknown_floor"], thresholds=thresholds,
                                 strictness=best["strictness"],
                                 check_plausibility=check_plausibility,
                                 iou_thr=iou_thr, full=True)
    full_rep["production"]["objective"] = objective
    full_rep["production"]["score"] = score_point(
        full_rep["production"], objective, max_fp_per_image, min_precision,
        min_recall)
    result["status"] = "swept"
    result["best"] = full_rep["production"]
    result["best_report"] = full_rep
    return result


def refine_per_class(gts: Sequence[Mapping], cache: CandidateCache,
                     decode_floor: float, unknown_floor: float,
                     objective: str = "f1", min_precision: float = 0.0,
                     rank_by: str = "production_score",
                     iou_thr: float = em.DEFAULT_IOU,
                     log=None) -> dict:
    """Derive per-class thresholds at an operating point and keep them only if
    they actually help.

    ``optimise_thresholds`` maximises each class's own F1 in isolation, which is
    not the same as improving the system. So the refined thresholds are re-run
    through the production path and adopted only if the ranking metric improves
    — otherwise the taxonomy defaults stand.
    """
    base = production_report(gts, cache, decode_floor, unknown_floor,
                             iou_thr=iou_thr, full=True)
    base_prod = base["production"]
    base_score = score_point(base_prod, rank_by, min_precision=0.0)

    asserted = replay_gate(cache, decode_floor,
                           gate_config(unknown_floor))["asserted"]
    rec = em.optimise_thresholds(gts, asserted, objective=objective,
                                 min_precision=min_precision)
    thresholds = {cid: v["recommended_threshold"] for cid, v in rec.items()}
    if not thresholds:
        return {"status": "no_recommendation", "baseline": base_prod,
                "adopted": False, "thresholds": {}}

    tuned = production_report(gts, cache, decode_floor, unknown_floor,
                              thresholds=thresholds, iou_thr=iou_thr, full=True)
    tuned_prod = tuned["production"]
    tuned_score = score_point(tuned_prod, rank_by, min_precision=0.0)
    adopted = (tuned_score is not None and base_score is not None
               and tuned_score > base_score)
    if log:
        log(f"  per-class thresholds: baseline {rank_by}="
            f"{base_score:.4f} -> tuned {tuned_score:.4f} "
            f"({'adopted' if adopted else 'rejected, defaults kept'})")
    return {
        "status": "refined",
        "rank_by": rank_by,
        "baseline": base_prod,
        "baseline_score": base_score,
        "tuned": tuned_prod,
        "tuned_score": tuned_score,
        "adopted": adopted,
        "thresholds": thresholds if adopted else {},
        "candidate_thresholds": thresholds,
        "detail": rec,
        "report": tuned if adopted else base,
    }


# --------------------------------------------------------------------------
# presentation
# --------------------------------------------------------------------------

_SWEEP_COLUMNS = (
    ("decode", "decode_floor", "{:.2f}"),
    ("unknown", "unknown_floor", "{:.2f}"),
    ("strict", "strictness", "{:.2f}"),
    ("P", "precision", "{:.3f}"),
    ("R", "recall", "{:.3f}"),
    ("F1", "f1", "{:.3f}"),
    ("mAP50", "map_50", "{:.3f}"),
    ("FP/img", "fp_per_image", "{:.2f}"),
    ("FN/img", "fn_per_image", "{:.2f}"),
    ("unk%", "unknown_rate", "{:.2f}"),
    ("acc", "accepted_detections", "{:.0f}"),
    ("rej", "rejected_detections", "{:.0f}"),
)


def format_sweep(result: Mapping, top: Optional[int] = None) -> str:
    """Render a sweep as a fixed-width table, best first."""
    rows = list(result.get("grid") or [])
    if not rows:
        return "(no operating points evaluated)"
    ranked = sorted(rows, key=lambda r: (r["score"] is not None,
                                         r["score"] if r["score"] is not None
                                         else 0.0), reverse=True)
    if top:
        ranked = ranked[:top]
    head = [h for h, _, _ in _SWEEP_COLUMNS] + ["score"]
    widths = [max(len(h), 7) for h in head]
    lines = ["  ".join(h.rjust(w) for h, w in zip(head, widths))]
    lines.append("  ".join("-" * w for w in widths))
    for r in ranked:
        cells = []
        for _, key, fmt in _SWEEP_COLUMNS:
            v = r.get(key)
            cells.append("-" if v is None else fmt.format(v))
        s = r.get("score")
        cells.append("rejected" if s is None else f"{s:.4f}")
        lines.append("  ".join(c.rjust(w) for c, w in zip(cells, widths)))
    best = result.get("best")
    if best:
        lines.append("")
        lines.append(f"chosen: decode_floor={best['decode_floor']} "
                     f"unknown_floor={best['unknown_floor']} "
                     f"strictness={best['strictness']} "
                     f"({result.get('objective')}={best.get('score')})")
    return "\n".join(lines)


def format_production(prod: Mapping) -> str:
    """Render the production metric block an acceptance report quotes."""
    return "\n".join([
        "Production metrics (production inference path, not Ultralytics val)",
        f"  images evaluated      {prod['images']}",
        f"  ground-truth boxes    {prod['ground_truth']}",
        f"  decode_floor          {prod['decode_floor']}",
        f"  unknown_floor         {prod['unknown_floor']}",
        f"  strictness            {prod['strictness']}",
        "",
        f"  Precision             {prod['precision']:.4f}",
        f"  Recall                {prod['recall']:.4f}",
        f"  F1                    {prod['f1']:.4f}",
        f"  mAP@0.5               {prod['map_50']:.4f}",
        f"  mAP@0.5:0.95          "
        + ("n/a" if prod['map_50_95'] is None else f"{prod['map_50_95']:.4f}"),
        "",
        f"  FP per image          {prod['fp_per_image']:.4f}",
        f"  FN per image          {prod['fn_per_image']:.4f}",
        f"  Unknown rate          {prod['unknown_rate']:.4f}",
        f"  Accepted detections   {prod['accepted_detections']}"
        f" (asserted {prod['accepted_asserted']},"
        f" unknown {prod['accepted_unknown']})",
        f"  Rejected detections   {prod['rejected_detections']}",
        "",
        f"  Localised recall      {prod['recall_localised']:.4f}"
        "   (class-agnostic ceiling)",
        f"  Classification gap    {prod['classification_shortfall']:.4f}"
        "   (localised but not named)",
    ])
