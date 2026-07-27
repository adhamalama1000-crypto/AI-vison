"""
Detection post-processing: the false-positive suppression and honesty gate.

The old pipeline had exactly one filter — a single global confidence threshold —
and a class-agnostic NMS applied across all classes at once. That combination is
what produced hundreds of junk boxes: any activation above 0.25 became a
detection, no matter how geometrically impossible, and legitimately adjacent
devices of different classes suppressed each other.

This module replaces it with an explicit, ordered and *measurable* cascade:

1. :func:`sanitise` — clip to the image, drop degenerate/zero-area boxes.
2. :func:`nms_per_class` — NMS **within** each class (keeps two adjacent
   devices of different classes; removes duplicate boxes on the same device).
3. :func:`dedupe_across_classes` — resolve genuine duplicates of the *same*
   physical device claimed by two classes, using IoU and containment, keeping
   the higher-scoring claim. Confusable pairs (mcb/mccb, relay/timer_relay) are
   resolved aggressively; unrelated classes are left alone.
4. :func:`plausibility_gate` — reject boxes whose aspect ratio or relative area
   cannot physically be that device (taxonomy priors).
5. :func:`confidence_gate` — per-class acceptance threshold. Above it, the class
   is asserted. Between the floor and the threshold the detection is kept but
   relabelled :data:`~.taxonomy.UNKNOWN_COMPONENT_ID` — "there is a device here,
   I will not guess what it is". Below the floor it is dropped.
6. :func:`group_rows` — cluster accepted devices into DIN-rail rows, which is
   real structural understanding of the panel and feeds the layout description.

Every stage records *why* it dropped each box, so the effect of the cascade is
quantifiable (see :class:`Diagnostics`) instead of a matter of opinion. All
functions are pure and dependency-light (numpy only) — they are unit-tested
directly against synthetic candidate sets.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Iterable, Optional, Sequence

import numpy as np

from . import taxonomy as tax

# Classes that a detector genuinely confuses because the devices look alike.
# Only within these groups do we resolve a same-device double claim by score;
# across unrelated classes an overlap is usually two real stacked devices
# (e.g. an overload relay bolted under a contactor) and must be preserved.
CONFUSABLE_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"mcb", "mccb", "acb", "rcbo", "rccb", "motor_starter"}),
    frozenset({"relay", "timer_relay", "safety_relay", "protection_relay",
               "signal_isolator"}),
    frozenset({"vfd", "soft_starter", "servo_drive"}),
    frozenset({"power_supply", "ups", "transformer"}),
    frozenset({"push_button", "indicator_lamp", "selector_switch",
               "emergency_stop"}),
    frozenset({"fuse", "fuse_holder"}),
    frozenset({"busbar", "neutral_bar", "earth_bar"}),
    frozenset({"terminal_block", "wire_duct", "din_rail"}),
    frozenset({"ethernet_switch", "industrial_router", "io_module"}),
    frozenset({"energy_meter", "ammeter", "pf_controller", "ats_controller"}),
)


def _confusable(a: str, b: str) -> bool:
    if a == b:
        return True
    return any(a in g and b in g for g in CONFUSABLE_GROUPS)


@dataclass
class Candidate:
    """A raw model output before gating. ``class_id`` must be canonical."""

    class_id: str
    score: float
    box: tuple[float, float, float, float]   # x1, y1, x2, y2 in image pixels
    source: str = "model"                    # which backend produced it
    raw_label: Optional[str] = None          # label as the model emitted it
    extra: dict = field(default_factory=dict)

    @property
    def width(self) -> float:
        return max(0.0, self.box[2] - self.box[0])

    @property
    def height(self) -> float:
        return max(0.0, self.box[3] - self.box[1])

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0)


@dataclass
class Diagnostics:
    """Why boxes were removed — the evidence that the cascade is doing work."""

    input_count: int = 0
    output_count: int = 0
    dropped: Counter = field(default_factory=Counter)
    relabelled_unknown: int = 0

    @property
    def dropped_total(self) -> int:
        return int(sum(self.dropped.values()))

    def suppression_rate(self) -> float:
        if not self.input_count:
            return 0.0
        return round(self.dropped_total / self.input_count, 4)

    def to_dict(self) -> dict:
        return {
            "input_count": self.input_count,
            "output_count": self.output_count,
            "dropped_total": self.dropped_total,
            "dropped_by_reason": dict(self.dropped),
            "relabelled_unknown": self.relabelled_unknown,
            "suppression_rate": self.suppression_rate(),
        }


@dataclass
class GateConfig:
    """Tunables for the cascade. Defaults are the production settings."""

    #: NMS IoU inside one class.
    nms_iou: float = 0.50
    #: IoU above which two confusable classes are treated as the same device.
    cross_class_iou: float = 0.65
    #: Containment fraction above which the smaller box is a duplicate.
    containment: float = 0.80
    #: Per-class acceptance thresholds; falls back to the taxonomy value.
    thresholds: dict[str, float] = field(default_factory=tax.default_thresholds)
    #: Global multiplier on every threshold (operator "strictness" dial).
    strictness: float = 1.0
    #: Below this score nothing is kept at all, not even as unknown.
    unknown_floor: float = 0.18
    #: Skip the geometric plausibility gate (for debugging a new detector).
    check_plausibility: bool = True
    #: Tolerance on the taxonomy aspect-ratio band (multiplicative slack).
    aspect_slack: float = 1.35
    #: Tolerance on the taxonomy relative-area band (multiplicative slack).
    area_slack: float = 2.0
    #: Hard cap on returned detections. A real panel photo has tens of devices,
    #: not hundreds; exceeding this means the model is misbehaving and we keep
    #: only the most confident, reporting the truncation honestly.
    max_detections: int = 200
    #: Row clustering tolerance as a fraction of mean device height.
    row_tolerance: float = 0.6

    def threshold_for(self, class_id: str) -> float:
        base = self.thresholds.get(class_id, tax.spec(class_id).min_conf)
        return float(min(0.99, max(0.01, base * self.strictness)))


@dataclass
class GateResult:
    accepted: list[Candidate]
    diagnostics: Diagnostics
    rows: list[list[int]] = field(default_factory=list)
    truncated: bool = False


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------

def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def containment(inner: Sequence[float], outer: Sequence[float]) -> float:
    """Fraction of ``inner``'s area that lies inside ``outer``."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    return float(inter / area) if area > 0 else 0.0


# --------------------------------------------------------------------------
# stage 1 — sanitise
# --------------------------------------------------------------------------

def sanitise(cands: Iterable[Candidate], image_shape: Sequence[int],
             diag: Optional[Diagnostics] = None,
             min_side: float = 4.0) -> list[Candidate]:
    """Clip boxes into the frame and remove degenerate ones.

    A box entirely outside the frame, inverted, or thinner than ``min_side``
    pixels on either side cannot be a device and is dropped. The old pipeline
    let these through, which is one reason overlays showed slivers.
    """
    h, w = int(image_shape[0]), int(image_shape[1])
    out: list[Candidate] = []
    for c in cands:
        x1, y1, x2, y2 = (float(v) for v in c.box)
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            if diag:
                diag.dropped["non_finite_box"] += 1
            continue
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        x1, y1 = max(0.0, min(x1, w - 1.0)), max(0.0, min(y1, h - 1.0))
        x2, y2 = max(0.0, min(x2, float(w))), max(0.0, min(y2, float(h)))
        if (x2 - x1) < min_side or (y2 - y1) < min_side:
            if diag:
                diag.dropped["degenerate_box"] += 1
            continue
        if not math.isfinite(c.score) or c.score <= 0.0:
            if diag:
                diag.dropped["invalid_score"] += 1
            continue
        out.append(replace(c, box=(x1, y1, x2, y2), score=float(min(1.0, c.score))))
    return out


# --------------------------------------------------------------------------
# stage 2 — per-class NMS
# --------------------------------------------------------------------------

def nms_per_class(cands: Sequence[Candidate], iou_thr: float,
                  diag: Optional[Diagnostics] = None) -> list[Candidate]:
    """Greedy NMS applied independently per class.

    Applying NMS per class (rather than across all classes at once, as the old
    ONNX path did) is what allows an overload relay bolted underneath a
    contactor, or two touching MCBs, to both survive.
    """
    by_class: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(cands):
        by_class[c.class_id].append(i)

    keep: list[int] = []
    for cls, idxs in by_class.items():
        idxs.sort(key=lambda i: cands[i].score, reverse=True)
        chosen: list[int] = []
        for i in idxs:
            if any(iou(cands[i].box, cands[j].box) > iou_thr for j in chosen):
                if diag:
                    diag.dropped["nms_same_class"] += 1
                continue
            chosen.append(i)
        keep.extend(chosen)
    keep.sort(key=lambda i: cands[i].score, reverse=True)
    return [cands[i] for i in keep]


# --------------------------------------------------------------------------
# stage 3 — cross-class duplicate resolution
# --------------------------------------------------------------------------

def dedupe_across_classes(cands: Sequence[Candidate], iou_thr: float,
                          contain_thr: float,
                          diag: Optional[Diagnostics] = None) -> list[Candidate]:
    """Resolve two *confusable* classes claiming the same physical device.

    Overlap between unrelated classes is left intact — in a real panel devices
    are physically stacked and nested (relay in a socket, CT around a busbar,
    overload under a contactor), so blanket class-agnostic suppression destroys
    correct detections.
    """
    order = sorted(range(len(cands)), key=lambda i: cands[i].score, reverse=True)
    kept: list[int] = []
    for i in order:
        drop = False
        for j in kept:
            if not _confusable(cands[i].class_id, cands[j].class_id):
                continue
            if iou(cands[i].box, cands[j].box) > iou_thr:
                drop = True
                break
            # a small box almost entirely inside a kept confusable box is the
            # same device detected at a tighter crop
            if containment(cands[i].box, cands[j].box) > contain_thr:
                drop = True
                break
        if drop:
            if diag:
                diag.dropped["duplicate_class_claim"] += 1
            continue
        kept.append(i)
    kept.sort(key=lambda i: cands[i].score, reverse=True)
    return [cands[i] for i in kept]


# --------------------------------------------------------------------------
# stage 4 — geometric plausibility
# --------------------------------------------------------------------------

def plausible(cand: Candidate, image_area: float, cfg: GateConfig
              ) -> tuple[bool, str]:
    """Can this box physically be this device? Returns ``(ok, reason)``."""
    if cand.class_id == tax.UNKNOWN_COMPONENT_ID:
        return True, ""
    sp = tax.spec(cand.class_id)
    if cand.height <= 0 or image_area <= 0:
        return False, "degenerate_box"

    ar = cand.width / cand.height
    lo, hi = sp.aspect_ratio
    if not (lo / cfg.aspect_slack) <= ar <= (hi * cfg.aspect_slack):
        return False, "implausible_aspect_ratio"

    rel = cand.area / image_area
    a_lo, a_hi = sp.rel_area
    if rel < (a_lo / cfg.area_slack):
        return False, "implausible_too_small"
    if rel > min(1.0, a_hi * cfg.area_slack):
        return False, "implausible_too_large"
    return True, ""


def plausibility_gate(cands: Sequence[Candidate], image_shape: Sequence[int],
                      cfg: GateConfig,
                      diag: Optional[Diagnostics] = None) -> list[Candidate]:
    if not cfg.check_plausibility:
        return list(cands)
    image_area = float(image_shape[0]) * float(image_shape[1])
    out: list[Candidate] = []
    for c in cands:
        ok, reason = plausible(c, image_area, cfg)
        if ok:
            out.append(c)
        elif diag:
            diag.dropped[reason] += 1
    return out


# --------------------------------------------------------------------------
# stage 5 — confidence gate with honest unknown fallback
# --------------------------------------------------------------------------

def confidence_gate(cands: Sequence[Candidate], cfg: GateConfig,
                    diag: Optional[Diagnostics] = None) -> list[Candidate]:
    """Accept, demote to unknown, or drop — never guess.

    This is the behaviour the specification demands: *"If confidence is low,
    classify the object as 'Unknown Industrial Component' instead of guessing."*
    """
    out: list[Candidate] = []
    for c in cands:
        thr = cfg.threshold_for(c.class_id)
        if c.score >= thr:
            out.append(c)
            continue
        if c.score >= cfg.unknown_floor:
            extra = dict(c.extra)
            extra["demoted_from"] = c.class_id
            extra["demoted_score"] = round(float(c.score), 4)
            extra["demotion_reason"] = (
                f"score {c.score:.2f} below the {thr:.2f} acceptance threshold "
                f"for {tax.display_name(c.class_id)}")
            out.append(replace(c, class_id=tax.UNKNOWN_COMPONENT_ID, extra=extra))
            if diag:
                diag.relabelled_unknown += 1
            continue
        if diag:
            diag.dropped["below_unknown_floor"] += 1
    return out


# --------------------------------------------------------------------------
# stage 6 — DIN-rail row structure
# --------------------------------------------------------------------------

def group_rows(cands: Sequence[Candidate], tolerance: float = 0.6
               ) -> list[list[int]]:
    """Cluster detections into horizontal rows (DIN rails / device banks).

    Devices in a control panel are organised in horizontal rows. Recovering
    that structure is genuine understanding of the panel layout: it drives the
    layout description in the report and gives an operator a spatial index
    ("row 2, position 4") instead of raw pixel coordinates.
    """
    if not cands:
        return []
    idxs = sorted(range(len(cands)), key=lambda i: cands[i].center[1])
    heights = [c.height for c in cands if c.height > 0]
    band = (float(np.median(heights)) if heights else 20.0) * max(0.1, tolerance)

    rows: list[list[int]] = []
    current: list[int] = [idxs[0]]
    anchor = cands[idxs[0]].center[1]
    for i in idxs[1:]:
        cy = cands[i].center[1]
        if abs(cy - anchor) <= band:
            current.append(i)
        else:
            rows.append(current)
            current = [i]
        # running anchor = mean of the row keeps long rows from drifting apart
        anchor = float(np.mean([cands[k].center[1] for k in current]))
    rows.append(current)
    for row in rows:
        row.sort(key=lambda i: cands[i].center[0])
    return rows


# --------------------------------------------------------------------------
# the cascade
# --------------------------------------------------------------------------

def run(cands: Sequence[Candidate], image_shape: Sequence[int],
        cfg: Optional[GateConfig] = None) -> GateResult:
    """Run the full post-processing cascade and report what it removed."""
    cfg = cfg or GateConfig()
    diag = Diagnostics(input_count=len(cands))

    stage = sanitise(cands, image_shape, diag)
    stage = nms_per_class(stage, cfg.nms_iou, diag)
    stage = dedupe_across_classes(stage, cfg.cross_class_iou, cfg.containment, diag)
    stage = plausibility_gate(stage, image_shape, cfg, diag)
    stage = confidence_gate(stage, cfg, diag)

    stage.sort(key=lambda c: c.score, reverse=True)
    truncated = False
    if len(stage) > cfg.max_detections:
        diag.dropped["over_detection_cap"] += len(stage) - cfg.max_detections
        stage = stage[:cfg.max_detections]
        truncated = True

    # Present results in reading order (top-to-bottom, left-to-right) — how an
    # engineer walks a panel — rather than by score.
    rows = group_rows(stage, cfg.row_tolerance)
    ordered: list[Candidate] = []
    ordered_rows: list[list[int]] = []
    for row in rows:
        start = len(ordered)
        for i in row:
            ordered.append(stage[i])
        ordered_rows.append(list(range(start, len(ordered))))

    diag.output_count = len(ordered)
    return GateResult(accepted=ordered, diagnostics=diag, rows=ordered_rows,
                      truncated=truncated)


# --------------------------------------------------------------------------
# aggregation helpers used by the report layer
# --------------------------------------------------------------------------

def panel_position(box: Sequence[float], image_shape: Sequence[int]) -> str:
    """Coarse 3×3 position label ("middle-left") for a box."""
    h, w = float(image_shape[0]), float(image_shape[1])
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    col = "left" if cx < w / 3 else ("center" if cx < 2 * w / 3 else "right")
    row = "top" if cy < h / 3 else ("middle" if cy < 2 * h / 3 else "bottom")
    return f"{row}-{col}"


def counts(cands: Sequence[Candidate]) -> dict[str, int]:
    c: Counter = Counter(x.class_id for x in cands)
    return dict(c)


def confidence_stats(cands: Sequence[Candidate]) -> dict:
    """Mean/median/min/max plus a low-confidence tally, for the report."""
    if not cands:
        return {"count": 0, "mean": None, "median": None, "min": None,
                "max": None, "below_0_5": 0, "unknown": 0}
    scores = np.array([c.score for c in cands], dtype=np.float64)
    return {
        "count": int(scores.size),
        "mean": round(float(scores.mean()), 4),
        "median": round(float(np.median(scores)), 4),
        "min": round(float(scores.min()), 4),
        "max": round(float(scores.max()), 4),
        "below_0_5": int((scores < 0.5).sum()),
        "unknown": sum(1 for c in cands if c.class_id == tax.UNKNOWN_COMPONENT_ID),
    }


__all__ = [
    "Candidate", "Diagnostics", "GateConfig", "GateResult",
    "CONFUSABLE_GROUPS", "iou", "containment", "sanitise", "nms_per_class",
    "dedupe_across_classes", "plausible", "plausibility_gate",
    "confidence_gate", "group_rows", "run", "panel_position", "counts",
    "confidence_stats",
]
