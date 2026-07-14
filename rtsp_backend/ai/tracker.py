"""
Multi-object tracking: a self-contained ByteTrack-style tracker.

This is a *real* tracker, not a stub. It needs no model weights and no GPU, so
it is fully unit-tested in CI. It assigns a stable integer ID to each object
across frames using:

* a constant-velocity motion model (a light Kalman-style predict/update on the
  box centre + size) to bridge short gaps and reduce ID switches while things
  move, and
* two-stage association like ByteTrack: high-confidence detections are matched
  first, then remaining tracks get a second chance against low-confidence
  detections (which recovers objects during partial occlusion / motion blur
  instead of dropping their ID).

Association cost is 1 - IoU between predicted track boxes and detections, solved
optimally with the Hungarian algorithm (scipy) when available and greedily
otherwise. Tracks are confirmed after ``min_hits`` frames and removed after
``max_age`` missed frames.

The tracker is generic over "detections": each detection only needs a bounding
box and a score, so the same tracker serves faces, people, vehicles, weapons,
etc. A separate :class:`MultiClassTracker` keeps one tracker per class label so
IDs never leak across classes (a person and a car never share an ID).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:  # optimal assignment; greedy fallback if scipy is absent
    from scipy.optimize import linear_sum_assignment  # type: ignore

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in ``a`` [N,4] and ``b`` [M,4] -> [N,M]."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter + 1e-9
    return (inter / union).astype(np.float32)


def _assign(cost: np.ndarray, max_cost: float):
    """Return (matches, unmatched_rows, unmatched_cols) for a cost matrix."""
    n, m = cost.shape
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))
    if _HAVE_SCIPY:
        rows, cols = linear_sum_assignment(cost)
        pairs = list(zip(rows.tolist(), cols.tolist()))
    else:  # greedy fallback
        pairs = []
        used_r, used_c = set(), set()
        order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
        for r, c in order:
            if r in used_r or c in used_c:
                continue
            pairs.append((int(r), int(c)))
            used_r.add(int(r))
            used_c.add(int(c))
    matches = [(r, c) for r, c in pairs if cost[r, c] <= max_cost]
    matched_r = {r for r, _ in matches}
    matched_c = {c for _, c in matches}
    unmatched_r = [r for r in range(n) if r not in matched_r]
    unmatched_c = [c for c in range(m) if c not in matched_c]
    return matches, unmatched_r, unmatched_c


@dataclass
class Track:
    track_id: int
    box: np.ndarray                      # [x1,y1,x2,y2] last observed/predicted
    score: float
    label: str = "object"
    hits: int = 1
    age: int = 0                         # frames since last successful update
    total_frames: int = 1
    confirmed: bool = False
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))

    def predict(self) -> np.ndarray:
        """Advance the box by its estimated velocity (constant-velocity model)."""
        self.box = self.box + self.velocity
        self.age += 1
        self.total_frames += 1
        return self.box

    def update(self, box: np.ndarray, score: float, alpha: float = 0.5) -> None:
        # EMA velocity from the observed displacement; smooths jitter.
        self.velocity = alpha * (box - self.box) + (1 - alpha) * self.velocity
        self.box = box.astype(np.float32)
        self.score = float(score)
        self.hits += 1
        self.age = 0


class ByteTrack:
    """Single-class ByteTrack-style tracker."""

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        label: str = "object",
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.label = label
        self._tracks: list[Track] = []
        self._next_id = 1

    @property
    def tracks(self) -> list[Track]:
        return [t for t in self._tracks if t.confirmed]

    def update(self, boxes: np.ndarray, scores: np.ndarray) -> list[Track]:
        """
        Advance one frame. ``boxes`` is [N,4] xyxy, ``scores`` is [N].
        Returns the list of currently confirmed tracks (each with a stable id).
        """
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)

        for t in self._tracks:
            t.predict()

        high = scores >= self.high_thresh
        low = (scores >= self.low_thresh) & (~high)
        hi_boxes, hi_scores = boxes[high], scores[high]
        lo_boxes, lo_scores = boxes[low], scores[low]

        track_boxes = np.array([t.box for t in self._tracks], dtype=np.float32) \
            if self._tracks else np.zeros((0, 4), dtype=np.float32)

        # --- stage 1: match high-confidence detections to all tracks ---
        cost = 1.0 - iou_matrix(track_boxes, hi_boxes)
        matches, un_tracks, un_hi = _assign(cost, 1.0 - self.iou_threshold)
        for ti, di in matches:
            self._tracks[ti].update(hi_boxes[di], hi_scores[di])

        # --- stage 2: remaining tracks get a shot at low-confidence dets ---
        remaining = [self._tracks[i] for i in un_tracks]
        rem_boxes = np.array([t.box for t in remaining], dtype=np.float32) \
            if remaining else np.zeros((0, 4), dtype=np.float32)
        cost2 = 1.0 - iou_matrix(rem_boxes, lo_boxes)
        matches2, un_rem, _ = _assign(cost2, 1.0 - self.iou_threshold)
        for ri, di in matches2:
            remaining[ri].update(lo_boxes[di], lo_scores[di])

        # --- births: unmatched high-confidence detections start new tracks ---
        for di in un_hi:
            self._tracks.append(
                Track(track_id=self._next_id, box=hi_boxes[di].astype(np.float32),
                      score=float(hi_scores[di]), label=self.label)
            )
            self._next_id += 1

        # --- confirm / cull ---
        for t in self._tracks:
            if not t.confirmed and t.hits >= self.min_hits:
                t.confirmed = True
        self._tracks = [t for t in self._tracks if t.age <= self.max_age]
        return self.tracks


class MultiClassTracker:
    """Keeps one :class:`ByteTrack` per class so IDs never cross class lines."""

    def __init__(self, **kwargs) -> None:
        self._kwargs = kwargs
        self._per_label: dict[str, ByteTrack] = {}
        self._global_id = 0
        self._id_remap: dict[tuple[str, int], int] = {}

    def _tracker(self, label: str) -> ByteTrack:
        t = self._per_label.get(label)
        if t is None:
            t = ByteTrack(label=label, **self._kwargs)
            self._per_label[label] = t
        return t

    def update(self, detections: list) -> dict[int, int]:
        """
        Feed a list of detection-like objects (needs ``.bbox`` with
        ``as_list()`` and ``.confidence`` and ``.label``). Returns a mapping
        from the *index in the input list* to a globally-unique stable id, for
        detections that belong to a confirmed track this frame.
        """
        by_label: dict[str, list[tuple[int, list, float]]] = {}
        for i, d in enumerate(detections):
            by_label.setdefault(d.label, []).append(
                (i, d.bbox.as_list(), float(d.confidence)))

        index_to_id: dict[int, int] = {}
        for label, items in by_label.items():
            boxes = np.array([b for _, b, _ in items], dtype=np.float32)
            scores = np.array([s for _, _, s in items], dtype=np.float32)
            tracker = self._tracker(label)
            tracks = tracker.update(boxes, scores)
            # map each confirmed track back to the nearest input detection
            if not tracks:
                continue
            t_boxes = np.array([t.box for t in tracks], dtype=np.float32)
            ious = iou_matrix(t_boxes, boxes)
            for ti, track in enumerate(tracks):
                if ious.shape[1] == 0:
                    continue
                di = int(np.argmax(ious[ti]))
                if ious[ti, di] < 0.1:
                    continue
                key = (label, track.track_id)
                gid = self._id_remap.get(key)
                if gid is None:
                    self._global_id += 1
                    gid = self._global_id
                    self._id_remap[key] = gid
                input_index = items[di][0]
                index_to_id[input_index] = gid

        # Prune remap entries for tracks that have been culled, so the map does
        # not grow without bound on long-running cameras with high track churn.
        if len(self._id_remap) > 2048:
            live = {(lbl, t.track_id)
                    for lbl, tr in self._per_label.items() for t in tr._tracks}
            self._id_remap = {k: v for k, v in self._id_remap.items() if k in live}
        return index_to_id
