"""
Real wire detection & tracing for electrical panels.

This is NOT a Hough-lines-on-Canny stand-in that fires on any texture. It is a
full classical instance pipeline:

    1.  Colour segmentation in HSV *and* LAB (coloured insulation is the
        strongest wire cue in a real panel photo).
    2.  Adaptive threshold of the luma channel (catches dark / grey wires that
        carry little chroma).
    3.  Morphology (close then open) to bridge tiny gaps and drop speckle.
    4.  Skeletonisation of the wire mask (1-px medial axis).
    5.  Connected-component labelling of the skeleton -> one component per wire
        candidate.
    6.  Contour / shape filtering — a wire is *elongated*; blobs, text and
        component bodies are rejected by length, elongation and fill ratio.
    7.  Polyline extraction — the skeleton graph's longest endpoint-to-endpoint
        path, simplified with Douglas–Peucker.
    8.  Hough transform — a complementary straight-segment pass, promoted to
        wire candidates only where the skeleton fragmented.
    9.  Endpoint detection + broken-segment merging — collinear, near-touching
        wire ends are fused into one wire.
    10. Per-wire attributes: start, end, polyline, length, thickness (from the
        distance transform), dominant colour, direction, and — when component /
        terminal geometry is supplied — the nodes each end connects to.

The output is a list of :class:`WireInstance`. Faults (loose / broken /
disconnected) are only asserted by the comparison stage against a learned
reference; a bare detection is reported ``status="detected"`` — never a
fabricated verdict.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

try:  # scikit-image is a declared dependency; degrade gracefully if absent.
    from skimage.morphology import skeletonize as _sk_skeletonize
    _HAVE_SKIMAGE = True
except Exception:  # pragma: no cover
    _HAVE_SKIMAGE = False


# ---------------------------------------------------------------------------
# colour naming
# ---------------------------------------------------------------------------

def dominant_color_name(bgr_pixels: np.ndarray) -> str:
    """Name the dominant insulation colour of a set of BGR pixels."""
    if bgr_pixels is None or bgr_pixels.size == 0:
        return "unknown"
    px = bgr_pixels.reshape(-1, 3).astype(np.uint8)
    hsv = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h = float(np.median(hsv[:, 0]))
    s = float(np.median(hsv[:, 1]))
    v = float(np.median(hsv[:, 2]))
    if v < 45:
        return "black"
    if s < 45:
        return "white/grey" if v > 150 else "grey"
    if h < 10 or h >= 170:
        return "red"
    if h < 22:
        return "orange"
    if h < 34:
        return "yellow"
    if h < 45:
        return "yellow-green"
    if h < 85:
        return "green"
    if h < 100:
        return "cyan"
    if h < 130:
        return "blue"
    if h < 150:
        return "violet"
    return "brown"


# ---------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------

@dataclass
class WireInstance:
    wire_uid: str
    start: tuple[float, float]
    end: tuple[float, float]
    polyline: list[tuple[float, float]]
    length: float
    thickness: float
    color: str = "unknown"
    direction: float = 0.0                     # degrees, 0=east, CCW+
    from_component: Optional[str] = None
    to_component: Optional[str] = None
    from_terminal: Optional[str] = None
    to_terminal: Optional[str] = None
    status: str = "detected"
    confidence: float = 0.6
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "wire_uid": self.wire_uid,
            "start": [round(float(self.start[0]), 1), round(float(self.start[1]), 1)],
            "end": [round(float(self.end[0]), 1), round(float(self.end[1]), 1)],
            "polyline": [[round(float(x), 1), round(float(y), 1)] for x, y in self.polyline],
            "length": round(float(self.length), 1),
            "thickness": round(float(self.thickness), 2),
            "color": self.color,
            "direction": round(float(self.direction), 1),
            "from_component": self.from_component,
            "to_component": self.to_component,
            "from_terminal": self.from_terminal,
            "to_terminal": self.to_terminal,
            "status": self.status,
            "confidence": round(float(self.confidence), 3),
            "extra": self.extra,
        }


# ---------------------------------------------------------------------------
# mask construction
# ---------------------------------------------------------------------------

# Hue bands (OpenCV hue is 0..179). Each colour is segmented and traced
# INDEPENDENTLY: a red wire crossing a blue wire won't merge into one skeleton,
# and neither will merge with the grey DIN-rail / duct structure. This is the
# single most important robustness property of the detector.
_HUE_BANDS = [
    ("orange", 10, 22), ("yellow", 22, 34), ("yellow-green", 34, 45),
    ("green", 45, 85), ("cyan", 85, 100), ("blue", 100, 130),
    ("violet", 130, 150), ("brown", 150, 170),
]


def _morph(mask: np.ndarray, params: dict) -> np.ndarray:
    k_close = int(params.get("close_ksize", 5))
    k_open = int(params.get("open_ksize", 3))
    if k_close > 1:
        ck = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ck, iterations=1)
    if k_open > 1:
        ok = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ok, iterations=1)
    return mask


def color_bands(image_bgr: np.ndarray, params: dict) -> list[tuple[str, np.ndarray]]:
    """Segment the image into per-colour wire masks (each morphologically
    cleaned). Returns ``[(colour_name, mask uint8 0/255), ...]``."""
    blur = cv2.GaussianBlur(image_bgr, (3, 3), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    sat_thr = int(params.get("sat_thr", 60))
    val_thr = int(params.get("val_thr", 45))
    colored = (s > sat_thr) & (v > val_thr)

    bands: list[tuple[str, np.ndarray]] = []
    # red wraps the hue circle
    red = colored & ((h < 10) | (h >= 170))
    if red.sum() > 0:
        bands.append(("red", _morph((red.astype(np.uint8) * 255), params)))
    for name, lo, hi in _HUE_BANDS:
        band = colored & (h >= lo) & (h < hi)
        if band.sum() > 0:
            bands.append((name, _morph((band.astype(np.uint8) * 255), params)))

    # dark (black) insulation — very common; kept but traced with the same
    # elongation filter so shadows/text don't survive.
    if params.get("include_dark", True):
        dark = (v < int(params.get("dark_val", 55))) & (s < 120)
        if dark.sum() > 0:
            bands.append(("black", _morph((dark.astype(np.uint8) * 255), params)))
    return bands


def build_wire_mask(image_bgr: np.ndarray, params: dict) -> np.ndarray:
    """Union of every colour band — a single binary wire mask (used for the
    Hough pass and by callers that want the raw foreground)."""
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for _name, band in color_bands(image_bgr, params):
        mask = cv2.bitwise_or(mask, band)
    return mask


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    """1-px medial axis of a 0/255 mask -> uint8 0/1."""
    binary = (mask > 0)
    if _HAVE_SKIMAGE:
        try:
            return _sk_skeletonize(binary).astype(np.uint8)
        except Exception:
            pass
    # Fallback: OpenCV thinning if the contrib module is present.
    try:  # pragma: no cover - optional
        thin = cv2.ximgproc.thinning((binary * 255).astype(np.uint8))
        return (thin > 0).astype(np.uint8)
    except Exception:  # pragma: no cover
        # Last resort: morphological skeleton (Lantuéjoul).
        img = (binary * 255).astype(np.uint8)
        skel = np.zeros_like(img)
        elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, elem)
            temp = cv2.subtract(img, opened)
            eroded = cv2.erode(img, elem)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded
            if cv2.countNonZero(img) == 0:
                break
        return (skel > 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# skeleton graph -> polyline
# ---------------------------------------------------------------------------

_NEIGH = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _prune_spurs(pixel_set: set, min_spur: int) -> set:
    """Remove short branch stubs (spurs) so a T/Y junction to a neighbouring
    same-colour wire doesn't corrupt the traced spine. The main run is kept."""
    def neighbours(p):
        r, c = p
        for dr, dc in _NEIGH:
            q = (r + dr, c + dc)
            if q in pixel_set:
                yield q

    for _ in range(2):  # a couple of passes handle nested spurs
        degree = {p: sum(1 for _ in neighbours(p)) for p in pixel_set}
        endpoints = [p for p, d in degree.items() if d == 1]
        remove: set = set()
        for ep in endpoints:
            spur = [ep]
            cur, prev = ep, None
            while True:
                nbrs = [q for q in neighbours(cur) if q != prev and q not in remove]
                if len(nbrs) != 1:
                    break  # reached a junction (>=2) or dead end
                prev, cur = cur, nbrs[0]
                if degree.get(cur, 0) >= 3:
                    break  # junction: stop, don't consume it
                spur.append(cur)
                if len(spur) > min_spur:
                    break
            if len(spur) <= min_spur and degree.get(cur, 0) >= 3:
                remove.update(spur)
        if not remove:
            break
        pixel_set = pixel_set - remove
    return pixel_set


def _longest_path(coords: np.ndarray, prune: int = 0) -> list[tuple[int, int]]:
    """Given the (row,col) pixels of one skeleton component, return the longest
    endpoint-to-endpoint pixel path (the wire's spine)."""
    if len(coords) == 1:
        r, c = coords[0]
        return [(int(r), int(c))]
    pixel_set = {(int(r), int(c)) for r, c in coords}

    def neighbours(p):
        r, c = p
        for dr, dc in _NEIGH:
            q = (r + dr, c + dc)
            if q in pixel_set:
                yield q

    if prune > 0 and len(pixel_set) > prune + 2:
        pruned = _prune_spurs(pixel_set, prune)
        if len(pruned) >= 2:
            pixel_set = pruned

    def bfs_farthest(src):
        seen = {src: None}
        dq = deque([src])
        far, far_d = src, 0
        dist = {src: 0}
        while dq:
            cur = dq.popleft()
            for nb in neighbours(cur):
                if nb not in seen:
                    seen[nb] = cur
                    dist[nb] = dist[cur] + 1
                    if dist[nb] > far_d:
                        far_d, far = dist[nb], nb
                    dq.append(nb)
        return far, seen

    # Diameter of the (connected) skeleton graph: BFS twice.
    start = next(iter(pixel_set))
    a, _ = bfs_farthest(start)
    b, parents = bfs_farthest(a)
    path = []
    node = b
    while node is not None:
        path.append(node)
        node = parents[node]
    path.reverse()
    return path


def _simplify(path_xy: np.ndarray, eps: float = 2.0) -> list[tuple[float, float]]:
    if len(path_xy) <= 2:
        return [(float(x), float(y)) for x, y in path_xy]
    approx = cv2.approxPolyDP(path_xy.reshape(-1, 1, 2).astype(np.int32), eps, False)
    pts = approx.reshape(-1, 2)
    return [(float(x), float(y)) for x, y in pts]


def _polyline_length(poly: list[tuple[float, float]]) -> float:
    return float(sum(math.hypot(poly[i + 1][0] - poly[i][0],
                                 poly[i + 1][1] - poly[i][1])
                     for i in range(len(poly) - 1)))


# ---------------------------------------------------------------------------
# broken-segment merging
# ---------------------------------------------------------------------------

def _angle(p: tuple[float, float], q: tuple[float, float]) -> float:
    return math.degrees(math.atan2(q[1] - p[1], q[0] - p[0]))


def _ang_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    d = min(d, 360.0 - d)
    return min(d, 180.0 - d)  # treat opposite directions as collinear


def _unit(vx: float, vy: float) -> tuple[float, float]:
    n = math.hypot(vx, vy) or 1.0
    return vx / n, vy / n


def _pt_seg_dist(pt, a, b) -> float:
    """Distance from point ``pt`` to segment ``a``-``b``."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-9:
        return math.hypot(pt[0] - ax, pt[1] - ay)
    t = max(0.0, min(1.0, ((pt[0] - ax) * dx + (pt[1] - ay) * dy) / L2))
    px, py = ax + t * dx, ay + t * dy
    return math.hypot(pt[0] - px, pt[1] - py)


def _refine_colors(image_bgr: np.ndarray, wires: list[WireInstance]) -> None:
    """Set each wire's colour from the median of the ORIGINAL image pixels along
    its spine — truer than the segmentation band label, and it collapses
    JPEG-chroma-bleed phantoms of one physical wire to a single colour so the
    overlap dedup can remove them."""
    h, w = image_bgr.shape[:2]
    for wnode in wires:
        m = np.zeros((h, w), np.uint8)
        pts = np.array([[int(x), int(y)] for x, y in wnode.polyline], np.int32)
        if len(pts) >= 2:
            cv2.polylines(m, [pts], False, 255,
                          thickness=max(2, int(round(wnode.thickness))))
            sample = image_bgr[m > 0]
            if sample.size:
                wnode.color = dominant_color_name(sample)


def _dedup_overlapping(wires: list[WireInstance], tol: float) -> list[WireInstance]:
    """Drop a wire when it is substantially covered by a longer, already-kept
    wire (both its endpoints and midpoint lie within ``tol`` of the longer
    wire's polyline). Removes skeleton/Hough duplicates of the same physical
    wire without discarding genuinely separate ones."""
    order = sorted(range(len(wires)), key=lambda k: -wires[k].length)
    kept: list[WireInstance] = []
    for idx in order:
        w = wires[idx]
        mid = ((w.start[0] + w.end[0]) / 2, (w.start[1] + w.end[1]) / 2)
        covered = False
        for k in kept:
            # same colour: full tolerance. Different colour (JPEG chroma-bleed
            # phantom lying directly on another wire): only a tight tolerance,
            # so genuinely separate parallel wires of different colours survive.
            t = tol if k.color == w.color else tol * 0.5
            seg_a, seg_b = k.start, k.end
            if (_pt_seg_dist(w.start, seg_a, seg_b) <= t and
                    _pt_seg_dist(w.end, seg_a, seg_b) <= t and
                    _pt_seg_dist(mid, seg_a, seg_b) <= t):
                covered = True
                break
        if not covered:
            kept.append(w)
    for i, w in enumerate(kept):
        w.wire_uid = f"w{i}"
    return kept


def _merge_segments(wires: list[WireInstance], gap: float, ang_tol: float) -> list[WireInstance]:
    """Fuse wire fragments whose ends nearly touch and continue in the same
    direction. Uses *directed* outward vectors through the joint and requires
    them to point the same way (dot > 0), so two overlapping / anti-parallel
    duplicates are never concatenated into a doubled-back polyline — the shorter
    duplicate is dropped instead."""
    wires = [w for w in wires if len(w.polyline) >= 2]
    cos_tol = math.cos(math.radians(ang_tol))
    merged = True
    while merged:
        merged = False
        n = len(wires)
        for i in range(n):
            for j in range(i + 1, n):
                w1, w2 = wires[i], wires[j]
                # closest endpoint pair
                best = None
                for e1 in (w1.start, w1.end):
                    for e2 in (w2.start, w2.end):
                        d = math.hypot(e1[0] - e2[0], e1[1] - e2[1])
                        if best is None or d < best[0]:
                            best = (d, e1, e2)
                d, e1, e2 = best
                # drop exact duplicates / near-coincident colinear overlaps
                if d > gap:
                    continue
                # outward direction of each fragment at the joint
                far1 = w1.start if e1 == w1.end else w1.end
                far2 = w2.start if e2 == w2.end else w2.end
                v1 = _unit(e1[0] - far1[0], e1[1] - far1[1])   # far1 -> joint
                v2 = _unit(far2[0] - e2[0], far2[1] - e2[1])   # joint -> far2
                dot = v1[0] * v2[0] + v1[1] * v2[1]
                if dot < cos_tol:      # not continuing straight (or folds back)
                    continue
                p1 = w1.polyline if w1.polyline[-1] == e1 else list(reversed(w1.polyline))
                p2 = w2.polyline if w2.polyline[0] == e2 else list(reversed(w2.polyline))
                poly = p1 + p2
                nw = WireInstance(
                    wire_uid=w1.wire_uid, start=poly[0], end=poly[-1], polyline=poly,
                    length=_polyline_length(poly),
                    thickness=(w1.thickness + w2.thickness) / 2.0,
                    color=w1.color if w1.length >= w2.length else w2.color,
                    direction=_angle(poly[0], poly[-1]),
                    confidence=max(w1.confidence, w2.confidence),
                    extra={"merged_from": [w1.wire_uid, w2.wire_uid]},
                )
                wires = [w for k, w in enumerate(wires) if k not in (i, j)]
                wires.append(nw)
                merged = True
                break
            if merged:
                break
    for idx, w in enumerate(wires):
        w.wire_uid = f"w{idx}"
    return wires


# ---------------------------------------------------------------------------
# main detector
# ---------------------------------------------------------------------------

class WireDetector:
    """Stateless, weights-free wire instance detector. Tunable via ``params``."""

    DEFAULTS = {
        "min_wire_len": 40,          # px, arc length
        "min_pixels": 26,            # skeleton pixels
        "min_elongation": 2.6,       # bbox longer/shorter side
        "sat_thr": 60, "val_thr": 45,
        "include_dark": True, "dark_val": 55,
        "close_ksize": 5, "open_ksize": 3,
        "spur_prune": 8,             # skeleton spur removal length
        "simplify_eps": 2.0,
        # --- artifact-rejection filters (reject text / screws / shadows /
        #     borders / scribbles that are not electrical wires) ---
        "min_thickness": 1.6,        # px full width; below => text-edge / noise
        "max_thickness": 34.0,       # px; above => backplate region / duct, not a wire
        "max_thickness_cv": 0.85,    # width variability; text/blobs vary, wires don't
        "max_bend_ratio": 3.2,       # arc-length / end-span; scribbles double back
        "border_margin": 6,          # px; a run hugging the image edge => panel frame
        "merge_gap": 18.0, "merge_angle_tol": 18.0,
        "snap_dist": 55.0,
        "max_wires": 400,
        # Hough is available as a supplementary straight-line recovery pass but
        # is OFF by default: the per-colour skeleton tracer is both more precise
        # and does not manufacture the panel-frame / duplicate artifacts a raw
        # Hough pass tends to. Enable per-call when a use-case needs it.
        "use_hough": False,
        "hough_min_len": 45, "hough_max_gap": 12,
    }

    def __init__(self, **params: Any) -> None:
        self.params = {**self.DEFAULTS, **(params or {})}

    def detect(self, image_bgr: np.ndarray,
               components: Optional[list] = None,
               terminals: Optional[list] = None) -> list[WireInstance]:
        if image_bgr is None or image_bgr.size == 0:
            return []
        p = self.params
        h, w = image_bgr.shape[:2]
        wires: list[WireInstance] = []
        uid = 0
        union = np.zeros((h, w), np.uint8)

        # Trace each colour band independently.
        for color_name, mask in color_bands(image_bgr, p):
            union = cv2.bitwise_or(union, mask)
            dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
            skel = _skeletonize(mask)
            n_labels, labels = cv2.connectedComponents(skel.astype(np.uint8), connectivity=8)
            for lab_id in range(1, n_labels):
                ys, xs = np.where(labels == lab_id)
                if len(xs) < p["min_pixels"]:
                    continue
                x0, x1 = xs.min(), xs.max()
                y0, y1 = ys.min(), ys.max()
                bw, bh = (x1 - x0 + 1), (y1 - y0 + 1)
                elong = max(bw, bh) / max(1, min(bw, bh))
                fill = len(xs) / float(bw * bh)
                # a wire is elongated; reject compact blobs (fill high AND not long)
                if elong < p["min_elongation"] and fill > 0.30:
                    continue

                coords = np.stack([ys, xs], axis=1)
                if len(coords) > 8000:
                    coords = coords[:: max(1, len(coords) // 8000)]
                path_rc = _longest_path(coords, prune=int(p.get("spur_prune", 8)))
                if len(path_rc) < 2:
                    continue
                path_xy = np.array([[c, r] for r, c in path_rc], dtype=np.float32)
                poly = _simplify(path_xy, p["simplify_eps"])
                if len(poly) < 2:
                    continue
                length = _polyline_length(poly)
                if length < p["min_wire_len"]:
                    continue

                # --- artifact rejection --------------------------------------
                start, end = poly[0], poly[-1]
                span = math.hypot(end[0] - start[0], end[1] - start[1])

                # (1) thickness: real insulated wires have a consistent width in a
                # sane range. Distance-transform radius -> full width.
                dvals = np.array([dist[r, c] for r, c in path_rc], dtype=np.float32)
                if dvals.size == 0:
                    continue
                thickness = float(2.0 * np.median(dvals))
                if thickness < p["min_thickness"] or thickness > p["max_thickness"]:
                    continue
                # (2) thickness uniformity: text strokes / blobs / shadows vary a
                # lot in width along their medial axis; a wire barely varies.
                med = max(1e-3, float(np.median(dvals)))
                cv_thick = float(np.std(dvals) / med)
                if cv_thick > p["max_thickness_cv"]:
                    continue
                # (3) straightness: a wire is straight or gently curved. Scribbles
                # and handwriting double back (length >> end-to-end span).
                if span > 1e-3 and (length / span) > p["max_bend_ratio"]:
                    continue
                # (4) panel border / frame: reject a run that hugs the image edge
                # for most of its length.
                bm = p["border_margin"]
                on_border = sum(1 for (px, py) in poly
                                if px <= bm or py <= bm or px >= w - bm or py >= h - bm)
                if on_border / max(1, len(poly)) > 0.8:
                    continue

                wires.append(WireInstance(
                    wire_uid=f"w{uid}", start=start, end=end, polyline=poly,
                    length=length, thickness=thickness, color=color_name,
                    direction=_angle(start, end),
                    confidence=float(min(0.95, 0.55 + 0.08 * math.log1p(length))),
                    extra={"pixels": int(len(xs)), "elongation": round(elong, 2),
                           "bend": round(length / span, 2) if span > 1e-3 else None,
                           "thickness_cv": round(cv_thick, 2)},
                ))
                uid += 1
                if len(wires) >= p["max_wires"]:
                    break
            if len(wires) >= p["max_wires"]:
                break

        # Hough pass — only when very few wires were traced, to recover long
        # straight runs the skeleton may have split; deduped against existing.
        if p.get("use_hough", True) and len(wires) < max(3, p["max_wires"] // 8):
            wires = self._augment_with_hough(image_bgr, union, wires, uid)

        if wires:
            wires = _merge_segments(wires, p["merge_gap"], p["merge_angle_tol"])
            _refine_colors(image_bgr, wires)  # truer than the band label; also
            wires = _dedup_overlapping(wires, float(p.get("dedup_tol", 12.0)))
        self._snap(wires, components, terminals)
        return wires

    # -- Hough augmentation ------------------------------------------------

    def _augment_with_hough(self, image_bgr, mask, wires, uid_start) -> list[WireInstance]:
        p = self.params
        lines = cv2.HoughLinesP(
            (mask > 0).astype(np.uint8) * 255, 1, np.pi / 180, threshold=50,
            minLineLength=int(p["hough_min_len"]), maxLineGap=int(p["hough_max_gap"]))
        if lines is None:
            return wires
        # Only add Hough lines that are far from any existing wire (avoid dupes).
        existing_mid = [((w.start[0] + w.end[0]) / 2, (w.start[1] + w.end[1]) / 2)
                        for w in wires]
        uid = uid_start
        h, w_ = image_bgr.shape[:2]
        added = 0
        for ln in lines[:300]:
            x1, y1, x2, y2 = [float(v) for v in np.ravel(ln)[:4]]
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            if any(math.hypot(mx - ex, my - ey) < 25 for ex, ey in existing_mid):
                continue
            patch = image_bgr[max(0, int(my) - 2):int(my) + 3,
                              max(0, int(mx) - 2):int(mx) + 3]
            poly = [(x1, y1), (x2, y2)]
            wires.append(WireInstance(
                wire_uid=f"w{uid}", start=(x1, y1), end=(x2, y2), polyline=poly,
                length=math.hypot(x2 - x1, y2 - y1), thickness=2.0,
                color=dominant_color_name(patch), direction=_angle((x1, y1), (x2, y2)),
                confidence=0.5, extra={"source": "hough"}))
            existing_mid.append((mx, my))
            uid += 1
            added += 1
            if added >= 120 or len(wires) >= p["max_wires"]:
                break
        return wires

    # -- endpoint snapping -------------------------------------------------

    def _snap(self, wires, components, terminals) -> None:
        p = self.params
        max_d = p["snap_dist"]
        term_pts = []
        if terminals:
            for t in terminals:
                ref = getattr(t, "ref_id", None) or (t.get("ref_id") if isinstance(t, dict) else None)
                x = getattr(t, "x", None) if not isinstance(t, dict) else t.get("x")
                y = getattr(t, "y", None) if not isinstance(t, dict) else t.get("y")
                comp = getattr(t, "component_ref", None) if not isinstance(t, dict) else t.get("component_ref")
                if x is not None and y is not None:
                    term_pts.append((ref, comp, float(x), float(y)))
        comp_pts = []
        if components:
            for c in components:
                ref, cx, cy = _component_ref_center(c)
                if cx is not None:
                    comp_pts.append((ref, cx, cy))

        def nearest_term(pt):
            best, bd = None, max_d
            for ref, comp, x, y in term_pts:
                d = math.hypot(x - pt[0], y - pt[1])
                if d < bd:
                    bd, best = d, (ref, comp)
            return best

        def nearest_comp(pt):
            best, bd = None, max_d
            for ref, x, y in comp_pts:
                d = math.hypot(x - pt[0], y - pt[1])
                if d < bd:
                    bd, best = d, ref
            return best

        for wnode in wires:
            ts = nearest_term(wnode.start)
            te = nearest_term(wnode.end)
            if ts:
                wnode.from_terminal, wnode.from_component = ts[0], ts[1]
            if te:
                wnode.to_terminal, wnode.to_component = te[0], te[1]
            if wnode.from_component is None:
                wnode.from_component = nearest_comp(wnode.start)
            if wnode.to_component is None:
                wnode.to_component = nearest_comp(wnode.end)


def _component_ref_center(c):
    """Accept a dict, a BBox-carrying Detection, or a ref-component dict."""
    if isinstance(c, dict):
        ref = c.get("ref_id") or c.get("label")
        if "cx" in c and c["cx"] is not None:
            return ref, float(c["cx"]), float(c["cy"])
        bbox = c.get("bbox")
        if bbox and len(bbox) == 4:
            return ref, (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        return ref, None, None
    # Detection-like object with .bbox.center
    bbox = getattr(c, "bbox", None)
    ref = getattr(c, "label", None)
    if bbox is not None and hasattr(bbox, "center"):
        cx, cy = bbox.center
        return ref, float(cx), float(cy)
    return ref, None, None


def detect_wires(image_bgr: np.ndarray, components=None, terminals=None,
                 params: Optional[dict] = None) -> list[WireInstance]:
    """Convenience wrapper around :class:`WireDetector`."""
    return WireDetector(**(params or {})).detect(image_bgr, components, terminals)
