"""
Terminal detection.

Detects, in order of preference:

* **Terminal blocks** — taken directly from the component detector when it
  labels ``terminal_block`` / ``din_rail`` regions; otherwise inferred from
  long, low, repetitive horizontal structures (the classic grey terminal rail).
* **Terminal screws / wire-entry points** — the regularly-spaced screw heads
  along a terminal block, found with Hough circles and, as a fallback, evenly
  spaced sampling across the block width.

Every terminal is returned as a :class:`Terminal` with a stable ``ref_id`` and
the owning component's ``ref_id`` (when known). Wire endpoints are snapped to
the nearest terminal by :class:`~rtsp_backend.panels.wire_detector.WireDetector`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

_TERMINAL_TYPES = {"terminal_block", "terminal", "din_rail", "busbar",
                   "neutral_bar", "earth_bar", "copper_bus"}


@dataclass
class Terminal:
    ref_id: str
    x: float
    y: float
    kind: str = "screw"                  # screw|block|entry
    label: Optional[str] = None
    component_ref: Optional[str] = None
    confidence: float = 0.6
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref_id": self.ref_id, "x": round(float(self.x), 1),
            "y": round(float(self.y), 1), "kind": self.kind,
            "label": self.label, "component_ref": self.component_ref,
            "confidence": round(float(self.confidence), 3), "extra": self.extra,
        }


def _screws_in_block(image_bgr: np.ndarray, box: tuple[int, int, int, int],
                     max_screws: int = 40) -> list[tuple[float, float]]:
    """Find screw-head centres inside a block bbox via Hough circles, falling
    back to evenly spaced entry points across the block's long axis."""
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(image_bgr.shape[1], x2)
    y2 = min(image_bgr.shape[0], y2)
    if x2 - x1 < 6 or y2 - y1 < 6:
        return []
    crop = image_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    ch, cw = gray.shape[:2]
    minr = max(2, int(min(ch, cw) * 0.12))
    maxr = max(minr + 2, int(min(ch, cw) * 0.5))
    pts: list[tuple[float, float]] = []
    try:
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(6, minr * 2),
            param1=90, param2=18, minRadius=minr, maxRadius=maxr)
        if circles is not None:
            for c in np.round(circles[0]).astype(int)[:max_screws]:
                pts.append((float(x1 + c[0]), float(y1 + c[1])))
    except Exception:
        pass
    if len(pts) >= 2:
        return pts
    # Fallback: evenly spaced entry points along the longer side.
    horizontal = (x2 - x1) >= (y2 - y1)
    n = min(max_screws, max(2, int((x2 - x1) / 18) if horizontal else int((y2 - y1) / 18)))
    pts = []
    for i in range(n):
        f = (i + 0.5) / n
        if horizontal:
            pts.append((x1 + f * (x2 - x1), (y1 + y2) / 2.0))
        else:
            pts.append(((x1 + x2) / 2.0, y1 + f * (y2 - y1)))
    return pts


def _detect_blocks_classical(image_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Infer terminal-rail blocks from morphology when no component model is
    available: long low-height horizontal bright/uniform bands."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    # emphasise horizontal structure
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    _, th = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    horiz = cv2.morphologyEx(
        th, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 15), 3)))
    contours, _ = cv2.findContours(horiz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blocks = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw > w * 0.18 and 6 <= bh <= h * 0.14 and bw > bh * 3:
            blocks.append((x, y, x + bw, y + bh))
    return blocks[:12]


def detect_terminals(image_bgr: np.ndarray, components: Optional[list] = None,
                     max_terminals: int = 400) -> list[Terminal]:
    """Return terminals for a panel image. ``components`` is the component list
    (dicts with ``ref_id``/``comp_type``/``bbox`` or Detection-like objects)."""
    if image_bgr is None or image_bgr.size == 0:
        return []
    terminals: list[Terminal] = []
    uid = 0

    block_boxes: list[tuple] = []          # (box, component_ref)
    if components:
        for c in components:
            ctype, ref, box = _component_fields(c)
            if ctype and ctype.lower() in _TERMINAL_TYPES and box is not None:
                block_boxes.append((box, ref))
    if not block_boxes:
        for box in _detect_blocks_classical(image_bgr):
            block_boxes.append((box, None))

    for box, comp_ref in block_boxes:
        x1, y1, x2, y2 = box
        # the block itself as one terminal-block node
        terminals.append(Terminal(
            ref_id=f"T{uid}", x=(x1 + x2) / 2.0, y=(y1 + y2) / 2.0,
            kind="block", component_ref=comp_ref, confidence=0.7,
            extra={"bbox": [float(x1), float(y1), float(x2), float(y2)]}))
        uid += 1
        for (sx, sy) in _screws_in_block(image_bgr, box):
            terminals.append(Terminal(
                ref_id=f"T{uid}", x=sx, y=sy, kind="screw",
                component_ref=comp_ref, confidence=0.55))
            uid += 1
            if uid >= max_terminals:
                return terminals
    return terminals


def _component_fields(c):
    """Return (comp_type, ref_id, bbox) from a dict or Detection-like object."""
    if isinstance(c, dict):
        box = c.get("bbox")
        box = tuple(box) if box and len(box) == 4 else None
        return (c.get("comp_type") or c.get("label"), c.get("ref_id") or c.get("label"), box)
    label = getattr(c, "label", None)
    bbox = getattr(c, "bbox", None)
    box = None
    if bbox is not None and hasattr(bbox, "as_list"):
        box = tuple(bbox.as_list())
    return (label, label, box)
