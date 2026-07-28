"""
Wire detection & topology-analysis backends.

Individual-wire instance segmentation, end-to-end tracing, and fault detection
(broken / loose / incorrect wiring) is at the frontier of computer vision and
has NO reliable public pretrained model. We therefore do NOT fake those
results. Instead we ship:

* ``classical_wires`` — a REAL, weights-free classical pipeline that finds
  wire-like line segments (edge + Hough/LSD), groups them by dominant colour,
  and builds a first-pass topology by linking segment endpoints to the nearest
  detected component terminal. It genuinely returns geometry from the frame,
  but it CANNOT reliably do instance-level tracing or fault classification —
  those fields are reported as ``status="unknown"``. This is an honest,
  inspectable baseline and a working integration harness, not a fabrication.

* ``null_wires`` — disabled state, returns nothing.

The topology data model, DB tables, visualisation, and API are complete, so a
future trained wire model only needs to populate :class:`Wire` objects with
per-instance status; nothing downstream changes.

.. warning::

   **Wiring analysis is disabled by default and is not part of the supported
   inspection path.** Both classical tracers below key off generic image
   gradients and colour, so in a real control panel they label cabinet seams,
   DIN-rail edges, device outlines, cable-duct lips, label borders, shadows and
   reflections as "wires" — routinely hundreds per frame — while missing the
   actual conductors, which are largely occluded inside ducting. That noise
   swamped the component result, so the platform no longer runs it.

   These backends remain registered as *experimental* for research only. Wire
   analysis will return when there is a trained, quantitatively validated
   instance-segmentation model behind it. See ``docs/AUDIT_PANEL_INSPECTOR.md``.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .base import Detection, Wire, WireAnalyzer
from .registry import register


def _dominant_color_name(bgr_patch: np.ndarray) -> str:
    if bgr_patch.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(bgr_patch.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h = float(np.median(hsv[:, 0]))
    s = float(np.median(hsv[:, 1]))
    v = float(np.median(hsv[:, 2]))
    if v < 50:
        return "black"
    if s < 40:
        return "white/grey"
    if h < 10 or h >= 170:
        return "red"
    if h < 25:
        return "orange"
    if h < 35:
        return "yellow"
    if h < 85:
        return "green"
    if h < 130:
        return "blue"
    return "other"


@register
class ClassicalWireAnalyzer(WireAnalyzer):
    backend_id = "classical_wires"
    task = "wires"
    display_name = ("Classical line/colour wire baseline — EXPERIMENTAL, "
                    "high false-positive rate")
    requires_weights = False
    experimental = True
    warning = ("Hough line segments are not wires. On a real panel this emits "
               "hundreds of false positives from seams, edges and shadows. "
               "Research use only.")

    def load(self) -> None:
        self._ready = True
        self._status = "ready"
        self._error = None

    def analyze(self, frame: np.ndarray, components: list[Detection]) -> list[Wire]:
        if not self._ready:
            self.load()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        min_len = int(self.params.get("min_wire_len", 40))
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=60,
            minLineLength=min_len, maxLineGap=10,
        )
        wires: list[Wire] = []
        if lines is None:
            return wires

        # component terminal points = bbox corners/centres, for endpoint linking
        terminals: list[tuple[int, tuple[float, float]]] = []
        for idx, comp in enumerate(components):
            cx, cy = comp.bbox.center
            terminals.append((idx, (cx, cy)))

        for i, ln in enumerate(lines[:200]):
            # HoughLinesP returns (N,1,4) on most OpenCV builds but (N,4) on
            # some; ravel handles both without assuming a leading singleton dim.
            x1, y1, x2, y2 = [float(v) for v in np.ravel(ln)[:4]]
            # sample colour along the midpoint neighbourhood
            mx, my = int((x1 + x2) / 2), int((y1 + y2) / 2)
            patch = frame[max(0, my - 2):my + 3, max(0, mx - 2):mx + 3]
            color = _dominant_color_name(patch)
            from_c = self._nearest(terminals, (x1, y1))
            to_c = self._nearest(terminals, (x2, y2))
            wires.append(
                Wire(
                    wire_uid=f"w{i}",
                    start=(x1, y1),
                    end=(x2, y2),
                    color=color,
                    status="unknown",  # honest: classical baseline can't classify faults
                    from_component=from_c,
                    to_component=to_c,
                    extra={"length": round(float(np.hypot(x2 - x1, y2 - y1)), 1)},
                )
            )
        return wires

    @staticmethod
    def _nearest(
        terminals: list[tuple[int, tuple[float, float]]],
        pt: tuple[float, float],
        max_dist: float = 60.0,
    ) -> Optional[int]:
        best_idx, best_d = None, max_dist
        for idx, (tx, ty) in terminals:
            d = np.hypot(tx - pt[0], ty - pt[1])
            if d < best_d:
                best_d, best_idx = d, idx
        return best_idx


@register
class AdvancedWireAnalyzer(WireAnalyzer):
    """Real classical wire instance detector (Feature 4).

    Wraps :class:`rtsp_backend.panels.wire_detector.WireDetector` — HSV/LAB
    colour segmentation, adaptive threshold, morphology, skeletonisation,
    connected components, contour filtering, polyline extraction, Hough
    transform, endpoint detection and broken-segment merging — and adapts its
    rich :class:`WireInstance` output to the platform's :class:`Wire` model so
    the live pipeline, overlays and API all benefit with no other changes.

    It returns genuine per-instance geometry (start/end/polyline/length/
    thickness/colour/direction). Fault *status* stays ``unknown`` on the live
    path — a fault can only be asserted against a learned reference panel, which
    the inspection engine does; a bare live detection never fabricates a verdict.
    """

    backend_id = "advanced_wires"
    task = "wires"
    display_name = ("Advanced classical wire detector — EXPERIMENTAL, "
                    "high false-positive rate")
    requires_weights = False
    experimental = True
    warning = ("Colour-segmentation + skeletonisation over the whole frame. It "
               "returns genuine line geometry but cannot tell a conductor from "
               "a cabinet seam, so it is unfit for inspection output. "
               "Research use only.")

    def load(self) -> None:
        # build the detector once, forwarding any tuning params
        from ..panels.wire_detector import WireDetector
        allowed = set(WireDetector.DEFAULTS.keys())
        kw = {k: v for k, v in self.params.items() if k in allowed}
        self._detector = WireDetector(**kw)
        self._ready = True
        self._status = "ready"
        self._error = None

    def analyze(self, frame: np.ndarray, components: list[Detection]) -> list[Wire]:
        if not self._ready:
            self.load()
        # index components so we can report integer node ids (compatible with
        # the existing Wire model + wires table).
        centers = []
        for idx, comp in enumerate(components or []):
            cx, cy = comp.bbox.center
            centers.append((idx, cx, cy))

        def nearest(pt, max_d=60.0):
            best, bd = None, max_d
            for idx, cx, cy in centers:
                d = ((cx - pt[0]) ** 2 + (cy - pt[1]) ** 2) ** 0.5
                if d < bd:
                    bd, best = d, idx
            return best

        instances = self._detector.detect(frame, components=None, terminals=None)
        wires: list[Wire] = []
        for inst in instances:
            wires.append(Wire(
                wire_uid=inst.wire_uid,
                start=inst.start, end=inst.end,
                color=inst.color,
                status="unknown",  # honest: no reference => no fault verdict
                from_component=nearest(inst.start),
                to_component=nearest(inst.end),
                extra={
                    "length": round(inst.length, 1),
                    "thickness": round(inst.thickness, 2),
                    "direction": round(inst.direction, 1),
                    "polyline": [[round(x, 1), round(y, 1)] for x, y in inst.polyline],
                    "confidence": round(inst.confidence, 3),
                },
            ))
        return wires


@register
class NullWireAnalyzer(WireAnalyzer):
    backend_id = "null_wires"
    task = "wires"
    display_name = "Disabled (no wire analysis)"
    requires_weights = False

    def load(self) -> None:
        self._ready = True
        self._status = "ready"

    def analyze(self, frame, components):
        return []
