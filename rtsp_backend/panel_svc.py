"""
Panel analysis engine (Part 8).

Runs the real electrical-component detector and the wire analyzer against a
single still image (uploaded or grabbed from a camera), then:

* counts every detected component by class,
* records every wire with its dominant colour and endpoints,
* builds an electrical topology (components = nodes, wires = edges linking the
  nearest component terminals),
* renders an annotated image, and
* returns a structured JSON result.

It never invents components: if no trained component model is loaded the
component list is empty (and ``components_note`` says why), while the classical
wire baseline still returns detected line geometry. This is the same honesty
contract as the live pipeline.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import cv2
import numpy as np

from .ai.pipeline import _COLORS, _draw_box
from .ai.base import BBox


def _panel_position(box: BBox, shape) -> str:
    h, w = shape[:2]
    cx, cy = box.center
    col = "left" if cx < w / 3 else ("center" if cx < 2 * w / 3 else "right")
    row = "top" if cy < h / 3 else ("middle" if cy < 2 * h / 3 else "bottom")
    return f"{row}-{col}"


def analyze(ai_manager, image_bgr: np.ndarray, annotate: bool = True) -> dict:
    result: dict[str, Any] = {
        "image_size": [int(image_bgr.shape[1]), int(image_bgr.shape[0])],
        "components": [], "component_counts": {}, "component_total": 0,
        "wires": [], "wire_color_counts": {}, "wire_total": 0,
        "topology": {"nodes": [], "edges": []},
        "notes": [],
    }
    annotated = image_bgr.copy() if annotate else None

    # -- components ------------------------------------------------------
    comps = []
    cbackend = ai_manager.backend("components")
    if cbackend is not None and getattr(cbackend, "ready", False):
        try:
            comps = cbackend.infer(image_bgr)
        except Exception as exc:
            result["notes"].append(f"component inference error: {exc}")
    else:
        result["notes"].append(
            "no trained component model loaded — drop a trained detector into "
            "models/components/ to populate components")

    counts: Counter = Counter()
    for i, c in enumerate(comps):
        pos = _panel_position(c.bbox, image_bgr.shape)
        counts[c.label] += 1
        d = c.to_dict()
        d["position"] = pos
        d["node_id"] = i
        result["components"].append(d)
        result["topology"]["nodes"].append(
            {"id": i, "label": c.label, "bbox": c.bbox.as_list(), "position": pos})
        if annotated is not None:
            _draw_box(annotated, c.bbox, c.label, _COLORS["component"], c.confidence)
    result["component_counts"] = dict(counts)
    result["component_total"] = len(comps)

    # -- wires -----------------------------------------------------------
    wires = []
    wbackend = ai_manager.backend("wires")
    if wbackend is not None:
        if not getattr(wbackend, "ready", False):
            try:
                wbackend.load()
            except Exception:
                pass
        try:
            wires = wbackend.analyze(image_bgr, comps)
        except Exception as exc:
            result["notes"].append(f"wire analysis error: {exc}")

    color_counts: Counter = Counter()
    for w in wires:
        color_counts[w.color or "unknown"] += 1
        result["wires"].append(w.to_dict())
        result["topology"]["edges"].append({
            "wire_uid": w.wire_uid, "start": list(w.start), "end": list(w.end),
            "color": w.color, "status": w.status,
            "from": w.from_component, "to": w.to_component})
        if annotated is not None:
            col = _COLORS["wire_ok"] if w.status in ("ok", "unknown") else _COLORS["wire_bad"]
            cv2.line(annotated, (int(w.start[0]), int(w.start[1])),
                     (int(w.end[0]), int(w.end[1])), col, 2)
    result["wire_color_counts"] = dict(color_counts)
    result["wire_total"] = len(wires)
    result["topology"]["node_count"] = len(result["topology"]["nodes"])
    result["topology"]["edge_count"] = len(result["topology"]["edges"])

    if annotated is not None:
        result["_annotated"] = annotated
    return result
