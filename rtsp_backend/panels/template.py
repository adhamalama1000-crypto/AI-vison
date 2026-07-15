"""
Reference-panel template learning.

Given one or more reference images of a *correct* panel, produce a reusable
template describing what the panel should look like:

* **Components** — bounding box, centre, size, rotation and (coarse) grid
  position for every detected component. Rotation is measured from the dominant
  orientation of the component's content (``minAreaRect``), so a
  "wrong rotation" fault is later detectable.
* **Terminals** — every terminal-block / screw / entry point.
* **Wires** — full instance geometry from the real wire detector.
* **Graph** — the electrical connection graph.
* **Feature embedding** — ORB descriptors + colour histogram of the primary
  image for later registration.

If no trained component model is loaded, the component list is empty and a note
says so — the wire, terminal and graph pipeline still run on real geometry.
This is the same honesty contract as the live pipeline; the template is
immediately useful for wire / topology inspection and becomes complete once a
component model is trained and dropped in.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Optional

import cv2
import numpy as np

from . import features as _features
from . import graph as _graph
from .terminal_detector import detect_terminals
from .wire_detector import detect_wires


def _grid_position(cx: float, cy: float, shape) -> str:
    h, w = shape[:2]
    col = "left" if cx < w / 3 else ("center" if cx < 2 * w / 3 else "right")
    row = "top" if cy < h / 3 else ("middle" if cy < 2 * h / 3 else "bottom")
    return f"{row}-{col}"


def _component_rotation(image_bgr: np.ndarray, box) -> float:
    """Dominant orientation (deg, 0..180) of the component's content inside box."""
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(image_bgr.shape[1], x2)
    y2 = min(image_bgr.shape[0], y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return 0.0
    crop = cv2.cvtColor(image_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(crop, 50, 150)
    pts = cv2.findNonZero(edges)
    if pts is None or len(pts) < 5:
        return 0.0
    rect = cv2.minAreaRect(pts)
    angle = rect[-1]
    # OpenCV angle convention -> normalise to [0,180)
    (w_, h_) = rect[1]
    if w_ < h_:
        angle += 90.0
    return float(angle % 180.0)


def analyze_image(ai_manager, image_bgr: np.ndarray,
                  wire_params: Optional[dict] = None) -> dict[str, Any]:
    """Run the full detector stack on a single image and return structured
    components / terminals / wires / graph plus notes."""
    notes: list[str] = []
    h, w = image_bgr.shape[:2]

    # -- components --------------------------------------------------------
    comps: list[dict] = []
    backend = ai_manager.backend("components") if ai_manager is not None else None
    detections = []
    if backend is not None and getattr(backend, "ready", False):
        try:
            detections = backend.infer(image_bgr)
        except Exception as exc:
            notes.append(f"component inference error: {exc}")
    else:
        notes.append("no trained component model loaded — components empty; drop "
                     "a trained detector into models/components/ to populate them")
    for i, d in enumerate(detections):
        box = d.bbox.as_list()
        cx, cy = d.bbox.center
        comps.append({
            "ref_id": f"C{i}", "comp_type": d.label, "label": d.label,
            "bbox": [round(float(v), 1) for v in box],
            "cx": round(float(cx), 1), "cy": round(float(cy), 1),
            "w": round(float(box[2] - box[0]), 1), "h": round(float(box[3] - box[1]), 1),
            "rotation": round(_component_rotation(image_bgr, box), 1),
            "confidence": round(float(d.confidence), 3),
            "position": _grid_position(cx, cy, image_bgr.shape),
        })

    # -- terminals ---------------------------------------------------------
    terminals = [t.to_dict() for t in detect_terminals(image_bgr, comps)]

    # -- wires -------------------------------------------------------------
    wires = [wnode.to_dict() for wnode in detect_wires(image_bgr, comps, terminals, wire_params)]

    # -- graph -------------------------------------------------------------
    g = _graph.build_graph(comps, terminals, wires)

    return {
        "image_size": [w, h],
        "components": comps, "component_total": len(comps),
        "component_counts": dict(Counter(c["comp_type"] for c in comps)),
        "terminals": terminals, "terminal_total": len(terminals),
        "wires": wires, "wire_total": len(wires),
        "wire_color_counts": dict(Counter(wnode["color"] for wnode in wires)),
        "graph": g,
        "notes": notes,
    }


def build_template(ai_manager, images: list[np.ndarray],
                   wire_params: Optional[dict] = None) -> dict[str, Any]:
    """Learn a reference template from one or more images.

    The first image is the *primary* (geometry + feature embedding come from
    it); additional images validate detection stability and contribute to the
    aggregated component-count expectation.
    """
    if not images:
        raise ValueError("build_template requires at least one image")

    primary = images[0]
    primary_analysis = analyze_image(ai_manager, primary, wire_params)

    # aggregate component counts across all images (stability signal)
    agg_counts: Counter = Counter(primary_analysis["component_counts"])
    per_image = [{"component_total": primary_analysis["component_total"],
                  "wire_total": primary_analysis["wire_total"],
                  "terminal_total": primary_analysis["terminal_total"]}]
    for img in images[1:]:
        a = analyze_image(ai_manager, img, wire_params)
        for k, v in a["component_counts"].items():
            agg_counts[k] = max(agg_counts.get(k, 0), v)
        per_image.append({"component_total": a["component_total"],
                          "wire_total": a["wire_total"],
                          "terminal_total": a["terminal_total"]})

    embedding = _features.extract_features(primary)

    template = {
        "image_size": primary_analysis["image_size"],
        "components": primary_analysis["components"],
        "terminals": primary_analysis["terminals"],
        "wires": primary_analysis["wires"],
        "graph": primary_analysis["graph"],
        "expected_component_counts": dict(agg_counts),
        "expected_wire_color_counts": primary_analysis["wire_color_counts"],
        "n_images": len(images),
        "per_image": per_image,
        "notes": primary_analysis["notes"],
    }
    return {"template": template, "features": embedding,
            "primary_analysis": primary_analysis}
