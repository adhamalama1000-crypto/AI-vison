"""
Industrial Panel Inspection engine.

A cohesive, dependency-light computer-vision package that turns an electrical
control panel into structured, comparable data:

* :mod:`wire_detector`     — real classical wire instance detection
  (HSV/LAB colour segmentation, adaptive threshold, morphology,
  skeletonisation, connected components, contour filtering, polyline
  extraction, Hough transform, endpoint detection, broken-segment merging).
* :mod:`terminal_detector` — terminal blocks / screws / wire-entry points.
* :mod:`features`          — ORB feature embedding + homography alignment used
  to register an observed panel against a learned reference.
* :mod:`graph`             — build the electrical graph (component + terminal
  nodes, wire edges).
* :mod:`template`          — learn a reusable reference-panel template from one
  or more images.
* :mod:`comparison`        — compare an observed panel against a reference and
  emit typed, confidence-scored errors.
* :mod:`datasheet`         — OCR a datasheet / schematic into an expected graph.
* :mod:`overlay`           — draw green/yellow/red inspection overlays.

Nothing here fabricates results. Where a step depends on a capability that is
not present (a trained component model, an OCR engine), the code degrades
gracefully and records *why* rather than inventing data — the same honesty
contract as the rest of the platform.
"""

from __future__ import annotations

from .wire_detector import WireDetector, WireInstance, detect_wires
from .terminal_detector import Terminal, detect_terminals
from . import features, graph, template, comparison, datasheet, overlay

__all__ = [
    "WireDetector", "WireInstance", "detect_wires",
    "Terminal", "detect_terminals",
    "features", "graph", "template", "comparison", "datasheet", "overlay",
]
