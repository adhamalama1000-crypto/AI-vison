"""
Inspection overlay rendering.

Draws the observed panel with an inspection verdict painted on top:

* **Green**  — element matches the reference (correct).
* **Yellow** — warning (moved, wrong rotation, wrong colour, loose wire, extra).
* **Red**    — error (missing, wrong, broken, disconnected).

Labels show component names, wire IDs, terminal IDs and confidence. All colours
are BGR (OpenCV). This is a pure drawing helper — it takes already-computed
detections + comparison errors and never runs inference itself.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

GREEN = (80, 200, 80)
YELLOW = (0, 200, 240)
RED = (0, 0, 235)
BLUE = (230, 170, 40)
GREY = (150, 150, 150)

_SEVERITY_COLOR = {"error": RED, "warning": YELLOW, "info": BLUE}


def _put_label(img, text, org, color) -> None:
    x, y = int(org[0]), int(org[1])
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(img, (x, y - th - 5), (x + tw + 4, y + 2), color, -1)
    cv2.putText(img, text, (x + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)


def draw_overlay(image_bgr: np.ndarray, observed: dict,
                 comparison: dict) -> np.ndarray:
    """Return a copy of ``image_bgr`` with components, terminals, wires and
    error markers drawn on top."""
    img = image_bgr.copy()
    errors = comparison.get("errors", []) if comparison else []

    # index which targets are in error/warning so matched elements can go green
    err_targets = {e.get("target"): e for e in errors if e.get("severity") == "error"}
    warn_targets = {e.get("target"): e for e in errors if e.get("severity") == "warning"}

    def tone_for(ref) -> tuple:
        if ref in err_targets:
            return RED
        if ref in warn_targets:
            return YELLOW
        return GREEN

    # components
    for c in observed.get("components", []) or []:
        box = c.get("bbox")
        if not box or len(box) != 4:
            continue
        color = tone_for(c.get("label"))
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        lbl = f"{c.get('comp_type', c.get('label', '?'))}"
        if c.get("confidence") is not None:
            lbl += f" {float(c['confidence']):.2f}"
        _put_label(img, lbl, (x1, max(12, y1)), color)

    # terminals
    for t in observed.get("terminals", []) or []:
        if t.get("x") is None:
            continue
        x, y = int(t["x"]), int(t["y"])
        if t.get("kind") == "screw":
            cv2.circle(img, (x, y), 4, BLUE, 1, cv2.LINE_AA)
        else:
            cv2.drawMarker(img, (x, y), BLUE, cv2.MARKER_SQUARE, 8, 1)

    # wires
    for w in observed.get("wires", []) or []:
        poly = w.get("polyline") or [w.get("start"), w.get("end")]
        pts = np.array([[int(p[0]), int(p[1])] for p in poly if p], np.int32)
        color = tone_for(w.get("wire_uid"))
        if len(pts) >= 2:
            cv2.polylines(img, [pts], False, color, 2, cv2.LINE_AA)
        if w.get("wire_uid") and len(pts):
            _put_label(img, str(w["wire_uid"]), (pts[0][0], pts[0][1]), color)

    # explicit error markers (missing elements have no observed box to colour)
    for e in errors:
        if e.get("x") is None:
            continue
        color = _SEVERITY_COLOR.get(e.get("severity"), GREY)
        x, y = int(e["x"]), int(e["y"])
        cv2.drawMarker(img, (x, y), color, cv2.MARKER_TILTED_CROSS, 16, 2)
        conf = e.get("confidence")
        tag = e.get("error_type", "")
        if conf is not None:
            tag += f" {float(conf):.2f}"
        _put_label(img, tag, (x + 8, y), color)

    _stamp_status(img, comparison)
    return img


def _stamp_status(img, comparison) -> None:
    if not comparison:
        return
    status = comparison.get("status", "unknown")
    color = {"pass": GREEN, "warning": YELLOW, "fail": RED}.get(status, GREY)
    score = comparison.get("score")
    txt = f"{status.upper()}"
    if score is not None:
        txt += f"  score={score:.2f}"
    txt += f"  errors={comparison.get('n_errors', 0)} warnings={comparison.get('n_warnings', 0)}"
    cv2.rectangle(img, (0, 0), (img.shape[1], 32), (30, 30, 30), -1)
    cv2.putText(img, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                cv2.LINE_AA)
