"""
Rendering for image analysis & comparison results.

* ``annotate_objects`` — draw detected-object boxes + labels on an image.
* ``build_comparison_visuals`` — from a comparison result, produce:
    - a difference **heatmap** (JET) blended over the current image,
    - the **current** image with changed-region / new-object boxes (red=major,
      amber=minor),
    - the **reference** image with missing-object boxes,
    - a **side-by-side** panel: reference | current | heatmap, captioned.

All BGR (OpenCV). Pure drawing — no inference.
"""

from __future__ import annotations

import cv2
import numpy as np

_MAJOR = (0, 0, 235)      # red
_MINOR = (0, 200, 240)    # amber
_OBJ = (230, 170, 40)     # blue
_MISS = (0, 0, 235)


def _label(img, text, org, color):
    x, y = int(org[0]), int(max(14, org[1]))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(img, (x, y - th - 5), (x + tw + 4, y + 2), color, -1)
    cv2.putText(img, text, (x + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def annotate_objects(image_bgr: np.ndarray, objects: list[dict]) -> np.ndarray:
    img = image_bgr.copy()
    for o in objects:
        b = o.get("bbox")
        if not b or len(b) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in b]
        cv2.rectangle(img, (x1, y1), (x2, y2), _OBJ, 2)
        lbl = o.get("label", "?")
        if o.get("confidence") is not None:
            lbl += f" {float(o['confidence']):.2f}"
        _label(img, lbl, (x1, y1), _OBJ)
    return img


def _draw_diffs(img, diffs, on="current"):
    out = img.copy()
    for d in diffs:
        b = d.get("bbox")
        if not b or len(b) != 4:
            continue
        want_ref = d["diff_type"] in ("missing_object",)
        if on == "reference" and not want_ref:
            continue
        if on == "current" and want_ref:
            continue
        x1, y1, x2, y2 = [int(v) for v in b]
        color = _MAJOR if d["severity"] == "major" else _MINOR
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        _label(out, d["diff_type"], (x1, y1), color)
    return out


def heatmap(current_bgr: np.ndarray, diff_map: np.ndarray) -> np.ndarray:
    dm = cv2.resize(diff_map, (current_bgr.shape[1], current_bgr.shape[0]))
    hm = cv2.applyColorMap(dm, cv2.COLORMAP_JET)
    return cv2.addWeighted(current_bgr, 0.6, hm, 0.4, 0)


def _caption_strip(img, text, h=28):
    strip = np.full((h, img.shape[1], 3), 25, np.uint8)
    cv2.putText(strip, text, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([strip, img])


def build_comparison_visuals(reference_bgr: np.ndarray, current_bgr: np.ndarray,
                             result: dict) -> dict[str, np.ndarray]:
    warped = result.get("_warped", current_bgr)
    diff_map = result.get("_diff_map")
    diffs = result.get("differences", [])

    cur_annot = _draw_diffs(warped, diffs, on="current")
    ref_annot = _draw_diffs(reference_bgr, diffs, on="reference")
    hm = heatmap(warped, diff_map) if diff_map is not None else warped.copy()

    # normalise heights for the side-by-side
    h = min(reference_bgr.shape[0], cur_annot.shape[0], hm.shape[0], 720)
    def fit(im):
        scale = h / im.shape[0]
        return cv2.resize(im, (int(im.shape[1] * scale), h))
    sim = result.get("similarity", 0)
    panels = [
        _caption_strip(fit(ref_annot), "REFERENCE"),
        _caption_strip(fit(cur_annot), "CURRENT"),
        _caption_strip(fit(hm), f"DIFFERENCES  ({sim:.1f}% similar)"),
    ]
    ph = max(pi.shape[0] for pi in panels)
    padded = []
    for pi in panels:
        if pi.shape[0] < ph:
            pad = np.full((ph - pi.shape[0], pi.shape[1], 3), 25, np.uint8)
            pi = np.vstack([pi, pad])
        padded.append(pi)
    side_by_side = np.hstack(padded)
    return {"overlay": side_by_side, "heatmap": hm,
            "annotated_current": cur_annot, "annotated_reference": ref_annot}
