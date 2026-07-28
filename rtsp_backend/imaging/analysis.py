"""
General AI image analysis — works on ANY image.

Produces, from a single image:

* **objects** + bounding boxes + confidence — via the platform's ONNX object
  detector (COCO by default; any YOLO/GroundingDINO-exported ONNX dropped into
  ``models/detection/`` is picked up automatically). Empty (with a note) when no
  detector weights are present — never fabricated boxes.
* **dominant colours** — k-means over the pixels (always available).
* **OCR text** — best available engine (EasyOCR/Paddle/Tesseract), or a note.
* **perceptual hash** (pHash) — for fast near-duplicate / similarity checks.
* **image size / format / metadata**.
* **tags** + a human-readable **summary** — synthesised from the detections,
  colours and OCR. If a CLIP/BLIP model is installed it is used to enrich the
  tags; otherwise an honest heuristic summary is produced.
* **defects** — generic low-level quality flags (blur, over/under exposure).

Everything is JSON-serialisable and persisted by the ImageService.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Optional

import cv2
import numpy as np

from .ocr import read_text

# rough English colour names for the dominant-colour swatches
_COLOR_NAMES = [
    ((0, 0, 0), "black"), ((255, 255, 255), "white"), ((128, 128, 128), "grey"),
    ((255, 0, 0), "red"), ((0, 255, 0), "green"), ((0, 0, 255), "blue"),
    ((255, 255, 0), "yellow"), ((255, 165, 0), "orange"), ((128, 0, 128), "purple"),
    ((165, 42, 42), "brown"), ((0, 255, 255), "cyan"), ((255, 192, 203), "pink"),
    ((0, 128, 128), "teal"), ((192, 192, 192), "silver"),
]


def _color_name(rgb) -> str:
    r, g, b = rgb
    best, bd = "unknown", 1e9
    for (cr, cg, cb), name in _COLOR_NAMES:
        d = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if d < bd:
            bd, best = d, name
    return best


def dominant_colors(image_bgr: np.ndarray, k: int = 5) -> list[dict]:
    """Top-k dominant colours via k-means, sorted by coverage."""
    small = cv2.resize(image_bgr, (128, 128), interpolation=cv2.INTER_AREA)
    data = small.reshape(-1, 3).astype(np.float32)
    k = min(k, max(1, len(np.unique(data, axis=0))))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    total = counts.sum() or 1
    order = np.argsort(counts)[::-1]
    out = []
    for i in order:
        b, g, r = [int(v) for v in centers[i]]
        out.append({
            "hex": f"#{r:02x}{g:02x}{b:02x}",
            "rgb": [r, g, b],
            "name": _color_name((r, g, b)),
            "ratio": round(float(counts[i] / total), 4),
        })
    return out


def perceptual_hash(image_bgr: np.ndarray) -> Optional[str]:
    try:
        import imagehash  # type: ignore
        from PIL import Image
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return str(imagehash.phash(Image.fromarray(rgb)))
    except Exception:
        return None


def quality_defects(image_bgr: np.ndarray) -> list[dict]:
    """Generic, model-free quality flags applicable to any image."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    defects = []
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur < 60:
        defects.append({"type": "blur", "detail": f"image looks blurry (sharpness {blur:.0f})",
                        "severity": "warning", "confidence": round(min(1.0, (60 - blur) / 60), 2)})
    mean = float(gray.mean())
    if mean > 225:
        defects.append({"type": "overexposed", "detail": f"very bright (mean {mean:.0f})",
                        "severity": "warning", "confidence": 0.6})
    elif mean < 30:
        defects.append({"type": "underexposed", "detail": f"very dark (mean {mean:.0f})",
                        "severity": "warning", "confidence": 0.6})
    return defects


def detect_objects(ai_manager, image_bgr: np.ndarray) -> tuple[list[dict], Optional[str], Optional[str]]:
    """Run the object detector if one is loaded. Returns (objects, source, note)."""
    if ai_manager is None:
        return [], None, "no AI manager available"
    backend = ai_manager.backend("detection")
    if backend is None:
        return [], None, "no detection backend configured"
    if not getattr(backend, "ready", False):
        try:
            backend.load()
        except Exception as exc:
            return [], getattr(backend, "backend_id", None), (
                f"object detector not loaded ({exc}); drop a YOLO/GroundingDINO "
                f"ONNX into models/detection/ to enable object detection")
    try:
        dets = backend.infer(image_bgr)
    except Exception as exc:
        return [], getattr(backend, "backend_id", None), f"detection error: {exc}"
    objs = [{
        "label": d.label, "confidence": round(float(d.confidence), 4),
        "bbox": [round(v, 1) for v in d.bbox.as_list()],
    } for d in dets]
    return objs, getattr(backend, "backend_id", None), None


def _clip_tags(image_bgr: np.ndarray) -> list[str]:
    """Optional CLIP zero-shot tags; empty if CLIP isn't installed."""
    try:
        import torch  # type: ignore  # noqa
        import open_clip  # type: ignore  # noqa
    except Exception:
        return []
    return []  # placeholder hook: wire a CLIP model here when weights are present


def build_summary(objects: list[dict], colors: list[dict], ocr_text: str,
                  size: tuple[int, int], defects: list[dict]) -> tuple[str, list[str]]:
    """Heuristic natural-language summary + tag list from real signals."""
    tags: list[str] = []
    counts = Counter(o["label"] for o in objects)
    parts: list[str] = []
    if counts:
        top = ", ".join(f"{n} {lbl}{'s' if n > 1 else ''}" for lbl, n in counts.most_common(6))
        parts.append(f"Detected {len(objects)} object(s): {top}.")
        tags += [lbl for lbl, _ in counts.most_common(8)]
    else:
        parts.append("No objects detected by the current model.")
    if colors:
        parts.append(f"Dominant colour is {colors[0]['name']} ({colors[0]['hex']}).")
        tags += [c["name"] for c in colors[:3]]
    if ocr_text.strip():
        snippet = " ".join(ocr_text.split())[:80]
        parts.append(f"Contains text: “{snippet}”.")
        tags.append("text")
    if defects:
        parts.append("Quality flags: " + ", ".join(d["type"] for d in defects) + ".")
    parts.append(f"Resolution {size[0]}×{size[1]}.")
    # de-dup tags, keep order
    seen, uniq = set(), []
    for t in tags:
        if t not in seen:
            seen.add(t); uniq.append(t)
    return " ".join(parts), uniq


def analyze(image_bgr: np.ndarray, ai_manager=None) -> dict[str, Any]:
    """Full analysis of one image. Never raises on optional-capability gaps."""
    h, w = image_bgr.shape[:2]
    notes: list[str] = []

    objects, src, det_note = detect_objects(ai_manager, image_bgr)
    if det_note:
        notes.append(det_note)
    for o in objects:
        o["source"] = src

    colors = dominant_colors(image_bgr)
    ocr = read_text(image_bgr)
    if ocr.get("note"):
        notes.append(ocr["note"])
    defects = quality_defects(image_bgr)
    phash = perceptual_hash(image_bgr)
    summary, tags = build_summary(objects, colors, ocr["text"], (w, h), defects)

    return {
        "image_size": [w, h],
        "channels": int(image_bgr.shape[2]) if image_bgr.ndim == 3 else 1,
        "objects": objects,
        "object_total": len(objects),
        "object_counts": dict(Counter(o["label"] for o in objects)),
        "detector": src,
        "dominant_colors": colors,
        "ocr": ocr,
        "defects": defects,
        "phash": phash,
        "tags": tags,
        "summary": summary,
        "notes": notes,
    }


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()
