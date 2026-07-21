"""
General-purpose OCR with automatic engine selection.

Tries the best available open-source engine at call time and reports which one
ran, so results are always honest about their source:

    EasyOCR  ->  PaddleOCR  ->  Tesseract (pytesseract)  ->  none

None of these is a hard dependency. If no engine is installed the reader returns
an empty result with ``engine="none"`` and a clear note — it never fabricates
text. Install any one to enable OCR:

    pip install easyocr            # bundles its own CRNN weights (heavy, torch)
    pip install paddleocr          # PP-OCR
    pip install pytesseract        # also needs the system `tesseract` binary
"""

from __future__ import annotations

from typing import Any

import numpy as np

_easyocr_reader = None


def _try_easyocr(image_bgr: np.ndarray):
    global _easyocr_reader
    import easyocr  # type: ignore
    import cv2
    if _easyocr_reader is None:
        _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    out = []
    for box, text, conf in _easyocr_reader.readtext(rgb):
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        out.append({"text": text, "confidence": float(conf),
                    "bbox": [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]})
    return out


def _try_paddle(image_bgr: np.ndarray):
    from paddleocr import PaddleOCR  # type: ignore
    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    res = ocr.ocr(image_bgr, cls=True)
    out = []
    for page in res or []:
        for box, (text, conf) in page or []:
            xs = [p[0] for p in box]; ys = [p[1] for p in box]
            out.append({"text": text, "confidence": float(conf),
                        "bbox": [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]})
    return out


def _try_tesseract(image_bgr: np.ndarray):
    import pytesseract  # type: ignore
    from pytesseract import Output  # type: ignore
    import cv2
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    data = pytesseract.image_to_data(gray, output_type=Output.DICT)
    out = []
    n = len(data["text"])
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        conf = float(data["conf"][i]) if str(data["conf"][i]).lstrip("-").isdigit() else -1.0
        if txt and conf > 0:
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            out.append({"text": txt, "confidence": conf / 100.0,
                        "bbox": [float(x), float(y), float(x + w), float(y + h)]})
    return out


def read_text(image_bgr: np.ndarray) -> dict[str, Any]:
    """Return ``{engine, items:[{text,confidence,bbox}], text, note}``."""
    for name, fn in (("easyocr", _try_easyocr), ("paddleocr", _try_paddle),
                     ("tesseract", _try_tesseract)):
        try:
            items = fn(image_bgr)
            return {"engine": name, "items": items,
                    "text": "\n".join(i["text"] for i in items), "note": None}
        except ModuleNotFoundError:
            continue
        except Exception as exc:  # engine present but failed on this image
            return {"engine": name, "items": [], "text": "",
                    "note": f"{name} error: {exc}"}
    return {"engine": "none", "items": [], "text": "",
            "note": ("no OCR engine installed — pip install easyocr / paddleocr / "
                     "pytesseract (+ tesseract binary) to enable text extraction")}
