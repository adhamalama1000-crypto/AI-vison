"""
General AI Image Analysis & Comparison engine.

Works on ANY image (panels, PCBs, machines, people, products, buildings,
documents, medical, screenshots, vehicles …). Everything degrades gracefully:
where a capability needs a model that isn't installed (object detector weights,
an OCR engine, a VLM), the pipeline still runs on real signals and records a
note explaining what's missing — it never fabricates results.

* :mod:`analysis`   — single-image analysis (objects, colours, OCR, pHash,
  tags, summary, defects, metadata).
* :mod:`comparison` — reference-vs-current diff (ORB registration, lighting
  compensation, SSIM regions, object/colour/text diff, similarity %).
* :mod:`visualize`  — heatmap / annotated / side-by-side overlays.
* :mod:`export`     — JSON + PDF reports.
* :mod:`service`    — DB-backed orchestration used by the REST API.
"""

from __future__ import annotations

from . import analysis, comparison, visualize, export, ocr
from .service import ImageService

__all__ = ["analysis", "comparison", "visualize", "export", "ocr", "ImageService"]
