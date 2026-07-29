"""
Panel analysis service — thin adapter over the industrial inspection engine.

The real work lives in :mod:`rtsp_backend.electrical.inspector`. This module
exists to give the API and the older consumers (reference-panel templating,
reference-vs-observed inspection) a stable, backwards-compatible dict shape
while the richer inspection result is carried alongside.

What changed from the previous implementation
---------------------------------------------
The old ``analyze`` ran the component detector *and* a classical wire tracer over
the frame. With no trained component model the component list was empty and the
wire tracer emitted hundreds of "wires" from cabinet seams, DIN-rail edges,
device outlines, duct lips and shadows — so every panel was reported as a mass
of wires and nothing else. That stage has been removed: ``wires`` is always
empty here and the result says why. Component recognition, panel-type inference
and expert analysis replace it.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np

from .electrical import inspector, postprocess as pp

_log = logging.getLogger(__name__)

#: Kept for consumers that still read the legacy wire keys.
_EMPTY_WIRE_KEYS: dict[str, Any] = {
    "wires": [],
    "wire_color_counts": {},
    "wire_total": 0,
}

#: Legacy note text that :mod:`rtsp_backend.inspection_svc` used to sniff for.
NO_MODEL_NOTE = ("no trained component model loaded — train and export a "
                 "detector into models/components/ (see "
                 "training/electrical/README.md) to populate components")


def analyze(ai_manager, image_bgr: np.ndarray, annotate: bool = True,
            gate_config: Optional[pp.GateConfig] = None) -> dict:
    """Inspect one panel image.

    Returns the full inspection result plus the legacy keys
    (``components``/``component_counts``/``component_total`` and the now-always-
    empty wire keys) so existing callers keep working unchanged.
    """
    backend = ai_manager.backend("components") if ai_manager is not None else None
    if backend is not None and not getattr(backend, "ready", False):
        # Give a lazily-configured backend one chance to load, exactly as the
        # previous implementation did, so a freshly-dropped model activates
        # without a restart.
        try:
            backend.load()
        except Exception:
            pass

    result = inspector.inspect_panel(backend, image_bgr, annotate=annotate,
                                     gate_config=gate_config)

    model_loaded = bool(backend is not None and getattr(backend, "ready", False))
    if not model_loaded:
        result.setdefault("notes", []).append(NO_MODEL_NOTE)
    result["component_model_loaded"] = model_loaded

    # Legacy-compatible view.
    result.update(_EMPTY_WIRE_KEYS)
    result["topology"] = {"nodes": [
        {"id": c["index"], "label": c["label"], "bbox": c["bbox"],
         "position": c["position"], "class_id": c["class_id"]}
        for c in result.get("components", [])
    ], "edges": [], "node_count": result.get("component_total", 0),
        "edge_count": 0}
    result["report"] = inspector.build_report(result)
    return result


def analyze_batch(ai_manager, images: Sequence[np.ndarray],
                  batch_size: int = 8, annotate: bool = False,
                  gate_config: Optional[pp.GateConfig] = None) -> list[dict]:
    """Inspect several panel images, batching the detector forward pass.

    Returns one result per input image, in order, each with the same shape as
    :func:`analyze`. ``annotate`` defaults to ``False`` here because rendering an
    overlay per image is the dominant cost for a large batch and the caller usually
    wants the data.

    Falls back to sequential :func:`analyze` when the selected backend has no
    batched path, so the result is always correct and only the throughput varies.
    """
    backend = ai_manager.backend("components") if ai_manager is not None else None
    if backend is not None and not getattr(backend, "ready", False):
        try:
            backend.load()
        except Exception:
            pass

    images = list(images)
    if not images:
        return []

    # Without a batched recogniser there is nothing to gain, and correctness must
    # not depend on the optimisation existing.
    if backend is None or not getattr(backend, "ready", False) \
            or not hasattr(backend, "recognize_batch"):
        return [analyze(ai_manager, img, annotate=annotate,
                        gate_config=gate_config) for img in images]

    try:
        gates = backend.recognize_batch(images, batch_size=batch_size)
    except Exception:
        _log.exception("batched recognition failed; falling back to per-image")
        return [analyze(ai_manager, img, annotate=annotate,
                        gate_config=gate_config) for img in images]

    out: list[dict] = []
    for img, gate in zip(images, gates):
        result = inspector.inspect_panel(
            _PrecomputedRecognizer(gate), img, annotate=annotate,
            gate_config=gate_config)
        result["component_model_loaded"] = True
        result.update(_EMPTY_WIRE_KEYS)
        result["topology"] = {"nodes": [
            {"id": c["index"], "label": c["label"], "bbox": c["bbox"],
             "position": c["position"], "class_id": c["class_id"]}
            for c in result.get("components", [])
        ], "edges": [], "node_count": result.get("component_total", 0),
            "edge_count": 0}
        result["report"] = inspector.build_report(result)
        out.append(result)
    return out


class _PrecomputedRecognizer:
    """Adapter presenting an already-computed GateResult as a recogniser.

    The batched forward pass happens once for the whole chunk, but
    :func:`~rtsp_backend.electrical.inspector.inspect_panel` does everything else —
    OCR, expert annotation, panel classification, risk — per image. Rather than
    duplicating that pipeline for the batch path (two copies that would drift), the
    precomputed result is handed back through the recogniser interface it expects.
    """

    ready = True

    def __init__(self, gate: "pp.GateResult") -> None:
        self._gate = gate

    def recognize(self, frame) -> "pp.GateResult":
        return self._gate


def analyze_and_report(ai_manager, image_bgr: np.ndarray,
                       annotate: bool = True) -> tuple[dict, str]:
    """Convenience: inspection result plus the rendered plain-text report."""
    result = analyze(ai_manager, image_bgr, annotate=annotate)
    return result, inspector.report_text(result["report"])


__all__ = ["analyze", "analyze_batch", "analyze_and_report", "NO_MODEL_NOTE"]
