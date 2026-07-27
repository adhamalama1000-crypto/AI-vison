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

from typing import Any, Optional

import numpy as np

from .electrical import inspector, postprocess as pp

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


def analyze_and_report(ai_manager, image_bgr: np.ndarray,
                       annotate: bool = True) -> tuple[dict, str]:
    """Convenience: inspection result plus the rendered plain-text report."""
    result = analyze(ai_manager, image_bgr, annotate=annotate)
    return result, inspector.report_text(result["report"])


__all__ = ["analyze", "analyze_and_report", "NO_MODEL_NOTE"]
