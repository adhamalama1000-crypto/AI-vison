"""
``POST /api/panel/analyze`` — the component-detection contract.

This is the endpoint the brief specifies, with the response shape it specifies:

.. code-block:: json

    {"components": [{"class": "MCB", "confidence": 0.98, "bbox": [x1,y1,x2,y2]}]}

It is a *new, additive* router. The existing ``/api/panels/analyze`` (plural)
keeps its richer response and its report/PDF persistence untouched — dashboards
and the reference-panel flow depend on it. Both routes run the same engine
(:mod:`rtsp_backend.panel_svc` → :mod:`rtsp_backend.electrical.inspector`), so
there is one detection path and one set of thresholds, not two that can drift.

On top of the minimal contract, ``?report=true`` (the default) adds the panel
report the brief asks for: detected components, missing components, unknown
components, confidence, and the annotated image.

Honesty rules enforced here
---------------------------
* With no trained model loaded, ``components`` is ``[]`` and ``model.loaded`` is
  ``false`` with the reason. It never returns invented boxes to look functional.
* A detection the model cannot classify confidently is reported with
  ``"class": "unknown_industrial_component"`` and listed under
  ``report.unknown_components`` — never promoted to a plausible-looking class.
* ``bbox`` is always ``[x1, y1, x2, y2]`` in **absolute pixels** of the submitted
  image, with ``image.width``/``image.height`` alongside so a client can
  normalise without guessing the convention.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, Query, UploadFile

from .. import panel_svc, reports_svc
from ..electrical import taxonomy as tax
from ..errors import RTSPBackendError
from .util import read_upload_capped, save_image

#: Cap on components returned in one response. A real panel photograph does not
#: contain more than this; hitting the cap means the model is misbehaving, and the
#: response says so rather than returning ten thousand boxes.
MAX_COMPONENTS = 500


async def _image_from(ctx, upload: Optional[UploadFile],
                      camera_id: Optional[str]) -> tuple[np.ndarray, str]:
    """Resolve the request to one BGR frame, from an upload or an RTSP camera."""
    if upload is not None:
        raw = await read_upload_capped(upload, ctx.max_upload_bytes)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise RTSPBackendError(
                "Uploaded file is not a decodable image.",
                status_code=400, code="bad_image")
        return img, "upload"

    cam = ctx.manager.get(camera_id) if camera_id else ctx.manager.get_active()
    frame, *_ = cam.buffer.latest()
    if frame is None:
        raise RTSPBackendError("Camera has no frame yet.", status_code=503,
                               code="frame_unavailable")
    return frame, f"rtsp:{getattr(cam, 'camera_id', camera_id) or 'active'}"


def _components_payload(result: dict) -> list[dict]:
    """The brief's minimal component list, from the inspection result.

    ``class`` carries the canonical taxonomy id (``"mcb"``), which is what an API
    client should branch on, and ``class_name`` carries the human label
    (``"MCB (Miniature Circuit Breaker)"``). Returning only a display name would
    make the contract fragile the moment a label is reworded.
    """
    out: list[dict] = []
    for c in (result.get("components") or [])[:MAX_COMPONENTS]:
        cid = c.get("class_id") or tax.UNKNOWN_COMPONENT_ID
        bbox = [float(v) for v in (c.get("bbox") or [0, 0, 0, 0])[:4]]
        out.append({
            "class": cid,
            "class_name": tax.display_name(cid),
            "confidence": round(float(c.get("confidence") or 0.0), 4),
            "bbox": [round(v, 1) for v in bbox],
            "is_unknown": cid == tax.UNKNOWN_COMPONENT_ID,
            # Extra identification the engine already worked out. Present when
            # OCR is installed and a nameplate was readable, null otherwise.
            "manufacturer": c.get("manufacturer"),
            "part_number": c.get("part_number"),
            "category": c.get("category"),
        })
    return out


def _report_payload(result: dict) -> dict:
    """Detected / missing / unknown components, plus confidence."""
    components = result.get("components") or []
    unknown = [c for c in components
               if (c.get("class_id") or "") == tax.UNKNOWN_COMPONENT_ID]
    known = [c for c in components
             if (c.get("class_id") or "") != tax.UNKNOWN_COMPONENT_ID]
    conf = result.get("confidence") or {}
    panel = result.get("panel") or {}

    return {
        "detected_components": [
            {"class": b.get("class_id"), "class_name": b.get("name"),
             "count": b.get("count"), "category": b.get("category"),
             "mean_confidence": b.get("mean_confidence")}
            for b in (result.get("bill_of_materials") or [])
        ],
        # "Missing" is an inference from the panel type, not a measurement: a
        # panel classified as a motor-control centre with no overload relay is
        # probably missing one. Each entry carries its own reasoning and
        # confidence so it is never mistaken for a detection.
        "missing_components": [
            {"class": m.get("class_id"), "class_name": m.get("name"),
             "reason": m.get("reason"), "confidence": m.get("confidence"),
             "severity": m.get("severity")}
            for m in (result.get("missing_components") or [])
        ],
        "unknown_components": {
            "count": len(unknown),
            "note": (
                "Detected as devices but not classified with enough confidence "
                "to name. Reported honestly rather than guessed; feed these "
                "crops back into the dataset — they are exactly the examples "
                "the model needs."),
            "items": [
                {"bbox": [round(float(v), 1) for v in (c.get("bbox") or [])[:4]],
                 "confidence": round(float(c.get("confidence") or 0.0), 4),
                 "position": c.get("position")}
                for c in unknown
            ],
        },
        "confidence": {
            "mean": conf.get("mean"),
            "min": conf.get("min"),
            "max": conf.get("max"),
            "identified": len(known),
            "unknown": len(unknown),
            "identification_rate": (round(len(known) / len(components), 4)
                                    if components else None),
        },
        "panel": {
            "type": panel.get("panel_type"),
            "type_name": panel.get("panel_type_name"),
            "confidence": panel.get("confidence"),
            "function": panel.get("function"),
        },
        "component_total": result.get("component_total", 0),
        "component_counts": result.get("component_counts") or {},
        "maintenance_notes": result.get("maintenance_notes") or [],
        "layout": result.get("layout") or {},
    }


def build_router(ctx) -> APIRouter:
    r = APIRouter(prefix="/api/panel", tags=["panel"])
    db = ctx.db

    @r.post("/analyze")
    async def analyze_panel(
        file: Optional[UploadFile] = File(
            None, description="Panel image. Omit to grab a frame from a camera."),
        camera_id: Optional[str] = Form(
            None, description="RTSP camera to grab from when no file is given."),
        report: bool = Query(
            True, description="Include the full panel report."),
        annotate: bool = Query(
            True, description="Render and persist an annotated image."),
        persist: bool = Query(
            True, description="Record the analysis in the reports table."),
        min_confidence: float = Query(
            0.0, ge=0.0, le=1.0,
            description="Drop components below this confidence from the "
                        "response. Filtering happens after the engine's "
                        "per-class gating, so it can only tighten the result."),
    ):
        """Detect every electrical component in a panel image.

        Returns ``components`` as specified in the brief; add ``report=false``
        for the bare list, or keep the default for the full panel report.
        """
        img, source = await _image_from(ctx, file, camera_id)
        height, width = img.shape[:2]

        # Inference is CPU-bound and blocking; off the event loop it goes, or a
        # single 960px analysis stalls every other request and the RTSP pipeline.
        result = await asyncio.to_thread(panel_svc.analyze, ctx.ai, img, annotate)
        annotated = result.pop("_annotated", None)

        components = _components_payload(result)
        if min_confidence > 0.0:
            components = [c for c in components
                          if c["confidence"] >= min_confidence]

        annotated_rel = None
        if annotate and annotated is not None:
            annotated_rel = save_image(ctx.data_dir, "panels", annotated,
                                       prefix="panel")

        payload: dict = {
            "components": components,
            "component_total": len(components),
            "image": {"width": int(width), "height": int(height),
                      "source": source},
            "model": {
                "loaded": bool(result.get("component_model_loaded")),
                "engine": result.get("engine"),
                "engine_version": result.get("engine_version"),
                "backend": (getattr(ctx.ai.backend("components"), "backend_id",
                                    None) if ctx.ai is not None else None),
            },
            "bbox_format": "xyxy_absolute_pixels",
            "duration_ms": result.get("duration_ms"),
            "notes": result.get("notes") or [],
        }
        if annotated_rel:
            payload["annotated_image"] = annotated_rel
        if result.get("component_total", 0) > MAX_COMPONENTS:
            payload["notes"].append(
                f"Detection count exceeded the {MAX_COMPONENTS}-component "
                f"response cap and the list was truncated. A real panel image "
                f"does not contain this many devices — review the model before "
                f"trusting this result.")
        if not payload["model"]["loaded"]:
            # The engine and panel_svc each already say something about the
            # missing model. Adding a third near-identical sentence buries the
            # actionable one, so replace that pile with a single clear note and
            # keep the engine's specific diagnostics.
            payload["notes"] = [
                n for n in payload["notes"]
                if "no trained component" not in n.lower()
            ] + [
                "No trained component model is loaded, so no components were "
                "detected. This is an honest empty result, not a failure to find "
                "anything: train and install a detector (see "
                "docs/ELECTRICAL_MODEL_TRAINING.md)."
            ]

        if report:
            payload["report"] = _report_payload(result)

        if persist:
            summary = {
                "component_total": len(components),
                "component_counts": result.get("component_counts") or {},
                "unknown_components": len(
                    [c for c in components if c["is_unknown"]]),
                "mean_confidence": (result.get("confidence") or {}).get("mean"),
                "panel_type": (result.get("panel") or {}).get("panel_type"),
                "model_loaded": payload["model"]["loaded"],
                "source": source,
                "annotated": annotated_rel,
                "duration_ms": result.get("duration_ms"),
            }
            json_rel = reports_svc.write_json(ctx.data_dir, "reports", result,
                                              "panel")
            summary["json"] = json_rel
            payload["id"] = db.insert(
                "INSERT INTO reports(kind,title,path,summary,created_at) "
                "VALUES(?,?,?,?,?)",
                ("panel_analysis", "Component Detection", json_rel,
                 json.dumps(summary), time.time()))

        return payload

    @r.get("/classes")
    async def panel_classes():
        """The label space the detector reports against.

        Clients that render component names or build filters should read this
        rather than hardcoding a list, because ``CLASS_ORDER`` is append-only and
        grows with each retrain.
        """
        return {
            "taxonomy_version": "5.1",
            "class_count": len(tax.CLASS_ORDER),
            "unknown_class": tax.UNKNOWN_COMPONENT_ID,
            "classes": [
                {"class": cid, "name": tax.display_name(cid),
                 "category": tax.SPECS[cid].category,
                 "domain": tax.SPECS[cid].domain,
                 "min_confidence": tax.SPECS[cid].min_conf}
                for cid in tax.CLASS_ORDER
            ],
        }

    @r.get("/model")
    async def panel_model_status():
        """Whether a trained detector is loaded, and what to do if not."""
        backend = ctx.ai.backend("components") if ctx.ai is not None else None
        ready = bool(backend is not None and getattr(backend, "ready", False))
        return {
            "loaded": ready,
            "backend": getattr(backend, "backend_id", None),
            "display_name": getattr(backend, "display_name", None),
            "status": getattr(backend, "_status", None),
            "reason": (getattr(backend, "_reason", None)
                       or getattr(backend, "_error", None)),
            "weights": getattr(backend, "_weights_path", None),
            "remedy": (None if ready else
                       "Install a trained bundle into models/components/ "
                       "(best.onnx + labels.txt + classes.json) — see "
                       "docs/ELECTRICAL_MODEL_TRAINING.md — then select the "
                       "'industrial_onnx' backend."),
        }

    return r


__all__ = ["build_router", "MAX_COMPONENTS"]
