"""Panel analysis endpoints (Part 8)."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, Query, UploadFile

from .. import panel_svc, reports_svc
from ..api.util import read_upload_capped, save_image
from ..errors import RTSPBackendError


async def _image_from(ctx, upload: Optional[UploadFile], camera_id: Optional[str]):
    if upload is not None:
        raw = await read_upload_capped(upload, ctx.max_upload_bytes)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise RTSPBackendError("Uploaded file is not a decodable image.",
                                   status_code=400, code="bad_image")
        return img
    cam = ctx.manager.get(camera_id) if camera_id else ctx.manager.get_active()
    frame, *_ = cam.buffer.latest()
    if frame is None:
        raise RTSPBackendError("Camera has no frame yet.", status_code=503,
                               code="frame_unavailable")
    return frame


def build_router(ctx) -> APIRouter:
    r = APIRouter(prefix="/api/panels", tags=["panels"])
    db = ctx.db

    @r.get("")
    async def list_panels(limit: int = Query(50, ge=1, le=500)):
        rows = db.query(
            "SELECT * FROM reports WHERE kind='panel_analysis' "
            "ORDER BY created_at DESC LIMIT ?", (limit,))
        out = []
        for x in rows:
            d = dict(x)
            if d.get("summary"):
                try:
                    d["summary"] = json.loads(d["summary"])
                except (TypeError, json.JSONDecodeError):
                    pass
            out.append(d)
        return {"panels": out, "total": len(out)}

    @r.post("/analyze")
    async def analyze(
        file: Optional[UploadFile] = File(None),
        camera_id: Optional[str] = Form(None),
        make_pdf: bool = Form(True),
    ):
        img = await _image_from(ctx, file, camera_id)
        result = await asyncio.to_thread(panel_svc.analyze, ctx.ai, img, True)
        annotated = result.pop("_annotated", None)

        annotated_rel = None
        if annotated is not None:
            annotated_rel = save_image(ctx.data_dir, "panels", annotated, prefix="panel")
        json_rel = reports_svc.write_json(ctx.data_dir, "reports", result, "panel")
        pdf_rel = None
        if make_pdf:
            pdf_rel = reports_svc.panel_pdf(ctx.data_dir, result, annotated_rel)

        summary = {
            "component_total": result["component_total"],
            "component_counts": result["component_counts"],
            "wire_total": result["wire_total"],
            "wire_color_counts": result["wire_color_counts"],
            "annotated": annotated_rel, "json": json_rel, "pdf": pdf_rel,
        }
        rid = db.insert(
            "INSERT INTO reports(kind,title,path,summary,created_at) "
            "VALUES(?,?,?,?,?)",
            ("panel_analysis", "Panel Analysis", pdf_rel or json_rel,
             json.dumps(summary), time.time()))
        return {"id": rid, "result": result, **summary}

    return r
