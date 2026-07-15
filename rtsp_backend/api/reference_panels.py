"""
Reference Panel REST API (Feature 11).

    POST   /api/reference-panels                 create a reference panel
    GET    /api/reference-panels                 list panels
    GET    /api/reference-panels/{id}            full panel (images/components/
                                                 terminals/wires/graph/template)
    DELETE /api/reference-panels/{id}            delete panel + its data
    POST   /api/reference-panels/{id}/capture    grab a frame from an RTSP camera
    POST   /api/reference-panels/{id}/upload     upload one or more images
    POST   /api/reference-panels/{id}/learn      learn the template + graph
    POST   /api/reference-panels/{id}/compare    inspect an image vs the panel
    GET    /api/reference-panels/{id}/result     latest (or ?result_id=) result
    GET    /api/reference-panels/{id}/results    inspection history

Capturing reads the *existing* camera frame buffer (never opens a second RTSP
connection), and the CPU-bound learn/compare/analysis runs in a worker thread so
the event loop and camera pipeline never block.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, Query, UploadFile
from pydantic import BaseModel

from ..api.util import read_upload_capped
from ..errors import RTSPBackendError
from ..reference_panels_svc import ReferencePanelService


class CreatePanelBody(BaseModel):
    name: str
    version: str = "v1"
    description: Optional[str] = None


def build_router(ctx) -> APIRouter:
    r = APIRouter(prefix="/api/reference-panels", tags=["reference-panels"])
    svc = ReferencePanelService(ctx.db, ctx.ai, ctx.data_dir)
    # expose for reuse (e.g. a future live-inspection worker)
    ctx.reference_panels = svc

    async def _decode_upload(file: UploadFile) -> np.ndarray:
        raw = await read_upload_capped(file, ctx.max_upload_bytes)
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RTSPBackendError("Uploaded file is not a decodable image.",
                                   status_code=400, code="bad_image")
        return img

    def _camera_frame(camera_id: Optional[str]) -> tuple[np.ndarray, str]:
        cam = ctx.manager.get(camera_id) if camera_id else ctx.manager.get_active()
        frame, *_ = cam.buffer.latest()
        if frame is None:
            raise RTSPBackendError("Camera has no frame yet.", status_code=503,
                                   code="frame_unavailable")
        return frame, cam.config.id

    # -- CRUD --------------------------------------------------------------

    @r.post("", status_code=201)
    async def create_panel(body: CreatePanelBody):
        return svc.create(body.name, body.version, body.description)

    @r.get("")
    async def list_panels(limit: int = Query(100, ge=1, le=1000)):
        panels = svc.list_panels(limit)
        return {"panels": panels, "total": len(panels)}

    @r.get("/{panel_id}")
    async def get_panel(panel_id: int):
        return svc.get(panel_id)

    @r.delete("/{panel_id}")
    async def delete_panel(panel_id: int):
        svc.delete(panel_id)
        return {"deleted": panel_id}

    # -- images ------------------------------------------------------------

    @r.post("/{panel_id}/capture")
    async def capture(panel_id: int, camera_id: Optional[str] = Form(None)):
        frame, cam_id = _camera_frame(camera_id)
        img = await asyncio.to_thread(
            svc.add_image, panel_id, frame.copy(), "camera", cam_id)
        return img

    @r.post("/{panel_id}/upload")
    async def upload(panel_id: int, files: list[UploadFile] = File(...)):
        added = []
        for f in files:
            img = await _decode_upload(f)
            added.append(await asyncio.to_thread(
                svc.add_image, panel_id, img, "upload", None))
        return {"panel_id": panel_id, "added": added, "count": len(added)}

    # -- learn -------------------------------------------------------------

    @r.post("/{panel_id}/learn")
    async def learn(panel_id: int):
        return await asyncio.to_thread(svc.learn, panel_id)

    # -- compare / inspect -------------------------------------------------

    @r.post("/{panel_id}/compare")
    async def compare(panel_id: int,
                      file: Optional[UploadFile] = File(None),
                      camera_id: Optional[str] = Form(None)):
        if file is not None:
            img = await _decode_upload(file)
            source, cam_id = "upload", None
        else:
            frame, cam_id = _camera_frame(camera_id)
            img, source = frame.copy(), "camera"
        return await asyncio.to_thread(svc.compare, panel_id, img, source, cam_id)

    @r.get("/{panel_id}/result")
    async def get_result(panel_id: int, result_id: Optional[int] = Query(None)):
        return svc.get_result(panel_id, result_id)

    @r.get("/{panel_id}/results")
    async def list_results(panel_id: int, limit: int = Query(50, ge=1, le=500)):
        results = svc.list_results(panel_id, limit)
        return {"panel_id": panel_id, "results": results, "total": len(results)}

    return r
