"""
AI Image Analysis & Comparison REST API.

    POST   /api/images/upload              upload an image (multipart `file`)
    POST   /api/images/analyze             analyze an uploaded `file` or `image_id`
    POST   /api/images/compare             compare reference vs current
                                           (`reference`/`current` files or
                                            `reference_id`/`current_id`)
    GET    /api/images                     list uploaded images
    GET    /api/images/{id}                one image + analysis
    DELETE /api/images/{id}                delete an image
    GET    /api/images/history             images + comparisons
    GET    /api/images/comparisons         list comparisons
    GET    /api/images/report/{id}         comparison report (JSON, `?pdf=1` link)

Uploads are size-capped (``max_upload_bytes``) and validated by actually
decoding them (a non-image is rejected 400). CPU-bound analysis/comparison runs
in a worker thread. Nothing here touches the camera/face/panel subsystems.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, File, Form, Query, UploadFile

from ..api.util import read_upload_capped
from ..errors import RTSPBackendError
from ..imaging import ImageService


def build_router(ctx) -> APIRouter:
    r = APIRouter(prefix="/api/images", tags=["images"])
    svc = ImageService(ctx.db, ctx.ai, ctx.data_dir)
    ctx.images = svc

    async def _store(file: UploadFile) -> dict:
        raw = await read_upload_capped(file, ctx.max_upload_bytes)
        return await asyncio.to_thread(svc.store_image, raw, file.filename)

    # -- upload ------------------------------------------------------------

    @r.post("/upload", status_code=201)
    async def upload(file: UploadFile = File(...)):
        return await _store(file)

    # -- analyze -----------------------------------------------------------

    @r.post("/analyze")
    async def analyze(file: Optional[UploadFile] = File(None),
                      image_id: Optional[int] = Form(None)):
        if file is not None:
            stored = await _store(file)
            image_id = stored["id"]
        if image_id is None:
            raise RTSPBackendError("Provide an image file or image_id.",
                                   status_code=400, code="bad_request")
        return await asyncio.to_thread(svc.analyze, image_id)

    # -- compare -----------------------------------------------------------

    @r.post("/compare")
    async def compare(reference: Optional[UploadFile] = File(None),
                      current: Optional[UploadFile] = File(None),
                      reference_id: Optional[int] = Form(None),
                      current_id: Optional[int] = Form(None),
                      make_pdf: bool = Form(True)):
        if reference is not None:
            reference_id = (await _store(reference))["id"]
        if current is not None:
            current_id = (await _store(current))["id"]
        if reference_id is None or current_id is None:
            raise RTSPBackendError(
                "Provide reference & current images (files or ids).",
                status_code=400, code="bad_request")
        return await asyncio.to_thread(svc.compare, reference_id, current_id, make_pdf)

    # -- reads -------------------------------------------------------------

    @r.get("")
    async def list_images(limit: int = Query(100, ge=1, le=1000)):
        imgs = svc.list_images(limit)
        return {"images": imgs, "total": len(imgs)}

    @r.get("/history")
    async def history(limit: int = Query(100, ge=1, le=1000)):
        return {"images": svc.list_images(limit),
                "comparisons": svc.list_comparisons(limit)}

    @r.get("/comparisons")
    async def comparisons(limit: int = Query(100, ge=1, le=1000)):
        rows = svc.list_comparisons(limit)
        return {"comparisons": rows, "total": len(rows)}

    @r.get("/report/{cmp_id}")
    async def report(cmp_id: int):
        return svc.get_comparison(cmp_id)

    @r.get("/{image_id}")
    async def get_image(image_id: int):
        return svc.get(image_id)

    @r.delete("/{image_id}")
    async def delete_image(image_id: int):
        svc.delete_image(image_id)
        return {"deleted": image_id}

    return r
