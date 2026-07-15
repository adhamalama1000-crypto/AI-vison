"""
Datasheet / schematic API (Feature 7).

    POST   /api/datasheets/upload        upload a PDF/PNG/JPG/DXF/SVG
    POST   /api/datasheets/{id}/extract  OCR + parse -> IDs + expected graph
    GET    /api/datasheets               list datasheets
    GET    /api/datasheets/{id}          full record (extraction + graph)
    DELETE /api/datasheets/{id}          delete

Extraction runs in a worker thread (OCR can be slow) and records which engine
was used; if no OCR engine is installed for a raster/PDF input it says so
instead of inventing component IDs.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Optional

from fastapi import APIRouter, File, Form, Query, UploadFile
from pydantic import BaseModel

from ..api.util import copy_upload_capped
from ..errors import RTSPBackendError
from ..panels import datasheet as _datasheet


def _row(r) -> dict:
    d = dict(r)
    for k in ("extracted", "expected_graph"):
        if d.get(k):
            try:
                d[k] = json.loads(d[k])
            except (TypeError, json.JSONDecodeError):
                pass
    return d


def build_router(ctx) -> APIRouter:
    r = APIRouter(prefix="/api/datasheets", tags=["datasheets"])
    db = ctx.db

    @r.get("")
    async def list_datasheets(limit: int = Query(100, ge=1, le=1000)):
        rows = db.query(
            "SELECT id,name,kind,path,panel_id,description,ocr_engine,status,"
            "created_at,updated_at FROM datasheets ORDER BY created_at DESC LIMIT ?",
            (limit,))
        return {"datasheets": [dict(x) for x in rows], "total": len(rows)}

    @r.get("/{ds_id}")
    async def get_datasheet(ds_id: int):
        row = db.query_one("SELECT * FROM datasheets WHERE id=?", (ds_id,))
        if not row:
            raise RTSPBackendError("Datasheet not found.", status_code=404,
                                   code="not_found")
        return _row(row)

    @r.post("/upload")
    async def upload(file: UploadFile = File(...),
                     name: Optional[str] = Form(None),
                     description: Optional[str] = Form(None),
                     panel_id: Optional[int] = Form(None)):
        fname = file.filename or "datasheet"
        kind = _datasheet.kind_for(fname)
        ts = int(time.time() * 1000)
        safe = os.path.basename(fname).replace("/", "_")
        rel = f"datasheets/{ts}_{safe}"
        path = os.path.join(ctx.data_dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        copy_upload_capped(file, path, ctx.max_upload_bytes)
        now = time.time()
        ds_id = db.insert(
            "INSERT INTO datasheets(name,kind,path,panel_id,description,status,"
            "created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
            (name or safe, kind, rel, panel_id, description, "uploaded", now, now))
        return {"id": ds_id, "name": name or safe, "kind": kind, "path": rel,
                "status": "uploaded"}

    @r.post("/{ds_id}/extract")
    async def extract(ds_id: int):
        row = db.query_one("SELECT * FROM datasheets WHERE id=?", (ds_id,))
        if not row:
            raise RTSPBackendError("Datasheet not found.", status_code=404,
                                   code="not_found")
        full = os.path.join(ctx.data_dir, row["path"])
        if not os.path.isfile(full):
            raise RTSPBackendError("Datasheet file missing on disk.", status_code=404,
                                   code="not_found")
        db.execute("UPDATE datasheets SET status='processing', updated_at=? WHERE id=?",
                   (time.time(), ds_id))
        result = await asyncio.to_thread(_datasheet.extract, full)
        db.execute(
            "UPDATE datasheets SET status='extracted', ocr_engine=?, extracted=?, "
            "expected_graph=?, updated_at=? WHERE id=?",
            (result["ocr_engine"], json.dumps(result["parsed"]),
             json.dumps(result["expected_graph"]), time.time(), ds_id))
        return {"id": ds_id, **result}

    @r.delete("/{ds_id}")
    async def delete_datasheet(ds_id: int):
        row = db.query_one("SELECT path FROM datasheets WHERE id=?", (ds_id,))
        if row:
            full = os.path.join(ctx.data_dir, row["path"])
            if os.path.isfile(full):
                try:
                    os.remove(full)
                except OSError:
                    pass
        db.execute("DELETE FROM datasheets WHERE id=?", (ds_id,))
        return {"deleted": ds_id}

    return r
