"""Reference design storage endpoints (Part 9)."""

from __future__ import annotations

import json
import os
import time
from typing import Optional

from fastapi import APIRouter, File, Form, Query, UploadFile
from pydantic import BaseModel

from ..api.util import copy_upload_capped
from ..errors import RTSPBackendError

_KIND_BY_EXT = {
    ".pdf": "pdf", ".png": "image", ".jpg": "image", ".jpeg": "image",
    ".bmp": "image", ".webp": "image", ".dxf": "dxf", ".dwg": "dwg",
}


class SpecBody(BaseModel):
    spec: dict


def _row(r) -> dict:
    d = dict(r)
    if d.get("spec"):
        try:
            d["spec"] = json.loads(d["spec"])
        except (TypeError, json.JSONDecodeError):
            pass
    return d


def build_router(ctx) -> APIRouter:
    r = APIRouter(prefix="/api/reference", tags=["reference"])
    db = ctx.db

    @r.get("")
    async def list_refs(limit: int = Query(100, ge=1, le=1000)):
        rows = db.query(
            "SELECT id,name,kind,path,description,created_at FROM reference_designs "
            "ORDER BY created_at DESC LIMIT ?", (limit,))
        return {"references": [dict(x) for x in rows], "total": len(rows)}

    @r.get("/{ref_id}")
    async def get_ref(ref_id: int):
        row = db.query_one("SELECT * FROM reference_designs WHERE id=?", (ref_id,))
        if not row:
            raise RTSPBackendError("Reference not found.", status_code=404,
                                   code="not_found")
        return _row(row)

    @r.post("/upload")
    async def upload(
        file: UploadFile = File(...),
        name: Optional[str] = Form(None),
        description: Optional[str] = Form(None),
    ):
        fname = file.filename or "reference"
        ext = os.path.splitext(fname)[1].lower()
        kind = _KIND_BY_EXT.get(ext, "other")
        ts = int(time.time() * 1000)
        safe = os.path.basename(fname).replace("/", "_")
        rel = f"reference/{ts}_{safe}"
        path = os.path.join(ctx.data_dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        copy_upload_capped(file, path, ctx.max_upload_bytes)
        ref_id = db.insert(
            "INSERT INTO reference_designs(name,kind,path,description,created_at) "
            "VALUES(?,?,?,?,?)",
            (name or safe, kind, rel, description, time.time()))
        return {"id": ref_id, "name": name or safe, "kind": kind, "path": rel}

    @r.put("/{ref_id}/spec")
    async def set_spec(ref_id: int, body: SpecBody):
        row = db.query_one("SELECT id FROM reference_designs WHERE id=?", (ref_id,))
        if not row:
            raise RTSPBackendError("Reference not found.", status_code=404,
                                   code="not_found")
        db.execute("UPDATE reference_designs SET spec=? WHERE id=?",
                   (json.dumps(body.spec), ref_id))
        return {"id": ref_id, "spec": body.spec}

    @r.delete("/{ref_id}")
    async def delete_ref(ref_id: int):
        row = db.query_one("SELECT path FROM reference_designs WHERE id=?", (ref_id,))
        if row:
            full = os.path.join(ctx.data_dir, row["path"])
            if os.path.isfile(full):
                try:
                    os.remove(full)
                except OSError:
                    pass
        db.execute("DELETE FROM reference_designs WHERE id=?", (ref_id,))
        return {"deleted": ref_id}

    return r
