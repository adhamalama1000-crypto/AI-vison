"""Dataset upload, auto-detection, and validation endpoints (Part 2)."""

from __future__ import annotations

import json
import os
import shutil
import time
from typing import Optional

from fastapi import APIRouter, File, Form, Query, UploadFile

from ..datasets_svc import detect_kind, safe_extract_zip, validate
from ..errors import RTSPBackendError


def _row(r) -> dict:
    d = dict(r)
    for k in ("classes", "report"):
        if d.get(k):
            try:
                d[k] = json.loads(d[k])
            except (TypeError, json.JSONDecodeError):
                pass
    return d


def build_router(ctx) -> APIRouter:
    r = APIRouter(prefix="/api/datasets", tags=["datasets"])
    db = ctx.db
    base = os.path.join(ctx.data_dir, "datasets")

    def _analyze_and_store(ds_id: int, root: str) -> dict:
        kind = detect_kind(root)
        report = validate(root, kind)
        n_images = report["n_images"]
        classes = report.get("classes", [])
        status = "valid" if report.get("ok") else "invalid"
        db.execute(
            "UPDATE datasets SET kind=?, status=?, n_images=?, n_labels=?, "
            "n_classes=?, classes=?, report=?, updated_at=? WHERE id=?",
            (kind, status, n_images, report.get("n_labels", 0),
             report.get("n_classes", 0), json.dumps(classes),
             json.dumps(report), time.time(), ds_id),
        )
        return report

    @r.get("")
    async def list_datasets(limit: int = Query(100, ge=1, le=1000)):
        rows = db.query(
            "SELECT id,name,kind,status,n_images,n_labels,n_classes,created_at,"
            "updated_at FROM datasets ORDER BY created_at DESC LIMIT ?", (limit,))
        return {"datasets": [dict(x) for x in rows], "total": len(rows)}

    @r.get("/{ds_id}")
    async def get_dataset(ds_id: int):
        row = db.query_one("SELECT * FROM datasets WHERE id=?", (ds_id,))
        if not row:
            raise RTSPBackendError("Dataset not found.", status_code=404, code="not_found")
        return _row(row)

    @r.post("/upload")
    async def upload(
        files: list[UploadFile] = File(...),
        name: Optional[str] = Form(None),
    ):
        ts = int(time.time() * 1000)
        ds_name = name or (files[0].filename or f"dataset_{ts}")
        rel = f"datasets/ds_{ts}"
        root = os.path.join(ctx.data_dir, rel)
        os.makedirs(root, exist_ok=True)

        for uf in files:
            # preserve any relative path a folder upload sends (webkitRelativePath
            # arrives as the filename with slashes); guard against traversal.
            fname = uf.filename or "file"
            safe_parts = [p for p in fname.replace("\\", "/").split("/")
                          if p not in ("", ".", "..")]
            dest = os.path.join(root, *safe_parts) if safe_parts else os.path.join(root, "file")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as fh:
                shutil.copyfileobj(uf.file, fh)
            # auto-extract a single uploaded zip in place
            if dest.lower().endswith(".zip"):
                try:
                    safe_extract_zip(dest, root)
                    os.remove(dest)
                except Exception as exc:
                    raise RTSPBackendError(
                        f"Failed to extract archive: {exc}", status_code=400,
                        code="bad_archive")

        ds_id = db.insert(
            "INSERT INTO datasets(name,kind,path,status,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?)",
            (ds_name, "unknown", rel, "validating", time.time(), time.time()))
        report = _analyze_and_store(ds_id, root)
        return {"id": ds_id, "name": ds_name, "path": rel, "report": report}

    @r.post("/{ds_id}/revalidate")
    async def revalidate(ds_id: int):
        row = db.query_one("SELECT * FROM datasets WHERE id=?", (ds_id,))
        if not row:
            raise RTSPBackendError("Dataset not found.", status_code=404, code="not_found")
        root = os.path.join(ctx.data_dir, row["path"])
        report = _analyze_and_store(ds_id, root)
        return {"id": ds_id, "report": report}

    @r.delete("/{ds_id}")
    async def delete_dataset(ds_id: int):
        row = db.query_one("SELECT path FROM datasets WHERE id=?", (ds_id,))
        if row:
            full = os.path.join(ctx.data_dir, row["path"])
            if os.path.isdir(full):
                shutil.rmtree(full, ignore_errors=True)
        db.execute("DELETE FROM datasets WHERE id=?", (ds_id,))
        return {"deleted": ds_id}

    return r
