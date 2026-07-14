"""Reference-vs-live inspection endpoints (Part 10)."""

from __future__ import annotations

import json
import os
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, Query, UploadFile

from .. import inspection_svc, panel_svc, reports_svc
from ..api.util import save_image
from ..errors import RTSPBackendError


def _load_spec_for(ctx, ref_row) -> dict:
    """Return the expected spec for a reference: an explicit stored spec, else
    derived by analysing the reference image with the same pipeline."""
    if ref_row["spec"]:
        try:
            return json.loads(ref_row["spec"])
        except (TypeError, json.JSONDecodeError):
            pass
    if ref_row["kind"] == "image":
        img_path = os.path.join(ctx.data_dir, ref_row["path"])
        img = cv2.imread(img_path)
        if img is not None:
            ref_result = panel_svc.analyze(ctx.ai, img, annotate=False)
            return inspection_svc.build_expected_from_analysis(ref_result)
    return {"component_counts": {}, "wire_color_counts": {}}


def build_router(ctx) -> APIRouter:
    r = APIRouter(prefix="/api/inspection", tags=["inspection"])
    db = ctx.db

    @r.get("")
    async def list_inspections(limit: int = Query(50, ge=1, le=500)):
        rows = db.query(
            "SELECT id,reference_id,camera_id,source,status,n_mismatches,"
            "report_path,created_at FROM inspections ORDER BY created_at DESC LIMIT ?",
            (limit,))
        return {"inspections": [dict(x) for x in rows], "total": len(rows)}

    @r.get("/{insp_id}")
    async def get_inspection(insp_id: int):
        row = db.query_one("SELECT * FROM inspections WHERE id=?", (insp_id,))
        if not row:
            raise RTSPBackendError("Inspection not found.", status_code=404,
                                   code="not_found")
        d = dict(row)
        if d.get("result"):
            try:
                d["result"] = json.loads(d["result"])
            except (TypeError, json.JSONDecodeError):
                pass
        return d

    @r.post("/run")
    async def run(
        reference_id: int = Form(...),
        file: Optional[UploadFile] = File(None),
        camera_id: Optional[str] = Form(None),
        make_pdf: bool = Form(True),
    ):
        ref = db.query_one("SELECT * FROM reference_designs WHERE id=?", (reference_id,))
        if not ref:
            raise RTSPBackendError("Reference not found.", status_code=404,
                                   code="not_found")

        # observed image
        if file is not None:
            raw = await file.read()
            arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            source = "upload"
            if img is None:
                raise RTSPBackendError("Uploaded file is not a decodable image.",
                                       status_code=400, code="bad_image")
        else:
            cam = ctx.manager.get(camera_id) if camera_id else ctx.manager.get_active()
            frame, *_ = cam.buffer.latest()
            if frame is None:
                raise RTSPBackendError("Camera has no frame yet.", status_code=503,
                                       code="frame_unavailable")
            img, source, camera_id = frame, "camera", cam.config.id

        observed = panel_svc.analyze(ctx.ai, img, annotate=True)
        annotated = observed.pop("_annotated", None)
        expected = _load_spec_for(ctx, ref)
        comparison = inspection_svc.compare(expected, observed)

        # highlight mismatches on the annotated frame
        if annotated is not None:
            _stamp_status(annotated, comparison)
            annotated_rel = save_image(ctx.data_dir, "inspections", annotated,
                                       prefix="insp")
        else:
            annotated_rel = None

        full = {**comparison, "observed_detail": observed,
                "reference": {"id": ref["id"], "name": ref["name"]}}
        pdf_rel = None
        if make_pdf:
            pdf_rel = reports_svc.inspection_pdf(ctx.data_dir, comparison, annotated_rel)
        json_rel = reports_svc.write_json(ctx.data_dir, "reports", full, "inspection")

        insp_id = db.insert(
            "INSERT INTO inspections(reference_id,camera_id,source,status,"
            "n_mismatches,result,snapshot,report_path,created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (reference_id, camera_id, source, comparison["status"],
             comparison["n_mismatches"], json.dumps(full), annotated_rel,
             pdf_rel or json_rel, time.time()))
        db.insert(
            "INSERT INTO reports(kind,title,ref_id,path,summary,created_at) "
            "VALUES(?,?,?,?,?,?)",
            ("inspection", f"Inspection vs {ref['name']}", insp_id,
             pdf_rel or json_rel,
             json.dumps({"status": comparison["status"],
                         "n_mismatches": comparison["n_mismatches"]}),
             time.time()))
        return {"id": insp_id, "status": comparison["status"],
                "n_mismatches": comparison["n_mismatches"],
                "mismatches": comparison["mismatches"],
                "annotated": annotated_rel, "pdf": pdf_rel, "json": json_rel,
                "result": full}

    return r


def _stamp_status(img, comparison) -> None:
    status = comparison["status"]
    color = {"pass": (80, 200, 80), "warning": (0, 180, 240),
             "fail": (0, 0, 230)}.get(status, (200, 200, 200))
    txt = f"{status.upper()}  ({comparison['n_mismatches']} mismatch)"
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (30, 30, 30), -1)
    cv2.putText(img, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                cv2.LINE_AA)
