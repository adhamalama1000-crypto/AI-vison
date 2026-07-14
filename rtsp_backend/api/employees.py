"""Employee management and face enrolment endpoints.

All CPU-bound work (image decode, face detection, embedding, disk I/O) is
executed with ``asyncio.to_thread`` so it runs in the threadpool and never
blocks the event loop. This is what keeps the MJPEG stream smooth while a
capture or enrolment is in progress — the earlier version ran detection inline
on the loop and froze streaming.
"""

from __future__ import annotations

import asyncio
import os
import time

from fastapi import APIRouter

from ..errors import RTSPBackendError
from .ai_schemas import CaptureRequest, EmployeeCreate, EmployeeUpdate, ImageUpload, RegisterFromCaptures
from .util import Context, decode_image, employee_dict, encode_jpeg_b64, save_image


class NotFound(RTSPBackendError):
    status_code = 404
    code = "not_found"


def build_router(ctx: Context) -> APIRouter:
    r = APIRouter(prefix="/api/employees", tags=["employees"])
    db = ctx.db

    def _get_or_404(emp_id: int):
        row = db.query_one("SELECT * FROM employees WHERE id=?", (emp_id,))
        if row is None:
            raise NotFound(f"Employee {emp_id} not found.")
        return row

    # ---- blocking helpers (run via asyncio.to_thread) -------------------

    def _enroll_blocking(emp_id: int, img):
        """Save the image and enrol a face. Pure blocking work for a thread."""
        rel = save_image(ctx.data_dir, f"employees/{emp_id}", img, "face")
        image_id = db.insert(
            "INSERT INTO employee_images(employee_id, path, created_at) VALUES(?,?,?)",
            (emp_id, rel, time.time()),
        )
        enroll = {"ok": False, "reason": "face_service_unavailable"}
        if ctx.ai.face_service is not None:
            try:
                enroll = ctx.ai.face_service.enroll_image(emp_id, img, image_id)
            except Exception as exc:
                # A malformed / degenerate frame must never crash enrolment.
                # Treat it as a rejected capture with a clear reason.
                enroll = {"ok": False, "reason": "enrollment_error",
                          "detail": f"{type(exc).__name__}: {exc}"}
        # If enrolment was rejected (blur / no face), don't keep a useless image row.
        if not enroll.get("ok"):
            db.execute("DELETE FROM employee_images WHERE id=?", (image_id,))
            try:
                os.remove(os.path.join(ctx.data_dir, rel))
            except OSError:
                pass
            return {"image_id": None, "path": None, "enrollment": enroll}
        # A face was just enrolled: guarantee recognition is live immediately,
        # with no manual model toggle and no server restart. The cache was
        # already rebuilt inside enroll_image(), so the very next frame matches.
        ctx.ai.ensure_enabled("face")
        return {"image_id": image_id, "path": rel, "enrollment": enroll}

    def _grab_frame(camera_id):
        cam = ctx.manager.get(camera_id) if camera_id else ctx.manager.get_active()
        frame, seq, _, _ = cam.buffer.latest()
        if frame is None:
            raise RTSPBackendError(
                "No frame available from the camera yet.",
                status_code=503, code="frame_unavailable", camera_id=cam.config.id,
            )
        return cam, frame

    # ---- read endpoints (light, stay async) -----------------------------

    @r.get("")
    async def list_employees():
        rows = db.query("SELECT * FROM employees ORDER BY full_name")
        out = []
        for row in rows:
            d = employee_dict(row)
            imgs = db.query(
                "SELECT id, path, created_at FROM employee_images WHERE employee_id=?",
                (row["id"],),
            )
            d["images"] = [dict(i) for i in imgs]
            d["embeddings"] = db.query_one(
                "SELECT COUNT(*) c FROM face_embeddings WHERE employee_id=?", (row["id"],)
            )["c"]
            out.append(d)
        return {"employees": out, "total": len(out)}

    @r.post("", status_code=201)
    async def create_employee(body: EmployeeCreate):
        now = time.time()
        if body.employee_code:
            existing = db.query_one(
                "SELECT id FROM employees WHERE employee_code=?", (body.employee_code,))
            if existing:
                raise RTSPBackendError(
                    f"Employee code '{body.employee_code}' already exists.",
                    status_code=409, code="duplicate_code")
        emp_id = db.insert(
            "INSERT INTO employees(employee_code, full_name, department, job_title, "
            "created_at, updated_at) VALUES(?,?,?,?,?,?)",
            (body.employee_code, body.full_name, body.department, body.job_title, now, now),
        )
        return employee_dict(_get_or_404(emp_id))

    @r.post("/register", status_code=201)
    async def register_from_captures(body: RegisterFromCaptures):
        """Full RTSP-camera enrolment in one call: create the employee, then
        enrol every captured frame. Rolls back completely if no frame yields a
        usable face, so recognition is guaranteed to work right after this
        returns and no faceless employee is ever left behind."""
        if ctx.ai.face_service is None:
            raise RTSPBackendError("Face service unavailable.", status_code=503,
                                   code="face_service_unavailable")
        if body.employee_code:
            existing = db.query_one(
                "SELECT id FROM employees WHERE employee_code=?", (body.employee_code,))
            if existing:
                raise RTSPBackendError(
                    f"Employee code '{body.employee_code}' already exists.",
                    status_code=409, code="duplicate_code")

        now = time.time()
        emp_id = db.insert(
            "INSERT INTO employees(employee_code, full_name, department, job_title, "
            "created_at, updated_at) VALUES(?,?,?,?,?,?)",
            (body.employee_code, body.full_name, body.department, body.job_title, now, now),
        )

        def _enroll_all():
            results, enrolled = [], 0
            for idx, data_url in enumerate(body.images):
                try:
                    img = decode_image(data_url)
                except RTSPBackendError as exc:
                    results.append({"ok": False, "reason": getattr(exc, "code", "bad_image")})
                    continue
                res = _enroll_blocking(emp_id, img)
                en = res.get("enrollment", {})
                if en.get("ok"):
                    enrolled += 1
                    if res.get("path") and enrolled == 1:
                        db.execute(
                            "UPDATE employees SET profile_image=?, updated_at=? WHERE id=?",
                            (res["path"], time.time(), emp_id))
                results.append(en)
            return results, enrolled

        results, enrolled = await asyncio.to_thread(_enroll_all)

        if enrolled == 0:
            # Nothing usable — roll the whole thing back.
            db.execute("DELETE FROM employees WHERE id=?", (emp_id,))
            reasons = [r.get("reason") for r in results if not r.get("ok")]
            raise RTSPBackendError(
                "No usable face in any captured frame; employee not created.",
                status_code=422, code="no_valid_face",
                details={"reasons": reasons, "captures": len(body.images)},
            )

        emp = employee_dict(_get_or_404(emp_id))
        emp["enrolled"] = enrolled
        emp["rejected"] = len(body.images) - enrolled
        emp["results"] = results
        emp["recognition_enabled"] = ctx.ai.is_enabled("face")
        return emp

    @r.get("/{emp_id}")
    async def get_employee(emp_id: int):
        d = employee_dict(_get_or_404(emp_id))
        imgs = db.query(
            "SELECT id, path, created_at FROM employee_images WHERE employee_id=?", (emp_id,)
        )
        d["images"] = [dict(i) for i in imgs]
        d["embeddings"] = db.query_one(
            "SELECT COUNT(*) c FROM face_embeddings WHERE employee_id=?", (emp_id,))["c"]
        return d

    @r.put("/{emp_id}")
    async def update_employee(emp_id: int, body: EmployeeUpdate):
        _get_or_404(emp_id)
        fields = body.model_dump(exclude_none=True)
        if fields:
            sets = ", ".join(f"{k}=?" for k in fields)
            db.execute(
                f"UPDATE employees SET {sets}, updated_at=? WHERE id=?",
                (*fields.values(), time.time(), emp_id),
            )
        return employee_dict(_get_or_404(emp_id))

    @r.delete("/{emp_id}")
    async def delete_employee(emp_id: int):
        _get_or_404(emp_id)
        db.execute("DELETE FROM employees WHERE id=?", (emp_id,))
        if ctx.ai.face_service is not None:
            await asyncio.to_thread(ctx.ai.face_service.reload_cache)
        return {"deleted": emp_id}

    # ---- capture / enrol (heavy -> threadpool) --------------------------

    @r.post("/validate")
    async def validate_capture(body: CaptureRequest):
        """Grab the latest buffered frame and report face suitability.

        Never opens a new RTSP connection; uses the frame already in memory.
        Returns the frame + primary-face crop as base64 so the UI can preview a
        candidate before the user decides to keep it.
        """
        if ctx.ai.face_service is None:
            raise RTSPBackendError("Face service unavailable.", status_code=503,
                                   code="face_service_unavailable")
        cam, frame = await asyncio.to_thread(_grab_frame, body.camera_id)

        def _work():
            verdict = ctx.ai.face_service.validate_frame(frame)
            crop_b64 = None
            if verdict.get("bbox"):
                x1, y1, x2, y2 = [int(v) for v in verdict["bbox"]]
                x1, y1 = max(0, x1), max(0, y1)
                crop = frame[y1:y2, x1:x2]
                if crop.size:
                    crop_b64 = encode_jpeg_b64(crop, quality=85)
            return verdict, encode_jpeg_b64(frame, quality=80), crop_b64

        verdict, frame_b64, crop_b64 = await asyncio.to_thread(_work)
        verdict["camera_id"] = cam.config.id
        verdict["image"] = frame_b64      # full frame (data URL) to enrol on save
        verdict["face_preview"] = crop_b64
        return verdict

    @r.post("/{emp_id}/images")
    async def add_image(emp_id: int, body: ImageUpload):
        _get_or_404(emp_id)
        img = await asyncio.to_thread(decode_image, body.image)
        res = await asyncio.to_thread(_enroll_blocking, emp_id, img)
        if body.make_profile and res.get("path"):
            db.execute("UPDATE employees SET profile_image=?, updated_at=? WHERE id=?",
                       (res["path"], time.time(), emp_id))
        return res

    @r.post("/{emp_id}/capture")
    async def capture_from_camera(emp_id: int, body: CaptureRequest):
        _get_or_404(emp_id)
        cam, frame = await asyncio.to_thread(_grab_frame, body.camera_id)
        res = await asyncio.to_thread(_enroll_blocking, emp_id, frame)
        if body.make_profile and res.get("path"):
            db.execute("UPDATE employees SET profile_image=?, updated_at=? WHERE id=?",
                       (res["path"], time.time(), emp_id))
        res["camera_id"] = cam.config.id
        return res

    @r.delete("/{emp_id}/images/{image_id}")
    async def delete_image(emp_id: int, image_id: int):
        row = db.query_one(
            "SELECT * FROM employee_images WHERE id=? AND employee_id=?", (image_id, emp_id)
        )
        if row is None:
            raise NotFound(f"Image {image_id} not found for employee {emp_id}.")
        db.execute("DELETE FROM employee_images WHERE id=?", (image_id,))
        db.execute("DELETE FROM face_embeddings WHERE image_id=?", (image_id,))
        if ctx.ai.face_service is not None:
            await asyncio.to_thread(ctx.ai.face_service.reload_cache)
        return {"deleted_image": image_id, "employee_id": emp_id}

    return r
