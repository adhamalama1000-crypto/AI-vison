"""AI model-manager, settings, and metrics endpoints."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter

from ..ai.manager import TASKS
from ..errors import RTSPBackendError
from .ai_schemas import FaceConfig, ModelEnable, ModelParams, ModelSelect, SettingSet


class BadTask(RTSPBackendError):
    status_code = 404
    code = "unknown_task"


def build_router(ctx) -> APIRouter:
    r = APIRouter(prefix="/api/ai", tags=["ai"])
    ai = ctx.ai

    def _check(task: str):
        if task not in TASKS:
            raise BadTask(f"Unknown AI task '{task}'. Valid: {', '.join(TASKS)}")

    @r.get("/status")
    async def ai_status():
        return ai.full_status()

    @r.get("/catalog")
    async def catalog():
        return ai.full_status()["catalog"]

    @r.get("/opencv")
    async def opencv_health():
        """Report whether the OpenCV install is healthy, and the fix command if
        conflicting opencv-python / opencv-python-headless wheels are present."""
        from ..opencv_guard import diagnose
        return diagnose()

    @r.get("/metrics")
    async def metrics():
        st = ai.full_status()
        return {
            "resources": st["resources"],
            "tasks": {t: st["tasks"][t]["metrics"] | {
                "enabled": st["tasks"][t]["enabled"],
                "backend": st["tasks"][t]["selected_backend"],
                "ready": st["tasks"][t]["backend"].get("ready", False),
                "state": st["tasks"][t]["state"],
                "reason": st["tasks"][t].get("reason"),
            } for t in TASKS},
        }

    @r.get("/models/{task}")
    async def task_status(task: str):
        _check(task)
        return ai.task_status(task)

    @r.post("/models/{task}/select")
    async def select_model(task: str, body: ModelSelect):
        _check(task)
        try:
            # model load (ONNX / InsightFace) is blocking CPU/IO — offload it so
            # it doesn't stall the event loop (and all live streams).
            return await asyncio.to_thread(ai.select, task, body.backend_id, body.params)
        except KeyError as exc:
            raise RTSPBackendError(str(exc), status_code=400, code="bad_backend")

    @r.post("/models/{task}/enable")
    async def enable_model(task: str, body: ModelEnable):
        _check(task)
        return await asyncio.to_thread(ai.set_enabled, task, body.enabled)

    @r.post("/models/{task}/params")
    async def set_params(task: str, body: ModelParams):
        _check(task)
        return await asyncio.to_thread(ai.update_params, task, body.params)

    # -- face recognition config + insight ---------------------------------

    @r.get("/face/config")
    async def face_config():
        """Live face-recognition tunables (threshold, margin, policy, quality
        floors) plus which real backend/index is active."""
        svc = ctx.ai.face_service
        cfg = svc.config() if svc is not None else {}
        st = ctx.ai.task_status("face")
        return {
            "config": cfg,
            "backend": st["selected_backend"],
            "backend_state": st["state"],
            "backend_detail": st.get("detail"),
            "backend_info": st.get("backend"),
            "params": st["backend"].get("params", {}),
        }

    @r.put("/face/config")
    async def set_face_config(body: FaceConfig):
        """Update recognition parameters (e.g. threshold from the frontend)."""
        params = {k: v for k, v in body.model_dump(exclude_none=True).items()}
        if not params:
            raise RTSPBackendError("No parameters provided.", status_code=400,
                                   code="empty_update")
        status = await asyncio.to_thread(ctx.ai.update_params, "face", params)
        svc = ctx.ai.face_service
        return {"ok": True, "config": svc.config() if svc else {},
                "task": status}

    @r.get("/face/messages")
    async def face_messages():
        """Quality reason-code -> human message map (used by the UI)."""
        from ..ai.face_service import QUALITY_MESSAGES
        return {"messages": QUALITY_MESSAGES}

    @r.get("/face/recognitions")
    async def face_recognitions(limit: int = 50):
        """Recent recognition history: recognised employees + unknown persons."""
        limit = max(1, min(int(limit), 500))
        rows = ctx.db.query(
            "SELECT id, type, camera_id, camera_name, label, confidence, "
            "employee_id, snapshot, created_at FROM events "
            "WHERE type IN ('face_recognized','unknown_person') "
            "ORDER BY created_at DESC LIMIT ?", (limit,))
        return {"recognitions": [dict(x) for x in rows], "total": len(rows)}

    # -- generic settings key/value ---------------------------------------

    @r.get("/settings")
    async def get_settings():
        return ctx.db.all_settings()

    @r.get("/settings/{key}")
    async def get_setting(key: str):
        return {"key": key, "value": ctx.db.get_setting(key)}

    @r.put("/settings/{key}")
    async def set_setting(key: str, body: SettingSet):
        ctx.db.set_setting(key, body.value)
        return {"key": key, "value": body.value}

    return r
