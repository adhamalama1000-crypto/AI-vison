"""AI model-manager, settings, and metrics endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from ..ai.manager import TASKS
from ..errors import RTSPBackendError
from .ai_schemas import ModelEnable, ModelParams, ModelSelect, SettingSet


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
            return ai.select(task, body.backend_id, body.params)
        except KeyError as exc:
            raise RTSPBackendError(str(exc), status_code=400, code="bad_backend")

    @r.post("/models/{task}/enable")
    async def enable_model(task: str, body: ModelEnable):
        _check(task)
        return ai.set_enabled(task, body.enabled)

    @r.post("/models/{task}/params")
    async def set_params(task: str, body: ModelParams):
        _check(task)
        return ai.update_params(task, body.params)

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
