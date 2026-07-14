"""Training job control + metrics endpoints (Parts 3, 4, 5)."""

from __future__ import annotations

import json

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from ..errors import RTSPBackendError
from ..training_svc import (CLASSIFIER_MODELS, DETECTION_MODELS, _ULTRA_WEIGHTS,
                            TrainingManager)


class StartBody(BaseModel):
    name: str = Field(default="training run", max_length=200)
    dataset_id: int | None = None
    task: str = Field(default="classification")   # classification|detection
    models: list[str] = Field(default_factory=lambda: ["mlp", "deep_mlp"])
    config: dict = Field(default_factory=dict)


def _job(row) -> dict:
    d = dict(row)
    for k in ("models", "config", "metrics", "history", "comparison", "artifacts"):
        if d.get(k):
            try:
                d[k] = json.loads(d[k])
            except (TypeError, json.JSONDecodeError):
                pass
    # never leak the in-process classifier handle
    if isinstance(d.get("comparison"), list):
        for c in d["comparison"]:
            c.pop("_clf", None)
    return d


def build_router(ctx) -> APIRouter:
    r = APIRouter(prefix="/api/training", tags=["training"])
    db = ctx.db
    sink = getattr(ctx.bus, "publish_threadsafe", None) or (lambda ev: None)
    mgr = TrainingManager(db, data_dir=ctx.data_dir,
                          models_dir=getattr(ctx, "models_dir", "models"),
                          event_sink=sink)
    ctx.training = mgr  # expose for other components / tests

    @r.get("/catalog")
    async def catalog():
        try:
            import ultralytics  # noqa: F401
            have_ultra = True
        except Exception:
            have_ultra = False
        return {
            "classification_models": list(CLASSIFIER_MODELS.keys()),
            "detection_models": DETECTION_MODELS,
            # detection archs that this build can actually train (need
            # ultralytics installed + a detection dataset). The rest are listed
            # but will be reported as "skipped" if selected.
            "detection_models_trainable": (
                [m for m in DETECTION_MODELS if m in _ULTRA_WEIGHTS] if have_ultra else []),
            "ultralytics_available": have_ultra,
            "tunable": ["learning_rate", "batch_size", "image_size", "epochs",
                        "weight_decay", "augment", "hpo", "hpo_trials",
                        "early_stopping_patience"],
        }

    @r.get("")
    async def list_jobs(limit: int = Query(50, ge=1, le=500)):
        rows = db.query(
            "SELECT id,name,dataset_id,task,status,progress,best_model,"
            "created_at,updated_at FROM training_jobs ORDER BY created_at DESC LIMIT ?",
            (limit,))
        return {"jobs": [dict(x) for x in rows], "total": len(rows)}

    @r.post("", status_code=201)
    async def start(body: StartBody):
        if body.dataset_id is not None:
            ds = db.query_one("SELECT id FROM datasets WHERE id=?", (body.dataset_id,))
            if not ds:
                raise RTSPBackendError("Dataset not found.", status_code=404,
                                       code="not_found")
        job_id = mgr.start(body.name, body.dataset_id, body.task,
                           body.models, body.config)
        return {"job_id": job_id, "status": "queued"}

    @r.get("/{job_id}")
    async def get_job(job_id: int):
        row = db.query_one("SELECT * FROM training_jobs WHERE id=?", (job_id,))
        if not row:
            raise RTSPBackendError("Job not found.", status_code=404, code="not_found")
        return _job(row)

    @r.get("/{job_id}/comparison")
    async def comparison(job_id: int):
        row = db.query_one("SELECT comparison,best_model FROM training_jobs WHERE id=?",
                           (job_id,))
        if not row:
            raise RTSPBackendError("Job not found.", status_code=404, code="not_found")
        comp = json.loads(row["comparison"]) if row["comparison"] else []
        for c in comp:
            c.pop("_clf", None)
        return {"comparison": comp, "best_model": row["best_model"]}

    @r.post("/{job_id}/pause")
    async def pause(job_id: int):
        return {"ok": mgr.pause(job_id)}

    @r.post("/{job_id}/resume")
    async def resume(job_id: int):
        return {"ok": mgr.resume(job_id)}

    @r.post("/{job_id}/stop")
    async def stop(job_id: int):
        return {"ok": mgr.stop(job_id)}

    return r
