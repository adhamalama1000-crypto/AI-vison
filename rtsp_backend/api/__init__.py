"""REST API routers for the AI Vision Platform."""

from __future__ import annotations

from fastapi import APIRouter

from . import ai as ai_router
from . import analysis, employees
from .util import Context


def build_all_routers(ctx: Context) -> list[APIRouter]:
    return [
        employees.build_router(ctx),
        ai_router.build_router(ctx),
        analysis.build_events_router(ctx),
        analysis.build_components_router(ctx),
        analysis.build_wires_router(ctx),
        analysis.build_stats_router(ctx),
    ]


__all__ = ["Context", "build_all_routers"]
