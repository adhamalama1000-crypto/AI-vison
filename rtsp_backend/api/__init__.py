"""REST API routers for the AI Vision Platform."""

from __future__ import annotations

from fastapi import APIRouter

from . import ai as ai_router
from . import analysis, attendance, datasets, datasheets, employees, images, inspection
from . import panels, reference, reference_panels, reports, training
from .util import Context


def build_all_routers(ctx: Context) -> list[APIRouter]:
    return [
        employees.build_router(ctx),
        ai_router.build_router(ctx),
        analysis.build_events_router(ctx),
        analysis.build_components_router(ctx),
        analysis.build_wires_router(ctx),
        analysis.build_stats_router(ctx),
        attendance.build_router(ctx),
        datasets.build_router(ctx),
        training.build_router(ctx),
        reference.build_router(ctx),
        reference_panels.build_router(ctx),
        datasheets.build_router(ctx),
        images.build_router(ctx),
        panels.build_router(ctx),
        inspection.build_router(ctx),
        reports.build_router(ctx),
    ]


__all__ = ["Context", "build_all_routers"]
