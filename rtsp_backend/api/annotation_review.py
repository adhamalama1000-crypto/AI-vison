"""
``/api/annotations/*`` — review and correct machine-generated annotations.

The human half of the auto-annotation loop. ``training.electrical.autolabel`` writes
YOLO pre-labels and a worst-first review queue; these endpoints serve that queue,
record per-box verdicts, and export corrected labels.

Scope is triage and reclassification — accept, reject, change the class — not drawing
new boxes. An image whose boxes need redrawing is flagged ``needs_redraw`` and
excluded from the export, so it goes to a real labelling tool instead of putting
known-bad labels into the training set. See :mod:`rtsp_backend.annotation_svc`.
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, Body, Query
from starlette.responses import FileResponse

from .. import annotation_svc as svc
from ..errors import RTSPBackendError


def _wrap(exc: Exception) -> RTSPBackendError:
    if isinstance(exc, KeyError):
        return RTSPBackendError(str(exc).strip("'\""), status_code=404,
                                code="not_found")
    if isinstance(exc, FileNotFoundError):
        return RTSPBackendError(str(exc), status_code=400, code="bad_batch")
    if isinstance(exc, ValueError):
        return RTSPBackendError(str(exc), status_code=400, code="invalid")
    return RTSPBackendError(str(exc), status_code=500, code="error")


def build_router(ctx) -> APIRouter:
    r = APIRouter(prefix="/api/annotations", tags=["annotations"])
    db = ctx.db
    svc.ensure_schema(db)

    @r.get("")
    async def list_batches():
        """Every registered review batch, with progress."""
        return {"batches": svc.list_batches(db)}

    @r.post("")
    async def register_batch(
        name: str = Body(..., embed=True),
        root: str = Body(..., embed=True),
        split: str = Body("train", embed=True),
    ):
        """Register an ``autolabel`` output directory for review."""
        try:
            return svc.register_batch(db, name, root, split)
        except Exception as exc:
            raise _wrap(exc)

    @r.get("/{batch}")
    async def batch_detail(batch: str):
        try:
            return svc.batch_detail(db, batch)
        except Exception as exc:
            raise _wrap(exc)

    @r.get("/{batch}/queue")
    async def review_queue(
        batch: str,
        state: Optional[str] = Query(
            "pending", description="Filter by image state; empty for all."),
        limit: int = Query(100, ge=1, le=1000),
    ):
        """Filenames to work, ordered worst-first (uncertain, then least confident)."""
        try:
            return svc.review_queue(db, batch, state or None, limit)
        except Exception as exc:
            raise _wrap(exc)

    @r.get("/{batch}/images/{filename}")
    async def get_image(batch: str, filename: str):
        """Serve the image bytes for review.

        Guarded against traversal: the resolved path must stay inside the batch's own
        image directory, and the filename must be one the batch actually contains.
        """
        try:
            row = svc._batch_row(db, batch)
        except Exception as exc:
            raise _wrap(exc)
        base = os.path.abspath(os.path.join(row["root"], "images", row["split"]))
        candidate = os.path.abspath(os.path.join(base, filename))
        if not (candidate == base or candidate.startswith(base + os.sep)) \
                or not os.path.isfile(candidate) \
                or os.path.basename(candidate) != filename:
            raise RTSPBackendError("Image not found in this batch.",
                                   status_code=404, code="not_found")
        return FileResponse(candidate)

    @r.get("/{batch}/items/{filename}")
    async def image_detail(batch: str, filename: str):
        """One image's boxes, with recorded verdicts merged in."""
        try:
            return svc.image_detail(db, batch, filename)
        except Exception as exc:
            raise _wrap(exc)

    @r.post("/{batch}/items/{filename}")
    async def record_review(
        batch: str, filename: str,
        boxes: list = Body(default_factory=list, embed=True),
        state: Optional[str] = Body(None, embed=True),
        note: Optional[str] = Body(None, embed=True),
        reviewer: Optional[str] = Body(None, embed=True),
    ):
        """Record box verdicts and optionally the image's state.

        ``boxes`` entries are ``{"index": int, "verdict": accepted|rejected|
        reclassified, "class_id": "<taxonomy id>"}``. A ``reclassified`` box without a
        valid taxonomy class id is rejected with 400 rather than written — a typo
        would otherwise become a label the trainer silently ignores.
        """
        try:
            out = svc.record_boxes(db, batch, filename, boxes, reviewer)
            if state:
                out.update(svc.set_image_state(db, batch, filename, state, note,
                                               reviewer))
            return out
        except Exception as exc:
            raise _wrap(exc)

    @r.post("/{batch}/export")
    async def export_batch(
        batch: str,
        dst_root: str = Body(..., embed=True),
        include_unreviewed: bool = Body(False, embed=True),
    ):
        """Write corrected YOLO labels for the reviewed images."""
        try:
            return svc.export_batch(db, batch, dst_root, include_unreviewed)
        except Exception as exc:
            raise _wrap(exc)

    return r


__all__ = ["build_router"]
