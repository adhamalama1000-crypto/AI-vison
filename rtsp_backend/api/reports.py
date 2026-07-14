"""Report listing endpoints (Parts 8, 10, 4)."""

from __future__ import annotations

import json

from fastapi import APIRouter, Query

from ..errors import RTSPBackendError


def _row(r) -> dict:
    d = dict(r)
    if d.get("summary"):
        try:
            d["summary"] = json.loads(d["summary"])
        except (TypeError, json.JSONDecodeError):
            pass
    return d


def build_router(ctx) -> APIRouter:
    r = APIRouter(prefix="/api/reports", tags=["reports"])
    db = ctx.db

    @r.get("")
    async def list_reports(
        limit: int = Query(100, ge=1, le=1000),
        kind: str | None = None,
    ):
        where, params = [], []
        if kind:
            where.append("kind=?"); params.append(kind)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        rows = db.query(
            f"SELECT * FROM reports {clause} ORDER BY created_at DESC LIMIT ?",
            (*params, limit))
        return {"reports": [_row(x) for x in rows], "total": len(rows)}

    @r.get("/summary")
    async def summary():
        rows = db.query("SELECT kind, COUNT(*) c FROM reports GROUP BY kind")
        return {"by_kind": {x["kind"]: x["c"] for x in rows}}

    @r.get("/{report_id}")
    async def get_report(report_id: int):
        row = db.query_one("SELECT * FROM reports WHERE id=?", (report_id,))
        if not row:
            raise RTSPBackendError("Report not found.", status_code=404,
                                   code="not_found")
        return _row(row)

    @r.delete("/{report_id}")
    async def delete_report(report_id: int):
        db.execute("DELETE FROM reports WHERE id=?", (report_id,))
        return {"deleted": report_id}

    return r
