"""Attendance endpoints (Part 1).

Attendance rows are created by the AI pipeline the first time a known employee
is recognised within the configured timeout window — never fabricated. These
endpoints expose the log, a daily roll-up, and the configurable timeout.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field


class TimeoutBody(BaseModel):
    seconds: float = Field(ge=0, le=604800)  # 0 .. one week


def build_router(ctx) -> APIRouter:
    r = APIRouter(prefix="/api/attendance", tags=["attendance"])
    db = ctx.db

    @r.get("")
    async def list_attendance(
        limit: int = Query(200, ge=1, le=2000),
        offset: int = Query(0, ge=0),
        employee_id: int | None = None,
        day: str | None = None,
    ):
        where, params = [], []
        if employee_id is not None:
            where.append("employee_id=?"); params.append(employee_id)
        if day:
            where.append("day=?"); params.append(day)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        rows = db.query(
            f"SELECT * FROM attendance {clause} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (*params, limit, offset),
        )
        total = db.query_one(
            f"SELECT COUNT(*) c FROM attendance {clause}", tuple(params))["c"]
        return {"attendance": [dict(x) for x in rows], "total": total}

    @r.get("/today")
    async def today():
        day = time.strftime("%Y-%m-%d", time.localtime())
        rows = db.query(
            "SELECT * FROM attendance WHERE day=? ORDER BY created_at DESC", (day,))
        present = db.query(
            "SELECT DISTINCT employee_id FROM attendance WHERE day=?", (day,))
        total_emp = db.query_one("SELECT COUNT(*) c FROM employees")["c"]
        return {
            "day": day,
            "records": [dict(x) for x in rows],
            "present": len(present),
            "employees_total": total_emp,
        }

    @r.get("/summary")
    async def summary(days: int = Query(7, ge=1, le=90)):
        rows = db.query(
            "SELECT day, COUNT(DISTINCT employee_id) present, COUNT(*) records "
            "FROM attendance GROUP BY day ORDER BY day DESC LIMIT ?", (days,))
        return {"summary": [dict(x) for x in rows]}

    @r.get("/config")
    async def get_config():
        return {"timeout_seconds": float(db.get_setting("attendance_timeout_s", 28800.0))}

    @r.put("/config")
    async def set_config(body: TimeoutBody):
        db.set_setting("attendance_timeout_s", body.seconds)
        # apply live to the running pipeline
        try:
            ctx.pipeline.set_attendance_timeout(body.seconds)
        except Exception:
            pass
        return {"timeout_seconds": body.seconds}

    @r.delete("")
    async def clear():
        db.execute("DELETE FROM attendance")
        return {"cleared": True}

    return r
