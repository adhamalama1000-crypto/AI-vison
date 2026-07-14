"""Events, detections, components, wire-analysis, and dashboard-stats endpoints."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Query


def _row_to_event(row) -> dict:
    d = dict(row)
    if d.get("payload"):
        try:
            d["payload"] = json.loads(d["payload"])
        except (TypeError, json.JSONDecodeError):
            pass
    return d


def build_events_router(ctx) -> APIRouter:
    r = APIRouter(prefix="/api/events", tags=["events"])
    db = ctx.db

    @r.get("")
    async def list_events(
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        type: str | None = None,
        camera_id: str | None = None,
    ):
        where, params = [], []
        if type:
            where.append("type=?"); params.append(type)
        if camera_id:
            where.append("camera_id=?"); params.append(camera_id)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        rows = db.query(
            f"SELECT * FROM events {clause} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (*params, limit, offset),
        )
        total = db.query_one(f"SELECT COUNT(*) c FROM events {clause}", tuple(params))["c"]
        return {"events": [_row_to_event(x) for x in rows], "total": total}

    @r.get("/types")
    async def event_types():
        rows = db.query("SELECT type, COUNT(*) c FROM events GROUP BY type ORDER BY c DESC")
        return {"types": [dict(x) for x in rows]}

    @r.delete("")
    async def clear_events():
        db.execute("DELETE FROM events")
        return {"cleared": True}

    return r


def build_components_router(ctx) -> APIRouter:
    r = APIRouter(prefix="/api/components", tags=["components"])
    db = ctx.db

    @r.get("")
    async def list_components(limit: int = Query(200, ge=1, le=2000),
                              camera_id: str | None = None):
        where, params = [], []
        if camera_id:
            where.append("camera_id=?"); params.append(camera_id)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        rows = db.query(
            f"SELECT * FROM components {clause} ORDER BY created_at DESC LIMIT ?",
            (*params, limit),
        )
        return {"components": [dict(x) for x in rows], "total": len(rows)}

    @r.get("/classes")
    async def component_classes():
        from ..ai.components import ELECTRICAL_CLASSES
        return {"classes": ELECTRICAL_CLASSES}

    @r.get("/summary")
    async def component_summary():
        rows = db.query(
            "SELECT comp_type, COUNT(*) c, AVG(confidence) avg_conf "
            "FROM components GROUP BY comp_type ORDER BY c DESC"
        )
        return {"summary": [dict(x) for x in rows]}

    return r


def build_wires_router(ctx) -> APIRouter:
    r = APIRouter(prefix="/api/wires", tags=["wires"])
    db = ctx.db

    @r.get("")
    async def list_wires(limit: int = Query(500, ge=1, le=5000),
                         camera_id: str | None = None):
        where, params = [], []
        if camera_id:
            where.append("camera_id=?"); params.append(camera_id)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        rows = db.query(
            f"SELECT * FROM wires {clause} ORDER BY created_at DESC LIMIT ?",
            (*params, limit),
        )
        return {"wires": [dict(x) for x in rows], "total": len(rows)}

    @r.get("/topology")
    async def topology(camera_id: str | None = None):
        """
        Live wire+component topology for a camera. Runs the enabled component and
        wire backends against the current frame and returns nodes (components)
        and edges (wires). With no trained component model the node list may be
        empty; the wire baseline still returns detected line segments.
        """
        cam = ctx.manager.get(camera_id) if camera_id else ctx.manager.get_active()
        frame, *_ = cam.buffer.latest()
        if frame is None:
            return {"camera_id": cam.config.id, "nodes": [], "edges": [],
                    "note": "no frame available yet"}
        res = await asyncio.to_thread(ctx.pipeline.process, cam.config.id,
                                      cam.config.name, frame, False, True)
        nodes = [{"id": i, "label": c["label"], "bbox": c["bbox"],
                  "position": c.get("position")} for i, c in enumerate(res["components"])]
        edges = [{"wire_uid": w["wire_uid"], "start": w["start"], "end": w["end"],
                  "color": w["color"], "status": w["status"],
                  "from": w["from_component"], "to": w["to_component"]}
                 for w in res["wires"]]
        return {"camera_id": cam.config.id, "nodes": nodes, "edges": edges,
                "node_count": len(nodes), "edge_count": len(edges)}

    return r


def build_stats_router(ctx) -> APIRouter:
    r = APIRouter(prefix="/api/stats", tags=["stats"])
    db = ctx.db

    @r.get("/dashboard")
    async def dashboard():
        def count(sql, params=()):
            return db.query_one(sql, params)["c"]

        overview = ctx.manager.overview()
        cams = overview["cameras"]
        total_fps = round(sum(c.get("fps", 0) or 0 for c in cams), 2)
        latencies = [c["latency"]["avg_ms"] for c in cams
                     if c.get("latency") and c["latency"].get("avg_ms") is not None]
        avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else None

        ai_status = ctx.ai.full_status()
        return {
            "employees": {
                "total": count("SELECT COUNT(*) c FROM employees"),
                "enrolled_faces": count("SELECT COUNT(*) c FROM face_embeddings"),
            },
            "recognition": {
                "recognized_events": count(
                    "SELECT COUNT(*) c FROM events WHERE type='face_recognized'"),
                "unknown_events": count(
                    "SELECT COUNT(*) c FROM events WHERE type='unknown_person'"),
            },
            "electrical": {
                "components_detected": count("SELECT COUNT(*) c FROM components"),
                "wiring_errors": count(
                    "SELECT COUNT(*) c FROM events WHERE type='wiring_error'"),
            },
            "cameras": {
                "total": overview["cameras_total"],
                "connected": overview["cameras_connected"],
                "active": overview["active_camera"],
                "total_fps": total_fps,
                "avg_latency_ms": avg_latency,
            },
            "resources": ai_status["resources"],
            "ai_tasks": {t: {
                "enabled": ai_status["tasks"][t]["enabled"],
                "backend": ai_status["tasks"][t]["selected_backend"],
                "ready": ai_status["tasks"][t]["backend"].get("ready", False),
            } for t in ai_status["tasks"]},
            "events_total": count("SELECT COUNT(*) c FROM events"),
        }

    return r
