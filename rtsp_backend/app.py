"""
The FastAPI application: REST status/control endpoints, JPEG snapshots, an MJPEG
stream, a WebSocket live-events feed, and the service lifecycle that starts and
stops cameras and the telemetry heartbeat.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

import os

from starlette.responses import FileResponse

from .ai import AIModelManager, AIPipeline
from .api import Context, build_all_routers
from .config import Settings, load_settings
from .db import Database
from .errors import RTSPBackendError
from .events import EventBus
from .manager import CameraManager
from .schemas import ActiveCameraRequest, CameraCreateRequest, CameraUpdateRequest

log = logging.getLogger("rtsp_backend")

MJPEG_BOUNDARY = "frame"


async def _ai_worker(manager: CameraManager, pipeline, ai_manager, interval: float = 0.25) -> None:
    """
    Continuous inference loop.

    Runs the enabled AI tasks on each camera's latest buffered frame so that
    recognition, attendance logging, and detections happen automatically even
    when no client is watching the stream. CPU-bound inference is offloaded to
    the threadpool, and the pipeline throttles per camera, so this never blocks
    the event loop or opens a second camera connection.
    """
    interval = interval if interval and interval > 0 else 0.25
    while True:
        await asyncio.sleep(interval)
        # skip all work cheaply if nothing is enabled
        if not any(ai_manager.is_enabled(t) for t in ai_manager._tasks):
            continue
        for camera in manager.list():
            frame, seq, _, _ = camera.buffer.latest()
            if frame is None:
                continue
            try:
                await asyncio.to_thread(
                    pipeline.process, camera.config.id, camera.config.name,
                    frame, False, False,
                )
            except Exception:
                # never let one bad frame kill the worker
                continue


async def _heartbeat(manager: CameraManager, bus: EventBus, interval: float) -> None:
    """Periodically broadcast per-camera telemetry as ``stats`` events."""
    interval = interval if interval and interval > 0 else 5.0
    while True:
        await asyncio.sleep(interval)
        for status in (c.status() for c in manager.list()):
            bus.publish(
                {
                    "type": "stats",
                    "camera_id": status["id"],
                    "timestamp": time.time(),
                    "state": status["state"],
                    "healthy": status["healthy"],
                    "fps": status["fps"],
                    "latency": status["latency"],
                    "frame_age_ms": status["frame_age_ms"],
                    "statistics": status["statistics"],
                }
            )


async def _mjpeg_stream(
    manager: CameraManager,
    camera_id: str,
    quality: Optional[int],
    fps_cap: Optional[float],
) -> AsyncIterator[bytes]:
    camera = manager.get(camera_id)
    last_seq = -1
    min_interval = (1.0 / fps_cap) if fps_cap else 0.0
    last_sent = 0.0
    while True:
        jpg, last_seq = await asyncio.to_thread(
            camera.next_jpeg, last_seq, 5.0, quality
        )
        if jpg is None:
            # No new frame within the wait window (e.g. reconnecting). Keep the
            # connection open and try again rather than dropping the client.
            await asyncio.sleep(0.05)
            continue
        if min_interval:
            wait = min_interval - (time.monotonic() - last_sent)
            if wait > 0:
                await asyncio.sleep(wait)
            last_sent = time.monotonic()
        header = (
            f"--{MJPEG_BOUNDARY}\r\n"
            f"Content-Type: image/jpeg\r\n"
            f"Content-Length: {len(jpg)}\r\n\r\n"
        ).encode("ascii")
        yield header + jpg + b"\r\n"


def build_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or load_settings()
    bus = EventBus()
    manager = CameraManager(event_sink=bus.publish_threadsafe)

    # Persistence + AI subsystem.
    db = Database(settings.db_path)
    ai_manager = AIModelManager(db, models_dir=settings.models_dir)
    pipeline = AIPipeline(
        db, ai_manager, event_sink=bus.publish_threadsafe,
        data_dir=settings.data_dir, min_interval=settings.ai_min_interval,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop = asyncio.get_running_loop()
        bus.bind_loop(loop)
        manager.load(
            settings.cameras, active=settings.active_camera, autostart=True
        )
        heartbeat = asyncio.create_task(
            _heartbeat(manager, bus, settings.stats_interval)
        )
        ai_worker = asyncio.create_task(
            _ai_worker(manager, pipeline, ai_manager, settings.ai_min_interval)
        )
        try:
            yield
        finally:
            heartbeat.cancel()
            ai_worker.cancel()
            for task in (heartbeat, ai_worker):
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            manager.stop_all()
            db.close()

    app = FastAPI(
        title="AI Vision Platform",
        version="3.1.5",
        description=(
            "RTSP-only camera backend with a pluggable AI subsystem (face "
            "recognition, object/component detection, wire analysis), employee "
            "management, a persistent database, and a live dashboard. Sources "
            "always come from configuration; there is no fallback to USB or local files."
        ),
        lifespan=lifespan,
    )
    app.state.manager = manager
    app.state.bus = bus
    app.state.settings = settings
    app.state.db = db
    app.state.ai = ai_manager
    app.state.pipeline = pipeline

    @app.exception_handler(RTSPBackendError)
    async def _handle_backend_error(request: Request, exc: RTSPBackendError):
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(request: Request, exc: RequestValidationError):
        # Bad request bodies / query params: render them in the same error
        # envelope as everything else instead of FastAPI's default {"detail": ...}.
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Request validation failed.",
                    "details": {"errors": jsonable_encoder(exc.errors())},
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_error(request: Request, exc: StarletteHTTPException):
        # Unknown routes (404), wrong method (405), etc. — keep the envelope
        # uniform so clients only ever parse one error shape.
        code = {
            400: "bad_request",
            404: "not_found",
            405: "method_not_allowed",
        }.get(exc.status_code, "http_error")
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": code, "message": str(exc.detail)}},
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected_error(request: Request, exc: Exception):
        # Catch-all: an unexpected bug must still produce the structured JSON
        # envelope (never Starlette's bare "Internal Server Error" text), and
        # the full traceback goes to the server log for debugging.
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": f"{type(exc).__name__}: {exc}",
                    "details": {"path": request.url.path},
                }
            },
        )

    # -- health / discovery ------------------------------------------------

    @app.get("/health", tags=["status"])
    async def health():
        overview = manager.overview()
        return {
            "status": "ok",
            "service": "rtsp-camera-backend",
            "rtsp_only": True,
            "cameras_total": overview["cameras_total"],
            "cameras_connected": overview["cameras_connected"],
            "active_camera": overview["active_camera"],
            "event_subscribers": bus.subscriber_count,
            "timestamp": time.time(),
        }

    # -- camera collection -------------------------------------------------

    @app.get("/cameras", tags=["cameras"])
    async def list_cameras():
        overview = manager.overview()
        return {
            "active_camera": overview["active_camera"],
            "cameras": overview["cameras"],
        }

    @app.post("/cameras", status_code=201, tags=["cameras"])
    async def create_camera(body: CameraCreateRequest):
        camera = manager.add_camera(body.to_config(), start=True)
        return camera.status()

    @app.get("/cameras/{camera_id}", tags=["cameras"])
    async def get_camera(camera_id: str):
        return manager.get(camera_id).status()

    @app.get("/cameras/{camera_id}/status", tags=["cameras"])
    async def camera_status(camera_id: str):
        # Dedicated status endpoint (health, latency, drops, connection stats).
        return manager.get(camera_id).status()

    @app.get("/cameras/{camera_id}/diagnose", tags=["cameras"])
    async def camera_diagnose(
        camera_id: str,
        timeout: float = Query(10.0, gt=0, le=60),
    ):
        """
        On-demand connection diagnostic: probes the camera's RTSP URL over TCP
        and UDP, capturing the real FFmpeg error output, plus a raw TCP
        reachability check and a plain-language verdict. Intended for
        troubleshooting, not the hot path (attempts run serially).
        """
        from .diagnostics import probe

        camera = manager.get(camera_id)
        return await asyncio.to_thread(probe, camera.config.url, timeout)

    @app.put("/cameras/{camera_id}", tags=["cameras"])
    async def update_camera(camera_id: str, body: CameraUpdateRequest):
        existing = manager.get(camera_id)
        camera = manager.update_camera(camera_id, body.to_config(existing.config))
        return camera.status()

    @app.delete("/cameras/{camera_id}", tags=["cameras"])
    async def delete_camera(camera_id: str):
        manager.remove_camera(camera_id)
        return {"deleted": camera_id}

    # -- active camera -----------------------------------------------------

    @app.get("/active-camera", tags=["cameras"])
    async def get_active_camera():
        camera = manager.get_active()
        return {"active_camera": manager.active_id, "camera": camera.status()}

    @app.post("/active-camera", tags=["cameras"])
    async def set_active_camera(body: ActiveCameraRequest):
        manager.set_active(body.id)
        return {"active_camera": manager.active_id}

    # -- snapshots ---------------------------------------------------------

    async def _snapshot_response(camera, quality: Optional[int]) -> Response:
        jpg, ts = await asyncio.to_thread(camera.snapshot_jpeg, quality)
        return Response(
            content=jpg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store", "X-Timestamp": str(ts)},
        )

    @app.get("/cameras/{camera_id}/snapshot", tags=["media"])
    async def snapshot(
        camera_id: str,
        quality: Optional[int] = Query(None, ge=1, le=100),
    ):
        return await _snapshot_response(manager.get(camera_id), quality)

    @app.get("/snapshot", tags=["media"])
    async def active_snapshot(quality: Optional[int] = Query(None, ge=1, le=100)):
        return await _snapshot_response(manager.get_active(), quality)

    # -- MJPEG stream ------------------------------------------------------

    def _stream_response(camera_id: str, quality, fps) -> StreamingResponse:
        return StreamingResponse(
            _mjpeg_stream(manager, camera_id, quality, fps),
            media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
            headers={"Cache-Control": "no-store", "Connection": "close"},
        )

    @app.get("/cameras/{camera_id}/stream", tags=["media"])
    async def stream(
        camera_id: str,
        quality: Optional[int] = Query(None, ge=1, le=100),
        fps: Optional[float] = Query(None, gt=0, le=60),
    ):
        manager.get(camera_id)  # 404 before opening the stream
        return _stream_response(camera_id, quality, fps)

    @app.get("/stream", tags=["media"])
    async def active_stream(
        quality: Optional[int] = Query(None, ge=1, le=100),
        fps: Optional[float] = Query(None, gt=0, le=60),
    ):
        camera = manager.get_active()
        return _stream_response(camera.config.id, quality, fps)

    # -- WebSocket live events --------------------------------------------

    @app.websocket("/ws/events")
    async def ws_events(websocket: WebSocket):
        await websocket.accept()
        queue = bus.subscribe()
        try:
            await websocket.send_json(
                {
                    "type": "hello",
                    "timestamp": time.time(),
                    "active_camera": manager.active_id,
                    "cameras": [c.status() for c in manager.list()],
                }
            )
            while True:
                event = await queue.get()
                await websocket.send_json(event)
        except WebSocketDisconnect:
            pass
        except Exception:
            log.exception("websocket event stream error")
        finally:
            bus.unsubscribe(queue)

    # -- AI-annotated media (overlays drawn on the live frame) -------------

    async def _ai_mjpeg(camera_id: str, quality: Optional[int], fps_cap: Optional[float]):
        camera = manager.get(camera_id)
        last_seq = -1
        min_interval = (1.0 / fps_cap) if fps_cap else 0.0
        last_sent = 0.0
        while True:
            frame, seq, _, _ = await asyncio.to_thread(
                camera.buffer.wait, last_seq, 5.0
            )
            if frame is None or seq == last_seq:
                await asyncio.sleep(0.03)
                continue
            last_seq = seq
            # Draw the most recent cached detections onto this fresh frame.
            # Inference runs on the background AI worker, never inline here, so
            # the AI video has the same low latency as the raw stream.
            jpg = await asyncio.to_thread(
                pipeline.annotated_jpeg_fast, camera.config.id, camera.config.name,
                frame, quality or camera.config.jpeg_quality,
            )
            if not jpg:
                continue
            if min_interval:
                wait = min_interval - (time.monotonic() - last_sent)
                if wait > 0:
                    await asyncio.sleep(wait)
                last_sent = time.monotonic()
            header = (
                f"--{MJPEG_BOUNDARY}\r\n"
                f"Content-Type: image/jpeg\r\n"
                f"Content-Length: {len(jpg)}\r\n\r\n"
            ).encode("ascii")
            yield header + jpg + b"\r\n"

    @app.get("/api/cameras/{camera_id}/ai-stream", tags=["ai-media"])
    async def ai_stream(
        camera_id: str,
        quality: Optional[int] = Query(None, ge=1, le=100),
        fps: Optional[float] = Query(None, gt=0, le=60),
    ):
        manager.get(camera_id)
        return StreamingResponse(
            _ai_mjpeg(camera_id, quality, fps),
            media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
            headers={"Cache-Control": "no-store", "Connection": "close"},
        )

    @app.get("/api/metrics", tags=["ai-media"])
    async def metrics():
        """
        Consolidated real-time performance metrics for the dashboard:
        per-camera capture FPS / stream latency (frame age) / read + encode
        time, per-task AI FPS + inference time, and host resource usage.
        """
        cams = []
        for c in manager.list():
            s = c.status()
            cams.append({
                "id": s["id"],
                "name": s["name"],
                "state": s["state"],
                "healthy": s["healthy"],
                "camera_fps": s["fps"],
                "stream_latency_ms": s["frame_age_ms"],
                "read_ms": s["latency"],
                "encode_ms": s.get("encode"),
                "frames_captured": s["statistics"]["frames_captured"],
                "frames_dropped": s["statistics"]["frames_dropped"],
                "reconnects": s["statistics"]["reconnect_count"],
            })
        ai_tasks = {}
        for t, st in ai_manager._tasks.items():
            ai_tasks[t] = {
                "enabled": st.enabled,
                "ai_fps": st.fps(),
                "inference_ms": st.avg_infer_ms(),
                "state": ai_manager._state(st),
            }
        return {
            "timestamp": time.time(),
            "cameras": cams,
            "ai": ai_tasks,
            "resources": ai_manager.resource_metrics(),
        }

    @app.get("/api/cameras/{camera_id}/ai-snapshot", tags=["ai-media"])
    async def ai_snapshot(camera_id: str, quality: Optional[int] = Query(None, ge=1, le=100)):
        camera = manager.get(camera_id)
        frame, *_ = camera.buffer.latest()
        if frame is None:
            raise RTSPBackendError(
                f"Camera '{camera_id}' has no frame yet.",
                status_code=503, code="frame_unavailable", camera_id=camera_id,
            )
        jpg = await asyncio.to_thread(
            pipeline.annotated_jpeg, camera.config.id, camera.config.name,
            frame, quality or 80,
        )
        return Response(content=jpg, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/cameras/{camera_id}/analyze", tags=["ai-media"])
    async def analyze_frame(camera_id: str):
        """Run all enabled AI on the current frame and return structured results (no image)."""
        camera = manager.get(camera_id)
        frame, *_ = camera.buffer.latest()
        if frame is None:
            raise RTSPBackendError(
                f"Camera '{camera_id}' has no frame yet.",
                status_code=503, code="frame_unavailable", camera_id=camera_id,
            )
        res = await asyncio.to_thread(
            pipeline.process, camera.config.id, camera.config.name, frame, False, True
        )
        return {k: v for k, v in res.items() if not k.startswith("_")}

    # -- stored media (snapshots, employee images) ------------------------

    @app.get("/api/media/{path:path}", tags=["media"])
    async def media(path: str):
        # Prevent path traversal; only serve from within the data dir.
        base = os.path.abspath(settings.data_dir)
        full = os.path.abspath(os.path.join(base, path))
        if not full.startswith(base + os.sep) or not os.path.isfile(full):
            raise RTSPBackendError("Media not found.", status_code=404, code="not_found")
        return FileResponse(full)

    # -- REST API routers --------------------------------------------------

    ctx = Context(db=db, manager=manager, ai=ai_manager, pipeline=pipeline,
                  bus=bus, data_dir=settings.data_dir)
    for router in build_all_routers(ctx):
        app.include_router(router)

    # -- frontend (single-page app, served last so /api and /docs win) -----
    #
    # The production build (Vite + React) lives in ``web/`` with its assets
    # under ``web/assets`` and a base path of ``/app/``. We serve real files
    # when they exist and fall back to ``index.html`` for any other ``/app/*``
    # path so client-side routes (e.g. /app/employees) survive a hard refresh.

    web_dir = os.path.join(os.path.dirname(__file__), "web")
    index_html = os.path.join(web_dir, "index.html")
    if os.path.isfile(index_html):
        from starlette.responses import RedirectResponse

        @app.get("/", include_in_schema=False)
        async def _root():
            return RedirectResponse(url="/app/")

        @app.get("/app", include_in_schema=False)
        @app.get("/app/{path:path}", include_in_schema=False)
        async def _spa(path: str = ""):
            # Resolve to a real file inside web_dir, guarding against traversal.
            base = os.path.abspath(web_dir)
            candidate = os.path.abspath(os.path.join(base, path))
            if (
                path
                and (candidate == base or candidate.startswith(base + os.sep))
                and os.path.isfile(candidate)
            ):
                return FileResponse(candidate)
            # SPA fallback: unknown route -> let the client router handle it.
            return FileResponse(index_html)

    return app
