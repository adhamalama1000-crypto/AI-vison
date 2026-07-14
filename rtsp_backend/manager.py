"""
Manages the set of live RTSP cameras: lifecycle, lookup, the notion of an
"active" camera, and runtime reconfiguration (adding cameras, removing them, and
switching a camera's RTSP source without restarting the service).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional

from .camera import RTSPCamera, redact_url
from .config import CameraConfig
from .errors import (
    CameraNotFoundError,
    DuplicateCameraError,
    NoActiveCameraError,
    validate_rtsp_url,
)

log = logging.getLogger("rtsp_backend.manager")


class CameraManager:
    def __init__(self, event_sink: Optional[Callable[[dict], None]] = None) -> None:
        self._cameras: dict[str, RTSPCamera] = {}
        self._lock = threading.RLock()
        self._active_id: Optional[str] = None
        self._event_sink = event_sink or (lambda ev: None)

    def _emit(self, event_type: str, **payload) -> None:
        event = {"type": event_type, "timestamp": time.time()}
        event.update(payload)
        try:
            self._event_sink(event)
        except Exception:
            log.exception("event sink failed")

    # -- bulk load ---------------------------------------------------------

    def load(
        self,
        configs: list[CameraConfig],
        active: Optional[str] = None,
        autostart: bool = True,
    ) -> None:
        for cfg in configs:
            self.add_camera(cfg, start=autostart, emit=False)
        with self._lock:
            if active and active in self._cameras:
                self._active_id = active
            elif self._active_id is None and self._cameras:
                self._active_id = next(iter(self._cameras))

    # -- CRUD --------------------------------------------------------------

    def add_camera(
        self, config: CameraConfig, start: bool = True, emit: bool = True
    ) -> RTSPCamera:
        # Authoritative RTSP-only gate.
        validate_rtsp_url(config.url, camera_id=config.id)
        with self._lock:
            if config.id in self._cameras:
                raise DuplicateCameraError(
                    f"Camera '{config.id}' already exists.", camera_id=config.id
                )
            camera = RTSPCamera(config, on_event=self._event_sink)
            self._cameras[config.id] = camera
            if self._active_id is None:
                self._active_id = config.id
        if start:
            camera.start()
        if emit:
            self._emit("camera_added", camera_id=config.id, name=config.name)
        return camera

    def get(self, camera_id: str) -> RTSPCamera:
        with self._lock:
            camera = self._cameras.get(camera_id)
        if camera is None:
            raise CameraNotFoundError(
                f"Camera '{camera_id}' not found.", camera_id=camera_id
            )
        return camera

    def list(self) -> list[RTSPCamera]:
        with self._lock:
            return list(self._cameras.values())

    def remove_camera(self, camera_id: str) -> None:
        camera = self.get(camera_id)
        camera.stop()
        with self._lock:
            self._cameras.pop(camera_id, None)
            if self._active_id == camera_id:
                self._active_id = next(iter(self._cameras), None)
        self._emit("camera_removed", camera_id=camera_id)

    def update_camera(self, camera_id: str, new_config: CameraConfig) -> RTSPCamera:
        """
        Switch a camera's configuration at runtime (e.g. point it at a different
        RTSP URL). The existing capture is stopped and a fresh one is started;
        the camera id is preserved.
        """
        validate_rtsp_url(new_config.url, camera_id=camera_id)
        existing = self.get(camera_id)
        existing.stop()
        config = new_config.model_copy(update={"id": camera_id})
        with self._lock:
            camera = RTSPCamera(config, on_event=self._event_sink)
            self._cameras[camera_id] = camera
        camera.start()
        self._emit("camera_updated", camera_id=camera_id, url=redact_url(config.url))
        return camera

    # -- active camera -----------------------------------------------------

    def set_active(self, camera_id: str) -> None:
        self.get(camera_id)  # ensure it exists
        with self._lock:
            self._active_id = camera_id
        self._emit("active_changed", camera_id=camera_id)

    def get_active(self) -> RTSPCamera:
        with self._lock:
            active_id = self._active_id
        if active_id is None:
            raise NoActiveCameraError("No cameras are configured.")
        return self.get(active_id)

    @property
    def active_id(self) -> Optional[str]:
        with self._lock:
            return self._active_id

    # -- lifecycle ---------------------------------------------------------

    def start_all(self) -> None:
        for camera in self.list():
            camera.start()

    def stop_all(self) -> None:
        for camera in self.list():
            camera.stop()

    def overview(self) -> dict[str, Any]:
        statuses = [c.status() for c in self.list()]
        connected = sum(1 for s in statuses if s["state"] == "connected")
        return {
            "cameras_total": len(statuses),
            "cameras_connected": connected,
            "active_camera": self.active_id,
            "cameras": statuses,
        }
