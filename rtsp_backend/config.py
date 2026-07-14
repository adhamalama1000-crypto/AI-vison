"""
Configuration for the RTSP backend.

Cameras are declared in a YAML file (default ``config.yaml``, override with the
``RTSP_CONFIG`` environment variable) or added at runtime through the API. In
every case the source is an RTSP URL that is validated by
:func:`rtsp_backend.errors.validate_rtsp_url`.
"""

from __future__ import annotations

import os
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

from .errors import validate_rtsp_url


class CameraConfig(BaseModel):
    """Full configuration for a single RTSP camera."""

    id: str = Field(min_length=1)
    url: str
    name: str = ""
    transport: Literal["tcp", "udp", "auto"] = "auto"

    # Reconnect behaviour (exponential backoff between reconnect_delay and max).
    reconnect_delay: float = Field(2.0, gt=0)
    max_reconnect_delay: float = Field(30.0, gt=0)

    # FFmpeg open / socket timeouts, in seconds.
    open_timeout: float = Field(10.0, gt=0)
    read_timeout: float = Field(10.0, gt=0)

    # Consecutive failed reads before we treat the stream as lost and reconnect.
    max_read_failures: int = Field(30, ge=1)

    # Optional cap on the *published* frame rate. ``None`` = as fast as the
    # stream delivers. When set, intermediate frames are cheaply grabbed and
    # discarded (not decoded) so throttling never lets latency accumulate.
    target_fps: Optional[float] = Field(None, gt=0, le=120)

    # Ultra-low-latency mode: pass aggressive no-buffer / low-delay flags to
    # FFmpeg and always publish only the newest decoded frame. Turn off only if
    # a particular camera misbehaves with these flags.
    low_latency: bool = True

    # Default JPEG quality for snapshot / MJPEG output (1-100).
    jpeg_quality: int = Field(80, ge=1, le=100)

    @model_validator(mode="after")
    def _default_name(self) -> "CameraConfig":
        if not self.name:
            self.name = self.id
        return self


class Settings(BaseModel):
    """Top-level service settings."""

    host: str = "0.0.0.0"
    port: int = 8000
    active_camera: Optional[str] = None
    stats_interval: float = Field(5.0, gt=0)
    cameras: list[CameraConfig] = Field(default_factory=list)

    # Data / AI storage locations.
    data_dir: str = "data"
    db_path: str = "data/platform.db"
    models_dir: str = "models"
    # Minimum seconds between AI inference passes per camera (throttle).
    ai_min_interval: float = Field(0.2, ge=0)

    # Optional API key. When set (RTSP_API_KEY), every /api and control request
    # must present it via the ``X-API-Key`` header or ``?api_key=`` query param;
    # /health and the static dashboard stay open. Unset => open (dev/test).
    api_key: Optional[str] = None
    # Allow an unauthenticated model-select to trigger `pip install insightface`.
    # Off by default (a public endpoint should not mutate the host environment);
    # set RTSP_ALLOW_AUTO_INSTALL=1 to enable.
    allow_auto_install: bool = False
    # Hard cap (bytes) on a single uploaded file / request body handled by the
    # dataset / reference / panel / inspection endpoints. Default 512 MiB.
    max_upload_bytes: int = 536870912


def _load_dotenv() -> None:
    """
    Best-effort load of a ``.env`` file into the process environment, so this
    service can be configured per-instance from a file.

    ``RTSP_ENV_FILE`` selects a specific file, which is what lets you run several
    independent instances side by side (each with its own env file / port /
    active camera). With it unset, a ``.env`` in the working directory is used if
    present. Real environment variables always win over ``.env`` values.
    """
    try:
        from dotenv import load_dotenv
    except Exception:
        # python-dotenv not installed: env vars still work, .env files are skipped.
        return

    env_file = os.environ.get("RTSP_ENV_FILE")
    if env_file:
        load_dotenv(env_file, override=False)
    else:
        load_dotenv(override=False)  # loads ./.env if it exists


def load_settings(path: Optional[str] = None) -> Settings:
    """
    Load settings from a YAML config file, with environment overrides.

    The camera list is validated as RTSP-only at load time, so a misconfigured
    source (a file path, ``0``, an HTTP URL, ...) fails fast at startup with a
    clear message rather than silently degrading at runtime.
    """
    _load_dotenv()
    path = path or os.environ.get("RTSP_CONFIG", "config.yaml")

    data: dict = {}
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

    host = os.environ.get("RTSP_HOST", data.get("host", "0.0.0.0"))
    port = int(os.environ.get("RTSP_PORT", data.get("port", 8000)))
    stats_interval = float(
        os.environ.get("RTSP_STATS_INTERVAL", data.get("stats_interval", 5.0))
    )
    active_camera = os.environ.get("RTSP_ACTIVE_CAMERA", data.get("active_camera"))

    cameras: list[CameraConfig] = []
    for raw in data.get("cameras", []) or []:
        # Authoritative RTSP-only gate before we build the model.
        validate_rtsp_url(raw.get("url", ""), camera_id=raw.get("id"))
        cameras.append(CameraConfig(**raw))

    data_dir = os.environ.get("RTSP_DATA_DIR", data.get("data_dir", "data"))
    db_path = os.environ.get(
        "RTSP_DB_PATH", data.get("db_path", os.path.join(data_dir, "platform.db"))
    )
    models_dir = os.environ.get("RTSP_MODELS_DIR", data.get("models_dir", "models"))
    ai_min_interval = float(
        os.environ.get("RTSP_AI_MIN_INTERVAL", data.get("ai_min_interval", 0.2))
    )

    api_key = os.environ.get("RTSP_API_KEY", data.get("api_key")) or None

    def _truthy(v) -> bool:
        return str(v).lower() in ("1", "true", "yes", "on")

    allow_auto_install = _truthy(
        os.environ.get("RTSP_ALLOW_AUTO_INSTALL", data.get("allow_auto_install", False))
    )
    max_upload_bytes = int(
        os.environ.get("RTSP_MAX_UPLOAD_BYTES", data.get("max_upload_bytes", 536870912))
    )

    return Settings(
        host=host,
        port=port,
        active_camera=active_camera,
        stats_interval=stats_interval,
        cameras=cameras,
        data_dir=data_dir,
        db_path=db_path,
        models_dir=models_dir,
        ai_min_interval=ai_min_interval,
        api_key=api_key,
        allow_auto_install=allow_auto_install,
        max_upload_bytes=max_upload_bytes,
    )
