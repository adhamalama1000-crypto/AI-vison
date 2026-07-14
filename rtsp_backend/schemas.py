"""
Request models for the REST API.

RTSP URLs are intentionally *not* validated in these models: validation happens
in :func:`rtsp_backend.errors.validate_rtsp_url` at the manager boundary so that
an invalid source returns the backend's structured JSON error (HTTP 400) rather
than a generic pydantic 422.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import AliasChoices, BaseModel, Field

from .config import CameraConfig


class CameraCreateRequest(BaseModel):
    id: str = Field(min_length=1)
    url: str
    name: Optional[str] = None
    transport: Literal["tcp", "udp", "auto"] = "auto"
    reconnect_delay: float = Field(2.0, gt=0)
    max_reconnect_delay: float = Field(30.0, gt=0)
    open_timeout: float = Field(10.0, gt=0)
    read_timeout: float = Field(10.0, gt=0)
    max_read_failures: int = Field(30, ge=1)
    target_fps: Optional[float] = Field(None, gt=0, le=120)
    jpeg_quality: int = Field(80, ge=1, le=100)

    def to_config(self) -> CameraConfig:
        return CameraConfig(
            id=self.id,
            url=self.url,
            name=self.name or self.id,
            transport=self.transport,
            reconnect_delay=self.reconnect_delay,
            max_reconnect_delay=self.max_reconnect_delay,
            open_timeout=self.open_timeout,
            read_timeout=self.read_timeout,
            max_read_failures=self.max_read_failures,
            target_fps=self.target_fps,
            jpeg_quality=self.jpeg_quality,
        )


class CameraUpdateRequest(BaseModel):
    """All fields optional except ``url`` — the new RTSP source to switch to."""

    url: str
    name: Optional[str] = None
    transport: Optional[Literal["tcp", "udp", "auto"]] = None
    reconnect_delay: Optional[float] = Field(None, gt=0)
    max_reconnect_delay: Optional[float] = Field(None, gt=0)
    open_timeout: Optional[float] = Field(None, gt=0)
    read_timeout: Optional[float] = Field(None, gt=0)
    max_read_failures: Optional[int] = Field(None, ge=1)
    target_fps: Optional[float] = Field(None, gt=0, le=120)
    jpeg_quality: Optional[int] = Field(None, ge=1, le=100)

    def to_config(self, base: CameraConfig) -> CameraConfig:
        data = base.model_dump()
        updates = {k: v for k, v in self.model_dump().items() if v is not None}
        data.update(updates)
        data["id"] = base.id  # id is immutable across an update
        return CameraConfig(**data)


class ActiveCameraRequest(BaseModel):
    """
    Body for ``POST /active-camera``.

    The documented field is ``id`` (``{"id": "cam"}``, matching ``POST /cameras``).
    ``camera_id`` is also accepted as an alias so older callers keep working.
    """

    id: str = Field(min_length=1, validation_alias=AliasChoices("id", "camera_id"))
