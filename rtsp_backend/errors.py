"""
Error types and the single authoritative RTSP-source validator.

This module is the gate that enforces the core guarantee of this backend:
the video source *always* comes from configuration and is *always* RTSP.
There is deliberately no code path anywhere in this project that opens
capture device ``0``, a USB device, an HTTP stream, or a local video file.
Anything that is not ``rtsp://`` / ``rtsps://`` is rejected here with a
descriptive JSON error instead of being silently substituted.
"""

from __future__ import annotations

from typing import Any, Optional
from urllib.parse import urlparse


class RTSPBackendError(Exception):
    """Base class for all errors that should be rendered as JSON to clients."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(
        self,
        message: str,
        *,
        camera_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        status_code: Optional[int] = None,
        code: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.camera_id = camera_id
        self.details = details or {}
        if status_code is not None:
            self.status_code = status_code
        if code is not None:
            self.code = code

    def to_dict(self) -> dict[str, Any]:
        err: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.camera_id is not None:
            err["camera_id"] = self.camera_id
        if self.details:
            err["details"] = self.details
        return {"error": err}


class InvalidRTSPURLError(RTSPBackendError):
    status_code = 400
    code = "invalid_rtsp_url"


class CameraNotFoundError(RTSPBackendError):
    status_code = 404
    code = "camera_not_found"


class DuplicateCameraError(RTSPBackendError):
    status_code = 409
    code = "duplicate_camera"


class CameraNotConnectedError(RTSPBackendError):
    status_code = 503
    code = "camera_not_connected"


class FrameUnavailableError(RTSPBackendError):
    status_code = 503
    code = "frame_unavailable"


class NoActiveCameraError(RTSPBackendError):
    status_code = 404
    code = "no_active_camera"


ALLOWED_SCHEMES = ("rtsp", "rtsps")


def validate_rtsp_url(url: Any, camera_id: Optional[str] = None) -> str:
    """
    Validate that ``url`` is a usable RTSP source and return the cleaned URL.

    Raises :class:`InvalidRTSPURLError` (HTTP 400) for anything that is not an
    RTSP URL. This is intentionally strict: numeric device indices, local file
    paths, ``file://``, and ``http(s)://`` streams are all refused so that the
    backend can never fall back to a USB webcam or a local video file.
    """
    if not isinstance(url, str) or not url.strip():
        raise InvalidRTSPURLError(
            "Camera source is empty. An RTSP URL (rtsp:// or rtsps://) is required.",
            camera_id=camera_id,
            details={"provided": url},
        )

    candidate = url.strip()

    # Explicitly reject integer indices such as "0" or "1" — these are the
    # OpenCV shorthands for USB / built-in capture devices.
    if candidate.isdigit():
        raise InvalidRTSPURLError(
            f"Numeric camera index '{candidate}' is not allowed. This backend is "
            "RTSP-only and never opens USB or local capture devices.",
            camera_id=camera_id,
            details={"provided": url, "reason": "numeric_device_index"},
        )

    parsed = urlparse(candidate)
    scheme = parsed.scheme.lower()

    if scheme in ("", "file"):
        raise InvalidRTSPURLError(
            "Local files and device paths are not supported. Provide an RTSP URL "
            f"(rtsp:// or rtsps://). Got: '{url}'.",
            camera_id=camera_id,
            details={"provided": url, "reason": "local_file_or_path"},
        )

    if scheme not in ALLOWED_SCHEMES:
        raise InvalidRTSPURLError(
            f"Unsupported source scheme '{parsed.scheme}'. This backend supports "
            "RTSP only (rtsp:// or rtsps://). USB devices, numeric indices, "
            "HTTP(S) streams, and local video files are not allowed.",
            camera_id=camera_id,
            details={
                "provided": url,
                "reason": "unsupported_scheme",
                "scheme": parsed.scheme,
            },
        )

    if not parsed.hostname:
        raise InvalidRTSPURLError(
            f"RTSP URL '{url}' is missing a host.",
            camera_id=camera_id,
            details={"provided": url, "reason": "missing_host"},
        )

    return candidate
