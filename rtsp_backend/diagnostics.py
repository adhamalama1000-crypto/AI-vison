"""
RTSP connection diagnostics.

Answers "why can't the backend open this stream?" with hard evidence instead of
guesses:

* exact URL parse (redacted) — proves what host/port/path FFmpeg is given
* raw TCP reachability of host:port — separates network problems from RTSP ones
* one open attempt per transport (tcp, udp), each capturing the *real*
  C-level FFmpeg stderr (401 Unauthorized, 404 Not Found, timeouts, ...)
* whether a frame could actually be decoded, and its size

Used by the ``diagnose.py`` CLI and the ``GET /cameras/{id}/diagnose`` endpoint.

Note: capturing FFmpeg's stderr requires temporarily redirecting the process's
stderr file descriptor, so probes are serialized by a module lock and intended
for on-demand troubleshooting, not the hot path.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from typing import Any, Optional
from urllib.parse import urlparse

import cv2  # type: ignore

from .camera import _TIMEOUT_OPTION, redact_url
from .errors import validate_rtsp_url

_PROBE_LOCK = threading.Lock()


def _tcp_reachable(host: str, port: int, timeout: float) -> dict[str, Any]:
    t0 = time.monotonic()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return {"reachable": True, "ms": round((time.monotonic() - t0) * 1000, 1)}
    except OSError as exc:
        return {
            "reachable": False,
            "ms": round((time.monotonic() - t0) * 1000, 1),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _attempt(url: str, transport: str, open_timeout: float) -> dict[str, Any]:
    """One open attempt over one transport, with FFmpeg stderr captured."""
    opts = f"rtsp_transport;{transport}"
    if _TIMEOUT_OPTION is not None:
        opts += f"|{_TIMEOUT_OPTION};{int(open_timeout * 1_000_000)}"

    read_fd, write_fd = os.pipe()
    saved_stderr = os.dup(2)
    os.dup2(write_fd, 2)
    os.close(write_fd)
    prev_opts = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
    t0 = time.monotonic()
    opened = frame_ok = False
    frame_shape = None
    attempt_error: Optional[str] = None
    try:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = opts
        params = []
        for name, ms in (
            ("CAP_PROP_OPEN_TIMEOUT_MSEC", open_timeout * 1000),
            ("CAP_PROP_READ_TIMEOUT_MSEC", open_timeout * 1000),
        ):
            const = getattr(cv2, name, None)
            if const is not None:
                params += [int(const), int(ms)]
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)
        try:
            opened = cap.isOpened()
            if opened:
                ok, frame = cap.read()
                frame_ok = bool(ok) and frame is not None
                if frame_ok:
                    frame_shape = list(frame.shape)
        finally:
            cap.release()
    except Exception as exc:  # a probe must report failures, never crash the API
        attempt_error = f"{type(exc).__name__}: {exc}"
    finally:
        if prev_opts is None:
            os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
        else:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = prev_opts
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
        captured = b""
        try:
            # drain what FFmpeg wrote (non-blocking-ish: pipe already closed for writing)
            os.set_blocking(read_fd, False)
            while True:
                chunk = os.read(read_fd, 65536)
                if not chunk:
                    break
                captured += chunk
        except (BlockingIOError, OSError):
            pass
        os.close(read_fd)

    stderr_lines = captured.decode(errors="replace").strip().splitlines()
    result: dict[str, Any] = {
        "transport": transport,
        "ffmpeg_options": opts,
        "opened": opened,
        "frame_decoded": frame_ok,
        "frame_shape": frame_shape,
        "seconds": round(time.monotonic() - t0, 2),
        "ffmpeg_stderr": stderr_lines[-12:],  # the tail carries the verdict
    }
    if attempt_error is not None:
        result["error"] = attempt_error
    return result


def probe(
    url: str,
    open_timeout: float = 10.0,
    transports: tuple[str, ...] = ("tcp", "udp"),
) -> dict[str, Any]:
    """Run the full diagnostic against an RTSP URL and return a JSON-able report."""
    validate_rtsp_url(url)  # raises the structured 400 if the URL itself is bad
    parsed = urlparse(url)
    host = parsed.hostname or ""
    try:
        port = parsed.port or 554
    except ValueError:  # e.g. non-numeric port slipped through
        port = 554

    report: dict[str, Any] = {
        "url": redact_url(url),
        "parsed": {
            "scheme": parsed.scheme,
            "host": host,
            "port": port,
            "username": parsed.username,
            "password_present": parsed.password is not None,
            "path": parsed.path,
            "query": parsed.query,
        },
        "ffmpeg": {
            "opencv_version": cv2.__version__,
            "socket_timeout_option": _TIMEOUT_OPTION,
        },
        "tcp_port": _tcp_reachable(host, port, timeout=min(open_timeout, 5.0)),
        "attempts": [],
    }

    with _PROBE_LOCK:
        for transport in transports:
            report["attempts"].append(_attempt(url, transport, open_timeout))

    # A plain-language verdict so the user can act immediately.
    if not report["tcp_port"]["reachable"]:
        verdict = (
            f"Cannot even open a TCP connection to {host}:{port} — this is a "
            "network problem (wrong IP/port, camera offline, firewall/VLAN), "
            "not an RTSP problem."
        )
    else:
        good = [a for a in report["attempts"] if a["frame_decoded"]]
        opened_only = [a for a in report["attempts"] if a["opened"] and not a["frame_decoded"]]
        if good:
            verdict = (
                f"SUCCESS over {good[0]['transport']}. Set 'transport: "
                f"{good[0]['transport']}' (or leave 'auto') and the backend will connect."
            )
        elif opened_only:
            verdict = (
                "RTSP session opens but no frame decodes — usually the wrong "
                "stream path/subtype or an unsupported codec. Check ffmpeg_stderr."
            )
        else:
            verdict = (
                "TCP port is reachable but every RTSP attempt failed. Read "
                "ffmpeg_stderr in each attempt: '401' means wrong credentials, "
                "'404'/'describe failed' means wrong stream path, a timeout "
                "with no HTTP-style error usually means the camera dislikes "
                "the transport or throttles concurrent clients (close VLC "
                "while testing!)."
            )
    report["verdict"] = verdict
    return report
