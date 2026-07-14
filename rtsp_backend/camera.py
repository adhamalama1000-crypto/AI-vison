"""
The RTSP camera: one background capture thread per camera, an always-latest
thread-safe frame buffer, automatic reconnect with exponential backoff, and
health / latency / drop / connection statistics.

Design notes
------------
* Low latency: the capture thread reads as fast as the stream delivers and keeps
  only the most recent frame (``FrameBuffer``). Consumers (snapshot / MJPEG)
  always get the freshest frame; stale frames are discarded rather than queued.
* TCP by default: RTSP transport is passed to FFmpeg via a process-global env
  var, so opening a capture is serialised with ``_OPEN_LOCK`` to avoid a race
  between cameras that use different transports.
* RTSP only: ``_open`` passes ``self.config.url`` (already validated as RTSP)
  straight to FFmpeg. There is no fallback source of any kind.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from enum import Enum
from typing import Any, Callable, Optional, Tuple
from urllib.parse import urlparse

import cv2  # type: ignore

from .config import CameraConfig
from .errors import CameraNotConnectedError, FrameUnavailableError

log = logging.getLogger("rtsp_backend.camera")

# Opening a VideoCapture reads a process-global FFmpeg env var for the RTSP
# transport, so opens must not overlap across cameras. Opening is fast.
_OPEN_LOCK = threading.Lock()


def _ffmpeg_timeout_option() -> Optional[str]:
    """
    Name of the RTSP socket-timeout option for the FFmpeg build OpenCV bundles.

    FFmpeg < 5 (libavformat < 59) calls it ``stimeout``; FFmpeg >= 5 renamed it
    to ``timeout`` and passing ``stimeout`` is silently ignored — which would
    silently disable the configured read timeout at the socket layer. Returns
    ``None`` if the version can't be determined (the CAP_PROP_*_TIMEOUT_MSEC
    parameters still enforce timeouts in that case).
    """
    try:
        import re

        m = re.search(r"avformat:\s*YES\s*\((\d+)\.", cv2.getBuildInformation())
        if not m:
            return None
        return "timeout" if int(m.group(1)) >= 59 else "stimeout"
    except Exception:
        return None


_TIMEOUT_OPTION = _ffmpeg_timeout_option()


class CameraState(str, Enum):
    INIT = "initializing"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    STOPPED = "stopped"
    ERROR = "error"


def redact_url(url: str) -> str:
    """Hide the password in an RTSP URL so it is safe to expose in status APIs."""
    try:
        parsed = urlparse(url)
        if parsed.password:
            host = parsed.hostname or ""
            if parsed.port:
                host = f"{host}:{parsed.port}"
            userinfo = f"{parsed.username or ''}:***@"
            query = f"?{parsed.query}" if parsed.query else ""
            return f"{parsed.scheme}://{userinfo}{host}{parsed.path}{query}"
    except Exception:
        pass
    return url


class FrameBuffer:
    """A single-slot, thread-safe buffer that always holds the latest frame."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0
        self._ts_mono = 0.0
        self._ts_wall = 0.0

    def put(self, frame) -> None:
        with self._cond:
            self._frame = frame
            self._seq += 1
            self._ts_mono = time.monotonic()
            self._ts_wall = time.time()
            self._cond.notify_all()

    def latest(self) -> Tuple[Any, int, float, float]:
        with self._cond:
            return self._frame, self._seq, self._ts_mono, self._ts_wall

    def wait(self, last_seq: int, timeout: float) -> Tuple[Any, int, float, float]:
        """Block until a frame newer than ``last_seq`` arrives (or timeout)."""
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout)
            return self._frame, self._seq, self._ts_mono, self._ts_wall

    def wake(self) -> None:
        """Release any threads blocked in :meth:`wait` (used on shutdown)."""
        with self._cond:
            self._cond.notify_all()


class RTSPCamera:
    def __init__(
        self,
        config: CameraConfig,
        on_event: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.config = config
        self._on_event = on_event or (lambda ev: None)

        self.buffer = FrameBuffer()

        self._state = CameraState.INIT
        self._state_lock = threading.Lock()

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cur_backoff = config.reconnect_delay

        # Statistics (guarded by _stats_lock).
        self._stats_lock = threading.Lock()
        self.frames_captured = 0
        self.frames_dropped = 0          # failed reads while nominally connected
        self.read_failures_total = 0
        self.reconnect_count = 0
        self.connected_since: Optional[float] = None
        self.last_frame_wall: Optional[float] = None
        self.last_error: Optional[str] = None
        self.last_error_at: Optional[float] = None
        self.transport_in_use: Optional[str] = None  # tcp/udp actually connected with
        self._frame_times: deque[float] = deque(maxlen=120)      # monotonic
        self._read_latencies: deque[float] = deque(maxlen=120)   # milliseconds
        self._encode_ms: deque[float] = deque(maxlen=120)        # JPEG encode ms
        self._started_at = time.time()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"rtsp-{self.config.id}", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self.buffer.wake()  # release any MJPEG consumer blocked on the buffer
        if self._thread:
            self._thread.join(timeout=timeout)
        self._set_state(CameraState.STOPPED)

    # -- state + events ----------------------------------------------------

    @property
    def state(self) -> CameraState:
        with self._state_lock:
            return self._state

    def _emit(self, event_type: str, **payload) -> None:
        event = {
            "type": event_type,
            "camera_id": self.config.id,
            "camera_name": self.config.name,
            "timestamp": time.time(),
        }
        event.update(payload)
        try:
            self._on_event(event)
        except Exception:  # never let the event sink break capture
            log.exception("event callback failed for camera %s", self.config.id)

    def _set_state(self, state: CameraState, error: Optional[str] = None) -> None:
        with self._state_lock:
            prev = self._state
            if prev == state and error is None:
                return
            self._state = state

        with self._stats_lock:
            if state == CameraState.CONNECTED:
                self.connected_since = time.time()
            else:
                self.connected_since = None
            if state == CameraState.RECONNECTING and prev != CameraState.RECONNECTING:
                self.reconnect_count += 1
            if error:
                self.last_error = error
                self.last_error_at = time.time()

        event_type = {
            CameraState.CONNECTED: "connected",
            CameraState.CONNECTING: "connecting",
            CameraState.RECONNECTING: "reconnecting",
            CameraState.ERROR: "error",
            CameraState.STOPPED: "stopped",
        }.get(state, "state_change")

        payload: dict[str, Any] = {"state": state.value, "previous_state": prev.value}
        if error:
            payload["message"] = error
        self._emit(event_type, **payload)

    # -- capture loop ------------------------------------------------------

    def _construct_capture(self) -> "cv2.VideoCapture":
        params: list[int] = []
        for name, seconds in (
            ("CAP_PROP_OPEN_TIMEOUT_MSEC", self.config.open_timeout * 1000),
            ("CAP_PROP_READ_TIMEOUT_MSEC", self.config.read_timeout * 1000),
        ):
            const = getattr(cv2, name, None)
            if const is not None:
                params += [int(const), int(seconds)]
        try:
            if params:
                return cv2.VideoCapture(self.config.url, cv2.CAP_FFMPEG, params)
        except Exception:
            pass
        return cv2.VideoCapture(self.config.url, cv2.CAP_FFMPEG)

    def _open(self) -> Optional["cv2.VideoCapture"]:
        """
        Open the configured RTSP URL. ``transport: auto`` tries TCP first, then
        UDP — some cameras only implement one of the two (the classic
        "works in VLC but not here" cause, since VLC negotiates transports).
        """
        if self.config.transport == "auto":
            transports = ("tcp", "udp")
        else:
            transports = (self.config.transport,)

        for transport in transports:
            opts = f"rtsp_transport;{transport}"
            if _TIMEOUT_OPTION is not None:
                stimeout_us = int(self.config.read_timeout * 1_000_000)
                opts += f"|{_TIMEOUT_OPTION};{stimeout_us}"
            if getattr(self.config, "low_latency", True):
                # Ultra-low-latency FFmpeg tuning:
                #   nobuffer     - don't buffer frames inside the demuxer
                #   low_delay    - decoder emits frames ASAP (no reordering wait)
                #   max_delay;0  - no added latency for stream interleaving
                #   reorder_queue_size;0 - don't hold RTP packets to reorder
                #   probesize / analyzeduration small - connect fast, less pre-buffer
                opts += (
                    "|fflags;nobuffer"
                    "|flags;low_delay"
                    "|max_delay;0"
                    "|reorder_queue_size;0"
                    "|probesize;100000"
                    "|analyzeduration;0"
                )
            with _OPEN_LOCK:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = opts
                # NOTE: the source is ALWAYS the configured RTSP URL, passed
                # verbatim. There is no fallback to device 0, a USB camera, or
                # a local file — only the *transport* may vary on "auto".
                cap = self._construct_capture()

            if cap.isOpened():
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimise buffering latency
                except Exception:
                    pass
                with self._stats_lock:
                    self.transport_in_use = transport
                return cap

            cap.release()
            if len(transports) > 1:
                log.info(
                    "camera %s: open failed over %s, trying next transport",
                    self.config.id,
                    transport,
                )

        with self._stats_lock:
            self.transport_in_use = None
        return None

    def _record_frame(self, latency_ms: float) -> None:
        now = time.monotonic()
        with self._stats_lock:
            self.frames_captured += 1
            self.last_frame_wall = time.time()
            self._frame_times.append(now)
            self._read_latencies.append(latency_ms)

    def _backoff(self) -> bool:
        """Sleep for the current backoff. Returns False if stopped meanwhile."""
        delay = self._cur_backoff
        interrupted = self._stop.wait(delay)
        self._cur_backoff = min(delay * 2, self.config.max_reconnect_delay)
        return not interrupted

    def _run(self) -> None:
        # target_fps caps the *published* rate. We never sleep between socket
        # reads (that would let the decoder's queue grow and add latency);
        # instead, when throttling, we cheaply grab() and discard intermediate
        # frames and only decode (retrieve) the newest one at the target cadence.
        min_interval = (1.0 / self.config.target_fps) if self.config.target_fps else 0.0

        while not self._stop.is_set():
            self._set_state(CameraState.CONNECTING)
            cap = self._open()

            if cap is None:
                self._set_state(
                    CameraState.RECONNECTING,
                    error="Failed to open RTSP stream (connection refused or timed out).",
                )
                if self._backoff():
                    continue
                break

            self._set_state(CameraState.CONNECTED)
            self._cur_backoff = self.config.reconnect_delay  # reset backoff
            consecutive_failures = 0
            last_publish = 0.0

            while not self._stop.is_set():
                t0 = time.monotonic()

                if min_interval:
                    # Keep the socket/decoder drained: grab (no decode) until it
                    # is time to publish, so the frame we finally decode is the
                    # freshest one, not a backlog entry.
                    now = time.monotonic()
                    if (now - last_publish) < min_interval:
                        if not cap.grab():
                            ok = False
                        else:
                            consecutive_failures = 0
                            continue
                    else:
                        ok, frame = cap.read()
                        last_publish = time.monotonic()
                else:
                    ok, frame = cap.read()

                latency_ms = (time.monotonic() - t0) * 1000.0

                if not ok or frame is None:
                    consecutive_failures += 1
                    with self._stats_lock:
                        self.frames_dropped += 1
                        self.read_failures_total += 1
                    if consecutive_failures >= self.config.max_read_failures:
                        break
                    if self._stop.wait(0.02):  # avoid a hot loop on transient errors
                        break
                    continue

                consecutive_failures = 0
                self._record_frame(latency_ms)
                self.buffer.put(frame)

            cap.release()
            if self._stop.is_set():
                break

            self._set_state(
                CameraState.RECONNECTING,
                error="RTSP stream lost; attempting to reconnect.",
            )
            if not self._backoff():
                break

        self._set_state(CameraState.STOPPED)

    # -- derived metrics ---------------------------------------------------

    def _fps(self) -> float:
        with self._stats_lock:
            times = list(self._frame_times)
        if len(times) < 2:
            return 0.0
        span = times[-1] - times[0]
        if span <= 0:
            return 0.0
        return round((len(times) - 1) / span, 2)

    def _latency(self) -> dict[str, Optional[float]]:
        with self._stats_lock:
            lat = list(self._read_latencies)
        if not lat:
            return {"last_ms": None, "avg_ms": None, "max_ms": None}
        return {
            "last_ms": round(lat[-1], 2),
            "avg_ms": round(sum(lat) / len(lat), 2),
            "max_ms": round(max(lat), 2),
        }

    def _record_encode(self, ms: float) -> None:
        with self._stats_lock:
            self._encode_ms.append(ms)

    def _encode_metric(self) -> dict[str, Optional[float]]:
        with self._stats_lock:
            enc = list(self._encode_ms)
        if not enc:
            return {"last_ms": None, "avg_ms": None}
        return {"last_ms": round(enc[-1], 2),
                "avg_ms": round(sum(enc) / len(enc), 2)}

    # -- public views ------------------------------------------------------

    def status(self) -> dict[str, Any]:
        state = self.state
        frame, seq, ts_mono, _ = self.buffer.latest()
        age_ms = round((time.monotonic() - ts_mono) * 1000.0, 1) if frame is not None else None

        with self._stats_lock:
            connected_since = self.connected_since
            transport_in_use = self.transport_in_use
            statistics = {
                "frames_captured": self.frames_captured,
                "frames_dropped": self.frames_dropped,
                "read_failures_total": self.read_failures_total,
                "reconnect_count": self.reconnect_count,
                "connected_since": connected_since,
                "last_frame_at": self.last_frame_wall,
                "last_error": self.last_error,
                "last_error_at": self.last_error_at,
                "uptime_seconds": round(time.time() - self._started_at, 1),
            }

        healthy = (
            state == CameraState.CONNECTED
            and frame is not None
            and age_ms is not None
            and age_ms < 5000
        )
        connected_for = round(time.time() - connected_since, 1) if connected_since else None

        return {
            "id": self.config.id,
            "name": self.config.name,
            "url": redact_url(self.config.url),
            "transport": self.config.transport,
            "transport_in_use": transport_in_use,
            "state": state.value,
            "healthy": healthy,
            "has_frame": frame is not None,
            "frame_seq": seq,
            "frame_age_ms": age_ms,
            "connected_for_seconds": connected_for,
            "fps": self._fps(),
            "latency": self._latency(),
            "encode": self._encode_metric(),
            "frame_age_ms": age_ms,
            "statistics": statistics,
        }

    # -- frame access ------------------------------------------------------

    def snapshot_jpeg(self, quality: Optional[int] = None) -> Tuple[bytes, float]:
        frame, _seq, _ts_mono, ts_wall = self.buffer.latest()
        state = self.state
        if frame is None:
            if state == CameraState.CONNECTED:
                raise FrameUnavailableError(
                    "Camera is connected but no frame is available yet.",
                    camera_id=self.config.id,
                    details={"state": state.value},
                )
            raise CameraNotConnectedError(
                f"Camera '{self.config.id}' is not connected (state: {state.value}); "
                "no snapshot is available.",
                camera_id=self.config.id,
                details={"state": state.value, "last_error": self.last_error},
            )

        q = quality if quality is not None else self.config.jpeg_quality
        _t = time.monotonic()
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
        self._record_encode((time.monotonic() - _t) * 1000.0)
        if not ok:
            raise FrameUnavailableError(
                "Failed to encode frame as JPEG.", camera_id=self.config.id
            )
        return buf.tobytes(), ts_wall

    def next_jpeg(
        self, last_seq: int, timeout: float, quality: Optional[int] = None
    ) -> Tuple[Optional[bytes], int]:
        """Wait for the next frame after ``last_seq`` and return it as JPEG."""
        frame, seq, _ts_mono, _ts_wall = self.buffer.wait(last_seq, timeout)
        if frame is None or seq == last_seq:
            return None, seq
        q = quality if quality is not None else self.config.jpeg_quality
        _t = time.monotonic()
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
        self._record_encode((time.monotonic() - _t) * 1000.0)
        if not ok:
            return None, seq
        return buf.tobytes(), seq
