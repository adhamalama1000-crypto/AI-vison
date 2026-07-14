"""
Verifies the capture -> buffer -> JPEG -> MJPEG pipeline WITHOUT a real RTSP
camera, by injecting synthetic frames straight into a camera's FrameBuffer.

This is the closest honest proxy for "live video is displayed": it proves that
once a frame lands in the buffer (which the capture thread does on every read),
the snapshot, next_jpeg, status, and the HTTP MJPEG stream all deliver correct,
decodable image bytes with low latency and correct multipart framing.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np
from fastapi.testclient import TestClient

from rtsp_backend.app import build_app
from rtsp_backend.camera import CameraState, RTSPCamera
from rtsp_backend.config import CameraConfig, Settings


def _frame(w=64, h=48, val=128):
    f = np.full((h, w, 3), val, dtype=np.uint8)
    f[0, 0] = (0, 0, 255)  # a distinguishable pixel
    return f


def test_snapshot_encodes_injected_frame():
    cam = RTSPCamera(CameraConfig(id="c", url="rtsp://127.0.0.1:554/x"))
    cam.buffer.put(_frame(val=200))
    # Pretend the capture loop marked us connected + recorded the frame.
    cam._set_state(CameraState.CONNECTED)
    cam._record_frame(3.0)

    jpg, ts = cam.snapshot_jpeg(quality=90)
    assert jpg[:2] == b"\xff\xd8" and jpg[-2:] == b"\xff\xd9"  # valid JPEG markers
    decoded = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None and decoded.shape == (48, 64, 3)

    st = cam.status()
    assert st["has_frame"] is True
    assert st["healthy"] is True
    assert st["fps"] >= 0.0
    assert st["latency"]["last_ms"] == 3.0


def test_next_jpeg_returns_new_frame():
    cam = RTSPCamera(CameraConfig(id="c", url="rtsp://127.0.0.1:554/x"))
    cam.buffer.put(_frame(val=50))
    jpg, seq = cam.next_jpeg(last_seq=-1, timeout=1.0, quality=80)
    assert jpg is not None and jpg[:2] == b"\xff\xd8"
    assert seq == 1
    # No new frame -> returns None for the same seq (stream keeps waiting).
    jpg2, seq2 = cam.next_jpeg(last_seq=seq, timeout=0.2)
    assert jpg2 is None and seq2 == seq


def test_mjpeg_generator_delivers_multipart_frames():
    """
    Exercise the exact async generator the /stream endpoint uses. A thread
    injects fresh frames; we pull several multipart parts and confirm each
    carries a decodable JPEG with correct boundary framing. Terminates
    deterministically (no live HTTP client that can block on an infinite body).
    """
    import asyncio

    from rtsp_backend.app import _mjpeg_stream
    from rtsp_backend.manager import CameraManager

    mgr = CameraManager()
    cam = mgr.add_camera(
        CameraConfig(id="live", url="rtsp://127.0.0.1:554/x"),
        start=False,
    )
    cam._set_state(CameraState.CONNECTED)

    stop = threading.Event()

    def pump():
        i = 0
        while not stop.is_set():
            cam.buffer.put(_frame(val=(i * 7) % 255))
            cam._record_frame(2.0)
            i += 1
            time.sleep(0.02)  # ~50 fps

    async def collect():
        parts, jpeg_ok = 0, False
        agen = _mjpeg_stream(mgr, "live", quality=70, fps_cap=None)
        try:
            deadline = time.time() + 5
            async for chunk in agen:
                assert chunk.startswith(b"--frame\r\n")
                assert b"Content-Type: image/jpeg" in chunk
                s = chunk.find(b"\xff\xd8")
                e = chunk.find(b"\xff\xd9", s + 2)
                if s != -1 and e != -1:
                    img = cv2.imdecode(
                        np.frombuffer(chunk[s:e + 2], np.uint8), cv2.IMREAD_COLOR
                    )
                    jpeg_ok = jpeg_ok or (img is not None and img.shape == (48, 64, 3))
                parts += 1
                if parts >= 3 and jpeg_ok:
                    break
                if time.time() > deadline:
                    break
        finally:
            await agen.aclose()
        return parts, jpeg_ok

    t = threading.Thread(target=pump, daemon=True)
    t.start()
    try:
        parts, jpeg_ok = asyncio.run(collect())
    finally:
        stop.set()
        t.join(timeout=2)
    assert parts >= 3, f"expected >=3 multipart parts, got {parts}"
    assert jpeg_ok, "stream did not deliver a decodable JPEG frame"


def test_status_reports_low_frame_age_when_fresh():
    cam = RTSPCamera(CameraConfig(id="c", url="rtsp://127.0.0.1:554/x"))
    cam._set_state(CameraState.CONNECTED)
    cam.buffer.put(_frame())
    cam._record_frame(1.0)
    st = cam.status()
    assert st["frame_age_ms"] is not None and st["frame_age_ms"] < 1000
    assert st["healthy"] is True
