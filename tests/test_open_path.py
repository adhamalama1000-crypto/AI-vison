"""
Tests for the RTSP open path:

* The configured URL reaches cv2.VideoCapture VERBATIM — nothing rewrites,
  sanitizes, or substitutes it (including odd-but-valid paths like
  '/ch=1&subtype=0' with a bare '&').
* 'transport: auto' tries tcp first, then udp, within one connect attempt.
* The FFmpeg socket-timeout option name matches the bundled FFmpeg.
* diagnostics.probe rejects invalid URLs with the structured error.
"""

from __future__ import annotations

import os

import pytest

from rtsp_backend import camera as camera_mod
from rtsp_backend.camera import RTSPCamera, _ffmpeg_timeout_option
from rtsp_backend.config import CameraConfig
from rtsp_backend.errors import InvalidRTSPURLError


USER_STYLE_URL = "rtsp://admin:RFID123456@192.168.100.5:554/ch=1&subtype=0"


class _FakeCapture:
    """Records constructor args; pretends the open failed."""

    calls: list[tuple] = []

    def __init__(self, *args):
        _FakeCapture.calls.append((args, os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")))

    def isOpened(self):
        return False

    def release(self):
        pass


@pytest.fixture
def fake_capture(monkeypatch):
    _FakeCapture.calls = []
    monkeypatch.setattr(camera_mod.cv2, "VideoCapture", _FakeCapture)
    return _FakeCapture


def _cam(transport: str) -> RTSPCamera:
    cfg = CameraConfig(id="t", url=USER_STYLE_URL, transport=transport,
                       open_timeout=1.0, read_timeout=1.0)
    return RTSPCamera(cfg)


def test_url_reaches_videocapture_verbatim(fake_capture):
    cam = _cam("tcp")
    assert cam._open() is None  # fake capture never opens
    assert len(fake_capture.calls) == 1
    args, _env = fake_capture.calls[0]
    # First positional arg is the source: must be the EXACT configured URL.
    assert args[0] == USER_STYLE_URL


def test_transport_auto_tries_tcp_then_udp(fake_capture):
    cam = _cam("auto")
    cam._open()
    envs = [env for _args, env in fake_capture.calls]
    assert len(envs) == 2
    assert "rtsp_transport;tcp" in envs[0]
    assert "rtsp_transport;udp" in envs[1]
    # and the URL is verbatim on every attempt
    assert all(args[0] == USER_STYLE_URL for args, _ in fake_capture.calls)


def test_transport_fixed_tries_only_that_transport(fake_capture):
    cam = _cam("udp")
    cam._open()
    assert len(fake_capture.calls) == 1
    assert "rtsp_transport;udp" in fake_capture.calls[0][1]


def test_ffmpeg_timeout_option_matches_bundled_ffmpeg():
    # On any FFmpeg >= 5 build (libavformat >= 59) the option must be
    # 'timeout'; 'stimeout' there would be silently ignored.
    opt = _ffmpeg_timeout_option()
    assert opt in ("timeout", "stimeout", None)
    import re

    import cv2

    m = re.search(r"avformat:\s*YES\s*\((\d+)\.", cv2.getBuildInformation())
    if m:
        expected = "timeout" if int(m.group(1)) >= 59 else "stimeout"
        assert opt == expected


def test_diagnostics_rejects_invalid_url():
    from rtsp_backend.diagnostics import probe

    with pytest.raises(InvalidRTSPURLError):
        probe("0")
