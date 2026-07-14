"""
Smoke tests for the guarantees that matter most:

* Non-RTSP sources ("0", file paths, http://) are rejected with a JSON error.
* An unreachable RTSP camera never falls back to another source; snapshots
  return a structured JSON error instead of a placeholder image.
* Basic REST surface (health, list, create, delete, active camera) works.

These do not require a real RTSP server. Run with:  pytest -q
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rtsp_backend.app import build_app
from rtsp_backend.config import CameraConfig, Settings
from rtsp_backend.errors import InvalidRTSPURLError, validate_rtsp_url


# -- validator unit tests -------------------------------------------------

@pytest.mark.parametrize("bad", ["0", "1", "", "   ", "/tmp/video.mp4", "video.mp4",
                                  "file:///tmp/v.mp4", "http://x/stream", "https://x/s"])
def test_validator_rejects_non_rtsp(bad):
    with pytest.raises(InvalidRTSPURLError):
        validate_rtsp_url(bad)


@pytest.mark.parametrize("good", [
    "rtsp://192.168.1.10:554/stream",
    "rtsp://user:pass@10.0.0.5:554/Streaming/Channels/101",
    "rtsps://cam.local/stream1",
])
def test_validator_accepts_rtsp(good):
    assert validate_rtsp_url(good) == good


# -- app fixtures ---------------------------------------------------------

def _fast_camera(cam_id: str, url: str) -> CameraConfig:
    # Short timeouts so an unreachable host fails (and shuts down) quickly.
    return CameraConfig(
        id=cam_id, url=url, name=cam_id,
        reconnect_delay=0.2, max_reconnect_delay=0.5,
        open_timeout=1.0, read_timeout=1.0, max_read_failures=3,
    )


@pytest.fixture
def client_with_unreachable():
    settings = Settings(
        cameras=[_fast_camera("cam1", "rtsp://127.0.0.1:1/nostream")],
        active_camera="cam1",
        stats_interval=60.0,
    )
    with TestClient(build_app(settings)) as client:
        yield client


@pytest.fixture
def client_empty():
    with TestClient(build_app(Settings()), raise_server_exceptions=False) as client:
        yield client


# -- REST tests -----------------------------------------------------------

def test_health(client_with_unreachable):
    r = client_with_unreachable.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["rtsp_only"] is True
    assert body["cameras_total"] == 1


def test_list_cameras(client_with_unreachable):
    r = client_with_unreachable.get("/cameras")
    assert r.status_code == 200
    cams = r.json()["cameras"]
    assert len(cams) == 1
    assert cams[0]["id"] == "cam1"
    # State should be a connection state, never "connected" to a fake source.
    assert cams[0]["state"] in {"initializing", "connecting", "reconnecting", "error"}
    assert cams[0]["has_frame"] is False


def test_create_rejects_usb_index(client_empty):
    r = client_empty.post("/cameras", json={"id": "usb", "url": "0"})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "invalid_rtsp_url"
    assert err["details"]["reason"] == "numeric_device_index"


def test_create_rejects_local_file(client_empty):
    r = client_empty.post("/cameras", json={"id": "file", "url": "/tmp/clip.mp4"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_rtsp_url"


def test_snapshot_unreachable_returns_json_error(client_with_unreachable):
    r = client_with_unreachable.get("/cameras/cam1/snapshot")
    # No frame + no fallback => structured 503, not an image.
    assert r.status_code == 503
    assert r.headers["content-type"].startswith("application/json")
    assert r.json()["error"]["code"] in {"camera_not_connected", "frame_unavailable"}


def test_snapshot_unknown_camera_404(client_with_unreachable):
    r = client_with_unreachable.get("/cameras/nope/snapshot")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "camera_not_found"


def test_no_active_camera_when_empty(client_empty):
    r = client_empty.get("/snapshot")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "no_active_camera"


def test_create_and_delete_cycle(client_empty):
    created = client_empty.post(
        "/cameras", json={"id": "c2", "url": "rtsp://127.0.0.1:1/x", "read_timeout": 1.0,
                          "open_timeout": 1.0, "reconnect_delay": 0.2, "max_reconnect_delay": 0.5,
                          "max_read_failures": 3},
    )
    assert created.status_code == 201
    assert created.json()["id"] == "c2"

    # Duplicate should conflict.
    dup = client_empty.post("/cameras", json={"id": "c2", "url": "rtsp://127.0.0.1:1/x"})
    assert dup.status_code == 409

    deleted = client_empty.delete("/cameras/c2")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] == "c2"


def test_websocket_hello(client_with_unreachable):
    with client_with_unreachable.websocket_connect("/ws/events") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello"
        assert "cameras" in hello


def test_unexpected_exception_returns_json_envelope(client_empty):
    # An internal crash must never surface as bare text "Internal Server Error";
    # it must use the same structured JSON envelope as every other error.
    app = client_empty.app
    original = app.state.manager.overview
    app.state.manager.overview = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        r = client_empty.get("/health")
        assert r.status_code == 500
        assert r.headers["content-type"].startswith("application/json")
        err = r.json()["error"]
        assert err["code"] == "internal_error"
        assert "boom" in err["message"]
    finally:
        app.state.manager.overview = original
