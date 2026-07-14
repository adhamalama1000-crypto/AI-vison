"""Shared test fixtures and real-image helpers.

Uses scikit-image's bundled ``astronaut`` photograph — a real human face that
OpenCV's Haar cascade reliably detects — so the face pipeline is exercised end
to end with genuine image data rather than synthetic stand-ins.
"""

from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest


def _rgb_to_bgr(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


@pytest.fixture(scope="session")
def astronaut_bgr():
    from skimage import data
    return _rgb_to_bgr(data.astronaut())


@pytest.fixture(scope="session")
def astronaut_variant(astronaut_bgr):
    """A mildly altered copy of the same face (brightness + slight blur).

    Represents 'another photo of the same person' for threshold testing.
    """
    img = cv2.convertScaleAbs(astronaut_bgr, alpha=1.05, beta=12)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def to_data_url(bgr) -> str:
    ok, buf = cv2.imencode(".jpg", bgr)
    assert ok
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


@pytest.fixture
def temp_settings(tmp_path):
    from rtsp_backend.config import Settings
    return Settings(
        db_path=str(tmp_path / "test.db"),
        data_dir=str(tmp_path / "data"),
        models_dir=str(tmp_path / "models"),
        cameras=[],
    )


@pytest.fixture
def client(temp_settings):
    import warnings
    warnings.filterwarnings("ignore")
    from fastapi.testclient import TestClient
    from rtsp_backend.app import build_app
    with TestClient(build_app(temp_settings)) as c:
        yield c


@pytest.fixture
def camera_client(temp_settings, astronaut_bgr):
    """A running app with a fake camera whose buffer holds a real face frame.

    No RTSP connection is opened; frames are injected straight into the buffer,
    which is exactly how snapshots/captures read them.
    """
    import warnings
    warnings.filterwarnings("ignore")
    from fastapi.testclient import TestClient
    from rtsp_backend.app import build_app
    from rtsp_backend.config import CameraConfig

    app = build_app(temp_settings)
    with TestClient(app) as c:
        cam = app.state.manager.add_camera(
            CameraConfig(id="cam1", name="Fake Cam", url="rtsp://example/stream"),
            start=False, emit=False,
        )
        cam.buffer.put(astronaut_bgr.copy())
        yield c, "cam1", cam
