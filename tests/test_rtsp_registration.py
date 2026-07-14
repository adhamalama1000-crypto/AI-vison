"""
Regression tests for the RTSP-camera-only employee-registration workflow.

These lock in the behaviour that was broken before:
  * face recognition must be ENABLED automatically after enrolment, so an
    employee is recognised immediately with no manual toggle and no restart;
  * the /register endpoint enrols captured frames atomically and rolls back
    completely if none contain a usable face (no faceless employees);
  * a degenerate/bad frame must never crash detection (it used to raise a raw
    cv2 range-check error and 500 the request).
"""

from __future__ import annotations

import cv2
import numpy as np

from conftest import to_data_url


def test_enrolment_auto_enables_recognition(camera_client, astronaut_bgr):
    """Enrolling a face turns recognition on by itself."""
    client, cam_id, cam = camera_client
    # explicitly turn face OFF first to prove enrolment turns it back on
    client.post("/api/ai/models/face/enable", json={"enabled": False})
    assert client.get("/api/ai/status").json()["tasks"]["face"]["enabled"] is False

    eid = client.post("/api/employees", json={"full_name": "Auto On"}).json()["id"]
    res = client.post(f"/api/employees/{eid}/capture", json={"camera_id": cam_id}).json()
    assert res["enrollment"]["ok"] is True

    # recognition is now enabled with no manual step
    assert client.get("/api/ai/status").json()["tasks"]["face"]["enabled"] is True


def test_register_endpoint_atomic_and_recognises_immediately(camera_client, astronaut_bgr):
    client, cam_id, cam = camera_client
    body = {
        "full_name": "Reg Person",
        "employee_code": "R-1",
        "images": [to_data_url(astronaut_bgr), to_data_url(astronaut_bgr)],
    }
    r = client.post("/api/employees/register", json=body)
    assert r.status_code == 201
    emp = r.json()
    assert emp["enrolled"] == 2 and emp["rejected"] == 0
    assert emp["recognition_enabled"] is True

    # the same face is now recognised on the live frame straight away
    cam.buffer.put(astronaut_bgr.copy())
    faces = client.get(f"/api/cameras/{cam_id}/analyze").json()["faces"]
    assert any(f["identity"] == "Reg Person" for f in faces)


def test_register_rolls_back_when_no_face(client):
    """No usable face -> 422 and the employee is NOT created."""
    before = client.get("/api/employees").json()["total"]
    blank = to_data_url(np.zeros((300, 300, 3), dtype=np.uint8))
    r = client.post("/api/employees/register",
                    json={"full_name": "Ghost", "images": [blank]})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "no_valid_face"
    after = client.get("/api/employees").json()["total"]
    assert after == before  # rolled back, no faceless employee left behind


def test_register_rejects_blurry_without_crashing(client, astronaut_bgr):
    blurry = to_data_url(cv2.GaussianBlur(astronaut_bgr, (31, 31), 0))
    r = client.post("/api/employees/register",
                    json={"full_name": "Blur", "images": [blurry]})
    # cleanly rejected (422), never a 500 from a detector crash
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "no_valid_face"


def test_detector_never_crashes_on_bad_frames():
    """detect_faces must swallow OpenCV internal errors and return []."""
    from rtsp_backend.ai.embedders import OpenCVFallbackEmbedder

    emb = OpenCVFallbackEmbedder()
    emb.load()
    # a variety of awkward frames that must not raise
    for frame in (
        np.zeros((0, 0, 3), dtype=np.uint8),
        np.zeros((480, 640, 3), dtype=np.uint8),
        np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8),
        np.full((512, 1024, 3), 200, dtype=np.uint8),
    ):
        assert isinstance(emb.detect_faces(frame), list)
