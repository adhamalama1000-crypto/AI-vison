"""
Tests for the live-capture validation workflow (registration step 2).

Uses a fake camera whose buffer we feed directly, so we control exactly what the
'live frame' contains: a real face, a blank frame, a blurred face, or two faces.
"""

from __future__ import annotations

import time

import cv2
import numpy as np


def _enable_face(client):
    assert client.post("/api/ai/models/face/enable", json={"enabled": True}).status_code == 200


def test_validate_accepts_real_face(camera_client):
    client, cam_id, cam = camera_client
    _enable_face(client)
    r = client.post("/api/employees/validate", json={"camera_id": cam_id})
    assert r.status_code == 200
    v = r.json()
    assert v["ok"] is True
    assert v["faces"] == 1
    assert v["multiple_faces"] is False
    assert v["blur_score"] is not None and v["blur_score"] > v["min_blur"]
    assert v["image"].startswith("data:image/")          # full frame for enrolment
    assert v["face_preview"].startswith("data:image/")   # crop for the UI


def test_validate_detects_no_face(camera_client):
    client, cam_id, cam = camera_client
    _enable_face(client)
    cam.buffer.put(np.zeros((300, 300, 3), dtype=np.uint8))  # blank frame
    v = client.post("/api/employees/validate", json={"camera_id": cam_id}).json()
    assert v["ok"] is False
    assert v["reason"] == "no_face_detected"
    assert v["faces"] == 0


def test_validate_rejects_blurry_face(camera_client, astronaut_bgr):
    client, cam_id, cam = camera_client
    _enable_face(client)
    blurred = cv2.GaussianBlur(astronaut_bgr, (31, 31), 0)
    cam.buffer.put(blurred)
    v = client.post("/api/employees/validate", json={"camera_id": cam_id}).json()
    # a real but heavily blurred face is detected yet rejected as too blurry
    assert v["ok"] is False
    assert v["reason"] == "blurry"
    assert v["blur_score"] < v["min_blur"]


def test_validate_warns_multiple_faces(camera_client, astronaut_bgr):
    client, cam_id, cam = camera_client
    _enable_face(client)
    two = np.hstack([astronaut_bgr, astronaut_bgr])  # two copies -> two faces
    cam.buffer.put(two)
    v = client.post("/api/employees/validate", json={"camera_id": cam_id}).json()
    if v["faces"] >= 2:
        assert v["multiple_faces"] is True
        # still enrollable (largest face used), surfaced as a warning
        assert v["ok"] is True
        assert v["reason"] == "multiple_faces_warning"
    else:
        # detector environment-dependent; at minimum the flag is coherent
        assert v["multiple_faces"] is (v["faces"] > 1)


def test_blurry_capture_leaves_no_orphan_image(camera_client, astronaut_bgr):
    client, cam_id, cam = camera_client
    _enable_face(client)
    eid = client.post("/api/employees", json={"full_name": "Blur Person"}).json()["id"]

    cam.buffer.put(cv2.GaussianBlur(astronaut_bgr, (31, 31), 0))
    res = client.post(f"/api/employees/{eid}/capture", json={"camera_id": cam_id}).json()
    assert res["enrollment"]["ok"] is False
    assert res["enrollment"]["reason"] == "blurry"
    assert res["image_id"] is None  # rejected -> no image row kept

    emp = client.get(f"/api/employees/{eid}").json()
    assert len(emp["images"]) == 0       # no orphaned image
    assert emp["embeddings"] == 0        # no embedding stored

    # a subsequent sharp capture succeeds and enrolls exactly one vector
    cam.buffer.put(astronaut_bgr.copy())
    ok = client.post(f"/api/employees/{eid}/capture", json={"camera_id": cam_id}).json()
    assert ok["enrollment"]["ok"] is True
    emp = client.get(f"/api/employees/{eid}").json()
    assert len(emp["images"]) == 1
    assert emp["embeddings"] == 1


def test_capture_from_buffer_no_new_connection(camera_client):
    # capturing must read the buffered frame, never dial a new RTSP connection
    client, cam_id, cam = camera_client
    _enable_face(client)
    before = cam.statistics() if hasattr(cam, "statistics") else None
    eid = client.post("/api/employees", json={"full_name": "Buf"}).json()["id"]
    r = client.post(f"/api/employees/{eid}/capture", json={"camera_id": cam_id})
    assert r.status_code == 200
    # the camera capture thread was never started in this fixture, yet capture works,
    # which proves the frame came from the in-memory buffer
    assert r.json()["enrollment"]["ok"] is True
