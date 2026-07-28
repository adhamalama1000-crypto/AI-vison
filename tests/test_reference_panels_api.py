"""
API integration tests for the Reference Panel + Datasheet subsystem.

Exercises the full REST surface (create/list/get/delete, upload, capture from an
RTSP camera buffer, learn, compare, results) plus datasheet upload/extract,
end to end through the FastAPI app — and asserts the existing employee/face and
camera systems are untouched.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest


def make_panel(shift=0, drop_green=False):
    img = np.full((400, 600, 3), 60, np.uint8)
    cv2.rectangle(img, (40, 340), (560, 370), (170, 170, 170), -1)
    cv2.line(img, (100 + shift, 100), (100 + shift, 345), (0, 0, 200), 4)
    if not drop_green:
        cv2.line(img, (200, 120), (400, 120), (0, 180, 0), 4)
    cv2.line(img, (300, 150), (300, 345), (200, 120, 0), 4)
    cv2.line(img, (430, 150), (520, 300), (0, 220, 220), 3)
    return img


def jpg(img):
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


# --------------------------------------------------------------------------
# CRUD
# --------------------------------------------------------------------------

def test_create_list_get_delete_panel(client):
    r = client.post("/api/reference-panels", json={"name": "P1", "version": "v2",
                                                    "description": "d"})
    assert r.status_code == 201
    pid = r.json()["id"]
    assert r.json()["status"] == "draft"

    r = client.get("/api/reference-panels")
    assert r.status_code == 200
    assert r.json()["total"] == 1

    r = client.get(f"/api/reference-panels/{pid}")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "P1" and body["version"] == "v2"
    assert body["images"] == [] and body["components"] == []

    assert client.delete(f"/api/reference-panels/{pid}").status_code == 200
    assert client.get("/api/reference-panels").json()["total"] == 0


def test_get_missing_panel_404(client):
    assert client.get("/api/reference-panels/999").status_code == 404


# --------------------------------------------------------------------------
# upload + learn + compare
# --------------------------------------------------------------------------

def test_upload_learn_compare_flow(client, monkeypatch):
    # The reference-panel wire diff is an experimental, opt-in feature since the
    # classical tracer's false-positive rate made it unfit for default use.
    monkeypatch.setenv("RTSP_ENABLE_WIRE_TRACING", "1")
    pid = client.post("/api/reference-panels", json={"name": "Main"}).json()["id"]

    # upload two images
    files = [("files", ("a.jpg", jpg(make_panel()), "image/jpeg")),
             ("files", ("b.jpg", jpg(make_panel()), "image/jpeg"))]
    r = client.post(f"/api/reference-panels/{pid}/upload", files=files)
    assert r.status_code == 200 and r.json()["count"] == 2

    # cannot compare before learning
    r = client.post(f"/api/reference-panels/{pid}/compare",
                    files={"file": ("o.jpg", jpg(make_panel()), "image/jpeg")})
    assert r.status_code == 409

    # learn
    r = client.post(f"/api/reference-panels/{pid}/learn")
    assert r.status_code == 200
    learned = r.json()
    assert learned["status"] == "ready"
    assert learned["n_wires"] >= 3
    assert learned["n_terminals"] >= 1
    assert learned["graph"]["edges"]

    # compare identical -> pass/warning
    r = client.post(f"/api/reference-panels/{pid}/compare",
                    files={"file": ("o.jpg", jpg(make_panel()), "image/jpeg")})
    assert r.status_code == 200
    res = r.json()
    assert res["status"] in ("pass", "warning")
    assert res["snapshot"]
    assert 0.0 <= res["score"] <= 1.0

    # compare with a wire removed -> fail + missing_wire
    r = client.post(f"/api/reference-panels/{pid}/compare",
                    files={"file": ("o.jpg", jpg(make_panel(drop_green=True)), "image/jpeg")})
    res = r.json()
    assert res["status"] == "fail"
    assert any(e["error_type"] == "missing_wire" for e in res["errors"])

    # latest result endpoint + history
    r = client.get(f"/api/reference-panels/{pid}/result")
    assert r.status_code == 200 and r.json()["status"] == "fail"
    r = client.get(f"/api/reference-panels/{pid}/results")
    assert r.json()["total"] == 2

    # the annotated overlay is servable media
    assert client.get(f"/api/media/{res['snapshot']}").status_code == 200


def test_learn_without_images_400(client):
    pid = client.post("/api/reference-panels", json={"name": "Empty"}).json()["id"]
    assert client.post(f"/api/reference-panels/{pid}/learn").status_code == 400


# --------------------------------------------------------------------------
# capture from an RTSP camera buffer (no second connection)
# --------------------------------------------------------------------------

def test_capture_from_camera(client):
    from rtsp_backend.config import CameraConfig
    app = client.app
    cam = app.state.manager.add_camera(
        CameraConfig(id="cam1", name="Cam", url="rtsp://example/s"),
        start=False, emit=False)
    cam.buffer.put(make_panel().copy())

    pid = client.post("/api/reference-panels", json={"name": "Cap"}).json()["id"]
    r = client.post(f"/api/reference-panels/{pid}/capture", data={"camera_id": "cam1"})
    assert r.status_code == 200
    assert r.json()["source"] == "camera"
    assert client.get(f"/api/reference-panels/{pid}").json()["n_images"] == 1


# --------------------------------------------------------------------------
# datasheets
# --------------------------------------------------------------------------

def test_datasheet_upload_extract_delete(client):
    txt = b"Q1 -> KM1. X1:1 - K1. Fuse F2. Selector S1. Lamp H1. Wire W12."
    r = client.post("/api/datasheets/upload",
                    files={"file": ("sld.txt", txt, "text/plain")})
    assert r.status_code == 200
    dsid = r.json()["id"]

    r = client.post(f"/api/datasheets/{dsid}/extract")
    assert r.status_code == 200
    body = r.json()
    assert "Q1" in body["parsed"]["component_ids"]
    assert body["expected_graph"]["node_count"] >= 1

    assert client.get("/api/datasheets").json()["total"] == 1
    assert client.get(f"/api/datasheets/{dsid}").json()["status"] == "extracted"
    assert client.delete(f"/api/datasheets/{dsid}").status_code == 200


# --------------------------------------------------------------------------
# non-regression: existing systems still respond
# --------------------------------------------------------------------------

def test_existing_endpoints_untouched(client):
    assert client.get("/health").status_code == 200
    assert client.get("/api/employees").status_code == 200
    assert client.get("/api/reference").status_code == 200      # old reference designs
    assert client.get("/api/panels").status_code == 200
    assert client.get("/api/inspection").status_code == 200
    # advanced wire backend is registered and is the default
    cat = client.get("/api/ai/status").json()["catalog"]["wires"]
    ids = {b["backend_id"] for b in cat}
    assert {"advanced_wires", "classical_wires", "null_wires"} <= ids
