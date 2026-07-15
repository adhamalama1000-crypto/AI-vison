"""
Tests for continuous (background) recognition — registration step 3.

The app's lifespan starts a background worker that runs the AI pipeline on each
camera's latest frame. With a face enrolled and the camera showing that face,
an attendance/recognition event must appear automatically, without anyone
hitting the stream — and must be de-duplicated, not logged every frame.
"""

from __future__ import annotations

import time

from conftest import to_data_url


def _poll_events(client, etype, timeout=8.0):
    deadline = time.time() + timeout
    last = []
    while time.time() < deadline:
        last = client.get(f"/api/events?type={etype}").json()["events"]
        if last:
            return last
        time.sleep(0.25)
    return last


def test_known_face_auto_recognized_and_deduped(camera_client, astronaut_bgr):
    client, cam_id, cam = camera_client
    client.post("/api/ai/models/face/enable", json={"enabled": True})

    # enrol the astronaut as an employee
    eid = client.post("/api/employees", json={"full_name": "Astro Naut"}).json()["id"]
    up = client.post(f"/api/employees/{eid}/images", json={"image": to_data_url(astronaut_bgr)})
    assert up.json()["enrollment"]["ok"] is True

    # keep the same face in front of the camera for a couple of seconds
    end = time.time() + 2.5
    while time.time() < end:
        cam.buffer.put(astronaut_bgr.copy())
        time.sleep(0.1)

    events = _poll_events(client, "face_recognized")
    assert events, "no face_recognized event was logged by the background worker"
    assert events[0]["label"] == "Astro Naut"
    assert events[0]["confidence"] is not None

    # dedup: despite ~25 frames processed, only a handful of events (not one/frame)
    total = client.get("/api/events?type=face_recognized").json()["total"]
    assert total <= 3, f"events were not de-duplicated: {total} logged"

    # the recognized event is the attendance record
    dash = client.get("/api/stats/dashboard").json()
    assert dash["recognition"]["recognized_events"] >= 1


def test_unknown_face_auto_logged(camera_client, astronaut_bgr):
    client, cam_id, cam = camera_client
    client.post("/api/ai/models/face/enable", json={"enabled": True})
    # nobody enrolled -> the visible face is an unknown person
    end = time.time() + 2.0
    while time.time() < end:
        cam.buffer.put(astronaut_bgr.copy())
        time.sleep(0.1)

    events = _poll_events(client, "unknown_person")
    assert events, "no unknown_person event logged"
    assert events[0]["label"] == "Unknown Employee"
    total = client.get("/api/events?type=unknown_person").json()["total"]
    assert total <= 3


def test_worker_idle_when_disabled(camera_client, astronaut_bgr):
    client, cam_id, cam = camera_client
    # face is on by default now; explicitly disable it to test the idle path
    assert client.post("/api/ai/models/face/enable", json={"enabled": False}).status_code == 200
    client.delete("/api/events")
    end = time.time() + 1.5
    while time.time() < end:
        cam.buffer.put(astronaut_bgr.copy())
        time.sleep(0.1)
    assert client.get("/api/events").json()["total"] == 0
