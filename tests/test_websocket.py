"""WebSocket tests: hello handshake and live ai_event delivery."""

from __future__ import annotations

import time


def test_ws_hello(client):
    with client.websocket_connect("/ws/events") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello"


def test_ws_receives_ai_event_on_recognition(client, astronaut_bgr):
    # enable face recognition and enroll a real face
    client.post("/api/ai/models/face/enable", json={"enabled": True})
    eid = client.post("/api/employees", json={"full_name": "Astro"}).json()["id"]
    from conftest import to_data_url
    res = client.post(f"/api/employees/{eid}/images", json={"image": to_data_url(astronaut_bgr)})
    assert res.json()["enrollment"]["ok"] is True

    pipeline = client.app.state.pipeline

    with client.websocket_connect("/ws/events") as ws:
        assert ws.receive_json()["type"] == "hello"
        # run the pipeline on the enrolled face; this should emit an ai_event
        pipeline.process("cam1", "Front Door", astronaut_bgr, annotate=False, force=True)

        got = None
        deadline = time.time() + 5
        while time.time() < deadline:
            msg = ws.receive_json()
            if msg.get("type") == "ai_event":
                got = msg
                break
        assert got is not None
        assert got["event_type"] == "face_recognized"
        assert got["employee_id"] == eid
        assert got["label"] == "Astro"


def test_ws_emits_unknown_person_event(client, astronaut_bgr):
    # face enabled but nobody enrolled -> unknown person event with a saved snapshot
    client.post("/api/ai/models/face/enable", json={"enabled": True})
    pipeline = client.app.state.pipeline

    with client.websocket_connect("/ws/events") as ws:
        assert ws.receive_json()["type"] == "hello"
        pipeline.process("cam9", "Lobby", astronaut_bgr, annotate=False, force=True)
        got = None
        deadline = time.time() + 5
        while time.time() < deadline:
            msg = ws.receive_json()
            if msg.get("type") == "ai_event":
                got = msg
                break
        assert got is not None
        assert got["event_type"] == "unknown_person"
        assert got["snapshot"]  # snapshot path recorded
