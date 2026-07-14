"""API tests for events, components, wires/topology, stats, and media guards."""

from __future__ import annotations

import time


def _seed_event(client, etype="system_alert", label="test"):
    # insert directly through the app's DB to simulate a logged event
    db = client.app.state.db
    db.insert(
        "INSERT INTO events(type,camera_id,camera_name,label,confidence,created_at) "
        "VALUES(?,?,?,?,?,?)",
        (etype, "cam1", "Front", label, 0.9, time.time()),
    )


def test_events_list_filter_clear(client):
    _seed_event(client, "face_recognized", "Grace")
    _seed_event(client, "unknown_person", "Unknown Person")

    r = client.get("/api/events")
    assert r.status_code == 200
    assert r.json()["total"] == 2

    # filter by type
    r = client.get("/api/events?type=unknown_person")
    assert r.json()["total"] == 1
    assert r.json()["events"][0]["type"] == "unknown_person"

    # types summary
    types = client.get("/api/events/types").json()["types"]
    assert {t["type"] for t in types} == {"face_recognized", "unknown_person"}

    # clear
    assert client.delete("/api/events").status_code == 200
    assert client.get("/api/events").json()["total"] == 0


def test_components_endpoints(client):
    classes = client.get("/api/components/classes").json()["classes"]
    assert "mcb" in classes and "vfd" in classes and len(classes) >= 18

    # with no trained model + no detections, summary is empty (not fabricated)
    assert client.get("/api/components").json()["total"] == 0
    assert client.get("/api/components/summary").json()["summary"] == []


def test_wires_list_empty(client):
    assert client.get("/api/wires").json()["total"] == 0


def test_stats_dashboard_shape(client):
    d = client.get("/api/stats/dashboard").json()
    for key in ("employees", "recognition", "electrical", "cameras", "resources", "ai_tasks"):
        assert key in d
    assert d["cameras"]["total"] == 0  # no cameras configured in tests
    assert d["resources"]["gpu_available"] is False


def test_media_path_traversal_blocked(client):
    # attempts to escape the data dir must 404, not leak files
    r = client.get("/api/media/../../etc/passwd")
    assert r.status_code in (400, 404)
    r = client.get("/api/media/does/not/exist.jpg")
    assert r.status_code == 404
