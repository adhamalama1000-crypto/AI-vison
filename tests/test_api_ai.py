"""API tests for the AI model manager, metrics, and settings persistence."""

from __future__ import annotations

from rtsp_backend.app import build_app


def test_ai_status_and_catalog(client):
    r = client.get("/api/ai/status")
    assert r.status_code == 200
    data = r.json()
    tasks = set(data["tasks"].keys())
    # core tasks are always present
    assert {"detection", "face", "components", "wires"} <= tasks
    # additional detection modules are registered as their own tasks
    assert {"fire", "violence", "fall", "weapon", "ppe", "human", "vehicle"} <= tasks
    assert "resources" in data
    assert data["resources"]["gpu_available"] is False

    cat = client.get("/api/ai/catalog").json()
    assert "face" in cat
    assert "fire" in cat and "weapon" in cat


def test_enable_select_params_flow(client):
    # enable + disable
    assert client.post("/api/ai/models/face/enable", json={"enabled": True}).json()["enabled"] is True
    assert client.post("/api/ai/models/face/enable", json={"enabled": False}).json()["enabled"] is False

    # select a different wire backend
    r = client.post("/api/ai/models/wires/select", json={"backend_id": "null_wires"})
    assert r.status_code == 200
    assert r.json()["selected_backend"] == "null_wires"

    # update params
    r = client.post("/api/ai/models/detection/params",
                    json={"params": {"conf": 0.4, "iou": 0.5, "device": "cpu"}})
    assert r.status_code == 200
    assert r.json()["backend"]["params"]["conf"] == 0.4

    # bad task + bad backend are handled
    assert client.post("/api/ai/models/nope/enable", json={"enabled": True}).status_code == 404
    assert client.post("/api/ai/models/face/select",
                       json={"backend_id": "does_not_exist"}).status_code == 400


def test_metrics_endpoint(client):
    m = client.get("/api/ai/metrics").json()
    assert "resources" in m and "tasks" in m
    for t in ("detection", "face", "components", "wires"):
        assert t in m["tasks"]
        assert "fps" in m["tasks"][t]


def test_settings_persist_across_restart(temp_settings):
    from fastapi.testclient import TestClient

    # write a setting + change model config on one app instance
    with TestClient(build_app(temp_settings)) as c:
        assert c.put("/api/ai/settings/overlay_opacity", json={"value": 0.7}).status_code == 200
        assert c.post("/api/ai/models/face/enable", json={"enabled": True}).status_code == 200
        assert c.post("/api/ai/models/wires/select",
                      json={"backend_id": "null_wires"}).status_code == 200

    # a fresh app instance backed by the same DB restores everything
    with TestClient(build_app(temp_settings)) as c2:
        assert c2.get("/api/ai/settings/overlay_opacity").json()["value"] == 0.7
        status = c2.get("/api/ai/status").json()
        assert status["tasks"]["face"]["enabled"] is True
        assert status["tasks"]["wires"]["selected_backend"] == "null_wires"


def test_status_state_vocabulary_and_reasons(client):
    st = client.get("/api/ai/status").json()["tasks"]
    # tasks needing weights report the precise reason, never a silent failure
    assert st["detection"]["state"] == "not_loaded"
    assert st["detection"]["reason"] == "weights_missing"
    assert st["components"]["state"] == "not_loaded"
    assert st["components"]["reason"] == "weights_missing"
    # wires stays opt-in (weightless but disabled); face is on by default so
    # a freshly enrolled employee is recognised immediately without a toggle.
    assert st["wires"]["state"] == "disabled"
    assert st["face"]["state"] in ("loaded", "running")
    assert st["face"]["enabled"] is True
    assert st["face"]["backend"]["ready"] is True

    # enabling face loads the tested fallback embedder
    r = client.post("/api/ai/models/face/enable", json={"enabled": True}).json()
    assert r["state"] in ("loaded", "running")
    assert r["backend"]["ready"] is True

    # Selecting the real InsightFace (SCRFD + ArcFace) backend is honest about
    # its outcome. Where the verified weights are available it loads and runs;
    # where they cannot be obtained (library missing / download blocked) it
    # reports an error state with a machine-readable reason and detail. It must
    # NEVER silently degrade to the weak fallback — so the outcome is one of
    # those two explicit states, never a fake "ready".
    r = client.post("/api/ai/models/face/select",
                    json={"backend_id": "insightface_arcface",
                          "auto_install": False}).json()
    assert r["selected_backend"] == "insightface_arcface"
    if r["state"] == "error":
        assert r["reason"] in ("insightface_missing", "weights_unavailable",
                               "init_failed", "onnxruntime_missing")
        assert r["detail"]
    else:
        # real backend actually loaded — verify it reports as ready
        assert r["state"] in ("loaded", "running")
        assert r["backend"]["ready"] is True
        assert r["backend"]["params"].get("model_pack") in ("buffalo_l", "buffalo_s")

    # metrics endpoint carries the same state so the UI can live-update
    m = client.get("/api/ai/metrics").json()
    assert m["tasks"]["face"]["state"] == r["state"]
    assert m["tasks"]["detection"]["reason"] == "weights_missing"


def test_duplicate_employee_code_rejected(client):
    assert client.post("/api/employees",
                       json={"full_name": "A", "employee_code": "E-1"}).status_code == 201
    dup = client.post("/api/employees", json={"full_name": "B", "employee_code": "E-1"})
    assert dup.status_code == 409
    # the second insert did not create a row
    assert client.get("/api/employees").json()["total"] == 1
