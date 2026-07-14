"""
End-to-end runtime checks that the new latency metrics endpoint and the module
event plumbing actually run inside the live app (not just import).

A real face frame is injected into a camera buffer (no RTSP needed); we then
drive the AI pipeline through the public HTTP surface.
"""

from __future__ import annotations


def test_metrics_endpoint_shape(camera_client):
    c, cam_id, cam = camera_client
    r = c.get("/api/metrics")
    assert r.status_code == 200
    m = r.json()
    assert "cameras" in m and "ai" in m and "resources" in m
    cam0 = next(x for x in m["cameras"] if x["id"] == cam_id)
    # the frame we injected has an age -> stream latency is reported
    assert "stream_latency_ms" in cam0
    assert "encode_ms" in cam0
    # every registered task shows up with fps + inference fields
    for task in ("detection", "face", "fire", "weapon", "fall", "human"):
        assert task in m["ai"]
        assert "ai_fps" in m["ai"][task]
        assert "inference_ms" in m["ai"][task]


def test_enable_new_module_and_analyze_runs(camera_client):
    c, cam_id, cam = camera_client
    # enable a couple of the new modules (no weights -> honest empty output)
    for task in ("human", "weapon", "fall"):
        r = c.post(f"/api/ai/models/{task}/enable", json={"enabled": True})
        assert r.status_code == 200

    # analyze the current frame through the full pipeline
    r = c.get(f"/api/cameras/{cam_id}/analyze")
    assert r.status_code == 200
    res = r.json()
    # face task is on by default and the astronaut face is present
    assert "faces" in res
    # Honest contract: a module enabled WITHOUT weights cannot run, so it is
    # skipped entirely — no key and definitely no fabricated detections.
    for task in ("human", "weapon", "fall"):
        assert res.get(task, []) == []


def test_module_status_reports_model_unavailable(camera_client):
    c, _, _ = camera_client
    status = c.get("/api/ai/status").json()
    fire = status["tasks"]["fire"]
    # honest: weights are missing, so the task is not runnable and says why
    assert fire["reason"] == "weights_missing"
    assert fire["state"] in ("not_loaded", "disabled")
