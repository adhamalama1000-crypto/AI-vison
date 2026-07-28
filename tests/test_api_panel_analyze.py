"""
``POST /api/panel/analyze`` — the component-detection API contract.

No trained checkpoint ships with the repository, so what these tests pin down is
the contract and the honesty of the empty case: the response shape the brief
specifies, and the guarantee that a platform with no model returns an empty
component list plus the reason rather than inventing plausible detections to look
functional. That second property is the one worth protecting with tests — an
endpoint that fabricates output is worse than one that returns nothing, because
nobody goes looking for a bug in a system that appears to work.

The additive-router property is also asserted: ``/api/panels/analyze`` (plural)
must keep working exactly as before, because the dashboard and the
reference-panel flow depend on its richer response.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest

from rtsp_backend.electrical import taxonomy as tax


def _panel_jpeg(width: int = 320, height: int = 240) -> bytes:
    """A plausible panel-ish image: grey cabinet with a row of dark modules."""
    img = np.full((height, width, 3), 120, np.uint8)
    for i in range(6):
        x = 20 + i * 45
        cv2.rectangle(img, (x, 80), (x + 34, 150), (40, 40, 45), -1)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _post(client, **params):
    files = {"file": ("panel.jpg", io.BytesIO(_panel_jpeg()), "image/jpeg")}
    return client.post("/api/panel/analyze", files=files, params=params)


# ==========================================================================
# the contract
# ==========================================================================

def test_analyze_returns_the_specified_response_shape(client):
    r = _post(client)
    assert r.status_code == 200, r.text
    body = r.json()

    # The brief's contract: components[] with class, confidence, bbox.
    assert isinstance(body["components"], list)
    for c in body["components"]:
        assert set(("class", "confidence", "bbox")).issubset(c)
        assert isinstance(c["bbox"], list) and len(c["bbox"]) == 4
        assert 0.0 <= c["confidence"] <= 1.0
        assert c["class"] in set(tax.CLASS_ORDER) | {tax.UNKNOWN_COMPONENT_ID}

    # And the metadata a client needs to interpret bbox without guessing.
    assert body["bbox_format"] == "xyxy_absolute_pixels"
    assert body["image"]["width"] == 320 and body["image"]["height"] == 240
    assert body["image"]["source"] == "upload"
    assert "loaded" in body["model"]


def test_analyze_is_honest_when_no_model_is_loaded(client):
    """No checkpoint must mean an empty list plus the reason — never fake boxes."""
    body = _post(client).json()
    if body["model"]["loaded"]:
        pytest.skip("a trained component model is installed in this environment")

    assert body["components"] == []
    assert body["component_total"] == 0
    joined = " ".join(body["notes"]).lower()
    assert "no trained component model" in joined
    assert "honest empty result" in joined


def test_analyze_includes_the_panel_report_by_default(client):
    body = _post(client).json()
    report = body["report"]
    # The five report sections the brief asks for.
    assert isinstance(report["detected_components"], list)
    assert isinstance(report["missing_components"], list)
    assert "count" in report["unknown_components"]
    assert "mean" in report["confidence"]
    assert "component_total" in report
    # 'annotated_image' is a media path the client fetches from /media.
    assert "annotated_image" in body


def test_report_can_be_switched_off_for_the_bare_list(client):
    body = _post(client, report=False).json()
    assert "report" not in body
    assert "components" in body


def test_annotation_can_be_switched_off(client):
    body = _post(client, annotate=False).json()
    assert "annotated_image" not in body


def test_unknown_components_are_reported_not_guessed(client):
    """Every unknown must be flagged in-band and counted in the report."""
    body = _post(client).json()
    unknown_in_list = [c for c in body["components"] if c["is_unknown"]]
    assert all(c["class"] == tax.UNKNOWN_COMPONENT_ID for c in unknown_in_list)
    assert body["report"]["unknown_components"]["count"] == len(unknown_in_list)
    assert "guessed" in body["report"]["unknown_components"]["note"]


def test_min_confidence_only_tightens_the_result(client):
    loose = _post(client, min_confidence=0.0).json()["components"]
    strict = _post(client, min_confidence=0.99).json()["components"]
    assert len(strict) <= len(loose)
    assert all(c["confidence"] >= 0.99 for c in strict)


def test_analyze_rejects_a_non_image_upload(client):
    files = {"file": ("notes.txt", io.BytesIO(b"this is not an image"),
                      "text/plain")}
    r = client.post("/api/panel/analyze", files=files)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "bad_image"


def test_analyze_without_a_file_or_camera_fails_cleanly(client):
    """No upload and no camera is a 4xx/5xx with a reason, never a 500 traceback."""
    r = client.post("/api/panel/analyze")
    assert r.status_code in (400, 404, 503)
    assert "error" in r.json()


def test_analyze_persists_a_report_row(client):
    body = _post(client).json()
    assert "id" in body
    rows = client.app.state.db.query(
        "SELECT * FROM reports WHERE kind='panel_analysis'")
    assert len(rows) == 1
    assert rows[0]["title"] == "Component Detection"


def test_persist_can_be_switched_off(client):
    body = _post(client, persist=False).json()
    assert "id" not in body
    rows = client.app.state.db.query(
        "SELECT * FROM reports WHERE kind='panel_analysis'")
    assert rows == [] or len(rows) == 0


# ==========================================================================
# supporting endpoints
# ==========================================================================

def test_classes_endpoint_exposes_the_full_label_space(client):
    body = client.get("/api/panel/classes").json()
    assert body["class_count"] == len(tax.CLASS_ORDER)
    assert body["unknown_class"] == tax.UNKNOWN_COMPONENT_ID
    ids = [c["class"] for c in body["classes"]]
    assert ids == list(tax.CLASS_ORDER)
    # The brief's target classes must all be addressable through the API.
    for cid in ("mcb", "mccb", "contactor", "relay", "plc", "power_supply",
                "vfd", "fuse", "terminal_block", "busbar", "push_button",
                "emergency_stop", "selector_switch", "indicator_lamp",
                "transformer", "current_transformer", "circuit_breaker",
                "timer_relay", "overload_relay", "din_rail", "wire_duct",
                "cooling_fan"):
        assert cid in ids, cid
    for c in body["classes"]:
        assert c["name"] and c["category"] and 0.0 < c["min_confidence"] <= 1.0


def test_model_endpoint_reports_status_and_the_remedy(client):
    body = client.get("/api/panel/model").json()
    assert "loaded" in body
    if not body["loaded"]:
        assert body["remedy"] and "models/components" in body["remedy"]
    else:
        assert body["remedy"] is None


# ==========================================================================
# the existing endpoint must be untouched
# ==========================================================================

def test_the_plural_panels_endpoint_still_works(client):
    """Additive change: /api/panels/analyze keeps its own richer response."""
    files = {"file": ("panel.jpg", io.BytesIO(_panel_jpeg()), "image/jpeg")}
    r = client.post("/api/panels/analyze", files=files,
                    data={"make_pdf": "false"})
    assert r.status_code == 200, r.text
    body = r.json()
    # Keys the dashboard and reference-panel flow read.
    for key in ("id", "result", "component_total", "component_counts",
                "wire_total", "panel_type"):
        assert key in body, key
    assert "report" in body["result"]


def test_both_endpoints_agree_on_the_component_count(client):
    """One detection path, not two that can drift apart."""
    a = _post(client, persist=False).json()
    files = {"file": ("panel.jpg", io.BytesIO(_panel_jpeg()), "image/jpeg")}
    b = client.post("/api/panels/analyze", files=files,
                    data={"make_pdf": "false"}).json()
    assert a["component_total"] == b["component_total"]
