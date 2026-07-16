"""
Regression tests for the AI-module fixes:

* Face backend auto-fallback (InsightFace missing / persisted but unavailable
  -> weights-free OpenCV backend, so recognition + enrolment keep working).
* Analyze never returns a bare 500 — a broken face model becomes a structured
  ``face_error`` in a 200 response.
* Component ONNX decode / NMS / class-mapping (synthetic YOLOv8-shaped model).
* Wire detector rejects text / borders / noise but keeps real wires.
* Full pipeline is exception-safe.
"""

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
import pytest

from rtsp_backend.ai import manager as mgr
from rtsp_backend.db import Database


# --------------------------------------------------------------------------
# face backend fallback
# --------------------------------------------------------------------------

def test_insightface_availability_probe_is_bool():
    assert isinstance(mgr._insightface_available(), bool)


def test_face_backend_resolves_to_fallback_without_insightface(monkeypatch):
    monkeypatch.delenv("RTSP_FACE_BACKEND", raising=False)
    if mgr._insightface_available():
        pytest.skip("InsightFace is installed in this environment")
    assert mgr._resolve_face_backend() == "opencv_fallback"


def test_explicit_override_is_respected(monkeypatch):
    monkeypatch.setenv("RTSP_FACE_BACKEND", "opencv_fallback")
    assert mgr._resolve_face_backend() == "opencv_fallback"


def test_persisted_insightface_config_falls_back(monkeypatch, tmp_path):
    """A DB that persisted 'insightface_arcface' must not leave face dead when
    InsightFace is unavailable — the manager falls back to opencv_fallback."""
    if mgr._insightface_available():
        pytest.skip("InsightFace is installed in this environment")
    monkeypatch.delenv("RTSP_FACE_BACKEND", raising=False)  # allow auto-fallback
    db = Database(str(tmp_path / "t.db"))
    db.execute(
        "INSERT INTO model_config(name,backend,enabled,params,updated_at) "
        "VALUES('face','insightface_arcface',1,'{}',0)")
    ai = mgr.AIModelManager(db, models_dir=str(tmp_path / "models"))
    st = ai.task_status("face")
    assert st["selected_backend"] == "opencv_fallback"
    assert ai.face_service is not None and ai.face_service.embedder.ready
    assert "fell back" in (st["last_error"] or "")
    db.close()


# --------------------------------------------------------------------------
# analyze never 500s, even with a broken face model
# --------------------------------------------------------------------------

def test_analyze_survives_broken_face_model(camera_client):
    c, cam_id, cam = camera_client
    # enable face, then sabotage the embedder so recognize_frame would raise
    c.post("/api/ai/models/face/enable", json={"enabled": True})
    fs = c.app.state.ai.face_service

    def boom(*_a, **_k):
        raise RuntimeError("simulated model failure")

    fs.recognize_frame = boom  # type: ignore
    r = c.get(f"/api/cameras/{cam_id}/analyze")
    assert r.status_code == 200                     # NOT a 500
    body = r.json()
    assert "face_error" in body
    assert "simulated model failure" in body["face_error"]


def test_analyze_recognizes_enrolled_employee(camera_client):
    import base64
    c, cam_id, cam = camera_client
    from skimage import data
    face = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    cam.buffer.put(face.copy())
    durl = "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", face)[1].tobytes()).decode()
    r = c.post("/api/employees/register", json={"full_name": "Grace", "images": [durl]})
    assert r.status_code == 201
    c.post("/api/ai/models/face/enable", json={"enabled": True})
    r = c.get(f"/api/cameras/{cam_id}/analyze")
    assert r.status_code == 200
    labels = [f["label"] for f in r.json().get("faces", [])]
    assert "Grace" in labels


# --------------------------------------------------------------------------
# component detection ONNX decode
# --------------------------------------------------------------------------

def _build_synthetic_yolo_onnx(path: str, nc: int = 27):
    onnx = pytest.importorskip("onnx")
    from onnx import helper, TensorProto
    out = np.zeros((1, 4 + nc, 8400), dtype=np.float32)
    out[0, 0, 0], out[0, 1, 0], out[0, 2, 0], out[0, 3, 0] = 320, 320, 100, 100
    out[0, 4 + 0, 0] = 0.9  # class 0 score
    nodes = [
        helper.make_node("Constant", [], ["raw"],
                         value=helper.make_tensor("v", TensorProto.FLOAT, out.shape, out.flatten())),
        helper.make_node("ReduceMean", ["images"], ["m"], keepdims=0),
        helper.make_node("Mul", ["m", "zero"], ["mz"]),
        helper.make_node("Add", ["mz", "one"], ["scale"]),
        helper.make_node("Mul", ["raw", "scale"], ["output"]),
    ]
    g = helper.make_graph(
        nodes, "m",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4 + nc, 8400])],
        [helper.make_tensor("zero", TensorProto.FLOAT, [], [0.0]),
         helper.make_tensor("one", TensorProto.FLOAT, [], [1.0])])
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    onnx.save(model, path)


def test_component_onnx_full_decode(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    md = tmp_path / "models" / "components"
    md.mkdir(parents=True)
    _build_synthetic_yolo_onnx(str(md / "test.onnx"))
    from rtsp_backend.ai.components import OnnxComponentDetector
    det = OnnxComponentDetector(models_dir=str(tmp_path / "models"), conf=0.25, iou=0.45)
    det.load()
    assert det.ready and det._imgsz == 640
    assert det.class_names[0] == "mcb"
    dets = det.infer(np.zeros((480, 640, 3), np.uint8))
    assert len(dets) == 1
    d = dets[0]
    assert d.label == "mcb" and d.confidence > 0.8
    # letterboxed center (320,320) size 100 -> ~[270,190,370,290]
    x1, y1, x2, y2 = d.bbox.as_list()
    assert abs(x1 - 270) < 6 and abs(x2 - 370) < 6


def test_component_backend_reports_missing_weights(tmp_path):
    from rtsp_backend.ai.components import OnnxComponentDetector
    det = OnnxComponentDetector(models_dir=str(tmp_path / "empty"))
    with pytest.raises(Exception):
        det.load()
    assert det.status()["reason"] == "weights_missing"
    assert not det.ready


# --------------------------------------------------------------------------
# wire artifact rejection
# --------------------------------------------------------------------------

def _panel():
    img = np.full((400, 600, 3), 60, np.uint8)
    cv2.rectangle(img, (40, 340), (560, 370), (170, 170, 170), -1)
    cv2.line(img, (100, 100), (100, 345), (0, 0, 200), 4)
    cv2.line(img, (200, 120), (400, 120), (0, 180, 0), 4)
    cv2.line(img, (300, 150), (300, 345), (200, 120, 0), 4)
    return img


def test_wire_keeps_real_rejects_artifacts():
    from rtsp_backend.panels import wire_detector
    clean = wire_detector.detect_wires(_panel())
    colors = {w.color for w in clean}
    assert {"red", "green", "blue"} <= colors
    assert len(clean) <= 4

    img = _panel()
    cv2.putText(img, "K1 CONTACTOR X1 X2", (360, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.rectangle(img, (2, 2), (597, 397), (40, 40, 40), 3)   # panel border
    for cx in range(60, 540, 40):
        cv2.circle(img, (cx, 355), 5, (90, 90, 90), -1)        # screws
    wires = wire_detector.detect_wires(img)
    # the full-width panel border (~599px) must never be reported as a wire
    assert all(w.length < 560 for w in wires)
    assert {"red", "green", "blue"} <= {w.color for w in wires}
    assert len(wires) <= 6


def test_wire_text_page_is_empty():
    from rtsp_backend.panels import wire_detector
    txt = np.full((400, 600, 3), 240, np.uint8)
    for i, line in enumerate(["Terminal block X1", "Wire label K1 KM2", "Notes abc def"]):
        cv2.putText(txt, line, (20, 60 + i * 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    assert len(wire_detector.detect_wires(txt)) <= 1


def test_wire_flat_image_empty():
    from rtsp_backend.panels import wire_detector
    assert wire_detector.detect_wires(np.full((400, 600, 3), 120, np.uint8)) == []


# --------------------------------------------------------------------------
# pipeline robustness
# --------------------------------------------------------------------------

def test_pipeline_process_is_exception_safe(camera_client):
    c, cam_id, cam = camera_client
    for task in ("face", "components", "wires", "detection"):
        c.post(f"/api/ai/models/{task}/enable", json={"enabled": True})
    # analyze must return 200 with a structured body, never a 500
    r = c.get(f"/api/cameras/{cam_id}/analyze")
    assert r.status_code == 200
    body = r.json()
    for key in ("faces", "objects", "components", "wires"):
        assert key in body
