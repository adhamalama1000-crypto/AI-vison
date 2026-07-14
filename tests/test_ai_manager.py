"""Tests for the plugin registry and AI model manager."""

from __future__ import annotations

import numpy as np

from rtsp_backend.ai import registry
from rtsp_backend.ai.manager import AIModelManager, TASKS
from rtsp_backend.db import Database


def test_registry_has_expected_backends():
    cat = registry.catalog()
    assert set(cat.keys()) == set(TASKS)
    assert any(b["backend_id"] == "opencv_fallback" for b in cat["face"])
    assert any(b["backend_id"] == "insightface_arcface" for b in cat["face"])
    assert any(b["backend_id"] == "onnx_yolo" for b in cat["detection"])
    assert any(b["backend_id"] == "onnx_components" for b in cat["components"])
    assert any(b["backend_id"] == "classical_wires" for b in cat["wires"])


def test_manager_defaults_and_face_service(tmp_path):
    db = Database(str(tmp_path / "m.db"))
    try:
        mgr = AIModelManager(db, models_dir=str(tmp_path / "models"))
        assert mgr.face_service is not None
        for t in TASKS:
            st = mgr.task_status(t)
            assert st["selected_backend"]
            # Face recognition is on by default so an enrolled employee is
            # recognised immediately; the other tasks stay opt-in.
            assert st["enabled"] is (t == "face")
    finally:
        db.close()


def test_enable_disable_and_select_persist(tmp_path):
    dbpath = str(tmp_path / "m.db")
    db = Database(dbpath)
    try:
        mgr = AIModelManager(db, models_dir=str(tmp_path / "models"))
        mgr.set_enabled("face", True)
        assert mgr.is_enabled("face") is True
        mgr.select("wires", "null_wires")
        mgr.update_params("face", {"threshold": 0.66})
        assert mgr.backend("wires").backend_id == "null_wires"
    finally:
        db.close()

    # new manager on same DB restores the persisted config
    db2 = Database(dbpath)
    try:
        mgr2 = AIModelManager(db2, models_dir=str(tmp_path / "models"))
        assert mgr2.task_status("face")["enabled"] is True
        assert mgr2.backend("wires").backend_id == "null_wires"
        assert abs(float(mgr2.face_service.threshold) - 0.66) < 1e-6
    finally:
        db2.close()


def test_component_backend_without_weights_is_honest(tmp_path):
    db = Database(str(tmp_path / "m.db"))
    try:
        mgr = AIModelManager(db, models_dir=str(tmp_path / "models"))
        mgr.set_enabled("components", True)
        st = mgr.task_status("components")
        # no weights present -> not ready, reports why, never fabricates
        assert st["backend"]["ready"] is False
        assert st["backend"]["requires_weights"] is True
        # running inference (if attempted) yields empty, not fake boxes
        backend = mgr.backend("components")
        try:
            out = backend.infer(np.zeros((64, 64, 3), dtype=np.uint8))
            assert out == []
        except RuntimeError:
            pass  # load() raising because no weights is also acceptable/honest
    finally:
        db.close()


def test_wire_baseline_returns_unknown_status(tmp_path):
    db = Database(str(tmp_path / "m.db"))
    try:
        mgr = AIModelManager(db, models_dir=str(tmp_path / "models"))
        mgr.select("wires", "classical_wires")
        mgr.set_enabled("wires", True)
        wa = mgr.backend("wires")
        # an image with strong lines yields segments, all flagged 'unknown'
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        import cv2
        cv2.line(img, (10, 40), (190, 44), (255, 255, 255), 2)
        cv2.line(img, (10, 120), (190, 118), (200, 200, 200), 2)
        wires = wa.analyze(img, [])
        for w in wires:
            assert w.status == "unknown"
    finally:
        db.close()


def test_resource_metrics_shape(tmp_path):
    db = Database(str(tmp_path / "m.db"))
    try:
        mgr = AIModelManager(db, models_dir=str(tmp_path / "models"))
        res = mgr.resource_metrics()
        assert "cpu_percent" in res and "ram_percent" in res
        assert res["gpu_available"] is False  # honest: no CUDA here
    finally:
        db.close()


def test_detection_autoloads_when_weights_present(tmp_path):
    """A valid .onnx dropped into models/detection/ is loaded automatically."""
    import pytest
    onnx = pytest.importorskip("onnx")
    from onnx import helper, TensorProto

    models = tmp_path / "models" / "detection"
    models.mkdir(parents=True)
    X = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 1, 1])
    node = helper.make_node("GlobalAveragePool", ["images"], ["output"])
    graph = helper.make_graph([node], "dummy", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    (models / "dummy.onnx").write_bytes(model.SerializeToString())

    db = Database(str(tmp_path / "m.db"))
    try:
        mgr = AIModelManager(db, models_dir=str(tmp_path / "models"))
        st = mgr.task_status("detection")
        # weights were discovered and the session initialised without enabling
        assert st["backend"]["ready"] is True
        assert st["reason"] != "weights_missing"
        assert st["state"] in ("disabled", "loaded")  # ready, just not enabled yet
    finally:
        db.close()
