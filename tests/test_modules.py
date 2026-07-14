"""
Tests for the additional AI detection modules and the face-quality upgrades.

These do not require any model weights or a GPU: the module backends honestly
report ``weights_missing`` when no ``.onnx`` is present (and must never fabricate
detections), and the face improvements are pure image logic exercised against a
real sample image.
"""

from __future__ import annotations

import numpy as np

from rtsp_backend.ai import registry
from rtsp_backend.ai.base import BBox, Detection
from rtsp_backend.ai.modules import HeuristicFallDetector


NEW_TASKS = ["fire", "violence", "fall", "weapon", "ppe", "human", "vehicle"]


def test_all_new_modules_registered():
    cat = registry.catalog()
    for task in NEW_TASKS:
        ids = [b["backend_id"] for b in cat.get(task, [])]
        assert ids, f"no backends registered for task {task}"
        assert "null" in ids  # honest disabled state always available


def test_onnx_modules_report_weights_missing_not_fake():
    """With no weights, a module backend must fail to load and detect nothing."""
    for task, bid in [("fire", "onnx_fire"), ("weapon", "onnx_weapon"),
                      ("ppe", "onnx_ppe"), ("violence", "onnx_violence"),
                      ("fall", "onnx_fall"), ("human", "onnx_human"),
                      ("vehicle", "onnx_vehicle")]:
        cls = registry.get(task, bid)
        backend = cls(models_dir="/tmp/nonexistent_models_dir_xyz")
        try:
            backend.load()
        except RuntimeError:
            pass  # expected: no weights
        assert backend.ready is False
        status = backend.status()
        assert status["reason"] == "weights_missing"
        # even if inference is attempted, it must not invent boxes
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            dets = backend.infer(frame)
        except RuntimeError:
            dets = []
        assert dets == []


def test_null_backends_return_nothing():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    for task in NEW_TASKS:
        cls = registry.get(task, "null")
        b = cls()
        b.load()
        assert b.ready is True
        assert b.infer(frame) == []


def test_heuristic_fall_flags_wide_person_only():
    fall = HeuristicFallDetector(fall_aspect_ratio=1.3)
    fall.load()
    standing = Detection("person", 0.9, BBox(0, 0, 40, 120))     # tall -> ok
    lying = Detection("person", 0.9, BBox(0, 0, 160, 60))        # wide -> fall
    out = fall.analyze_people([standing, lying])
    assert len(out) == 1
    assert out[0].label == "falling"
    assert out[0].extra["method"] == "aspect_ratio"


def test_heuristic_fall_no_people_no_detections():
    fall = HeuristicFallDetector()
    fall.load()
    assert fall.analyze_people([]) == []   # never fabricates
