"""
End-to-end face-recognition tests using a real detectable face.

These exercise the genuine pipeline: Haar detection -> deterministic embedding
-> persistent SQLite store -> cosine matching -> threshold gating. They do NOT
assert production-grade inter-person accuracy (that needs the InsightFace
backend); they assert the pipeline is correct, persistent, and honest about
unknowns.
"""

from __future__ import annotations

import time

import numpy as np

from rtsp_backend.ai.embedders import OpenCVFallbackEmbedder
from rtsp_backend.ai.base import BBox
from rtsp_backend.ai.face_service import FaceRecognitionService
from rtsp_backend.db import Database


def _mk(tmp_path):
    db = Database(str(tmp_path / "faces.db"))
    emb = OpenCVFallbackEmbedder()
    emb.load()
    return db, emb


def test_embedder_detects_real_face(astronaut_bgr):
    emb = OpenCVFallbackEmbedder()
    emb.load()
    boxes = emb.detect_faces(astronaut_bgr)
    assert len(boxes) >= 1
    vec = emb.embed(astronaut_bgr, boxes[0])
    assert vec is not None
    assert vec.shape == (emb.dim,)
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-3


def test_embedding_is_deterministic(astronaut_bgr):
    emb = OpenCVFallbackEmbedder()
    emb.load()
    box = emb.detect_faces(astronaut_bgr)[0]
    v1 = emb.embed(astronaut_bgr, box)
    v2 = emb.embed(astronaut_bgr, box)
    assert np.allclose(v1, v2)


def test_enroll_then_recognize_same_person(tmp_path, astronaut_bgr):
    db, emb = _mk(tmp_path)
    try:
        svc = FaceRecognitionService(db, emb, threshold=0.5)
        now = time.time()
        emp_id = db.insert(
            "INSERT INTO employees(full_name, created_at, updated_at) VALUES(?,?,?)",
            ("Astro Naut", now, now),
        )
        res = svc.enroll_image(emp_id, astronaut_bgr)
        assert res["ok"] is True
        assert svc.enrolled_vectors == 1

        # recognising the same image should return that employee with high score
        dets = svc.recognize_frame(astronaut_bgr)
        assert len(dets) >= 1
        face = max(dets, key=lambda d: d.confidence)
        assert face.employee_id == emp_id
        assert face.identity == "Astro Naut"
        assert face.confidence > 0.9
    finally:
        db.close()


def test_unknown_when_db_empty(tmp_path, astronaut_bgr):
    db, emb = _mk(tmp_path)
    try:
        svc = FaceRecognitionService(db, emb, threshold=0.5)
        dets = svc.recognize_frame(astronaut_bgr)
        assert len(dets) >= 1
        for d in dets:
            assert d.employee_id is None
            assert d.label == "Unknown Person"
    finally:
        db.close()


def test_high_threshold_forces_unknown(tmp_path, astronaut_bgr, astronaut_variant):
    db, emb = _mk(tmp_path)
    try:
        svc = FaceRecognitionService(db, emb, threshold=0.5)
        now = time.time()
        emp_id = db.insert(
            "INSERT INTO employees(full_name, created_at, updated_at) VALUES(?,?,?)",
            ("Astro", now, now),
        )
        svc.enroll_image(emp_id, astronaut_bgr)

        # an impossibly high threshold rejects even a near-identical variant
        svc.threshold = 0.9999
        dets = svc.recognize_frame(astronaut_variant)
        assert all(d.employee_id is None for d in dets)

        # a reasonable threshold accepts the same person again
        svc.threshold = 0.4
        dets = svc.recognize_frame(astronaut_bgr)
        assert any(d.employee_id == emp_id for d in dets)
    finally:
        db.close()


def test_persistence_across_service_reload(tmp_path, astronaut_bgr):
    db, emb = _mk(tmp_path)
    try:
        svc = FaceRecognitionService(db, emb, threshold=0.5)
        now = time.time()
        emp_id = db.insert(
            "INSERT INTO employees(full_name, created_at, updated_at) VALUES(?,?,?)",
            ("Persist", now, now),
        )
        svc.enroll_image(emp_id, astronaut_bgr)
    finally:
        db.close()

    # brand new DB connection + service instance reads the stored vectors
    db2 = Database(str(tmp_path / "faces.db"))
    emb2 = OpenCVFallbackEmbedder(); emb2.load()
    try:
        svc2 = FaceRecognitionService(db2, emb2, threshold=0.4)
        assert svc2.enrolled_vectors == 1
        dets = svc2.recognize_frame(astronaut_bgr)
        assert any(d.employee_id == emp_id for d in dets)
    finally:
        db2.close()


def test_cache_rebuilds_after_enroll(tmp_path, astronaut_bgr):
    db, emb = _mk(tmp_path)
    try:
        svc = FaceRecognitionService(db, emb, threshold=0.4)
        # initially unknown
        assert all(d.employee_id is None for d in svc.recognize_frame(astronaut_bgr))
        now = time.time()
        emp_id = db.insert(
            "INSERT INTO employees(full_name, created_at, updated_at) VALUES(?,?,?)",
            ("Late", now, now),
        )
        svc.enroll_image(emp_id, astronaut_bgr)
        # immediately recognised without any manual cache call
        assert any(d.employee_id == emp_id for d in svc.recognize_frame(astronaut_bgr))
    finally:
        db.close()


# --- face-quality upgrades (Priority 2) ---

def test_recognize_attaches_quality_metadata(tmp_path, astronaut_bgr):
    from rtsp_backend.db import Database
    from rtsp_backend.ai.embedders import OpenCVFallbackEmbedder
    from rtsp_backend.ai.face_service import FaceRecognitionService

    db = Database(str(tmp_path / "q.db"))
    try:
        emb = OpenCVFallbackEmbedder(); emb.load()
        svc = FaceRecognitionService(db, emb, threshold=0.5)
        dets = svc.recognize_frame(astronaut_bgr)
        assert dets, "expected to detect the astronaut face"
        d = dets[0]
        assert "quality" in d.extra and 0.0 <= d.extra["quality"] <= 1.0
        assert "blur" in d.extra and "min_side" in d.extra
    finally:
        db.close()


def test_tiny_faces_are_ignored(tmp_path, astronaut_bgr):
    from rtsp_backend.db import Database
    from rtsp_backend.ai.embedders import OpenCVFallbackEmbedder
    from rtsp_backend.ai.face_service import FaceRecognitionService

    db = Database(str(tmp_path / "t.db"))
    try:
        emb = OpenCVFallbackEmbedder(); emb.load()
        svc = FaceRecognitionService(db, emb, threshold=0.5)
        # an impossibly large min_face_size rejects every real face
        svc.min_face_size = 100000
        assert svc.recognize_frame(astronaut_bgr) == []
    finally:
        db.close()


def test_duplicate_boxes_are_merged():
    from rtsp_backend.ai.face_service import FaceRecognitionService
    from rtsp_backend.ai.base import BBox
    boxes = [BBox(0, 0, 100, 100), BBox(2, 2, 101, 101), BBox(400, 400, 460, 460)]
    merged = FaceRecognitionService._nms_boxes(boxes, 0.4)
    assert len(merged) == 2  # the two heavily-overlapping boxes collapse to one
