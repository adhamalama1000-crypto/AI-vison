"""Integration tests for the AI overlay pipeline and AI media endpoints."""

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np

from rtsp_backend.ai.manager import AIModelManager
from rtsp_backend.ai.pipeline import AIPipeline
from rtsp_backend.db import Database


def test_annotated_jpeg_is_valid_and_overlaid(astronaut_bgr):
    d = tempfile.mkdtemp()
    db = Database(os.path.join(d, "p.db"))
    try:
        mgr = AIModelManager(db, models_dir=os.path.join(d, "models"))
        mgr.set_enabled("face", True)
        pipe = AIPipeline(db, mgr, data_dir=os.path.join(d, "data"))
        emp = db.insert(
            "INSERT INTO employees(full_name,created_at,updated_at) VALUES(?,?,?)",
            ("Astro", 0, 0),
        )
        assert mgr.face_service.enroll_image(emp, astronaut_bgr)["ok"] is True

        jpg = pipe.annotated_jpeg("cam1", "Front", astronaut_bgr)
        assert len(jpg) > 1000
        decoded = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None
        assert decoded.shape == astronaut_bgr.shape
    finally:
        db.close()


def test_process_persists_and_structures_results(astronaut_bgr):
    d = tempfile.mkdtemp()
    db = Database(os.path.join(d, "p.db"))
    try:
        mgr = AIModelManager(db, models_dir=os.path.join(d, "models"))
        mgr.set_enabled("face", True)
        pipe = AIPipeline(db, mgr, data_dir=os.path.join(d, "data"))
        res = pipe.process("cam1", "Front", astronaut_bgr, annotate=False, force=True)
        assert "faces" in res and isinstance(res["faces"], list)
        assert len(res["faces"]) >= 1
        # an unknown face should have created an event with a snapshot on disk
        ev = db.query_one("SELECT * FROM events WHERE type='unknown_person'")
        assert ev is not None
        assert ev["snapshot"]
        assert os.path.isfile(os.path.join(d, "data", ev["snapshot"]))
    finally:
        db.close()


def test_ai_media_endpoints_error_cleanly(client):
    # unknown camera -> 404 (not a crash)
    assert client.get("/api/cameras/ghost/analyze").status_code == 404
    assert client.get("/api/cameras/ghost/ai-snapshot").status_code == 404


def test_pipeline_dedups_repeated_identity(astronaut_bgr):
    """Processing the same face many times logs one event per identity, not N."""
    d = tempfile.mkdtemp()
    db = Database(os.path.join(d, "p.db"))
    try:
        mgr = AIModelManager(db, models_dir=os.path.join(d, "models"))
        mgr.set_enabled("face", True)
        pipe = AIPipeline(db, mgr, data_dir=os.path.join(d, "data"))
        emp = db.insert("INSERT INTO employees(full_name,created_at,updated_at) VALUES(?,?,?)",
                        ("Astro", 0, 0))
        assert mgr.face_service.enroll_image(emp, astronaut_bgr)["ok"] is True

        for _ in range(30):
            pipe.process("cam1", "Front", astronaut_bgr, annotate=False, force=True)

        # de-duplicated within the window: a single recognised event, not 30
        n = db.query_one(
            "SELECT COUNT(*) c FROM events WHERE type='face_recognized' AND label='Astro'")["c"]
        assert n == 1, f"expected 1 deduped event, got {n}"
        # and no duplicate embedding rows were created by recognition
        emb = db.query_one("SELECT COUNT(*) c FROM face_embeddings WHERE employee_id=?", (emp,))["c"]
        assert emb == 1
    finally:
        db.close()
