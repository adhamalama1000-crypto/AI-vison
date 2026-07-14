"""Tests for the SQLite persistence layer."""

from __future__ import annotations

import time

import numpy as np

from rtsp_backend.db import Database


def test_schema_and_settings_roundtrip(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    try:
        db.set_setting("theme", {"mode": "dark", "n": 3})
        assert db.get_setting("theme") == {"mode": "dark", "n": 3}
        assert db.get_setting("missing", "fallback") == "fallback"
        db.set_setting("theme", "light")  # overwrite
        assert db.get_setting("theme") == "light"
        assert "theme" in db.all_settings()
    finally:
        db.close()


def test_employee_and_embedding_cascade(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    try:
        now = time.time()
        emp_id = db.insert(
            "INSERT INTO employees(full_name, created_at, updated_at) VALUES(?,?,?)",
            ("Ada Lovelace", now, now),
        )
        img_id = db.insert(
            "INSERT INTO employee_images(employee_id, path, created_at) VALUES(?,?,?)",
            (emp_id, "employees/1/a.jpg", now),
        )
        vec = np.ones(8, dtype=np.float32)
        db.insert(
            "INSERT INTO face_embeddings(employee_id, image_id, embedder, dim, vector, created_at) "
            "VALUES(?,?,?,?,?,?)",
            (emp_id, img_id, "opencv_fallback", 8, vec.tobytes(), now),
        )
        assert db.query_one("SELECT COUNT(*) c FROM face_embeddings")["c"] == 1

        # deleting the employee cascades to images + embeddings (FK ON)
        db.execute("DELETE FROM employees WHERE id=?", (emp_id,))
        assert db.query_one("SELECT COUNT(*) c FROM employee_images")["c"] == 0
        assert db.query_one("SELECT COUNT(*) c FROM face_embeddings")["c"] == 0
    finally:
        db.close()


def test_model_config_upsert(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    try:
        db.execute(
            "INSERT INTO model_config(name, backend, enabled, params, updated_at) "
            "VALUES(?,?,?,?,?)",
            ("face", "opencv_fallback", 1, "{}", time.time()),
        )
        db.execute(
            "INSERT INTO model_config(name, backend, enabled, params, updated_at) "
            "VALUES(?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET enabled=excluded.enabled",
            ("face", "opencv_fallback", 0, "{}", time.time()),
        )
        row = db.query_one("SELECT enabled FROM model_config WHERE name='face'")
        assert row["enabled"] == 0
    finally:
        db.close()
