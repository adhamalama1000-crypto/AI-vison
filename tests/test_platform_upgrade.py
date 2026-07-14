"""
Tests for the industrial-platform upgrade: attendance, dataset management,
training orchestration, panel analysis, reference designs, and inspection.

Everything here exercises the REAL code paths on REAL (small) data — training
actually trains and exports an ONNX model, dataset validation walks actual
files, panel/inspection run the actual pipeline. Nothing is mocked away.
"""

from __future__ import annotations

import io
import os
import time
import zipfile

import cv2
import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _jpg_bytes(color=(40, 40, 200), size=64):
    img = np.full((size, size, 3), color, np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


def _panel_image_bytes():
    img = np.full((400, 600, 3), (60, 60, 60), np.uint8)
    cv2.line(img, (50, 60), (550, 60), (0, 0, 220), 3)
    cv2.line(img, (50, 140), (550, 140), (0, 200, 0), 3)
    cv2.line(img, (50, 220), (550, 220), (220, 0, 0), 3)
    return cv2.imencode(".jpg", img)[1].tobytes()


def _classification_zip(with_corrupt=True):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for i in range(5):
            z.writestr(f"mcb/img{i}.jpg", _jpg_bytes((30, 30, 200)))
        for i in range(2):
            z.writestr(f"relay/img{i}.jpg", _jpg_bytes((200, 30, 30)))
        if with_corrupt:
            z.writestr("relay/broken.jpg", b"definitely-not-an-image")
    buf.seek(0)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# attendance (Part 1)
# --------------------------------------------------------------------------- #

def test_attendance_config_roundtrip(client):
    r = client.put("/api/attendance/config", json={"seconds": 120})
    assert r.status_code == 200 and r.json()["timeout_seconds"] == 120
    assert client.get("/api/attendance/config").json()["timeout_seconds"] == 120


def test_attendance_recorded_once_within_timeout(client):
    app = client.app
    db, pipeline = app.state.db, app.state.pipeline
    emp_id = db.insert(
        "INSERT INTO employees(full_name,created_at,updated_at) VALUES(?,?,?)",
        ("Jane Doe", time.time(), time.time()))
    pipeline.set_attendance_timeout(3600)  # short-timeout branch (per-interval)
    frame = np.zeros((48, 48, 3), np.uint8)
    pipeline._record_attendance(emp_id, "Jane Doe", "cam1", "Cam", 0.9, frame)
    pipeline._record_attendance(emp_id, "Jane Doe", "cam1", "Cam", 0.9, frame)
    today = client.get("/api/attendance/today").json()
    assert today["present"] == 1
    assert len([r for r in today["records"] if r["employee_id"] == emp_id]) == 1


def test_attendance_records_again_after_timeout(client):
    app = client.app
    db, pipeline = app.state.db, app.state.pipeline
    emp_id = db.insert(
        "INSERT INTO employees(full_name,created_at,updated_at) VALUES(?,?,?)",
        ("Bob", time.time(), time.time()))
    pipeline.set_attendance_timeout(0)  # no throttle
    frame = np.zeros((48, 48, 3), np.uint8)
    pipeline._record_attendance(emp_id, "Bob", "c", "C", 0.8, frame)
    pipeline._record_attendance(emp_id, "Bob", "c", "C", 0.8, frame)
    rows = client.get(f"/api/attendance?employee_id={emp_id}").json()
    assert rows["total"] == 2


# --------------------------------------------------------------------------- #
# dataset management (Part 2)
# --------------------------------------------------------------------------- #

def test_dataset_upload_classification_and_validation(client):
    data = _classification_zip(with_corrupt=True)
    r = client.post("/api/datasets/upload",
                    files={"files": ("ds.zip", data, "application/zip")})
    assert r.status_code == 200
    rep = r.json()["report"]
    assert rep["kind"] == "classification"
    assert set(rep["classes"]) == {"mcb", "relay"}
    assert len(rep["corrupt_images"]) == 1     # the broken file is detected
    assert rep["ok"] is False                  # corrupt image => not ok
    assert rep["imbalance_ratio"] is not None


def test_dataset_list_and_delete(client):
    client.post("/api/datasets/upload",
                files={"files": ("ds.zip", _classification_zip(False), "application/zip")})
    lst = client.get("/api/datasets").json()
    assert lst["total"] >= 1
    ds_id = lst["datasets"][0]["id"]
    assert client.delete(f"/api/datasets/{ds_id}").json()["deleted"] == ds_id


def test_detect_kind_variants(tmp_path):
    from rtsp_backend import datasets_svc as dsv

    # YOLO
    yolo = tmp_path / "yolo"
    (yolo / "images").mkdir(parents=True)
    (yolo / "labels").mkdir(parents=True)
    cv2.imwrite(str(yolo / "images" / "a.jpg"), np.zeros((10, 10, 3), np.uint8))
    (yolo / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    assert dsv.detect_kind(str(yolo)) == "yolo"

    # COCO
    coco = tmp_path / "coco"
    coco.mkdir()
    (coco / "ann.json").write_text(
        '{"images":[{"id":1}],"annotations":[{"image_id":1,"category_id":1}],'
        '"categories":[{"id":1,"name":"mcb"}]}')
    assert dsv.detect_kind(str(coco)) == "coco"

    # VOC
    voc = tmp_path / "voc"
    voc.mkdir()
    (voc / "a.xml").write_text(
        "<annotation><object><name>relay</name></object></annotation>")
    cv2.imwrite(str(voc / "a.jpg"), np.zeros((10, 10, 3), np.uint8))
    assert dsv.detect_kind(str(voc)) == "voc"


# --------------------------------------------------------------------------- #
# training orchestration (Parts 3,4,5)
# --------------------------------------------------------------------------- #

def _wait_job(client, job_id, timeout=90):
    for _ in range(timeout * 2):
        j = client.get(f"/api/training/{job_id}").json()
        if j["status"] in ("completed", "failed", "stopped"):
            return j
        time.sleep(0.5)
    return client.get(f"/api/training/{job_id}").json()


def test_training_trains_compares_selects_and_exports(client):
    body = {"name": "demo", "task": "classification",
            "models": ["mlp", "logreg", "yolov8"],
            "config": {"epochs": 8, "augment": True}}
    r = client.post("/api/training", json=body)
    assert r.status_code == 201
    job_id = r.json()["job_id"]
    j = _wait_job(client, job_id)
    assert j["status"] == "completed", j.get("error")
    assert j["best_model"] in ("mlp", "logreg")
    comp = client.get(f"/api/training/{job_id}/comparison").json()["comparison"]
    trained = [c for c in comp if c["status"] == "trained"]
    assert len(trained) >= 2                       # multiple real models compared
    best = [c for c in comp if c.get("selected")]
    assert len(best) == 1
    assert best[0]["onnx"]["verification"]["ok"]   # exported ONNX reloads & matches
    # detection arch honestly skipped, not faked
    assert any(c["model"] == "yolov8" and c["status"] == "skipped" for c in comp)


def test_training_history_has_real_learning_curve(client):
    body = {"name": "curve", "task": "classification", "models": ["mlp"],
            "config": {"epochs": 10, "augment": False}}
    job_id = client.post("/api/training", json=body).json()["job_id"]
    j = _wait_job(client, job_id)
    hist = [h for h in j["history"] if h.get("model") == "mlp"]
    assert len(hist) >= 3
    # loss should generally decrease from first to last recorded epoch
    assert hist[-1]["train_loss"] <= hist[0]["train_loss"]


def test_training_stop(client):
    body = {"name": "stopme", "task": "classification", "models": ["mlp"],
            "config": {"epochs": 500}}
    job_id = client.post("/api/training", json=body).json()["job_id"]
    time.sleep(1.0)
    assert client.post(f"/api/training/{job_id}/stop").json()["ok"] is True
    j = _wait_job(client, job_id, timeout=30)
    assert j["status"] in ("stopped", "completed")


# --------------------------------------------------------------------------- #
# panel analysis (Part 8)
# --------------------------------------------------------------------------- #

def test_panel_analysis_produces_report(client):
    r = client.post("/api/panels/analyze",
                    files={"file": ("p.jpg", _panel_image_bytes(), "image/jpeg")},
                    data={"make_pdf": "true"})
    assert r.status_code == 200
    d = r.json()
    assert d["wire_total"] >= 1                    # classical wire baseline finds lines
    assert d["annotated"] and d["json"]
    # no trained component model => zero components, with an honest note
    assert d["component_total"] == 0
    assert any("component model" in n for n in d["result"]["notes"])
    assert client.get("/api/panels").json()["total"] >= 1


# --------------------------------------------------------------------------- #
# reference + inspection (Parts 9,10)
# --------------------------------------------------------------------------- #

def test_reference_upload_and_inspection(client):
    r = client.post("/api/reference/upload",
                    files={"file": ("ref.png", _panel_image_bytes(), "image/png")},
                    data={"name": "panel-v1"})
    ref_id = r.json()["id"]
    client.put(f"/api/reference/{ref_id}/spec",
               json={"spec": {"component_counts": {"mcb": 3},
                              "wire_color_counts": {}}})
    r = client.post("/api/inspection/run",
                    files={"file": ("obs.jpg", _panel_image_bytes(), "image/jpeg")},
                    data={"reference_id": str(ref_id), "make_pdf": "true"})
    assert r.status_code == 200
    ins = r.json()
    # expected 3 mcb but none detected (no model) => a real mismatch, status fail
    assert ins["status"] == "fail"
    assert any(m["type"] == "missing_component" for m in ins["mismatches"])
    assert client.get("/api/inspection").json()["total"] >= 1


def test_reports_registry(client):
    client.post("/api/panels/analyze",
                files={"file": ("p.jpg", _panel_image_bytes(), "image/jpeg")},
                data={"make_pdf": "false"})
    reports = client.get("/api/reports?kind=panel_analysis").json()
    assert reports["total"] >= 1
    assert reports["reports"][0]["kind"] == "panel_analysis"


# --------------------------------------------------------------------------- #
# inspection comparison unit logic
# --------------------------------------------------------------------------- #

def test_inspection_compare_logic():
    from rtsp_backend import inspection_svc as isv
    expected = {"component_counts": {"mcb": 2, "relay": 1},
                "wire_color_counts": {"red": 2}}
    observed = {"component_counts": {"mcb": 2, "contactor": 1},
                "wire_color_counts": {"red": 2}, "notes": []}
    res = isv.compare(expected, observed)
    types = {m["type"] for m in res["mismatches"]}
    assert "missing_component" in types      # relay expected, absent
    assert "extra_component" in types        # contactor found, unexpected
    assert res["status"] == "fail"


# --------------------------------------------------------------------------- #
# security hardening (audit fixes)
# --------------------------------------------------------------------------- #

def _build_client(tmp_path, **overrides):
    import warnings
    warnings.filterwarnings("ignore")
    from fastapi.testclient import TestClient
    from rtsp_backend.app import build_app
    from rtsp_backend.config import Settings
    s = Settings(db_path=str(tmp_path / "platform.db"),
                 data_dir=str(tmp_path / "data"),
                 models_dir=str(tmp_path / "models"), cameras=[], **overrides)
    return TestClient(build_app(s))


def test_media_never_serves_the_database(client):
    # The SQLite DB lives under data_dir; it must not be downloadable.
    for p in ("platform.db", "platform.db-wal", "platform.db-shm",
              "../platform.db", "etc/passwd"):
        assert client.get(f"/api/media/{p}").status_code == 404


def test_api_key_required_when_configured(tmp_path):
    with _build_client(tmp_path, api_key="topsecret") as c:
        assert c.get("/health").status_code == 200                 # health open
        assert c.get("/api/datasets").status_code == 401           # gated
        assert c.get("/api/datasets",
                     headers={"X-API-Key": "topsecret"}).status_code == 200
        assert c.get("/api/reports?api_key=topsecret").status_code == 200
        assert c.get("/api/datasets",
                     headers={"X-API-Key": "wrong"}).status_code == 401


def test_upload_size_cap_enforced(tmp_path):
    import cv2
    import numpy as np
    with _build_client(tmp_path, max_upload_bytes=4096) as c:
        big = cv2.imencode(".jpg", (np.random.rand(300, 300, 3) * 255).astype("uint8"))[1].tobytes()
        assert len(big) > 4096
        r = c.post("/api/panels/analyze",
                   files={"file": ("b.jpg", big, "image/jpeg")},
                   data={"make_pdf": "false"})
        assert r.status_code == 413


def test_zip_bomb_and_slip_guarded(tmp_path):
    import io
    import zipfile
    from rtsp_backend import datasets_svc as dsv

    # zip-slip
    slip = tmp_path / "slip.zip"
    with zipfile.ZipFile(slip, "w") as z:
        z.writestr("../../evil.txt", b"x")
    with pytest.raises(ValueError):
        dsv.safe_extract_zip(str(slip), str(tmp_path / "out1"))

    # declared-size bomb (tiny file, huge declared size is hard to fake; instead
    # assert the file-count cap triggers)
    many = tmp_path / "many.zip"
    with zipfile.ZipFile(many, "w") as z:
        for i in range(50):
            z.writestr(f"f{i}.txt", b"x")
    with pytest.raises(ValueError):
        dsv.safe_extract_zip(str(many), str(tmp_path / "out2"), max_files=10)


def test_training_reports_unknown_models(client):
    body = {"name": "u", "task": "classification",
            "models": ["mlp", "totally_made_up_arch"],
            "config": {"epochs": 4}}
    job_id = client.post("/api/training", json=body).json()["job_id"]
    j = _wait_job(client, job_id)
    comp = client.get(f"/api/training/{job_id}/comparison").json()["comparison"]
    assert any(c["model"] == "totally_made_up_arch" and c["status"] == "skipped"
               for c in comp)
