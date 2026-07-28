"""
Unit + API tests for the AI Image Analysis & Comparison platform.

Uses synthetic scenes (coloured shapes + text). Asserts real behaviour of the
analysis and comparison engines and the full REST surface, and that no existing
endpoint regresses.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from rtsp_backend.imaging import analysis as an
from rtsp_backend.imaging import comparison as cmp


def scene(shift=0, drop=False, recolor=False):
    img = np.full((360, 480, 3), 205, np.uint8)
    cv2.rectangle(img, (40, 40), (160, 200), (60, 200, 200) if recolor else (180, 120, 60), -1)
    if not drop:
        cv2.circle(img, (330, 110), 45, (40, 40, 200), -1)
    cv2.rectangle(img, (250 + shift, 220), (410 + shift, 320), (60, 180, 60), -1)
    cv2.putText(img, "PANEL A1", (50, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    return img


def jpg(img):
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


# --------------------------------------------------------------------------
# analysis unit
# --------------------------------------------------------------------------

def test_dominant_colors():
    colors = an.dominant_colors(scene(), k=5)
    assert 1 <= len(colors) <= 5
    assert abs(sum(c["ratio"] for c in colors) - 1.0) < 0.05
    for c in colors:
        assert c["hex"].startswith("#") and "name" in c


def test_analyze_full_shape():
    r = an.analyze(scene(), ai_manager=None)
    for key in ("image_size", "objects", "dominant_colors", "ocr", "phash",
                "tags", "summary", "defects", "notes"):
        assert key in r
    assert r["image_size"] == [480, 360]
    # no detector loaded => honest empty objects + at least one explanatory note
    assert r["object_total"] == 0
    assert len(r["notes"]) >= 1


def test_perceptual_hash_stable():
    h1 = an.perceptual_hash(scene())
    h2 = an.perceptual_hash(scene())
    assert h1 == h2  # identical image -> identical hash


def test_quality_defects_flags_blur():
    blurry = cv2.GaussianBlur(scene(), (21, 21), 0)
    flags = {d["type"] for d in an.quality_defects(blurry)}
    assert "blur" in flags


# --------------------------------------------------------------------------
# comparison unit
# --------------------------------------------------------------------------

def test_compare_identical_high_similarity():
    r = cmp.compare(scene(), scene())
    assert r["similarity"] >= 95.0
    assert r["status"] in ("identical", "minor")


def test_compare_detects_changes():
    r = cmp.compare(scene(), scene(shift=70, drop=True, recolor=True))
    assert r["similarity"] < 99.0
    assert r["n_diffs"] >= 1
    for d in r["differences"]:
        assert 0.0 <= d["confidence"] <= 1.0
        assert d["severity"] in ("info", "minor", "major")


def test_compare_registration_handles_perspective():
    # a mild affine warp of the scene should still register (perspective comp)
    img = scene()
    M = cv2.getRotationMatrix2D((240, 180), 3, 1.03)
    warped = cv2.warpAffine(img, M, (480, 360), borderValue=(205, 205, 205))
    r = cmp.compare(img, warped)
    # registration keeps a mild perspective change well above the score a
    # genuinely different image would get (~40-55%).
    assert r["similarity"] >= 72.0


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------

def test_upload_analyze_flow(client):
    r = client.post("/api/images/upload", files={"file": ("a.jpg", jpg(scene()), "image/jpeg")})
    assert r.status_code == 201
    img_id = r.json()["id"]

    r = client.post("/api/images/analyze", data={"image_id": img_id})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "analyzed"
    assert body["dominant_colors"] and "summary" in body

    assert client.get(f"/api/images/{img_id}").status_code == 200
    assert client.get("/api/images").json()["total"] >= 1


def test_analyze_inline_upload(client):
    r = client.post("/api/images/analyze", files={"file": ("a.jpg", jpg(scene()), "image/jpeg")})
    assert r.status_code == 200
    assert r.json()["status"] == "analyzed"


def test_bad_upload_rejected(client):
    r = client.post("/api/images/upload", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "bad_image"


def test_compare_api_and_report(client):
    files = {"reference": ("r.jpg", jpg(scene()), "image/jpeg"),
             "current": ("c.jpg", jpg(scene(shift=70, drop=True)), "image/jpeg")}
    r = client.post("/api/images/compare", files=files, data={"make_pdf": "true"})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["similarity"] <= 100
    assert body["overlay_path"] and body["n_diffs"] >= 1
    cmp_id = body["id"]

    rep = client.get(f"/api/images/report/{cmp_id}")
    assert rep.status_code == 200
    assert rep.json()["id"] == cmp_id

    # rendered overlay is servable media
    assert client.get("/api/media/" + body["overlay_path"]).status_code == 200
    # history + comparisons
    assert client.get("/api/images/history").status_code == 200
    assert client.get("/api/images/comparisons").json()["total"] >= 1


def test_compare_requires_two_images(client):
    r = client.post("/api/images/compare", files={"reference": ("r.jpg", jpg(scene()), "image/jpeg")})
    assert r.status_code == 400


def test_existing_endpoints_untouched(client):
    assert client.get("/health").status_code == 200
    assert client.get("/api/employees").status_code == 200
    assert client.get("/api/reference-panels").status_code == 200
    assert client.get("/api/stats/dashboard").status_code == 200
