"""API tests for employee management and face enrolment."""

from __future__ import annotations

from conftest import to_data_url


def test_employee_crud(client):
    # create
    r = client.post("/api/employees", json={
        "full_name": "Grace Hopper", "employee_code": "EMP-1",
        "department": "R&D", "job_title": "Engineer"})
    assert r.status_code == 201
    emp = r.json()
    eid = emp["id"]
    assert emp["full_name"] == "Grace Hopper"

    # list
    r = client.get("/api/employees")
    assert r.status_code == 200
    assert r.json()["total"] == 1
    assert r.json()["employees"][0]["embeddings"] == 0

    # get
    assert client.get(f"/api/employees/{eid}").status_code == 200

    # update
    r = client.put(f"/api/employees/{eid}", json={"department": "Compilers"})
    assert r.status_code == 200
    assert r.json()["department"] == "Compilers"

    # 404s
    assert client.get("/api/employees/9999").status_code == 404
    assert client.put("/api/employees/9999", json={"full_name": "x"}).status_code == 404

    # delete
    assert client.delete(f"/api/employees/{eid}").status_code == 200
    assert client.get("/api/employees").json()["total"] == 0


def test_enroll_via_upload_and_recognize_stats(client, astronaut_bgr):
    # enable face recognition
    assert client.post("/api/ai/models/face/enable", json={"enabled": True}).status_code == 200

    eid = client.post("/api/employees", json={"full_name": "Astro Naut"}).json()["id"]

    # upload a real face image -> should enroll a vector
    r = client.post(f"/api/employees/{eid}/images",
                    json={"image": to_data_url(astronaut_bgr), "make_profile": True})
    assert r.status_code == 200
    body = r.json()
    assert body["enrollment"]["ok"] is True
    assert body["image_id"] > 0

    # employee now reports an embedding and the image is retrievable
    emp = client.get(f"/api/employees/{eid}").json()
    assert len(emp["images"]) == 1
    media = client.get(f"/api/media/{emp['images'][0]['path']}")
    assert media.status_code == 200
    assert media.headers["content-type"].startswith("image/")

    # dashboard reflects the enrolled face
    stats = client.get("/api/stats/dashboard").json()
    assert stats["employees"]["total"] == 1
    assert stats["employees"]["enrolled_faces"] == 1

    # delete the image -> embedding removed
    img_id = emp["images"][0]["id"]
    assert client.delete(f"/api/employees/{eid}/images/{img_id}").status_code == 200
    stats = client.get("/api/stats/dashboard").json()
    assert stats["employees"]["enrolled_faces"] == 0


def test_upload_non_face_reports_reason(client):
    import base64
    import cv2
    import numpy as np
    assert client.post("/api/ai/models/face/enable", json={"enabled": True}).status_code == 200
    eid = client.post("/api/employees", json={"full_name": "Blank"}).json()["id"]
    blank = np.zeros((120, 120, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", blank)
    url = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
    r = client.post(f"/api/employees/{eid}/images", json={"image": url})
    assert r.status_code == 200
    # image saved, but no face -> honest reason, no fabricated enrolment
    assert r.json()["enrollment"]["ok"] is False
    assert r.json()["enrollment"]["reason"] == "no_face_detected"


def test_bad_image_rejected(client):
    eid = client.post("/api/employees", json={"full_name": "X"}).json()["id"]
    r = client.post(f"/api/employees/{eid}/images", json={"image": "not-base64!!"})
    assert r.status_code == 400
