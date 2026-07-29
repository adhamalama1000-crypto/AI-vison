"""
CSV report export, and the annotation review loop.

Two things worth protecting with tests here.

**CSV quoting.** Nameplate text comes from OCR and routinely contains commas, quotes
and newlines. Written naively those corrupt the row and shift every later column, and
the corruption is invisible until somebody opens the file in a spreadsheet. The tests
round-trip through :mod:`csv` to prove the escaping holds.

**Review is not a rubber stamp.** ``autolabel`` output is raw model prediction. The
export must refuse to emit un-reviewed images by default, must exclude anything a human
flagged as needing a redraw, and must reject a reclassification to a class id that is
not in the taxonomy — a typo there would become a label the trainer silently ignores.
"""

from __future__ import annotations

import csv
import io
import json
import os

import cv2
import numpy as np
import pytest

from rtsp_backend import annotation_svc as asvc
from rtsp_backend import reports_svc
from rtsp_backend.electrical import taxonomy as tax


# ==========================================================================
# CSV reports
# ==========================================================================

def _result(components=None, nasty_text: bool = False) -> dict:
    comps = components if components is not None else [
        {"index": 0, "class_id": "mcb", "label": "MCB", "title": "MCB 1",
         "confidence": 0.93, "bbox": [10.0, 20.0, 50.0, 100.0],
         "category": "protection", "domain": "power", "position": "row 1 left",
         "row": 0, "row_position": 0, "manufacturer": "Schneider Electric",
         "product_family": "Acti9", "part_number": "A9F74216",
         "identification_basis": "nameplate", "nameplate_text": "C16 6kA",
         "purpose": "final circuit protection"},
        {"index": 1, "class_id": "contactor", "label": "Contactor",
         "title": "Contactor 1", "confidence": 0.88,
         "bbox": [60.0, 20.0, 130.0, 120.0], "category": "switching",
         "domain": "power", "position": "row 1 middle", "row": 0,
         "row_position": 1, "manufacturer": None, "product_family": None,
         "part_number": None, "identification_basis": "model",
         "nameplate_text": None, "purpose": "motor switching"},
    ]
    if nasty_text:
        # Exactly what OCR produces from a real nameplate.
        comps[0]["nameplate_text"] = 'C16, 6kA "AC-3"\nIn=16A; Ue=230V'
        comps[0]["purpose"] = "protection, isolation"
    return {
        "components": comps,
        "component_total": len(comps),
        "component_counts": {"MCB": 1, "Contactor": 1},
        "bill_of_materials": [
            {"class_id": "mcb", "name": "MCB", "quantity": 1,
             "category": "protection"},
            {"class_id": "contactor", "name": "Contactor", "quantity": 1,
             "category": "switching"},
        ],
        "missing_components": [
            {"class_id": "overload_relay", "severity": "important",
             "rationale": "An MCC normally includes an overload relay, per feeder."},
        ],
        "maintenance_notes": [
            {"code": "starter_protection_mismatch", "severity": "important",
             "message": "1 contactor but 0 overload relays detected."},
        ],
        "confidence": {"mean": 0.905, "min": 0.88, "max": 0.93, "unknown": 0},
        "panel": {"panel_type": "motor_control_centre",
                  "panel_type_name": "Motor Control Centre",
                  "confidence": 0.81, "function": "motor control"},
        "notes": ["a note, with a comma"],
        "report": {"risk_assessment": {
            "level": "elevated", "score": 5.5, "confidence": "high",
            "assessable": True, "headline": "2 indicators, 1 important.",
            "drivers": [{"code": "missing_overload_relay", "weight": 4.5,
                         "message": "Safety-critical device not detected."}],
            "recommendations": ["PRIORITY: verify the overload relay."],
            "limits": ["Derived from one photograph."],
        }},
    }


def _read_csv(data_dir: str, rel: str) -> list[list[str]]:
    with open(os.path.join(data_dir, rel), encoding="utf-8", newline="") as fh:
        return list(csv.reader(fh))


def test_component_csv_has_one_row_per_device(tmp_path):
    rel = reports_svc.panel_csv(str(tmp_path), _result())
    rows = _read_csv(str(tmp_path), rel)
    assert rows[0] == list(reports_svc.COMPONENT_CSV_COLUMNS)
    assert len(rows) == 3, "header plus two devices"
    assert rows[1][1] == "mcb"
    assert rows[2][1] == "contactor"


def test_component_csv_carries_geometry_and_nameplate(tmp_path):
    rel = reports_svc.panel_csv(str(tmp_path), _result())
    rows = _read_csv(str(tmp_path), rel)
    header = rows[0]
    row = dict(zip(header, rows[1]))
    assert row["confidence"] == "0.93"
    assert (row["x1"], row["y1"], row["x2"], row["y2"]) == \
           ("10.0", "20.0", "50.0", "100.0")
    assert row["width"] == "40.0" and row["height"] == "80.0"
    assert row["manufacturer"] == "Schneider Electric"
    assert row["part_number"] == "A9F74216"


def test_csv_survives_ocr_text_containing_commas_quotes_and_newlines(tmp_path):
    """Written naively this corrupts the row and shifts every later column."""
    rel = reports_svc.panel_csv(str(tmp_path), _result(nasty_text=True))
    rows = _read_csv(str(tmp_path), rel)
    assert len(rows) == 3, "quoting failed — the nasty text split the row"
    row = dict(zip(rows[0], rows[1]))
    assert row["nameplate_text"] == 'C16, 6kA "AC-3"\nIn=16A; Ue=230V'
    assert row["purpose"] == "protection, isolation"


def test_component_csv_of_an_empty_result_is_a_header_only(tmp_path):
    rel = reports_svc.panel_csv(str(tmp_path), {"components": []})
    rows = _read_csv(str(tmp_path), rel)
    assert len(rows) == 1 and rows[0][0] == "index"


def test_summary_csv_sections(tmp_path):
    rel = reports_svc.panel_summary_csv(str(tmp_path), _result())
    rows = _read_csv(str(tmp_path), rel)
    assert rows[0] == ["section", "key", "value", "detail"]
    sections = {r[0] for r in rows[1:]}
    for expected in ("panel", "risk", "bill_of_materials", "possible_missing",
                     "risk_driver", "recommendation", "maintenance_note",
                     "limit", "note"):
        assert expected in sections, f"{expected} section missing"


def test_summary_csv_records_the_risk_level_and_its_meaning(tmp_path):
    rel = reports_svc.panel_summary_csv(str(tmp_path), _result())
    rows = _read_csv(str(tmp_path), rel)
    risk = {r[1]: (r[2], r[3]) for r in rows if r[0] == "risk"}
    assert risk["level"][0] == "elevated"
    assert risk["score"][0] == "5.5"
    # A reader of the CSV alone must learn what assessable=false means.
    assert "NOT a pass" in risk["assessable"][1]


def test_summary_csv_of_an_unassessable_result_says_so(tmp_path):
    result = _result()
    result["report"]["risk_assessment"] = {
        "level": "unknown", "assessable": False, "score": 0.0,
        "confidence": "none", "headline": "No model loaded.",
        "drivers": [], "recommendations": [], "limits": [],
    }
    rel = reports_svc.panel_summary_csv(str(tmp_path), result)
    rows = _read_csv(str(tmp_path), rel)
    risk = {r[1]: (r[2], r[3]) for r in rows if r[0] == "risk"}
    assert risk["level"][0] == "unknown"
    assert risk["assessable"][0] == "False"


def test_api_writes_csv_and_pdf_on_request(client):
    img = np.full((240, 320, 3), 120, np.uint8)
    payload = cv2.imencode(".jpg", img)[1].tobytes()
    body = client.post(
        "/api/panel/analyze",
        files={"file": ("p.jpg", io.BytesIO(payload), "image/jpeg")},
        params={"csv": True, "pdf": True}).json()
    exports = body["exports"]
    assert exports["csv_components"].endswith(".csv")
    assert exports["csv_summary"].endswith(".csv")
    assert "json" in exports
    # The files must actually be fetchable through /api/media.
    for rel in (exports["csv_components"], exports["csv_summary"]):
        r = client.get(f"/api/media/{rel}")
        assert r.status_code == 200, rel


def test_api_omits_exports_unless_asked(client):
    img = np.full((240, 320, 3), 120, np.uint8)
    payload = cv2.imencode(".jpg", img)[1].tobytes()
    body = client.post(
        "/api/panel/analyze",
        files={"file": ("p.jpg", io.BytesIO(payload), "image/jpeg")},
        params={"persist": False}).json()
    assert "exports" not in body


# ==========================================================================
# annotation review
# ==========================================================================

def _autolabel_output(root: str, n: int = 4) -> str:
    """A directory shaped like `cli autolabel` output.

    Two classified boxes per image in the YOLO file, plus one *unclassified* box in a
    sidecar — which is where they genuinely go, because
    ``unknown_industrial_component`` has no class index and cannot be written into a
    label file.
    """
    idx = tax.class_index()
    os.makedirs(os.path.join(root, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(root, "labels", "train"), exist_ok=True)
    for i in range(n):
        cv2.imwrite(os.path.join(root, "images", "train", f"p{i}.jpg"),
                    np.full((200, 300, 3), 100 + i * 10, np.uint8))
        with open(os.path.join(root, "labels", "train", f"p{i}.txt"), "w",
                  encoding="utf-8") as fh:
            fh.write(f"{idx['mcb']} 0.3 0.4 0.1 0.2\n")
            fh.write(f"{idx['contactor']} 0.7 0.5 0.15 0.25\n")
        with open(os.path.join(root, "labels", "train",
                               f"p{i}.unclassified.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"image": f"p{i}.jpg", "width": 300, "height": 200,
                       "boxes": [{"bbox": [150.0, 120.0, 210.0, 180.0],
                                  "norm": {"cx": 0.6, "cy": 0.75,
                                           "w": 0.2, "h": 0.3},
                                  "confidence": 0.21,
                                  "raw_class_id":
                                      tax.UNKNOWN_COMPONENT_ID}]}, fh)
    with open(os.path.join(root, "autolabel_manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"status": "labelled", "backend": "openvocab_owlv2",
                   "thresholds": {"accept": 0.35, "review": 0.15},
                   "by_verdict": {"auto": 2, "uncertain": 2},
                   "boxes_unclassified": n,
                   "review_queue": [f"p{i}.jpg" for i in (3, 1)]}, fh)
    return root


@pytest.fixture()
def batch(client, tmp_path):
    root = _autolabel_output(str(tmp_path / "prelabelled"))
    r = client.post("/api/annotations",
                    json={"name": "round1", "root": root, "split": "train"})
    assert r.status_code == 200, r.text
    return "round1", root


def test_register_and_list_batches(client, batch):
    name, _root = batch
    body = client.get("/api/annotations").json()
    assert any(b["name"] == name for b in body["batches"])
    detail = client.get(f"/api/annotations/{name}").json()
    assert detail["images"] == 4
    assert detail["remaining"] == 4
    assert detail["backend"] == "openvocab_owlv2"
    assert "PRE-LABELS" in detail["guidance"]


def test_registering_a_non_autolabel_directory_fails_clearly(client, tmp_path):
    r = client.post("/api/annotations",
                    json={"name": "bad", "root": str(tmp_path / "nope")})
    assert r.status_code == 400
    assert "autolabel" in r.json()["error"]["message"]


def test_queue_is_ordered_worst_first(client, batch):
    """The manifest's review queue ordering must be honoured."""
    name, _root = batch
    items = client.get(f"/api/annotations/{name}/queue").json()["items"]
    assert [i["filename"] for i in items][:2] == ["p3.jpg", "p1.jpg"]
    assert all(i["state"] == "pending" for i in items)


def test_image_detail_returns_boxes_and_the_rules(client, batch):
    name, _root = batch
    d = client.get(f"/api/annotations/{name}/items/p0.jpg").json()
    assert d["width"] == 300 and d["height"] == 200
    # Two classified boxes from the YOLO file, plus one from the sidecar.
    assert len(d["boxes"]) == 3
    assert d["boxes"][0]["class_id"] == "mcb"
    assert d["boxes"][0]["unclassified"] is False
    assert d["boxes"][2]["unclassified"] is True
    assert d["boxes"][2]["class_id"] is None
    assert d["boxes"][2]["confidence"] == 0.21
    assert d["unclassified_boxes"] == 1
    # Absolute pixels alongside normalised, so a client need not guess.
    assert d["boxes"][0]["bbox"] == [75.0, 60.0, 105.0, 100.0]
    joined = " ".join(d["labelling_rules"])
    assert "TWO boxes" in joined and "per pole" in joined


def test_unclassified_sidecar_boxes_are_surfaced_for_review(client, batch):
    """The regression: these boxes were previously discarded entirely.

    `unknown_industrial_component` has no class index, so an earlier version of
    autolabel looked it up, got None, and silently dropped every unclassified box —
    exactly the boxes that show where the model is blind.
    """
    name, _root = batch
    d = client.get(f"/api/annotations/{name}/items/p1.jpg").json()
    unclassified = [b for b in d["boxes"] if b["unclassified"]]
    assert len(unclassified) == 1, "the sidecar box was not surfaced"
    box = unclassified[0]
    assert box["bbox"] == [150.0, 120.0, 210.0, 180.0]
    assert box["name"] == "Unclassified"
    # Its index continues the classified boxes' index space, so the ordinary
    # 'reclassified' verdict works on it.
    assert box["index"] == 2


def test_serving_an_image_is_guarded_against_traversal(client, batch):
    name, _root = batch
    assert client.get(f"/api/annotations/{name}/images/p0.jpg").status_code == 200
    for evil in ("../autolabel_manifest.json", "..%2Fautolabel_manifest.json"):
        r = client.get(f"/api/annotations/{name}/images/{evil}")
        assert r.status_code == 404, evil


def test_recording_verdicts_and_state(client, batch):
    name, _root = batch
    r = client.post(f"/api/annotations/{name}/items/p0.jpg", json={
        "boxes": [{"index": 0, "verdict": "accepted"},
                  {"index": 1, "verdict": "reclassified",
                   "class_id": "relay"}],
        "state": "reviewed", "reviewer": "hatem"})
    assert r.status_code == 200
    body = r.json()
    assert body["boxes_recorded"] == 2
    assert body["reviewed"] == 1

    d = client.get(f"/api/annotations/{name}/items/p0.jpg").json()
    assert d["state"] == "reviewed"
    assert d["boxes"][0]["verdict"] == "accepted"
    assert d["boxes"][1]["verdict"] == "reclassified"
    assert d["boxes"][1]["class_id"] == "relay"
    assert d["boxes"][1]["original_class_id"] == "contactor"


def test_reclassifying_to_an_unknown_class_is_refused(client, batch):
    """A typo would otherwise become a label the trainer silently ignores."""
    name, _root = batch
    r = client.post(f"/api/annotations/{name}/items/p0.jpg", json={
        "boxes": [{"index": 0, "verdict": "reclassified",
                   "class_id": "definitely_not_a_class"}]})
    assert r.status_code == 400
    assert "valid taxonomy class id" in r.json()["error"]["message"]


def test_an_invalid_verdict_is_refused(client, batch):
    name, _root = batch
    r = client.post(f"/api/annotations/{name}/items/p0.jpg",
                    json={"boxes": [{"index": 0, "verdict": "looks_fine"}]})
    assert r.status_code == 400


def test_an_invalid_state_is_refused(client, batch):
    name, _root = batch
    r = client.post(f"/api/annotations/{name}/items/p0.jpg",
                    json={"boxes": [], "state": "probably_ok"})
    assert r.status_code == 400


def test_export_drops_rejected_and_applies_reclassification(client, batch,
                                                            tmp_path):
    name, _root = batch
    client.post(f"/api/annotations/{name}/items/p0.jpg", json={
        "boxes": [{"index": 0, "verdict": "accepted"},
                  {"index": 1, "verdict": "rejected"}],
        "state": "reviewed"})
    client.post(f"/api/annotations/{name}/items/p1.jpg", json={
        "boxes": [{"index": 0, "verdict": "reclassified",
                   "class_id": "mccb"},
                  {"index": 1, "verdict": "accepted"}],
        "state": "reviewed"})

    dst = str(tmp_path / "corrected")
    body = client.post(f"/api/annotations/{name}/export",
                       json={"dst_root": dst}).json()
    assert body["images_exported"] == 2
    assert body["boxes_dropped"] == 1
    assert body["boxes_reclassified"] == 1

    idx = tax.class_index()
    p0_lines = open(os.path.join(dst, "labels", "train", "p0.txt"),
                    encoding="utf-8").read().strip().splitlines()
    assert len(p0_lines) == 1, "the rejected box must be gone"
    assert p0_lines[0].split()[0] == str(idx["mcb"])
    p1_lines = open(os.path.join(dst, "labels", "train", "p1.txt"),
                    encoding="utf-8").read().strip().splitlines()
    assert p1_lines[0].split()[0] == str(idx["mccb"]), "reclassification applied"


def test_a_classified_sidecar_box_enters_the_export(client, batch, tmp_path):
    """An unclassified box becomes trainable data only once a human names it."""
    name, _root = batch
    client.post(f"/api/annotations/{name}/items/p0.jpg", json={
        "boxes": [{"index": 0, "verdict": "accepted"},
                  {"index": 1, "verdict": "accepted"},
                  # index 2 is the sidecar box
                  {"index": 2, "verdict": "reclassified", "class_id": "vfd"}],
        "state": "reviewed"})
    dst = str(tmp_path / "corrected")
    body = client.post(f"/api/annotations/{name}/export",
                       json={"dst_root": dst}).json()

    assert body["unclassified_promoted"] == 1
    idx = tax.class_index()
    lines = open(os.path.join(dst, "labels", "train", "p0.txt"),
                 encoding="utf-8").read().strip().splitlines()
    assert len(lines) == 3
    assert lines[2].split()[0] == str(idx["vfd"])
    # And its geometry came from the sidecar, normalised.
    assert lines[2].split()[1:] == ["0.600000", "0.750000", "0.200000", "0.300000"]
    assert body["instances_per_class"].get("vfd") == 1


def test_an_unclassified_box_left_unnamed_is_not_exported(client, batch,
                                                          tmp_path):
    """There is no valid label index for 'unclassified' — it must not be invented."""
    name, _root = batch
    client.post(f"/api/annotations/{name}/items/p0.jpg", json={
        "boxes": [{"index": 0, "verdict": "accepted"},
                  {"index": 1, "verdict": "accepted"}],
        "state": "reviewed"})
    dst = str(tmp_path / "corrected")
    body = client.post(f"/api/annotations/{name}/export",
                       json={"dst_root": dst}).json()

    assert body["unclassified_promoted"] == 0
    assert body["unclassified_still_unresolved"] == 1
    lines = open(os.path.join(dst, "labels", "train", "p0.txt"),
                 encoding="utf-8").read().strip().splitlines()
    assert len(lines) == 2, "the unnamed box must not be in the labels"
    assert any("where the model is blind" in w for w in body["warnings"])


def test_export_excludes_unreviewed_images_by_default(client, batch, tmp_path):
    """Un-reviewed labels are raw model output, not ground truth."""
    name, _root = batch
    client.post(f"/api/annotations/{name}/items/p0.jpg",
                json={"boxes": [{"index": 0, "verdict": "accepted"}],
                      "state": "reviewed"})
    body = client.post(f"/api/annotations/{name}/export",
                       json={"dst_root": str(tmp_path / "out")}).json()
    assert body["images_exported"] == 1
    assert body["skipped_unreviewed"] == 3
    assert any("raw model output" in w for w in body["warnings"])


def test_export_can_include_unreviewed_but_warns_loudly(client, batch, tmp_path):
    name, _root = batch
    body = client.post(f"/api/annotations/{name}/export",
                       json={"dst_root": str(tmp_path / "out"),
                             "include_unreviewed": True}).json()
    assert body["images_exported"] == 4
    assert any("RAW MODEL OUTPUT" in w for w in body["warnings"])


def test_needs_redraw_images_are_excluded_and_listed(client, batch, tmp_path):
    """Their boxes are wrong in ways this interface cannot fix."""
    name, _root = batch
    client.post(f"/api/annotations/{name}/items/p0.jpg",
                json={"boxes": [], "state": "needs_redraw",
                      "note": "boxes are all on the wrong devices"})
    client.post(f"/api/annotations/{name}/items/p1.jpg",
                json={"boxes": [{"index": 0, "verdict": "accepted"},
                                {"index": 1, "verdict": "accepted"}],
                      "state": "reviewed"})
    body = client.post(f"/api/annotations/{name}/export",
                       json={"dst_root": str(tmp_path / "out")}).json()
    assert body["skipped_needs_redraw"] == ["p0.jpg"]
    assert body["images_exported"] == 1
    assert any("needs_redraw" in w and "labelling tool" in w
               for w in body["warnings"])
    assert not os.path.exists(os.path.join(str(tmp_path / "out"), "images",
                                           "train", "p0.jpg"))


def test_export_writes_a_dataset_yaml_and_next_step(client, batch, tmp_path):
    name, _root = batch
    client.post(f"/api/annotations/{name}/items/p0.jpg",
                json={"boxes": [{"index": 0, "verdict": "accepted"}],
                      "state": "reviewed"})
    dst = str(tmp_path / "out")
    body = client.post(f"/api/annotations/{name}/export",
                       json={"dst_root": dst}).json()
    assert os.path.exists(os.path.join(dst, "dataset.yaml"))
    assert "cli split" in body["next_step"]


def test_progress_tracks_reviewed_images(client, batch):
    name, _root = batch
    for fn, state in (("p0.jpg", "reviewed"), ("p1.jpg", "needs_redraw"),
                      ("p2.jpg", "skipped")):
        client.post(f"/api/annotations/{name}/items/{fn}",
                    json={"boxes": [], "state": state})
    d = client.get(f"/api/annotations/{name}").json()
    assert d["reviewed"] == 1
    assert d["needs_redraw"] == 1
    assert d["skipped"] == 1
    assert d["remaining"] == 1
    assert d["progress"] == pytest.approx(0.75)


def test_verdicts_survive_a_fresh_service_call(client, batch):
    """State lives in the database, not in memory."""
    name, _root = batch
    client.post(f"/api/annotations/{name}/items/p2.jpg",
                json={"boxes": [{"index": 0, "verdict": "reclassified",
                                 "class_id": "relay"}],
                      "state": "reviewed", "reviewer": "a"})
    db = client.app.state.db
    detail = asvc.image_detail(db, name, "p2.jpg")
    assert detail["boxes"][0]["class_id"] == "relay"
    assert detail["state"] == "reviewed"


def test_unknown_batch_is_404(client):
    assert client.get("/api/annotations/nope").status_code == 404
    assert client.get("/api/annotations/nope/queue").status_code == 404


def test_unknown_filename_is_404(client, batch):
    name, _root = batch
    assert client.get(
        f"/api/annotations/{name}/items/nothere.jpg").status_code == 404
