"""
Unit tests for the industrial panel vision engine (rtsp_backend.panels).

Uses a synthetic but realistic control-panel image: a grey backplate, a grey
terminal rail, and several coloured wires. These assert *real* behaviour of the
detectors (geometry, colour, stability, fault detection), not stubs.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from rtsp_backend.panels import (
    wire_detector, terminal_detector, features, graph, template,
    comparison, datasheet, overlay,
)


def make_panel(shift=0, drop_green=False, extra=False):
    """A synthetic panel: grey backplate + terminal rail + coloured wires."""
    img = np.full((400, 600, 3), 60, np.uint8)
    cv2.rectangle(img, (40, 340), (560, 370), (170, 170, 170), -1)   # terminal rail
    cv2.line(img, (100 + shift, 100), (100 + shift, 345), (0, 0, 200), 4)   # red
    if not drop_green:
        cv2.line(img, (200, 120), (400, 120), (0, 180, 0), 4)               # green
    cv2.line(img, (300, 150), (300, 345), (200, 120, 0), 4)                 # blue
    cv2.line(img, (430, 150), (520, 300), (0, 220, 220), 3)                 # yellow
    if extra:
        cv2.line(img, (150, 200), (250, 300), (0, 120, 255), 4)             # orange
    return img


def jpg_roundtrip(img, q=90):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)


# --------------------------------------------------------------------------
# wire detector
# --------------------------------------------------------------------------

def test_wire_detector_finds_colored_wires():
    wires = wire_detector.detect_wires(make_panel())
    assert 4 <= len(wires) <= 6           # 4 real wires, tolerate <=2 phantoms
    colors = {w.color for w in wires}
    assert {"red", "green", "blue"} <= colors
    # every wire has full geometry
    for w in wires:
        assert len(w.polyline) >= 2
        assert w.length > 0
        assert w.thickness > 0
        d = w.to_dict()
        for k in ("start", "end", "polyline", "length", "thickness", "color", "direction"):
            assert k in d


def test_wire_lengths_are_physically_sane():
    wires = {w.color: w for w in wire_detector.detect_wires(make_panel())}
    # red ≈245px, green ≈200px, blue ≈195px (allow generous tolerance)
    assert 200 <= wires["red"].length <= 320
    assert 150 <= wires["green"].length <= 260
    assert 150 <= wires["blue"].length <= 260


def test_wire_detector_stable_under_jpeg():
    a = wire_detector.detect_wires(make_panel())
    b = wire_detector.detect_wires(jpg_roundtrip(make_panel()))
    # count stable within a small margin
    assert abs(len(a) - len(b)) <= 2


def test_wire_detector_flat_image_yields_nothing():
    flat = np.full((400, 600, 3), 120, np.uint8)
    assert wire_detector.detect_wires(flat) == []


def test_wire_detector_handles_empty():
    assert wire_detector.detect_wires(np.zeros((0, 0, 3), np.uint8)) == []


def test_dominant_color_name():
    red = np.full((10, 10, 3), (0, 0, 220), np.uint8)
    assert wire_detector.dominant_color_name(red) == "red"
    black = np.full((10, 10, 3), (10, 10, 10), np.uint8)
    assert wire_detector.dominant_color_name(black) == "black"


# --------------------------------------------------------------------------
# terminal detector
# --------------------------------------------------------------------------

def test_terminal_detector_finds_rail_and_screws():
    terms = terminal_detector.detect_terminals(make_panel())
    assert len(terms) >= 2
    kinds = {t.kind for t in terms}
    assert "block" in kinds or "screw" in kinds
    for t in terms:
        d = t.to_dict()
        assert d["x"] is not None and d["y"] is not None


def test_terminal_detector_uses_component_boxes():
    comps = [{"ref_id": "C0", "comp_type": "terminal_block",
              "bbox": [40, 340, 560, 370], "cx": 300, "cy": 355}]
    terms = terminal_detector.detect_terminals(make_panel(), comps)
    assert any(t.component_ref == "C0" for t in terms)


# --------------------------------------------------------------------------
# features / alignment
# --------------------------------------------------------------------------

def test_feature_extraction_and_self_alignment():
    img = make_panel()
    feats = features.extract_features(img)
    assert feats["n_keypoints"] > 0
    assert "descriptors_b64" in feats
    al = features.align(feats, img)
    assert al["ok"] is True
    assert al["inliers"] > 8
    # a point maps to ~itself under the self-homography
    x, y = features.warp_point(al["homography"], 100.0, 100.0)
    assert abs(x - 100) < 15 and abs(y - 100) < 15


def test_alignment_without_descriptors_is_graceful():
    al = features.align({}, make_panel())
    assert al["ok"] is False
    assert al["note"]


# --------------------------------------------------------------------------
# graph
# --------------------------------------------------------------------------

def test_build_graph_nodes_and_edges():
    comps = [{"ref_id": "C0", "comp_type": "mcb", "cx": 100, "cy": 100, "label": "mcb"}]
    terms = [{"ref_id": "T0", "x": 100, "y": 340, "kind": "block"}]
    wires = [{"wire_uid": "w0", "from_terminal": "T0", "to_component": "C0",
              "color": "red", "start": [100, 100], "end": [100, 340]}]
    g = graph.build_graph(comps, terms, wires)
    assert g["node_count"] == 2
    assert g["edge_count"] == 1
    adj = graph.adjacency(g)
    assert "T0" in adj["C0"] or "C0" in adj["T0"]


# --------------------------------------------------------------------------
# template + comparison (the inspection core)
# --------------------------------------------------------------------------

def test_template_analyze_without_component_model():
    result = template.analyze_image(None, make_panel())
    assert result["component_total"] == 0          # no model => honest empty
    # wire tracing is off by default: it produced hundreds of false "wires"
    assert result["wire_total"] == 0
    assert any("wire tracing disabled" in n for n in result["notes"])
    assert result["terminal_total"] >= 1
    assert any("no trained component model" in n for n in result["notes"])
    assert result["graph"]["node_count"] >= 1


def test_template_wire_tracing_opt_in_still_available():
    """The experimental tracer must still be reachable for research, explicitly."""
    result = template.analyze_image(None, make_panel(),
                                    wire_params={"enabled": True})
    assert result["wire_total"] >= 1
    assert any("explicitly enabled" in n for n in result["notes"])


def test_build_template_multi_image():
    built = template.build_template(None, [make_panel(), make_panel()],
                                    wire_params={"enabled": True})
    tmpl = built["template"]
    assert tmpl["n_images"] == 2
    assert tmpl["wires"]
    assert "descriptors_b64" in built["features"]


def test_compare_identical_passes():
    built = template.build_template(None, [make_panel()])
    tmpl, feats = built["template"], built["features"]
    obs = template.analyze_image(None, make_panel())
    result = comparison.compare(tmpl, obs, make_panel(), feats)
    assert result["status"] in ("pass", "warning")
    assert result["score"] >= 0.7


def test_compare_missing_wire_detected():
    """Wire comparison is experimental and opt-in; exercise it explicitly."""
    on = {"enabled": True}
    built = template.build_template(None, [make_panel()], wire_params=on)
    obs = template.analyze_image(None, make_panel(drop_green=True), wire_params=on)
    result = comparison.compare(tmpl := built["template"], obs, make_panel(drop_green=True),
                                built["features"])
    types = {e["error_type"] for e in result["errors"]}
    assert "missing_wire" in types
    assert result["status"] == "fail"
    # every error carries a confidence in [0,1]
    for e in result["errors"]:
        assert 0.0 <= e["confidence"] <= 1.0


def test_compare_extra_wire_detected():
    on = {"enabled": True}
    built = template.build_template(None, [make_panel()], wire_params=on)
    obs = template.analyze_image(None, make_panel(extra=True), wire_params=on)
    result = comparison.compare(built["template"], obs, make_panel(extra=True),
                                built["features"])
    types = {e["error_type"] for e in result["errors"]}
    assert "extra_wire" in types


def test_overlay_renders():
    on = {"enabled": True}
    built = template.build_template(None, [make_panel()], wire_params=on)
    obs = template.analyze_image(None, make_panel(drop_green=True), wire_params=on)
    result = comparison.compare(built["template"], obs, make_panel(drop_green=True),
                                built["features"])
    img = overlay.draw_overlay(make_panel(drop_green=True), obs, result)
    assert img.shape == (400, 600, 3)


# --------------------------------------------------------------------------
# datasheet
# --------------------------------------------------------------------------

def test_datasheet_parse_text():
    parsed = datasheet.parse_text(
        "Q1 -> KM1. Fuse F2 protects motor. X1:1 - K1. Selector S1. Lamp H1. Wire W12.")
    assert "Q1" in parsed["component_ids"]
    assert "KM1" in parsed["component_ids"]
    assert "X1:1" in parsed["terminal_ids"]
    assert parsed["n_connections"] >= 2


def test_datasheet_expected_graph():
    parsed = datasheet.parse_text("Q1 -> KM1. KM1 -> M1.")
    g = datasheet.build_expected_graph(parsed)
    assert g["edge_count"] == 2
    assert g["node_count"] >= 3


def test_datasheet_extract_textfile(tmp_path):
    p = tmp_path / "sld.txt"
    p.write_text("Q1 -> KM1. X1:1 - K1. F2 fuse.")
    out = datasheet.extract(str(p))
    assert out["ocr_engine"] == "text"
    assert out["parsed"]["n_components"] >= 3
    assert out["expected_graph"]["node_count"] >= 1
