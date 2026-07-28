"""
Recogniser backends: preprocessing, output decoding, class-map resolution and
the honest failure modes.

The decoding tests are regression tests for the bug that made a trained model
useless: the old decoder chose its output format from the raw column count
(``>= 85`` meant YOLOv5), which is only true for an 80-class COCO model. With
the 53-class electrical taxonomy every label came out shifted by one.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from rtsp_backend.ai import registry
from rtsp_backend.electrical import recognizer as rec
from rtsp_backend.electrical import taxonomy as tax


# ==========================================================================
# preprocessing
# ==========================================================================

def test_letterbox_preserves_aspect_ratio_and_pads():
    img = np.zeros((480, 1280, 3), np.uint8)
    out, r, dw, dh = rec.letterbox(img, 640)
    assert out.shape == (640, 640, 3)
    assert r == pytest.approx(0.5)
    assert dh > 0 and dw == 0          # wide image → vertical padding


def test_letterbox_never_upscales_beyond_one():
    img = np.zeros((100, 100, 3), np.uint8)
    _, r, _, _ = rec.letterbox(img, 640)
    assert r == pytest.approx(6.4)     # ratio may exceed 1 for a small image
    big = np.zeros((2000, 2000, 3), np.uint8)
    _, r2, _, _ = rec.letterbox(big, 640)
    assert r2 < 1.0


def test_to_blob_shape_and_range():
    img = np.full((64, 64, 3), 255, np.uint8)
    blob = rec.to_blob(img)
    assert blob.shape == (1, 3, 64, 64)
    assert blob.dtype == np.float32
    assert blob.max() == pytest.approx(1.0)


def test_to_blob_converts_bgr_to_rgb():
    img = np.zeros((2, 2, 3), np.uint8)
    img[:, :, 0] = 255                 # pure blue in BGR
    blob = rec.to_blob(img)
    assert blob[0, 2].max() == pytest.approx(1.0)   # blue -> RGB channel 2
    assert blob[0, 0].max() == pytest.approx(0.0)   # red channel stays empty


# ==========================================================================
# class-map resolution — the labels.txt disaster
# ==========================================================================

def test_numeric_labels_txt_is_rejected(tmp_path):
    """The shipped labels.txt was the literal lines 0..9.

    Honouring it produced components labelled "0"…"9". It must be rejected in
    favour of the canonical taxonomy order.
    """
    d = tmp_path / "components"
    d.mkdir()
    (d / "labels.txt").write_text("\n".join(str(i) for i in range(10)))
    names, source = rec.load_class_map(str(tmp_path))
    assert source == "taxonomy"
    assert names == list(tax.CLASS_ORDER)


def test_real_labels_txt_is_honoured(tmp_path):
    d = tmp_path / "components"
    d.mkdir()
    (d / "labels.txt").write_text("contactor\nmcb\nplc\n")
    names, source = rec.load_class_map(str(tmp_path))
    assert source == "labels.txt"
    assert names == ["contactor", "mcb", "plc"]


def test_classes_json_wins_over_labels_txt(tmp_path):
    d = tmp_path / "components"
    d.mkdir()
    (d / "labels.txt").write_text("wrong\nalso_wrong\n")
    (d / "classes.json").write_text(json.dumps({"classes": ["contactor", "mcb"]}))
    names, source = rec.load_class_map(str(tmp_path))
    assert source == "classes.json"
    assert names == ["contactor", "mcb"]


def test_explicit_labels_param_wins():
    names, source = rec.load_class_map("nonexistent", explicit=["a", "b"])
    assert source == "params" and names == ["a", "b"]


def test_shipped_classes_json_matches_the_taxonomy():
    """models/components/classes.json is the authoritative label map."""
    path = os.path.join("models", "components", "classes.json")
    assert os.path.exists(path), "classes.json must ship with the repo"
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["classes"] == list(tax.CLASS_ORDER)
    assert not os.path.exists(os.path.join("models", "components", "labels.txt")), \
        "the numeric-placeholder labels.txt must not come back"


def test_resolve_names_flags_unmappable_classes():
    canon, unmapped = rec.resolve_names(["contactor", "mcb", "zorp_widget"])
    assert canon == ["contactor", "mcb", None]
    assert unmapped == ["zorp_widget"]


# ==========================================================================
# decoding — YOLOv8 / v11 layout
# ==========================================================================

def _v8_output(nc: int, entries):
    """Build a (1, 4+nc, N) Ultralytics-style output tensor."""
    n = len(entries)
    arr = np.zeros((4 + nc, n), np.float32)
    for i, (cid, score, (cx, cy, w, h)) in enumerate(entries):
        arr[0, i], arr[1, i], arr[2, i], arr[3, i] = cx, cy, w, h
        arr[4 + cid, i] = score
    return arr[None, ...]


def test_decode_v8_maps_classes_correctly():
    nc = len(tax.CLASS_ORDER)
    out = _v8_output(nc, [(7, 0.9, (100, 100, 40, 60)),
                          (0, 0.8, (300, 200, 30, 50))])
    boxes, scores, ids = rec.decode_yolo(out, nc, 0.25, 1.0, 0.0, 0.0)
    assert list(ids) == [7, 0]
    assert scores[0] == pytest.approx(0.9)
    assert boxes[0].tolist() == pytest.approx([80.0, 70.0, 120.0, 130.0])


def test_decode_applies_letterbox_inverse():
    nc = 5
    out = _v8_output(nc, [(1, 0.9, (200, 200, 100, 100))])
    boxes, _, _ = rec.decode_yolo(out, nc, 0.25, r=0.5, dw=20.0, dh=10.0)
    # x1 = (200 - 50 - 20) / 0.5 = 260 ; y1 = (200 - 50 - 10) / 0.5 = 280
    assert boxes[0].tolist() == pytest.approx([260.0, 280.0, 460.0, 480.0])


def test_decode_respects_the_floor():
    nc = 5
    out = _v8_output(nc, [(1, 0.9, (200, 200, 40, 40)),
                          (2, 0.05, (300, 300, 40, 40))])
    _, scores, _ = rec.decode_yolo(out, nc, 0.25, 1.0, 0.0, 0.0)
    assert len(scores) == 1


# ==========================================================================
# decoding — YOLOv5 layout: the label-shift regression
# ==========================================================================

def _v5_output(nc: int, entries):
    """Build an (N, 4+1+nc) YOLOv5-style output tensor."""
    arr = np.zeros((len(entries), 4 + 1 + nc), np.float32)
    for i, (cid, score, (cx, cy, w, h)) in enumerate(entries):
        arr[i, 0:4] = (cx, cy, w, h)
        arr[i, 4] = 0.95                         # objectness
        arr[i, 5 + cid] = score
    return arr


def test_decode_v5_with_taxonomy_class_count_is_not_shifted():
    """The exact case the old decoder got wrong: 53 classes → 58 columns."""
    nc = len(tax.CLASS_ORDER)
    assert nc + 5 != 85, "if this ever equals 85 the regression test is void"
    out = _v5_output(nc, [(12, 0.9, (100, 100, 40, 60)),
                          (33, 0.85, (300, 200, 50, 70))])
    _, _, ids = rec.decode_yolo(out, nc, 0.25, 1.0, 0.0, 0.0)
    assert list(ids) == [12, 33]


def test_legacy_column_heuristic_would_have_shifted_every_label():
    """Documents the old behaviour so the fix cannot silently regress."""
    nc = len(tax.CLASS_ORDER)
    out = _v5_output(nc, [(12, 0.9, (100, 100, 40, 60))])
    row = out[0]
    # the old code: `if row.shape[0] >= 85` -> False for 58 columns, so it took
    # the v8 branch and argmax'd over columns 4..end, which includes objectness
    legacy_cls = int(np.argmax(row[4:]))
    assert legacy_cls == 0, "objectness (0.95) wins the argmax → everything is class 0"
    _, _, ids = rec.decode_yolo(out, nc, 0.25, 1.0, 0.0, 0.0)
    assert int(ids[0]) == 12


def test_decode_v5_with_80_classes_still_works():
    """The COCO case the old heuristic happened to get right must not break."""
    nc = 80
    out = _v5_output(nc, [(41, 0.9, (100, 100, 40, 60))])
    _, _, ids = rec.decode_yolo(out, nc, 0.25, 1.0, 0.0, 0.0)
    assert int(ids[0]) == 41


def test_decode_handles_garbage_gracefully():
    for bad in (np.zeros((0,), np.float32), np.zeros((3, 2), np.float32),
                np.zeros((2, 2, 2, 2), np.float32)):
        boxes, scores, ids = rec.decode_yolo(bad, 53, 0.25, 1.0, 0.0, 0.0)
        assert boxes.shape == (0, 4) and len(scores) == 0 and len(ids) == 0


# ==========================================================================
# decoding — RT-DETR
# ==========================================================================

def test_decode_rtdetr_two_tensor_export():
    nc = 6
    boxes_n = np.array([[0.5, 0.5, 0.2, 0.4], [0.25, 0.25, 0.1, 0.1]], np.float32)
    logits = np.full((2, nc), -5.0, np.float32)
    logits[0, 3] = 5.0
    logits[1, 1] = 4.0
    boxes, scores, ids = rec.decode_rtdetr([boxes_n, logits], nc, 0.25,
                                           1000.0, 800.0)
    assert list(ids) == [3, 1]
    assert scores[0] > 0.9
    assert boxes[0].tolist() == pytest.approx([400.0, 240.0, 600.0, 560.0])


def test_decode_rtdetr_accepts_already_sigmoided_scores():
    nc = 4
    boxes_n = np.array([[0.5, 0.5, 0.2, 0.2]], np.float32)
    probs = np.array([[0.01, 0.02, 0.95, 0.02]], np.float32)
    _, scores, ids = rec.decode_rtdetr([boxes_n, probs], nc, 0.25, 100.0, 100.0)
    assert int(ids[0]) == 2 and scores[0] == pytest.approx(0.95)


def test_decode_rtdetr_rejects_single_tensor():
    boxes, _, _ = rec.decode_rtdetr([np.zeros((2, 4), np.float32)], 4, 0.25,
                                    100.0, 100.0)
    assert boxes.shape == (0, 4)


# ==========================================================================
# backend registration and honest failure
# ==========================================================================

def test_all_backends_registered():
    cat = registry.catalog()["components"]
    ids = {b["backend_id"] for b in cat}
    assert {"industrial_onnx", "industrial_ultralytics", "openvocab_owlv2",
            "openvocab_grounding_dino", "openvocab_florence2",
            "industrial_ensemble", "industrial_disabled"} <= ids


def test_onnx_backend_without_weights_reports_precisely(tmp_path):
    b = rec.OnnxIndustrialRecognizer(models_dir=str(tmp_path))
    with pytest.raises(RuntimeError):
        b.load()
    st = b.status()
    assert st["ready"] is False
    assert st["reason"] == "weights_missing"
    assert "training/electrical" in st["error"]
    assert st["class_count"] == len(tax.CLASS_ORDER)
    assert st["class_source"] == "taxonomy"


def test_ultralytics_backend_without_weights_reports_precisely(tmp_path):
    b = rec.UltralyticsIndustrialRecognizer(models_dir=str(tmp_path))
    with pytest.raises(RuntimeError):
        b.load()
    assert b.status()["reason"] == "weights_missing"


def test_openvocab_reports_missing_dependency_not_a_crash():
    b = rec.Owlv2Recognizer()
    try:
        b.load()
    except RuntimeError:
        pass
    st = b.status()
    if not st["ready"]:
        assert st["reason"] in ("transformers_missing", "weights_unavailable")
        assert st["error"]


def test_openvocab_prompts_come_from_the_taxonomy():
    b = rec.Owlv2Recognizer()
    pairs = b.prompt_pairs()
    assert len(pairs) > 50
    assert all(cid in tax.SPECS for _, cid in pairs)


def test_openvocab_prompt_filtering_by_class():
    b = rec.Owlv2Recognizer(classes=["contactor", "plc"])
    assert {cid for _, cid in b.prompt_pairs()} == {"contactor", "plc"}


def test_disabled_backend_is_ready_and_returns_nothing():
    b = rec.NullIndustrialRecognizer()
    b.load()
    assert b.ready is True
    frame = np.zeros((100, 100, 3), np.uint8)
    assert b.infer(frame) == []
    assert b.recognize(frame).accepted == []


def test_ensemble_without_members_fails_loudly(tmp_path):
    b = rec.EnsembleRecognizer(models_dir=str(tmp_path),
                               members=["industrial_onnx"])
    with pytest.raises(RuntimeError):
        b.load()
    assert b.status()["reason"] == "no_members_loaded"


def test_ensemble_loads_when_one_member_works(tmp_path):
    b = rec.EnsembleRecognizer(models_dir=str(tmp_path),
                               members=["industrial_onnx",
                                        "industrial_disabled"])
    b.load()
    assert b.ready is True
    # the failing member's reason is retained, not swallowed
    assert "industrial_onnx" in (b.status()["error"] or "")


def test_gate_config_from_params():
    b = rec.NullIndustrialRecognizer(strictness=1.5, nms_iou=0.6,
                                     unknown_floor=0.3, max_detections=25,
                                     thresholds={"contactor": 0.77})
    cfg = b.gate_config()
    assert cfg.strictness == 1.5
    assert cfg.nms_iou == 0.6
    assert cfg.unknown_floor == 0.3
    assert cfg.max_detections == 25
    assert cfg.thresholds["contactor"] == 0.77


# ==========================================================================
# ensemble fusion
# ==========================================================================

def test_fusion_boosts_cross_source_agreement():
    from rtsp_backend.electrical import postprocess as pp

    cands = [
        pp.Candidate("contactor", 0.60, (100, 100, 180, 200), source="a"),
        pp.Candidate("contactor", 0.55, (102, 102, 182, 202), source="b"),
    ]
    fused = rec.fuse(cands)
    assert all(f.score > 0.60 for f in fused)
    assert fused[0].extra["corroborated_by"] == ["b"]


def test_fusion_does_not_boost_same_source_duplicates():
    from rtsp_backend.electrical import postprocess as pp

    cands = [
        pp.Candidate("contactor", 0.60, (100, 100, 180, 200), source="a"),
        pp.Candidate("contactor", 0.55, (102, 102, 182, 202), source="a"),
    ]
    fused = rec.fuse(cands)
    assert fused[0].score == pytest.approx(0.60)
    assert "corroborated_by" not in fused[0].extra


def test_fusion_does_not_boost_disagreement():
    from rtsp_backend.electrical import postprocess as pp

    cands = [
        pp.Candidate("contactor", 0.60, (100, 100, 180, 200), source="a"),
        pp.Candidate("plc", 0.60, (100, 100, 180, 200), source="b"),
    ]
    fused = rec.fuse(cands)
    assert all(f.score == pytest.approx(0.60) for f in fused)
