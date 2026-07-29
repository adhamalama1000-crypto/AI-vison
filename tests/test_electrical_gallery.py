"""Tests for the false-positive / false-negative galleries."""

from __future__ import annotations

import os

import cv2
import numpy as np

from training.electrical import gallery as gal

SHAPE = (400, 600)


def _img(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = np.full((SHAPE[0], SHAPE[1], 3), 120, np.uint8)
    cv2.rectangle(img, (100, 100), (160, 220), (60, 60, 60), -1)
    cv2.imwrite(path, img)


def _gt(image_id, cid, box):
    return {"image_id": image_id, "class_id": cid, "box": box}


def _pred(image_id, cid, box, score):
    return {"image_id": image_id, "class_id": cid, "box": box, "score": score}


def _dataset(tmp_path):
    image_dir = str(tmp_path / "images")
    for name in ("a", "b"):
        _img(os.path.join(image_dir, f"{name}.jpg"))
    return image_dir


def test_galleries_render_crops_and_contact_sheets(tmp_path):
    image_dir = _dataset(tmp_path)
    out = str(tmp_path / "gallery")
    gts = [
        _gt("a", "mcb", (100, 100, 160, 220)),        # matched
        _gt("a", "contactor", (300, 100, 380, 200)),  # confused -> FN + FP
        _gt("b", "relay", (50, 50, 120, 160)),        # missed entirely -> FN
    ]
    preds = [
        _pred("a", "mcb", (100, 100, 160, 220), 0.91),        # TP
        _pred("a", "relay", (300, 100, 380, 200), 0.77),      # class_confusion FP
        _pred("a", "plc", (450, 250, 520, 330), 0.55),        # spurious FP
    ]
    res = gal.write_galleries(gts, preds, image_dir, out, top=10, cols=3)

    fp, fn = res["false_positives"], res["false_negatives"]
    assert fp["total"] == 2
    assert fp["rendered"] == 2
    assert fp["by_cause_all"]["class_confusion"] == 1
    assert fp["by_cause_all"]["spurious_detection"] == 1
    assert os.path.exists(fp["contact_sheet"])

    assert fn["total"] == 2
    assert fn["rendered"] == 2
    assert set(fn["by_class_all"]) == {"contactor", "relay"}
    assert os.path.exists(fn["contact_sheet"])

    # crops land under a per-cause / per-class subdirectory
    assert any(f.startswith("class_confusion") for f in fp["files"])
    assert any(f.startswith("spurious_detection") for f in fp["files"])
    for rel in fp["files"]:
        assert os.path.exists(os.path.join(out, "false_positives", rel))
    for rel in fn["files"]:
        assert os.path.exists(os.path.join(out, "false_negatives", rel))


def test_false_positives_are_ordered_most_confident_first(tmp_path):
    """A confident mistake is the damaging one — it reaches the report."""
    image_dir = _dataset(tmp_path)
    out = str(tmp_path / "g")
    preds = [
        _pred("a", "plc", (10, 10, 60, 60), 0.42),
        _pred("a", "vfd", (200, 200, 260, 280), 0.95),
        _pred("a", "relay", (300, 40, 340, 120), 0.61),
    ]
    res = gal.write_galleries([], preds, image_dir, out, top=10)
    names = [os.path.basename(f) for f in res["false_positives"]["files"]]
    assert names[0].startswith("001_vfd_0.95")
    assert names[1].startswith("002_relay_0.61")
    assert names[2].startswith("003_plc_0.42")


def test_false_negatives_are_ordered_largest_first(tmp_path):
    image_dir = _dataset(tmp_path)
    out = str(tmp_path / "g")
    gts = [
        _gt("a", "mcb", (10, 10, 30, 40)),          # small
        _gt("a", "vfd", (100, 100, 300, 350)),      # large
        _gt("b", "relay", (10, 10, 90, 110)),       # medium
    ]
    res = gal.write_galleries(gts, [], image_dir, out, top=10)
    names = [os.path.basename(f) for f in res["false_negatives"]["files"]]
    assert names[0].startswith("001_vfd")
    assert names[1].startswith("002_relay")
    assert names[2].startswith("003_mcb")


def test_top_limits_rendering_but_not_the_reported_total(tmp_path):
    """Truncating the gallery must not understate how many failures there were."""
    image_dir = _dataset(tmp_path)
    out = str(tmp_path / "g")
    preds = [_pred("a", "plc", (10 + i, 10, 40 + i, 60), 0.5 + i / 100.0)
             for i in range(9)]
    res = gal.write_galleries([], preds, image_dir, out, top=3)
    assert res["false_positives"]["total"] == 9
    assert res["false_positives"]["rendered"] == 3


def test_a_missing_image_is_skipped_without_crashing(tmp_path):
    image_dir = _dataset(tmp_path)
    out = str(tmp_path / "g")
    preds = [_pred("nope", "plc", (10, 10, 60, 60), 0.9),
             _pred("a", "vfd", (10, 10, 60, 60), 0.8)]
    res = gal.write_galleries([], preds, image_dir, out, top=10)
    assert res["false_positives"]["total"] == 2
    assert res["false_positives"]["rendered"] == 1


def test_empty_inputs_produce_no_contact_sheet(tmp_path):
    image_dir = _dataset(tmp_path)
    res = gal.write_galleries([], [], image_dir, str(tmp_path / "g"))
    assert res["false_positives"]["total"] == 0
    assert res["false_positives"]["contact_sheet"] is None
    assert res["false_negatives"]["contact_sheet"] is None


def test_find_image_matches_any_supported_extension(tmp_path):
    d = str(tmp_path / "im")
    os.makedirs(d, exist_ok=True)
    cv2.imwrite(os.path.join(d, "x.png"), np.zeros((10, 10, 3), np.uint8))
    assert gal.find_image(d, "x").endswith("x.png")
    assert gal.find_image(d, "missing") is None


def test_a_tiny_box_is_widened_so_the_crop_is_inspectable(tmp_path):
    img = np.full((300, 300, 3), 90, np.uint8)
    crop, origin = gal._crop(img, (150, 150, 156, 158), min_side=64)
    assert crop is not None
    assert crop.shape[0] >= 60 and crop.shape[1] >= 60


def test_a_box_outside_the_image_is_reported_as_uncroppable(tmp_path):
    img = np.full((100, 100, 3), 90, np.uint8)
    crop, _ = gal._crop(img, (500, 500, 520, 520))
    assert crop is None
