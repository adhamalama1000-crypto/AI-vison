"""
Dataset quality inspection.

Every problem this module finds is one that trains *silently*: a YOLO trainer skips an
unreadable image with a warning nobody reads, clamps an out-of-range box without
comment, and happily trains on a class index that does not exist in the label space.
The result is a model that is worse than it should be for reasons that never appear in
the metrics — so these tests check that each defect is actually detected, and that the
severity assigned to it matches what it costs.

The other property under test is restraint. Blur and exposure thresholds are
deliberately permissive: field panel photography is badly lit by nature, and filtering
aggressively on image statistics would discard the hardest and most valuable training
examples. Low quality is a ``warning`` that is kept by default, never a ``fatal``.
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
import pytest

from rtsp_backend.electrical import taxonomy as tax
from training.electrical import quality as ql


# ==========================================================================
# fixtures
# ==========================================================================

def _panel(seed: int, w: int = 640, h: int = 480) -> np.ndarray:
    """A sharp, well-exposed synthetic cabinet photograph."""
    r = np.random.default_rng(seed)
    img = np.full((h, w, 3), 110, np.float64)
    for i in range(8):
        x = 20 + i * 75
        cv2.rectangle(img, (x, 120), (x + 60, 260),
                      r.integers(20, 90, 3).tolist(), -1)
    img += r.normal(0.0, 4.0, img.shape)
    return img.clip(0, 255).astype(np.uint8)


def _write(root: str, split: str, name: str, img,
           label: str = "0 0.5 0.5 0.1 0.2\n") -> str:
    os.makedirs(os.path.join(root, "images", split), exist_ok=True)
    os.makedirs(os.path.join(root, "labels", split), exist_ok=True)
    path = os.path.join(root, "images", split, name + ".jpg")
    if img is not None:
        cv2.imwrite(path, img)
    if label is not None:
        with open(os.path.join(root, "labels", split, name + ".txt"), "w",
                  encoding="utf-8") as fh:
            fh.write(label)
    return path


def _clean_dataset(root: str, n: int = 6) -> str:
    for i in range(n):
        _write(root, "train", f"ok{i}", _panel(i),
               "0 0.5 0.5 0.1 0.2\n8 0.3 0.4 0.08 0.15\n")
    _write(root, "val", "okv", _panel(99))
    return root


# ==========================================================================
# image checks
# ==========================================================================

def test_a_good_image_reports_no_issues(tmp_path):
    path = str(tmp_path / "good.jpg")
    cv2.imwrite(path, _panel(1))
    dims, issues = ql.check_image(path)
    assert dims == (480, 640)
    assert issues == []


def test_a_zero_byte_file_is_fatal(tmp_path):
    path = tmp_path / "empty.jpg"
    path.write_bytes(b"")
    dims, issues = ql.check_image(str(path))
    assert dims is None
    assert issues[0][0] == "empty_file" and issues[0][1] == "fatal"


def test_a_truncated_image_is_fatal(tmp_path):
    path = tmp_path / "trunc.jpg"
    path.write_bytes(b"\xff\xd8\xff\xe0 not actually a jpeg")
    dims, issues = ql.check_image(str(path))
    assert dims is None
    assert issues[0][0] == "unreadable" and issues[0][1] == "fatal"
    assert "truncated" in issues[0][2]


def test_a_lens_cap_frame_is_flagged(tmp_path):
    path = str(tmp_path / "dark.jpg")
    cv2.imwrite(path, np.full((480, 640, 3), 4, np.uint8))
    _dims, issues = ql.check_image(path)
    codes = {c for c, _s, _d in issues}
    assert "too_dark" in codes or "featureless" in codes


def test_a_blown_out_frame_is_flagged(tmp_path):
    path = str(tmp_path / "bright.jpg")
    cv2.imwrite(path, np.full((480, 640, 3), 252, np.uint8))
    _dims, issues = ql.check_image(path)
    codes = {c for c, _s, _d in issues}
    assert "too_bright" in codes or "clipped" in codes


def test_a_tiny_image_is_flagged(tmp_path):
    path = str(tmp_path / "small.jpg")
    cv2.imwrite(path, cv2.resize(_panel(2), (90, 70)))
    _dims, issues = ql.check_image(path)
    assert "small_image" in {c for c, _s, _d in issues}


def test_low_quality_is_a_warning_never_fatal(tmp_path):
    """A dim or soft photograph of a real panel is deployment input, not a reject."""
    cases = {
        "dark": np.full((480, 640, 3), 10, np.uint8),
        "bright": np.full((480, 640, 3), 250, np.uint8),
        "blurred": cv2.GaussianBlur(_panel(3), (31, 31), 0),
        "small": cv2.resize(_panel(4), (100, 80)),
    }
    for name, img in cases.items():
        path = str(tmp_path / f"{name}.jpg")
        cv2.imwrite(path, img)
        dims, issues = ql.check_image(path)
        assert dims is not None, f"{name} must stay usable"
        assert all(s != "fatal" for _c, s, _d in issues), \
            f"{name} was marked fatal; low quality must be a warning"


# ==========================================================================
# label checks
# ==========================================================================

def test_valid_labels_parse_with_no_issues(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("0 0.5 0.5 0.1 0.2\n8 0.3 0.4 0.08 0.15\n")
    rows, issues = ql.check_labels(str(path), len(tax.CLASS_ORDER))
    assert len(rows) == 2
    assert issues == []


def test_a_class_index_outside_the_label_space_is_fatal(tmp_path):
    """The failure that silently ruins a model: the trainer accepts it."""
    path = tmp_path / "a.txt"
    path.write_text("999 0.5 0.5 0.1 0.1\n")
    rows, issues = ql.check_labels(str(path), len(tax.CLASS_ORDER))
    assert rows == []
    assert issues[0][0] == "class_out_of_range" and issues[0][1] == "fatal"


def test_absolute_pixel_labels_are_fatal(tmp_path):
    """Unnormalised labels are a common merge mistake."""
    path = tmp_path / "a.txt"
    path.write_text("0 320 240 60 80\n")
    rows, issues = ql.check_labels(str(path), len(tax.CLASS_ORDER))
    assert rows == []
    assert issues[0][0] == "centre_out_of_range"
    assert "not normalised" in issues[0][2]


def test_a_malformed_row_is_fatal(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("0 0.5 0.5\n")
    _rows, issues = ql.check_labels(str(path), len(tax.CLASS_ORDER))
    assert issues[0][0] == "malformed_row" and issues[0][1] == "fatal"


def test_a_non_numeric_row_is_fatal(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("0 abc 0.5 0.1 0.1\n")
    _rows, issues = ql.check_labels(str(path), len(tax.CLASS_ORDER))
    assert issues[0][0] == "unparseable_row"


def test_a_degenerate_box_is_fatal(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("0 0.5 0.5 0 0.1\n")
    rows, issues = ql.check_labels(str(path), len(tax.CLASS_ORDER))
    assert rows == []
    assert issues[0][0] == "degenerate_box"


def test_a_whole_frame_box_is_a_warning(tmp_path):
    """Usually a cabinet box mislabelled as a device — worth flagging, not fatal."""
    path = tmp_path / "a.txt"
    path.write_text("0 0.5 0.5 0.98 0.97\n")
    rows, issues = ql.check_labels(str(path), len(tax.CLASS_ORDER))
    assert len(rows) == 1, "the box is still usable"
    codes = {c: s for c, s, _d in issues}
    assert codes.get("box_covers_frame") == "warning"


def test_a_missing_label_file_is_info_not_an_error(tmp_path):
    """An unlabelled image is a legitimate negative example."""
    rows, issues = ql.check_labels(str(tmp_path / "nope.txt"),
                                   len(tax.CLASS_ORDER))
    assert rows == []
    assert issues[0][0] == "no_label_file" and issues[0][1] == "info"


def test_an_empty_label_file_is_info(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("")
    _rows, issues = ql.check_labels(str(path), len(tax.CLASS_ORDER))
    assert issues[0][0] == "empty_annotation" and issues[0][1] == "info"


def test_a_box_past_the_image_edge_is_a_warning(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("0 0.05 0.5 0.4 0.2\n")
    rows, issues = ql.check_labels(str(path), len(tax.CLASS_ORDER))
    assert len(rows) == 1
    assert "box_outside_frame" in {c for c, _s, _d in issues}


# ==========================================================================
# dataset scan
# ==========================================================================

def test_a_clean_dataset_reports_clean(tmp_path):
    root = _clean_dataset(str(tmp_path / "d"))
    report = ql.inspect(root, log=lambda m: None)
    assert report.images == 7
    assert report.fatal_files == []
    assert report.per_severity.get("fatal", 0) == 0
    assert "structurally usable" in report.verdict


def test_every_defect_class_is_detected(tmp_path):
    root = str(tmp_path / "d")
    _clean_dataset(root)
    (tmp_path / "d" / "images" / "train" / "zero.jpg").write_bytes(b"")
    _write(root, "train", "badclass", _panel(20), "999 0.5 0.5 0.1 0.1\n")
    _write(root, "train", "absolute", _panel(21), "0 320 240 60 80\n")
    _write(root, "train", "malformed", _panel(22), "0 0.5 0.5\n")
    _write(root, "train", "wholeframe", _panel(23), "0 0.5 0.5 0.98 0.97\n")
    _write(root, "train", "nolabel", _panel(24), label=None)

    report = ql.inspect(root, log=lambda m: None)
    codes = report.per_code
    for expected in ("empty_file", "class_out_of_range", "centre_out_of_range",
                     "malformed_row", "box_covers_frame", "no_label_file"):
        assert expected in codes, f"{expected} was not detected"
    fatal_names = {os.path.basename(f) for f in report.fatal_files}
    assert {"zero.jpg", "badclass.jpg", "absolute.jpg",
            "malformed.jpg"} <= fatal_names
    # A whole-frame box and a missing label are NOT fatal.
    assert "wholeframe.jpg" not in fatal_names
    assert "nolabel.jpg" not in fatal_names


def test_the_out_of_range_class_recommendation_names_the_cause(tmp_path):
    root = str(tmp_path / "d")
    _clean_dataset(root)
    _write(root, "train", "bad", _panel(30), "999 0.5 0.5 0.1 0.1\n")
    report = ql.inspect(root, log=lambda m: None)
    joined = " ".join(report.recommendations)
    assert "remap" in joined, "the fix (re-run cli remap) must be stated"


def test_class_balance_is_measured(tmp_path):
    root = str(tmp_path / "d")
    # 20 instances of class 0, 1 of class 8 → a 20:1 imbalance.
    for i in range(10):
        _write(root, "train", f"a{i}", _panel(i),
               "0 0.5 0.5 0.1 0.2\n0 0.3 0.3 0.1 0.1\n")
    _write(root, "train", "rare", _panel(50), "8 0.5 0.5 0.1 0.1\n")
    report = ql.inspect(root, log=lambda m: None)
    balance = report.class_balance
    assert balance["classes_present"] == 2
    assert balance["imbalance_ratio"] == pytest.approx(20.0)
    assert balance["imbalanced"] is True
    assert any("imbalance" in r for r in report.recommendations)


def test_absent_classes_are_reported(tmp_path):
    root = _clean_dataset(str(tmp_path / "d"))
    report = ql.inspect(root, log=lambda m: None)
    assert report.class_balance["classes_absent"] == len(tax.CLASS_ORDER) - 2
    assert any("cli gap" in r for r in report.recommendations)


def test_inspect_of_an_empty_root_is_honest(tmp_path):
    report = ql.inspect(str(tmp_path / "nothing"), log=lambda m: None)
    assert report.images == 0
    assert "No images found" in report.verdict


def test_no_pixels_mode_skips_decoding(tmp_path):
    """Structural-only mode must still catch label problems."""
    root = str(tmp_path / "d")
    _clean_dataset(root)
    (tmp_path / "d" / "images" / "train" / "zero.jpg").write_bytes(b"")
    _write(root, "train", "badclass", _panel(20), "999 0.5 0.5 0.1 0.1\n")
    report = ql.inspect(root, check_pixels=False, log=lambda m: None)
    assert "class_out_of_range" in report.per_code
    # The zero-byte file is not decoded, so its image defect is not reported...
    assert "empty_file" not in report.per_code
    # ...and the report is still serialisable.
    json.dumps(report.to_dict())


def test_report_is_json_serialisable(tmp_path):
    root = _clean_dataset(str(tmp_path / "d"))
    d = ql.inspect(root, log=lambda m: None).to_dict()
    json.dumps(d)
    assert "thresholds" in d and "verdict" in d


# ==========================================================================
# cleaning
# ==========================================================================

def test_clean_removes_only_the_unusable(tmp_path):
    root, out = str(tmp_path / "d"), str(tmp_path / "out")
    _clean_dataset(root)
    (tmp_path / "d" / "images" / "train" / "zero.jpg").write_bytes(b"")
    _write(root, "train", "badclass", _panel(20), "999 0.5 0.5 0.1 0.1\n")
    _write(root, "train", "blurry", cv2.GaussianBlur(_panel(21), (31, 31), 0))

    res = ql.clean(root, out, log=lambda m: None)
    kept = os.listdir(os.path.join(out, "images", "train"))
    assert "zero.jpg" not in kept
    assert "badclass.jpg" not in kept
    # The blurred image is kept by default.
    assert "blurry.jpg" in kept
    assert res["images_dropped"] == 2


def test_clean_can_drop_named_warning_codes(tmp_path):
    root, out = str(tmp_path / "d"), str(tmp_path / "out")
    _clean_dataset(root)
    _write(root, "train", "blurry", cv2.GaussianBlur(_panel(21), (31, 31), 0))
    res = ql.clean(root, out, drop_warnings=("blurred",), log=lambda m: None)
    assert "blurry.jpg" not in os.listdir(os.path.join(out, "images", "train"))
    assert res["dropped_warning_codes"] == ["blurred"]


def test_clean_quarantines_with_reasons(tmp_path):
    root, out = str(tmp_path / "d"), str(tmp_path / "out")
    _clean_dataset(root)
    _write(root, "train", "badclass", _panel(20), "999 0.5 0.5 0.1 0.1\n")
    res = ql.clean(root, out, log=lambda m: None)
    assert res["quarantine"]
    assert os.path.exists(os.path.join(out, "quarantine", "train", "badclass.jpg"))
    reasons = json.load(open(os.path.join(out, "quarantine", "_reasons.json"),
                             encoding="utf-8"))
    assert any("badclass" in k for k in reasons)


def test_the_cleaned_dataset_is_actually_clean(tmp_path):
    root, out = str(tmp_path / "d"), str(tmp_path / "out")
    _clean_dataset(root)
    (tmp_path / "d" / "images" / "train" / "zero.jpg").write_bytes(b"")
    _write(root, "train", "badclass", _panel(20), "999 0.5 0.5 0.1 0.1\n")
    _write(root, "train", "absolute", _panel(21), "0 320 240 60 80\n")
    ql.clean(root, out, log=lambda m: None)
    after = ql.inspect(out, log=lambda m: None)
    assert after.fatal_files == [], "cleaning left unusable files behind"


def test_clean_leaves_the_source_untouched(tmp_path):
    root, out = str(tmp_path / "d"), str(tmp_path / "out")
    _clean_dataset(root)
    _write(root, "train", "badclass", _panel(20), "999 0.5 0.5 0.1 0.1\n")
    before = sorted(os.listdir(os.path.join(root, "images", "train")))
    ql.clean(root, out, log=lambda m: None)
    assert sorted(os.listdir(os.path.join(root, "images", "train"))) == before


def test_clean_pairs_every_image_with_a_label(tmp_path):
    root, out = str(tmp_path / "d"), str(tmp_path / "out")
    _clean_dataset(root)
    _write(root, "train", "nolabel", _panel(40), label=None)
    ql.clean(root, out, log=lambda m: None)
    for split in ("train", "val"):
        img_dir = os.path.join(out, "images", split)
        if not os.path.isdir(img_dir):
            continue
        for fn in os.listdir(img_dir):
            stem = os.path.splitext(fn)[0]
            assert os.path.exists(os.path.join(out, "labels", split,
                                               stem + ".txt"))


def test_clean_writes_a_canonical_dataset_yaml(tmp_path):
    root, out = str(tmp_path / "d"), str(tmp_path / "out")
    _clean_dataset(root)
    ql.clean(root, out, log=lambda m: None)
    text = open(os.path.join(out, "dataset.yaml"), encoding="utf-8").read()
    assert f"nc: {len(tax.CLASS_ORDER)}" in text
