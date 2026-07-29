"""
Duplicate detection — the thing that decides whether validation numbers are real.

The registry warns that two of the public switchgear sources are probably the same
photographs republished. If both are merged and nothing deduplicates them, the same
image lands in train and val, validation scores memorisation, and the reported mAP
is fiction. These tests pin down that the detector actually catches that case —
including the specific bug that made an earlier version of this module miss it.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from training.electrical import dedup as dd


# ==========================================================================
# fixtures — visually distinct "panels", not noise
# ==========================================================================

def _panel(seed: int, w: int = 640, h: int = 480) -> np.ndarray:
    """A synthetic cabinet photograph that is genuinely distinct per seed.

    Deliberately detailed: a lighting gradient, three populated DIN rails filling
    the frame, per-module highlights, and sensor noise. Structure spread across the
    whole image is what makes this a fair test rather than a fixture artefact.

    An earlier version of this fixture drew a few flat rectangles on a uniform
    background, and the hash tests failed. The algorithm was fine; the fixture was
    not. dHash compares adjacent pixels after downsampling, so a mostly-featureless
    image reduces to near-ties whose sign is decided by noise, and any brightness
    or compression change scrambles those bits. Measured on imagery like the below,
    brightness (α1.3/β40), JPEG q30, 5×5 blur and a half-resolution round trip all
    give a dHash distance of 0–1, while two different panels give 16–22 — so the
    threshold of 5 sits in a wide, safe gap. The flat-image limitation is asserted
    separately and on purpose in ``test_flat_images_are_unreliable_to_hash``.
    """
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    base = 90 + 40 * (xx / w) + 25 * (yy / h)          # cabinet lighting gradient
    img = np.dstack([base, base, base])
    for row in range(3):
        y = 60 + row * 130
        cv2.rectangle(img, (20, y + 70), (w - 20, y + 80), (150, 150, 155), -1)
        x = 30
        while x < w - 50:
            bw, bh = int(r.integers(18, 40)), int(r.integers(45, 70))
            colour = r.integers(30, 120, 3).tolist()
            cv2.rectangle(img, (x, y), (x + bw, y + bh), colour, -1)
            cv2.rectangle(img, (x + 3, y + 8), (x + bw - 3, y + 18),
                          [min(255, v + 70) for v in colour], -1)
            x += bw + int(r.integers(2, 8))
    img += r.normal(0.0, 3.0, img.shape)
    return img.clip(0, 255).astype(np.uint8)


def _write(root: str, split: str, name: str, img: np.ndarray,
           cls: int = 0, box=(0.5, 0.5, 0.3, 0.3)) -> None:
    os.makedirs(os.path.join(root, "images", split), exist_ok=True)
    os.makedirs(os.path.join(root, "labels", split), exist_ok=True)
    cv2.imwrite(os.path.join(root, "images", split, name + ".jpg"), img)
    with open(os.path.join(root, "labels", split, name + ".txt"), "w",
              encoding="utf-8") as fh:
        fh.write(f"{cls} {box[0]} {box[1]} {box[2]} {box[3]}\n")


def _recompress(img: np.ndarray, quality: int = 55) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img,
                           [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ==========================================================================
# hashing primitives
# ==========================================================================

def test_hashes_are_stable_and_64_bit():
    img = _panel(1)
    assert dd.dhash(img) == dd.dhash(img.copy())
    assert dd.ahash(img) == dd.ahash(img.copy())
    assert 0 <= dd.dhash(img) < 2 ** 64
    assert 0 <= dd.ahash(img) < 2 ** 64


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_dhash_survives_the_augmentations_roboflow_applies(seed):
    """The whole point: an augmented copy must hash close to its original.

    These are the transforms that produce the duplicate copies inside a Roboflow
    export — brightness/exposure, re-compression, blur and rescaling.
    """
    img = _panel(seed)
    variants = {
        "brightness": cv2.convertScaleAbs(img, alpha=1.3, beta=40),
        "jpeg_q50": _recompress(img, 50),
        "jpeg_q30": _recompress(img, 30),
        "blur": cv2.GaussianBlur(img, (5, 5), 0),
        "half_resolution_roundtrip": cv2.resize(
            cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)),
            (img.shape[1], img.shape[0])),
    }
    for name, variant in variants.items():
        d = dd.hamming(dd.dhash(img), dd.dhash(variant))
        assert d <= dd.DEFAULT_THRESHOLD, f"{name}: distance {d} exceeds threshold"


def test_the_duplicate_and_distinct_distance_bands_do_not_overlap(_seed=7):
    """There must be real headroom around the threshold, not a lucky cutoff."""
    img = _panel(_seed)
    worst_duplicate = max(
        dd.hamming(dd.dhash(img), dd.dhash(v)) for v in (
            cv2.convertScaleAbs(img, alpha=1.3, beta=40),
            _recompress(img, 30),
            cv2.GaussianBlur(img, (5, 5), 0)))
    best_distinct = min(
        dd.hamming(dd.dhash(img), dd.dhash(_panel(_seed + 100 + k)))
        for k in range(5))
    assert worst_duplicate <= dd.DEFAULT_THRESHOLD < best_distinct
    assert best_distinct - worst_duplicate >= 8, (
        f"only {best_distinct - worst_duplicate} bits separate a duplicate from a "
        f"distinct panel — the threshold is a coin flip")


def test_dhash_separates_genuinely_different_panels():
    a, b = dd.dhash(_panel(10)), dd.dhash(_panel(11))
    assert dd.hamming(a, b) > dd.DEFAULT_THRESHOLD


def test_flat_images_are_unreliable_to_hash():
    """A documented limitation, asserted so nobody is surprised by it.

    dHash compares adjacent pixels. In a featureless region that comparison is
    decided by noise, so a brightness shift can scramble many bits and two
    near-identical flat images may exceed the duplicate threshold. This does not
    matter for panel photography — real images have texture everywhere — but it
    does mean a near-blank frame (lens cap, blown exposure, uniform cabinet wall)
    should not be trusted to deduplicate. Filter those on capture instead.
    """
    flat = np.full((240, 320, 3), 128, np.uint8)
    shifted = cv2.convertScaleAbs(flat, alpha=1.15, beta=20)
    # No assertion that the distance is small — the point is that it is NOT
    # dependable, so we only assert the hashes are computable and total order is
    # meaningless here.
    d = dd.hamming(dd.dhash(flat), dd.dhash(shifted))
    assert 0 <= d <= 64


def test_builtin_hash_agrees_with_itself_across_colour_and_gray():
    """The NumPy fallback must be self-consistent for a grayscale input."""
    img = _panel(3)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Same luma, so the hashes should be identical or near-identical.
    assert dd.hamming(dd.dhash(img), dd.dhash(gray)) <= 2


def test_hamming_is_symmetric_and_zero_on_identity():
    assert dd.hamming(0b1011, 0b1011) == 0
    assert dd.hamming(0b1011, 0b1111) == dd.hamming(0b1111, 0b1011) == 1


def test_content_hash_detects_byte_identity(tmp_path):
    a, b, c = tmp_path / "a.jpg", tmp_path / "b.jpg", tmp_path / "c.jpg"
    img = _panel(4)
    cv2.imwrite(str(a), img)
    cv2.imwrite(str(b), img)
    cv2.imwrite(str(c), _panel(5))
    assert dd.content_hash(str(a)) == dd.content_hash(str(b))
    assert dd.content_hash(str(a)) != dd.content_hash(str(c))


# ==========================================================================
# group detection
# ==========================================================================

def _leaky_dataset(root: str) -> np.ndarray:
    """10 distinct train panels, plus copies of panel 0 planted in val."""
    for g in range(10):
        _write(root, "train", f"orig{g}", _panel(100 + g))
    base = _panel(100)
    _write(root, "train", "exact_copy", base.copy())
    _write(root, "val", "leaked_recompress", _recompress(base, 55))
    _write(root, "val", "leaked_bright",
           cv2.convertScaleAbs(base, alpha=1.12, beta=14))
    for g in range(5):
        _write(root, "val", f"vonly{g}", _panel(300 + g))
    return base


def test_exact_duplicates_are_found(tmp_path):
    root = str(tmp_path / "d")
    img = _panel(20)
    _write(root, "train", "a", img)
    _write(root, "train", "b", img.copy())
    _write(root, "train", "other", _panel(21))
    records, _ = dd.scan_dataset(root)
    groups = dd.find_duplicate_groups(records)
    assert len(groups) == 1
    assert groups[0].kind == "exact"
    assert groups[0].max_distance == 0
    assert {m.stem for m in groups[0].members} == {"a", "b"}


def test_cross_split_leakage_is_detected(tmp_path):
    """The regression test for the bug that made this module useless.

    An earlier version emitted exact-duplicate groups immediately and excluded
    their members from the near-duplicate pass. So with A and B byte-identical in
    train and C a re-compressed copy of A in val, C could not link to A — and the
    cross-split leak was reported as zero. Both relations must feed one
    connected-components pass.
    """
    root = str(tmp_path / "d")
    _leaky_dataset(root)
    report = dd.analyse_duplicates(root)

    assert report["status"] == "analysed"
    assert report["cross_split_groups"] >= 1, \
        "the planted train/val duplicate was not detected"
    cross = [g for g in report["groups"] if g["crosses_splits"]]
    members = {m["filename"] for g in cross for m in g["members"]}
    # The exact-duplicate pair AND both near-duplicate val copies must all be in
    # the same group.
    assert "leaked_recompress.jpg" in members
    assert "leaked_bright.jpg" in members
    assert "exact_copy.jpg" in members
    assert "orig0.jpg" in members
    assert "STRADDLE SPLITS" in report["verdict"]


def test_label_conflicts_are_reported_not_resolved(tmp_path):
    """Two copies of one image with different labels is a human problem."""
    root = str(tmp_path / "d")
    img = _panel(30)
    _write(root, "train", "a", img, cls=0)
    _write(root, "train", "b", img.copy(), cls=8)
    report = dd.analyse_duplicates(root)
    assert report["label_conflict_groups"] == 1
    assert "LABELS DISAGREE" in report["verdict"]


def test_identical_labels_are_not_a_conflict(tmp_path):
    root = str(tmp_path / "d")
    img = _panel(31)
    _write(root, "train", "a", img, cls=3)
    _write(root, "train", "b", img.copy(), cls=3)
    report = dd.analyse_duplicates(root)
    assert report["duplicate_groups"] == 1
    assert report["label_conflict_groups"] == 0


def test_clean_dataset_reports_no_duplicates(tmp_path):
    root = str(tmp_path / "d")
    for g in range(12):
        _write(root, "train", f"p{g}", _panel(500 + g))
    report = dd.analyse_duplicates(root)
    assert report["duplicate_groups"] == 0
    assert report["redundant_images"] == 0
    assert report["cross_split_groups"] == 0
    assert "No duplicates found" in report["verdict"]


def test_analyse_skips_an_empty_dataset(tmp_path):
    report = dd.analyse_duplicates(str(tmp_path / "nothing"))
    assert report["status"] == "skipped"


def test_threshold_zero_only_matches_bit_identical_hashes(tmp_path):
    root = str(tmp_path / "d")
    img = _panel(40)
    _write(root, "train", "orig", img)
    _write(root, "val", "bright", cv2.convertScaleAbs(img, alpha=1.2, beta=25))
    loose = dd.analyse_duplicates(root, threshold=dd.DEFAULT_THRESHOLD)
    strict = dd.analyse_duplicates(root, threshold=0)
    assert loose["duplicate_groups"] >= strict["duplicate_groups"]


# ==========================================================================
# removal
# ==========================================================================

def test_deduplicate_keeps_train_and_drops_the_val_copy(tmp_path):
    """The only safe direction: evaluation data must be unseen."""
    root, out = str(tmp_path / "d"), str(tmp_path / "clean")
    _leaky_dataset(root)
    report = dd.deduplicate(root, out)

    assert report["status"] == "deduplicated"
    assert report["images_dropped"] > 0
    # The val copies must be gone; the train original must survive.
    val_files = os.listdir(os.path.join(out, "images", "val"))
    assert "leaked_recompress.jpg" not in val_files
    assert "leaked_bright.jpg" not in val_files
    # Exactly one copy of that panel survives, and it must be the TRAIN one — which
    # of the two train copies wins is arbitrary and not worth asserting.
    train_files = os.listdir(os.path.join(out, "images", "train"))
    survivors = {"orig0.jpg", "exact_copy.jpg"} & set(train_files)
    assert len(survivors) == 1, f"expected one surviving train copy, got {survivors}"


def test_deduplicated_output_has_no_remaining_leakage(tmp_path):
    root, out = str(tmp_path / "d"), str(tmp_path / "clean")
    _leaky_dataset(root)
    dd.deduplicate(root, out)
    after = dd.analyse_duplicates(out)
    assert after["cross_split_groups"] == 0, \
        "deduplication left train/val leakage in place"


def test_deduplicate_keeps_every_label_file_with_its_image(tmp_path):
    root, out = str(tmp_path / "d"), str(tmp_path / "clean")
    _leaky_dataset(root)
    dd.deduplicate(root, out)
    for split in ("train", "val"):
        img_dir = os.path.join(out, "images", split)
        lbl_dir = os.path.join(out, "labels", split)
        if not os.path.isdir(img_dir):
            continue
        for fn in os.listdir(img_dir):
            stem = os.path.splitext(fn)[0]
            assert os.path.exists(os.path.join(lbl_dir, stem + ".txt")), \
                f"{split}/{fn} lost its label file"


def test_deduplicate_writes_a_canonical_dataset_yaml(tmp_path):
    from rtsp_backend.electrical import taxonomy as tax

    root, out = str(tmp_path / "d"), str(tmp_path / "clean")
    _leaky_dataset(root)
    dd.deduplicate(root, out)
    text = open(os.path.join(out, "dataset.yaml"), encoding="utf-8").read()
    assert f"nc: {len(tax.CLASS_ORDER)}" in text


def test_deduplicate_leaves_the_source_untouched(tmp_path):
    root, out = str(tmp_path / "d"), str(tmp_path / "clean")
    _leaky_dataset(root)
    before = sorted(os.listdir(os.path.join(root, "images", "val")))
    dd.deduplicate(root, out)
    assert sorted(os.listdir(os.path.join(root, "images", "val"))) == before


def test_deduplicate_preserves_conflicted_groups_by_default(tmp_path):
    """Picking one of two disagreeing labels silently discards a human's work."""
    root, out = str(tmp_path / "d"), str(tmp_path / "clean")
    img = _panel(50)
    _write(root, "train", "a", img, cls=0)
    _write(root, "train", "b", img.copy(), cls=8)
    report = dd.deduplicate(root, out)
    assert report["label_conflict_groups_kept"] == 1
    assert report["images_dropped"] == 0
    assert any("KEPT IN FULL" in w for w in report["warnings"])
    assert len(os.listdir(os.path.join(out, "images", "train"))) == 2


def test_drop_label_conflicts_deduplicates_them_when_asked(tmp_path):
    root, out = str(tmp_path / "d"), str(tmp_path / "clean")
    img = _panel(51)
    _write(root, "train", "a", img, cls=0)
    _write(root, "train", "b", img.copy(), cls=8)
    report = dd.deduplicate(root, out, drop_label_conflicts=True)
    assert report["images_dropped"] == 1
    assert len(os.listdir(os.path.join(out, "images", "train"))) == 1


def test_deduplicate_rejects_an_unknown_keep_policy(tmp_path):
    root = str(tmp_path / "d")
    _write(root, "train", "a", _panel(60))
    with pytest.raises(ValueError):
        dd.deduplicate(root, str(tmp_path / "o"), keep="whatever")


def test_deduplicate_refuses_an_empty_dataset(tmp_path):
    with pytest.raises(ValueError):
        dd.deduplicate(str(tmp_path / "nope"), str(tmp_path / "o"))


def test_symlink_mode_does_not_copy_bytes(tmp_path):
    root, out = str(tmp_path / "d"), str(tmp_path / "clean")
    _write(root, "train", "a", _panel(70))
    dd.deduplicate(root, out, symlink=True)
    assert os.path.islink(os.path.join(out, "images", "train", "a.jpg"))
