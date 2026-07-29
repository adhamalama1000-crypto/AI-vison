"""
SAM2 box refinement — and specifically its guards.

SAM2 cannot be run in CI (no GPU, no checkpoint, no model-hub route), so what is
tested here is everything that decides whether a refinement is *safe to accept*.
That is the part that matters: SAM has no idea what a contactor is, and on panel
imagery it fails in predictable ways — segmenting the whole DIN-rail row because the
modules are visually continuous, segmenting only the toggle lever, or segmenting the
neighbouring device. Accepting those silently would corrupt the labels the tool
exists to produce, and corrupt them in a way that looks like tighter boxes.

So the guards are the contract, and a rejected refinement must always fall back to
the detector's original box rather than losing the detection.
"""

from __future__ import annotations

import numpy as np
import pytest

from rtsp_backend.electrical import taxonomy as tax
from training.electrical import refine as rf


# ==========================================================================
# availability and honest degradation
# ==========================================================================

def test_refiner_never_raises_when_sam_is_missing():
    """A 500-image batch must not die halfway because SAM is not installed."""
    r = rf.SamRefiner(log=lambda m: None)
    assert isinstance(r.ready, bool)
    if not r.ready:
        assert r.reason and "SAM" in r.reason
        assert "pip install" in r.reason, "the reason must say how to fix it"


def test_unavailable_refiner_returns_original_boxes_untouched():
    r = rf.SamRefiner(log=lambda m: None)
    if r.ready:
        pytest.skip("a SAM backend is installed in this environment")
    img = np.zeros((200, 300, 3), np.uint8)
    boxes = [(10.0, 20.0, 60.0, 90.0), (100.0, 30.0, 140.0, 95.0)]
    out = r.refine(img, boxes, ["mcb", "contactor"])
    assert len(out) == len(boxes)
    for original, refined in zip(boxes, out):
        assert refined.accepted is False
        assert refined.reason == "sam_unavailable"
        # The critical property: the usable box is still the original.
        assert refined.box == original


def test_refine_with_no_boxes_is_a_no_op():
    r = rf.SamRefiner(log=lambda m: None)
    assert r.refine(np.zeros((10, 10, 3), np.uint8), [], []) == []


# ==========================================================================
# the guards — checked directly, since they are the safety contract
# ==========================================================================

ORIGINAL = (100.0, 100.0, 140.0, 180.0)      # 40x80 — a plausible MCB


def _check(refined, class_id="mcb"):
    return rf.SamRefiner._check(ORIGINAL, refined, class_id)


def test_a_modest_tightening_is_accepted():
    ok, reason = _check((104.0, 106.0, 137.0, 176.0))
    assert ok and reason == "ok"


def test_growing_beyond_the_limit_is_rejected():
    """The DIN-rail-row failure: SAM segments every module as one object."""
    ok, reason = _check((20.0, 100.0, 600.0, 180.0))
    assert not ok and reason == "grew_beyond_limit"


def test_collapsing_to_a_subpart_is_rejected():
    """The toggle-lever failure: SAM segments the switch, not the device."""
    ok, reason = _check((115.0, 105.0, 125.0, 120.0))
    assert not ok and reason == "collapsed_to_subpart"


def test_drifting_onto_the_neighbour_is_rejected():
    """Same size, same aspect, wrong device."""
    ok, reason = _check((160.0, 100.0, 200.0, 180.0))
    assert not ok and reason == "centre_drifted"


def test_a_degenerate_mask_is_rejected():
    ok, reason = _check((120.0, 140.0, 120.5, 140.5))
    assert not ok and reason == "degenerate_mask"


def test_an_implausible_aspect_ratio_is_rejected():
    """Reuses the taxonomy's geometric prior rather than a second rule set."""
    # Same area as the original but 8:1 wide — an MCB is never that shape.
    ok, reason = _check((60.0, 130.0, 240.0, 152.0))
    assert not ok
    assert reason in ("aspect_implausible", "grew_beyond_limit",
                      "centre_drifted")


def test_the_aspect_guard_uses_the_real_taxonomy_prior():
    """The same shape is plausible for a duct and implausible for an MCB.

    The guard reads the per-class band from the taxonomy rather than hardcoding a
    second set of geometry rules, so this must hold without touching refine.py.
    """
    assert tax.SPECS["wire_duct"].aspect_ratio[1] > 4, \
        "wire_duct should permit a long thin box"
    assert tax.SPECS["mcb"].aspect_ratio[1] < 4, \
        "an MCB should not permit a long thin box"

    # A long thin duct-shaped original, refined slightly tighter — same geometry,
    # two different classes.
    original = (100.0, 125.0, 270.0, 155.0)
    refined = (102.0, 128.0, 266.0, 152.0)
    ok_duct, reason_duct = rf.SamRefiner._check(original, refined, "wire_duct")
    _ok_mcb, reason_mcb = rf.SamRefiner._check(original, refined, "mcb")

    assert ok_duct and reason_duct == "ok"
    assert reason_mcb == "aspect_implausible"


def test_an_unknown_class_skips_the_aspect_guard_but_not_the_others():
    """No prior exists for the unknown class, so geometry cannot be judged."""
    ok, reason = rf.SamRefiner._check(
        ORIGINAL, (104.0, 106.0, 137.0, 176.0), tax.UNKNOWN_COMPONENT_ID)
    assert ok and reason == "ok"
    # But size and drift guards still apply.
    ok2, reason2 = rf.SamRefiner._check(
        ORIGINAL, (20.0, 100.0, 600.0, 180.0), tax.UNKNOWN_COMPONENT_ID)
    assert not ok2 and reason2 == "grew_beyond_limit"


def test_a_degenerate_original_is_rejected_rather_than_dividing_by_zero():
    ok, reason = rf.SamRefiner._check((10.0, 10.0, 10.0, 10.0),
                                      (10.0, 10.0, 40.0, 60.0), "mcb")
    assert not ok and reason in ("degenerate_original", "grew_beyond_limit")


@pytest.mark.parametrize("growth", [1.0, 1.2, 1.55])
def test_growth_within_the_limit_is_allowed(growth):
    """Refinement may legitimately grow a box the detector cropped too tightly."""
    w, h = 40.0 * growth ** 0.5, 80.0 * growth ** 0.5
    cx, cy = 120.0, 140.0
    ok, reason = _check((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    assert ok, reason


# ==========================================================================
# mask geometry
# ==========================================================================

def test_mask_box_is_the_tight_bounds_of_the_mask():
    mask = np.zeros((100, 100), bool)
    mask[20:41, 30:51] = True
    assert rf.SamRefiner._mask_box(mask) == (30.0, 20.0, 51.0, 41.0)


def test_mask_box_of_an_empty_mask_is_none():
    assert rf.SamRefiner._mask_box(np.zeros((10, 10), bool)) is None


def test_iou_is_sane():
    assert rf._iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert rf._iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert 0.0 < rf._iou((0, 0, 10, 10), (5, 5, 15, 15)) < 1.0


def test_iou_of_degenerate_boxes_is_zero_not_an_exception():
    assert rf._iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


# ==========================================================================
# reporting — "is SAM actually helping?" must have a number
# ==========================================================================

def _box(cid, orig, ref, accepted, reason="ok"):
    return rf.RefinedBox(class_id=cid, original=orig, refined=ref,
                         accepted=accepted, reason=reason,
                         iou=rf._iou(orig, ref))


def test_summary_of_nothing_is_honest():
    s = rf.refine_summary([])
    assert s["boxes"] == 0
    assert s["accept_rate"] is None


def test_summary_reports_accept_rate_and_rejection_reasons():
    boxes = [
        _box("mcb", (0, 0, 40, 80), (4, 4, 36, 76), True),
        _box("mcb", (0, 0, 40, 80), (0, 0, 400, 80), False,
             "grew_beyond_limit"),
        _box("relay", (0, 0, 40, 80), (0, 0, 400, 80), False,
             "grew_beyond_limit"),
        _box("relay", (0, 0, 40, 80), (10, 10, 20, 20), False,
             "collapsed_to_subpart"),
    ]
    s = rf.refine_summary(boxes)
    assert s["boxes"] == 4
    assert s["accepted"] == 1
    assert s["accept_rate"] == 0.25
    assert s["rejected_reasons"]["grew_beyond_limit"] == 2
    assert s["rejected_reasons"]["collapsed_to_subpart"] == 1


def test_summary_warns_when_refinement_is_not_helping():
    """A low accept rate must be called out, not buried in a number."""
    boxes = [_box("mcb", (0, 0, 40, 80), (0, 0, 400, 80), False,
                  "grew_beyond_limit") for _ in range(10)]
    s = rf.refine_summary(boxes)
    assert s["accept_rate"] == 0.0
    assert "not helping" in s["interpretation"]
    assert "DIN-rail" in s["interpretation"]


def test_summary_warns_when_refinement_changes_nothing():
    """High accept rate but near-identical boxes is wasted compute."""
    boxes = [_box("mcb", (0, 0, 40, 80), (0, 0, 40, 80), True)
             for _ in range(10)]
    s = rf.refine_summary(boxes)
    assert s["accept_rate"] == 1.0
    assert s["mean_iou_original_vs_refined"] == pytest.approx(1.0)
    assert "changing almost nothing" in s["interpretation"]


def test_summary_measures_the_tightening():
    boxes = [_box("mcb", (0, 0, 40, 80), (5, 10, 35, 70), True)
             for _ in range(4)]
    s = rf.refine_summary(boxes)
    assert s["boxes_tightened"] == 4
    # 30x60 = 1800 out of 40x80 = 3200 → ~44% area reduction.
    assert s["mean_area_reduction"] == pytest.approx(1 - 1800 / 3200, abs=1e-6)


def test_refined_box_falls_back_to_the_original_when_rejected():
    b = _box("mcb", (0, 0, 40, 80), (0, 0, 400, 80), False, "grew_beyond_limit")
    assert b.box == (0, 0, 40, 80)
    d = b.to_dict()
    assert d["accepted"] is False and d["reason"] == "grew_beyond_limit"


# ==========================================================================
# integration with auto-annotation
# ==========================================================================

def test_autolabel_reports_refinement_status(tmp_path):
    import cv2
    from training.electrical import autolabel as al

    images = tmp_path / "imgs"
    images.mkdir()
    cv2.imwrite(str(images / "a.jpg"), np.full((120, 160, 3), 90, np.uint8))
    manifest = al.autolabel_directory(
        str(images), str(tmp_path / "out"), backends=("null_components",),
        refine_boxes=True)
    if manifest["status"] != "labelled":
        pytest.skip(f"no usable backend: {manifest.get('reason')}")
    ref = manifest["box_refinement"]
    # Whether or not SAM is installed, the manifest must say which it was.
    assert "enabled" in ref and "reason" in ref


def test_autolabel_can_disable_refinement(tmp_path):
    import cv2
    from training.electrical import autolabel as al

    images = tmp_path / "imgs"
    images.mkdir()
    cv2.imwrite(str(images / "a.jpg"), np.full((120, 160, 3), 90, np.uint8))
    manifest = al.autolabel_directory(
        str(images), str(tmp_path / "out"), backends=("null_components",),
        refine_boxes=False)
    if manifest["status"] != "labelled":
        pytest.skip(f"no usable backend: {manifest.get('reason')}")
    assert manifest["box_refinement"]["enabled"] is False
    assert manifest["box_refinement"]["reason"] == "not requested"
