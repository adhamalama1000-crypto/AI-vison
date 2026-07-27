"""
The false-positive suppression cascade — the core of the fix.

These tests pin down the specific behaviours that were broken before:
per-class (not class-agnostic) NMS, geometric plausibility rejection, the honest
"Unknown Industrial Component" demotion instead of a guess, and row structure
recovery. The quantitative before/after comparison lives in
``scripts/validate_panel_inspector.py``; here we assert the mechanisms.
"""

from __future__ import annotations

import pytest

from rtsp_backend.electrical import postprocess as pp
from rtsp_backend.electrical import taxonomy as tax

SHAPE = (768, 1024)   # h, w


def cand(cid, score, box, **kw):
    return pp.Candidate(cid, score, box, **kw)


# --------------------------------------------------------------------------
# geometry primitives
# --------------------------------------------------------------------------

def test_iou_basic():
    assert pp.iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert pp.iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert pp.iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3, abs=1e-6)


def test_containment():
    assert pp.containment((2, 2, 4, 4), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert pp.containment((0, 0, 10, 10), (2, 2, 4, 4)) == pytest.approx(0.04)
    assert pp.containment((0, 0, 0, 0), (0, 0, 10, 10)) == 0.0


# --------------------------------------------------------------------------
# stage 1 — sanitise
# --------------------------------------------------------------------------

def test_sanitise_drops_degenerate_and_clips():
    diag = pp.Diagnostics()
    out = pp.sanitise([
        cand("contactor", 0.9, (100, 100, 180, 200)),      # fine
        cand("contactor", 0.9, (10, 10, 12, 400)),         # 2px wide sliver
        cand("contactor", 0.9, (-50, -50, 2000, 2000)),    # clipped to frame
        cand("contactor", 0.9, (200, 100, 100, 200)),      # inverted -> fixed
        cand("contactor", float("nan"), (0, 0, 50, 50)),   # bad score
        cand("contactor", 0.9, (float("inf"), 0, 50, 50)),  # non-finite box
    ], SHAPE, diag)
    assert diag.dropped["degenerate_box"] == 1
    assert diag.dropped["invalid_score"] == 1
    assert diag.dropped["non_finite_box"] == 1
    assert len(out) == 3
    for c in out:
        assert 0 <= c.box[0] <= c.box[2] <= SHAPE[1]
        assert 0 <= c.box[1] <= c.box[3] <= SHAPE[0]


# --------------------------------------------------------------------------
# stage 2 — per-class NMS
# --------------------------------------------------------------------------

def test_nms_removes_same_class_duplicates():
    diag = pp.Diagnostics()
    out = pp.nms_per_class([
        cand("contactor", 0.9, (100, 100, 180, 200)),
        cand("contactor", 0.7, (104, 104, 184, 204)),   # same device
    ], 0.5, diag)
    assert len(out) == 1
    assert out[0].score == pytest.approx(0.9)
    assert diag.dropped["nms_same_class"] == 1


def test_nms_is_per_class_so_stacked_devices_both_survive():
    """An overload relay bolted under a contactor overlaps it heavily.

    The old class-agnostic NMS destroyed one of them. This is the regression.
    """
    out = pp.nms_per_class([
        cand("contactor", 0.9, (100, 100, 180, 210)),
        cand("overload_relay", 0.85, (100, 190, 180, 265)),
    ], 0.3)
    assert {c.class_id for c in out} == {"contactor", "overload_relay"}


def test_full_cascade_keeps_contactor_and_overload_stack():
    res = pp.run([
        cand("contactor", 0.90, (100, 100, 180, 205)),
        cand("overload_relay", 0.86, (100, 200, 180, 270)),
    ], SHAPE)
    assert {c.class_id for c in res.accepted} == {"contactor", "overload_relay"}


# --------------------------------------------------------------------------
# stage 3 — cross-class dedupe
# --------------------------------------------------------------------------

def test_confusable_double_claim_resolved_by_score():
    diag = pp.Diagnostics()
    out = pp.dedupe_across_classes([
        cand("mcb", 0.55, (100, 100, 130, 160)),
        cand("mccb", 0.82, (101, 101, 131, 161)),   # same device, higher score
    ], 0.65, 0.8, diag)
    assert len(out) == 1
    assert out[0].class_id == "mccb"
    assert diag.dropped["duplicate_class_claim"] == 1


def test_unrelated_overlap_is_preserved():
    """A CT around a busbar overlaps it completely and both are real."""
    out = pp.dedupe_across_classes([
        cand("busbar", 0.8, (100, 100, 900, 130)),
        cand("current_transformer", 0.7, (400, 95, 460, 140)),
    ], 0.65, 0.8)
    assert len(out) == 2


def test_confusable_groups_are_symmetric_and_disjoint_enough():
    for group in pp.CONFUSABLE_GROUPS:
        for cid in group:
            assert cid in tax.SPECS or cid == tax.UNKNOWN_COMPONENT_ID, cid
    assert pp._confusable("mcb", "mccb")
    assert not pp._confusable("mcb", "plc")
    assert pp._confusable("plc", "plc")


# --------------------------------------------------------------------------
# stage 4 — geometric plausibility
# --------------------------------------------------------------------------

def test_sliver_claimed_as_plc_is_rejected():
    cfg = pp.GateConfig()
    area = SHAPE[0] * SHAPE[1]
    ok, reason = pp.plausible(cand("plc", 0.95, (0, 300, 1000, 306)), area, cfg)
    assert not ok
    assert reason == "implausible_aspect_ratio"


def test_full_frame_box_claimed_as_indicator_lamp_is_rejected():
    cfg = pp.GateConfig()
    area = SHAPE[0] * SHAPE[1]
    ok, reason = pp.plausible(
        cand("indicator_lamp", 0.99, (0, 0, 1024, 768)), area, cfg)
    assert not ok
    assert reason == "implausible_too_large"


def test_plausible_device_passes():
    cfg = pp.GateConfig()
    area = SHAPE[0] * SHAPE[1]
    ok, reason = pp.plausible(cand("contactor", 0.9, (100, 100, 180, 200)),
                              area, cfg)
    assert ok and reason == ""


def test_unknown_class_bypasses_plausibility():
    cfg = pp.GateConfig()
    area = SHAPE[0] * SHAPE[1]
    ok, _ = pp.plausible(
        cand(tax.UNKNOWN_COMPONENT_ID, 0.5, (0, 0, 1024, 700)), area, cfg)
    assert ok


def test_plausibility_can_be_disabled_for_debugging():
    cfg = pp.GateConfig(check_plausibility=False)
    out = pp.plausibility_gate([cand("plc", 0.9, (0, 300, 1000, 306))],
                               SHAPE, cfg)
    assert len(out) == 1


# --------------------------------------------------------------------------
# stage 5 — confidence gate / honest unknown
# --------------------------------------------------------------------------

def test_low_confidence_becomes_unknown_not_a_guess():
    cfg = pp.GateConfig()
    diag = pp.Diagnostics()
    thr = cfg.threshold_for("contactor")
    out = pp.confidence_gate(
        [cand("contactor", thr - 0.05, (100, 100, 180, 200))], cfg, diag)
    assert len(out) == 1
    assert out[0].class_id == tax.UNKNOWN_COMPONENT_ID
    assert out[0].extra["demoted_from"] == "contactor"
    assert "threshold" in out[0].extra["demotion_reason"]
    assert diag.relabelled_unknown == 1


def test_very_low_confidence_is_dropped_entirely():
    cfg = pp.GateConfig()
    diag = pp.Diagnostics()
    out = pp.confidence_gate(
        [cand("contactor", 0.05, (100, 100, 180, 200))], cfg, diag)
    assert out == []
    assert diag.dropped["below_unknown_floor"] == 1


def test_confident_detection_keeps_its_class():
    out = pp.confidence_gate([cand("contactor", 0.95, (100, 100, 180, 200))],
                             pp.GateConfig())
    assert out[0].class_id == "contactor"


def test_strictness_dial_raises_every_threshold():
    lax = pp.GateConfig(strictness=1.0)
    strict = pp.GateConfig(strictness=1.8)
    assert strict.threshold_for("contactor") > lax.threshold_for("contactor")
    score = lax.threshold_for("contactor") + 0.01
    assert pp.confidence_gate([cand("contactor", score, (100, 100, 180, 200))],
                             lax)[0].class_id == "contactor"
    assert pp.confidence_gate([cand("contactor", score, (100, 100, 180, 200))],
                             strict)[0].class_id == tax.UNKNOWN_COMPONENT_ID


def test_per_class_threshold_override():
    cfg = pp.GateConfig()
    cfg.thresholds = {**cfg.thresholds, "contactor": 0.9}
    assert cfg.threshold_for("contactor") == pytest.approx(0.9)


# --------------------------------------------------------------------------
# stage 6 — rows
# --------------------------------------------------------------------------

def test_group_rows_recovers_din_rail_structure():
    cands = [
        cand("mcb", 0.9, (100, 100, 130, 160)),
        cand("mcb", 0.9, (140, 102, 170, 162)),
        cand("mcb", 0.9, (180, 100, 210, 160)),
        cand("relay", 0.9, (100, 300, 140, 360)),
        cand("relay", 0.9, (150, 302, 190, 362)),
    ]
    rows = pp.group_rows(cands)
    assert len(rows) == 2
    assert len(rows[0]) == 3 and len(rows[1]) == 2
    # each row is ordered left to right
    for row in rows:
        xs = [cands[i].center[0] for i in row]
        assert xs == sorted(xs)


def test_group_rows_handles_empty():
    assert pp.group_rows([]) == []


# --------------------------------------------------------------------------
# the whole cascade
# --------------------------------------------------------------------------

def test_cascade_suppresses_junk_and_keeps_real_devices():
    real = [
        cand("contactor", 0.91, (100, 100, 180, 200)),
        cand("contactor", 0.88, (200, 100, 280, 200)),
        cand("plc", 0.82, (400, 100, 600, 200)),
    ]
    junk = [cand("contactor", 0.30, (i * 7, 400 + i, i * 7 + 600, 404 + i))
            for i in range(120)]
    res = pp.run(real + junk, SHAPE)

    assert {c.class_id for c in res.accepted} == {"contactor", "plc"}
    assert len(res.accepted) == 3
    d = res.diagnostics
    assert d.input_count == 123
    assert d.output_count == 3
    assert d.dropped_total == 120
    assert d.suppression_rate() > 0.95


def test_cascade_returns_reading_order():
    res = pp.run([
        cand("relay", 0.9, (500, 400, 540, 460)),     # row 2, right
        cand("mcb", 0.9, (100, 100, 130, 160)),       # row 1, left
        cand("mcb", 0.9, (300, 100, 330, 160)),       # row 1, right
    ], SHAPE)
    xs_ys = [c.center for c in res.accepted]
    assert xs_ys[0][1] < xs_ys[2][1]                  # top row first
    assert xs_ys[0][0] < xs_ys[1][0]                  # left to right within row
    assert res.rows == [[0, 1], [2]]


def test_detection_cap_is_reported_not_hidden():
    cfg = pp.GateConfig(max_detections=5)
    cands = [cand("mcb", 0.9, (20 * i, 100, 20 * i + 18, 160))
             for i in range(20)]
    res = pp.run(cands, SHAPE, cfg)
    assert len(res.accepted) == 5
    assert res.truncated is True
    assert res.diagnostics.dropped["over_detection_cap"] == 15


def test_empty_input_is_safe():
    res = pp.run([], SHAPE)
    assert res.accepted == []
    assert res.rows == []
    assert res.diagnostics.suppression_rate() == 0.0


# --------------------------------------------------------------------------
# aggregation helpers
# --------------------------------------------------------------------------

def test_counts_and_confidence_stats():
    cands = [
        cand("mcb", 0.8, (0, 0, 20, 60)),
        cand("mcb", 0.6, (30, 0, 50, 60)),
        cand(tax.UNKNOWN_COMPONENT_ID, 0.3, (60, 0, 90, 60)),
    ]
    assert pp.counts(cands) == {"mcb": 2, tax.UNKNOWN_COMPONENT_ID: 1}
    stats = pp.confidence_stats(cands)
    assert stats["count"] == 3
    assert stats["max"] == pytest.approx(0.8)
    assert stats["below_0_5"] == 1
    assert stats["unknown"] == 1


def test_confidence_stats_empty():
    stats = pp.confidence_stats([])
    assert stats["count"] == 0 and stats["mean"] is None


@pytest.mark.parametrize("box,expected", [
    ((10, 10, 20, 20), "top-left"),
    ((500, 380, 520, 400), "middle-center"),
    ((1000, 700, 1020, 760), "bottom-right"),
])
def test_panel_position(box, expected):
    assert pp.panel_position(box, SHAPE) == expected


def test_diagnostics_serialisation():
    diag = pp.Diagnostics(input_count=10, output_count=4)
    diag.dropped["nms_same_class"] = 6
    d = diag.to_dict()
    assert d["dropped_total"] == 6
    assert d["suppression_rate"] == pytest.approx(0.6)
