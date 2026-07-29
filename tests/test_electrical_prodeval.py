"""Tests for the production-path evaluator and the acceptance sweep.

These build candidate caches by hand rather than running a model, so the gate
replay, the metric accounting and the sweep ranking are tested in isolation from
whether any checkpoint happens to exist.

``check_plausibility=False`` is used wherever the subject of the test is the
confidence/threshold accounting: the geometric gate is exercised directly in
:mod:`tests.test_electrical_postprocess`, and leaving it on here would make the
expected counts depend on the taxonomy's aspect-ratio priors.
"""

import pytest

from rtsp_backend.electrical import postprocess as pp
from rtsp_backend.electrical import taxonomy as tax
from training.electrical import prodeval as pe

MCB_THR = tax.spec("mcb").min_conf
BOX = (100.0, 100.0, 140.0, 200.0)
SHAPE = (1000, 1000)


def _cand(cid: str, score: float, box=BOX) -> pp.Candidate:
    return pp.Candidate(class_id=cid, score=float(score), box=tuple(box))


def _cache(images, base=0.01) -> pe.CandidateCache:
    """``images`` is a sequence of (image_id, [Candidate])."""
    return pe.CandidateCache(
        images=[pe.ImageCandidates(image_id=i, shape=SHAPE, candidates=list(c))
                for i, c in images],
        backend_id="test", split="val", base_decode_floor=base)


def _gt(image_id: str, cid: str = "mcb", box=BOX) -> dict:
    return {"image_id": image_id, "class_id": cid, "box": tuple(box)}


# --------------------------------------------------------------------------
# replay_gate / decode floor
# --------------------------------------------------------------------------

def test_decode_floor_filters_candidates_and_is_accounted_for():
    cache = _cache([("a", [_cand("mcb", 0.9), _cand("mcb", 0.03, (300, 100, 340, 200))])])
    gated = pe.replay_gate(cache, 0.05, pe.gate_config(0.18, check_plausibility=False))
    assert gated["raw_candidates"] == 2
    assert gated["kept_after_decode_floor"] == 1
    assert gated["dropped_below_decode_floor"] == 1
    assert len(gated["asserted"]) == 1


def test_replay_at_lower_floor_keeps_more():
    cache = _cache([("a", [_cand("mcb", 0.9), _cand("mcb", 0.03, (300, 100, 340, 200))])])
    cfg = pe.gate_config(0.02, check_plausibility=False)
    low = pe.replay_gate(cache, 0.01, cfg)
    high = pe.replay_gate(cache, 0.05, cfg)
    assert low["kept_after_decode_floor"] > high["kept_after_decode_floor"]


def test_production_report_refuses_a_floor_below_the_cached_one():
    cache = _cache([("a", [_cand("mcb", 0.9)])], base=0.10)
    with pytest.raises(ValueError, match="below the 0.1"):
        pe.production_report([_gt("a")], cache, 0.05, 0.18)


# --------------------------------------------------------------------------
# metric accounting
# --------------------------------------------------------------------------

def test_perfect_detections_score_perfectly():
    gts = [_gt("a"), _gt("b")]
    cache = _cache([("a", [_cand("mcb", 0.9)]), ("b", [_cand("mcb", 0.9)])])
    prod = pe.production_report(gts, cache, 0.05, 0.18,
                               check_plausibility=False)["production"]
    assert prod["precision"] == 1.0
    assert prod["recall"] == 1.0
    assert prod["false_positives"] == 0
    assert prod["false_negatives"] == 0
    assert prod["fp_per_image"] == 0.0
    assert prod["fn_per_image"] == 0.0
    assert prod["accepted_asserted"] == 2
    assert prod["accepted_unknown"] == 0
    assert prod["unknown_rate"] == 0.0
    assert prod["images"] == 2


def test_low_confidence_becomes_an_abstention_not_a_wrong_answer():
    """Between the unknown floor and the class threshold the box is kept as
    unknown: recall collapses, but it is recorded as an abstention, and the
    class-agnostic recall shows the device was still localised."""
    gts = [_gt("a")]
    score = (0.18 + MCB_THR) / 2          # above unknown_floor, below threshold
    cache = _cache([("a", [_cand("mcb", score)])])
    prod = pe.production_report(gts, cache, 0.05, 0.18,
                               check_plausibility=False)["production"]
    assert prod["accepted_asserted"] == 0
    assert prod["accepted_unknown"] == 1
    assert prod["unknown_rate"] == 1.0
    assert prod["recall"] == 0.0
    assert prod["recall_localised"] == 1.0
    assert prod["classification_shortfall"] == 1.0


def test_below_the_unknown_floor_nothing_is_kept():
    gts = [_gt("a")]
    cache = _cache([("a", [_cand("mcb", 0.12)])])
    prod = pe.production_report(gts, cache, 0.05, 0.18,
                               check_plausibility=False)["production"]
    assert prod["accepted_detections"] == 0
    assert prod["unknown_rate"] == 0.0
    assert prod["recall"] == 0.0
    assert prod["dropped_by_reason"].get("below_unknown_floor") == 1


def test_fn_per_image_is_divided_by_the_image_count():
    """Regression guard: the per-image rates must use the number of images
    evaluated as the denominator. Dividing by the ground-truth or prediction
    count makes the figure move when the gate changes and means nothing."""
    gts = [_gt(i) for i in ("a", "b", "c", "d")]
    cache = _cache([(i, []) for i in ("a", "b", "c", "d")])
    prod = pe.production_report(gts, cache, 0.05, 0.18,
                               check_plausibility=False)["production"]
    assert prod["images"] == 4
    assert prod["false_negatives"] == 4
    assert prod["fn_per_image"] == 1.0


def test_ground_truth_is_aligned_to_the_images_actually_inferred():
    """With --limit, predictions cover a subset while load_ground_truth reads the
    whole split. Scoring the two against each other turns every unevaluated
    image's boxes into false negatives and reports a recall that describes the
    limit rather than the model."""
    gts = [_gt("a")] + [_gt(f"skipped_{i}") for i in range(9)]
    cache = _cache([("a", [_cand("mcb", 0.9)])])          # only image "a" inferred
    prod = pe.production_report(gts, cache, 0.05, 0.18,
                               check_plausibility=False)["production"]
    assert prod["ground_truth"] == 1
    assert prod["ground_truth_outside_evaluated_images"] == 9
    assert prod["recall"] == 1.0
    assert prod["false_negatives"] == 0
    assert prod["fn_per_image"] == 0.0


def test_fp_per_image_counts_spurious_boxes_per_image():
    gts = [_gt("a")]
    cache = _cache([("a", [_cand("mcb", 0.9),
                           _cand("mcb", 0.85, (500, 100, 540, 200)),
                           _cand("mcb", 0.80, (700, 100, 740, 200))])])
    prod = pe.production_report(gts, cache, 0.05, 0.18,
                               check_plausibility=False)["production"]
    assert prod["accepted_asserted"] == 3
    assert prod["false_positives"] == 2
    assert prod["fp_per_image"] == 2.0


def test_rejected_total_accounts_for_both_stages():
    cache = _cache([("a", [_cand("mcb", 0.9),
                           _cand("mcb", 0.02, (300, 100, 340, 200)),
                           _cand("mcb", 0.12, (500, 100, 540, 200))])])
    gated = pe.replay_gate(cache, 0.05, pe.gate_config(0.18, check_plausibility=False))
    assert gated["dropped_below_decode_floor"] == 1     # the 0.02
    assert gated["dropped_by_gate"] >= 1                # the 0.12
    assert gated["rejected_total"] == (gated["dropped_below_decode_floor"]
                                      + gated["dropped_by_gate"])


def test_full_report_carries_map_50_95_and_cheap_one_does_not():
    gts = [_gt("a")]
    cache = _cache([("a", [_cand("mcb", 0.9)])])
    full = pe.production_report(gts, cache, 0.05, 0.18, full=True,
                                check_plausibility=False)["production"]
    cheap = pe.production_report(gts, cache, 0.05, 0.18, full=False,
                                 check_plausibility=False)["production"]
    assert full["map_50_95"] is not None
    assert cheap["map_50_95"] is None


def test_report_names_the_path_it_measured():
    cache = _cache([("a", [_cand("mcb", 0.9)])])
    rep = pe.production_report([_gt("a")], cache, 0.05, 0.18,
                               check_plausibility=False)
    assert "production inference path" in rep["evaluated_via"]
    assert rep["backend_id"] == "test"


# --------------------------------------------------------------------------
# scoring and the sweep
# --------------------------------------------------------------------------

def test_score_point_rejects_a_point_over_the_fp_budget():
    prod = {"precision": 0.9, "recall": 0.9, "f1": 0.9, "map_50": 0.9,
            "fp_per_image": 5.0}
    assert pe.score_point(prod, "f1", max_fp_per_image=1.0) is None
    assert pe.score_point(prod, "f1", max_fp_per_image=10.0) == 0.9


def test_score_point_rejects_below_precision_and_recall_floors():
    prod = {"precision": 0.4, "recall": 0.8, "f1": 0.5, "map_50": 0.5,
            "fp_per_image": 0.1}
    assert pe.score_point(prod, "f1", min_precision=0.6) is None
    assert pe.score_point(prod, "f1", min_recall=0.9) is None
    assert pe.score_point(prod, "f1") == 0.5


def test_production_score_penalises_false_positives():
    clean = {"precision": 0.9, "recall": 0.9, "f1": 0.9, "map_50": 0.9,
             "fp_per_image": 0.0}
    noisy = {**clean, "fp_per_image": 4.0}
    assert (pe.score_point(clean, "production_score")
            > pe.score_point(noisy, "production_score"))


def test_sweep_covers_the_grid_and_picks_a_winner():
    gts = [_gt("a"), _gt("b")]
    cache = _cache([("a", [_cand("mcb", 0.9)]), ("b", [_cand("mcb", 0.9)])])
    res = pe.sweep(gts, cache, decode_floors=(0.01, 0.05, 0.10),
                   unknown_floors=(0.10, 0.20), check_plausibility=False)
    assert res["status"] == "swept"
    assert res["points_evaluated"] == 6
    assert len(res["grid"]) == 6
    # the winner is recomputed in full, so its mAP@0.5:0.95 is real
    assert res["best"]["map_50_95"] is not None
    assert res["best"]["decode_floor"] in (0.01, 0.05, 0.10)


def test_sweep_reports_when_no_point_satisfies_the_constraints():
    gts = [_gt("a")]
    cache = _cache([("a", [_cand("mcb", 0.9)])])
    res = pe.sweep(gts, cache, decode_floors=(0.05,), unknown_floors=(0.18,),
                   min_precision=1.5, check_plausibility=False)
    assert res["status"] == "no_eligible_operating_point"
    assert res["points_eligible"] == 0
    assert "relax" in res["reason"]


def test_sweep_refuses_floors_below_the_cache():
    cache = _cache([("a", [_cand("mcb", 0.9)])], base=0.10)
    with pytest.raises(ValueError, match="below the cached"):
        pe.sweep([_gt("a")], cache, decode_floors=(0.01, 0.10),
                 unknown_floors=(0.18,))


def test_sweep_rejects_an_unknown_objective():
    cache = _cache([("a", [_cand("mcb", 0.9)])])
    with pytest.raises(ValueError, match="objective must be one of"):
        pe.sweep([_gt("a")], cache, objective="accuracy")


def test_default_grid_matches_the_documented_sweep():
    assert pe.DECODE_FLOORS == (0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20)
    assert pe.UNKNOWN_FLOORS == (0.10, 0.15, 0.18, 0.20, 0.25)


# --------------------------------------------------------------------------
# per-class threshold refinement
# --------------------------------------------------------------------------

def test_per_class_refinement_only_adopts_when_it_helps():
    gts = [_gt("a"), _gt("b")]
    cache = _cache([("a", [_cand("mcb", 0.9)]), ("b", [_cand("mcb", 0.9)])])
    res = pe.refine_per_class(gts, cache, 0.05, 0.18)
    assert res["status"] in ("refined", "no_recommendation")
    assert isinstance(res["adopted"], bool)
    # already perfect, so there is nothing to gain and defaults must stand
    if res["status"] == "refined":
        assert res["tuned_score"] <= res["baseline_score"] or res["adopted"]
        if not res["adopted"]:
            assert res["thresholds"] == {}


# --------------------------------------------------------------------------
# presentation
# --------------------------------------------------------------------------

def test_format_sweep_renders_a_table_with_the_chosen_point():
    gts = [_gt("a")]
    cache = _cache([("a", [_cand("mcb", 0.9)])])
    res = pe.sweep(gts, cache, decode_floors=(0.05,), unknown_floors=(0.18,),
                   check_plausibility=False)
    out = pe.format_sweep(res)
    assert "decode" in out and "unknown" in out and "FP/img" in out
    assert "chosen:" in out


def test_format_sweep_handles_an_empty_grid():
    assert "no operating points" in pe.format_sweep({"grid": []})


def test_format_production_lists_every_required_metric():
    gts = [_gt("a")]
    cache = _cache([("a", [_cand("mcb", 0.9)])])
    prod = pe.production_report(gts, cache, 0.05, 0.18,
                                check_plausibility=False)["production"]
    out = pe.format_production(prod)
    for label in ("Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95",
                  "FP per image", "FN per image", "Unknown rate",
                  "Accepted detections", "Rejected detections"):
        assert label in out
