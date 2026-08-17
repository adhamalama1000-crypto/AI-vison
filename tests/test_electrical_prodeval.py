"""
Tests for production-path evaluation and the confidence sweep.

Two things here are easy to get wrong and expensive to get wrong:

1. Sweeping only ``decode_floor`` measures nothing, because ``unknown_floor`` (0.18) and
   the per-class thresholds sit above it and do the actual cutting. A sweep like that
   returns identical rows for 0.01/0.03/0.05/0.10 and looks like a plateau in the model
   rather than a misconfigured harness.
2. Selecting an operating point that violates a stated constraint, silently. If a
   caller asks for precision >= 0.9 and nothing reaches it, the answer is "nothing
   reached it", not the best row with the constraint quietly dropped.
"""

import json

import pytest

from rtsp_backend.electrical import taxonomy as tax
from training.electrical import prodeval as pe


# --------------------------------------------------------------------------
# production_params: all three gates move together
# --------------------------------------------------------------------------

class TestProductionParams:
    def test_it_moves_all_three_gates(self):
        p = pe.production_params(0.05, "w.pt")
        assert p["decode_floor"] == 0.05
        assert p["unknown_floor"] == 0.05
        assert set(p["thresholds"].values()) == {0.05}

    def test_every_taxonomy_class_gets_a_threshold(self):
        # A class left out would silently fall back to its taxonomy min_conf, which
        # for some devices is well above the swept value -- so that class would not
        # participate in the sweep at all.
        p = pe.production_params(0.02, "w.pt")
        assert set(p["thresholds"]) == set(tax.CLASS_ORDER)

    def test_strictness_is_pinned(self):
        # strictness multiplies every per-class threshold; left free it would rescale
        # the uniform value being set here.
        assert pe.production_params(0.05, "w.pt")["strictness"] == 1.0

    def test_weights_imgsz_and_device_are_passed_through(self):
        p = pe.production_params(0.1, "best.pt", imgsz=512, device="cuda")
        assert p["weights"] == "best.pt"
        assert p["imgsz"] == 512
        assert p["device"] == "cuda"

    def test_extra_params_override(self):
        p = pe.production_params(0.1, "w.pt", extra={"nms_iou": 0.7})
        assert p["nms_iou"] == 0.7

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 1.5])
    def test_out_of_range_conf_is_refused(self, bad):
        with pytest.raises(ValueError, match="conf must be in"):
            pe.production_params(bad, "w.pt")

    def test_the_swept_points_are_the_documented_ones(self):
        assert pe.OPERATING_POINTS == (0.01, 0.03, 0.05, 0.10, 0.20)


# --------------------------------------------------------------------------
# evaluate_at / sweep
# --------------------------------------------------------------------------

def _fake_env(monkeypatch, tmp_path, by_conf, n_images=10, n_gt=20):
    """Stub the inference path, recording the params each call received."""
    seen = []

    root = tmp_path / "ds"
    (root / "images" / "val").mkdir(parents=True)
    (root / "labels" / "val").mkdir(parents=True)
    for i in range(n_images):
        (root / "images" / "val" / f"i{i}.jpg").write_bytes(b"x")

    gts = [{"image_id": f"i{i % n_images}", "class_id": "mcb",
            "box": (0, 0, 10, 10)} for i in range(n_gt)]
    monkeypatch.setattr(pe.tr, "load_ground_truth", lambda r, s: list(gts))

    class FakeBackend:
        def __init__(self, **params):
            seen.append(params)
            self.params = params

        def load(self):
            pass

    def fake_get(task, backend_id):
        return FakeBackend

    import rtsp_backend.ai.registry as registry
    monkeypatch.setattr(registry, "get", fake_get)

    def fake_collect(inst, image_dir, limit=None):
        conf = inst.params["decode_floor"]
        return list(by_conf.get(conf, []))

    monkeypatch.setattr(pe.tr, "collect_predictions", fake_collect)
    return str(root), seen, gts


class TestEvaluateAt:
    def test_it_reports_per_image_error_rates(self, monkeypatch, tmp_path):
        # 10 predictions, all correct, against 20 gt on 10 images -> 10 fn.
        preds = [{"image_id": f"i{i}", "class_id": "mcb",
                  "box": (0, 0, 10, 10), "score": 0.9} for i in range(10)]
        root, _, _ = _fake_env(monkeypatch, tmp_path, {0.05: preds})
        row = pe.evaluate_at(0.05, "w.pt", root, "val")

        assert row["status"] == "evaluated"
        assert row["images"] == 10
        assert row["tp"] == 10 and row["fn"] == 10
        assert row["fn_per_image"] == 1.0
        assert row["fp_per_image"] == 0.0

    def test_it_counts_unknown_demotions_separately(self, monkeypatch, tmp_path):
        preds = [{"image_id": "i0", "class_id": tax.UNKNOWN_COMPONENT_ID,
                  "box": (0, 0, 10, 10), "score": 0.2} for _ in range(4)]
        root, _, _ = _fake_env(monkeypatch, tmp_path, {0.01: preds})
        row = pe.evaluate_at(0.01, "w.pt", root, "val")

        assert row["unknown_predictions"] == 4
        assert row["unknown_per_image"] == 0.4
        # Unknowns match no gt class, so they land in fp -- and the row says so
        # rather than leaving the reader to infer it.
        assert row["fp"] >= 4
        assert "honest low-confidence output" in row["note_on_unknowns"]

    def test_the_threshold_policy_is_stated_on_every_row(self, monkeypatch, tmp_path):
        root, _, _ = _fake_env(monkeypatch, tmp_path, {0.05: []})
        row = pe.evaluate_at(0.05, "w.pt", root, "val")
        assert "decode_floor, unknown_floor and every per-class threshold" in \
            row["threshold_policy"]

    def test_it_passes_the_swept_conf_into_the_backend(self, monkeypatch, tmp_path):
        root, seen, _ = _fake_env(monkeypatch, tmp_path, {0.03: []})
        pe.evaluate_at(0.03, "w.pt", root, "val")
        assert seen[0]["decode_floor"] == 0.03
        assert seen[0]["unknown_floor"] == 0.03

    def test_no_ground_truth_is_skipped_with_a_reason(self, monkeypatch, tmp_path):
        root, _, _ = _fake_env(monkeypatch, tmp_path, {})
        monkeypatch.setattr(pe.tr, "load_ground_truth", lambda r, s: [])
        row = pe.evaluate_at(0.05, "w.pt", root, "val")
        assert row["status"] == "skipped"
        assert "no ground truth" in row["reason"]

    def test_limit_restricts_the_ground_truth_to_the_images_inferred(
            self, monkeypatch, tmp_path):
        """The bug this pins: --limit scored N images' predictions against all labels.

        Every instance in the unvisited images became a false negative, so recall and
        FN/image were pinned to a wrong value that did not move as the threshold swept.
        It read as a model plateau and was really the harness comparing two sets.
        """
        # 10 images, one gt each; predict correctly on the 3 we will evaluate.
        preds = [{"image_id": f"i{i}", "class_id": "mcb",
                  "box": (0, 0, 10, 10), "score": 0.9} for i in range(3)]
        root, _, _ = _fake_env(monkeypatch, tmp_path, {0.05: preds},
                               n_images=10, n_gt=10)

        row = pe.evaluate_at(0.05, "w.pt", root, "val", limit=3)
        assert row["images"] == 3
        assert row["ground_truth_instances"] == 3, \
            "ground truth was not restricted to the evaluated images"
        # All 3 matched, so recall is perfect and there are no false negatives.
        assert row["recall"] == 1.0
        assert row["fn"] == 0
        assert row["fn_per_image"] == 0.0

    def test_without_a_limit_all_ground_truth_is_used(self, monkeypatch, tmp_path):
        preds = [{"image_id": "i0", "class_id": "mcb",
                  "box": (0, 0, 10, 10), "score": 0.9}]
        root, _, _ = _fake_env(monkeypatch, tmp_path, {0.05: preds},
                               n_images=10, n_gt=10)
        row = pe.evaluate_at(0.05, "w.pt", root, "val")
        assert row["images"] == 10
        assert row["ground_truth_instances"] == 10
        assert row["fn"] == 9

    def test_a_limit_covering_only_unlabelled_images_is_skipped(self, monkeypatch,
                                                               tmp_path):
        root, _, _ = _fake_env(monkeypatch, tmp_path, {0.05: []},
                               n_images=10, n_gt=10)
        # Ground truth only for i5..i9, but we evaluate i0..i2.
        monkeypatch.setattr(
            pe.tr, "load_ground_truth",
            lambda r, s: [{"image_id": f"i{i}", "class_id": "mcb",
                           "box": (0, 0, 10, 10)} for i in range(5, 10)])
        row = pe.evaluate_at(0.05, "w.pt", root, "val", limit=3)
        assert row["status"] == "skipped"
        assert "raise --limit" in row["reason"]

    def test_an_unloadable_backend_is_skipped_not_raised(self, monkeypatch, tmp_path):
        root, _, _ = _fake_env(monkeypatch, tmp_path, {0.05: []})
        import rtsp_backend.ai.registry as registry

        def boom(task, bid):
            raise KeyError("no such backend")

        monkeypatch.setattr(registry, "get", boom)
        row = pe.evaluate_at(0.05, "w.pt", root, "val")
        assert row["status"] == "skipped"
        assert "unavailable" in row["reason"]


class TestSweep:
    def test_it_evaluates_every_point_and_adds_the_production_default(
            self, monkeypatch, tmp_path):
        root, seen, _ = _fake_env(monkeypatch, tmp_path, {})
        rows = pe.sweep("w.pt", root, "val")
        confs = sorted(r["conf"] for r in rows)
        assert confs == sorted(set(pe.OPERATING_POINTS)
                               | {pe.PRODUCTION_DEFAULT_CONF})
        assert sum(1 for r in rows if r["is_production_default"]) == 1

    def test_the_default_can_be_excluded(self, monkeypatch, tmp_path):
        root, _, _ = _fake_env(monkeypatch, tmp_path, {})
        rows = pe.sweep("w.pt", root, "val", include_production_default=False)
        assert sorted(r["conf"] for r in rows) == sorted(pe.OPERATING_POINTS)

    def test_each_point_really_gets_a_different_threshold(self, monkeypatch,
                                                          tmp_path):
        # The regression this guards: sweeping only decode_floor left unknown_floor at
        # 0.18, so every row below that returned identical numbers.
        root, seen, _ = _fake_env(monkeypatch, tmp_path, {})
        pe.sweep("w.pt", root, "val", confs=(0.01, 0.05),
                 include_production_default=False)
        floors = [(s["decode_floor"], s["unknown_floor"],
                   sorted(set(s["thresholds"].values()))) for s in seen]
        assert floors == [(0.01, 0.01, [0.01]), (0.05, 0.05, [0.05])]

    def test_lower_conf_yields_more_predictions_in_the_report(self, monkeypatch,
                                                             tmp_path):
        many = [{"image_id": f"i{i}", "class_id": "mcb", "box": (0, 0, 10, 10),
                 "score": 0.5} for i in range(10)]
        few = many[:2]
        root, _, _ = _fake_env(monkeypatch, tmp_path, {0.01: many, 0.20: few})
        rows = {r["conf"]: r for r in
                pe.sweep("w.pt", root, "val", confs=(0.01, 0.20),
                         include_production_default=False)}
        assert rows[0.01]["recall"] > rows[0.20]["recall"]


# --------------------------------------------------------------------------
# operating point selection
# --------------------------------------------------------------------------

def _row(conf, precision=0.5, recall=0.5, map50=0.5, f1=None, fpi=1.0, fni=1.0):
    if f1 is None:
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
    return {"conf": conf, "status": "evaluated", "precision": precision,
            "recall": recall, "map_50": map50, "map_50_95": map50 * 0.6,
            "f1": round(f1, 4), "fp_per_image": fpi, "fn_per_image": fni,
            "tp": 10, "fp": 5, "fn": 5, "images": 10,
            "unknown_predictions": 0}


class TestSelectOperatingPoint:
    def test_it_maximises_the_objective(self):
        rows = [_row(0.01, 0.30, 0.95), _row(0.10, 0.70, 0.70),
                _row(0.20, 0.90, 0.30)]
        best = pe.select_operating_point(rows, objective="f1")
        assert best["conf"] == 0.10

    def test_objective_can_be_precision(self):
        rows = [_row(0.01, 0.30, 0.95), _row(0.20, 0.90, 0.30)]
        assert pe.select_operating_point(
            rows, objective="precision")["conf"] == 0.20

    def test_objective_can_be_recall(self):
        rows = [_row(0.01, 0.30, 0.95), _row(0.20, 0.90, 0.30)]
        assert pe.select_operating_point(
            rows, objective="recall")["conf"] == 0.01

    def test_a_min_precision_constraint_is_honoured(self):
        rows = [_row(0.01, 0.30, 0.99), _row(0.10, 0.70, 0.70),
                _row(0.20, 0.90, 0.40)]
        best = pe.select_operating_point(rows, min_precision=0.85)
        assert best["conf"] == 0.20
        assert best["constraints_met"] is True

    def test_a_max_fp_per_image_constraint_is_honoured(self):
        rows = [_row(0.01, 0.4, 0.9, fpi=8.0), _row(0.20, 0.8, 0.5, fpi=0.5)]
        best = pe.select_operating_point(rows, max_fp_per_image=1.0)
        assert best["conf"] == 0.20

    def test_an_unsatisfiable_constraint_is_reported_not_hidden(self):
        rows = [_row(0.01, 0.30, 0.95), _row(0.20, 0.50, 0.40)]
        best = pe.select_operating_point(rows, min_precision=0.95)
        assert best["status"] == "selected"
        assert best["constraints_met"] is False
        assert "does NOT meet the requirement" in best["warning"]
        assert best["constraints_applied"] == {"min_precision": 0.95}

    def test_nothing_scored_is_a_failure(self):
        best = pe.select_operating_point(
            [{"conf": 0.05, "status": "skipped", "reason": "no weights"}])
        assert best["status"] == "failed"
        assert "no operating point" in best["reason"]

    def test_the_rationale_quotes_the_error_rates(self):
        rows = [_row(0.05, 0.6, 0.8, fpi=2.5, fni=0.75)]
        best = pe.select_operating_point(rows)
        assert "2.5 false positives" in best["rationale"]
        assert "0.75 false negatives" in best["rationale"]

    def test_the_rationale_states_it_used_the_deployed_path(self):
        best = pe.select_operating_point([_row(0.05)])
        assert "deployed inference path" in best["rationale"]

    def test_it_explains_unknown_inflated_false_positives(self):
        r = _row(0.01, 0.3, 0.9)
        r["unknown_predictions"] = 40
        best = pe.select_operating_point([r])
        assert "understates classification accuracy" in best["rationale"]

    def test_how_to_apply_names_the_real_params(self):
        best = pe.select_operating_point([_row(0.05)])
        assert "decode_floor=0.05" in best["how_to_apply"]
        assert "unknown_floor=0.05" in best["how_to_apply"]


# --------------------------------------------------------------------------
# acceptance verdict
# --------------------------------------------------------------------------

class TestVerdict:
    def test_meeting_the_target_passes(self):
        v = pe._verdict([_row(0.05, map50=0.90)],
                        {"conf": 0.05, "map_50": 0.90}, 0.85)
        assert v["passed"] is True
        assert "meets the 0.85 target" in v["statement"]

    def test_missing_the_target_fails_and_says_why_lowering_conf_is_not_the_fix(self):
        v = pe._verdict([_row(0.05, map50=0.40)],
                        {"conf": 0.05, "map_50": 0.40}, 0.85)
        assert v["passed"] is False
        assert "not acceptable for production" in v["statement"]
        assert "trades the shortfall for false" in v["statement"]

    def test_a_missing_metric_does_not_crash_the_verdict(self):
        v = pe._verdict([], {"conf": 0.05, "map_50": None}, 0.85)
        assert v["passed"] is False
        assert "0.0000" in v["statement"]


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

class TestFormatting:
    def test_the_sweep_table_marks_the_production_default(self):
        rows = [_row(0.05), {**_row(0.18), "is_production_default": True}]
        t = pe.format_sweep(rows)
        assert "0.18*" in t
        assert "shipped production default" in t

    def test_skipped_rows_render_without_crashing(self):
        t = pe.format_sweep([{"conf": 0.05, "status": "skipped",
                              "reason": "backend unavailable"}])
        assert "skipped" in t

    def test_the_table_has_the_requested_columns(self):
        t = pe.format_sweep([_row(0.05)])
        for col in ("mAP50", "precision", "recall", "FP/img", "FN/img", "unknown"):
            assert col in t

    def test_per_class_table_sorts_by_support(self):
        row = {"per_class": {
            "mcb": {"ap": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5,
                    "support": 3, "name": "MCB"},
            "relay": {"ap": 0.9, "precision": 0.9, "recall": 0.9, "f1": 0.9,
                      "support": 50, "name": "Relay"}}}
        lines = pe.per_class_table(row).splitlines()
        assert lines[2].startswith("Relay")

    def test_per_class_table_handles_no_data(self):
        assert pe.per_class_table({}) == "no per-class data"


class TestWriters:
    def test_report_json_round_trips(self, tmp_path):
        p = str(tmp_path / "deep" / "acceptance.json")
        pe.write_report({"best_operating_point": {"conf": 0.05}}, p)
        assert json.load(open(p))["best_operating_point"]["conf"] == 0.05

    def test_sweep_csv_has_a_header_and_one_row_per_point(self, tmp_path):
        p = str(tmp_path / "sweep.csv")
        pe.write_sweep_csv([_row(0.05), _row(0.10)], p)
        lines = open(p).read().strip().splitlines()
        assert lines[0].startswith("conf,status,map_50")
        assert len(lines) == 3

    def test_sweep_csv_is_sorted_by_conf(self, tmp_path):
        p = str(tmp_path / "s.csv")
        pe.write_sweep_csv([_row(0.20), _row(0.01)], p)
        rows = open(p).read().strip().splitlines()[1:]
        assert rows[0].startswith("0.01")
