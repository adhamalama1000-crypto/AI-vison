"""
Tests for synthetic -> real domain transfer.

The load-bearing invariant here is that validation is REAL ONLY. Everything else in
this module is a convenience; that one property is what makes the reported numbers mean
anything, so it is tested from several directions.
"""

import json
import os

import pytest

from training.electrical import transfer as xf


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def _write_pair(root, split, stem, cls=0, ext=".jpg"):
    """Create an image/label pair. The image content is irrelevant to these tests."""
    img_d = os.path.join(root, "images", split)
    lbl_d = os.path.join(root, "labels", split)
    os.makedirs(img_d, exist_ok=True)
    os.makedirs(lbl_d, exist_ok=True)
    with open(os.path.join(img_d, stem + ext), "wb") as fh:
        fh.write(b"\xff\xd8\xff\xe0not-a-real-jpeg")
    with open(os.path.join(lbl_d, stem + ".txt"), "w", encoding="utf-8") as fh:
        fh.write(f"{cls} 0.5 0.5 0.2 0.2\n")


def _dataset(root, counts, prefix="img"):
    """counts: {"train": n, "val": n, "test": n}"""
    for split, n in counts.items():
        for i in range(n):
            _write_pair(root, split, f"{prefix}_{split}_{i:03d}")
    with open(os.path.join(root, "dataset.yaml"), "w", encoding="utf-8") as fh:
        fh.write(f"path: {os.path.abspath(root)}\n"
                 "train: images/train\nval: images/val\ntest: images/test\n"
                 "nc: 2\nnames:\n  0: mcb\n  1: contactor\n")
    return root


@pytest.fixture
def real(tmp_path):
    return _dataset(str(tmp_path / "real"),
                    {"train": 40, "val": 10, "test": 5}, prefix="real")


@pytest.fixture
def synth(tmp_path):
    return _dataset(str(tmp_path / "synth"),
                    {"train": 200, "val": 40}, prefix="syn")


def _count(root, split):
    d = os.path.join(root, "images", split)
    return len(os.listdir(d)) if os.path.isdir(d) else 0


def _names(root, split):
    d = os.path.join(root, "images", split)
    return sorted(os.listdir(d)) if os.path.isdir(d) else []


# --------------------------------------------------------------------------
# build_mixed: the real-only validation invariant
# --------------------------------------------------------------------------

class TestBuildMixedValidationIsRealOnly:
    def test_val_and_test_contain_no_synthetic_images(self, real, synth, tmp_path):
        dst = str(tmp_path / "mixed")
        xf.build_mixed(real, synth, dst, synth_fraction=0.3)

        # Every val/test image carries the real_ prefix. Nothing synthetic leaks in,
        # which is the entire reason this function exists rather than a plain merge.
        for split in ("val", "test"):
            assert _names(dst, split), f"{split} is empty"
            assert all(n.startswith("real_") for n in _names(dst, split)), \
                f"synthetic image leaked into {split}"

    def test_report_states_the_policy_and_the_zero_counts(self, real, synth, tmp_path):
        rep = xf.build_mixed(real, synth, str(tmp_path / "mixed"),
                             synth_fraction=0.3)
        assert rep["val"]["synthetic"] == 0
        assert rep["test"]["synthetic"] == 0
        assert rep["val"]["real"] == 10
        assert "REAL ONLY" in rep["validation_policy"]

    def test_refuses_a_real_dataset_with_no_val_split(self, synth, tmp_path):
        bare = _dataset(str(tmp_path / "novl"), {"train": 20}, prefix="real")
        with pytest.raises(ValueError, match="no val split"):
            xf.build_mixed(bare, synth, str(tmp_path / "out"))

    def test_refuses_a_real_dataset_with_no_train_split(self, synth, tmp_path):
        bare = _dataset(str(tmp_path / "notr"), {"val": 10}, prefix="real")
        with pytest.raises(ValueError, match="no train split"):
            xf.build_mixed(bare, synth, str(tmp_path / "out"))


class TestBuildMixedSynthFraction:
    def test_fraction_is_honoured(self, real, synth, tmp_path):
        rep = xf.build_mixed(real, synth, str(tmp_path / "m"), synth_fraction=0.3)
        # 40 real, so n_synth = 0.3/0.7 * 40 = 17.14 -> 17; 17/57 = 0.298
        assert rep["train"]["real"] == 40
        assert rep["train"]["synthetic"] == 17
        assert rep["train"]["synthetic_fraction"] == pytest.approx(0.3, abs=0.01)
        assert _count(str(tmp_path / "m"), "train") == 57

    def test_zero_fraction_admits_no_synthetic_images(self, real, synth, tmp_path):
        dst = str(tmp_path / "ro")
        rep = xf.build_mixed(real, synth, dst, synth_fraction=0.0)
        assert rep["train"]["synthetic"] == 0
        assert _count(dst, "train") == 40
        assert all(n.startswith("real_") for n in _names(dst, "train"))

    def test_half_fraction_matches_real_count(self, real, synth, tmp_path):
        rep = xf.build_mixed(real, synth, str(tmp_path / "h"), synth_fraction=0.5)
        assert rep["train"]["synthetic"] == 40      # 0.5/0.5 * 40

    @pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
    def test_out_of_range_fraction_is_refused(self, real, synth, tmp_path, bad):
        with pytest.raises(ValueError, match="synth_fraction"):
            xf.build_mixed(real, synth, str(tmp_path / "b"), synth_fraction=bad)

    def test_warns_when_too_few_synthetic_images_exist(self, real, tmp_path):
        thin = _dataset(str(tmp_path / "thin"), {"train": 3}, prefix="syn")
        rep = xf.build_mixed(real, thin, str(tmp_path / "m"), synth_fraction=0.5)
        assert rep["train"]["synthetic"] == 3
        assert rep["synth_requested"] == 40
        assert any("but only 3 exist" in w for w in rep["warnings"])

    def test_seed_makes_the_selection_reproducible(self, real, synth, tmp_path):
        a, b = str(tmp_path / "a"), str(tmp_path / "b")
        xf.build_mixed(real, synth, a, synth_fraction=0.3, seed=7)
        xf.build_mixed(real, synth, b, synth_fraction=0.3, seed=7)
        assert _names(a, "train") == _names(b, "train")

    def test_different_seeds_select_differently(self, real, synth, tmp_path):
        a, b = str(tmp_path / "a"), str(tmp_path / "b")
        xf.build_mixed(real, synth, a, synth_fraction=0.3, seed=1)
        xf.build_mixed(real, synth, b, synth_fraction=0.3, seed=2)
        assert _names(a, "train") != _names(b, "train")


class TestBuildMixedFileHandling:
    def test_every_image_gets_a_label_file(self, real, synth, tmp_path):
        dst = str(tmp_path / "m")
        xf.build_mixed(real, synth, dst, synth_fraction=0.3)
        for split in ("train", "val", "test"):
            for name in _names(dst, split):
                stem = os.path.splitext(name)[0]
                lbl = os.path.join(dst, "labels", split, stem + ".txt")
                assert os.path.exists(lbl), f"missing label for {split}/{name}"

    def test_an_unlabelled_image_becomes_an_empty_label_not_a_crash(self, real, synth,
                                                                   tmp_path):
        # A negative (background) image is legitimate training data and must survive
        # with an empty label file rather than being dropped or raising.
        img_d = os.path.join(real, "images", "train")
        with open(os.path.join(img_d, "negative.jpg"), "wb") as fh:
            fh.write(b"\xff\xd8\xff")
        dst = str(tmp_path / "m")
        xf.build_mixed(real, synth, dst, synth_fraction=0.0)
        lbl = os.path.join(dst, "labels", "train", "real_negative.txt")
        assert os.path.exists(lbl)
        assert open(lbl).read() == ""

    def test_a_name_in_both_synth_splits_produces_two_files(self, real, tmp_path):
        # A synthetic set may hold the same filename in train and val. Both copies must
        # land, each paired with its own label — an earlier version resolved the origin
        # split by membership test and mapped the val copy to the train label.
        s = str(tmp_path / "dup")
        _write_pair(s, "train", "same", cls=0)
        _write_pair(s, "val", "same", cls=1)
        with open(os.path.join(s, "dataset.yaml"), "w") as fh:
            fh.write(f"path: {s}\nnc: 2\nnames:\n  0: mcb\n  1: contactor\n")

        dst = str(tmp_path / "m")
        xf.build_mixed(real, s, dst, synth_fraction=0.5)
        train = _names(dst, "train")
        synth_names = [n for n in train if n.startswith("synth_")]
        assert len(synth_names) == 2, synth_names
        assert len(set(synth_names)) == 2, "one synthetic copy overwrote the other"
        # And each kept its own class index.
        classes = set()
        for n in synth_names:
            stem = os.path.splitext(n)[0]
            body = open(os.path.join(dst, "labels", "train", stem + ".txt")).read()
            classes.add(body.split()[0])
        assert classes == {"0", "1"}, f"labels were mispaired: {classes}"

    def test_symlinks_by_default_and_copies_on_request(self, real, synth, tmp_path):
        a = str(tmp_path / "link")
        xf.build_mixed(real, synth, a, synth_fraction=0.0, symlink=True)
        one = os.path.join(a, "images", "train", _names(a, "train")[0])
        assert os.path.islink(one)

        b = str(tmp_path / "copy")
        xf.build_mixed(real, synth, b, synth_fraction=0.0, symlink=False)
        one = os.path.join(b, "images", "train", _names(b, "train")[0])
        assert not os.path.islink(one)

    def test_rebuilding_over_an_existing_root_does_not_fail_on_symlinks(
            self, real, synth, tmp_path):
        dst = str(tmp_path / "m")
        xf.build_mixed(real, synth, dst, synth_fraction=0.0)
        xf.build_mixed(real, synth, dst, synth_fraction=0.0)   # must not raise
        assert _count(dst, "train") == 40

    def test_dataset_yaml_points_at_the_new_root(self, real, synth, tmp_path):
        dst = str(tmp_path / "m")
        rep = xf.build_mixed(real, synth, dst, synth_fraction=0.3)
        assert os.path.exists(rep["dataset_yaml"])
        body = open(rep["dataset_yaml"]).read()
        assert os.path.abspath(dst) in body
        # The stale path from the source dataset must be gone, or training reads the
        # wrong images while reporting the new root.
        assert os.path.abspath(real) not in body

    def test_explicit_classes_write_a_profile_yaml(self, real, synth, tmp_path):
        dst = str(tmp_path / "m")
        rep = xf.build_mixed(real, synth, dst, synth_fraction=0.0,
                            classes=("mcb", "contactor", "relay"))
        body = open(rep["dataset_yaml"]).read()
        assert "relay" in body


class TestBuildMixedWarnings:
    def test_warns_on_a_thin_validation_split(self, synth, tmp_path):
        r = _dataset(str(tmp_path / "r"), {"train": 200, "val": 8}, prefix="real")
        rep = xf.build_mixed(r, synth, str(tmp_path / "m"), synth_fraction=0.0)
        assert any("8 real validation image" in w for w in rep["warnings"])

    def test_warns_on_a_thin_training_split(self, synth, tmp_path):
        r = _dataset(str(tmp_path / "r"), {"train": 30, "val": 60}, prefix="real")
        rep = xf.build_mixed(r, synth, str(tmp_path / "m"), synth_fraction=0.0)
        assert any("30 real training image" in w for w in rep["warnings"])

    def test_no_warnings_on_a_healthy_split(self, synth, tmp_path):
        r = _dataset(str(tmp_path / "r"), {"train": 400, "val": 60}, prefix="real")
        rep = xf.build_mixed(r, synth, str(tmp_path / "m"), synth_fraction=0.3)
        assert rep["warnings"] == []


# --------------------------------------------------------------------------
# domain gap measurement
# --------------------------------------------------------------------------

class TestMeasureDomainGap:
    def _stub(self, monkeypatch, synth_map, real_map):
        calls = []

        def fake_eval(backend, root, split, params=None, limit=None):
            calls.append(root)
            # Match on the basename only. A substring test against the whole path
            # matches pytest's tmp_path, which embeds the test's own name — and this
            # test is called ..._synthetic_minus_real, so "synth" was in both roots
            # and every gap measured as zero.
            m = synth_map if os.path.basename(root) == "synth" else real_map
            if m is None:
                return {"status": "skipped", "reason": "no ground truth"}
            return {"status": "evaluated", "map_50": m[0], "map_50_95": m[1],
                    "overall": {"precision": 0.7, "recall": 0.6},
                    "classes": {}, "confusion_matrix": [[1]],
                    "false_positives": 3, "false_negatives": 4}

        monkeypatch.setattr(xf.tr, "evaluate_backend", fake_eval)
        return calls

    def test_gap_is_synthetic_minus_real(self, monkeypatch, tmp_path):
        self._stub(monkeypatch, (0.85, 0.60), (0.05, 0.02))
        out = xf.measure_domain_gap("w.pt", str(tmp_path / "synth"),
                                    str(tmp_path / "real"))
        assert out["gap_map_50"] == pytest.approx(0.80)
        assert out["synthetic"]["map_50"] == 0.85
        assert out["real"]["map_50"] == 0.05

    def test_a_large_gap_says_the_model_must_not_ship(self, monkeypatch, tmp_path):
        self._stub(monkeypatch, (0.85, 0.60), (0.05, 0.02))
        out = xf.measure_domain_gap("w.pt", str(tmp_path / "synth"),
                                    str(tmp_path / "real"))
        assert "must not be shipped" in out["interpretation"]

    def test_a_small_gap_is_reported_as_suspicious_not_as_success(
            self, monkeypatch, tmp_path):
        self._stub(monkeypatch, (0.60, 0.40), (0.58, 0.38))
        out = xf.measure_domain_gap("w.pt", str(tmp_path / "synth"),
                                    str(tmp_path / "real"))
        assert "too small to be discriminating" in out["interpretation"]

    def test_a_middling_gap_points_at_the_comparison(self, monkeypatch, tmp_path):
        self._stub(monkeypatch, (0.80, 0.50), (0.50, 0.30))
        out = xf.measure_domain_gap("w.pt", str(tmp_path / "synth"),
                                    str(tmp_path / "real"))
        assert "Substantial" in out["interpretation"]

    def test_no_real_evaluation_means_no_claim(self, monkeypatch, tmp_path):
        self._stub(monkeypatch, (0.85, 0.60), None)
        out = xf.measure_domain_gap("w.pt", str(tmp_path / "synth"),
                                    str(tmp_path / "real"))
        assert out["gap_map_50"] is None
        assert out["real"]["map_50"] is None
        assert "no claim about domain transfer can be made" in out["interpretation"]

    def test_it_carries_the_real_confusion_matrix_and_error_counts(
            self, monkeypatch, tmp_path):
        self._stub(monkeypatch, (0.85, 0.60), (0.05, 0.02))
        out = xf.measure_domain_gap("w.pt", str(tmp_path / "synth"),
                                    str(tmp_path / "real"))
        assert out["real_confusion_matrix"] == [[1]]
        assert out["real_false_positives"] == 3
        assert out["real_false_negatives"] == 4

    def test_it_scores_both_domains(self, monkeypatch, tmp_path):
        calls = self._stub(monkeypatch, (0.85, 0.60), (0.05, 0.02))
        xf.measure_domain_gap("w.pt", str(tmp_path / "synth"), str(tmp_path / "real"))
        assert len(calls) == 2


# --------------------------------------------------------------------------
# fine-tuning
# --------------------------------------------------------------------------

class _FakeResult:
    def __init__(self, status="trained", weights="best.pt", reason=None):
        self.status, self.weights, self.reason = status, weights, reason

    def to_dict(self):
        return {"status": self.status, "weights": self.weights,
                "reason": self.reason}


class TestFineTune:
    def test_init_from_reaches_the_trainer(self, monkeypatch):
        seen = []

        def fake_train(cfg, export_onnx=True, log=None):
            seen.append(cfg)
            return _FakeResult(weights=f"{cfg.name}/best.pt")

        monkeypatch.setattr(xf.tr, "train", fake_train)
        out = xf.fine_tune("d.yaml", "synth_best.pt", epochs=30)
        assert out["status"] == "completed"
        assert seen[0].init_from == "synth_best.pt"

    def test_staged_run_freezes_then_unfreezes_and_halves_the_lr(self, monkeypatch):
        seen = []

        def fake_train(cfg, export_onnx=True, log=None):
            seen.append(cfg)
            return _FakeResult(weights=f"{cfg.name}/best.pt")

        monkeypatch.setattr(xf.tr, "train", fake_train)
        out = xf.fine_tune("d.yaml", "init.pt", epochs=30, lr0=0.002,
                           freeze_layers=10)

        assert len(seen) == 2
        assert seen[0].freeze == 10 and seen[0].epochs == 10
        assert seen[1].freeze == 0 and seen[1].epochs == 20
        assert seen[1].lr0 == pytest.approx(0.001)
        # Stage 2 must continue from stage 1's weights, not restart from init.
        assert seen[1].init_from == "finetune_s1_frozen/best.pt"
        assert out["weights"] == "finetune_s2_full/best.pt"
        assert [s["stage"] for s in out["stages"]] == [1, 2]

    def test_unstaged_run_is_a_single_full_run(self, monkeypatch):
        seen = []
        monkeypatch.setattr(xf.tr, "train",
                            lambda cfg, export_onnx=True, log=None:
                            (seen.append(cfg), _FakeResult())[1])
        out = xf.fine_tune("d.yaml", "init.pt", epochs=30, staged=False)
        assert len(seen) == 1
        assert seen[0].epochs == 30
        assert out["staged"] is False

    def test_stage1_epochs_override(self, monkeypatch):
        seen = []
        monkeypatch.setattr(xf.tr, "train",
                            lambda cfg, export_onnx=True, log=None:
                            (seen.append(cfg), _FakeResult(
                                weights=f"{cfg.name}/best.pt"))[1])
        xf.fine_tune("d.yaml", None, epochs=40, stage1_epochs=5)
        assert seen[0].epochs == 5 and seen[1].epochs == 35

    def test_a_failed_stage1_stops_and_reports(self, monkeypatch):
        monkeypatch.setattr(xf.tr, "train",
                            lambda cfg, export_onnx=True, log=None:
                            _FakeResult("failed", None, "cuda oom"))
        out = xf.fine_tune("d.yaml", "init.pt", epochs=30)
        assert out["status"] == "failed"
        assert "cuda oom" in out["reason"]
        assert len(out["stages"]) == 1        # it did not attempt stage 2

    def test_a_failed_stage2_keeps_the_stage1_weights(self, monkeypatch):
        calls = {"n": 0}

        def fake_train(cfg, export_onnx=True, log=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return _FakeResult(weights="s1/best.pt")
            return _FakeResult("failed", None, "diverged")

        monkeypatch.setattr(xf.tr, "train", fake_train)
        out = xf.fine_tune("d.yaml", "init.pt", epochs=30)
        assert out["status"] == "failed"
        assert out["weights"] == "s1/best.pt"

    def test_the_default_lr_is_well_below_from_scratch(self, monkeypatch):
        seen = []
        monkeypatch.setattr(xf.tr, "train",
                            lambda cfg, export_onnx=True, log=None:
                            (seen.append(cfg), _FakeResult(
                                weights=f"{cfg.name}/best.pt"))[1])
        xf.fine_tune("d.yaml", "init.pt", epochs=9)
        # Fine-tuning at the 0.01 from-scratch rate is the standard way to destroy a
        # good checkpoint, so the default must stay small.
        assert seen[0].lr0 <= 0.005


# --------------------------------------------------------------------------
# ranking
# --------------------------------------------------------------------------

def _res(strategy, map50, map5095=None, recall=0.5, status="completed"):
    r = xf.TransferResult(strategy, status)
    r.map_50 = map50
    r.map_50_95 = map5095 if map5095 is not None else (
        None if map50 is None else map50 * 0.6)
    r.precision = 0.7
    r.recall = recall
    r.weights = f"{strategy}/best.pt"
    return r


class TestRankRequiresARealMargin:
    def test_a_small_win_does_not_count_as_synthetic_helping(self):
        # 0.015 over the control is inside the noise band for a small real val set.
        out = xf._rank({"real_only": _res("real_only", 0.500),
                        "mixed": _res("mixed", 0.515)}, {}, None, 0.3)
        assert out["winner"] == "mixed"
        assert out["synthetic_data_helped"] is False
        assert "did NOT help" in out["rationale"]

    def test_a_clear_win_counts(self):
        out = xf._rank({"real_only": _res("real_only", 0.500),
                        "mixed": _res("mixed", 0.560)}, {}, None, 0.3)
        assert out["winner"] == "mixed"
        assert out["synthetic_data_helped"] is True
        assert "helped" in out["rationale"]

    def test_exactly_at_the_threshold_does_not_count(self):
        out = xf._rank({"real_only": _res("real_only", 0.500),
                        "mixed": _res("mixed", 0.520)}, {}, None, 0.3)
        assert out["synthetic_data_helped"] is False

    def test_the_control_winning_is_not_synthetic_helping(self):
        out = xf._rank({"real_only": _res("real_only", 0.600),
                        "mixed": _res("mixed", 0.400)}, {}, None, 0.3)
        assert out["winner"] == "real_only"
        assert out["synthetic_data_helped"] is False

    def test_ranking_is_ordered_by_map50(self):
        out = xf._rank({"a": _res("real_only", 0.30), "b": _res("mixed", 0.70),
                        "c": _res("coco_to_synth_to_real", 0.50)}, {}, None, 0.3)
        assert [r["strategy"] for r in out["ranking"]] == [
            "mixed", "coco_to_synth_to_real", "real_only"]

    def test_map_50_95_breaks_a_tie(self):
        out = xf._rank({"a": _res("real_only", 0.50, 0.20),
                        "b": _res("mixed", 0.50, 0.35)}, {}, None, 0.3)
        assert out["ranking"][0]["strategy"] == "mixed"

    def test_unscored_strategies_are_excluded_from_the_ranking(self):
        out = xf._rank({"real_only": _res("real_only", 0.50),
                        "mixed": _res("mixed", None, status="failed")},
                       {}, None, 0.3)
        assert [r["strategy"] for r in out["ranking"]] == ["real_only"]
        assert out["strategies"]["mixed"]["status"] == "failed"

    def test_no_scored_strategy_is_a_failure_with_an_explanation(self):
        out = xf._rank({"mixed": _res("mixed", None, status="failed")},
                       {}, None, 0.3)
        assert out["status"] == "failed"
        assert out["winner"] is None
        assert "No strategy produced a scored model" in out["rationale"]

    def test_a_missing_recall_does_not_crash_the_rationale(self):
        out = xf._rank({"real_only": _res("real_only", 0.50, recall=None)},
                       {}, None, 0.3)
        assert "unavailable" in out["rationale"]

    def test_the_rationale_always_states_the_validation_policy(self):
        out = xf._rank({"real_only": _res("real_only", 0.50)}, {}, None, 0.3)
        assert "REAL-ONLY validation" in out["rationale"]


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

class TestFormatRanking:
    def test_it_tabulates_the_ranking(self):
        out = xf._rank({"real_only": _res("real_only", 0.50),
                        "mixed": _res("mixed", 0.70)}, {}, None, 0.3)
        table = xf.format_ranking(out)
        assert "mixed" in table and "real_only" in table
        assert "0.7000" in table
        # The winner is on the first data row.
        assert table.splitlines()[2].startswith("1")

    def test_empty_ranking_says_so_rather_than_crashing(self):
        assert xf.format_ranking({"ranking": []}) == "no scored strategies"
        assert xf.format_ranking({}) == "no scored strategies"

    def test_missing_metrics_render_as_dashes(self):
        table = xf.format_ranking({"ranking": [
            {"strategy": "mixed", "map_50": 0.5, "map_50_95": None,
             "precision": None, "recall": None}]})
        assert "-" in table


class TestWriteResult:
    def test_it_writes_parseable_json_and_creates_the_directory(self, tmp_path):
        p = str(tmp_path / "nested" / "deeper" / "out.json")
        xf.write_result({"status": "completed", "winner": "mixed"}, p)
        assert json.load(open(p))["winner"] == "mixed"


class TestCompareStrategiesFailsCleanly:
    def test_a_bad_real_dataset_is_reported_not_raised(self, synth, tmp_path):
        bare = _dataset(str(tmp_path / "novl"), {"train": 10}, prefix="real")
        out = xf.compare_strategies(bare, synth, str(tmp_path / "w"),
                                    strategies=("real_only",))
        assert out["status"] == "failed"
        assert "no val split" in out["reason"]

    def test_a_bad_synth_fraction_is_reported_not_raised(self, real, synth, tmp_path):
        # The real_only build always uses 0.0, so it survives an out-of-range fraction;
        # only the mixed build sees the bad value.
        out = xf.compare_strategies(real, synth, str(tmp_path / "w"),
                                    strategies=("mixed",), synth_fraction=1.5)
        assert out["status"] == "failed"
        assert "synth_fraction" in out["reason"]

    def test_an_unknown_strategy_is_skipped_with_the_known_names(
            self, real, synth, tmp_path, monkeypatch):
        monkeypatch.setattr(xf.tr, "train",
                            lambda cfg, export_onnx=True, log=None:
                            _FakeResult(weights=f"{cfg.name}/best.pt"))
        monkeypatch.setattr(xf.tr, "evaluate_backend",
                            lambda *a, **k: {"status": "skipped",
                                             "reason": "stubbed"})
        out = xf.compare_strategies(real, synth, str(tmp_path / "w"),
                                    strategies=("wishful_thinking",))
        s = out["strategies"]["wishful_thinking"]
        assert s["status"] == "skipped"
        assert "real_only" in s["reason"]


class TestStrategyCatalogue:
    def test_the_control_is_present_and_documented(self):
        assert "real_only" in xf.STRATEGIES
        assert "control" in xf.STRATEGIES["real_only"]

    def test_the_two_stage_plan_is_documented_as_possibly_worse(self):
        # The docs must not oversell sim2real pretraining; that is how a pipeline
        # acquires a stage that costs compute and buys nothing.
        assert "WORSE" in xf.STRATEGIES["coco_to_synth_to_real"]

    def test_the_synthetic_fraction_default_is_a_minority(self):
        assert 0 < xf.DEFAULT_SYNTH_FRACTION <= 0.5
