"""
Hyperparameter optimisation for the detector.

A full search cannot run in CI — every trial is a real training run and there is no
GPU, no ultralytics and no dataset — so what is tested here is the part that decides
whether a search is *trustworthy*:

* the sampled space covers what the brief asks for and nothing invalid;
* domain priors are enforced, so the search cannot buy a fraction of a point of mAP
  by teaching the model that mirrored nameplates are normal;
* a trial that cannot train is pruned rather than recorded as a score of 0.0, which
  would teach the sampler to avoid a region of the space for the wrong reason;
* sampled parameters actually reach the trainer, including the ones that are not
  ``TrainConfig`` fields;
* an unavailable dependency is reported, never faked.
"""

from __future__ import annotations

import pytest

from training.electrical import hpo
from training.electrical import train as tr


# ==========================================================================
# a fake Optuna trial, so the space can be exercised without a study
# ==========================================================================

class _FakeTrial:
    """Records every suggestion and returns a deterministic point in the space."""

    def __init__(self, pick: str = "low", number: int = 0):
        assert pick in ("low", "high", "mid")
        self.pick = pick
        self.number = number
        self.suggested: dict = {}
        self.reported: list = []
        self.user_attrs: dict = {}
        self._should_prune = False

    def _value(self, lo, hi):
        return {"low": lo, "high": hi, "mid": (lo + hi) / 2}[self.pick]

    def suggest_float(self, name, low, high, log=False, step=None):
        v = float(self._value(low, high))
        self.suggested[name] = v
        return v

    def suggest_int(self, name, low, high, step=1, log=False):
        v = int(self._value(low, high))
        self.suggested[name] = v
        return v

    def suggest_categorical(self, name, choices):
        v = choices[0] if self.pick == "low" else choices[-1]
        self.suggested[name] = v
        return v

    def report(self, value, step):
        self.reported.append((step, value))

    def should_prune(self):
        return self._should_prune

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value


# ==========================================================================
# availability
# ==========================================================================

def test_optuna_availability_is_reported_not_assumed():
    ok, version = hpo.optuna_available()
    assert isinstance(ok, bool)
    if ok:
        assert version


def test_optimise_skips_without_ultralytics(tmp_path):
    """No trainer means no search — and it must say so, not fake a result."""
    ok, _ = tr.ultralytics_available()
    if ok:
        pytest.skip("ultralytics is installed in this environment")
    res = hpo.optimise(str(tmp_path / "data.yaml"), trials=1)
    assert res.status == "skipped"
    assert "ultralytics" in res.reason
    assert res.best_params == {}
    assert res.best_value is None


def test_optimise_skips_a_missing_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr(tr, "ultralytics_available", lambda: (True, "8.3.0"))
    monkeypatch.setattr(tr, "arch_available", lambda a: True)
    if not hpo.optuna_available()[0]:
        pytest.skip("optuna is not installed")
    res = hpo.optimise(str(tmp_path / "nope.yaml"), trials=1)
    assert res.status == "skipped"
    assert "not found" in res.reason


def test_optimise_skips_an_unavailable_architecture(monkeypatch, tmp_path):
    yaml = tmp_path / "data.yaml"
    yaml.write_text("names: [mcb]\nnc: 1\n")
    monkeypatch.setattr(tr, "ultralytics_available", lambda: (True, "8.3.0"))
    monkeypatch.setattr(tr, "arch_available", lambda a: False)
    if not hpo.optuna_available()[0]:
        pytest.skip("optuna is not installed")
    res = hpo.optimise(str(yaml), arch="yolo12n", trials=1)
    assert res.status == "skipped"
    assert "not available" in res.reason


# ==========================================================================
# the search space
# ==========================================================================

def test_the_space_covers_every_knob_the_brief_names():
    """Learning rate, batch, image size, optimizer, scheduler, augmentation,
    early stopping, weight decay."""
    params = hpo._suggest(_FakeTrial("mid"), hpo.HpoSpace(),
                          respect_domain_priors=False)
    for knob in ("lr0", "batch", "imgsz", "optimizer", "cos_lr",
                 "warmup_epochs", "weight_decay", "patience",
                 "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale",
                 "shear", "perspective", "mosaic", "mixup", "copy_paste"):
        assert knob in params, f"{knob} is not searched"


def test_sampled_image_sizes_are_multiples_of_32():
    """A non-multiple-of-32 imgsz silently changes shape inside the model."""
    for size in hpo.IMGSZ_CHOICES:
        assert size % 32 == 0, size


def test_sampled_batch_sizes_are_bounded():
    """A 960px batch of 32 OOMs on most hardware and wastes the trial slot."""
    assert max(hpo.BATCH_CHOICES) <= 16
    assert all(b > 0 for b in hpo.BATCH_CHOICES)


@pytest.mark.parametrize("pick", ["low", "mid", "high"])
def test_every_sampled_point_is_inside_its_declared_bounds(pick):
    space = hpo.HpoSpace()
    params = hpo._suggest(_FakeTrial(pick), space,
                          respect_domain_priors=False)
    for name, value in params.items():
        bound = getattr(space, name, None)
        if isinstance(bound, tuple) and len(bound) == 2 and \
                all(isinstance(b, (int, float)) for b in bound):
            assert bound[0] <= value <= bound[1], f"{name}={value} outside {bound}"


def test_frozen_knobs_are_not_searched():
    space = hpo.HpoSpace(frozen=("lr0", "imgsz", "mosaic"))
    params = hpo._suggest(_FakeTrial("mid"), space,
                          respect_domain_priors=False)
    assert "lr0" not in params
    assert "imgsz" not in params
    assert "mosaic" not in params
    assert "batch" in params, "freezing must be selective"


# ==========================================================================
# domain priors — the part that stops the search buying a wrong model
# ==========================================================================

@pytest.mark.parametrize("pick", ["low", "mid", "high"])
def test_domain_priors_pin_horizontal_flip_to_zero(pick):
    """A mirrored nameplate is not a real thing.

    A search maximising validation mAP on a small dataset will happily switch
    horizontal flip on — it looks like free augmentation — and produce a model that
    has learned reversed device markings are normal. Physical correctness is not a
    tunable, so with priors respected these are held at 0 and never suggested.
    """
    trial = _FakeTrial(pick)
    params = hpo._suggest(trial, hpo.HpoSpace(), respect_domain_priors=True)
    assert params["fliplr"] == 0.0
    assert params["flipud"] == 0.0
    assert "fliplr" not in trial.suggested, "flip must not even be sampled"
    assert "flipud" not in trial.suggested


def test_domain_priors_bound_rotation():
    """Panels are gravity-oriented; a 45-degree view is not a real one."""
    trial = _FakeTrial("high")
    params = hpo._suggest(trial, hpo.HpoSpace(), respect_domain_priors=True)
    assert params["degrees"] <= hpo.DOMAIN_DEGREES_RANGE[1]


def test_disabling_domain_priors_searches_flip_explicitly():
    """The escape hatch must work, so the decision can be made deliberately."""
    trial = _FakeTrial("high")
    params = hpo._suggest(trial, hpo.HpoSpace(), respect_domain_priors=False)
    assert "fliplr" in trial.suggested
    assert params["fliplr"] > 0.0
    assert params["degrees"] > hpo.DOMAIN_DEGREES_RANGE[1]


# ==========================================================================
# sampled params must actually reach the trainer
# ==========================================================================

def test_sampled_params_reach_the_train_config():
    params = hpo._suggest(_FakeTrial("mid"), hpo.HpoSpace(),
                          respect_domain_priors=True)
    cfg = hpo._config_from(params, "d.yaml", "yolo11s", 25, "cpu", "t0")
    assert cfg.data == "d.yaml" and cfg.arch == "yolo11s" and cfg.epochs == 25
    kwargs = cfg.to_kwargs()
    for knob in ("lr0", "batch", "imgsz", "optimizer", "cos_lr",
                 "warmup_epochs", "patience", "mosaic", "mixup", "hsv_v"):
        assert kwargs[knob] == params[knob], f"{knob} did not reach the trainer"
    assert kwargs["fliplr"] == 0.0


def test_params_that_are_not_config_fields_ride_along_in_extra():
    """lrf and weight_decay are valid Ultralytics args but not TrainConfig fields.

    Dropping them silently would mean the search reported tuning weight decay while
    every trial actually trained at the default.
    """
    params = hpo._suggest(_FakeTrial("mid"), hpo.HpoSpace(),
                          respect_domain_priors=True)
    cfg = hpo._config_from(params, "d.yaml", "yolo11s", 10, "cpu", "t0")
    assert cfg.extra.get("weight_decay") == params["weight_decay"]
    assert cfg.extra.get("lrf") == params["lrf"]
    kwargs = cfg.to_kwargs()
    assert kwargs["weight_decay"] == params["weight_decay"]
    assert kwargs["lrf"] == params["lrf"]


# ==========================================================================
# scoring
# ==========================================================================

def test_score_reads_the_ultralytics_map_key():
    res = tr.TrainResult("yolo11s", "trained",
                         ultralytics_metrics={"metrics/mAP50-95(B)": 0.4231})
    assert hpo._score(res) == pytest.approx(0.4231)


@pytest.mark.parametrize("key", list(hpo._MAP_KEYS))
def test_score_accepts_every_known_metric_key_spelling(key):
    res = tr.TrainResult("yolo11s", "trained", ultralytics_metrics={key: 0.5})
    assert hpo._score(res) == pytest.approx(0.5)


def test_a_missing_metric_scores_none_not_zero():
    """0.0 would be a lie: it says 'this hyperparameter set is bad'.

    A trial that failed to train tells you nothing about its hyperparameters, and
    recording it as the worst possible score teaches the sampler to avoid that
    region of the space for a reason that has nothing to do with the region.
    """
    assert hpo._score(tr.TrainResult("yolo11s", "failed", "boom")) is None
    assert hpo._score(tr.TrainResult("yolo11s", "trained",
                                     ultralytics_metrics={})) is None


def test_epoch_map_reads_a_trainer_mid_run():
    class _Trainer:
        metrics = {"metrics/mAP50-95(B)": 0.31}

    assert hpo._epoch_map(_Trainer()) == pytest.approx(0.31)


def test_epoch_map_is_none_before_validation_metrics_exist():
    """Reporting 0.0 in epoch 1 would get every trial pruned immediately."""
    class _Empty:
        metrics = {}

    class _NoMetrics:
        pass

    assert hpo._epoch_map(_Empty()) is None
    assert hpo._epoch_map(_NoMetrics()) is None


# ==========================================================================
# pruning needs a real mechanism
# ==========================================================================

def test_training_aborted_propagates_instead_of_becoming_a_failed_result():
    """The contract that makes per-epoch pruning possible.

    train() deliberately swallows exceptions so one broken architecture cannot
    abort a benchmark of six. TrainingAborted is the documented exception: it must
    escape, or a pruned trial would be reported as a training failure.
    """
    def explode(_trainer):
        raise tr.TrainingAborted("pruned at epoch 3", 0.21)

    ok, _ = tr.ultralytics_available()
    if not ok:
        # Without ultralytics train() returns 'skipped' before reaching callbacks,
        # so assert the exception's own contract instead.
        with pytest.raises(tr.TrainingAborted) as exc:
            explode(None)
        assert exc.value.value == pytest.approx(0.21)
        assert "epoch 3" in exc.value.reason
        return
    cfg = tr.TrainConfig(data="missing.yaml", arch="yolo11n", epochs=1)
    with pytest.raises(tr.TrainingAborted):
        tr.train(cfg, export_onnx=False,
                 callbacks={"on_fit_epoch_end": explode})


def test_training_aborted_carries_the_reason_and_value():
    a = tr.TrainingAborted("behind the median", 0.12)
    assert a.reason == "behind the median"
    assert a.value == pytest.approx(0.12)
    assert "behind the median" in str(a)


def test_train_still_reports_ordinary_failures_as_results():
    """Everything other than TrainingAborted must stay a reported result."""
    cfg = tr.TrainConfig(data="definitely-missing.yaml", arch="yolo11s",
                         epochs=1)
    res = tr.train(cfg, export_onnx=False)
    assert res.status in ("skipped", "failed")
    assert res.reason


# ==========================================================================
# dataset health — HPO cannot fix a data problem
# ==========================================================================

def test_a_thin_dataset_is_called_out(tmp_path):
    """Tuning a 30-instance class is compute spent on nothing."""
    import cv2
    import numpy as np

    root = tmp_path / "d"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(3):
            cv2.imwrite(str(root / "images" / split / f"{i}.jpg"),
                        np.zeros((64, 64, 3), np.uint8))
            (root / "labels" / split / f"{i}.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n")

    warnings, gap = _dataset_health_of(str(root))
    assert warnings, "a 6-image dataset must produce a warning"
    joined = " ".join(warnings)
    assert "ZERO" in joined or "noisy" in joined
    assert gap is not None


def _dataset_health_of(root: str):
    return hpo._dataset_health(root)


def test_dataset_health_of_a_missing_root_is_silent():
    warnings, gap = hpo._dataset_health("/definitely/not/here")
    assert warnings == [] and gap is None


def test_dataset_health_of_none_is_silent():
    assert hpo._dataset_health(None) == ([], None)


# ==========================================================================
# reporting
# ==========================================================================

def test_history_table_ranks_by_score():
    res = hpo.HpoResult(
        status="completed",
        history=[
            {"number": 0, "value": 0.31, "state": "COMPLETE",
             "params": {"lr0": 0.01, "batch": 8, "imgsz": 960,
                        "optimizer": "SGD"}},
            {"number": 1, "value": 0.42, "state": "COMPLETE",
             "params": {"lr0": 0.005, "batch": 4, "imgsz": 640,
                        "optimizer": "AdamW"}},
            {"number": 2, "value": None, "state": "PRUNED", "params": {}},
        ])
    text = hpo.format_history(res)
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("-")]
    # Header, then trial 1 (0.42) ahead of trial 0 (0.31); the pruned trial has no
    # score and is excluded.
    assert "mAP50-95" in lines[0]
    assert lines[1].split()[1] == "1"
    assert lines[2].split()[1] == "0"


def test_history_table_of_nothing_is_not_a_crash():
    assert "no completed trials" in hpo.format_history(hpo.HpoResult("completed"))


def test_result_is_json_serialisable(tmp_path):
    res = hpo.HpoResult(status="completed", best_value=0.4,
                        best_params={"lr0": 0.01}, trials_run=3)
    path = hpo.write_result(res, str(tmp_path / "out" / "hpo.json"))
    import json

    with open(path, encoding="utf-8") as fh:
        loaded = json.load(fh)
    assert loaded["best_value"] == 0.4
    assert loaded["best_params"]["lr0"] == 0.01
