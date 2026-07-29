"""
Runtime benchmarking and automatic model selection.

The decision this module automates is the one most often got wrong: ranking
detectors on mAP alone always picks the largest, and for a platform whose default
deployment is ONNX Runtime on CPU that is usually the wrong answer. A few points of
mAP for six times the latency is a bad trade on a 4-core box.

So these tests pin down three things:

1. Timing is measured honestly — warmup excluded, percentiles computed, unmeasurable
   quantities reported as ``None`` with a reason rather than estimated.
2. Selection trades accuracy against speed, disqualifies rather than penalises a
   candidate that misses a hard requirement, and **says what it gave up** so a human
   can overrule it.
3. Weights can override the trade, because the right balance is a deployment
   decision and not something this module can know.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from training.electrical import bench as bm


# ==========================================================================
# fakes — a real backend needs weights that do not exist in CI
# ==========================================================================

class _FakeBackend:
    """A loaded backend whose inference cost is controllable."""

    ready = True
    backend_id = "fake"

    def __init__(self, delay_s: float = 0.0, fail: bool = False,
                 warmup_penalty_s: float = 0.0):
        self.delay_s = delay_s
        self.fail = fail
        self.warmup_penalty_s = warmup_penalty_s
        self.calls = 0

    def infer(self, frame):
        self.calls += 1
        if self.fail:
            raise RuntimeError("inference exploded")
        import time

        # The first few calls are slower, mimicking lazy graph init / arena
        # allocation. If warmup were included in the measurement, the mean would
        # be inflated by an amount that depends on nothing interesting.
        delay = self.delay_s
        if self.calls <= 3:
            delay += self.warmup_penalty_s
        if delay:
            time.sleep(delay)
        return []


class _NotReadyBackend:
    ready = False
    _reason = "weights_missing"


def _image_dir(tmp_path, n: int = 8) -> str:
    d = tmp_path / "images" / "val"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        cv2.imwrite(str(d / f"p{i}.jpg"),
                    np.full((120, 160, 3), 90 + i, np.uint8))
    return str(d)


# ==========================================================================
# profiling
# ==========================================================================

def test_profile_measures_latency_and_fps(tmp_path):
    prof = bm.profile_backend(_FakeBackend(delay_s=0.004),
                              _image_dir(tmp_path), "fake", warmup=2, runs=12)
    assert prof.status == "measured"
    assert prof.runs == 12
    assert prof.latency_ms_p50 is not None and prof.latency_ms_p50 >= 3.0
    assert prof.latency_ms_p95 >= prof.latency_ms_p50
    assert prof.latency_ms_max >= prof.latency_ms_p95
    assert prof.latency_ms_min <= prof.latency_ms_p50
    assert prof.fps and prof.fps > 0


def test_warmup_runs_are_excluded_from_the_measurement(tmp_path):
    """The regression that makes a benchmark meaningless if it is not handled.

    The backend's first three calls take an extra 40 ms. With warmup=3 those must
    not appear in the measured samples, so the measured max stays far below the
    warmup cost.
    """
    backend = _FakeBackend(delay_s=0.002, warmup_penalty_s=0.040)
    prof = bm.profile_backend(backend, _image_dir(tmp_path), "fake",
                              warmup=3, runs=10)
    assert prof.status == "measured"
    assert backend.calls == 13, "warmup calls must still be executed"
    assert prof.latency_ms_max < 30.0, (
        f"measured max {prof.latency_ms_max:.1f} ms includes warmup cost")


def test_profile_calls_the_backend_exactly_warmup_plus_runs(tmp_path):
    backend = _FakeBackend()
    bm.profile_backend(backend, _image_dir(tmp_path), "f", warmup=4, runs=9)
    assert backend.calls == 13


def test_profile_skips_a_backend_that_is_not_ready(tmp_path):
    prof = bm.profile_backend(_NotReadyBackend(), _image_dir(tmp_path), "nr")
    assert prof.status == "skipped"
    assert prof.reason == "weights_missing"


def test_profile_skips_when_there_are_no_real_images(tmp_path):
    """Timing on random noise would be unrepresentative, so it is refused."""
    prof = bm.profile_backend(_FakeBackend(), str(tmp_path / "empty"), "f")
    assert prof.status == "skipped"
    assert "REAL images" in prof.reason
    assert "candidate count" in prof.reason


def test_profile_reports_a_failing_backend_without_raising(tmp_path):
    prof = bm.profile_backend(_FakeBackend(fail=True), _image_dir(tmp_path), "f",
                              warmup=0, runs=3)
    assert prof.status == "failed"
    assert "inference exploded" in prof.reason


def test_profile_warns_when_too_few_runs_for_a_meaningful_p95(tmp_path):
    prof = bm.profile_backend(_FakeBackend(), _image_dir(tmp_path), "f",
                              warmup=0, runs=4)
    assert any("p95" in n for n in prof.notes)


def test_profile_records_the_environment(tmp_path):
    """A benchmark that does not say what it ran on is not reproducible."""
    prof = bm.profile_backend(_FakeBackend(), _image_dir(tmp_path), "f",
                              warmup=1, runs=5)
    env = prof.environment
    assert env["python"] and env["platform"]
    assert "cpu_count" in env and "device_requested" in env
    # onnxruntime/torch presence must be recorded, even as None.
    assert "onnxruntime" in env and "torch" in env


def test_unmeasurable_fields_are_none_not_guessed(tmp_path):
    """A fake backend exposes no torch module and no weights file."""
    prof = bm.profile_backend(_FakeBackend(), _image_dir(tmp_path), "f",
                              warmup=1, runs=5)
    assert prof.parameters is None
    assert prof.model_file_mb is None
    assert prof.cuda_peak_mb is None


def test_to_dict_rounds_without_losing_none(tmp_path):
    prof = bm.profile_backend(_FakeBackend(), _image_dir(tmp_path), "f",
                              warmup=1, runs=5)
    d = prof.to_dict()
    assert isinstance(d["latency_ms_mean"], float)
    assert d["parameters"] is None


def test_percentile_is_nearest_rank():
    """Standard nearest-rank: rank = ceil(p/100 * N), 1-based.

    Regression: an earlier ``round(x + 0.5)`` formula was off by one whenever
    ``p/100 * N`` landed on an integer, so the p50 of 1..10 came out as 6.0 and
    every reported median was overstated by one sample.
    """
    s = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert bm._pct(s, 50) == 5.0
    assert bm._pct(s, 90) == 9.0
    assert bm._pct(s, 95) == 10.0
    assert bm._pct(s, 100) == 10.0
    assert bm._pct(s, 10) == 1.0
    assert bm._pct([], 50) == 0.0
    assert bm._pct([7.0], 50) == 7.0


def test_percentiles_are_monotonic_over_a_real_sample():
    s = sorted(float(x) for x in range(1, 101))
    values = [bm._pct(s, p) for p in (5, 25, 50, 75, 90, 95, 99)]
    assert values == sorted(values)
    assert bm._pct(s, 50) == 50.0
    assert bm._pct(s, 95) == 95.0


# ==========================================================================
# speed score
# ==========================================================================

def test_speed_score_is_monotonic_and_bounded():
    budget = 4000.0
    scores = [bm._speed_score(ms, budget)
              for ms in (10, 50, 200, 800, 2000, 3999)]
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores == sorted(scores, reverse=True), "faster must score higher"


def test_speed_score_saturates_at_the_bounds():
    assert bm._speed_score(5.0, 4000.0) == 1.0
    assert bm._speed_score(4000.0, 4000.0) == 0.0
    assert bm._speed_score(9999.0, 4000.0) == 0.0


def test_speed_score_of_unknown_latency_is_none():
    assert bm._speed_score(None, 4000.0) is None
    assert bm._speed_score(0.0, 4000.0) is None


def test_speed_score_is_ratio_based_not_absolute():
    """100→200 ms should cost about as much score as 1000→2000 ms."""
    budget = 8000.0
    a = bm._speed_score(100, budget) - bm._speed_score(200, budget)
    b = bm._speed_score(1000, budget) - bm._speed_score(2000, budget)
    assert a == pytest.approx(b, abs=1e-9)


# ==========================================================================
# selection
# ==========================================================================

def _ev(map95, map50, f1):
    return {"status": "evaluated", "map_50_95": map95, "map_50": map50,
            "overall": {"f1": f1}}


def _pr(p95, fps=None, params=None):
    return {"status": "measured", "latency_ms_p95": p95,
            "fps": fps or (1000.0 / p95), "parameters": params}


#: A realistic accuracy/latency curve for CPU ONNX at 960px.
_EVALS = {
    "yolo11n": _ev(0.41, 0.62, 0.64),
    "yolo11s": _ev(0.49, 0.71, 0.72),
    "yolo11m": _ev(0.52, 0.74, 0.75),
    "yolo11x": _ev(0.55, 0.77, 0.77),
}
_PROFS = {
    "yolo11n": _pr(210, params=2_600_000),
    "yolo11s": _pr(520, params=9_400_000),
    "yolo11m": _pr(1450, params=20_100_000),
    "yolo11x": _pr(6200, params=56_900_000),
}


def test_selection_does_not_just_pick_the_most_accurate():
    """The whole reason this module exists."""
    sel = bm.select_best(_EVALS, _PROFS)
    assert sel["winner"] is not None
    assert sel["winner"] != "yolo11x", \
        "accuracy-only ranking would pick yolo11x; that is the bug being fixed"


def test_a_model_over_the_latency_budget_is_disqualified_not_penalised():
    sel = bm.select_best(_EVALS, _PROFS, latency_budget_ms=4000.0)
    dq = {d["label"] for d in sel["disqualified"]}
    assert "yolo11x" in dq
    assert "yolo11x" not in {r["label"] for r in sel["ranking"]}
    assert "exceeds" in next(d["reason"] for d in sel["disqualified"]
                             if d["label"] == "yolo11x")


def test_an_accuracy_floor_disqualifies_the_weakest():
    sel = bm.select_best(_EVALS, _PROFS, min_map_50_95=0.45)
    dq = {d["label"] for d in sel["disqualified"]}
    assert "yolo11n" in dq
    assert sel["winner"] != "yolo11n"


def test_selection_discloses_the_accuracy_it_traded_away():
    """A human must be able to overrule it, which needs the cost stated."""
    sel = bm.select_best(_EVALS, _PROFS, latency_budget_ms=10_000.0)
    rationale = sel["rationale"]
    if sel["winner"] != "yolo11x":
        assert "more accurate" in rationale
        assert "override with --weights" in rationale


def test_weights_can_prioritise_accuracy():
    accuracy_first = {"map_50_95": 0.80, "map_50": 0.10, "f1": 0.08,
                      "speed": 0.02}
    sel = bm.select_best(_EVALS, _PROFS, weights=accuracy_first,
                         latency_budget_ms=10_000.0)
    assert sel["winner"] == "yolo11x"


def test_weights_can_prioritise_speed():
    speed_first = {"map_50_95": 0.15, "map_50": 0.05, "f1": 0.05, "speed": 0.75}
    sel = bm.select_best(_EVALS, _PROFS, weights=speed_first)
    assert sel["winner"] == "yolo11n"


def test_ranking_is_ordered_by_score():
    sel = bm.select_best(_EVALS, _PROFS)
    scores = [r["score"] for r in sel["ranking"]]
    assert scores == sorted(scores, reverse=True)


def test_a_model_without_timing_is_scored_on_accuracy_and_flagged():
    """Silently ranking an unmeasured model last would be misleading."""
    sel = bm.select_best(_EVALS, {"yolo11s": _pr(520)})
    rows = {r["label"]: r for r in sel["ranking"]}
    assert rows["yolo11m"]["note"] and "accuracy only" in rows["yolo11m"]["note"]
    assert rows["yolo11s"]["note"] is None
    assert "not comparable" in sel["rationale"]


def test_unevaluated_candidates_are_disqualified_with_the_reason():
    evals = dict(_EVALS)
    evals["broken"] = {"status": "skipped", "reason": "no ground truth"}
    sel = bm.select_best(evals, _PROFS)
    dq = {d["label"]: d["reason"] for d in sel["disqualified"]}
    assert "broken" in dq and "no ground truth" in dq["broken"]


def test_selection_with_nothing_qualifying_says_so():
    sel = bm.select_best(_EVALS, _PROFS, min_map_50_95=0.99)
    assert sel["winner"] is None
    assert "No candidate qualified" in sel["rationale"]


def test_selection_with_no_input_says_so():
    sel = bm.select_best({}, {})
    assert sel["winner"] is None
    assert "nothing to choose between" in sel["rationale"]


def test_selection_accepts_runtime_profile_objects(tmp_path):
    """select_best must take RuntimeProfile as well as plain dicts."""
    prof = bm.profile_backend(_FakeBackend(delay_s=0.002),
                              _image_dir(tmp_path), "yolo11s",
                              warmup=1, runs=6)
    sel = bm.select_best({"yolo11s": _ev(0.49, 0.71, 0.72)},
                         {"yolo11s": prof})
    assert sel["winner"] == "yolo11s"
    assert sel["ranking"][0]["latency_ms_p95"] is not None


# ==========================================================================
# tables
# ==========================================================================

def test_profile_table_renders_measured_and_skipped_rows(tmp_path):
    good = bm.profile_backend(_FakeBackend(), _image_dir(tmp_path), "good",
                              warmup=1, runs=5)
    bad = bm.profile_backend(_NotReadyBackend(), _image_dir(tmp_path), "bad")
    text = bm.format_profile_table({"good": good, "bad": bad})
    assert "good" in text and "bad" in text
    assert "measured" in text and "skipped" in text
    assert "p95 ms" in text


def test_selection_table_lists_every_qualifying_model():
    text = bm.format_selection_table(bm.select_best(_EVALS, _PROFS))
    for label in ("yolo11n", "yolo11s", "yolo11m"):
        assert label in text
    assert "mAP50-95" in text


def test_selection_table_of_nothing_is_not_a_crash():
    assert "no qualifying" in bm.format_selection_table({"ranking": []})


# ==========================================================================
# integration with the training driver
# ==========================================================================

def test_benchmark_reports_selection_even_when_training_is_unavailable(tmp_path):
    """Without ultralytics every arch is skipped — the shape must still hold."""
    from training.electrical import train as tr

    out = tr.benchmark(str(tmp_path / "missing.yaml"), str(tmp_path),
                       archs=["yolo11s"], epochs=1)
    assert "selection" in out and "runtime" in out
    assert out["selection"]["winner"] is None
    assert out["recommended_arch"] is None
    assert "accuracy alone" in out["note"]


def test_profile_trained_skips_cleanly_for_a_missing_checkpoint(tmp_path):
    from training.electrical import train as tr

    prof = tr.profile_trained(str(tmp_path / "nope.pt"), str(tmp_path), "x")
    assert prof.status in ("skipped", "failed")
    assert prof.reason
