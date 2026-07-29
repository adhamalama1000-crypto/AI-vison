"""
Runtime benchmarking and automatic model selection.

:func:`training.electrical.train.benchmark` already trains several architectures on
one split and ranks them by measured mAP. Ranking on accuracy alone picks
``yolo11x`` every time, and for this platform that is usually the wrong answer: the
default deployment is ONNX Runtime on CPU, where ``yolo11x`` at 960px is on the order
of six times slower than ``yolo11s`` for a couple of points of mAP. A panel
inspection that takes twelve seconds instead of two is a different product.

So this module measures the other half — latency, throughput, memory — and
:func:`select_best` combines both halves into one decision that **states its
reasoning**, including what it gave up.

How the timing is done, and why it is trustworthy
-------------------------------------------------
Naive benchmarking of an inference call produces numbers that are wrong in
predictable ways, so:

* **Warmup runs are discarded.** The first inference pays lazy graph
  initialisation, memory-arena allocation and (on GPU) kernel autotuning. Including
  it inflates the mean by a factor that depends on nothing interesting.
* **Real images, not random noise.** Detector latency is data-dependent: NMS cost
  scales with the number of candidate boxes above threshold, and random noise
  produces an unrepresentative candidate count. A dataset directory is required.
* **Percentiles, not just the mean.** A mean hides tail latency. p95 is what a user
  experiences when the system feels slow; for a queue it is what determines whether
  work piles up.
* **Peak RSS delta, not absolute.** Absolute process memory includes the whole
  Python interpreter and every already-imported library, so it says nothing about
  the model. The delta across the measured window is attributable.
* **The thread count is recorded.** ONNX Runtime latency on CPU depends heavily on
  it, and a benchmark that does not say how many threads it used is not
  reproducible.

Nothing here estimates. If a measurement cannot be taken — no ``psutil`` for
memory, no torch for parameter counts — the field is ``None`` with a reason
alongside, never a plausible-looking guess.
"""

from __future__ import annotations

import gc
import os
import platform
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from . import datasets as ds

#: Inferences discarded before measurement starts.
DEFAULT_WARMUP = 3
#: Measured inferences. Enough for a stable p95 without making a benchmark a
#: coffee break; raise it for a release-gating measurement.
DEFAULT_RUNS = 30

#: Latency above which a model is unsuitable for interactive panel analysis,
#: milliseconds. Used as a hard filter by :func:`select_best`, not as a score
#: term — a model nobody will wait for is disqualified rather than penalised.
DEFAULT_LATENCY_BUDGET_MS = 4000.0

#: Weights for the composite score. Accuracy dominates because a fast wrong answer
#: is worthless, but speed is not free either.
DEFAULT_WEIGHTS = {"map_50_95": 0.45, "map_50": 0.20, "f1": 0.15, "speed": 0.20}


@dataclass
class RuntimeProfile:
    """Measured runtime characteristics of one loaded model."""

    label: str
    status: str                            # measured | skipped | failed
    reason: Optional[str] = None
    images: int = 0
    warmup: int = 0
    runs: int = 0
    latency_ms_mean: Optional[float] = None
    latency_ms_p50: Optional[float] = None
    latency_ms_p95: Optional[float] = None
    latency_ms_p99: Optional[float] = None
    latency_ms_min: Optional[float] = None
    latency_ms_max: Optional[float] = None
    latency_ms_stdev: Optional[float] = None
    fps: Optional[float] = None
    peak_rss_delta_mb: Optional[float] = None
    cuda_peak_mb: Optional[float] = None
    parameters: Optional[int] = None
    model_file_mb: Optional[float] = None
    device: str = "cpu"
    threads: Optional[int] = None
    environment: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        out = {k: v for k, v in self.__dict__.items()}
        for key in ("latency_ms_mean", "latency_ms_p50", "latency_ms_p95",
                    "latency_ms_p99", "latency_ms_min", "latency_ms_max",
                    "latency_ms_stdev", "fps", "peak_rss_delta_mb",
                    "cuda_peak_mb", "model_file_mb"):
            if out.get(key) is not None:
                out[key] = round(float(out[key]), 3)
        return out


def _environment(device: str) -> dict:
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "device_requested": device,
    }
    try:
        import onnxruntime as ort  # type: ignore
        env["onnxruntime"] = ort.__version__
        env["onnxruntime_providers"] = ort.get_available_providers()
    except Exception:
        env["onnxruntime"] = None
    try:
        import torch  # type: ignore
        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:
        env["torch"] = None
        env["cuda_available"] = False
    return env


def _peak_rss_mb() -> Optional[float]:
    try:
        import psutil  # type: ignore
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except Exception:
        return None


def _model_size_mb(backend) -> Optional[float]:
    path = getattr(backend, "_weights_path", None)
    if path and os.path.exists(str(path)):
        return os.path.getsize(str(path)) / (1024 ** 2)
    return None


def _parameter_count(backend) -> Optional[int]:
    """Parameter count, when the backend exposes a torch module."""
    model = getattr(backend, "_model", None)
    inner = getattr(model, "model", None)
    if inner is None:
        return None
    try:
        return int(sum(p.numel() for p in inner.parameters()))
    except Exception:
        return None


def _load_images(image_dir: str, limit: int) -> list:
    """Read up to ``limit`` real images for timing."""
    import cv2

    if not os.path.isdir(image_dir):
        return []
    files = [f for f in sorted(os.listdir(image_dir))
             if f.lower().endswith(ds.IMAGE_EXTS)][:limit]
    out = []
    for fn in files:
        img = cv2.imread(os.path.join(image_dir, fn), cv2.IMREAD_COLOR)
        if img is not None:
            out.append(img)
    return out


def profile_backend(backend, image_dir: str, label: str,
                    warmup: int = DEFAULT_WARMUP,
                    runs: int = DEFAULT_RUNS,
                    device: str = "cpu",
                    log: Optional[Callable[[str], None]] = None
                    ) -> RuntimeProfile:
    """Measure latency, throughput and memory for one loaded backend.

    ``backend`` must already be loaded and ready. Returns a profile in every case;
    a failure is reported rather than raised, so one broken model does not abort a
    comparison of six.
    """
    say = log or (lambda m: None)
    env = _environment(device)
    prof = RuntimeProfile(label=label, status="failed", device=device,
                          warmup=warmup, environment=env)

    if backend is None or not getattr(backend, "ready", False):
        prof.status = "skipped"
        prof.reason = (getattr(backend, "_reason", None)
                       or getattr(backend, "_error", None)
                       or "backend is not loaded")
        return prof

    images = _load_images(image_dir, max(1, warmup + runs))
    if not images:
        prof.status = "skipped"
        prof.reason = (
            f"no readable images in {image_dir}. Timing needs REAL images: "
            f"detector latency is data-dependent because NMS cost scales with the "
            f"candidate count, so random noise would give an unrepresentative "
            f"number.")
        return prof
    prof.images = len(images)

    def run_once(img):
        if hasattr(backend, "recognize"):
            return backend.recognize(img)
        return backend.infer(img)

    try:
        # -- warmup: discarded, and that is the point ---------------------
        for i in range(warmup):
            run_once(images[i % len(images)])

        gc.collect()
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

        rss_before = _peak_rss_mb()
        rss_peak = rss_before or 0.0
        samples: list[float] = []

        for i in range(runs):
            img = images[i % len(images)]
            t0 = time.perf_counter()
            run_once(img)
            samples.append((time.perf_counter() - t0) * 1000.0)
            current = _peak_rss_mb()
            if current is not None:
                rss_peak = max(rss_peak, current)

        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                prof.cuda_peak_mb = (torch.cuda.max_memory_allocated()
                                     / (1024 ** 2))
        except Exception:
            pass
    except Exception as exc:
        prof.status = "failed"
        prof.reason = f"{type(exc).__name__}: {exc}"
        return prof

    samples.sort()
    prof.runs = len(samples)
    prof.latency_ms_mean = statistics.fmean(samples)
    prof.latency_ms_p50 = _pct(samples, 50)
    prof.latency_ms_p95 = _pct(samples, 95)
    prof.latency_ms_p99 = _pct(samples, 99)
    prof.latency_ms_min = samples[0]
    prof.latency_ms_max = samples[-1]
    prof.latency_ms_stdev = (statistics.stdev(samples) if len(samples) > 1
                             else 0.0)
    prof.fps = (1000.0 / prof.latency_ms_mean) if prof.latency_ms_mean else None
    if rss_before is not None:
        prof.peak_rss_delta_mb = max(0.0, rss_peak - rss_before)
    else:
        prof.notes.append("psutil is not installed, so memory was not measured")
    prof.parameters = _parameter_count(backend)
    prof.model_file_mb = _model_size_mb(backend)
    prof.threads = _thread_count(backend)
    prof.status = "measured"

    if prof.runs < 10:
        prof.notes.append(
            f"only {prof.runs} measured run(s) — p95 from this few samples is not "
            f"meaningful; raise --runs for a number worth quoting")
    if prof.latency_ms_stdev and prof.latency_ms_mean:
        cv = prof.latency_ms_stdev / prof.latency_ms_mean
        if cv > 0.35:
            prof.notes.append(
                f"latency varied a lot (stdev/mean = {cv:.2f}); the machine was "
                f"probably doing something else. Re-run on an idle host before "
                f"trusting these numbers for a deployment decision.")

    say(f"[{label}] p50 {prof.latency_ms_p50:.0f} ms  p95 "
        f"{prof.latency_ms_p95:.0f} ms  {prof.fps:.2f} FPS"
        + (f"  +{prof.peak_rss_delta_mb:.0f} MB"
           if prof.peak_rss_delta_mb is not None else ""))
    return prof


def _pct(sorted_samples: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile of an already-sorted sample list.

    ``rank = ceil(pct/100 * N)``, 1-based, clamped — the standard definition. An
    earlier version used ``round(x + 0.5)``, which is off by one whenever
    ``pct/100 * N`` lands on an integer: the p50 of 1..10 came out as 6.0 instead
    of 5.0, quietly overstating every median.
    """
    if not sorted_samples:
        return 0.0
    import math

    n = len(sorted_samples)
    rank = math.ceil((pct / 100.0) * n)
    index = max(0, min(n - 1, rank - 1))
    return float(sorted_samples[index])


def _thread_count(backend) -> Optional[int]:
    sess = getattr(backend, "_sess", None)
    opts = getattr(sess, "_sess_options", None) if sess is not None else None
    for attr in ("intra_op_num_threads",):
        value = getattr(opts, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    env = os.environ.get("OMP_NUM_THREADS")
    if env and env.isdigit():
        return int(env)
    return None


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------

def _speed_score(latency_ms: Optional[float],
                 budget_ms: float) -> Optional[float]:
    """Map latency onto 0..1 — 1.0 at instant, 0.0 at the budget.

    Linear in log-latency rather than in latency, because the *ratio* is what a
    user perceives: 100 ms → 200 ms matters as much as 1 s → 2 s.
    """
    if latency_ms is None or latency_ms <= 0:
        return None
    import math

    floor_ms = 10.0
    if latency_ms <= floor_ms:
        return 1.0
    if latency_ms >= budget_ms:
        return 0.0
    return float(1.0 - math.log(latency_ms / floor_ms)
                 / math.log(budget_ms / floor_ms))


def select_best(evaluations: dict, profiles: dict,
                weights: Optional[dict] = None,
                latency_budget_ms: float = DEFAULT_LATENCY_BUDGET_MS,
                min_map_50_95: float = 0.0) -> dict:
    """Pick the model to deploy, and explain the choice.

    ``evaluations`` maps a label to an accuracy report from
    :func:`rtsp_backend.electrical.metrics.evaluate`; ``profiles`` maps the same
    labels to :class:`RuntimeProfile` dicts.

    A candidate is **disqualified** (not merely penalised) when it misses the
    latency budget or the accuracy floor, because those are requirements rather
    than preferences. Survivors are scored on a weighted sum of accuracy and speed.
    The result names the winner, the runner-up, and what choosing the winner cost —
    which is the part a human actually needs in order to overrule it.
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    total_w = sum(w.values()) or 1.0

    rows: list[dict] = []
    disqualified: list[dict] = []

    for label, ev in evaluations.items():
        if not isinstance(ev, dict) or ev.get("status") not in (None,
                                                               "evaluated"):
            disqualified.append({"label": label,
                                 "reason": f"not evaluated: "
                                           f"{ev.get('reason') if isinstance(ev, dict) else ev}"})
            continue
        prof = profiles.get(label) or {}
        if isinstance(prof, RuntimeProfile):
            prof = prof.to_dict()

        map_50_95 = float(ev.get("map_50_95") or 0.0)
        map_50 = float(ev.get("map_50") or 0.0)
        f1 = float((ev.get("overall") or {}).get("f1") or 0.0)
        latency = prof.get("latency_ms_p95") or prof.get("latency_ms_mean")

        if map_50_95 < min_map_50_95:
            disqualified.append({
                "label": label,
                "reason": f"mAP@50-95 {map_50_95:.3f} is below the required "
                          f"floor {min_map_50_95:.3f}"})
            continue
        if latency is not None and latency > latency_budget_ms:
            disqualified.append({
                "label": label,
                "reason": f"p95 latency {latency:.0f} ms exceeds the "
                          f"{latency_budget_ms:.0f} ms budget"})
            continue

        speed = _speed_score(latency, latency_budget_ms)
        if speed is None:
            # No timing available. Score on accuracy alone and say so, rather than
            # inventing a speed term or silently ranking it last.
            score = ((w["map_50_95"] * map_50_95 + w["map_50"] * map_50
                      + w["f1"] * f1)
                     / (total_w - w["speed"] or 1.0))
            timing_note = "no runtime measurement — scored on accuracy only"
        else:
            score = (w["map_50_95"] * map_50_95 + w["map_50"] * map_50
                     + w["f1"] * f1 + w["speed"] * speed) / total_w
            timing_note = None

        rows.append({
            "label": label,
            "score": round(score, 5),
            "map_50_95": round(map_50_95, 4),
            "map_50": round(map_50, 4),
            "f1": round(f1, 4),
            "latency_ms_p95": (round(latency, 1) if latency is not None
                               else None),
            "fps": prof.get("fps"),
            "peak_rss_delta_mb": prof.get("peak_rss_delta_mb"),
            "parameters": prof.get("parameters"),
            "speed_score": (round(speed, 4) if speed is not None else None),
            "note": timing_note,
        })

    rows.sort(key=lambda r: -r["score"])
    winner = rows[0] if rows else None
    runner_up = rows[1] if len(rows) > 1 else None

    return {
        "winner": winner["label"] if winner else None,
        "ranking": rows,
        "disqualified": disqualified,
        "weights": w,
        "latency_budget_ms": latency_budget_ms,
        "min_map_50_95": min_map_50_95,
        "rationale": _rationale(winner, runner_up, rows, disqualified,
                                latency_budget_ms),
    }


def _rationale(winner: Optional[dict], runner_up: Optional[dict],
               rows: Sequence[dict], disqualified: Sequence[dict],
               budget_ms: float) -> str:
    if not winner:
        if disqualified:
            return ("No candidate qualified. "
                    + "; ".join(f"{d['label']}: {d['reason']}"
                                for d in disqualified))
        return "No models were evaluated, so there is nothing to choose between."

    parts = [f"{winner['label']} wins with a composite score of "
             f"{winner['score']:.3f} "
             f"(mAP@50-95 {winner['map_50_95']:.3f}, "
             f"mAP@50 {winner['map_50']:.3f}, F1 {winner['f1']:.3f}"
             + (f", p95 {winner['latency_ms_p95']:.0f} ms"
                if winner.get("latency_ms_p95") else "") + ")."]

    if runner_up:
        d_map = winner["map_50_95"] - runner_up["map_50_95"]
        parts.append(
            f"Runner-up {runner_up['label']} scored {runner_up['score']:.3f} "
            f"({'+' if d_map < 0 else '-'}{abs(d_map):.3f} mAP@50-95 versus the "
            f"winner"
            + (f", p95 {runner_up['latency_ms_p95']:.0f} ms"
               if runner_up.get("latency_ms_p95") else "") + ").")
        # The honest disclosure: when the winner is not the most accurate model,
        # say what accuracy was traded away and let a human overrule it.
        most_accurate = max(rows, key=lambda r: r["map_50_95"])
        if most_accurate["label"] != winner["label"]:
            lost = most_accurate["map_50_95"] - winner["map_50_95"]
            parts.append(
                f"NOTE: {most_accurate['label']} is more accurate "
                f"(mAP@50-95 {most_accurate['map_50_95']:.3f}, i.e. {lost:.3f} "
                f"higher) but slower"
                + (f" ({most_accurate['latency_ms_p95']:.0f} ms vs "
                   f"{winner['latency_ms_p95']:.0f} ms p95)"
                   if most_accurate.get("latency_ms_p95")
                   and winner.get("latency_ms_p95") else "")
                + ". If accuracy matters more than latency for your deployment, "
                  "override with --weights or raise the latency budget.")

    if disqualified:
        parts.append(
            "Disqualified: "
            + "; ".join(f"{d['label']} ({d['reason']})" for d in disqualified)
            + ".")

    if any(r.get("note") for r in rows):
        parts.append(
            "Some candidates had no runtime measurement and were scored on "
            "accuracy alone — their ranking against timed models is not "
            "comparable. Re-run with a dataset directory to time them.")
    return " ".join(parts)


def format_profile_table(profiles: dict) -> str:
    """A fixed-width runtime table for a terminal."""
    rows = []
    for label, p in profiles.items():
        d = p.to_dict() if isinstance(p, RuntimeProfile) else dict(p)
        if d.get("status") != "measured":
            rows.append((label, d.get("status", "?"), "-", "-", "-", "-", "-"))
            continue
        rows.append((
            label, "measured",
            f"{d['latency_ms_p50']:.0f}" if d.get("latency_ms_p50") else "-",
            f"{d['latency_ms_p95']:.0f}" if d.get("latency_ms_p95") else "-",
            f"{d['fps']:.2f}" if d.get("fps") else "-",
            f"{d['peak_rss_delta_mb']:.0f}"
            if d.get("peak_rss_delta_mb") is not None else "n/a",
            f"{d['parameters'] / 1e6:.1f}M" if d.get("parameters") else "-",
        ))
    header = ("model", "status", "p50 ms", "p95 ms", "FPS", "ΔRSS MB", "params")
    widths = [max(len(str(r[i])) for r in [header] + rows)
              for i in range(len(header))]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    out = [line, "-" * len(line)]
    for r in rows:
        out.append("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
    return "\n".join(out)


def format_selection_table(selection: dict) -> str:
    rows = selection.get("ranking") or []
    if not rows:
        return "no qualifying candidates"
    header = ("rank", "model", "score", "mAP50-95", "mAP50", "F1", "p95 ms",
              "FPS")
    body = []
    for i, r in enumerate(rows, 1):
        body.append((
            str(i), r["label"], f"{r['score']:.3f}", f"{r['map_50_95']:.3f}",
            f"{r['map_50']:.3f}", f"{r['f1']:.3f}",
            f"{r['latency_ms_p95']:.0f}" if r.get("latency_ms_p95") else "-",
            f"{r['fps']:.2f}" if r.get("fps") else "-",
        ))
    widths = [max(len(str(r[i])) for r in [header] + body)
              for i in range(len(header))]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    out = [line, "-" * len(line)]
    for r in body:
        out.append("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
    return "\n".join(out)


__all__ = [
    "DEFAULT_WARMUP", "DEFAULT_RUNS", "DEFAULT_LATENCY_BUDGET_MS",
    "DEFAULT_WEIGHTS", "RuntimeProfile", "profile_backend", "select_best",
    "format_profile_table", "format_selection_table",
]
