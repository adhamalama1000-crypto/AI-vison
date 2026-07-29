"""
Hyperparameter optimisation for the component detector, with Optuna.

:class:`training.electrical.train.TrainConfig` carries hand-reasoned defaults —
``imgsz=960`` because an MCB in a wide cabinet shot is a handful of pixels,
``fliplr=0`` because a mirrored nameplate is not a real thing, ``close_mosaic=10``
because mosaic destroys the row context the model should learn. Those are good
priors and they are documented, but they were never *searched*. This module searches
them.

Search space (the brief's list, mapped onto Ultralytics knobs):

=========================  ===========================================
Learning rate              ``lr0`` (log), ``lrf`` (final LR fraction)
Batch size                 ``batch`` (categorical, VRAM-bounded)
Image size                 ``imgsz`` (categorical, multiples of 32)
Optimizer                  ``optimizer`` (SGD / Adam / AdamW / auto)
Scheduler                  ``cos_lr`` + ``warmup_epochs``
Weight decay               ``weight_decay`` (log)
Augmentation               HSV, geometric, mosaic/mixup/copy-paste
Early stopping             ``patience``
=========================  ===========================================

Two things make this honest rather than a knob-turning exercise:

**Domain priors are respected by default.** ``--respect-domain-priors`` (on by
default) keeps ``fliplr`` and ``flipud`` at 0 and bounds rotation to a sane band,
because a search that maximises validation mAP on a small dataset will happily turn
on horizontal flip — it looks like free augmentation — and produce a model that has
learned mirrored nameplates are normal. The search is not permitted to trade away
physical correctness for a fraction of a point. Pass
``--no-respect-domain-priors`` to search them anyway; the flag exists so the
decision is explicit.

**The objective is the metric you actually care about.** Optuna maximises mAP@50-95
from Ultralytics' own validation, not training loss. Loss is not comparable across
different ``imgsz`` values, so optimising it would silently prefer whichever image
size makes the loss number smallest.

Practicalities that matter for a real run:

* **Budget.** Each trial is a full (short) training run. With ``--trials 20
  --epochs 20`` on one GPU this is hours, not minutes. Start with a small
  ``--epochs`` to rank the space, then train the winner properly with
  ``cli train``. A hyperparameter ranking from 20 epochs transfers reasonably to
  120; a ranking from 2 epochs does not.
* **Pruning.** Median pruning stops a trial that is clearly behind at an
  intermediate epoch, which roughly halves wall-clock for the same information.
* **Resumability.** Studies are stored in SQLite, so an interrupted search resumes
  where it stopped instead of starting over.
* **It cannot fix a data problem.** If ``cli gap`` says a class has 30 instances, no
  hyperparameter makes that class work. HPO is worth running once the dataset is
  adequate, and is close to worthless before then — :func:`optimise` says so
  explicitly when it detects a thin dataset.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from . import datasets as ds
from . import train as tr

#: Batch sizes worth trying. Bounded above because a 960px batch of 32 will OOM on
#: anything short of an A100, and a trial that dies on OOM wastes the slot.
BATCH_CHOICES: tuple[int, ...] = (4, 8, 12, 16)

#: Image sizes, all multiples of 32 as the architecture requires. 640 is included
#: because it may win on latency grounds even if it loses a little mAP; 1280 because
#: small-object recall sometimes justifies the cost.
IMGSZ_CHOICES: tuple[int, ...] = (640, 800, 960, 1280)

OPTIMIZER_CHOICES: tuple[str, ...] = ("SGD", "Adam", "AdamW", "auto")

#: Rotation band when domain priors are respected. Inspectors do not hold a camera
#: level, but a panel is gravity-oriented — a 45° rotation is not a real view.
DOMAIN_DEGREES_RANGE = (0.0, 15.0)


@dataclass
class HpoSpace:
    """Which knobs to search, and within what bounds."""

    lr0: tuple[float, float] = (1e-4, 5e-2)
    lrf: tuple[float, float] = (0.005, 0.2)
    weight_decay: tuple[float, float] = (1e-5, 1e-2)
    warmup_epochs: tuple[float, float] = (0.0, 5.0)
    batch: tuple[int, ...] = BATCH_CHOICES
    imgsz: tuple[int, ...] = IMGSZ_CHOICES
    optimizer: tuple[str, ...] = OPTIMIZER_CHOICES
    patience: tuple[int, int] = (10, 40)
    # augmentation
    hsv_h: tuple[float, float] = (0.0, 0.03)
    hsv_s: tuple[float, float] = (0.2, 0.9)
    hsv_v: tuple[float, float] = (0.2, 0.8)
    degrees: tuple[float, float] = (0.0, 30.0)
    translate: tuple[float, float] = (0.0, 0.25)
    scale: tuple[float, float] = (0.2, 0.7)
    shear: tuple[float, float] = (0.0, 8.0)
    perspective: tuple[float, float] = (0.0, 0.002)
    mosaic: tuple[float, float] = (0.0, 1.0)
    mixup: tuple[float, float] = (0.0, 0.3)
    copy_paste: tuple[float, float] = (0.0, 0.3)
    fliplr: tuple[float, float] = (0.0, 0.5)
    flipud: tuple[float, float] = (0.0, 0.3)
    #: Knobs to leave alone entirely.
    frozen: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HpoResult:
    status: str                            # completed | skipped | failed
    reason: Optional[str] = None
    study_name: Optional[str] = None
    storage: Optional[str] = None
    trials_run: int = 0
    trials_pruned: int = 0
    trials_failed: int = 0
    best_value: Optional[float] = None
    best_params: dict = field(default_factory=dict)
    best_trial_number: Optional[int] = None
    baseline_value: Optional[float] = None
    history: list = field(default_factory=list)
    importances: dict = field(default_factory=dict)
    recommended_config: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    next_step: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def optuna_available() -> tuple[bool, Optional[str]]:
    try:
        import optuna
        return True, getattr(optuna, "__version__", None)
    except Exception:
        return False, None


def _suggest(trial, space: HpoSpace, respect_domain_priors: bool) -> dict:
    """Sample one hyperparameter set from the space."""
    frozen = set(space.frozen)

    def use(name: str) -> bool:
        return name not in frozen

    params: dict[str, Any] = {}

    if use("lr0"):
        params["lr0"] = trial.suggest_float("lr0", *space.lr0, log=True)
    if use("lrf"):
        params["lrf"] = trial.suggest_float("lrf", *space.lrf, log=True)
    if use("weight_decay"):
        params["weight_decay"] = trial.suggest_float(
            "weight_decay", *space.weight_decay, log=True)
    if use("optimizer"):
        params["optimizer"] = trial.suggest_categorical(
            "optimizer", list(space.optimizer))
    if use("batch"):
        params["batch"] = trial.suggest_categorical("batch", list(space.batch))
    if use("imgsz"):
        params["imgsz"] = trial.suggest_categorical("imgsz", list(space.imgsz))
    if use("cos_lr"):
        params["cos_lr"] = trial.suggest_categorical("cos_lr", [True, False])
    if use("warmup_epochs"):
        params["warmup_epochs"] = trial.suggest_float(
            "warmup_epochs", *space.warmup_epochs)
    if use("patience"):
        params["patience"] = trial.suggest_int("patience", *space.patience)

    # -- augmentation ---------------------------------------------------
    for name in ("hsv_h", "hsv_s", "hsv_v", "translate", "scale", "shear",
                 "perspective", "mosaic", "mixup", "copy_paste"):
        if use(name):
            lo, hi = getattr(space, name)
            params[name] = trial.suggest_float(name, lo, hi)

    if respect_domain_priors:
        # Not searched, and this is the point. A search maximising val mAP on a
        # small set will switch horizontal flip on because it looks like free
        # augmentation, and produce a model that believes mirrored nameplates and
        # reversed device markings are normal. Physical correctness is not a
        # tunable.
        params["fliplr"] = 0.0
        params["flipud"] = 0.0
        lo = max(space.degrees[0], DOMAIN_DEGREES_RANGE[0])
        hi = min(space.degrees[1], DOMAIN_DEGREES_RANGE[1])
        if use("degrees"):
            params["degrees"] = trial.suggest_float("degrees", lo, hi)
    else:
        if use("degrees"):
            params["degrees"] = trial.suggest_float("degrees", *space.degrees)
        if use("fliplr"):
            params["fliplr"] = trial.suggest_float("fliplr", *space.fliplr)
        if use("flipud"):
            params["flipud"] = trial.suggest_float("flipud", *space.flipud)

    return params


def _config_from(params: dict, data: str, arch: str, epochs: int,
                 device: str, name: str) -> tr.TrainConfig:
    """Build a TrainConfig from sampled params, routing unknown keys to `extra`."""
    known = {
        "lr0", "cos_lr", "warmup_epochs", "optimizer", "batch", "imgsz",
        "patience", "degrees", "translate", "scale", "shear", "perspective",
        "fliplr", "flipud", "hsv_h", "hsv_s", "hsv_v", "mosaic", "mixup",
        "copy_paste",
    }
    kwargs = {k: v for k, v in params.items() if k in known}
    # lrf / weight_decay are valid Ultralytics arguments but not TrainConfig
    # fields, so they ride along in `extra` rather than being dropped silently.
    extra = {k: v for k, v in params.items() if k not in known}
    return tr.TrainConfig(data=data, arch=arch, epochs=epochs, device=device,
                          name=name, extra=extra, **kwargs)


def _score(result: tr.TrainResult) -> Optional[float]:
    """Pull mAP@50-95 out of an Ultralytics result dict.

    Ultralytics key names have moved between versions, so several are accepted.
    Returns ``None`` rather than 0.0 when the metric is absent — a missing metric
    is not a bad score, and scoring it as 0.0 would make a broken trial look like
    a genuinely poor hyperparameter set.
    """
    metrics = result.ultralytics_metrics or {}
    for key in _MAP_KEYS:
        if key in metrics:
            return float(metrics[key])
    return None


#: Metric keys Ultralytics has used for mAP@50-95 across versions, most recent
#: first. Accepting several beats pinning one and silently losing the metric on an
#: upgrade.
_MAP_KEYS: tuple[str, ...] = (
    "metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95",
    "metrics/mAP_0.5:0.95",
)


def _epoch_map(trainer) -> Optional[float]:
    """Read mAP@50-95 from an Ultralytics trainer mid-run, for pruning.

    Returns ``None`` when the metric is not present yet — the first epoch or two
    often have no validation metrics — so the caller skips reporting rather than
    reporting a zero that would get the trial pruned for the wrong reason.
    """
    metrics = getattr(trainer, "metrics", None)
    if not isinstance(metrics, dict):
        return None
    for key in _MAP_KEYS:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _dataset_health(dataset_root: Optional[str]) -> tuple[list, Optional[dict]]:
    """Warn when the dataset is too thin for HPO to be worth the compute."""
    warnings: list[str] = []
    if not dataset_root or not os.path.isdir(dataset_root):
        return warnings, None
    try:
        analysis = ds.analyse_dataset(dataset_root)
        gap = ds.requirements_report(analysis, priority_only=True)
    except Exception:
        return warnings, None

    if gap["missing_classes"]:
        warnings.append(
            f"{len(gap['missing_classes'])} priority class(es) have ZERO "
            f"annotations in this dataset. Hyperparameter search cannot create "
            f"data — those classes will not work at any setting. Spend the compute "
            f"on collection first: {', '.join(gap['missing_classes'][:8])}"
            + ("..." if len(gap['missing_classes']) > 8 else ""))
    if analysis["images"] < 500:
        warnings.append(
            f"only {analysis['images']} image(s). On a set this small, validation "
            f"mAP is noisy enough that the search will largely be fitting that "
            f"noise — trial rankings will not reproduce. Treat any result as a "
            f"hint, not a conclusion.")
    return warnings, gap


def optimise(dataset_yaml: str,
             dataset_root: Optional[str] = None,
             arch: str = tr.DEFAULT_ARCH,
             trials: int = 20,
             epochs: int = 20,
             device: str = "cpu",
             space: Optional[HpoSpace] = None,
             respect_domain_priors: bool = True,
             study_name: str = "electrical_detector_hpo",
             storage: Optional[str] = None,
             prune: bool = True,
             seed: int = 1234,
             baseline: bool = True,
             log: Optional[Callable[[str], None]] = None) -> HpoResult:
    """Search hyperparameters for the detector and return the best config.

    Returns a result object in every case. Without ``optuna`` or ``ultralytics``
    installed it reports ``skipped`` with the reason and never pretends to have
    searched.
    """
    say = log or (lambda m: None)
    space = space or HpoSpace()

    ok_optuna, optuna_version = optuna_available()
    if not ok_optuna:
        return HpoResult("skipped",
                         "optuna is not installed (pip install optuna)")
    ok_ultra, ultra_version = tr.ultralytics_available()
    if not ok_ultra:
        return HpoResult("skipped",
                         "ultralytics is not installed, so no trial can train "
                         "(pip install ultralytics)")
    if not os.path.exists(dataset_yaml):
        return HpoResult("skipped", f"dataset yaml not found: {dataset_yaml}")
    if not tr.arch_available(arch):
        return HpoResult("skipped",
                         f"'{arch}' is not available in the installed "
                         f"ultralytics {ultra_version}")

    import optuna
    from optuna.pruners import MedianPruner, NopPruner
    from optuna.samplers import TPESampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    warnings, _gap = _dataset_health(dataset_root)
    for w in warnings:
        say(f"warning: {w}")

    storage_url = storage
    if storage_url is None:
        # SQLite by default so an interrupted 6-hour search resumes rather than
        # starting over.
        os.makedirs("runs/electrical", exist_ok=True)
        storage_url = f"sqlite:///runs/electrical/{study_name}.db"

    study = optuna.create_study(
        study_name=study_name, storage=storage_url, load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0)
        if prune else NopPruner())

    already = len(study.trials)
    if already:
        say(f"resuming study '{study_name}' with {already} existing trial(s)")

    # -- baseline: the hand-reasoned defaults ----------------------------
    # Run outside the study, as a reference value only. It is deliberately NOT
    # enqueued as a trial: the defaults are a fixed point to beat, and feeding them
    # in as a sampled trial would let the TPE sampler treat them as evidence about
    # a region of the space it did not choose.
    baseline_value: Optional[float] = None
    if baseline and not already:
        say("running the hand-tuned defaults first, so the search has something "
            "to beat")
        base_cfg = tr.TrainConfig(data=dataset_yaml, arch=arch, epochs=epochs,
                                  device=device, name=f"hpo_{arch}_baseline")
        base_res = tr.train(base_cfg, export_onnx=False, log=say)
        baseline_value = _score(base_res)
        if baseline_value is None:
            say(f"baseline produced no mAP metric ({base_res.status}: "
                f"{base_res.reason}) — the search will run without a reference")
        else:
            say(f"baseline mAP@50-95 = {baseline_value:.4f}")

    failed = 0

    def objective(trial) -> float:
        nonlocal failed
        params = _suggest(trial, space, respect_domain_priors)
        cfg = _config_from(params, dataset_yaml, arch, epochs, device,
                           name=f"hpo_{arch}_t{trial.number}")
        say(f"--- trial {trial.number}: lr0={params.get('lr0'):.5g} "
            f"batch={params.get('batch')} imgsz={params.get('imgsz')} "
            f"opt={params.get('optimizer')}")

        # Per-epoch reporting so the pruner has something to act on. Ultralytics
        # training is one blocking call, so without this callback MedianPruner
        # would never fire and "pruning enabled" would be a claim with no
        # mechanism behind it.
        state = {"epoch": 0, "last": None}

        def on_epoch_end(trainer) -> None:
            state["epoch"] += 1
            value = _epoch_map(trainer)
            if value is None:
                return
            state["last"] = value
            trial.report(value, state["epoch"])
            if trial.should_prune():
                raise tr.TrainingAborted(
                    f"pruned at epoch {state['epoch']} (mAP@50-95 {value:.4f} is "
                    f"behind the median of completed trials)", value)

        try:
            res = tr.train(cfg, export_onnx=False, log=say,
                           callbacks={"on_fit_epoch_end": on_epoch_end}
                           if prune else None)
        except tr.TrainingAborted as abort:
            say(f"    trial {trial.number}: {abort.reason}")
            raise optuna.TrialPruned(abort.reason) from None

        value = _score(res)
        if value is None:
            failed += 1
            # A trial that could not train is not a data point about the
            # hyperparameters. Prune it rather than recording a fake 0.0, which
            # would teach the sampler to avoid a region for the wrong reason.
            raise optuna.TrialPruned(
                f"no mAP metric ({res.status}: {res.reason})")
        trial.set_user_attr("weights", res.weights or "")
        trial.set_user_attr("duration_s", res.duration_s or 0.0)
        say(f"    trial {trial.number}: mAP@50-95 = {value:.4f}")
        return value

    try:
        study.optimize(objective, n_trials=trials, gc_after_trial=True)
    except KeyboardInterrupt:
        say("interrupted — the study is stored, so re-running resumes it")
    except Exception as exc:
        return HpoResult("failed", f"{type(exc).__name__}: {exc}",
                         study_name=study_name, storage=storage_url,
                         warnings=warnings)

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials
              if t.state == optuna.trial.TrialState.PRUNED]
    if not completed:
        return HpoResult(
            "failed",
            "no trial completed. Every training run failed or was pruned — check "
            "the reasons above; this is usually a dataset or VRAM problem rather "
            "than a search-space problem.",
            study_name=study_name, storage=storage_url,
            trials_pruned=len(pruned), trials_failed=failed,
            warnings=warnings)

    best = study.best_trial
    importances: dict = {}
    try:
        if len(completed) >= 4:
            from optuna.importance import get_param_importances
            importances = {k: round(float(v), 4) for k, v in
                           get_param_importances(study).items()}
    except Exception:
        importances = {}

    recommended = _config_from(dict(best.params), dataset_yaml, arch,
                               epochs, device, name=f"{arch}_hpo_best")
    result = HpoResult(
        status="completed",
        study_name=study_name,
        storage=storage_url,
        trials_run=len(completed),
        trials_pruned=len(pruned),
        trials_failed=failed,
        best_value=float(best.value),
        best_params=dict(best.params),
        best_trial_number=best.number,
        baseline_value=baseline_value,
        history=[{"number": t.number, "value": t.value,
                  "state": str(t.state), "params": t.params}
                 for t in study.trials],
        importances=importances,
        recommended_config=recommended.to_kwargs(),
        warnings=list(warnings),
    )

    if respect_domain_priors:
        result.warnings.append(
            "fliplr and flipud were held at 0 and rotation bounded to "
            f"{DOMAIN_DEGREES_RANGE[1]:.0f}° because panels are gravity-oriented "
            "and device markings are directional. The search was not allowed to "
            "turn on horizontal flip, which it otherwise would have — it looks "
            "like free augmentation and teaches the model that mirrored "
            "nameplates are normal. Re-run with respect_domain_priors=False to "
            "search them anyway.")
    if baseline_value is not None and result.best_value is not None:
        delta = result.best_value - baseline_value
        if delta <= 0.005:
            result.warnings.append(
                f"the search improved mAP@50-95 by only {delta:+.4f} over the "
                f"hand-tuned defaults. That is within noise for most validation "
                f"sets — keep the defaults, and spend the compute on data "
                f"instead.")
        else:
            result.warnings.append(
                f"the search improved mAP@50-95 by {delta:+.4f} over the "
                f"hand-tuned defaults.")

    result.next_step = (
        f"Train properly with the winning settings (the search used only "
        f"{epochs} epochs to rank the space):\n"
        f"  python -m training.electrical.cli train --data {dataset_yaml} "
        f"--arch {arch} --epochs 120 --device {device} "
        + " ".join(f"--{k} {v}" for k, v in best.params.items()
                   if k in ("imgsz", "batch"))
        + "\nThe full winning parameter set is in recommended_config; pass any "
          "not exposed as a CLI flag through TrainConfig.extra.")
    say(f"best trial {best.number}: mAP@50-95 = {best.value:.4f}")
    return result


def write_result(result: HpoResult, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2, default=str)
    return path


def format_history(result: HpoResult, top: int = 15) -> str:
    """A ranked trial table for a terminal."""
    rows = [h for h in result.history if h.get("value") is not None]
    rows.sort(key=lambda h: -h["value"])
    rows = rows[:top]
    if not rows:
        return "no completed trials"
    header = ("rank", "trial", "mAP50-95", "lr0", "batch", "imgsz", "optimizer")
    body = []
    for i, h in enumerate(rows, 1):
        p = h.get("params") or {}
        body.append((
            str(i), str(h["number"]), f"{h['value']:.4f}",
            f"{p.get('lr0', float('nan')):.5g}" if "lr0" in p else "-",
            str(p.get("batch", "-")), str(p.get("imgsz", "-")),
            str(p.get("optimizer", "-")),
        ))
    widths = [max(len(str(r[i])) for r in [header] + body)
              for i in range(len(header))]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    out = [line, "-" * len(line)]
    for r in body:
        out.append("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
    return "\n".join(out)


__all__ = [
    "BATCH_CHOICES", "IMGSZ_CHOICES", "OPTIMIZER_CHOICES",
    "DOMAIN_DEGREES_RANGE", "HpoSpace", "HpoResult", "optuna_available",
    "optimise", "write_result", "format_history",
]
