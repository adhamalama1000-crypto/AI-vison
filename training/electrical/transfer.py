"""
Synthetic → real domain transfer.

The recipe is proven on synthetic imagery. The product needs it on photographs of real
cabinets, and those are different distributions in ways that matter: procedural renders
have flat shading, hard edges, no specular highlights, no dust, no cable shadows, no
depth-of-field falloff, and a background that is literally a constant colour. A model
that scores 0.85 on those has learned "dark rectangle on grey" — which is not what an
MCB looks like in a torch-lit cabinet.

This module is the machinery for closing that gap **and for measuring whether each step
actually closes it**, because the intuition here is unreliable in a specific and
important way (see below).

The four things it provides
--------------------------
1. :func:`build_mixed` — a dataset whose *training* split combines synthetic and real
   images at a controlled ratio, and whose *validation* split is **real only**. This is
   non-negotiable. Fine-tuning on real data and validating on a mix measures nothing:
   the synthetic val images are easy, they dominate the mean, and the number goes up
   while real-world performance does not.
2. :func:`measure_domain_gap` — score an existing checkpoint on real validation data
   before any fine-tuning. That number is the gap, and it is the baseline every later
   claim is measured against.
3. :func:`fine_tune` — staged transfer: freeze the backbone while the head adapts, then
   unfreeze at a lower learning rate. Both stages are real training runs with real
   metrics, and the report shows what each stage bought.
4. :func:`compare_strategies` — trains the plausible strategies on the same data and
   ranks them.

Why (4) exists, and why you should not skip it
---------------------------------------------
The obvious plan — pretrain on synthetic, fine-tune on real — is **often worse than
simply fine-tuning from COCO**, and it is worth being blunt about why.

A COCO-pretrained backbone has seen millions of real photographs. Its early layers
encode real optics: soft shadows, specular highlights, sensor noise, motion blur,
depth-of-field. A backbone pretrained on a few hundred procedural renders has instead
learned features tuned to flat fills and hard synthetic edges, and fine-tuning has to
*unlearn* that before it can learn anything useful. Sim-to-real pretraining pays off
when the synthetic data is photorealistic and abundant — domain-randomised renders in
the tens of thousands. Flat-shaded procedural panels are neither.

So :data:`STRATEGIES` includes:

``coco_to_real``
    Straight fine-tune from COCO onto real data. **The baseline to beat**, and
    frequently the winner.
``coco_to_synth_to_real``
    The two-stage plan. Included because it might win, not because it will.
``mixed``
    Synthetic images mixed into a real training set as extra augmentation, capped at a
    modest fraction, validated on real only. Usually the best of the three when real
    data is scarce, because the synthetic images add shape variety without getting a
    vote on what "photograph" means.
``real_only``
    Fine-tune from COCO on real data with no synthetic at all. The control that tells
    you whether the synthetic data contributed anything.

Run the comparison, read the numbers, keep the winner. Do not assume.
"""

from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

from . import datasets as ds
from . import train as tr

#: Transfer strategies, described for the report so a reader knows what was compared.
STRATEGIES: dict[str, str] = {
    "real_only": (
        "COCO-pretrained backbone, fine-tuned on real images only. The control: if it "
        "matches or beats the others, the synthetic data contributed nothing and the "
        "pipeline is simpler without it."),
    "coco_to_real": (
        "COCO-pretrained backbone, fine-tuned on real images with synthetic images "
        "available but unused. Identical to real_only unless a mix ratio is set — kept "
        "as an explicit name because it is the baseline people mean by 'just train on "
        "real data'."),
    "coco_to_synth_to_real": (
        "Two stages: pretrain on synthetic, then fine-tune on real. The intuitive plan, "
        "and frequently WORSE than going straight from COCO — a backbone tuned on a few "
        "hundred flat-shaded renders has to unlearn that before it can learn real "
        "optics. Included so the numbers decide."),
    "mixed": (
        "Synthetic images mixed into the real training set at a capped fraction, "
        "validated on real only. Usually the best of the three when real data is "
        "scarce: the synthetic images add shape and layout variety without getting a "
        "vote on what a photograph looks like."),
}

#: Default share of the *training* split allowed to be synthetic in ``mixed``.
#: Above roughly a third, the synthetic distribution starts dominating the gradient and
#: the model drifts back toward "dark rectangle on grey".
DEFAULT_SYNTH_FRACTION = 0.3

#: Freeze depth for stage 1 of a staged fine-tune. 10 is the YOLO backbone.
BACKBONE_FREEZE_LAYERS = 10


@dataclass
class TransferResult:
    strategy: str
    status: str                        # completed | skipped | failed
    reason: Optional[str] = None
    stages: list = field(default_factory=list)
    weights: Optional[str] = None
    real_eval: dict = field(default_factory=dict)
    baseline_eval: dict = field(default_factory=dict)
    map_50: Optional[float] = None
    map_50_95: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy, "status": self.status,
            "reason": self.reason, "stages": self.stages,
            "weights": self.weights,
            "map_50": self.map_50, "map_50_95": self.map_50_95,
            "precision": self.precision, "recall": self.recall,
            "real_eval": self.real_eval, "baseline_eval": self.baseline_eval,
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------
# dataset construction
# --------------------------------------------------------------------------

def _images(root: str, split: str) -> list[str]:
    d = os.path.join(root, "images", split)
    if not os.path.isdir(d):
        return []
    return [f for f in sorted(os.listdir(d))
            if f.lower().endswith(ds.IMAGE_EXTS)]


def _copy_pair(src_root: str, split: str, fn: str,
               dst_root: str, dst_split: str, prefix: str,
               symlink: bool) -> None:
    stem, ext = os.path.splitext(fn)
    out_stem = f"{prefix}{stem}"
    d_img = os.path.join(dst_root, "images", dst_split)
    d_lbl = os.path.join(dst_root, "labels", dst_split)
    os.makedirs(d_img, exist_ok=True)
    os.makedirs(d_lbl, exist_ok=True)

    s_img = os.path.join(src_root, "images", split, fn)
    dst_img = os.path.join(d_img, out_stem + ext)
    if symlink:
        if os.path.lexists(dst_img):
            os.remove(dst_img)
        os.symlink(os.path.abspath(s_img), dst_img)
    else:
        shutil.copy2(s_img, dst_img)

    s_lbl = os.path.join(src_root, "labels", split, stem + ".txt")
    d = os.path.join(d_lbl, out_stem + ".txt")
    if os.path.exists(s_lbl):
        shutil.copy2(s_lbl, d)
    else:
        open(d, "w", encoding="utf-8").close()


def build_mixed(real_root: str, synth_root: str, dst_root: str,
                synth_fraction: float = DEFAULT_SYNTH_FRACTION,
                classes: Optional[Sequence[str]] = None,
                seed: int = 1234,
                symlink: bool = True,
                log: Optional[Callable[[str], None]] = None) -> dict:
    """Build a training set of real + synthetic, validated on **real only**.

    ``synth_fraction`` is the share of the *training* split that may be synthetic.
    Validation and test are taken exclusively from ``real_root``, and that is the point
    of this function: fine-tuning on real data while validating on a mix produces a
    rising number and a model that has not improved, because the synthetic validation
    images are easy and dominate the mean.

    Both roots must already share one label space — run ``cli scope`` on each with the
    same profile first, or the indices mean different things and the merge is silently
    wrong.
    """
    say = log or (lambda m: None)
    if not 0.0 <= synth_fraction < 1.0:
        raise ValueError(
            f"synth_fraction must be in [0, 1), got {synth_fraction}")

    real_train = _images(real_root, "train")
    real_val = _images(real_root, "val")
    real_test = _images(real_root, "test")
    # Carry the origin split with each name. Resolving it later by re-listing the
    # directory is both quadratic and wrong: a synthetic set may legitimately hold
    # ``panel_004.jpg`` in *both* train and val, and a membership test would map the
    # val copy back to train, silently pairing it with the wrong label file.
    synth_train: list[tuple[str, str]] = (
        [("train", f) for f in _images(synth_root, "train")]
        + [("val", f) for f in _images(synth_root, "val")])

    if not real_val:
        raise ValueError(
            f"{real_root} has no val split. Real-only validation is the whole point "
            f"of this function — without real validation images there is nothing to "
            f"measure domain transfer against.")
    if not real_train:
        raise ValueError(f"{real_root} has no train split")

    # How many synthetic images to admit so that they are `synth_fraction` of the
    # final training split: n_synth = f/(1-f) * n_real.
    rng = random.Random(seed)
    want_synth = (int(round(len(real_train) * synth_fraction
                            / (1.0 - synth_fraction)))
                  if synth_fraction > 0 else 0)
    take_synth = min(want_synth, len(synth_train))
    chosen_synth = rng.sample(synth_train, take_synth) if take_synth else []

    for fn in real_train:
        _copy_pair(real_root, "train", fn, dst_root, "train", "real_", symlink)
    for src_split, fn in chosen_synth:
        # The split goes in the prefix so a name present in both synthetic splits
        # produces two distinct destination files instead of one overwriting the other.
        _copy_pair(synth_root, src_split, fn, dst_root, "train",
                   f"synth_{src_split}_", symlink)
    for fn in real_val:
        _copy_pair(real_root, "val", fn, dst_root, "val", "real_", symlink)
    for fn in real_test:
        _copy_pair(real_root, "test", fn, dst_root, "test", "real_", symlink)

    # The label space comes from the real dataset, which is authoritative — the
    # synthetic set must already have been mapped onto it.
    if classes:
        from . import profiles as pf

        pf.write_profile_yaml(
            dst_root, pf.ClassProfile(name="transfer", classes=tuple(classes)))
    else:
        src_yaml = os.path.join(real_root, "dataset.yaml")
        if os.path.exists(src_yaml):
            shutil.copy2(src_yaml, os.path.join(dst_root, "dataset.yaml"))
            _rewrite_yaml_path(os.path.join(dst_root, "dataset.yaml"), dst_root)
        else:
            ds.write_dataset_yaml(dst_root)
        for extra in ("classes.json",):
            src = os.path.join(real_root, extra)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dst_root, extra))

    total_train = len(real_train) + len(chosen_synth)
    report = {
        "dst_root": dst_root,
        "dataset_yaml": os.path.join(dst_root, "dataset.yaml"),
        "train": {"real": len(real_train), "synthetic": len(chosen_synth),
                  "total": total_train,
                  "synthetic_fraction": (round(len(chosen_synth) / total_train, 4)
                                         if total_train else 0.0)},
        "val": {"real": len(real_val), "synthetic": 0},
        "test": {"real": len(real_test), "synthetic": 0},
        "synth_requested": want_synth,
        "synth_available": len(synth_train),
        "validation_policy": (
            "Validation and test are REAL ONLY, by construction. Validating on a "
            "synth/real mix makes the metric rise without the model improving: the "
            "synthetic images are easy and they dominate the mean."),
        "warnings": [],
    }
    if want_synth > len(synth_train):
        report["warnings"].append(
            f"asked for {want_synth} synthetic image(s) to hit a "
            f"{synth_fraction:.0%} fraction but only {len(synth_train)} exist, so the "
            f"achieved fraction is "
            f"{report['train']['synthetic_fraction']:.0%}.")
    if len(real_val) < 50:
        report["warnings"].append(
            f"only {len(real_val)} real validation image(s). Per-class metrics from "
            f"this few are noisy enough that a 5-point mAP difference between "
            f"strategies is not a real difference — collect more before drawing "
            f"conclusions from a comparison.")
    if len(real_train) < 100:
        report["warnings"].append(
            f"only {len(real_train)} real training image(s). Fine-tuning on this "
            f"little will overfit quickly; keep the freeze stage and watch for val "
            f"mAP turning over.")
    say(f"train: {len(real_train)} real + {len(chosen_synth)} synthetic; "
        f"val: {len(real_val)} real (real-only by design)")
    for w in report["warnings"]:
        say(f"warning: {w}")
    return report


def _rewrite_yaml_path(yaml_path: str, root: str) -> None:
    """Point a copied dataset.yaml at its new root."""
    try:
        with open(yaml_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        with open(yaml_path, "w", encoding="utf-8") as fh:
            for line in lines:
                if line.startswith("path:"):
                    fh.write(f"path: {os.path.abspath(root)}\n")
                else:
                    fh.write(line)
    except OSError:
        pass


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------

def measure_domain_gap(weights: str, synth_root: str, real_root: str,
                       split: str = "val", imgsz: int = 640,
                       device: str = "cpu",
                       log: Optional[Callable[[str], None]] = None) -> dict:
    """Score one checkpoint on synthetic and on real data, and name the gap.

    This is the number that matters before any fine-tuning starts. A synthetic-trained
    model at 0.85 on synthetic and 0.05 on real has a gap of 0.80, and that gap — not
    the 0.85 — is what the transfer phase has to close.
    """
    say = log or (lambda m: None)
    params = {"weights": weights, "imgsz": imgsz, "device": device}

    say("scoring on synthetic validation data")
    synth = tr.evaluate_backend("industrial_ultralytics", synth_root, split,
                                params=params)
    say("scoring on REAL validation data")
    real = tr.evaluate_backend("industrial_ultralytics", real_root, split,
                               params=params)

    def _m(rep: dict, key: str):
        if rep.get("status") != "evaluated":
            return None
        if key in ("precision", "recall", "f1"):
            return (rep.get("overall") or {}).get(key)
        return rep.get(key)

    s50, r50 = _m(synth, "map_50"), _m(real, "map_50")
    gap = (s50 - r50) if (s50 is not None and r50 is not None) else None

    out = {
        "weights": weights,
        "synthetic": {"status": synth.get("status"),
                      "reason": synth.get("reason"),
                      "map_50": s50, "map_50_95": _m(synth, "map_50_95"),
                      "precision": _m(synth, "precision"),
                      "recall": _m(synth, "recall")},
        "real": {"status": real.get("status"), "reason": real.get("reason"),
                 "map_50": r50, "map_50_95": _m(real, "map_50_95"),
                 "precision": _m(real, "precision"),
                 "recall": _m(real, "recall")},
        "gap_map_50": (round(gap, 4) if gap is not None else None),
        "real_confusion_matrix": real.get("confusion_matrix"),
        "real_per_class": real.get("classes"),
        "real_false_positives": real.get("false_positives"),
        "real_false_negatives": real.get("false_negatives"),
    }
    out["interpretation"] = _gap_interpretation(s50, r50, gap)
    say(out["interpretation"])
    return out


def _gap_interpretation(s50: Optional[float], r50: Optional[float],
                        gap: Optional[float]) -> str:
    if r50 is None:
        return ("Real evaluation did not run, so there is no gap measurement. Without "
                "real validation data no claim about domain transfer can be made at "
                "all.")
    if s50 is None:
        return (f"Real mAP@50 is {r50:.3f}; synthetic evaluation did not run, so the "
                f"gap cannot be quantified.")
    if gap is None:
        return "Gap could not be computed."
    if gap <= 0.05:
        return (f"Synthetic {s50:.3f} vs real {r50:.3f} — a gap of {gap:.3f}, which is "
                f"small. Either the synthetic imagery is unusually representative or "
                f"the real validation set is too small to be discriminating; check the "
                f"image count before believing it.")
    if gap >= 0.5:
        return (f"Synthetic {s50:.3f} vs real {r50:.3f} — a gap of {gap:.3f}. The "
                f"synthetic score is essentially meaningless as a predictor of real "
                f"performance. This is the expected result for procedurally-rendered "
                f"panels, and it is why the synthetic model must not be shipped. Real "
                f"data is the only thing that closes it.")
    return (f"Synthetic {s50:.3f} vs real {r50:.3f} — a gap of {gap:.3f}. Substantial. "
            f"Fine-tuning on real data should close much of it; run "
            f"compare_strategies to find out which route closes most.")


# --------------------------------------------------------------------------
# fine-tuning
# --------------------------------------------------------------------------

def fine_tune(dataset_yaml: str, init_from: Optional[str],
              arch: str = "yolo11s",
              epochs: int = 60,
              imgsz: int = 640,
              batch: int = 16,
              device: str = "cpu",
              staged: bool = True,
              freeze_layers: int = BACKBONE_FREEZE_LAYERS,
              stage1_epochs: Optional[int] = None,
              lr0: float = 0.002,
              name: str = "finetune",
              log: Optional[Callable[[str], None]] = None) -> dict:
    """Fine-tune onto a new domain, optionally in two stages.

    ``staged`` runs the standard transfer schedule: freeze the backbone so the detection
    head adapts to the new domain without the pretrained features being destroyed by
    early large gradients, then unfreeze and train end to end at a lower learning rate.
    On a small real dataset this is meaningfully better than full fine-tuning from the
    first step, which tends to wash out the pretrained features before the head is
    producing useful gradients.

    ``lr0`` defaults well below the from-scratch 0.01: fine-tuning at a training-scale
    learning rate is the most common way to destroy a good checkpoint.
    """
    say = log or (lambda m: None)
    stages: list[dict] = []
    weights = init_from

    if staged:
        s1_epochs = stage1_epochs or max(1, epochs // 3)
        say(f"stage 1: backbone frozen ({freeze_layers} layers), {s1_epochs} epoch(s)")
        cfg1 = tr.TrainConfig(
            data=dataset_yaml, arch=arch, epochs=s1_epochs, imgsz=imgsz,
            batch=batch, device=device, init_from=weights, freeze=freeze_layers,
            lr0=lr0, name=f"{name}_s1_frozen")
        res1 = tr.train(cfg1, export_onnx=False, log=say)
        stages.append({"stage": 1, "frozen_layers": freeze_layers,
                       "epochs": s1_epochs, **res1.to_dict()})
        if res1.status != "trained" or not res1.weights:
            return {"status": "failed",
                    "reason": f"stage 1 did not train: {res1.reason}",
                    "stages": stages}
        weights = res1.weights

        s2_epochs = max(1, epochs - s1_epochs)
        say(f"stage 2: unfrozen, {s2_epochs} epoch(s) at lr0={lr0 / 2:g}")
        cfg2 = tr.TrainConfig(
            data=dataset_yaml, arch=arch, epochs=s2_epochs, imgsz=imgsz,
            batch=batch, device=device, init_from=weights, freeze=0,
            lr0=lr0 / 2, name=f"{name}_s2_full")
        res2 = tr.train(cfg2, export_onnx=False, log=say)
        stages.append({"stage": 2, "frozen_layers": 0, "epochs": s2_epochs,
                       **res2.to_dict()})
        if res2.status != "trained" or not res2.weights:
            return {"status": "failed",
                    "reason": f"stage 2 did not train: {res2.reason}",
                    "stages": stages, "weights": weights}
        weights = res2.weights
    else:
        cfg = tr.TrainConfig(
            data=dataset_yaml, arch=arch, epochs=epochs, imgsz=imgsz,
            batch=batch, device=device, init_from=weights, lr0=lr0,
            name=f"{name}_full")
        res = tr.train(cfg, export_onnx=False, log=say)
        stages.append({"stage": 1, "frozen_layers": None, "epochs": epochs,
                       **res.to_dict()})
        if res.status != "trained" or not res.weights:
            return {"status": "failed", "reason": res.reason, "stages": stages}
        weights = res.weights

    return {"status": "completed", "weights": weights, "stages": stages,
            "staged": staged}


# --------------------------------------------------------------------------
# strategy comparison
# --------------------------------------------------------------------------

def compare_strategies(real_root: str, synth_root: str, work_dir: str,
                       strategies: Sequence[str] = (
                           "real_only", "coco_to_synth_to_real", "mixed"),
                       arch: str = "yolo11s",
                       epochs: int = 40,
                       synth_pretrain_epochs: int = 20,
                       imgsz: int = 640,
                       batch: int = 16,
                       device: str = "cpu",
                       synth_fraction: float = DEFAULT_SYNTH_FRACTION,
                       synth_weights: Optional[str] = None,
                       log: Optional[Callable[[str], None]] = None) -> dict:
    """Train each strategy on the same data and rank them on REAL validation.

    Every strategy is scored on the same real-only validation split, which is the only
    comparison that means anything. Returns a ranking plus the reasoning, including
    whether the synthetic data helped at all — a question worth answering explicitly,
    because the intuitive answer is often wrong.
    """
    say = log or (lambda m: None)
    os.makedirs(work_dir, exist_ok=True)
    results: dict[str, TransferResult] = {}

    # Real-only dataset (validation is already real; this just drops synthetic).
    real_only_root = os.path.join(work_dir, "real_only")
    try:
        real_only_info = build_mixed(real_root, synth_root, real_only_root,
                                     synth_fraction=0.0, log=say)
    except ValueError as exc:
        return {"status": "failed", "reason": str(exc)}

    mixed_root = os.path.join(work_dir, "mixed")
    mixed_info = None
    if "mixed" in strategies:
        # Guarded like the real_only build above. The real_only build always passes
        # synth_fraction=0.0, so an out-of-range synth_fraction survives it and only
        # raises here — and this function's contract is to report a failure, not to
        # throw after having already built a dataset.
        try:
            mixed_info = build_mixed(real_root, synth_root, mixed_root,
                                     synth_fraction=synth_fraction, log=say)
        except ValueError as exc:
            return {"status": "failed",
                    "reason": f"could not build the mixed dataset: {exc}"}

    for strategy in strategies:
        say(f"\n=== strategy: {strategy} ===")
        if strategy not in STRATEGIES:
            results[strategy] = TransferResult(
                strategy, "skipped",
                f"unknown strategy; known: {', '.join(STRATEGIES)}")
            continue

        if strategy in ("real_only", "coco_to_real"):
            data_yaml = real_only_info["dataset_yaml"]
            init = None
        elif strategy == "mixed":
            data_yaml = mixed_info["dataset_yaml"]
            init = None
        else:                                     # coco_to_synth_to_real
            data_yaml = real_only_info["dataset_yaml"]
            init = synth_weights
            if not init:
                say("  pretraining on synthetic data first")
                pre = tr.TrainConfig(
                    data=os.path.join(synth_root, "dataset.yaml"), arch=arch,
                    epochs=synth_pretrain_epochs, imgsz=imgsz, batch=batch,
                    device=device, name=f"transfer_{strategy}_pretrain")
                pre_res = tr.train(pre, export_onnx=False, log=say)
                if pre_res.status != "trained" or not pre_res.weights:
                    results[strategy] = TransferResult(
                        strategy, "failed",
                        f"synthetic pretraining did not train: {pre_res.reason}")
                    continue
                init = pre_res.weights

        ft = fine_tune(data_yaml, init, arch=arch, epochs=epochs, imgsz=imgsz,
                       batch=batch, device=device,
                       name=f"transfer_{strategy}", log=say)
        if ft["status"] != "completed":
            results[strategy] = TransferResult(
                strategy, "failed", ft.get("reason"), stages=ft.get("stages", []))
            continue

        # Score on REAL validation only — the same split for every strategy.
        rep = tr.evaluate_backend(
            "industrial_ultralytics", real_only_root, "val",
            params={"weights": ft["weights"], "imgsz": imgsz, "device": device})
        res = TransferResult(
            strategy, "completed", stages=ft["stages"], weights=ft["weights"],
            real_eval=rep)
        if rep.get("status") == "evaluated":
            res.map_50 = rep.get("map_50")
            res.map_50_95 = rep.get("map_50_95")
            res.precision = (rep.get("overall") or {}).get("precision")
            res.recall = (rep.get("overall") or {}).get("recall")
            # recall can be absent even when mAP is present, so it is formatted
            # defensively — a crash here would throw away every strategy already trained.
            rec = f"{res.recall:.4f}" if res.recall is not None else "unavailable"
            say(f"  real mAP@50 {res.map_50:.4f}  recall {rec}")
        results[strategy] = res

    return _rank(results, real_only_info, mixed_info, synth_fraction)


def _rank(results: Mapping, real_info: dict, mixed_info: Optional[dict],
          synth_fraction: float) -> dict:
    scored = [r for r in results.values()
              if r.status == "completed" and r.map_50 is not None]
    scored.sort(key=lambda r: (-(r.map_50 or 0), -(r.map_50_95 or 0)))

    winner = scored[0] if scored else None
    control = results.get("real_only")
    synth_helped = None
    if winner and control and control.map_50 is not None \
            and winner.map_50 is not None:
        # Rounded to the precision the report prints. Unrounded, 0.520 - 0.500 is
        # 0.020000000000000018, so a delta sitting exactly on the threshold would clear
        # it on float representation error alone — the verdict has to be decided by the
        # number the reader actually sees, not by the binary expansion behind it.
        delta = round(winner.map_50 - control.map_50, 4)
        # 0.02 is a deliberately generous threshold: with a small real validation set,
        # anything at or under it is noise, and calling noise an improvement is how a
        # pipeline acquires a stage that costs compute and buys nothing.
        synth_helped = bool(winner.strategy != "real_only" and delta > 0.02)

    return {
        "status": "completed" if scored else "failed",
        "strategies": {k: v.to_dict() for k, v in results.items()},
        "ranking": [
            {"strategy": r.strategy, "map_50": r.map_50,
             "map_50_95": r.map_50_95, "precision": r.precision,
             "recall": r.recall, "weights": r.weights}
            for r in scored
        ],
        "winner": winner.strategy if winner else None,
        "winner_weights": winner.weights if winner else None,
        "synthetic_data_helped": synth_helped,
        "dataset": {"real_only": real_info,
                    "mixed": mixed_info,
                    "synth_fraction": synth_fraction},
        "strategy_descriptions": dict(STRATEGIES),
        "rationale": _rank_rationale(scored, control, synth_helped),
    }


def _rank_rationale(scored: Sequence[TransferResult],
                    control: Optional[TransferResult],
                    synth_helped: Optional[bool]) -> str:
    if not scored:
        return ("No strategy produced a scored model. Check the per-strategy reasons — "
                "on a small real dataset the usual cause is too few validation images "
                "for the evaluator to score anything.")
    top = scored[0]
    rec = f"{top.recall:.4f}" if top.recall is not None else "unavailable"
    parts = [f"{top.strategy} wins on REAL validation data with mAP@50 "
             f"{top.map_50:.4f} (recall {rec})."]
    if len(scored) > 1:
        parts.append(
            "Ranking: " + ", ".join(
                f"{r.strategy} {r.map_50:.4f}" for r in scored) + ".")
    if synth_helped is False and control is not None:
        parts.append(
            f"The synthetic data did NOT help: real_only scored "
            f"{control.map_50:.4f} and nothing beat it by more than noise. Drop the "
            f"synthetic stage — it costs compute and buys nothing. This is a common "
            f"outcome: a COCO backbone has seen millions of real photographs, and "
            f"pretraining on a few hundred flat-shaded renders replaces those features "
            f"with worse ones.")
    elif synth_helped and control is not None:
        parts.append(
            f"The synthetic data helped: {scored[0].strategy} beat the real_only "
            f"control ({control.map_50:.4f}) by "
            f"{scored[0].map_50 - control.map_50:+.4f} mAP@50.")
    parts.append(
        "All strategies were scored on the same REAL-ONLY validation split, which is "
        "the only comparison that means anything here.")
    return " ".join(parts)


def format_ranking(comparison: dict) -> str:
    rows = comparison.get("ranking") or []
    if not rows:
        return "no scored strategies"
    header = ("rank", "strategy", "mAP50", "mAP50-95", "precision", "recall")
    body = [(str(i), r["strategy"],
             f"{r['map_50']:.4f}" if r["map_50"] is not None else "-",
             f"{r['map_50_95']:.4f}" if r["map_50_95"] is not None else "-",
             f"{r['precision']:.4f}" if r["precision"] is not None else "-",
             f"{r['recall']:.4f}" if r["recall"] is not None else "-")
            for i, r in enumerate(rows, 1)]
    widths = [max(len(str(r[i])) for r in [header] + body)
              for i in range(len(header))]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    return "\n".join([line, "-" * len(line)]
                     + ["  ".join(str(c).ljust(w)
                                  for c, w in zip(r, widths)) for r in body])


def write_result(payload: dict, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path


__all__ = [
    "STRATEGIES", "DEFAULT_SYNTH_FRACTION", "BACKBONE_FREEZE_LAYERS",
    "TransferResult", "build_mixed", "measure_domain_gap", "fine_tune",
    "compare_strategies", "format_ranking", "write_result",
]
