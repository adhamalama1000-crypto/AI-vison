"""
Production-path evaluation and confidence sweep.

The trainer's validator is not the product. Ultralytics scores a checkpoint at a
confidence floor around 0.001, counting detections no deployed system would ever
surface; the API decodes at a floor, applies per-class acceptance thresholds, runs NMS
and a geometric plausibility gate, and demotes anything it is unsure about to
``unknown_industrial_component``. Those two numbers can differ by a factor of four, and
only the second one describes what a user will see.

This module evaluates through **exactly** the deployed path — the same registered
backend, the same preprocessing, the same gates, the same NMS — and sweeps the operating
point so the choice of threshold is a measurement rather than a default.

The three stacked gates
-----------------------
Sweeping "confidence" is ambiguous in this system, because there are three thresholds in
series and the highest one wins:

``decode_floor``
    Passed to the detector itself. Nothing below it is decoded at all.
``GateConfig.unknown_floor`` (production default 0.18)
    Below this nothing is kept, not even as unknown.
``GateConfig.thresholds`` (per class, from the taxonomy's ``min_conf``)
    Acceptance threshold for a *named* class. Below it, a surviving box is demoted to
    unknown rather than dropped.

So lowering only ``decode_floor`` to 0.01 changes nothing: ``unknown_floor`` at 0.18
still discards everything beneath it, and a sweep over 0.01/0.03/0.05/0.10 would return
four identical rows. That looks like a bug and is really a misconfiguration.

:func:`production_params` therefore moves **all three** together, which is what a single
"conf = 0.05" operating point has to mean. The per-class shape from the taxonomy is
deliberately flattened during a sweep so the comparison is one-dimensional; recover it
afterwards with ``cli tune``, which fits a threshold per class.

Reading the numbers
-------------------
Lowering the threshold does not only add true positives. It adds boxes the gate cannot
name, and those are accepted as ``unknown_industrial_component`` — which matches no
ground-truth class and therefore scores as a false positive. That is honest behaviour
(the alternative is inventing a label), but it means ``fp_per_image`` at a low threshold
is partly unknowns rather than misclassifications. Every row reports the unknown count
separately so the two causes stay distinguishable.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Optional, Sequence

from rtsp_backend.electrical import metrics as em
from rtsp_backend.electrical import taxonomy as tax

from . import train as tr

#: The operating points to sweep. A named constant so the report and the CLI cannot
#: disagree about what was measured.
OPERATING_POINTS: tuple[float, ...] = (0.01, 0.03, 0.05, 0.10, 0.20)

#: The shipped default, evaluated alongside the sweep for reference.
PRODUCTION_DEFAULT_CONF = 0.18

#: Backend that runs a .pt checkpoint through the production gate cascade.
DEFAULT_BACKEND = "industrial_ultralytics"


def production_params(conf: float, weights: str, imgsz: int = 640,
                      device: str = "cpu",
                      extra: Optional[dict] = None) -> dict:
    """Backend params that put the whole gate cascade at one operating point.

    All three thresholds move together — see the module docstring for why moving only
    ``decode_floor`` measures nothing. ``strictness`` is pinned to 1.0 so it cannot
    silently rescale the uniform thresholds being set here.
    """
    if not 0.0 < conf < 1.0:
        raise ValueError(f"conf must be in (0, 1), got {conf}")
    params = {
        "weights": weights,
        "imgsz": imgsz,
        "device": device,
        "decode_floor": conf,
        "unknown_floor": conf,
        # Flattened on purpose: a one-number sweep needs a one-dimensional axis.
        "thresholds": {cid: conf for cid in tax.CLASS_ORDER},
        "strictness": 1.0,
    }
    if extra:
        params.update(extra)
    return params


def _evaluated_images(dataset_root: str, split: str,
                      limit: Optional[int] = None) -> list[str]:
    """The image stems the recogniser will actually be run over, in the same order.

    Must mirror :func:`train.collect_predictions` exactly — same extensions, same
    ``sorted()``, same ``[:limit]`` slice. It is the basis for both the per-image
    denominator and the ground-truth restriction below, and if it diverges from what
    the collector iterates, every rate is computed against the wrong set.
    """
    d = os.path.join(dataset_root, "images", split)
    if not os.path.isdir(d):
        return []
    files = [f for f in sorted(os.listdir(d))
             if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]
    if limit:
        files = files[:limit]
    return [os.path.splitext(f)[0] for f in files]


def evaluate_at(conf: float, weights: str, dataset_root: str,
                split: str = "val", imgsz: int = 640, device: str = "cpu",
                backend: str = DEFAULT_BACKEND,
                limit: Optional[int] = None,
                iou_thr: float = em.DEFAULT_IOU,
                extra_params: Optional[dict] = None,
                log: Optional[Callable[[str], None]] = None) -> dict:
    """Evaluate one operating point through the deployed inference path."""
    say = log or (lambda m: None)
    from rtsp_backend import electrical
    from rtsp_backend.ai import registry

    # Imported for its import-time side effect: it registers the component backends,
    # so registry.get below would raise without it. Referenced explicitly so neither a
    # linter nor a well-meaning import cleanup deletes it.
    assert electrical is not None

    gts = tr.load_ground_truth(dataset_root, split)
    if not gts:
        return {"conf": conf, "status": "skipped",
                "reason": f"no ground truth under {dataset_root}/labels/{split}"}

    evaluated = _evaluated_images(dataset_root, split, limit)
    n_images = len(evaluated)
    if limit:
        # Restrict the ground truth to the images actually inferred. Without this,
        # `--limit 40` on a 120-image split scores 40 images' predictions against 120
        # images' labels: every instance in the 80 unvisited images becomes a false
        # negative, so recall and FN/image are pinned to a wrong value that does not
        # move as the threshold sweeps. It reads as "the model plateaued" and is
        # really the harness comparing two different sets.
        keep = set(evaluated)
        gts = [g for g in gts if g.get("image_id") in keep]
        if not gts:
            return {"conf": conf, "status": "skipped",
                    "reason": (f"none of the first {limit} image(s) of "
                               f"{split} have labels; raise --limit")}

    params = production_params(conf, weights, imgsz, device, extra_params)
    try:
        inst = registry.get("components", backend)(**params)
        inst.load()
    except Exception as exc:
        return {"conf": conf, "status": "skipped",
                "reason": f"{backend} unavailable: {exc}"}

    say(f"conf={conf:.2f}: running the production path")
    preds = tr.collect_predictions(
        inst, os.path.join(dataset_root, "images", split), limit=limit)

    unknown = sum(1 for p in preds
                  if p.get("class_id") == tax.UNKNOWN_COMPONENT_ID)

    rep = em.evaluate(gts, preds, iou_thr=iou_thr)
    overall = rep.get("overall") or {}
    tp = int(overall.get("tp", 0))
    fp = int(overall.get("fp", 0))
    fn = int(overall.get("fn", 0))

    row = {
        "conf": conf,
        "status": "evaluated",
        "map_50": rep.get("map_50"),
        "map_50_95": rep.get("map_50_95"),
        "precision": overall.get("precision"),
        "recall": overall.get("recall"),
        "f1": overall.get("f1"),
        "tp": tp, "fp": fp, "fn": fn,
        "images": n_images,
        "ground_truth_instances": len(gts),
        "predictions": len(preds),
        "unknown_predictions": unknown,
        # The two headline rates the acceptance decision is made on.
        "fp_per_image": (round(fp / n_images, 4) if n_images else None),
        "fn_per_image": (round(fn / n_images, 4) if n_images else None),
        "unknown_per_image": (round(unknown / n_images, 4) if n_images else None),
        "per_class": rep.get("per_class"),
        "false_positive_analysis": rep.get("false_positive_analysis"),
        "false_negative_analysis": rep.get("false_negative_analysis"),
        "macro_precision": rep.get("macro_precision"),
        "macro_recall": rep.get("macro_recall"),
        "params": {k: v for k, v in params.items() if k != "thresholds"},
        "threshold_policy": (
            f"decode_floor, unknown_floor and every per-class threshold set to "
            f"{conf}; strictness 1.0. NMS, cross-class suppression and the geometric "
            f"plausibility gate are at their production defaults."),
    }
    if unknown:
        row["note_on_unknowns"] = (
            f"{unknown} of {len(preds)} prediction(s) were accepted as "
            f"'{tax.UNKNOWN_COMPONENT_ID}' because no class cleared its threshold. "
            f"These match no ground-truth class and so count toward fp "
            f"({fp}); they are honest low-confidence output, not misclassifications.")
    say(f"conf={conf:.2f}: mAP50={row['map_50']} P={row['precision']} "
        f"R={row['recall']} FP/img={row['fp_per_image']} "
        f"FN/img={row['fn_per_image']}")
    return row


def sweep(weights: str, dataset_root: str, split: str = "val",
          confs: Sequence[float] = OPERATING_POINTS,
          imgsz: int = 640, device: str = "cpu",
          backend: str = DEFAULT_BACKEND,
          limit: Optional[int] = None,
          include_production_default: bool = True,
          log: Optional[Callable[[str], None]] = None) -> list[dict]:
    """Evaluate every operating point on the same split, through the deployed path."""
    points = list(confs)
    if include_production_default and PRODUCTION_DEFAULT_CONF not in points:
        points.append(PRODUCTION_DEFAULT_CONF)
    rows = []
    for c in sorted(set(points)):
        row = evaluate_at(c, weights, dataset_root, split, imgsz=imgsz,
                          device=device, backend=backend, limit=limit, log=log)
        row["is_production_default"] = (c == PRODUCTION_DEFAULT_CONF)
        rows.append(row)
    return rows


def select_operating_point(rows: Sequence[dict], objective: str = "f1",
                           min_precision: Optional[float] = None,
                           min_recall: Optional[float] = None,
                           max_fp_per_image: Optional[float] = None) -> dict:
    """Pick the best operating point, and say why it was picked.

    ``objective`` is what gets maximised among the rows that satisfy the constraints.
    Constraints exist because a single scalar is the wrong way to choose a threshold for
    an inspection report: a false positive puts a device on a report that is not in the
    panel, which costs an engineer a site visit to disprove, while a false negative is
    a gap a human reviewer can still catch. If you care about that asymmetry, say so
    with ``min_precision`` or ``max_fp_per_image`` rather than trusting F1 to encode it.
    """
    scored = [r for r in rows if r.get("status") == "evaluated"
              and r.get(objective) is not None]
    if not scored:
        return {"status": "failed",
                "reason": ("no operating point produced a scored evaluation; check "
                           "the per-row reasons"),
                "constraints_applied": {}}

    constraints = {
        "min_precision": min_precision, "min_recall": min_recall,
        "max_fp_per_image": max_fp_per_image,
    }

    def ok(r: dict) -> bool:
        if min_precision is not None and (r.get("precision") or 0) < min_precision:
            return False
        if min_recall is not None and (r.get("recall") or 0) < min_recall:
            return False
        if max_fp_per_image is not None:
            fpi = r.get("fp_per_image")
            if fpi is None or fpi > max_fp_per_image:
                return False
        return True

    feasible = [r for r in scored if ok(r)]
    relaxed = False
    if not feasible:
        # Report the unconstrained best plus the fact that nothing met the bar, rather
        # than silently returning a point that violates a stated requirement.
        feasible = scored
        relaxed = True

    best = max(feasible, key=lambda r: (r.get(objective) or 0.0,
                                        r.get("map_50") or 0.0))
    out = {
        "status": "selected",
        "objective": objective,
        "conf": best["conf"],
        "map_50": best.get("map_50"),
        "precision": best.get("precision"),
        "recall": best.get("recall"),
        "f1": best.get("f1"),
        "fp_per_image": best.get("fp_per_image"),
        "fn_per_image": best.get("fn_per_image"),
        "constraints_applied": {k: v for k, v in constraints.items()
                                if v is not None},
        "constraints_met": not relaxed,
    }
    if relaxed:
        out["warning"] = (
            "No operating point satisfied the stated constraints "
            f"({', '.join(f'{k}={v}' for k, v in constraints.items() if v is not None)}"
            "). The point below is the unconstrained best and does NOT meet the "
            "requirement — treat it as a measurement of how far short the model is, "
            "not as an acceptable configuration.")
    out["rationale"] = _selection_rationale(best, scored, objective, relaxed)
    out["how_to_apply"] = (
        f"Set the components backend params to decode_floor={best['conf']}, "
        f"unknown_floor={best['conf']} (POST /api/ai/models/components/params). "
        f"For better results than a uniform threshold, run `cli tune` to fit a "
        f"per-class threshold at this operating point.")
    return out


def _selection_rationale(best: dict, scored: Sequence[dict], objective: str,
                         relaxed: bool) -> str:
    parts = [
        f"conf={best['conf']} maximises {objective} "
        f"({best.get(objective):.4f}) among {len(scored)} measured point(s), "
        f"at precision {best.get('precision'):.4f}, recall "
        f"{best.get('recall'):.4f}, {best.get('fp_per_image')} false positives and "
        f"{best.get('fn_per_image')} false negatives per image."
    ]
    lo = min(scored, key=lambda r: r["conf"])
    hi = max(scored, key=lambda r: r["conf"])
    if lo is not hi:
        parts.append(
            f"Across the sweep, recall runs {lo.get('recall'):.3f} -> "
            f"{hi.get('recall'):.3f} and precision {lo.get('precision'):.3f} -> "
            f"{hi.get('precision'):.3f} as conf goes {lo['conf']} -> {hi['conf']}.")
    unk = best.get("unknown_predictions") or 0
    if unk:
        parts.append(
            f"{unk} prediction(s) at this point are unnamed "
            f"('{tax.UNKNOWN_COMPONENT_ID}') and count as false positives, so the "
            f"precision figure understates classification accuracy on the boxes the "
            f"model was willing to name.")
    if relaxed:
        parts.append("This point does not satisfy the requested constraints.")
    parts.append(
        "Measured through the deployed inference path — same preprocessing, gates, "
        "NMS and thresholds the API uses — so it predicts served behaviour rather "
        "than trainer-reported behaviour.")
    return " ".join(parts)


def format_sweep(rows: Sequence[dict]) -> str:
    header = ("conf", "mAP50", "mAP50-95", "precision", "recall", "F1",
              "TP", "FP", "FN", "FP/img", "FN/img", "unknown")
    body = []
    for r in sorted(rows, key=lambda x: x["conf"]):
        if r.get("status") != "evaluated":
            body.append((f"{r['conf']:.2f}", "skipped", r.get("reason", "")[:28],
                         "", "", "", "", "", "", "", "", ""))
            continue
        mark = "*" if r.get("is_production_default") else ""

        def f(v, nd=4):
            return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "-"

        body.append((f"{r['conf']:.2f}{mark}", f(r.get("map_50")),
                     f(r.get("map_50_95")), f(r.get("precision")),
                     f(r.get("recall")), f(r.get("f1")),
                     str(r.get("tp", "-")), str(r.get("fp", "-")),
                     str(r.get("fn", "-")), f(r.get("fp_per_image"), 3),
                     f(r.get("fn_per_image"), 3),
                     str(r.get("unknown_predictions", "-"))))
    widths = [max(len(str(row[i])) for row in [header] + body)
              for i in range(len(header))]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    out = [line, "-" * len(line)]
    out += ["  ".join(str(c).ljust(w) for c, w in zip(row, widths))
            for row in body]
    out.append("")
    out.append("* = shipped production default. Every row is measured through the "
               "deployed inference path.")
    return "\n".join(out)


def per_class_table(row: dict, top: Optional[int] = None) -> str:
    """Per-class AP / precision / recall at one operating point."""
    pc = row.get("per_class") or {}
    if not pc:
        return "no per-class data"
    header = ("class", "AP", "precision", "recall", "F1", "support", "FN")
    fn_by = ((row.get("false_negative_analysis") or {}).get("by_class") or {})
    items = sorted(pc.items(), key=lambda kv: -(kv[1].get("support") or 0))
    if top:
        items = items[:top]
    body = []
    for cid, v in items:
        name = v.get("name") or cid
        body.append((name[:26],
                     f"{v.get('ap', 0):.4f}", f"{v.get('precision', 0):.4f}",
                     f"{v.get('recall', 0):.4f}", f"{v.get('f1', 0):.4f}",
                     str(v.get("support", 0)), str(fn_by.get(name, 0))))
    widths = [max(len(str(r[i])) for r in [header] + body)
              for i in range(len(header))]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    return "\n".join([line, "-" * len(line)]
                     + ["  ".join(str(c).ljust(w) for c, w in zip(r, widths))
                        for r in body])


def acceptance_report(weights: str, dataset_root: str, split: str = "val",
                      confs: Sequence[float] = OPERATING_POINTS,
                      imgsz: int = 640, device: str = "cpu",
                      backend: str = DEFAULT_BACKEND,
                      limit: Optional[int] = None,
                      objective: str = "f1",
                      min_precision: Optional[float] = None,
                      min_recall: Optional[float] = None,
                      max_fp_per_image: Optional[float] = None,
                      target_map50: Optional[float] = None,
                      log: Optional[Callable[[str], None]] = None) -> dict:
    """The full production-path acceptance report: sweep, best point, verdict."""
    rows = sweep(weights, dataset_root, split, confs=confs, imgsz=imgsz,
                 device=device, backend=backend, limit=limit, log=log)
    best = select_operating_point(rows, objective=objective,
                                  min_precision=min_precision,
                                  min_recall=min_recall,
                                  max_fp_per_image=max_fp_per_image)
    report = {
        "weights": weights,
        "dataset_root": dataset_root,
        "split": split,
        "backend": backend,
        "imgsz": imgsz,
        "evaluation_path": (
            "The registered production backend, via recognize() -> accepted. Same "
            "preprocessing, per-class gates, NMS, cross-class suppression and "
            "plausibility checks as POST /api/panel/analyze. This is not the "
            "trainer's validator and the numbers will be lower than results.csv."),
        "sweep": rows,
        "best_operating_point": best,
    }
    if target_map50 is not None:
        report["acceptance"] = _verdict(rows, best, target_map50)
    return report


def _verdict(rows: Sequence[dict], best: dict, target: float) -> dict:
    got = best.get("map_50")
    passed = bool(got is not None and got >= target)
    v = {"target_map_50": target, "achieved_map_50": got, "passed": passed}
    if passed:
        v["statement"] = (
            f"Production-path mAP@50 {got:.4f} meets the {target:.2f} target at "
            f"conf={best.get('conf')}.")
    else:
        best_any = max((r.get("map_50") or 0.0) for r in rows) if rows else 0.0
        v["statement"] = (
            f"Production-path mAP@50 {got if got is not None else 0.0:.4f} does NOT "
            f"meet the {target:.2f} target (best across the sweep: {best_any:.4f}). "
            f"This is the number that describes served behaviour, so the model is not "
            f"acceptable for production regardless of what the trainer reported. "
            f"Lowering the threshold further trades the shortfall for false "
            f"positives rather than fixing it — the gap closes with more or better "
            f"training data.")
    return v


def write_report(report: dict, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    return path


def write_sweep_csv(rows: Sequence[dict], path: str) -> str:
    """The sweep as CSV, for pasting into an acceptance record."""
    import csv

    cols = ["conf", "status", "map_50", "map_50_95", "precision", "recall", "f1",
            "tp", "fp", "fn", "images", "fp_per_image", "fn_per_image",
            "unknown_predictions", "is_production_default"]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: x["conf"]):
            w.writerow(r)
    return path


__all__ = [
    "OPERATING_POINTS", "PRODUCTION_DEFAULT_CONF", "DEFAULT_BACKEND",
    "production_params", "evaluate_at", "sweep", "select_operating_point",
    "format_sweep", "per_class_table", "acceptance_report", "write_report",
    "write_sweep_csv",
]
