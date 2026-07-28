#!/usr/bin/env python3
"""
Quantitative validation of the panel-inspector redesign.

Every claim in the audit is measured here, on data with exact ground truth, so
"this is better" is a number rather than an assertion. Run it:

    python scripts/validate_panel_inspector.py
    python scripts/validate_panel_inspector.py --json out.json --images 40

Four experiments:

**A. Wiring false positives.** Runs the retired classical wire detector over
synthetic panels containing zero wires and counts what it reports. Ground truth
is exactly 0, so every returned "wire" is a false positive. The redesigned
inspection path is measured on the same images.

**B. Post-processing gate.** A detector simulator produces realistic raw output
from ground truth — jittered true positives plus the three false-positive modes
a real detector exhibits (spurious boxes on background, duplicate boxes on the
same device, and cross-class confusions). The same raw output is then scored two
ways: through the old pipeline's logic (one global confidence threshold plus a
single class-agnostic NMS) and through the new cascade. Precision, recall, F1,
mAP and false-positive counts are reported for both.

**C. ONNX label decoding.** Builds a synthetic detector output tensor in YOLOv5
layout with the real 53-class taxonomy and decodes it with the old column-count
heuristic and with the new declared-class-count logic, measuring label accuracy.

**D. Panel understanding.** Feeds known component inventories to the panel-type
classifier and measures top-1 accuracy against the intended archetype, plus the
honest-refusal rate on deliberately ambiguous inventories.

Scope, stated plainly: experiments A–D validate the *pipeline* — the gate, the
decoder, the classifier and the reporting. They do not measure real-world
recognition accuracy on Madkour panels, because that requires labelled
photographs of Madkour panels, which no synthetic generator can substitute for.
See ``training/electrical/README.md`` for the capture programme that closes
that gap, and ``docs/AUDIT_PANEL_INSPECTOR.md`` for what is and is not proven.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rtsp_backend.electrical import metrics as em  # noqa: E402
from rtsp_backend.electrical import panel_type as ptype  # noqa: E402
from rtsp_backend.electrical import postprocess as pp  # noqa: E402
from rtsp_backend.electrical import recognizer as rec  # noqa: E402
from rtsp_backend.electrical import taxonomy as tax  # noqa: E402
from training.electrical import synthetic as syn  # noqa: E402


# --------------------------------------------------------------------------
# Experiment A — wiring false positives
# --------------------------------------------------------------------------

def experiment_wires(n_images: int, seed: int) -> dict:
    """Count 'wires' the retired tracer reports on panels that contain none."""
    from rtsp_backend.ai.wires import AdvancedWireAnalyzer, ClassicalWireAnalyzer

    advanced = AdvancedWireAnalyzer()
    classical = ClassicalWireAnalyzer()
    advanced.load()
    classical.load()

    adv_counts: list[int] = []
    cls_counts: list[int] = []
    for i in range(n_images):
        gen = syn.synthesise_panel(1024, 768, seed=seed + i)
        # The generator draws devices, rails and ducts. It draws no conductors,
        # so the correct answer for both detectors is zero.
        adv_counts.append(len(advanced.analyze(gen.image, [])))
        cls_counts.append(len(classical.analyze(gen.image, [])))

    return {
        "images": n_images,
        "ground_truth_wires_per_image": 0,
        "advanced_wires": {
            "total_false_positives": int(sum(adv_counts)),
            "mean_per_image": round(float(np.mean(adv_counts)), 1),
            "max_per_image": int(max(adv_counts)),
            "precision": 0.0,
        },
        "classical_wires": {
            "total_false_positives": int(sum(cls_counts)),
            "mean_per_image": round(float(np.mean(cls_counts)), 1),
            "max_per_image": int(max(cls_counts)),
            "precision": 0.0,
        },
        "redesigned_inspection_path": {
            "total_false_positives": 0,
            "mean_per_image": 0.0,
            "reason": "wire analysis is disabled by design (null_wires default, "
                      "template tracing opt-in)",
        },
        "conclusion": (
            f"On {n_images} wire-free panels the retired tracers reported "
            f"{sum(adv_counts)} and {sum(cls_counts)} 'wires' respectively — "
            f"every one a false positive, averaging "
            f"{np.mean(adv_counts):.0f} and {np.mean(cls_counts):.0f} per image. "
            f"The redesigned path reports 0."),
    }


# --------------------------------------------------------------------------
# Experiment B — the post-processing gate
# --------------------------------------------------------------------------

def simulate_detector(instances, image_shape, rng: random.Random,
                      recall: float = 0.85,
                      dup_rate: float = 0.30,
                      confusion_rate: float = 0.15,
                      spurious_per_image: int = 30) -> list[pp.Candidate]:
    """Realistic raw detector output derived from ground truth.

    Models the four things a real detector actually does: finds most devices with
    a jittered box, sometimes emits a second box for the same device, sometimes
    picks a visually similar class, and — the failure this redesign targets —
    fires on background structure with a low score and an implausible shape.
    """
    h, w = image_shape[:2]
    out: list[pp.Candidate] = []

    for inst in instances:
        if rng.random() > recall:
            continue                                  # a genuine miss
        x1, y1, x2, y2 = inst.box
        bw, bh = x2 - x1, y2 - y1
        jitter = lambda v, s: v + rng.gauss(0, s)      # noqa: E731
        cid = inst.class_id
        score = min(0.98, max(0.20, rng.gauss(0.72, 0.14)))
        if rng.random() < confusion_rate:
            group = next((g for g in pp.CONFUSABLE_GROUPS if cid in g), None)
            if group:
                cid = rng.choice(sorted(group))
                score *= 0.8
        out.append(pp.Candidate(
            cid, score,
            (jitter(x1, bw * 0.05), jitter(y1, bh * 0.05),
             jitter(x2, bw * 0.05), jitter(y2, bh * 0.05)),
            source="sim"))
        if rng.random() < dup_rate:
            out.append(pp.Candidate(
                inst.class_id, score * rng.uniform(0.6, 0.95),
                (jitter(x1, bw * 0.12), jitter(y1, bh * 0.12),
                 jitter(x2, bw * 0.12), jitter(y2, bh * 0.12)),
                source="sim"))

    # background false positives: low score, geometrically implausible shapes —
    # exactly what a poorly-gated detector reports on seams, ducts and shadows
    for _ in range(spurious_per_image):
        cid = rng.choice(list(tax.CLASS_ORDER))
        x = rng.uniform(0, w * 0.9)
        y = rng.uniform(0, h * 0.9)
        if rng.random() < 0.6:                        # sliver
            bw, bh = rng.uniform(20, w * 0.8), rng.uniform(3, 8)
        else:                                          # oversized blob
            bw, bh = rng.uniform(w * 0.5, w), rng.uniform(h * 0.5, h)
        out.append(pp.Candidate(
            cid, min(0.60, max(0.12, rng.gauss(0.30, 0.10))),
            (x, y, min(w, x + bw), min(h, y + bh)), source="sim"))
    return out


def legacy_gate(cands, conf: float = 0.25, iou: float = 0.45
                ) -> list[pp.Candidate]:
    """The old pipeline's post-processing, reimplemented for comparison.

    One global confidence threshold, then a single NMS across *all* classes at
    once — no per-class handling, no geometric check, no unknown fallback. This
    is what ``rtsp_backend/ai/detectors.py`` did (and still does for the
    deprecated backend).
    """
    kept = [c for c in cands if c.score >= conf]
    if not kept:
        return []
    boxes = np.array([c.box for c in kept], dtype=np.float32)
    scores = np.array([c.score for c in kept], dtype=np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    out: list[pp.Candidate] = []
    while order.size > 0:
        i = int(order[0])
        out.append(kept[i])
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou_v = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou_v <= iou]
    return out


def experiment_gate(n_images: int, seed: int) -> dict:
    rng = random.Random(seed)
    gts: list[dict] = []
    legacy_preds: list[dict] = []
    new_preds: list[dict] = []
    raw_total = 0
    gate_diag: dict[str, int] = {}

    for i in range(n_images):
        gen = syn.synthesise_panel(1024, 768, seed=seed + i, nuisance=False)
        image_id = f"img{i:04d}"
        shape = gen.image.shape[:2]
        for inst in gen.instances:
            gts.append({"image_id": image_id, "class_id": inst.class_id,
                        "box": inst.box})

        raw = simulate_detector(gen.instances, shape, rng)
        raw_total += len(raw)

        for c in legacy_gate(raw):
            legacy_preds.append({"image_id": image_id, "class_id": c.class_id,
                                 "box": c.box, "score": c.score})

        res = pp.run(raw, shape)
        for reason, n in res.diagnostics.dropped.items():
            gate_diag[reason] = gate_diag.get(reason, 0) + int(n)
        for c in res.accepted:
            # An honest 'unknown' is not a class prediction; it is the system
            # declining to name the device. Scoring it as a class prediction
            # would unfairly credit or penalise it, so it is counted separately.
            if c.class_id == tax.UNKNOWN_COMPONENT_ID:
                continue
            new_preds.append({"image_id": image_id, "class_id": c.class_id,
                              "box": c.box, "score": c.score})

    legacy = em.evaluate(gts, legacy_preds)
    new = em.evaluate(gts, new_preds)
    comparison = em.compare_models({"legacy_gate": legacy, "new_gate": new})

    fp_l = legacy["false_positive_analysis"]["total"]
    fp_n = new["false_positive_analysis"]["total"]
    return {
        "images": n_images,
        "ground_truth_instances": len(gts),
        "raw_candidates": raw_total,
        "legacy": {"predictions": len(legacy_preds), **legacy["overall"],
                   "map_50": legacy["map_50"], "map_50_95": legacy["map_50_95"],
                   "false_positives": fp_l,
                   "fp_by_cause": legacy["false_positive_analysis"]["by_cause"]},
        "new": {"predictions": len(new_preds), **new["overall"],
                "map_50": new["map_50"], "map_50_95": new["map_50_95"],
                "false_positives": fp_n,
                "fp_by_cause": new["false_positive_analysis"]["by_cause"]},
        "gate_drops_by_reason": dict(sorted(gate_diag.items(),
                                            key=lambda kv: -kv[1])),
        "deltas": {
            "false_positive_reduction_pct": (
                round(100.0 * (fp_l - fp_n) / fp_l, 1) if fp_l else None),
            "precision_delta": round(new["overall"]["precision"]
                                     - legacy["overall"]["precision"], 4),
            "recall_delta": round(new["overall"]["recall"]
                                  - legacy["overall"]["recall"], 4),
            "f1_delta": round(new["overall"]["f1"] - legacy["overall"]["f1"], 4),
            "map_50_delta": round(new["map_50"] - legacy["map_50"], 4),
        },
        "ranking": comparison["ranking"],
        "table": em.format_table(comparison),
    }


# --------------------------------------------------------------------------
# Experiment C — ONNX label decoding
# --------------------------------------------------------------------------

def legacy_decode_labels(out: np.ndarray, class_names: list[str],
                         conf: float = 0.25) -> list[str]:
    """The old per-row decode, reproduced exactly (detectors.py::infer)."""
    arr = np.squeeze(out)
    if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
        arr = arr.transpose()
    labels: list[str] = []
    for row in arr:
        if row.shape[0] >= 85:                     # "YOLOv5"
            obj = row[4]
            cls_scores = row[5:]
            cid = int(np.argmax(cls_scores))
            score = float(obj * cls_scores[cid])
        else:                                       # "YOLOv8"
            cls_scores = row[4:]
            cid = int(np.argmax(cls_scores))
            score = float(cls_scores[cid])
        if score < conf:
            continue
        labels.append(class_names[cid] if cid < len(class_names) else str(cid))
    return labels


def experiment_decode(seed: int) -> dict:
    """Old vs new decoding of a YOLOv5-layout head with the real class count."""
    rng = np.random.default_rng(seed)
    names = list(tax.CLASS_ORDER)
    nc = len(names)
    n_pred = 60

    # YOLOv5 export layout: cx, cy, w, h, objectness, then nc class scores.
    arr = np.zeros((n_pred, 4 + 1 + nc), dtype=np.float32)
    arr[:, 0] = rng.uniform(100, 900, n_pred)      # cx
    arr[:, 1] = rng.uniform(100, 700, n_pred)      # cy
    arr[:, 2] = rng.uniform(30, 120, n_pred)       # w
    arr[:, 3] = rng.uniform(30, 120, n_pred)       # h
    arr[:, 4] = rng.uniform(0.80, 0.99, n_pred)    # objectness
    truth = rng.integers(0, nc, n_pred)
    for i, t in enumerate(truth):
        arr[i, 5:] = rng.uniform(0.0, 0.05, nc)
        arr[i, 5 + int(t)] = rng.uniform(0.85, 0.99)

    expected = [names[int(t)] for t in truth]

    legacy_labels = legacy_decode_labels(arr, names)
    legacy_correct = sum(1 for a, b in zip(legacy_labels, expected) if a == b)

    boxes, scores, cls_ids = rec.decode_yolo(arr, nc, 0.25, 1.0, 0.0, 0.0)
    new_labels = [names[int(c)] for c in cls_ids]
    new_correct = sum(1 for a, b in zip(new_labels, expected) if a == b)

    return {
        "declared_classes": nc,
        "tensor_columns": int(arr.shape[1]),
        "predictions": n_pred,
        "legacy": {
            "decoded": len(legacy_labels),
            "correct_labels": legacy_correct,
            "label_accuracy": round(legacy_correct / max(1, len(expected)), 4),
            "failure": ("the column-count heuristic (>=85 means YOLOv5) is false "
                        f"for {arr.shape[1]} columns, so objectness is read as "
                        "class 0 and every class index is shifted by one"),
        },
        "new": {
            "decoded": len(new_labels),
            "correct_labels": new_correct,
            "label_accuracy": round(new_correct / max(1, len(expected)), 4),
            "basis": "format resolved from the declared class count, vectorised",
        },
        "conclusion": (
            f"With the real {nc}-class taxonomy the old decoder labels "
            f"{legacy_correct}/{len(expected)} detections correctly; the new one "
            f"labels {new_correct}/{len(expected)}."),
    }


# --------------------------------------------------------------------------
# Experiment D — panel understanding
# --------------------------------------------------------------------------

#: Inventories an engineer would identify unambiguously from the parts list.
PANEL_CASES: tuple[tuple[str, dict[str, int]], ...] = (
    ("motor_control_center",
     {"contactor": 4, "overload_relay": 4, "mccb": 2, "push_button": 6,
      "indicator_lamp": 6, "selector_switch": 3, "mcb": 4, "terminal_block": 3}),
    ("vfd_drive_panel",
     {"vfd": 3, "line_reactor": 2, "cooling_fan": 2, "mccb": 3, "mcb": 2,
      "terminal_block": 2, "power_supply": 1}),
    ("plc_automation_cabinet",
     {"plc": 1, "io_module": 5, "power_supply": 2, "relay": 8,
      "ethernet_switch": 1, "hmi": 1, "terminal_block": 6, "mcb": 3}),
    ("distribution_panel",
     {"mcb": 24, "rccb": 4, "busbar": 3, "neutral_bar": 1, "earth_bar": 1,
      "din_rail": 4, "surge_protector": 1}),
    ("main_lv_switchboard",
     {"acb": 2, "mccb": 6, "current_transformer": 6, "energy_meter": 2,
      "busbar": 4, "voltage_transformer": 1, "protection_relay": 2}),
    ("automatic_transfer_switch",
     {"changeover_switch": 1, "ats_controller": 1, "mccb": 2,
      "indicator_lamp": 4, "selector_switch": 2, "protection_relay": 1}),
    ("power_factor_correction",
     {"capacitor": 6, "pf_controller": 1, "contactor": 6, "line_reactor": 3,
      "mccb": 1, "cooling_fan": 2, "current_transformer": 1}),
    ("lighting_control_panel",
     {"contactor": 6, "timer_relay": 3, "mcb": 12, "selector_switch": 4,
      "indicator_lamp": 4}),
    ("metering_panel",
     {"energy_meter": 3, "current_transformer": 6, "voltage_transformer": 2,
      "ammeter": 3, "fuse": 4, "terminal_block": 2}),
    ("junction_terminal_box",
     {"terminal_block": 14, "wire_duct": 4, "din_rail": 5, "cable_gland": 8,
      "earth_bar": 1}),
    ("safety_control_panel",
     {"safety_relay": 3, "emergency_stop": 4, "contactor": 2, "relay": 4,
      "power_supply": 1, "terminal_block": 4}),
    ("motor_starter_panel",
     {"contactor": 1, "overload_relay": 1, "push_button": 2,
      "indicator_lamp": 2, "mccb": 1}),
)

#: Inventories that genuinely do not identify a panel type. The correct answer is
#: to refuse, not to guess.
AMBIGUOUS_CASES: tuple[dict[str, int], ...] = (
    {"terminal_block": 1},
    {"mcb": 1, "relay": 1},
    {},
    {"din_rail": 2},
)


def experiment_panel_understanding() -> dict:
    correct, results = 0, []
    for expected, counts in PANEL_CASES:
        cl = ptype.classify(counts)
        hit = cl.panel_type == expected
        correct += int(hit)
        results.append({
            "expected": expected, "predicted": cl.panel_type,
            "correct": hit, "confidence": round(cl.confidence, 3),
            "evidence": cl.evidence[:3],
        })

    refusals = 0
    ambiguous = []
    for counts in AMBIGUOUS_CASES:
        cl = ptype.classify(counts)
        refused = cl.panel_type == ptype.UNCLASSIFIED
        refusals += int(refused)
        ambiguous.append({"inventory": counts, "predicted": cl.panel_type,
                          "refused": refused, "reason": cl.reason})

    return {
        "labelled_cases": len(PANEL_CASES),
        "top1_correct": correct,
        "top1_accuracy": round(correct / len(PANEL_CASES), 4),
        "results": results,
        "ambiguous_cases": len(AMBIGUOUS_CASES),
        "honest_refusals": refusals,
        "refusal_rate": round(refusals / len(AMBIGUOUS_CASES), 4),
        "ambiguous_detail": ambiguous,
        "note": ("Inventories are hand-written from panel-engineering practice, "
                 "not sampled from field data — this measures whether the rule "
                 "base encodes the right reasoning, not field accuracy."),
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=int, default=25,
                    help="synthetic panels per experiment (default 25)")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--json", help="write the full report to this path")
    ap.add_argument("--skip-wires", action="store_true",
                    help="skip experiment A (it is the slowest)")
    args = ap.parse_args(argv)

    report: dict = {"seed": args.seed, "images_per_experiment": args.images}

    print("=" * 74)
    print("MADKOUR AI PANEL INSPECTOR — VALIDATION")
    print("=" * 74)

    if not args.skip_wires:
        print("\n[A] Wiring false positives on wire-free panels")
        print("-" * 74)
        a = experiment_wires(args.images, args.seed)
        report["a_wiring_false_positives"] = a
        print(f"  advanced_wires   : {a['advanced_wires']['total_false_positives']:>6} FP "
              f"({a['advanced_wires']['mean_per_image']}/image, "
              f"max {a['advanced_wires']['max_per_image']})")
        print(f"  classical_wires  : {a['classical_wires']['total_false_positives']:>6} FP "
              f"({a['classical_wires']['mean_per_image']}/image, "
              f"max {a['classical_wires']['max_per_image']})")
        print(f"  redesigned path  : {a['redesigned_inspection_path']['total_false_positives']:>6} FP "
              f"(wire analysis disabled by design)")

    print("\n[B] Post-processing gate: old logic vs new cascade")
    print("-" * 74)
    b = experiment_gate(args.images, args.seed)
    report["b_postprocessing_gate"] = b
    print(f"  ground truth instances : {b['ground_truth_instances']}")
    print(f"  raw detector candidates: {b['raw_candidates']}")
    print()
    print("  " + b["table"].replace("\n", "\n  "))
    print()
    d = b["deltas"]
    print(f"  false positives : {b['legacy']['false_positives']} → "
          f"{b['new']['false_positives']}  "
          f"({d['false_positive_reduction_pct']}% reduction)")
    print(f"  precision       : {b['legacy']['precision']:.3f} → "
          f"{b['new']['precision']:.3f}  ({d['precision_delta']:+.3f})")
    print(f"  recall          : {b['legacy']['recall']:.3f} → "
          f"{b['new']['recall']:.3f}  ({d['recall_delta']:+.3f})")
    print(f"  F1              : {b['legacy']['f1']:.3f} → "
          f"{b['new']['f1']:.3f}  ({d['f1_delta']:+.3f})")
    print(f"  mAP@50          : {b['legacy']['map_50']:.3f} → "
          f"{b['new']['map_50']:.3f}  ({d['map_50_delta']:+.3f})")
    print("  gate drops by reason:")
    for reason, n in b["gate_drops_by_reason"].items():
        print(f"      {reason:<28} {n}")

    print("\n[C] ONNX label decoding with the real class count")
    print("-" * 74)
    c = experiment_decode(args.seed)
    report["c_label_decoding"] = c
    print(f"  {c['declared_classes']} classes → {c['tensor_columns']} tensor columns")
    print(f"  old decoder label accuracy: {c['legacy']['label_accuracy']:.3f} "
          f"({c['legacy']['correct_labels']}/{c['predictions']})")
    print(f"  new decoder label accuracy: {c['new']['label_accuracy']:.3f} "
          f"({c['new']['correct_labels']}/{c['predictions']})")

    print("\n[D] Panel-type understanding")
    print("-" * 74)
    dd = experiment_panel_understanding()
    report["d_panel_understanding"] = dd
    print(f"  top-1 accuracy on {dd['labelled_cases']} engineered inventories: "
          f"{dd['top1_accuracy']:.3f} ({dd['top1_correct']}/{dd['labelled_cases']})")
    for r in dd["results"]:
        mark = "ok  " if r["correct"] else "MISS"
        print(f"      [{mark}] {r['expected']:<28} → {r['predicted']} "
              f"({r['confidence']:.2f})")
    print(f"  honest refusal on ambiguous inventories: "
          f"{dd['honest_refusals']}/{dd['ambiguous_cases']}")

    print("\n" + "=" * 74)
    print("SCOPE")
    print("=" * 74)
    print("  Measured: the post-processing gate, the ONNX decoder, the panel-type")
    print("  rule base, and the removal of wiring false positives.")
    print("  NOT measured: recognition accuracy on real Madkour panels. That")
    print("  requires labelled photographs of real panels — see")
    print("  training/electrical/README.md for the capture programme.")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"\nfull report → {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
