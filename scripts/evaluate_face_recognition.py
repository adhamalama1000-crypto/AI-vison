"""
Run a real end-to-end evaluation of the production face pipeline.

Drives the genuine InsightFace (SCRFD + ArcFace) embedder over a real LFW subset
and reports Accuracy, Precision, Recall, F1, ROC/AUC, EER, a confusion matrix,
and — most importantly for this system — FAR (stranger accepted) and FRR
(employee missed). Nothing is mocked: every number comes from the model running
on real photographs.

    python -m scripts.evaluate_face_recognition [--threshold 0.65] [--policy average]

Exits non-zero (and prints a clear message) if the real model or the dataset is
unavailable, so a CI run can tell "not validated" from "validated".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import numpy as np


def build_embedder(model_pack: str, det_size: int):
    from rtsp_backend.ai.embedders import InsightFaceEmbedder

    models_dir = os.environ.get("RTSP_MODELS_DIR", "models")
    emb = InsightFaceEmbedder(models_dir=models_dir, model_pack=model_pack,
                              det_size=det_size, det_thresh=0.5)
    emb.load()  # raises with a clear reason if weights are unavailable
    return emb


def make_embed_fn(embedder):
    def embed_fn(img) -> Optional[np.ndarray]:
        faces = embedder.detect_and_embed(img)
        if not faces:
            return None
        # largest detected face
        box, _score, vec = max(
            faces, key=lambda f: (f[0].x2 - f[0].x1) * (f[0].y2 - f[0].y1))
        return vec
    return embed_fn


def split_gallery_probe(embs_by_id, known_ids, gallery_n=15):
    """Split KNOWN identities into gallery (enrol) and probe (query)."""
    gallery, probes = {}, []
    for ident, vecs in embs_by_id.items():
        if ident in known_ids:
            gallery[ident] = vecs[:gallery_n]
            for v in vecs[gallery_n:]:
                probes.append((v, ident))
        else:  # stranger -> impostor probe (never enrolled)
            for v in vecs:
                probes.append((v, None))
    return gallery, probes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.65)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--policy", choices=["average", "nearest"], default="average")
    ap.add_argument("--model-pack", default="buffalo_l")
    ap.add_argument("--det-size", type=int, default=640)
    ap.add_argument("--gallery-n", type=int, default=15)
    ap.add_argument("--out", default=None, help="write metrics JSON here")
    args = ap.parse_args()

    from rtsp_backend.ai import evaluation as ev
    from scripts.fetch_lfw_eval import KNOWN, load_dataset

    print("Loading real LFW evaluation subset …")
    dataset = load_dataset(verbose=True)
    if not dataset:
        print("ERROR: evaluation dataset unavailable (download blocked?).",
              file=sys.stderr)
        return 2

    print(f"\nLoading real model pack '{args.model_pack}' (SCRFD + ArcFace) …")
    try:
        embedder = build_embedder(args.model_pack, args.det_size)
    except Exception as exc:
        print(f"ERROR: real InsightFace model unavailable: {exc}", file=sys.stderr)
        return 3
    print(f"  backend={embedder.backend_id} dim={embedder.dim} "
          f"ready={embedder.ready}")

    embed_fn = make_embed_fn(embedder)
    t0 = time.time()
    embs_by_id, stats = ev.embed_dataset(embed_fn, dataset)
    print(f"  embedded {stats['faces_detected']}/{stats['images']} images "
          f"(detection rate {stats['detection_rate']:.1%}) in {time.time()-t0:.1f}s")

    known_ids = [k for k in KNOWN if k in embs_by_id]
    gallery, probes = split_gallery_probe(embs_by_id, set(known_ids),
                                          gallery_n=args.gallery_n)

    # verification distributions over the KNOWN identities
    known_embs = {k: v for k, v in embs_by_id.items() if k in known_ids}
    genuine, impostor = ev.genuine_impostor_scores(known_embs)
    roc = ev.roc_eer(genuine, impostor)
    far_frr = ev.far_frr_at(genuine, impostor, args.threshold)

    # open-set identification (the deployed decision)
    report = ev.open_set_report(gallery, probes, args.threshold, args.margin,
                                policy=args.policy)

    summary = {
        "model_pack": args.model_pack,
        "det_size": args.det_size,
        "threshold": args.threshold,
        "margin": args.margin,
        "policy": args.policy,
        "dataset": {
            "identities": len(dataset),
            "known": len(known_ids),
            "strangers": len(dataset) - len(known_ids),
            "gallery_per_id": args.gallery_n,
            **stats,
        },
        "verification": {
            "genuine_pairs": int(genuine.size),
            "impostor_pairs": int(impostor.size),
            "genuine_mean": round(float(genuine.mean()), 4) if genuine.size else None,
            "impostor_mean": round(float(impostor.mean()), 4) if impostor.size else None,
            "auc": roc["auc"],
            "eer": roc["eer"],
            "eer_threshold": roc["eer_threshold"],
            "far_at_threshold": far_frr["far"],
            "frr_at_threshold": far_frr["frr"],
        },
        "identification": {k: v for k, v in report.items()
                           if k not in ("details", "confusion_matrix")},
        "confusion_matrix": {"labels": report["labels"],
                             "matrix": report["confusion_matrix"]},
    }

    print("\n================  FACE RECOGNITION EVALUATION  ================")
    print(json.dumps(summary, indent=2))
    idr = summary["identification"]
    ver = summary["verification"]
    print("\n---- headline ----")
    print(f"  genuine mean sim : {ver['genuine_mean']}   impostor mean sim: {ver['impostor_mean']}")
    print(f"  ROC AUC          : {ver['auc']}")
    print(f"  EER              : {ver['eer']} (thr {ver['eer_threshold']})")
    print(f"  Accuracy         : {idr['accuracy']}")
    print(f"  Precision/Recall : {idr['precision_macro']} / {idr['recall_macro']}")
    print(f"  F1               : {idr['f1_macro']}")
    print(f"  FAR (stranger->employee): {idr['far']}")
    print(f"  FRR (employee missed)   : {idr['frr']}")

    out = args.out or os.path.join(
        os.environ.get("AIVISION_EVAL_DIR", os.path.expanduser("~/.cache/aivision_eval")),
        "face_eval_metrics.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"summary": summary, "roc": roc, "report": report}, fh, indent=2)
    print(f"\nFull metrics written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
