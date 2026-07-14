"""
Face-recognition evaluation engine.

Computes the metrics a production attendance system must be judged on, from real
embeddings — never fabricated numbers:

* **Verification** (are these two faces the same person?): genuine vs impostor
  cosine-similarity distributions, ROC curve, AUC, and the Equal Error Rate
  (EER) with its operating threshold.
* **Open-set identification** (who is this, or is it a stranger?): at a fixed
  acceptance threshold + identity margin, every probe is classified as a
  specific enrolled employee or ``Unknown``. Yields the confusion matrix,
  Accuracy / Precision / Recall / F1, and the two rates this system cares about
  most: **FAR** (False Acceptance Rate — a stranger accepted as an employee) and
  **FRR** (False Rejection Rate — a genuine employee missed or misidentified).

The engine is backend-agnostic: it takes an ``embed`` callable and a dataset of
``{identity: [images]}``. The real InsightFace pipeline is driven through it by
``scripts/evaluate_face_recognition.py`` and the evaluation tests.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


# ----------------------------------------------------------------------------
# similarity distributions + verification metrics
# ----------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n else v


def genuine_impostor_scores(
    identity_embeddings: dict[str, list[np.ndarray]]
) -> tuple[np.ndarray, np.ndarray]:
    """All pairwise cosine similarities split into genuine (same identity) and
    impostor (different identity) sets."""
    labels: list[str] = []
    vecs: list[np.ndarray] = []
    for label, embs in identity_embeddings.items():
        for e in embs:
            labels.append(label)
            vecs.append(_unit(np.asarray(e, dtype=np.float32)))
    if len(vecs) < 2:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    mat = np.vstack(vecs)
    sims = mat @ mat.T
    lab = np.asarray(labels)
    n = len(vecs)
    iu, ju = np.triu_indices(n, k=1)  # unique unordered pairs
    same = lab[iu] == lab[ju]
    pair_sims = sims[iu, ju]
    return (pair_sims[same].astype(np.float32),
            pair_sims[~same].astype(np.float32))


def roc_eer(genuine: np.ndarray, impostor: np.ndarray) -> dict:
    """ROC curve, AUC and Equal Error Rate from genuine/impostor scores."""
    if genuine.size == 0 or impostor.size == 0:
        return {"auc": None, "eer": None, "eer_threshold": None,
                "fpr": [], "tpr": [], "thresholds": []}
    from sklearn.metrics import roc_curve, auc as _auc

    y = np.concatenate([np.ones_like(genuine), np.zeros_like(impostor)])
    scores = np.concatenate([genuine, impostor])
    fpr, tpr, thr = roc_curve(y, scores)
    roc_auc = float(_auc(fpr, tpr))
    fnr = 1.0 - tpr
    # EER: where FPR and FNR cross
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return {
        "auc": round(roc_auc, 4),
        "eer": round(eer, 4),
        "eer_threshold": round(float(thr[idx]), 4),
        "fpr": [round(float(x), 4) for x in fpr],
        "tpr": [round(float(x), 4) for x in tpr],
        "thresholds": [round(float(x), 4) for x in thr],
    }


def far_frr_at(genuine: np.ndarray, impostor: np.ndarray,
               threshold: float) -> dict:
    """FAR/FRR for a plain verification decision at ``threshold``."""
    far = float((impostor >= threshold).mean()) if impostor.size else None
    frr = float((genuine < threshold).mean()) if genuine.size else None
    return {"threshold": round(float(threshold), 4),
            "far": None if far is None else round(far, 4),
            "frr": None if frr is None else round(frr, 4)}


# ----------------------------------------------------------------------------
# open-set identification (the deployed decision)
# ----------------------------------------------------------------------------

def build_gallery(gallery_embeddings: dict[str, list[np.ndarray]],
                  policy: str = "average") -> dict:
    """Build the matcher gallery from enrolled identities.

    ``average`` -> one unit centroid per identity. ``nearest`` -> keep every
    sample. Returns a structure consumed by :func:`classify`.
    """
    ids = sorted(gallery_embeddings)
    if policy == "nearest":
        rows, row_id = [], []
        for i, ident in enumerate(ids):
            for e in gallery_embeddings[ident]:
                rows.append(_unit(np.asarray(e, dtype=np.float32)))
                row_id.append(i)
        mat = np.vstack(rows).astype(np.float32) if rows else None
        return {"policy": "nearest", "ids": ids, "matrix": mat,
                "row_id": np.asarray(row_id, dtype=np.int64)}
    centroids = []
    for ident in ids:
        embs = [_unit(np.asarray(e, dtype=np.float32)) for e in gallery_embeddings[ident]]
        mean = np.mean(embs, axis=0)
        centroids.append(_unit(mean))
    mat = np.vstack(centroids).astype(np.float32) if centroids else None
    return {"policy": "average", "ids": ids, "matrix": mat}


def classify(gallery: dict, vec: np.ndarray, threshold: float,
             margin: float) -> dict:
    """Classify one probe embedding against the gallery with the SAME rule the
    service uses: accept the best identity only if it clears ``threshold`` and
    beats the runner-up identity by ``margin``; else ``Unknown``."""
    ids = gallery["ids"]
    mat = gallery["matrix"]
    vec = _unit(np.asarray(vec, dtype=np.float32))
    if mat is None or not ids:
        return {"identity": None, "similarity": 0.0, "margin": 0.0}
    sims = mat @ vec
    if gallery["policy"] == "nearest":
        scores = np.full(len(ids), -1.0, dtype=np.float32)
        np.maximum.at(scores, gallery["row_id"], sims)
    else:
        scores = sims
    order = np.argsort(scores)[::-1]
    best = int(order[0])
    best_s = float(scores[best])
    runner = float(scores[int(order[1])]) if len(order) > 1 else -1.0
    m = best_s - runner
    if best_s >= threshold and m >= margin:
        return {"identity": ids[best], "similarity": best_s, "margin": m}
    return {"identity": None, "similarity": best_s, "margin": m,
            "closest": ids[best]}


UNKNOWN = "__unknown__"


def open_set_report(gallery_embeddings: dict[str, list[np.ndarray]],
                    probes: list[tuple[np.ndarray, Optional[str]]],
                    threshold: float, margin: float,
                    policy: str = "average") -> dict:
    """Full open-set evaluation.

    ``probes`` is a list of ``(embedding, true_identity_or_None)`` where None (or
    an identity not in the gallery) marks a stranger. Returns confusion counts,
    Accuracy/Precision/Recall/F1, FAR and FRR.
    """
    from sklearn.metrics import precision_recall_fscore_support

    gallery = build_gallery(gallery_embeddings, policy=policy)
    enrolled = set(gallery["ids"])

    y_true: list[str] = []
    y_pred: list[str] = []
    far_num = far_den = 0     # impostor accepted as some employee
    frr_num = frr_den = 0     # genuine rejected or misidentified
    correct_known = 0
    details = []

    for emb, truth in probes:
        is_known = truth in enrolled and truth is not None
        pred = classify(gallery, emb, threshold, margin)
        pred_id = pred["identity"]
        t_lab = truth if is_known else UNKNOWN
        p_lab = pred_id if pred_id is not None else UNKNOWN
        y_true.append(t_lab)
        y_pred.append(p_lab)

        if is_known:
            frr_den += 1
            if pred_id == truth:
                correct_known += 1
            else:
                # missed (unknown) OR matched to the wrong employee: both are
                # false rejections of the true identity
                frr_num += 1
        else:
            far_den += 1
            if pred_id is not None:
                far_num += 1  # a stranger became an employee — the worst case

        details.append({
            "true": t_lab, "pred": p_lab,
            "similarity": round(float(pred["similarity"]), 4),
            "margin": round(float(pred["margin"]), 4),
        })

    labels = sorted(enrolled) + [UNKNOWN]
    # confusion matrix as nested dict
    idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    accuracy = float(np.mean([t == p for t, p in zip(y_true, y_pred)])) if y_true else 0.0

    far = (far_num / far_den) if far_den else None
    frr = (frr_num / frr_den) if frr_den else None

    return {
        "threshold": round(float(threshold), 4),
        "margin": round(float(margin), 4),
        "policy": policy,
        "n_probes": len(probes),
        "n_known_probes": frr_den,
        "n_unknown_probes": far_den,
        "n_enrolled_identities": len(enrolled),
        "accuracy": round(accuracy, 4),
        "precision_macro": round(float(prec), 4),
        "recall_macro": round(float(rec), 4),
        "f1_macro": round(float(f1), 4),
        "far": None if far is None else round(far, 4),
        "frr": None if frr is None else round(frr, 4),
        "known_correct": correct_known,
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "details": details,
    }


# ----------------------------------------------------------------------------
# dataset embedding
# ----------------------------------------------------------------------------

def embed_dataset(embed_fn: Callable[[np.ndarray], Optional[np.ndarray]],
                  images_by_identity: dict[str, list[np.ndarray]]
                  ) -> tuple[dict[str, list[np.ndarray]], dict]:
    """Embed every image; drop those with no detectable face. Returns
    ``(embeddings_by_identity, stats)``."""
    out: dict[str, list[np.ndarray]] = {}
    total = detected = 0
    for ident, imgs in images_by_identity.items():
        vecs = []
        for img in imgs:
            total += 1
            v = embed_fn(img)
            if v is not None:
                vecs.append(np.asarray(v, dtype=np.float32))
                detected += 1
        if vecs:
            out[ident] = vecs
    return out, {"images": total, "faces_detected": detected,
                 "detection_rate": round(detected / total, 4) if total else 0.0}
