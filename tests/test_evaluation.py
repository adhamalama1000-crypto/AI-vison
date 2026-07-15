"""
Evaluation-engine tests.

Two layers:

* **Unit** — the metrics engine (ROC/EER/FAR/FRR/confusion/open-set) is checked
  against synthetic embeddings with known, hand-computable answers. These run
  everywhere with no model or network.
* **Real end-to-end** — the genuine InsightFace (SCRFD + ArcFace) pipeline is run
  over a real LFW subset and the headline metrics are asserted. This test is the
  honest proof that recognition works and that a stranger is not accepted; it
  SKIPS (never fails) when the real weights or the dataset are unavailable, so
  "not validated here" is distinguishable from "validated and wrong".
"""

from __future__ import annotations

import numpy as np
import pytest

from rtsp_backend.ai import evaluation as ev


# ---- synthetic fixtures: 3 well-separated identities in 8-D -----------------

def _identity_embeddings(seed=0, n_ids=3, per=6, dim=8, noise=0.02):
    rng = np.random.default_rng(seed)
    centres = np.eye(n_ids, dim, dtype=np.float32)  # orthogonal -> ~0 cross-sim
    out = {}
    for i in range(n_ids):
        vecs = []
        for _ in range(per):
            v = centres[i] + rng.normal(0, noise, dim).astype(np.float32)
            vecs.append(v / np.linalg.norm(v))
        out[f"id{i}"] = vecs
    return out


def test_genuine_impostor_separation():
    data = _identity_embeddings()
    genuine, impostor = ev.genuine_impostor_scores(data)
    assert genuine.size > 0 and impostor.size > 0
    # same-identity similarity must dominate cross-identity similarity
    assert genuine.mean() > 0.9
    assert impostor.mean() < 0.3


def test_roc_eer_perfect_separation():
    genuine = np.array([0.9, 0.85, 0.95, 0.88], dtype=np.float32)
    impostor = np.array([0.1, 0.2, 0.05, 0.15], dtype=np.float32)
    roc = ev.roc_eer(genuine, impostor)
    assert roc["auc"] == 1.0
    assert roc["eer"] == 0.0
    # a threshold between the two clusters separates them perfectly
    assert 0.2 < roc["eer_threshold"] <= 0.9


def test_far_frr_monotonic():
    genuine = np.array([0.7, 0.8, 0.9], dtype=np.float32)
    impostor = np.array([0.1, 0.2, 0.6], dtype=np.float32)
    low = ev.far_frr_at(genuine, impostor, 0.5)
    high = ev.far_frr_at(genuine, impostor, 0.85)
    # raising the threshold cannot increase FAR and cannot decrease FRR
    assert high["far"] <= low["far"]
    assert high["frr"] >= low["frr"]


def test_open_set_rejects_stranger():
    data = _identity_embeddings(per=8)
    gallery = {k: v[:5] for k, v in data.items()}
    probes = [(v, k) for k, vs in data.items() for v in vs[5:]]  # known probes
    # a stranger: orthogonal to every enrolled identity
    stranger = np.zeros(8, dtype=np.float32); stranger[7] = 1.0
    probes += [(stranger, None), (stranger, None)]

    rep = ev.open_set_report(gallery, probes, threshold=0.65, margin=0.05,
                             policy="average")
    assert rep["far"] == 0.0                 # stranger never accepted
    assert rep["frr"] == 0.0                 # every known probe identified
    assert rep["accuracy"] == 1.0
    assert rep["n_unknown_probes"] == 2
    # confusion matrix is square over labels incl. the Unknown row/col
    assert len(rep["confusion_matrix"]) == len(rep["labels"])


def test_open_set_margin_blocks_ambiguous():
    # two identities whose centroids are very close: a probe near both should be
    # rejected by the margin rule even if it clears the threshold.
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.98, 0.199], dtype=np.float32); b /= np.linalg.norm(b)
    gallery = {"a": [a], "b": [b]}
    probe = (a + b); probe /= np.linalg.norm(probe)   # sits between a and b
    rep = ev.open_set_report(gallery, [(probe, "a")], threshold=0.6,
                             margin=0.2, policy="average")
    # high margin requirement -> ambiguous probe is not accepted as "a"
    assert rep["far"] is None or rep["n_unknown_probes"] == 0
    assert rep["known_correct"] == 0


# ---- real end-to-end evaluation (skips if model/dataset unavailable) --------

def _load_real_embedder():
    from rtsp_backend.ai.embedders import InsightFaceEmbedder
    emb = InsightFaceEmbedder(models_dir="models", model_pack="buffalo_l",
                              det_size=512, det_thresh=0.5)
    emb.load()
    return emb


def _capped_dataset(max_per=10):
    from scripts.fetch_lfw_eval import load_dataset
    data = load_dataset(verbose=False)
    return {k: v[:max_per] for k, v in data.items()}


@pytest.mark.slow
def test_real_pipeline_low_far_high_accuracy():
    try:
        embedder = _load_real_embedder()
    except Exception as exc:  # weights not available in this environment
        pytest.skip(f"real InsightFace weights unavailable: {exc}")

    try:
        dataset = _capped_dataset(max_per=10)
    except Exception as exc:
        pytest.skip(f"evaluation dataset unavailable: {exc}")
    if len(dataset) < 4:
        pytest.skip("evaluation dataset too small / download blocked")

    from scripts.fetch_lfw_eval import KNOWN

    def embed_fn(img):
        faces = embedder.detect_and_embed(img)
        if not faces:
            return None
        _b, _s, vec = max(faces, key=lambda f: (f[0].x2 - f[0].x1) * (f[0].y2 - f[0].y1))
        return vec

    embs, stats = ev.embed_dataset(embed_fn, dataset)
    assert stats["detection_rate"] > 0.8, "SCRFD failed to detect most faces"

    known_ids = {k for k in KNOWN if k in embs}
    gallery, probes = {}, []
    for ident, vecs in embs.items():
        if ident in known_ids:
            gallery[ident] = vecs[:6]
            probes += [(v, ident) for v in vecs[6:]]
        else:
            probes += [(v, None) for v in vecs]

    # verification: genuine vs impostor must be strongly separated
    genuine, impostor = ev.genuine_impostor_scores(
        {k: v for k, v in embs.items() if k in known_ids})
    roc = ev.roc_eer(genuine, impostor)
    assert genuine.mean() > impostor.mean() + 0.3
    assert roc["auc"] is not None and roc["auc"] > 0.95

    # open-set at the production default threshold: the anti-FAR guarantee
    rep = ev.open_set_report(gallery, probes, threshold=0.65, margin=0.05,
                             policy="average")
    assert rep["n_unknown_probes"] > 0, "need strangers to measure FAR"
    assert rep["far"] == 0.0, f"a stranger was accepted (FAR={rep['far']})"
    assert rep["accuracy"] > 0.9, f"identification accuracy too low: {rep['accuracy']}"
