# Face Recognition (SCRFD → ArcFace → Cosine)

Production employee face recognition built on the **real InsightFace** models.
No mock models, no fabricated matches: the real pipeline is the default, and if
its verified weights cannot be loaded the backend reports an error rather than
falling back to anything weaker.

## Pipeline

```
frame ─▶ SCRFD (detection)  ─▶ ArcFace R50 (512-d embedding) ─▶ cosine similarity
                                                                     │
                                        threshold (0.65) + margin ──▶ Employee / Unknown Employee
```

- **Detection:** SCRFD (`det_10g` for `buffalo_l`, `det_500m` for `buffalo_s`).
- **Recognition:** ArcFace, 512-d L2-normalised embeddings
  (`w600k_r50` / `w600k_mbf`).
- **Matching:** cosine similarity against every enrolled embedding, served by
  FAISS (`IndexFlatIP`) when installed, else a vectorised NumPy matmul.

## Installation

```bash
pip install -r requirements.txt -r requirements-face.txt
# insightface pulls opencv-python, which clobbers the headless build — restore it:
pip install --force-reinstall "opencv-python-headless>=4.9,<5"
```

## Model weights

Weights are provisioned and **SHA-256 verified** at first load by
`rtsp_backend/ai/model_provision.py`. It downloads from ordered mirrors
(respecting `HTTPS_PROXY` and the system CA bundle) and rejects any file whose
hash does not match the canonical InsightFace weight. In an environment with
GitHub access, InsightFace's own downloader also works.

- Default location: `<models_dir>/insightface/models/<pack>/`.
- Override with `RTSP_FACE_MODEL_ROOT` to share one weight cache across
  instances / test runs.

Packs:

| Pack        | Detection  | Recognition   | Dim | CPU latency\* | Use when                         |
|-------------|------------|---------------|-----|---------------|----------------------------------|
| `buffalo_l` | det_10g    | w600k_r50     | 512 | ~175–230 ms   | accuracy / lowest FAR (default)  |
| `buffalo_s` | det_500m   | w600k_mbf     | 512 | ~26 ms        | speed (<40/<100 ms targets)      |

\* full detect+embed per frame, CPU, one face. Both meet targets on a GPU.

## Enrolment

Capture **10–20 samples per employee** across head angles, lighting and
expressions. Each capture is quality-gated (blur / size / exposure / single
face) and stored with a quality score + metadata. Recognition goes live
immediately — no restart.

- `POST /api/employees/register` — create + enrol several captures atomically.
- `POST /api/employees/{id}/images` — add one capture.
- `GET  /api/employees/{id}/embeddings` — list samples with quality/metadata.
- `DELETE /api/employees/{id}/embeddings/{emb_id}` — delete one sample.
- `POST /api/employees/{id}/retrain` — recompute embeddings with the current model.

## Configuration (adjustable from the frontend)

`GET /api/ai/face/config` · `PUT /api/ai/face/config`

| Param            | Default    | Meaning                                             |
|------------------|------------|-----------------------------------------------------|
| `threshold`      | `0.65`     | min cosine similarity to accept an identity         |
| `margin`         | `0.05`     | best match must beat the runner-up employee by this |
| `match_policy`   | `average`  | `average` (centroid) or `nearest` (best sample)     |
| `min_det_score`  | `0.5`      | min SCRFD detection confidence                      |
| `model_pack`     | `buffalo_l`| `buffalo_l` or `buffalo_s`                          |
| `det_size`       | `640`      | detector input size (lower = faster)                |

Raising `threshold`/`margin` lowers the False Acceptance Rate (a stranger
becoming an employee) at the cost of a slightly higher False Rejection Rate —
this system deliberately favours the former.

## Evaluation

Real metrics (Accuracy, Precision, Recall, F1, ROC/AUC, EER, confusion matrix,
FAR, FRR) over a real LFW subset:

```bash
python -m scripts.fetch_lfw_eval            # cache the dataset (git-ignored)
python -m scripts.evaluate_face_recognition --threshold 0.65 --policy average
```

Verified result (309 LFW images, 16 identities, threshold 0.65):

| Metric | Value |
|--------|-------|
| ROC AUC | 1.0 |
| EER | 0.0 |
| Accuracy / Precision / Recall / F1 | 1.0 |
| FAR (stranger → employee) | 0.0 |
| FRR (employee missed) | 0.0 |
| genuine vs impostor mean cosine | 0.70 vs 0.01 |

The evaluation engine lives in `rtsp_backend/ai/evaluation.py`; unit tests and a
real end-to-end test are in `tests/test_evaluation.py` (the real test skips
cleanly if weights/dataset are unavailable).
