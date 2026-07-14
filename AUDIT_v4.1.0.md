# Full-project audit (v4.1.0)

A whole-project review — every backend endpoint, frontend page, DB table, AI
pipeline, training workflow, and test — with fixes applied. Findings were
gathered by two focused review passes (frontend; backend security+correctness)
plus behavioural probing (path-traversal attempts, upload caps, job control,
concurrency), then verified and fixed.

## Security fixes applied

| Severity | Issue | Fix |
|---|---|---|
| CRITICAL | `/api/media` served any file under `data_dir`, incl. the SQLite DB (`platform.db`, `-wal`, `-shm`) → full PII/embeddings/attendance download | `/api/media` now only serves a whitelist of media subdirs (`snapshots, employees, panels, inspections, reports, reference`) and never `*.db*`. |
| CRITICAL | No authentication on any endpoint | Optional API-key gate: set `RTSP_API_KEY` (env or config) → every `/api` + control request must send `X-API-Key` (or `?api_key=`); `/health` and the dashboard shell stay open. Unset = open (dev/LAN/behind-proxy). |
| HIGH | Unbounded upload read into memory (`panels`, `inspection`) → memory-exhaustion DoS | Chunked, capped reads (`read_upload_capped`) abort past `RTSP_MAX_UPLOAD_BYTES` (default 512 MiB) with HTTP 413. |
| HIGH | Zip-bomb / unbounded dataset upload → disk exhaustion | `safe_extract_zip` now rejects archives over a declared-uncompressed-size cap (5 GiB) and a file-count cap (200k) before extracting; dataset/reference uploads stream to disk with the same per-file cap. |
| HIGH | A public model-select could trigger `pip install`/`uninstall` on the host | InsightFace auto-install now OFF unless `RTSP_ALLOW_AUTO_INSTALL=1` (or an explicit `auto_install` param). |

Verified NOT vulnerable (checked): the WHERE-clause builders (only fixed column
names are interpolated; all values use `?` params) — no SQL injection; upload
path handling and zip extraction — no path-traversal / zip-slip bypass; no
`eval`/`exec`/`pickle`/user-fed `subprocess`.

## Correctness / performance fixes applied

| Severity | Issue | Fix |
|---|---|---|
| HIGH | A JPEG snapshot was written on **every** worker tick while an unknown person / event subject stayed in frame (only the DB event was deduped) → disk fill | Snapshot now gated by the same dedup window as the event; the JPEG is written only when a new event is actually logged. |
| HIGH | One `components` DB row inserted per component **per frame** (≈5×/s) → unbounded table growth + DB-lock churn | Component persistence throttled to once per 2 s per camera; the live result still reflects every frame. |
| MEDIUM | `AIPipeline.process()` was called concurrently (background worker + `analyze`/`ai-snapshot`/`topology`) mutating unlocked shared state (trackers, dedup, attendance) → races / double events | `process()` now holds a per-camera lock. |
| MEDIUM | Attendance dedup used only an in-memory map → a restart within the 8 h window created a duplicate row | Always consults the DB (last row within the window / one-per-day for ≥12 h timeouts). |
| MEDIUM | Heavy CPU/IO on the event loop: panel/inspection analysis, `wires/topology`, and model `select/enable/params` (ONNX load) blocked all streams | All offloaded via `asyncio.to_thread`. |
| MEDIUM | Training models neither classifier nor detection were silently dropped; defaults silently substituted | Unknown models now reported as `skipped` with a reason; the demo default is used only when *no* models are requested. |
| MEDIUM | Daemon training threads could write to the DB after `db.close()` at shutdown | Lifespan now stops + joins training threads (`TrainingManager.shutdown()`) before closing the DB. |
| LOW | Tracker `_id_remap` grew forever | Pruned to live tracks past a threshold. |
| LOW | `sys.path` mutated from worker threads on every job | Done once at import. |
| LOW | Training catalog advertised 9 detection archs as trainable; only 5 have adapters | Catalog now reports `detection_models_trainable` + `ultralytics_available`. |

## Frontend fixes applied

- **Crash**: `Reports`/`PanelAnalysis` rendered a parsed-JSON `summary` object directly as a React child (would throw). Now rendered via a `summaryText()` helper.
- **Wrong data**: dataset status badge compared against `ok`/`error` instead of `valid`/`invalid`; the Metrics camera table read nested fields that the flat `/api/metrics` payload doesn't have (every row showed "—"). Both fixed to the real shapes.
- **Runaway/again-stuck polling**: `TrainingProgress` only polled while `running` (never advanced from `queued`); job list polled forever. Both now poll all non-terminal states and stop on terminal.
- Comparison links now also show for `stopped` jobs; report kind labels/tones corrected.

## Test coverage added

`tests/test_platform_upgrade.py` now also covers: media never serves the DB,
API-key gating, upload size cap (413), zip-slip + file-count guards, and unknown
training models being reported as skipped. Full suite green.

## Still requires a real dataset / external resources (unchanged, honest)

- **Electrical component/wire detection** returns empty until you upload a
  labelled electrical dataset and train a detector (install `ultralytics` for
  real YOLO/RT-DETR training). Only 5 of 9 advertised detection archs have
  Ultralytics adapters; the rest report `skipped`.
- **GPU/CUDA** absent → CPU training only, no TensorRT, GPU metrics `unavailable`.
- **InsightFace ArcFace** model pack downloads from GitHub (blocked in this
  sandbox); works on a normal host.

## Known limitations / partial (documented, not bugs)

- **No auth by default** — the API-key gate exists but is off unless configured;
  a production deployment must set `RTSP_API_KEY` (or sit behind an auth proxy).
- **TensorBoard, cross-validation, EMA, MixUp/Mosaic/CopyPaste** — partial:
  augmentation + early stopping + Optuna + cosine-LR reporting are implemented;
  detection-specific augmentations are delegated to Ultralytics.
- **Wire fault classification** (broken/loose) — the classical baseline reports
  geometry + colour honestly as `unknown`; true fault classification needs a
  trained model.
- **Real-time live-stream mismatch highlighting** — inspection runs on a
  frame/upload and annotates that frame; it is not yet a continuous overlay on
  the live MJPEG stream.
