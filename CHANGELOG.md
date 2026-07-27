# Changelog

## [5.1.0] — Madkour AI Panel Inspector: component recognition redesign

The panel-inspection AI has been rebuilt rather than tuned. The previous system
reported hundreds of phantom wires and zero components; that was the exact,
predictable output of the configuration that shipped. Full root-cause analysis
with reproducible before/after numbers:
[`docs/AUDIT_PANEL_INSPECTOR.md`](docs/AUDIT_PANEL_INSPECTOR.md).

### Removed: wiring detection

- The `wires` task now defaults to **`null_wires`** and is **disabled**. A
  persisted `advanced_wires` / `classical_wires` selection is **migrated away on
  startup** and reported in `GET /api/ai/status` — an existing installation stops
  producing phantom wires on upgrade with no operator action.
- Measured on 25 synthetic panels containing **zero** conductors:
  `advanced_wires` reported **715** false wires (28.6/image, worst 57);
  `classical_wires` reported **4 494** (179.8/image, hitting its own 200-line
  cap). Precision 0.00. The redesigned path reports **0**.
- `panel_svc.analyze` has no wire stage; results carry
  `wire_analysis: {enabled: false, reason: …}` so the decision is stated, not
  silently zero.
- `panels/template.py` (reference-panel learning) no longer traces wires unless
  `RTSP_ENABLE_WIRE_TRACING=1` or `wire_params={"enabled": True}`.
- Both tracers stay registered and flagged `experimental` with a warning string
  surfaced in the backend catalogue — research reproducibility, not product use.

### Fixed: three defects that made a trained model useless

- **Label shift.** The ONNX decoder chose its output format from the raw column
  count (`>= 85` meant YOLOv5), true only for an 80-class COCO model. Every
  electrical class set fell below it, so objectness was read as class 0 and every
  label shifted by one. Measured label accuracy with the 53-class taxonomy:
  **0.017 → 1.000**. Format is now resolved from the declared class count.
- **`models/components/labels.txt` contained the lines `0`–`9`**, and was
  preferred over the class list — components would have been named `"0"`…`"9"`.
  Deleted; replaced by generated `models/components/classes.json`, pinned by a test.
- **Class-agnostic NMS.** One NMS across all classes suppressed genuinely stacked
  devices (an overload relay bolted under its contactor, a CT around a busbar).
  NMS is now per class, with cross-class dedupe restricted to confusable groups.

Also fixed: per-row Python decode loop (now vectorised), boxes never clipped to
the frame, a `np.squeeze` that silently dropped **single**-detection outputs, and
missing RT-DETR support.

### New: `rtsp_backend/electrical/`

- `taxonomy.py` — **53 industrial component classes**, each with engineering
  function, panel role, electrical domain, mounting style, aspect-ratio and
  relative-area priors, dataset aliases, zero-shot prompts and a per-class
  confidence threshold. Ambiguous labels (`"circuit breaker"`) deliberately do
  **not** resolve — they become unknown rather than a guess.
- `postprocess.py` — six-stage suppression cascade (sanitise → per-class NMS →
  confusable cross-class dedupe → geometric plausibility → confidence gate with
  **Unknown Industrial Component** demotion → DIN-rail row grouping), with
  per-stage drop accounting exposed in the API and UI. Measured against the old
  logic: false positives **505 → 66 (−86.9 %)**, precision **0.591 → 0.916**,
  F1 **0.673 → 0.834**; recall −0.016 and mAP@50 −0.012 are the accepted trade.
- `recognizer.py` — `industrial_onnx`, `industrial_ultralytics`, and **zero-shot**
  `openvocab_owlv2` / `openvocab_grounding_dino` / `openvocab_florence2` driven by
  the taxonomy prompts (no dataset needed), plus a corroboration-weighted
  `industrial_ensemble`.
- `nameplate.py` — **90 manufacturer part-number signatures** across 21
  manufacturers → manufacturer, product family, and a cross-check against the
  detector's class. A disagreement is reported, not hidden.
- `expert.py` — per-component engineering record with context-sensitive purpose (a
  contactor next to an overload relay is a motor starter; next to capacitors it is
  a PFC stage), bill of materials, row layout description.
- `panel_type.py` — **12 panel archetypes** as weighted evidence rules, expected-BOM
  gap analysis, and 12 engineering maintenance checks. Measured: **12/12** top-1 on
  engineered inventories, **4/4** honest refusals on ambiguous ones.
- `inspector.py` — the engine and the report builder (all nine required sections).
- `metrics.py` — precision/recall/F1, AP, mAP@50, mAP@50-95, confusion matrix with
  background row **and** column, false-positive cause analysis, false-negative
  analysis, per-class threshold optimiser, model comparison table.

### New: `training/electrical/`

- `datasets.py` — public-source registry with per-source label maps, YOLO
  label-space remapping onto the taxonomy (unmappable classes **dropped with a
  count**, never folded), multi-dataset merge, trainability/coverage report, and
  the Madkour field-capture protocol.
- `synthetic.py` — labelled data generation: composition from real device crops
  (the useful mode) and procedural stand-ins, with lighting, perspective,
  rotation, occlusion, dust, shadow, reflection, blur and JPEG artefacts. Every
  manifest records whether the data is real-sourced or procedural.
- `train.py` — Ultralytics driver for YOLOv11 / YOLOv8 / RT-DETR (and YOLOv12
  **only if the installed build exposes it** — otherwise `skipped`, never
  substituted), panel-specific augmentation (no horizontal mirroring, 960 px,
  `close_mosaic`), ONNX export with a pinned class map, and an architecture
  benchmark ranked on measured mAP.
- `cli.py` — `plan · synth · remap · merge · analyse · train · bench · eval · tune`.

### New: validation harness

`scripts/validate_panel_inspector.py` reproduces every number above. It also
states its own limits: it validates the pipeline on exact ground truth and does
**not** measure real-world accuracy on Madkour panels.

### API

- `GET /api/components/classes` now returns the full taxonomy, not a bare list.
- `GET /api/components/panel-types` — the archetypes and their evidence rules.
- `GET /api/components/nameplate-catalogue` — manufacturer signature coverage.
- `GET /api/ai/status` reports `migrations`, and each backend carries
  `deprecated` / `experimental` / `warning`.
- `POST /api/panels/analyze` returns panel type, function, application, bill of
  materials, missing components, maintenance notes, gate diagnostics and the
  full report alongside the legacy keys.

### UI — renamed to **Madkour AI Panel Inspector**

- Industrial control-room redesign: **dark by default** (light theme retained),
  engineering-blue `#2D8CDC` with signal amber, machined card elevation, DIN-rail
  texture, scan-sweep and count-up animations.
- New **Panel Inspector** page: annotated panel, panel-type verdict with
  confidence bar and evidence, layout rows, bill of materials, an expandable
  component table (class · confidence · bbox · centre · row/position, expanding to
  function, purpose, manufacturer, part number, nameplate text), possible missing
  components, maintenance notes, confidence statistics and the detection-gate
  breakdown.
- Original brand mark (a DIN rail carrying modular devices) — no Madkour brand
  asset is reproduced.
- Fixed: a loaded-but-disabled AI task was badged "ready", which read as running.

### Face recognition

Unchanged and fully functional. The entire pre-existing face suite passes.

### Tests

340 → 400+ tests. New: `test_electrical_taxonomy.py`,
`test_electrical_postprocess.py`, `test_electrical_intelligence.py`,
`test_electrical_recognizer.py`, `test_electrical_training.py`, plus startup
migration coverage in `test_ai_manager.py`.

## [5.0.0] — Real production face recognition (SCRFD → ArcFace → cosine)

The face pipeline has been rebuilt around the **real InsightFace models** and
tuned to **avoid false recognition** above all else. It is no longer possible
for the system to silently run on a weak fallback: the real model is the default
and, if its verified weights cannot be loaded, the backend reports a precise
error instead of degrading.

### Recognition pipeline
- **SCRFD detection → ArcFace 512-d embeddings → cosine similarity**, via
  InsightFace `buffalo_l` (default) or `buffalo_s` (faster). The active backend,
  embedding dimension and index engine are surfaced in the UI and at
  `GET /api/ai/face/config`.
- **Model provisioning with integrity verification** (`ai/model_provision.py`):
  every ONNX weight is checked against its canonical **SHA-256** before use and
  fetched from ordered mirrors, so it works behind restricted networks and can
  never load a corrupted/substituted model.
- **Never guess.** Below the threshold (default **0.65**, configurable from the
  frontend), or when the best match does not beat the runner-up employee by the
  identity **margin**, the face is reported as **“Unknown Employee”** — never the
  closest employee. Tuned for a **low False Acceptance Rate**.
- **Multiple embeddings per employee** with `average` (centroid) or `nearest`
  matching policy. Per-sample **quality score + metadata** (blur, brightness,
  detection score, bbox) are stored; individual samples can be deleted and an
  employee can be **re-trained** with the current model.
- **Quality gating** rejects blurry, tiny, over/under-exposed and multi-face
  captures with clear messages (No face detected / Face too blurry / Face too
  small / Multiple faces / Bad lighting).
- **Efficient search:** a FAISS `IndexFlatIP` (cosine) is used when available,
  otherwise a single vectorised NumPy matmul — never a per-frame Python scan.

### Live camera & frontend
- Every detected face draws a **green (employee) / red (unknown)** box with
  **name, similarity %, confidence and status**.
- New **Face Recognition** page: live recognition feed, per-face detail
  (name / similarity / unknown warnings), employee gallery with sample counts,
  a **threshold + margin + policy** control, recognition history and attendance
  log.

### Evaluation (real, verified — not claimed)
- `ai/evaluation.py` + `scripts/evaluate_face_recognition.py` compute Accuracy,
  Precision, Recall, F1, ROC/AUC, EER, a confusion matrix, and **FAR/FRR** from
  the real model over a real LFW subset. Measured on 309 LFW images / 16
  identities at threshold 0.65: **AUC 1.0, EER 0.0, Accuracy/Precision/Recall/F1
  = 1.0, FAR 0.0, FRR 0.0** (genuine mean cosine 0.70 vs impostor 0.01).
- Performance (CPU, this environment): `buffalo_l` full detect+embed ≈
  175–230 ms/frame (best on GPU); `buffalo_s` ≈ 26 ms/frame, meeting the
  <40 ms detection / <100 ms recognition targets. Inference is offloaded to the
  threadpool and throttled, so streaming FPS stays stable.

## [3.1.5] — Production React frontend + RTSP-camera-only registration

### Frontend — complete rebuild (React + TypeScript + Vite)

The previous plain-HTML/JS interface has been replaced by a modern, responsive
single-page application suitable for commercial deployment. It is pre-built into
`rtsp_backend/web/` and served directly by the backend — **no Node.js is needed
at runtime**. Source lives in `frontend/`.

- **Stack:** React 18 + TypeScript + Vite, Tailwind CSS design system with
  light/dark theme tokens, TanStack Query for data/polling, Recharts for
  charts, lucide-react icons, React Router.
- **Dark & light mode** with a persisted preference and instant toggle.
- **Dashboard** — live stat cards, real-time FPS area chart (fed by the events
  WebSocket), CPU/RAM radial gauges, a recognition donut, per-task AI status,
  and a recent-events feed. All values come from `/api/stats/dashboard` and the
  `/ws/events` socket.
- **Live Cameras** — large RTSP preview with an AI-overlay/raw toggle,
  fullscreen support, one-click snapshot download, a camera switcher with live
  thumbnails, and per-camera stats (fps, latency, frame age, uptime, transport,
  frame counters).
- **Employees** — searchable, paginated table with add / edit / delete. The
  add/edit dialog shows the **live camera inside the dialog**, captures faces
  **directly from the RTSP stream**, supports **multiple captures** with a
  **thumbnail gallery** and per-image **retake/delete**, saves atomically via
  `/api/employees/register`, then switches to a **live verification** view that
  draws recognition overlays and confirms the match on-screen.
- **Events** — filter by event type and camera, snapshot thumbnails with a
  full-size preview dialog, absolute + relative timestamps, and clear-all.
- **AI Models** — a card per task showing status, device, throughput (fps),
  inference time, an enable/disable toggle, a backend selector, and a match
  threshold slider, plus compute-resource gauges.
- **Settings** — RTSP camera CRUD (with `rtsp://` validation), appearance/theme
  controls, and system health.
- **Serving:** the backend redirects `/` → `/app/`, serves the built assets, and
  falls back to `index.html` for client-side routes so deep links survive a
  refresh. The app is fully self-contained (no external CDN/font dependencies).

### Backend

## [3.1.5-backend] — RTSP-camera-only employee registration

This release makes employee enrolment work **entirely from the live RTSP
camera** and guarantees recognition works **immediately after enrolment** — no
manual image upload, no manual model toggle, and no backend restart.

### Fixed

- **Recognition did not start by itself after enrolment.** The `face` task
  shipped *disabled*, so an enrolled employee stored a face vector but was never
  recognised until someone manually enabled the model on the AI Models page.
  - Face recognition is now **enabled by default** on a fresh install.
  - Enrolment now **auto-enables** the face task
    (`AIModelManager.ensure_enabled("face")`), so recognition is live on the
    very next frame.
- **The UI fell back to local file upload.** With no `config.yaml`, zero
  cameras loaded and the wizard offered "Upload image", which is exactly the
  manual path that was not wanted.
  - A working `config.yaml` is now shipped (RTSP camera declared there).
  - The **file-upload path was removed entirely** from the employee wizard.
- **A bad/degenerate frame could return HTTP 500 and interrupt capture.**
  OpenCV's Haar `detectMultiScale` can raise an internal range-check error on
  some frames.
  - `OpenCVFallbackEmbedder.detect_faces` now guards empty frames and swallows
    OpenCV internal errors, returning `[]` instead of crashing.
  - Enrolment wraps the face-service call so a bad frame becomes a clean
    rejection (`enrollment_error` / `no_face_detected` / `blurry`), never a 500.

### Added

- **`POST /api/employees/register`** — atomic RTSP enrolment: creates the
  employee and enrols every captured frame in one call. If no frame yields a
  usable face the whole operation is **rolled back** (no faceless employees).
  Response includes `enrolled`, `rejected`, and `recognition_enabled`.
- **Live in-dialog verification.** After saving, the Add-employee dialog
  switches to the AI-annotated stream (name + confidence drawn on the feed) and
  reports the recognition result live, so the operator confirms it works
  on the spot.
- Regression tests in `tests/test_rtsp_registration.py`:
  auto-enable-on-enrolment, atomic register + immediate recognition, no-face
  rollback, blurry rejection without crashing, and detector robustness on bad
  frames.

### Changed

- The Add-employee wizard is now **RTSP-only**: shows the live camera, captures
  straight from the stream (validated — rejects no-face and blurry frames, warns
  on multiple faces), supports **multiple captures with delete/retake**, then
  saves via `/register`.

### Behaviour preserved

- Capture and snapshot run in the thread pool: the MJPEG stream **does not
  freeze** during capture, and **snapshots do not interrupt** the stream (both
  verified under concurrent load).
- Recognition and unknown-person events (attendance) are written to the
  `events` table automatically by the background worker.
- The RTSP-only source policy is unchanged — there is no USB / local-file
  fallback anywhere.

---

## Verification (this release)

Performed against a live RTSP camera (a real H.264 RTSP stream carrying a real
face) and a fresh database:

1. Clean boot via `python run.py` (reads `config.yaml` + `.env`) — fresh DB.
2. Camera connects and streams (~15 fps, 0 dropped frames); MJPEG live.
3. Face recognition auto-enabled on boot (`state: running`, `ready: true`).
4. Validate live capture — single sharp face accepted; no upload used.
5. Register from captures — 2 enrolled, 0 rejected, recognition auto-enabled.
6. Recognition immediate — enrolled employee matched at ~0.9996 confidence.
7. Attendance/events stored (`face_recognized`, `unknown_person`).
8. AI overlay (`ai-snapshot` / `ai-stream`) draws name + confidence.
9. No freeze — stream keeps advancing while 6 captures + 6 snapshots run
   concurrently (capture max ~0.25 s).
10. AI Models endpoints — enable/disable/select/params/metrics all functional.
11. Full browser UI (headless Chromium) — wizard capture → save → live
    "Recognized ✓ … · 100.0% confidence", zero console errors.

Test suite: **88 passed** (`pytest`).

## Running

```bash
pip install -r requirements.txt        # runtime deps
# edit config.yaml -> cameras[0].url with your RTSP camera URL
python run.py                          # serves on the host/port from .env / config.yaml
# open the dashboard at http://<host>:<port>/  (default port 8090)
```

For tests: `pip install -r requirements-dev.txt && pytest`.

## Rebuilding the frontend (optional)

The UI is already built into `rtsp_backend/web/`, so running the backend is all
that's required. To modify and rebuild it:

```bash
cd frontend
npm install
npm run build     # outputs to ../rtsp_backend/web
# or: npm run dev  (Vite dev server on :5173, proxies the API to :8090)
```
