# Changelog

## [5.3.0] — Closing the production gaps

A full audit against the twelve-task production brief
([`docs/AUDIT_v5.2.0.md`](docs/AUDIT_v5.2.0.md)) found **no TODOs, no stubs and no
mocked AI paths** in production code — the `NotImplementedError`s are legitimate ABC
contracts. What it found instead were eight capabilities the brief requires that had
never been built. All eight are now built. No existing feature was removed,
simplified or rewritten.

### Dataset builder (Task 2)

- **`training/electrical/dedup.py`** — near-duplicate detection and removal. The
  registry already warned that two public switchgear sources are probably the same
  photographs republished; nothing detected it. dHash + aHash within a Hamming
  distance of 5, plus an exact-content pass, one connected-components sweep over both
  relations, and **cross-split leak detection reported separately** because that is
  the case that corrupts metrics. `cli dedup` is read-only by default and **exits
  non-zero on cross-split leakage** so CI can gate on it; `cli merge --dedup` wires it
  into the merge. Removal keeps the *training* copy and drops the val/test one — the
  only direction that leaves evaluation data unseen. Duplicates whose **labels
  disagree** are kept in full and reported rather than resolved by picking one.
  Measured on panel imagery: brightness, JPEG q30, blur and a half-resolution round
  trip give a distance of 0–1; two different panels give 16–22.
- **Open Images and GitHub fetchers.** `kind="openimages"` (FiftyOne partial download
  — the only practical way to pull a class subset of a 500 GB release) and
  `kind="github"` (pinned archives and release assets; an unpinned default-branch
  download is refused as irreproducible). The Open Images entry declares **zero
  positive classes on purpose**: OIv7 has no industrial electrical classes, its
  nearest neighbours are domestic light switches and sockets, and its value here is as
  **hard negatives** so the detector does not fire `push_button` on every round button
  in frame. The GitHub entry is a working fetcher with **no verified source behind
  it** — GitHub was searched and returned nothing usable, and naming a dataset would
  be a fake citation.

### Auto-annotation (Task 4)

- **`training/electrical/refine.py`** — SAM2 box refinement. Open-vocabulary
  detectors return loose boxes that include DIN rail, neighbouring modules and wire
  looms, and correcting a loose box costs a labeller as much as drawing a new one —
  which destroys the speed-up that justifies auto-annotation. SAM2 is used as a
  promptable segmenter: each detector box becomes a box prompt and the mask's tight
  bounds replace it. Prefers SAM2 via Ultralytics, falls back to SAM 1 / MobileSAM,
  then native `sam2`.
- **Four guards, because a "tighter" box can be worse than a loose one.** Rejects a
  refinement that grows past 1.6× area (SAM segmenting the whole DIN-rail row — the
  modules are visually continuous), collapses below 0.35× (only the toggle lever),
  drifts more than 0.35 of the diagonal (the neighbouring device), or lands on an
  aspect ratio the class's taxonomy prior rules out. A failed guard keeps the original
  box. The manifest reports accept rate and mean IoU shift, so "is SAM helping on my
  imagery?" has a number.

### Model selection (Task 5)

- **`training/electrical/bench.py`** — runtime measurement and automatic selection.
  Ranking on mAP alone always picks the largest model, which for a CPU ONNX
  deployment is usually wrong. Now measures p50/p95/p99 latency, FPS, peak RSS delta
  and parameter count, and `select_best()` scores accuracy against speed with a **hard
  latency budget that disqualifies rather than penalises** — then prints what it traded
  away, so a human can overrule it. Timing discards warmup, requires real images
  (detector latency is data-dependent through NMS), and records the environment and
  thread count. `cli profile` measures one model; `cli bench` trains and picks.

### Hyperparameter optimisation (Task 6)

- **`training/electrical/hpo.py`** — Optuna search over learning rate, batch size,
  image size, optimizer, LR schedule, warmup, weight decay, patience and the full
  augmentation block. Runs the hand-tuned defaults first as a reference, and says so
  when the search fails to beat them by more than noise.
- **`fliplr`/`flipud` are held at 0 and never sampled by default.** A search
  maximising val mAP on a small set will switch horizontal flip on because it looks
  like free augmentation, and produce a model that has learned mirrored nameplates are
  normal. `--no-respect-domain-priors` searches them anyway; the flag exists so the
  decision is deliberate.
- **Pruning is real, not claimed.** A new `on_fit_epoch_end` callback path in
  `train()` reports intermediate mAP to the median pruner. `TrainingAborted` is the
  documented exception that propagates out of `train()` so a pruned trial is not
  reported as a training failure.
- Refuses to waste GPU hours: pointed at a dataset where `cli gap` reports classes
  with zero annotations, it says so up front.

### Export artifacts (Task 7)

- Bundles now carry **`metrics.json`** (headline + per-class accuracy, confusion
  matrix, training curves, runtime, and the caveats that make the headline readable)
  and an **`artifacts/`** directory: confusion matrix, PR/F1/P/R curves, loss curves,
  `results.csv`, `args.yaml`. Harvested from the Ultralytics run, and **rendered from
  `results.csv` when Ultralytics did not plot them** (RT-DETR runs, `plots=False`).
  The confusion matrix is rendered from our own evaluation, so its axes are readable
  device names rather than integer indices, and only active classes are plotted.
  `install_bundle` carries the evidence with the weights — a deployed model that
  cannot be audited is the problem this solves.

### Panel intelligence (Task 9)

- **`rtsp_backend/electrical/risk.py`** — aggregate risk level, in the report, the
  API and the PDF. Per-finding severities existed; nothing combined them, so one
  important finding and twelve looked alike at a glance.
- **It returns `unknown`, never `low`, when there is no basis to score** — no model
  loaded, nothing detected, or more than half the detections unidentified. "We found
  nothing wrong" and "we could not look" read identically while meaning opposite
  things, and somebody may decide not to open a cabinet based on this output.
- The score is the sum of its listed `drivers` — no hidden terms, no learned weights.
  Detection quality is itself a driver. Too few devices means an absence is not
  treated as evidence. A low score with low confidence explicitly does **not** claim a
  clean panel. It adds no findings of its own; it only weighs what the rule engine
  found.

### Performance (Task 11)

- **Batch inference.** `recognize_batch()` / `infer_batch()` on the industrial
  recognisers, a genuinely batched forward pass for `industrial_ultralytics`, an honest
  sequential fallback elsewhere, and `supports_true_batching` in `status()` so a
  throughput claim is checkable. A backend returning the wrong number of results is
  **refused** rather than paired up positionally, because that misattributes one
  panel's detections to another image. `POST /api/panel/analyze/batch` (50-image cap,
  per-file rejection) and `cli analyse-batch`. Deliberately **not** applied to the RTSP
  path: a cabinet does not change between frames, so buffering to fill a batch adds
  latency for nothing.

### Fixed

Three bugs, all caught by the tests written alongside the code:

- **Deduplication missed every cross-split leak.** Exact-duplicate groups were emitted
  immediately and their members excluded from the near-duplicate pass, so with A and B
  byte-identical in train and C a re-compressed copy of A in val, C could not link to A
  and the leak reported as zero. Both relations now feed one connected-components pass.
- **`_pct` was off by one rank.** `round(x + 0.5)` overstated every percentile whenever
  `p/100 × N` landed on an integer — the p50 of 1..10 came back as 6.0. Now
  `ceil(p/100 × N)`.
- **`train.py` and `export.py` each wrote `classes.json` differently**, so the
  taxonomy version in one was already stale. `train.py` now delegates, and a test
  asserts the two agree.

### Tests

498 → 688 passing. New suites for deduplication (27), refinement guards (27),
benchmarking and selection (33), HPO search space (35), risk assessment (30) and
batch inference (23), plus export-artifact coverage in the pipeline suite.

## [5.2.0] — Electrical component detection: real datasets, real pipeline

The component-recognition engine shipped in 5.1.0 with everything except a
trained model. This release builds the machinery that produces one, and — more
usefully — establishes exactly what it will cost, with measured numbers instead of
optimism. No existing feature was removed or rewritten.

### The dataset question, answered

- `training/electrical/datasets.py` **`SOURCES` no longer contains placeholders.**
  Every Roboflow entry is a verified `workspace/project/version` with its licence,
  image count and the per-class instance counts read off the upstream project. A
  test now fails if a `<placeholder>` locator returns.
- `plan()` is a **forecast, not a wish list**: it sums the measured per-class counts
  and reports which classes will be reliable, weak, untrainable, or have no public
  data at all.
- **The finding:** public data does not cover this taxonomy. ~3,500 usable images
  and ~5,300 instances across all sources; **6 of 54** classes reach the
  300-instance reliability bar; **6 priority classes have zero public instances**
  (VFD, SMPS, busbar, DIN rail, cable duct, emergency stop). Two sources are
  recorded and **deliberately excluded** with reasons (thermal imagery; consumer
  push buttons that would generate false positives) so they are not rediscovered
  and merged by mistake. Three further traps are annotated in the registry,
  including one dataset that is a single panel photographed 256 times.
- `requirements_report()` / `cli gap` states the shortfall in actionable units:
  which classes are missing, how many annotations are required, how many
  photographs that implies, and **where on site each missing class is found**.
  Exits non-zero while a priority class has no data, so CI can gate a release.

### New

- **`training/electrical/download.py`** — real fetchers (Roboflow SDK with a REST
  fallback, Kaggle CLI, URL archives), YOLO layout normalisation, and remapping
  onto the canonical taxonomy in one step. Failures name the fix (missing API key,
  un-versioned upstream project) instead of producing an empty dataset that looks
  like success. Archive extraction rejects path traversal and link members.
- **`training/electrical/split.py`** — 80/10/10 splitting **grouped by capture**.
  A random image-level split leaks near-duplicate framings of the same cabinet into
  validation, which is the standard way an industrial detector reports mAP 0.95 and
  then fails on site. Group keys are derived from filenames (source prefixes,
  Roboflow `_jpg.rf.<hash>` mangling, augmentation verbs, shot counters) or supplied
  via `groups.json`. Assignment is quota-driven rarest-class-first so scarce classes
  still reach val. The report names `leaking_groups` and
  `classes_absent_from_val` — classes silently excluded from mAP.
- **`training/electrical/autolabel.py`** — model-assisted pre-labelling for human
  *correction*, using a trained checkpoint when one exists and zero-shot OWLv2 /
  Grounding DINO before that. Per-image verdicts (`auto` / `review` / `uncertain` /
  `empty`) and a worst-first review queue. Boxes the model cannot classify are
  written as `unknown_industrial_component`, never guessed. Ships the full human
  annotation guide (`cli labelguide`).
- **`training/electrical/export.py`** — `best.pt` + `best.onnx` + `labels.txt` +
  `classes.json` + `model_card.json` bundles, with verification that re-reads the
  bundle the way the runtime will: rejects a numeric `labels.txt`, a
  labels/classes disagreement, a reordered label space, and an ONNX head whose class
  count does not match the labels. `install_bundle` **refuses** a bundle that would
  mislabel. TensorRT is documented rather than automated, because an engine is not
  portable across GPU/driver/TensorRT versions.
- **`POST /api/panel/analyze`** — the specified contract
  (`components[].class/confidence/bbox`, xyxy absolute pixels) plus the panel report
  (detected / missing / unknown components, confidence, annotated image). Additive:
  `/api/panels/analyze` (plural) is unchanged and both run the same engine. Also
  `GET /api/panel/classes` and `GET /api/panel/model`.
- **`docs/ELECTRICAL_MODEL_TRAINING.md`** and
  **`docs/PRODUCTION_DEPLOYMENT.md`**.
- CLI subcommands: `download`, `split`, `gap`, `autolabel`, `labelguide`, `export`,
  `verify`, `tensorrt`.

### Taxonomy 5.1

- Appended one class, **`circuit_breaker`** ("type unspecified"), as an honest home
  for the many public datasets that label every protective device "circuit breaker"
  and for medium-voltage VCB/SF6 breakers with no LV equivalent. `resolve()` maps a
  bare "circuit breaker" here and **never** to MCB/MCCB/ACB/RCCB — it does not
  invent specificity the label lacks — while "miniature circuit breaker" still
  resolves to `mcb`.
- Append-only respected: indices 0–52 are unchanged, so a taxonomy-5.0 checkpoint
  stays valid and `verify` reports it as a valid prefix.
- `models/components/labels.txt` is **not** shipped (it is an export artefact); the
  runtime's rejection of a numeric labels file is unchanged.

### Fixed

Two bugs in the new splitter, both caught by its own tests before release:

- The shot-counter heuristic stripped trailing digits without requiring a
  separator, so `panel12` became `panel`, **every panel collapsed into one capture
  group, and an entire dataset would have landed in a single split.** A separator or
  parentheses is now mandatory.
- Pooling input splits could silently overwrite same-named files from different
  splits, losing both the image and its labels. Collisions are now detected,
  renamed and reported.

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
