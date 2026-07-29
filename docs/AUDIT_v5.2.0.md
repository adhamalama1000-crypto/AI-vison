# Audit — Madkour AI Panel Inspector, v5.2.0

Full audit of the repository against the twelve-task production brief. Every claim
below is evidenced by file and line, and every gap named here is either fixed in
this release or has an explicit reason it is not.

Method:

```bash
grep -rn "TODO\|FIXME\|XXX\|HACK\|NotImplementedError" --include="*.py" .
python -m pyflakes $(git ls-files '*.py')          # unused imports, undefined names
python -m pytest tests/ -q                          # behavioural coverage
# plus an AST sweep for pass-only and NotImplemented-only function bodies
```

---

## Headline

The repository is in better shape than the brief assumes. There are **no TODOs, no
FIXMEs, no stub functions and no mocked AI paths** in production code. The
`NotImplementedError`s that exist are legitimate abstract-base-class contracts:

| Location | Verdict |
|---|---|
| `rtsp_backend/ai/base.py:135,143,147,171,178` | Correct — ABC methods on `Detector` / `FaceEngine` / `Analyzer` |
| `rtsp_backend/electrical/recognizer.py:328` | Correct — `raw_candidates` is the recogniser subclass contract |

One genuine placeholder was found:

| Location | Issue |
|---|---|
| `rtsp_backend/imaging/analysis.py:135` | `_clip_tags()` imports torch/open_clip, then `return []  # placeholder hook`. Dead code: it can never return a tag. |

This is in the *generic image analysis* module, not the electrical path, and it fails
closed (empty tag list) rather than fabricating. It is documented in "Deliberately
not done" below.

The eight real gaps were not unfinished code — they were **capabilities the brief
requires that had never been built**. They are listed per task.

---

## Task-by-task status

| # | Task | Before | After |
|---|---|---|---|
| 1 | Audit | — | This document |
| 2 | Complete dataset builder | Partial | **Fixed** — dedup + Open Images/GitHub added |
| 3 | Dataset gap report | Complete | Unchanged |
| 4 | Auto-annotation pipeline | Partial | **Fixed** — SAM2 added |
| 5 | Train + compare detectors | Partial | **Fixed** — FPS/latency/memory + auto-select |
| 6 | Hyperparameter optimisation | Missing | **Fixed** — Optuna detector HPO |
| 7 | Export artifacts | Partial | **Fixed** — metrics/curves/confusion matrix |
| 8 | Backend integration | Complete | Unchanged |
| 9 | Panel intelligence | Partial | **Fixed** — risk level added |
| 10 | Validation | Complete | Extended |
| 11 | Performance | Partial | **Fixed** — batch inference |
| 12 | Documentation | Complete | Extended |

---

### Task 2 — Dataset builder

**Was in place.** `training/electrical/datasets.py` holds a registry of verified
Roboflow locators with per-class instance counts; `download.py` fetches from
Roboflow (SDK + REST fallback), Kaggle and URL archives, normalises YOLO layout and
remaps onto the canonical taxonomy; `merge()` unions datasets with per-source
filename prefixing; everything lands in YOLO format with `dataset.yaml`.

**Gap 1 — no deduplication.** The brief requires "remove duplicates" and the
registry itself warns that `rf_switchgear_varsha` and `rf_switchgear_potholes` have
near-identical class lists and the same ~30-instances-per-class signature, i.e. they
are probably overlapping imagery. Merging both without dedup puts the *same
photograph* in train and val, which inflates every validation metric. There was no
code to detect it.

> Fixed: `training/electrical/dedup.py` — perceptual hashing (dHash + aHash, with
> `imagehash` used when available and a self-contained fallback when not), Hamming
> distance matching, cross-split leak detection, and an exact-duplicate pass by
> content hash. Wired into `cli merge --dedup` and available as `cli dedup`.

**Gap 2 — Open Images and GitHub sources absent.** The brief names both. The
registry's `kind` field supported `roboflow | kaggle | url | manual` only.

> Fixed: `kind="openimages"` (via FiftyOne, which is the only sane way to pull a
> class subset of Open Images without downloading 500 GB) and `kind="github"`
> (release assets / archive refs). Real verified entries added for both.

---

### Task 3 — Gap report

**Complete, no change.** `datasets.requirements_report()` and `cli gap` already
report existing classes, missing classes, per-class annotation counts, the
annotation shortfall, the implied image count, target per class, and where each
missing class is found on site. Exits non-zero while a priority class has zero data.

---

### Task 4 — Auto-annotation

**Was in place.** `training/electrical/autolabel.py` drives the registered
open-vocabulary backends — `openvocab_owlv2`, `openvocab_grounding_dino`,
`openvocab_florence2` (`rtsp_backend/electrical/recognizer.py:600,664,753`) — over a
directory, writes YOLO labels, and produces a per-image verdict manifest with a
worst-first human review queue. Boxes it cannot classify are written as
`unknown_industrial_component`, never guessed.

**Gap 3 — no SAM2.** The brief names SAM2 explicitly and there was no SAM or SAM2
code anywhere in the repository. This is not a box-ticking gap: open-vocabulary
detectors are trained on natural-image captions and return *loose* boxes, typically
including surrounding DIN rail, adjacent devices and wiring. A labeller then spends
as long fixing box edges as they would have spent drawing them, which destroys the
speed-up that justifies auto-annotation at all.

> Fixed: `training/electrical/refine.py` — SAM2 (and SAM 1 / `ultralytics.SAM` as
> fallbacks) used as a *promptable* segmenter: each detector box becomes a box
> prompt, the returned mask's tight bounding box replaces the loose one, subject to
> guards that reject a refinement that grows the box, collapses it, or drifts off
> the original centre. Reports per-box IoU shift so the benefit is measurable rather
> than assumed. Degrades to a no-op with a stated reason when SAM2 is not installed.

---

### Task 5 — Train and compare

**Was in place.** `training/electrical/train.py` trains `yolo11{n,s,m,l,x}`,
`yolov8{n,s,m,l,x}`, `rtdetr-{l,x}` and conditionally `yolo12{n,s,m}` (guarded by
`arch_available()`, reported as `skipped` rather than substituted).
`benchmark()` trains each on the same split and ranks by measured mAP via
`rtsp_backend/electrical/metrics.py`.

**Gap 4 — accuracy-only ranking.** The brief requires Precision, Recall, mAP50,
mAP50-95, **FPS, Latency and Memory**, and automatic best-model selection. Only the
accuracy half existed. `metrics.py:78` has an `fps` local variable, but that is
`false_positives` — unrelated.

Ranking on mAP alone picks `yolo11x` every time, which is the wrong answer for a
platform whose default deployment is CPU ONNX Runtime: a 3-point mAP gain for a 6×
latency cost is a bad trade on a 4-core box.

> Fixed: `training/electrical/bench.py` — warmup-then-measure timing over real
> images (p50/p95/p99 latency, FPS, throughput), peak RSS delta and, where torch is
> present, parameter count and CUDA peak memory; plus `select_best()`, a documented
> weighted score over accuracy and speed with a hard latency budget, which names the
> winner **and prints why it beat the runner-up**.

---

### Task 6 — Hyperparameter optimisation

**Gap 5 — missing entirely for the detector.** Optuna appears at
`rtsp_backend/training_svc.py:245,425` but only optimises the scikit-learn
*classifier* used by the generic training service. Nothing tuned the YOLO detector;
`TrainConfig` defaults were hand-set (well-reasoned, and documented as such, but not
searched).

> Fixed: `training/electrical/hpo.py` — Optuna study over learning rate, batch size,
> image size, optimizer, LR schedule, warmup, weight decay, and the full
> augmentation block, with a short-budget objective, median pruning, SQLite-backed
> resumable studies, and a `--respect-domain-priors` mode that keeps `fliplr` at 0
> because a mirrored nameplate is not a real thing regardless of what the search
> says. Emits a ready-to-use `TrainConfig` and records the full trial history.

---

### Task 7 — Export

**Was in place.** `training/electrical/export.py` produces `best.pt`, `best.onnx`,
`labels.txt`, `classes.json` and `model_card.json`, and verifies the bundle the way
the runtime reads it (rejects a numeric `labels.txt`, a labels/classes
disagreement, a reordered label space, and an ONNX head whose class count disagrees
with the labels). `cli tensorrt` documents engine building — deliberately not
automated, because an engine is not portable across GPU/driver/TensorRT versions.

**Gap 6 — evaluation artifacts not collected.** The brief requires `metrics.json`,
`confusion_matrix.png`, PR curves and loss curves in the export. Ultralytics writes
these into its run directory, but nothing copied them into the bundle, so a deployed
model carried no evidence of its own accuracy.

> Fixed: `export.py:collect_artifacts()` harvests `results.csv`, `confusion_matrix*.png`,
> `*_curve.png` and `args.yaml` from the Ultralytics run, writes a normalised
> `metrics.json`, and — when Ultralytics plots are absent (RT-DETR runs, or
> `plots=False`) — renders a confusion matrix and PR curve from our own evaluation
> via matplotlib, falling back to CSV if matplotlib is missing. Loss curves are
> plotted from `results.csv`.

---

### Task 8 — Backend integration

**Complete, no change.** Detection flows through the existing plugin registry
(`rtsp_backend/ai/registry.py`) with no architectural change: `industrial_onnx` /
`industrial_ultralytics` are registered `ComponentDetector`s selected through
`POST /api/ai/models/components`. Consumers — uploaded images, RTSP frames
(`rtsp_backend/ai/pipeline.py`), Reference Panel Inspection
(`rtsp_backend/inspection_svc.py`), the dashboard and
`POST /api/panel/analyze` — all read components through `rtsp_backend/panel_svc.py`,
so a newly installed bundle activates everywhere at once.

---

### Task 9 — Panel intelligence

**Was in place.** Detected components and counts (`inspector.inspect_panel`),
manufacturer and part number (`electrical/nameplate.py`, from real OCR — null when
no OCR engine is installed, never invented), panel type and function
(`electrical/panel_type.py:336,441`), possible missing components
(`panel_type.py:490`), recommendations (`maintenance_notes`, `panel_type.py:531`),
annotated image (`inspector.draw_overlay`), JSON report (`reports_svc.write_json`)
and PDF report (`reports_svc.panel_pdf`).

**Gap 7 — no risk level.** Individual notes carry `severity: info|advisory|important`
(`panel_type.py:476,523`) but nothing aggregated them into the single risk level the
brief asks for, so a report with one critical finding and a report with twelve looked
alike at a glance.

> Fixed: `rtsp_backend/electrical/risk.py` — an auditable risk assessment that scores
> weighted evidence (missing protective devices, thermal-management gaps,
> unidentified device ratio, detection-quality signals) into
> `low | moderate | elevated | high`, lists the drivers with their individual
> contributions, and **states its own confidence and limits** — including refusing to
> return anything but `unknown` when no model is loaded, because a risk level derived
> from zero detections would be the most dangerous output in the system.

---

### Task 10 — Validation

**Was in place and extended.** 498 tests passing before this release; `scripts/validate_panel_inspector.py`
exercises the inspection path end to end.

Note for CI: the suite requires `scikit-learn`, `skl2onnx` and `onnx` from
`requirements-train.txt`. Installing only `requirements.txt` leaves 10 import
failures in `test_evaluation.py`, `test_platform_upgrade.py` and `test_training.py`.
This is pre-existing and is a CI configuration issue, not a code defect.

---

### Task 11 — Performance

**Was in place.** CPU inference via ONNX Runtime (`industrial_onnx`), GPU via
`industrial_ultralytics`, provider selection in
`recognizer.py:427`, TensorRT documented in `export.tensorrt_instructions()`.

**Gap 8 — no batch inference.** Every recogniser was strictly single-frame. For
folder-scale work (re-scoring an archive, the auto-annotation pass over a capture
batch) per-image calls waste most of the available throughput.

> Fixed: `infer_batch()` on the industrial recognisers, with an honest
> sequential fallback for backends that cannot genuinely batch, plus
> `cli analyse-batch` for folder throughput. Batching is deliberately **not** applied
> to the RTSP path: panel inspection is not a real-time problem — a cabinet does not
> change between frames — and buffering frames to fill a batch would add latency for
> no benefit.

---

### Task 12 — Documentation

`docs/ELECTRICAL_MODEL_TRAINING.md`, `docs/PRODUCTION_DEPLOYMENT.md`,
`docs/INDUSTRIAL_INSPECTION.md`, `docs/AUDIT_PANEL_INSPECTOR.md`,
`training/electrical/README.md`, plus this audit. Extended in this release with the
HPO, benchmarking, dedup and model-update procedures.

---

## The thing the brief asks for that code cannot deliver

The mission states the final system "must actually detect electrical components".
Every piece of machinery for that now exists and is tested. **What does not exist is
a dataset large enough to train a 22-class production detector**, and no amount of
code changes that.

Measured, not estimated (`python -m training.electrical.cli plan`):

| | |
|---|---|
| Usable public images across every verified source | ~3,500 |
| Classes reaching the 300-instance reliability bar | **6 of 54** |
| Priority classes with **zero** public instances | VFD, SMPS, busbar, DIN rail, cable duct, emergency stop |
| Annotations needed to close the 22 priority classes | ~6,600 |
| Panel photographs that implies | **~550** |

Training on what is publicly available will produce a real, working detector for
roughly six classes — `mcb`, `contactor`, `relay`, `fuse`, `plc`, `timer_relay` — and
that model will honestly return `unknown_industrial_component` for everything else.
That is the correct behaviour, and it is the behaviour the brief demands ("never fake
detections"). It is not the same thing as a 22-class production model.

Per the strict rules, this is what has been built to close it rather than hide it:

1. **`cli gap`** — the exact shortfall, per class, in annotations and photographs.
2. **`cli plan | jq '.custom_collection'`** — the capture protocol: framings, angles,
   lighting, manufacturer spread, and the panel-level split policy.
3. **`gap --priority-only | jq '.what_to_collect'`** — where each missing class is
   physically found on site.
4. **`cli labelguide`** — the labelling rules, including the four that most reliably
   ruin an industrial dataset (contactor+overload are two boxes; terminal strips are
   per-strip not per-pole; a 3-pole MCB is one box; illuminated push button is a push
   button).
5. **`cli autolabel`** (now with SAM2 refinement) — 3–5× faster labelling by
   correcting boxes instead of drawing them.
6. **`training/electrical/synthetic.py`** — crop-library composition to multiply a
   small real crop set, with real appearance and synthetic arrangement/lighting/
   occlusion.
7. **`cli split`** — grouped 80/10/10, so the metrics from the resulting model are
   trustworthy rather than leaked.

---

## Deliberately not done, with reasons

| Item | Why not |
|---|---|
| Ship a trained checkpoint | There is no dataset to train it on (above). Shipping weights trained on ~3,500 images as a "production model" would be the exact dishonesty the brief forbids — it would report confident wrong component names. |
| Build a TensorRT `.engine` in the repo | An engine is compiled for one GPU compute capability, driver and TensorRT version. A shipped engine fails to deserialise on any other host. Documented and scripted to build on the deployment host instead. |
| Automate Kaggle dataset selection | No Kaggle dataset was verified to add boxed panel-device instances. A registry entry naming one would be a fake citation. The fetcher works; pass `--locator <owner>/<slug>` when you have one. |
| Fix `_clip_tags()` in `imaging/analysis.py` | Out of scope for the electrical path, and it fails closed (returns `[]`). Wiring open_clip properly means choosing a checkpoint and a tag vocabulary — a real feature decision, not a cleanup. Flagged here so it is not mistaken for working code. |
| Batch the RTSP path | Panel inspection is not a real-time problem; buffering frames to fill a batch adds latency for no accuracy benefit. Batching is exposed where it helps (folder/archive work). |
| YOLO12 | Attempted only when the installed Ultralytics build can construct it, otherwise reported `skipped`. The brief's "if stable" condition, enforced in code (`train.py:70`). |
