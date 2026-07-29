# Production Deployment — Electrical Component Detection

Deploying the component detector into the AI Vision Platform: CPU, GPU, ONNX,
TensorRT, and the operational practices that keep the results trustworthy.

Training the model: `docs/ELECTRICAL_MODEL_TRAINING.md`.

---

## Before you deploy: is the model fit to deploy?

The platform will happily serve a model trained on 200 images. It will report
confident component names with 0.9 confidence and be wrong most of the time, and
the failure will not look like a failure. So gate the deployment:

```bash
# 1. Does the model have data behind it? Exits non-zero if a priority class has none.
python -m training.electrical.cli gap --root data/final --priority-only

# 2. Are the labels and the graph consistent? Exits non-zero on any mismatch.
python -m training.electrical.cli verify --bundle models/components

# 3. What is the measured per-class accuracy — not the headline mAP?
python -m training.electrical.cli eval --root data/final --backend industrial_onnx
```

Deployment checklist:

- [ ] `verify` reports `ok: true`.
- [ ] Per-class recall is acceptable **for each class you will act on**. Any class
      the gap report calls weak or untrainable is unvalidated, whatever the mean says.
- [ ] `classes_absent_from_val` from the split report is empty, or you have written
      down which classes are unmeasured and told the users.
- [ ] Per-class thresholds came from `cli tune`, not from guesswork.
- [ ] `model_card.json` is deployed alongside the weights.
- [ ] Dataset licences are honoured — CC BY 4.0 requires attribution; keep
      `download_manifest.json` with the model.

Deploying a model that fails these is a decision, not an accident. Make it
explicitly and tell the people reading the reports.

---

## Bundle layout

A deployment needs one directory:

```
models/components/
├── best.onnx          # CPU/GPU inference graph
├── best.pt            # Ultralytics checkpoint (GPU, fine-tuning)
├── labels.txt         # one canonical class id per line, training index order
├── classes.json       # authoritative label order (preferred by the runtime)
├── model_card.json    # what was trained, on what, and what it cannot do
├── metrics.json       # measured accuracy: headline, per-class, curves, runtime
└── artifacts/         # confusion matrix, PR/F1 curves, loss curves, args.yaml
```

`metrics.json` and `artifacts/` are the audit trail. They are not optional in
practice: without them, nobody can answer "what was this model's recall on
contactors?" after the training box has been recycled. `install_bundle` copies them
alongside the weights for exactly that reason.

The runtime resolves labels in this order: explicit `labels` param →
`classes.json` → `labels.txt` (**only if it does not look like bare integers**) →
the taxonomy's `CLASS_ORDER`. That numeric-file rejection exists because a previous
release shipped `labels.txt` as the lines `0`…`9` and every detection came back
named `"0"`.

`best.onnx` and `labels.txt` must agree on class count. `verify` enforces it; a
mismatch shifts every label by one and is invisible until someone notices a
contactor being reported as a relay.

Install a bundle:

```bash
python -m training.electrical.cli export --weights best.pt --out dist/model --install
# or
cp dist/model/* models/components/
```

---

## Choosing a runtime

| Backend | Weights | Hardware | When |
|---|---|---|---|
| `industrial_onnx` | `best.onnx` | CPU (or GPU via `onnxruntime-gpu`) | **Default.** Portable, no torch dependency, adequate for on-demand analysis. |
| `industrial_ultralytics` | `best.pt` / `.engine` | GPU | Continuous RTSP analysis, or when TensorRT is in play. Pulls torch. |
| `openvocab_owlv2` / `openvocab_grounding_dino` | pulled by `transformers` | GPU strongly preferred | Zero-shot bootstrap before a checkpoint exists, and for auto-labelling. Orders of magnitude slower per image. |
| `null_components` | — | — | Honest disabled state. |

Select at runtime:

```bash
curl -X POST localhost:8000/api/ai/models/components \
     -H 'Content-Type: application/json' \
     -d '{"backend_id": "industrial_onnx"}'
```

Keep `best.onnx` installed even when running TensorRT, so the platform degrades to
CPU instead of failing when the GPU is unavailable.

### CPU sizing

Rough guidance for `yolo11s` at 960px, ONNX Runtime, modern x86:

| Cores | Latency / image | Suitable for |
|---|---|---|
| 2 | ~1.5–3 s | Uploaded images, occasional analysis |
| 4 | ~0.8–1.5 s | Uploads plus one RTSP camera at low frame rate |
| 8+ | ~0.4–0.8 s | Several cameras at a few frames per minute |

Measure on your hardware; these are starting points, not commitments. Panel
inspection is not a real-time problem — a cabinet does not change between frames —
so analysing one frame every few seconds per camera is normally right, and is far
cheaper than continuous inference.

Tune ONNX Runtime threads to the container's real CPU allocation:

```bash
export OMP_NUM_THREADS=4
export ORT_DISABLE_ALL_OPTIMIZATION=0
```

Over-subscribing threads inside a CPU-limited container makes inference *slower*,
and this is a common and confusing production regression.

### GPU

```bash
pip install ultralytics onnxruntime-gpu
```

Then select `industrial_ultralytics`. Verify the GPU is actually visible to the
process (`nvidia-smi` inside the container; `--gpus all` plus the NVIDIA Container
Toolkit for Docker) — Ultralytics silently falls back to CPU otherwise, and you get
CPU latency while believing you have GPU.

### TensorRT

```bash
python -m training.electrical.cli tensorrt
```

Build **on the deployment host**:

```bash
yolo export model=models/components/best.pt format=engine imgsz=960 half=True device=0
# or
trtexec --onnx=models/components/best.onnx \
        --saveEngine=models/components/best.engine --fp16 --workspace=4096
```

An engine is compiled for one GPU compute capability, one TensorRT version and one
driver. Build it in CI and ship the file and it will fail to deserialise on any
other host — that is why this is documented rather than automated.

**Re-measure after conversion.** FP16 changes outputs slightly; INT8 changes them
materially. Run `cli eval` against the engine rather than assuming parity with the
ONNX model. Expect roughly 2–4× over ONNX Runtime CPU for `yolo11s` at 960px, but
measure — the ratio depends heavily on batch size and GPU contention.

---

## The API

### `POST /api/panel/analyze`

```bash
curl -X POST localhost:8000/api/panel/analyze -F file=@panel.jpg
```

```json
{
  "components": [
    {
      "class": "mcb",
      "class_name": "MCB (Miniature Circuit Breaker)",
      "confidence": 0.94,
      "bbox": [412.0, 188.5, 448.0, 262.0],
      "is_unknown": false,
      "manufacturer": "Schneider Electric",
      "part_number": "A9F74216",
      "category": "protection"
    }
  ],
  "component_total": 1,
  "image": {"width": 1920, "height": 1080, "source": "upload"},
  "model": {"loaded": true, "backend": "industrial_onnx", "engine_version": "5.0"},
  "bbox_format": "xyxy_absolute_pixels",
  "annotated_image": "panels/panel_1730000000000.jpg",
  "duration_ms": 812.4,
  "report": { "...": "see below" }
}
```

`bbox` is `[x1, y1, x2, y2]` in **absolute pixels** of the submitted image, with
`image.width`/`image.height` alongside so clients normalise without guessing.
Branch on `class` (the stable taxonomy id), display `class_name`.

Parameters:

| Parameter | Default | Effect |
|---|---|---|
| `file` | — | Image upload. Omit to grab a frame from a camera. |
| `camera_id` | active camera | RTSP source when no file is given. |
| `report` | `true` | Include the full panel report. `false` for the bare list. |
| `annotate` | `true` | Render and persist the annotated image. |
| `persist` | `true` | Record the analysis in the reports table. |
| `min_confidence` | `0.0` | Post-filter. Can only tighten the engine's per-class gating. |

The report section carries what the brief asks for:

```json
{
  "detected_components": [{"class": "mcb", "count": 12, "mean_confidence": 0.91}],
  "missing_components":  [{"class": "overload_relay", "reason": "...", "severity": "..."}],
  "unknown_components":  {"count": 2, "items": [{"bbox": [], "confidence": 0.41}]},
  "confidence": {"mean": 0.88, "min": 0.41, "max": 0.98,
                 "identified": 12, "unknown": 2, "identification_rate": 0.857},
  "panel": {"type": "motor_control_centre", "confidence": 0.79}
}
```

`missing_components` is an **inference from the panel type, not a measurement** — a
panel classified as a motor-control centre with no overload relay is probably
missing one. Each entry carries its own reasoning and confidence. Do not present it
to users as a detection.

### Risk level

The report also carries an aggregate risk assessment:

```json
{
  "risk": {
    "level": "elevated",
    "score": 6.5,
    "confidence": "high",
    "assessable": true,
    "headline": "4 risk indicator(s), 2 important, including 1 safety-critical device not detected. Risk level: ELEVATED (score 6.5).",
    "drivers": [
      {"code": "missing_emergency_stop", "category": "missing_protection",
       "severity": "important", "weight": 4.5,
       "message": "... This is a safety-critical device."}
    ],
    "recommendations": ["PRIORITY: verify by eye whether ..."],
    "limits": ["..."]
  }
}
```

Four properties of this output matter operationally, and they are all deliberate:

1. **`level` is `"unknown"`, never `"low"`, when there is no basis to score.** No
   model loaded, nothing detected, or more than half the detections unidentified all
   produce `unknown` with `assessable: false`. "We found nothing wrong" and "we could
   not look" read identically to somebody skimming a report while meaning opposite
   things — and somebody may decide not to open a cabinet based on this. **Do not
   render `unknown` as a green badge.** Grey or amber; never a pass.
2. **The score is the sum of its `drivers`.** No hidden terms, no learned weights.
   "Why is this elevated?" is answerable from the JSON alone, and the weights in
   `rtsp_backend/electrical/risk.py` are declared engineering judgements you can
   change.
3. **Detection quality is itself a driver.** A panel where 30% of devices came back
   unidentified scores higher *and* carries a `limits` entry saying the
   missing-component findings are weakened — one of the unknowns may be the device
   reported absent.
4. **It adds no findings of its own.** It only weighs what the detector and rule
   engine already found. A clean panel scores 0.0 with an empty `drivers` list.

`confidence` is the assessment's confidence in *itself* (`high`/`moderate`/`low`),
lowered by unidentified detections, low mean confidence, too few devices to treat an
absence as evidence, or an unclassifiable panel type. When it is `low` the headline
says so in capitals, and a low score explicitly does **not** claim a clean panel.

The PDF renders the level first, colour-coded, with the recommendations and the
limits — `unknown` renders grey, not green.

### Report formats

```bash
curl -X POST 'localhost:8000/api/panel/analyze?csv=true&pdf=true' -F file=@panel.jpg
```

```json
{"exports": {
  "json": "reports/panel_1730000000000.json",
  "pdf": "reports/panel_1730000000000.pdf",
  "csv_components": "reports/panel_1730000000001.csv",
  "csv_summary": "reports/panel_summary_1730000000002.csv"
}}
```

Fetch any of them from `/api/media/<path>`.

| Format | Contents |
|---|---|
| JSON | The complete inspection result — everything the engine produced |
| PDF | Risk level (colour-coded), annotated image, counts, recommendations, limits |
| `csv_components` | **One row per detected device**: class, confidence, xyxy, size, position, row, manufacturer, part number, nameplate text, purpose |
| `csv_summary` | Sectioned: panel, risk, bill of materials, possible-missing, risk drivers, recommendations, maintenance notes, limits |

The per-component CSV is the one to paste next to an as-built bill of materials and diff
the counts. It is one row per *device* rather than per component type, because the
position and nameplate columns are per device and are the reason to open it in a
spreadsheet at all.

Both CSVs are written through Python's `csv` module, so a nameplate string containing a
comma, a quote or a newline is quoted correctly instead of corrupting the row — OCR
output routinely contains all three, and the corruption is invisible until somebody
opens the file.

The summary CSV's `risk`/`assessable` row carries `false means no basis to score — NOT
a pass` in its detail column, so a reader of the CSV alone is not misled by a
`level=unknown`.

### `POST /api/panel/analyze/batch`

For folder-scale work: re-scoring an archive, or checking a capture batch.

```bash
curl -X POST localhost:8000/api/panel/analyze/batch \
     -F files=@p1.jpg -F files=@p2.jpg -F files=@p3.jpg \
     -G --data-urlencode 'batch_size=8'
```

Per-image results are identical to `/analyze` — same engine, same thresholds, same
post-processing gate. Only throughput differs, and only on a backend with a real
batched forward pass. `true_batching` in the response says which you got; when it is
`false` the images were processed sequentially and a `note` says so, so a timing
number is never presented as a batching win when it is not one.

Capped at 50 images per request (unbounded batching is a memory-exhaustion vector).
One undecodable file is reported in `rejected` and the rest of the batch proceeds.
`report=false` is the default because for a 50-image batch the full reports dominate
the response.

**Batching is deliberately not used on the RTSP path.** Panel inspection is not a
real-time problem — a cabinet does not change between frames — so buffering frames to
fill a batch would add latency for no accuracy benefit. Analyse one frame every few
seconds per camera instead.

Same thing from the CLI, with throughput measurement:

```bash
python -m training.electrical.cli analyse-batch --images captures/ \
    --backend industrial_onnx --batch 8
```

### Supporting endpoints

```bash
curl localhost:8000/api/panel/classes   # the label space — read this, do not hardcode
curl localhost:8000/api/panel/model     # is a model loaded, and if not, the remedy
```

`/api/panel/classes` matters because `CLASS_ORDER` is append-only and grows with
each retrain. A client with a hardcoded list silently stops showing new classes.

### Unchanged endpoints

`/api/panel/*` is **additive**. `POST /api/panels/analyze` (plural) keeps its
richer response and PDF report flow, and both routes run the same engine, so there
is one detection path and one set of thresholds. Reference Panel Inspection, the
dashboard, Live view and the RTSP pipeline all consume the components through
`rtsp_backend.panel_svc` and pick up a newly installed model automatically.

---

## Deployment shapes

### Docker

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Ship the model bundle as a mounted volume, not baked into the image: weights
# change on a different cadence from code, and a 50 MB layer per retrain is waste.
VOLUME /app/models/components
EXPOSE 8000
CMD ["python", "run.py"]
```

```bash
docker run -d --name ai-vision \
  -p 8000:8000 \
  -v /srv/ai-vision/models:/app/models \
  -v /srv/ai-vision/data:/app/data \
  --env-file /srv/ai-vision/.env \
  --restart unless-stopped \
  ai-vision:latest
```

GPU: add `--gpus all`, base on `nvidia/cuda:12.4.0-runtime-ubuntu22.04`, and install
a CUDA torch build.

### systemd

```ini
[Unit]
Description=AI Vision Platform
After=network-online.target

[Service]
Type=simple
User=aivision
WorkingDirectory=/srv/ai-vision
EnvironmentFile=/srv/ai-vision/.env
Environment=OMP_NUM_THREADS=4
ExecStart=/srv/ai-vision/.venv/bin/python run.py
Restart=on-failure
RestartSec=5
# The process needs to write only its own data directory.
ProtectSystem=strict
ReadWritePaths=/srv/ai-vision/data /srv/ai-vision/models
PrivateTmp=true
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
```

Put a reverse proxy (nginx/Caddy) in front for TLS. The API is not designed to be
exposed to the internet unauthenticated.

### Persistence

| Path | Contents | Backup |
|---|---|---|
| `data/` | SQLite DB, annotated images, JSON reports, PDFs | Yes — this is the inspection record |
| `models/` | Model bundles | Yes, or keep them rebuildable from a tagged dataset |
| `runs/` | Training artefacts | Optional; regenerable |

---

## Operating it

### Health

```bash
curl localhost:8000/api/panel/model     # model loaded?
curl localhost:8000/api/ai/models       # every backend's status
curl localhost:8000/api/stats           # throughput and events
```

`/api/panel/model` returning `{"loaded": false}` with a `remedy` is the single most
useful check after a deployment — it distinguishes "no model installed" from "model
installed but broken", which look identical in the UI (both show zero components).

### Monitor these, not just uptime

- **Unknown rate** (`report.confidence.identification_rate`). A rising unknown rate
  means the field distribution has drifted from the training set — new panel
  families, a new manufacturer, worse lighting. It is the earliest warning you get,
  and it arrives long before anyone complains.
- **Components per panel** against the as-built bill of materials. Systematic
  under-counting is usually a threshold set too high, or a class the model never
  learned.
- **Inference latency.** A sudden increase usually means GPU fallback to CPU or
  thread over-subscription, not a model change.

---

## Model update guide

The procedure for replacing a deployed model, end to end. Because `CLASS_ORDER` is
append-only, a retrained model is a drop-in replacement — but the steps below are
what stop a "drop-in" from silently mislabelling everything.

### 1. Gather the new data

Every `unknown_industrial_component` returned in production is a labelled example the
model is asking for. Collect those crops, plus captures of whatever `cli gap` still
reports as short.

```bash
python -m training.electrical.cli gap --root data/final --priority-only
python -m training.electrical.cli autolabel --images new_captures/ --out data/round2
# → correct in a labelling tool, working autolabel_manifest.json's review_queue first
```

### 2. Rebuild the dataset

```bash
python -m training.electrical.cli merge --roots data/final data/round2 \
    --dst data/merged_v2 --dedup
python -m training.electrical.cli split --src data/merged_v2_dedup --dst data/final_v2
python -m training.electrical.cli gap --root data/final_v2
```

Deduplicate. New captures of a panel you already have will otherwise land in both
train and val, and the v2 metrics will look better than v1 for reasons that have
nothing to do with the model.

Keep `data/final` — reproducing the *old* model is the only way to attribute a
regression to the data rather than the training run.

### 3. Train and compare on the same footing

```bash
python -m training.electrical.cli train --data data/final_v2/dataset.yaml \
    --arch yolo11s --epochs 120 --device 0

# score the NEW model and the OLD model on the SAME validation split
python -m training.electrical.cli eval --root data/final_v2 \
    --backend industrial_ultralytics \
    --params '{"weights":"runs/electrical/yolo11s/weights/best.pt"}' > dist/eval_v2.json
python -m training.electrical.cli eval --root data/final_v2 \
    --backend industrial_onnx \
    --params '{"weights":"/srv/ai-vision/models/components/best.onnx"}' > dist/eval_v1.json
```

Both on `final_v2`. Comparing v2-on-v2 against v1-on-v1 compares two different
questions and tells you nothing.

### 4. Decide on per-class recall, not headline mAP

Headline mAP moves for uninteresting reasons when the validation set changes — a new
class appearing, or a rare class gaining val instances, shifts the mean without any
model change. Compare the per-class tables in the two `eval` outputs and ask:

- Did any class that was working get **worse**? That is a regression, whatever the
  mean did.
- Did the classes you collected data for actually improve? If not, the labelling is
  more likely at fault than the training.
- Are there classes still absent from val? Those remain unvalidated in v2 too.

### 5. Export, verify, stage

```bash
python -m training.electrical.cli export \
    --weights runs/electrical/yolo11s/weights/best.pt \
    --out dist/model_v2 --imgsz 960 --data data/final_v2/dataset.yaml \
    --eval-json dist/eval_v2.json --notes "v2: +410 field captures, adds VFD/SMPS"

python -m training.electrical.cli verify --bundle dist/model_v2
```

`verify` must report `ok: true` before this goes anywhere near production.

### 6. Install atomically, keeping the rollback

```bash
cd /srv/ai-vision/models
rm -rf components.prev && cp -r components components.prev   # keep the rollback
python -m training.electrical.cli export --weights ... --out dist/model_v2 --install
python -m training.electrical.cli verify --bundle components
curl -X POST localhost:8000/api/ai/models/components \
     -H 'Content-Type: application/json' -d '{"backend_id":"industrial_onnx"}'
curl localhost:8000/api/panel/model     # confirm it loaded, and which weights
```

Copy the whole bundle or none of it. A half-copied bundle — new `best.onnx` with the
old `labels.txt` — is exactly the class-count mismatch that shifts every label by one,
and it is invisible until somebody notices a contactor being reported as a relay.

### 7. Watch the first day

- **Unknown rate** (`report.confidence.identification_rate`) should not get worse. If
  it does, v2 is less confident on real field imagery than the validation split
  suggested — which usually means the new training data was less diverse than it
  looked.
- **Components per panel** against the as-built BOM on a few known panels.
- **Latency**, in case the architecture or image size changed.

### Rollback

```bash
rm -rf /srv/ai-vision/models/components
cp -r /srv/ai-vision/models/components.prev /srv/ai-vision/models/components
python -m training.electrical.cli verify --bundle /srv/ai-vision/models/components
curl -X POST localhost:8000/api/ai/models/components -d '{"backend_id":"industrial_onnx"}'
```

Always `verify` after a rollback, for the same half-copied-bundle reason.

### Adding a class

Append a `ComponentSpec` to `rtsp_backend/electrical/taxonomy.py` — **append only**,
never insert, or every existing checkpoint's class indices become wrong. Give it its
function, role, mounting, geometric priors, aliases and zero-shot prompts; the class
id alone is not enough for the rest of the system to reason about the device. Then
regenerate `models/components/classes.json`, collect data for it, and retrain.

An older bundle with fewer classes stays valid: `verify` recognises a proper prefix of
`CLASS_ORDER` and reports that the newer classes simply cannot be detected by that
checkpoint.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `components: []`, `model.loaded: false` | No bundle installed | Install one; check `/api/panel/model` for the reason |
| Detections labelled `"0"`, `"1"` | `labels.txt` is bare integers | `cli verify`; re-export. The runtime already rejects this file and falls back to the taxonomy |
| Every label off by one | ONNX head class count ≠ label count | `cli verify`; re-export from a checkpoint trained on this label space |
| Excellent mAP, poor on-site accuracy | Capture groups leaked across the split | `cli split` with `--groups`; check `leaking_groups == 0` |
| One class never detected | No validation instances, so never measured | `classes_absent_from_val` in the split report; collect data for it |
| Everything is `unknown` | Thresholds too high, or genuine distribution drift | `cli tune`; if thresholds are right, this is the model telling you it needs data |
| Contactors reported as relays | Overlapping classes with too little data | Per-class confusion matrix from `cli eval`; usually needs more instances, not more epochs |
| GPU latency looks like CPU | Silent CPU fallback | `nvidia-smi` inside the container; `--gpus all` |
| Slower with more threads | Thread over-subscription in a CPU-limited container | Set `OMP_NUM_THREADS` to the real allocation |
| `no version component` on download | Upstream project has no generated version | Fork it, generate a version, pass `--locator your-workspace/project/1` |
| Risk level shows `unknown` | No model loaded, nothing detected, or >50% unidentified | `assessable: false` and `headline` say which. **Not** a pass — do not render it green |
| Risk `low` but `confidence: low` | Too few devices detected to treat an absence as evidence | Re-photograph the whole cabinet; the headline already says not to read it as a pass |
| v2 model looks better but is worse on site | Duplicates leaked between the new and old data | Re-merge with `--dedup`, then re-evaluate both models on the same split |
| `cli dedup` exits 1 | A duplicate group straddles train/val | Intended — that leakage inflates every metric. Run with `--dst` to fix |
| SAM refinement accept rate is low | SAM is segmenting whole DIN-rail rows, not devices | Check `rejected_reasons`; if mostly `grew_beyond_limit`, use `--no-refine` |
| HPO gains nothing over defaults | The dataset is the bottleneck, not the hyperparameters | The result says so explicitly; spend the compute on `cli gap`'s capture list |
| Batch endpoint no faster | Backend has no real batched forward pass | `true_batching: false` in the response says so; `industrial_ultralytics` batches, ONNX does not |
| `metrics.json` has null accuracy | Exported without `--eval-json` | Run `cli eval`, re-export. A model with no accuracy record cannot be audited |

---

## Security and licensing

- Keep `ROBOFLOW_API_KEY` and any Kaggle token in a gitignored `.env`. Never commit
  them; never paste them into an issue.
- The API has no authentication of its own. Terminate TLS and authenticate at the
  reverse proxy; do not expose it directly.
- Uploads are size-capped (`max_upload_bytes`) and decoded defensively; archive
  extraction in the downloader rejects path traversal and link members.
- **Public dataset licences bind the model.** Most sources here are CC BY 4.0,
  which requires attribution — ship `download_manifest.json` and credit each
  dataset in the model card. Manufacturer catalogue photography is copyrighted:
  clear it with the vendor or your legal team *before* training on it, not after.
