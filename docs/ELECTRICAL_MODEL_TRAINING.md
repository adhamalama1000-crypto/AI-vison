# Training the Electrical Component Detection Model

Reproducible, end-to-end procedure for training the YOLOv11 detector that powers
component recognition in the AI Vision Platform.

**Read this section first, before running anything.**

---

## The honest starting position

No trained checkpoint ships with this repository, and the reason is not that the
code was missing — the whole pipeline (training, export, inference, post-processing,
API, UI) exists and is tested. The reason is **data**.

A survey of every public dataset relevant to industrial electrical panels was run
against Roboflow Universe, Kaggle and Open Images, and the verified results are
recorded in `training/electrical/datasets.py` with real locators, real image
counts, and the real per-class instance counts read off each upstream project.
The conclusion:

| Question | Answer |
|---|---|
| Usable public images, all sources combined | **~3,500** |
| Public annotated instances, all classes | **~5,300** |
| Taxonomy classes with *any* public source | 34 of 54 |
| Classes reaching the 300-instance reliability bar | **6** (`mcb`, `contactor`, `relay`, `fuse`, `plc`, `timer_relay`) |
| Priority classes with **zero** public instances | **6** — `vfd`, `power_supply`, `busbar`, `din_rail`, `wire_duct`, `emergency_stop` |

Run it yourself; the numbers are computed, not asserted:

```bash
python -m training.electrical.cli plan | jq '.dataset_plan | {
  forecast_images,
  reliable: .forecast_reliable_classes,
  weak: .forecast_weak_classes,
  no_public_data: .priority_classes_with_no_public_instances
}'
```

Three specific traps in the public data, which is why the registry annotates
rather than just lists:

1. **`rf_electrical_panel_imgpro`** has ~255 instances of each of five on-target
   classes across 256 images — but it is roughly one instance per image of *the
   same panel*. Effective diversity is one cabinet, not 256. It will teach the
   model what a contactor looks like from many angles; it will not teach
   manufacturer or layout invariance.
2. **The two `switchgear_components` sets** have near-identical class lists and the
   same ~30-instances-per-class signature. They are very probably overlapping
   imagery. Merging both inflates validation scores without adding diversity —
   deduplicate by perceptual hash, or pick one. They are also *medium-voltage*
   switchgear, so their "Circuit Breaker" is a cubicle VCB, not a DIN-rail MCB.
3. **`rf_terminal_block`** boxes each terminal individually, while this project's
   labelling rule is one box per contiguous strip. Merging it as-is teaches
   per-pole boxes and wrecks terminal-block counts in the bill of materials.

Two sources are recorded and **deliberately excluded** so nobody rediscovers and
merges them: `rf_thermal_panel` (IR imagery — wrong input distribution for
visible-light RTSP cameras) and `rf_pushbutton_generic` (1,958 tempting images of
consumer appliance buttons, which would turn the detector into a false-positive
generator for every round button in frame).

### What this means for you

Public data will bootstrap perhaps six classes to usable accuracy. It will not
produce the 22-class production model the platform is designed around. **Budget
for a capture programme.** The exact shortfall, in units you can act on:

```bash
python -m training.electrical.cli gap --priority-only
```

From zero, that reports **~6,600 annotations ≈ ~550 labelled panel photographs**
to bring all 22 priority classes to 300 instances each. That is roughly 2–4 weeks
of one engineer photographing panels on site plus labelling, and it is the single
highest-value investment in the whole system. Everything else here is already
built.

---

## Requirements

```bash
pip install -r requirements.txt          # runtime
pip install -r requirements-train.txt    # training extras
pip install ultralytics                  # YOLOv11 — required to train and export
pip install roboflow                      # optional; REST fallback works without it
```

GPU training needs a CUDA build of torch matching your driver. CPU inference needs
only `onnxruntime`, which is already in `requirements.txt`.

```bash
export ROBOFLOW_API_KEY=...   # from app.roboflow.com → Settings → API
```

Put it in a **gitignored** `.env`. Never commit it.

---

## The pipeline

### 1. Survey

```bash
python -m training.electrical.cli plan          # sources, licences, forecast
python -m training.electrical.cli gap           # the shortfall, from zero
```

`plan` verifies nothing about your disk — it tells you what exists upstream, what
each source contributes, under which licence, and what it cannot give you.

### 2. Download

```bash
python -m training.electrical.cli download --all --dst data/raw
```

Each source is fetched, its layout normalised to `images/<split>` +
`labels/<split>`, and its labels remapped onto the canonical taxonomy using the
per-source `label_map`. A label that cannot be mapped is **dropped with a count**,
never folded into a neighbouring class — check `unmapped_source_classes` in the
output, because that is where silent label noise would otherwise enter.

One failing upstream project does not abort the batch. Read the `failed` reasons:

- `ROBOFLOW_API_KEY is not set` — export the key.
- `'<locator>' has no version component` — the upstream project never generated a
  version, so it cannot be downloaded. Fork it at app.roboflow.com, generate a
  version, and pass your fork:
  ```bash
  python -m training.electrical.cli download \
      --sources rf_control_panels_azure \
      --locator my-workspace/control-panels/1 --dst data/raw
  ```
  This applies to `rf_control_panels_azure`, which is the best public source of
  modular DIN-rail instances (~703 MCB boxes) — worth the five minutes.

`data/raw/download_manifest.json` records every licence. **CC BY 4.0 requires
attribution**: keep that manifest with the trained model and credit each dataset
in the model card.

### 3. Merge, deduplicate, and split

```bash
python -m training.electrical.cli merge --roots data/raw/rf_* --dst data/merged --dedup
python -m training.electrical.cli split --src data/merged_dedup --dst data/final
```

**Do not skip `--dedup`.** Two of the registry's sources are probably the same
photographs republished, and Roboflow exports contain augmented copies of every
original. If a duplicate straddles train and val, validation is scoring
memorisation and the mAP you get back is fiction. Check it independently:

```bash
python -m training.electrical.cli dedup --root data/merged
```

Read-only by default, and **exits non-zero when any duplicate group straddles a
split**, so CI can gate on it. Detection is dHash + aHash within a Hamming distance
of 5, plus an exact-content pass; measured on panel imagery, brightness, JPEG q30,
blur and a half-resolution round trip all give a distance of 0–1 while two different
panels give 16–22, so the threshold sits in a wide gap. Add `--dst` to write a
cleaned copy; it keeps the **training** copy and drops the val/test one, which is the
only direction that leaves evaluation data unseen.

Two behaviours worth knowing: duplicates whose **labels disagree** are kept in full
and reported rather than resolved by picking one (that is a human's call, and one of
the two annotations is wrong); and near-featureless images — lens cap, blown
exposure, uniform wall — cannot be hashed reliably, so filter those at capture
rather than trusting dedup on them.

`split` produces the 80/10/10 division — **grouped by capture, not by image**.

This matters more than any hyperparameter. Panel photography comes in bursts: the
same cabinet from five angles, a wide shot plus two close-ups, plus Roboflow's
augmented copies. Split that at random and near-duplicates land in both train and
val, so validation measures memorisation. A model reported at mAP@50 = 0.95 then
fails on site, and nobody can work out why.

The splitter derives a capture-group key from each filename (stripping source
prefixes, Roboflow's `_jpg.rf.<hash>` mangling, augmentation verbs and shot
counters) and keeps every image of one group in exactly one split. Group
assignment is quota-driven, rarest-class-first, so a 30-instance class still
reaches val instead of being randomly excluded from mAP altogether.

If your capture programme recorded which panel each photograph came from — and it
should — pass it and beat the heuristic outright:

```bash
python -m training.electrical.cli split --src data/merged --dst data/final \
    --groups groups.json      # {"IMG_1001.jpg": "cabinet-A", ...}
```

Read the split report before training:

- `leaking_groups` must be `0`.
- `classes_absent_from_val` — these are excluded from mAP entirely. Their accuracy
  is **unmeasured**, whatever the headline number says.
- `input_leaking_groups` — how much contamination the merged input had.
- Achieved ratios will not be exactly 80/10/10. With 30 capture groups the
  smallest increment is 3.3%. That is correct behaviour: group integrity beats
  ratio precision.

### 4. Confirm the data is worth training on

```bash
python -m training.electrical.cli analyse --root data/final
python -m training.electrical.cli gap     --root data/final
```

`gap` exits non-zero while any priority class has zero annotations, so CI can gate
a release on "the model has data behind it".

Thresholds (`training/electrical/datasets.py`): below **50** instances a class will
not train usefully; below **300** it trains but stays unreliable. These are
working rules of thumb for detection fine-tuning, not guarantees.

### 5. Train

```bash
python -m training.electrical.cli train \
    --data data/final/dataset.yaml --arch yolo11s \
    --epochs 120 --imgsz 960 --batch 8 --device 0
```

Defaults are tuned for panel imagery rather than copied from COCO, and each choice
has a reason (`training/electrical/train.py`):

| Setting | Value | Why |
|---|---|---|
| `imgsz` | 960 | At 640 an MCB in a wide cabinet shot is a handful of pixels. Use 640 only if inference latency forces it. |
| `fliplr` | **0.0** | Panels are gravity-oriented and nameplates are directional. A mirrored nameplate is not a real thing; horizontal flip teaches wrong geometry. |
| `degrees` | 8 | Inspectors do not hold a camera level. Mild rotation only. |
| `mosaic` + `close_mosaic=10` | 0.8, off for last 10 epochs | Mosaic helps small-object recall but destroys the row/layout context the model should learn. |
| `hsv_v` / `hsv_s` | 0.50 / 0.55 | Field lighting varies enormously; device colour is a real class signal and must survive. |
| `perspective` | 0.0008 | Off-axis framing is normal; extreme warp is not. |

Requested augmentations map as follows: rotation → `degrees`, brightness/contrast →
`hsv_v`, hue → `hsv_h`, flip → `flipud`/`fliplr` (both deliberately 0 — see above),
perspective → `perspective`, mosaic → `mosaic`, MixUp → `mixup`. Blur and noise are
not Ultralytics knobs; they arrive through the Albumentations transforms
Ultralytics applies automatically when `albumentations` is installed, and through
`training.electrical.synthetic`'s nuisance factors for composited data.

### 5b. Pick the architecture by measurement, not opinion

```bash
python -m training.electrical.cli bench --data data/final/dataset.yaml \
    --root data/final --archs yolo11n yolo11s yolo11m rtdetr-l --epochs 60
```

This trains each architecture on the same split, then measures **both halves**:
accuracy (mAP@50, mAP@50-95, precision, recall, F1) and cost (p50/p95/p99 latency,
FPS, peak RSS delta, parameter count), and picks a winner automatically.

Ranking on mAP alone always picks the largest model, and for a platform whose
default deployment is ONNX Runtime on CPU that is usually wrong — a few points of
mAP for six times the latency is a bad trade on a 4-core box. So selection is a
weighted score over accuracy and speed, with a **hard latency budget** (default
4000 ms p95) that *disqualifies* rather than penalises, and it prints what it traded
away:

```
1  yolo11s   0.539  mAP50-95 0.490  p95  520 ms
2  yolo11m   0.528  mAP50-95 0.520  p95 1450 ms
NOTE: rtdetr-l is more accurate (mAP@50-95 0.530, 0.040 higher) but slower
(2100 ms vs 520 ms p95). If accuracy matters more than latency, override with
--weights or raise the latency budget.
Disqualified: yolo11x (p95 latency 6200 ms exceeds the 4000 ms budget).
```

Override the trade when your deployment justifies it:

```bash
# accuracy first (GPU deployment, latency is not the constraint)
--weights '{"map_50_95":0.80,"map_50":0.10,"f1":0.08,"speed":0.02}' \
    --latency-budget 10000
# speed first (many cameras on modest CPU)
--weights '{"map_50_95":0.15,"map_50":0.05,"f1":0.05,"speed":0.75}'
```

Timing is done properly: warmup runs are discarded (the first inference pays lazy
graph init), real images are required because detector latency is data-dependent
through NMS, percentiles are reported rather than just a mean, and the thread count
and environment are recorded so the numbers are reproducible. Anything unmeasurable
— no `psutil`, no torch for parameter counts — comes back `None` with a reason, never
an estimate.

Profile a single already-trained model:

```bash
python -m training.electrical.cli profile --root data/final \
    --weights runs/electrical/yolo11s/weights/best.pt --runs 50
```

RT-DETR is worth including: it tends to win on densely packed scenes, and a DIN rail
full of adjacent modular devices is exactly that. An architecture the installed
Ultralytics build cannot construct is reported as `skipped` with the reason — never
silently substituted.

### 5c. Hyperparameter search

The `TrainConfig` defaults are hand-reasoned and documented, but they were never
searched. This searches them:

```bash
python -m training.electrical.cli hpo --data data/final/dataset.yaml \
    --root data/final --arch yolo11s --trials 20 --epochs 20 --device 0
```

Covers learning rate, batch size, image size, optimizer, LR schedule, warmup, weight
decay, early-stopping patience and the full augmentation block. The reference run
uses the hand-tuned defaults first, so the search has something to beat — and if it
does not beat them by more than noise, the output says so and tells you to keep the
defaults.

Three things to know:

- **`fliplr` and `flipud` are held at 0 by default and never sampled.** A search
  maximising validation mAP on a small dataset will switch horizontal flip on, because
  it looks like free augmentation, and produce a model that has learned mirrored
  nameplates and reversed device markings are normal. Physical correctness is not a
  tunable. `--no-respect-domain-priors` searches them anyway; the flag exists so the
  decision is deliberate.
- **Pruning is real.** A per-epoch Ultralytics callback reports intermediate mAP to
  Optuna's median pruner, so a trial that is clearly behind stops instead of running
  to completion.
- **It cannot fix a data problem.** Pointed at a dataset where `cli gap` reports
  classes with zero annotations, it says so up front rather than spending GPU hours
  tuning a class that has no examples.

Studies are stored in SQLite under `runs/electrical/`, so an interrupted six-hour
search resumes rather than restarting. The search uses short runs to *rank* the
space; train the winner properly afterwards with `cli train`.

### 6. Evaluate

```bash
python -m training.electrical.cli eval --root data/final \
    --backend industrial_ultralytics \
    --params '{"weights":"runs/electrical/yolo11s/weights/best.pt"}'
```

Reports mAP@50, mAP@50-95, per-class precision/recall/F1, a confusion matrix and
FP/FN analysis. Ultralytics writes PR curves, loss curves and its own confusion
matrix into the run directory.

Read the **per-class** table, not the headline mAP. With a long-tailed dataset the
mean is dominated by whichever classes have data, and a class absent from val
contributes nothing at all.

Derive per-class confidence thresholds from the validation split rather than
guessing:

```bash
python -m training.electrical.cli tune --root data/final \
    --backend industrial_ultralytics --objective f1
```

Compare against the zero-shot baseline to know whether training actually helped —
`training.electrical.train.evaluate_zero_shot`.

### 7. Export and install

```bash
# capture the accuracy evidence first, so the bundle can carry it
python -m training.electrical.cli eval --root data/final \
    --backend industrial_ultralytics \
    --params '{"weights":"runs/electrical/yolo11s/weights/best.pt"}' \
    > dist/eval.json

python -m training.electrical.cli export \
    --weights runs/electrical/yolo11s/weights/best.pt \
    --out dist/model --imgsz 960 --data data/final/dataset.yaml \
    --eval-json dist/eval.json --install
```

Produces `best.pt`, `best.onnx`, `labels.txt`, `classes.json`,
`model_card.json`, `metrics.json` and an `artifacts/` directory holding the
confusion matrix, PR/F1/P/R curves, loss curves, `results.csv` and the exact
training `args.yaml`. It then verifies the bundle the way the runtime will read it,
and refuses to install one that would mislabel.

The evidence matters as much as the weights. A deployed model with no record of its
own measured accuracy cannot be audited, and "what was this model's per-class recall?"
has no answer six months later. `--eval-json` is what puts that record in the bundle;
without it the export warns. Curves come from Ultralytics' own plots when it wrote
them, and are rendered from `results.csv` when it did not (an RT-DETR run, or
`plots=False`). The confusion matrix is rendered from *our* evaluation, so its axes are
readable device names rather than integer indices, and it plots only the classes that
actually appear — a 54×54 grid of zeros tells you nothing.

That verification is not ceremony. An earlier version of this platform shipped a
`labels.txt` containing the literal lines `0`…`9`, so every detection came back
named `"0"`. `verify` now rejects a numeric labels file, a labels/classes
disagreement, a reordered label space, and an ONNX head whose class count does not
match the label count — the failure that shifts every label by one and is invisible
until someone notices a contactor being reported as a relay.

```bash
python -m training.electrical.cli verify --bundle models/components
```

Select the backend: `industrial_onnx` for CPU, `industrial_ultralytics` to run
`best.pt` on a GPU. For TensorRT:

```bash
python -m training.electrical.cli tensorrt
```

It prints instructions rather than building an engine, because a TensorRT engine is
compiled for one GPU architecture, driver and TensorRT version. Building it in CI
and shipping the file gives a binary that fails to deserialise in production — it
must be built on the deployment host.

---

## Closing the gap: the capture and labelling loop

This is the part that actually gets you to production.

### Capture

```bash
python -m training.electrical.cli plan | jq '.custom_collection'
```

The protocol in brief:

- Three framings per panel: whole cabinet door-open, each device row filling the
  frame, and a close-up of every nameplate. **Row framing is what the model sees
  in service.**
- Vary the angle deliberately: straight on, and ±30° horizontally and vertically.
- Capture in the lighting that actually exists — fluorescent, torch, flash,
  backlit through the window. Do not normalise it; that variation is the signal.
- **Include the panels that look bad**: dusty, oil-filmed, cable bundles crossing
  devices, faded labels, mixed manufacturers, retrofits. Clean panels alone produce
  a fragile model.
- Photograph the same device family from several manufacturers. Manufacturer
  invariance has to be learned from examples.
- Record the as-built bill of materials per panel. It validates counts
  independently of the boxes and costs nothing at capture time.
- **Record the panel id per photograph** into `groups.json`. This is what makes the
  split trustworthy.
- Where each hard class is found on site: `gap --priority-only | jq '.what_to_collect'`.

### Pre-label, then correct

```bash
python -m training.electrical.cli labelguide       # hand this to the labeller
python -m training.electrical.cli autolabel --images captures/ --out data/prelabelled
```

Auto-labelling makes a human *correct* boxes instead of drawing them — a 3–5×
speed-up on the several hundred hours the shortfall represents. It uses a trained
checkpoint when one exists, and falls back to zero-shot OWLv2 / Grounding DINO for
round one (`pip install -r requirements-openvocab.txt`).

**Boxes are tightened with SAM2 by default.** This is not decoration:
open-vocabulary detectors are trained on natural-image captions, so their boxes
routinely include a strip of DIN rail, the neighbouring module and the wire loom —
and correcting a loose box costs as much as drawing a new one, which destroys the
speed-up. SAM2 is used as a *promptable segmenter*: each detector box becomes a box
prompt, and the tight bounds of the returned mask replace the loose box. Detection
stays the detector's job; localisation becomes SAM2's.

Every refinement is guard-checked, because SAM fails in predictable ways on panel
imagery and an unchecked "tighter" box is worse than a loose one:

| Guard | Catches |
|---|---|
| must not grow beyond 1.6× area | segmented the **whole DIN-rail row** (modules are visually continuous) |
| must not shrink below 0.35× area | segmented only the **toggle lever** or one terminal |
| centre must not drift >0.35 of the box diagonal | segmented the **neighbouring device** |
| aspect ratio must fit the class's taxonomy prior | an MCB is never 8:1 wide |

A failed guard keeps the original box and counts the reason. The manifest reports the
accept rate and mean IoU shift, so "is SAM helping on my imagery?" has a number: a low
accept rate dominated by `grew_beyond_limit` means it is segmenting rows, and
`--no-refine` saves you the compute. Without a SAM backend installed, refinement is
skipped with a stated reason and labelling proceeds on the detector's own boxes.

Every image gets a verdict in `autolabel_manifest.json`:

| Verdict | Meaning |
|---|---|
| `auto` | All boxes cleared the accept threshold. Fast glance. |
| `review` | At least one box is between review and accept. Look properly. |
| `uncertain` | Boxes found, classes not confident — written as `unknown_industrial_component`, **not guessed**. |
| `empty` | Nothing detected. A valid negative, but flagged so a batch of empties is not mistaken for finished work. |

Boxes below the review threshold are discarded: a label file full of junk is slower
to fix than an empty one.

**These are pre-labels, not ground truth.** Import into Roboflow / CVAT / Label
Studio as YOLO, work the `review_queue` first (worst predictions are both most
likely wrong and most informative), correct, export. Training directly on
un-reviewed output teaches the model its own mistakes.

### The labelling rules that decide whether the dataset is any good

Full guide: `python -m training.electrical.cli labelguide`. The ones that get
broken most often:

- **Contactor + overload relay = two boxes.** Separately replaceable devices,
  separately reported.
- **Terminal blocks: one box per contiguous strip, never per pole.** Per-pole
  labelling produces hundreds of boxes per image and destroys class balance.
- **MCB: one box per device, not per pole.** A 3-pole MCB with a common toggle is
  one box; three 1-pole MCBs side by side are three, even when linked by a comb.
- **DIN rail / cable duct / busbar: one box per continuous run.** Label visibly
  empty rail too, or the class is learned as "gap between devices".
- **Illuminated push button is a push button.** An indicator lamp has no actuator
  travel.
- **Emergency stop is never a push button.**
- **Generic `circuit_breaker` is for genuine ambiguity, not convenience.** If you
  can see it is an MCB, MCCB or ACB, label that.
- **Do not label** wires, legend plates, stickers, the enclosure, or anything
  visible only as a reflection in the door glass.
- **If unsure, label `unknown_industrial_component`. Never guess.** A wrong
  confident label is actively harmful and very hard to find later.

Two labellers on a 10% sample; measure agreement. Below ~0.85 IoU or ~0.90 class
agreement, **fix the guide before labelling more** — you are otherwise buying noise
at scale.

### The loop

Every `unknown_industrial_component` returned in production is a labelled example
the model is asking for. Feed them back into `autolabel`, correct, retrain. That
loop is what closes the long tail; nothing else does.

---

## Taxonomy and label-space rules

`rtsp_backend/electrical/taxonomy.py` holds 54 classes with engineering function,
geometric priors, zero-shot prompts and per-class thresholds. The brief's 23 target
classes all map onto it.

**`CLASS_ORDER` is append-only.** Inserting or reordering invalidates every
previously trained checkpoint, because the class index in a label file and in a
model head is positional. Taxonomy 5.1 appended one class, `circuit_breaker`
("type unspecified"), as an honest home for the many public datasets that label
every protective device "circuit breaker" and for medium-voltage VCB/SF6 breakers
that have no LV equivalent. Indices 0–52 are unchanged, so a 5.0 checkpoint remains
valid and `verify` reports it as a valid prefix.

`resolve()` never invents specificity a label does not carry: "miniature circuit
breaker" → `mcb`, but a bare "circuit breaker" → `circuit_breaker`, never MCB or
MCCB. Where a source *is* specific but oddly worded ("circuit breaker 1-pole" in
the German distribution-board sets really is an MCB), the mapping goes in that
source's `label_map`, explicitly, where it is reviewable.

---

## Reproducibility

- `--seed` fixes training and splitting. `split` is deterministic for a seed.
- `dataset.yaml` and `classes.json` pin the label space next to the data and next
  to every export.
- `download_manifest.json` records every source, version and licence.
- `model_card.json` records classes, input/output contract, provenance and
  **limitations** — including that any class the gap report calls weak or
  untrainable is unvalidated regardless of headline mAP.

## Files

| Path | Purpose |
|---|---|
| `training/electrical/datasets.py` | Verified source registry, remapping, merging, coverage + gap reports |
| `training/electrical/download.py` | Roboflow / Kaggle / URL fetchers, layout normalisation |
| `training/electrical/split.py` | 80/10/10 group-aware splitting |
| `training/electrical/autolabel.py` | Model-assisted pre-labelling + the human annotation guide |
| `training/electrical/synthetic.py` | Crop-composited dataset multiplication |
| `training/electrical/train.py` | Training, architecture benchmarking, evaluation |
| `training/electrical/export.py` | Bundle export, verification, install, TensorRT guidance |
| `training/electrical/cli.py` | Every step above, as a subcommand |
| `rtsp_backend/electrical/` | Runtime: taxonomy, post-processing, recognisers, inspector |
| `rtsp_backend/api/panel_analyze.py` | `POST /api/panel/analyze` |

Deployment: `docs/PRODUCTION_DEPLOYMENT.md`.
