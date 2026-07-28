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

### 3. Merge and split

```bash
python -m training.electrical.cli merge --roots data/raw/rf_* --dst data/merged
python -m training.electrical.cli split --src data/merged --dst data/final
```

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

Pick the architecture with a measurement, not an opinion:

```bash
python -m training.electrical.cli bench --data data/final/dataset.yaml \
    --root data/final --archs yolo11s yolo11m rtdetr-l --epochs 60
```

RT-DETR is worth measuring here: it tends to win on densely packed scenes, and a
DIN rail full of adjacent modular devices is exactly that. An architecture the
installed Ultralytics build cannot construct is reported as `skipped` with the
reason — never silently substituted.

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
python -m training.electrical.cli export \
    --weights runs/electrical/yolo11s/weights/best.pt \
    --out dist/model --imgsz 960 --data data/final/dataset.yaml --install
```

Produces `best.pt`, `best.onnx`, `labels.txt`, `classes.json` and
`model_card.json`, then verifies the bundle the way the runtime will read it and
refuses to install one that would mislabel.

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
