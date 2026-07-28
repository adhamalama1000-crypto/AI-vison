# Industrial component detector — data, training, evaluation

How to go from "no model, zero components" to a detector that recognises
industrial electrical equipment on real Madkour panels, and how to prove it did.

Everything here is driven from one entry point:

```bash
python -m training.electrical.cli --help
```

> **Where this repository stands.** No trained checkpoint ships with the code, and
> none could be produced in the environment it was built in: no GPU, and no
> network route to the dataset or model hubs. The recogniser therefore reports
> `weights_missing` and returns **zero components** rather than inventing any.
> This document is the procedure that closes that gap. See
> `docs/AUDIT_PANEL_INSPECTOR.md` for exactly what is and is not proven.

---

## 0. The label space

Every dataset, checkpoint and export in this project shares one label space:
`rtsp_backend.electrical.taxonomy.CLASS_ORDER` (53 classes).

- It is **append-only**. Inserting a class in the middle invalidates every
  previously trained checkpoint.
- `models/components/classes.json` is the authoritative on-disk copy. The
  recogniser reads it; training writes it next to every export. A test asserts
  the two match.
- Adding a class means adding a `ComponentSpec` to `taxonomy.py` — its function,
  role, mounting, geometric priors, aliases and zero-shot prompts. The taxonomy
  entry is what makes the rest of the system able to reason about the device;
  the class id alone is not enough.

Check what you have:

```bash
python -m training.electrical.cli plan
```

This prints the taxonomy, the dataset sources with their licences and how to
fetch each, which classes have a downloadable source and which need capture, and
which architectures are actually available in your environment.

---

## 1. Get data

### 1.1 The realistic assessment

Public coverage of this taxonomy is thin and heavily skewed toward modular
breakers and terminal blocks. Drives, ACBs, safety relays, meters and instrument
transformers are effectively absent from public detection datasets. `plan()`
reports downloadable coverage separately from "needs manual capture" precisely so
this is not glossed over.

So the plan is three-legged:

| Leg | Covers | Effort | Value |
|---|---|---|---|
| Public datasets | common modular protection, terminal blocks, some operator devices | low | bootstrap |
| Vendor catalogue crops → synthetic composition | the whole taxonomy, with ground-truth part numbers | medium | high |
| Madkour field capture | the deployment distribution | high | decisive |

### 1.2 Public datasets

Search Roboflow Universe for `electrical panel`, `control panel components`,
`switchgear`, `MCB detection`, `distribution board`. Individually the projects are
too small; merged they cover the common classes. Download in YOLOv8 format, then
remap onto the canonical label space:

```bash
python -m training.electrical.cli remap \
  --src raw/rf_panel_1 --dst data/rf1 \
  --names-from raw/rf_panel_1/data.yaml \
  --source-key rf_electrical_panel_components
```

`remap` resolves each source label through that source's declared label map and
then the taxonomy resolver. **Anything it cannot map is dropped with a count and
listed on stderr** — never folded into a nearby class, because a wrong label is
worse than a missing one. Read that list; it usually tells you the source has a
class worth adding to the taxonomy.

Check the licence of every project before using it commercially. The registry
records what is known, but upstream terms change.

### 1.3 Vendor catalogue crops (highest value per hour)

Every manufacturer publishes clean product photography of every product, labelled
with the exact part number. Build a crop library:

```
data/crops/
  contactor/          lc1d09.jpg  lc1d32.jpg  3rt2027.jpg  af16.jpg ...
  overload_relay/     lrd10.jpg   3ru2116.jpg ...
  mccb/               nsx100.jpg  3va1.jpg    xt1.jpg ...
  plc/                s7-1214.jpg cp1e.jpg    ...
```

Directory names are resolved through the taxonomy, so `MCCB`,
`magnetic contactor` and `mccb` all land correctly. Aim for **10+ crops per
class, from at least 3 manufacturers** — manufacturer-invariance has to be
learned from examples.

These images are copyrighted. Clear their use with the vendor or your legal team
before training on them.

Then multiply:

```bash
python -m training.electrical.cli synth \
  --out data/composed --crops data/crops --train 4000 --val 800
```

`compose_from_crops` seats real device crops on synthetic back plates with DIN
rails and cable ducts, then applies lighting variation, perspective, rotation,
occlusion (cable bundles, opaque patches), dust film, cast shadow, specular
reflection, blur, sensor noise and JPEG artefacts. Device *appearance* is real;
only the arrangement and the nuisance factors are synthetic. The manifest records
`source: composed_real_crops`.

### 1.4 Procedural data — what it is and is not for

```bash
python -m training.electrical.cli synth --out data/procedural --train 400 --val 80
```

With no crop library the generator draws device housings from the taxonomy's
geometric priors, with terminals, screws, label plates, status LEDs, ventilation
and dials. This is **not photorealistic** and a model trained only on it will not
generalise to a real cabinet. Its manifest carries an explicit
`PROCEDURAL DATA ONLY` warning.

It is genuinely useful for two things: exercising and measuring the whole
pipeline against exact ground truth (this is what
`scripts/validate_panel_inspector.py` uses), and pre-training layout priors
before real images exist.

### 1.5 Madkour field capture — the decisive leg

```bash
python -c "import json;from training.electrical import datasets as d;print(json.dumps(d.custom_collection_plan(),indent=2))"
```

The protocol in brief:

**Capture**
- Three framings per panel: whole cabinet door-open, each device row filling the
  frame, close-up of every nameplate. The row framing is what the model sees in
  service; the nameplate close-ups train and validate the part-number reader.
- Vary the angle: square-on and roughly ±30° horizontally and vertically. A model
  trained only square-on fails the moment an inspector stands to one side.
- Capture in the lighting that exists — overhead fluorescent, torch, flash,
  backlit through the window. Do not correct it. That variation *is* the signal.
- Include the panels that look bad: dusty, oil-filmed, cable bundles crossing
  devices, faded labels, mixed manufacturers, retrofits.
- Record the as-built bill of materials per panel. It validates counts and
  panel-type inference independently of the boxes and costs nothing at capture
  time.

**Labelling**
- One tight box per physical device, including terminals, excluding wiring.
- An overload relay bolted under a contactor is **two boxes**. They are
  separately replaceable and separately reported. (The per-class NMS exists so
  both survive.)
- Terminal blocks: **one box per contiguous strip**, not per pole. Per-pole
  labelling produces hundreds of boxes per image and wrecks the class balance.
- Structural items (DIN rail, duct, busbar): one box per continuous run.
- If a labeller is not certain, label it `unknown_industrial_component`. Honest
  unknowns become the next capture list; wrong labels become permanent model
  error.
- Two labellers on a 10 % sample; measure agreement. Below ~0.85 IoU the guide
  needs work, not the model.

**Splitting**
- Split by **panel**, never by image. Several framings of the same cabinet
  appearing in both train and val leaks and inflates every metric. This is the
  single most common way an industrial detector is reported as excellent and then
  fails on site.

---

## 2. Merge and assess before training

```bash
python -m training.electrical.cli merge \
  --roots data/rf1 data/rf2 data/composed data/madkour --dst data/merged

python -m training.electrical.cli analyse --root data/merged
```

`merge` prefixes filenames per source so identically-named images from different
datasets cannot silently overwrite each other.

`analyse` reports per-class instance and image counts and box-size distribution.
`coverage_report` then splits the taxonomy into four buckets against working
rules of thumb for detection fine-tuning:

| Bucket | Instances | Expect |
|---|---|---|
| reliable | ≥ 300 | usable in production |
| weak | 50–299 | trains, stays unreliable |
| untrainable | 1–49 | will not work |
| absent | 0 | cannot work |

**Read this before training, not after.** Anything outside `reliable` should be
expected to fall through to *Unknown Industrial Component* — which is the correct
behaviour, and better than a confident wrong answer.

---

## 3. Train

```bash
python -m training.electrical.cli train \
  --data data/merged/dataset.yaml --arch yolo11s \
  --epochs 150 --imgsz 960 --batch 16 --device 0 --install
```

Requires `pip install ultralytics`. `--install` copies the exported ONNX into
`models/components/` together with `classes.json`, and the backend picks it up on
next load — no code change.

### The recipe, and why it differs from a COCO recipe

| Setting | Value | Reason |
|---|---|---|
| `imgsz` | 960 | Modular devices are small relative to a cabinet photograph. At 640 an MCB in a wide shot is a handful of pixels. |
| `fliplr` / `flipud` | **0.0** | Panels are gravity-oriented and nameplates are directional. A mirrored device is not a real thing; mirroring teaches wrong geometry. |
| `degrees` | 8 | Real hand-held capture is a few degrees off level, not upside down. |
| `mosaic` + `close_mosaic=10` | 0.8, off for the last 10 epochs | Mosaic helps small-object recall but destroys the row/layout context. Turning it off at the end restores it. |
| `hsv_v` 0.50 / `hsv_s` 0.55 | strong value, moderate saturation | Lighting varies enormously in the field, but device colour is a real class signal and must not be destroyed. |
| `perspective` | 0.0008 | Mild — the camera-angle variation belongs in the data, not only the augmenter. |

---

## 4. Choose the architecture by measurement

```bash
python -m training.electrical.cli bench \
  --data data/merged/dataset.yaml --root data/merged \
  --archs yolo11s yolo11m yolov8s rtdetr-l --epochs 80 --device 0
```

Trains each architecture on the same split with the same budget, evaluates each
with `rtsp_backend.electrical.metrics`, and prints a ranked table ordered by
mAP@0.5:0.95 then F1.

Candidates and why each is in the list:

- **YOLOv11** — current generation, best accuracy-per-millisecond here. Default.
- **YOLOv8** — the most widely reproduced baseline; a control.
- **RT-DETR** — NMS-free transformer detector; tends to win on densely packed
  scenes, and a DIN rail full of adjacent modular devices is exactly that. Worth
  measuring rather than assuming.
- **YOLOv12** — attempted only if the installed Ultralytics exposes it, and
  reported as `skipped` otherwise. The brief said "if stable"; that condition is
  enforced in code, not by hand.

An architecture that is unavailable is reported as `skipped` with the reason. It
is never silently substituted.

**Open-vocabulary models are compared on the same split:**

```python
from training.electrical.train import evaluate_zero_shot
evaluate_zero_shot("data/merged", backends=["openvocab_owlv2"], limit=50)
```

This is the honest baseline question: does the trained model actually beat
zero-shot OWLv2 driven by the taxonomy prompts? If it does not, the dataset is
the problem, not the architecture.

---

## 5. Evaluate, and tune the thresholds

```bash
python -m training.electrical.cli eval --root data/merged --backend industrial_onnx
python -m training.electrical.cli tune --root data/merged --backend industrial_onnx \
  --objective f1 --min-precision 0.9
```

`eval` produces precision, recall, F1, AP per class, mAP@50, mAP@50-95, a
confusion matrix **with background row and column** (so misses and hallucinations
are visible, not hidden), false positives broken down by cause — spurious /
class-confusion / localisation — and false negatives by class.

`tune` sweeps per-class confidence thresholds and recommends the value maximising
your objective, optionally subject to a precision floor. Industrial inspection
usually wants `--min-precision 0.9`: a false alarm costs an engineer a site visit.
Apply the result either at runtime:

```
POST /api/ai/models/components/params
{"thresholds": {"contactor": 0.42, "mcb": 0.55, ...}}
```

or make it the default by editing `min_conf` in `taxonomy.py`. **Derive these
numbers; do not leave them at the hand-set defaults once you have a validation
set.**

---

## 6. The improvement loop

One pass is not a project. Iterate:

```
1. analyse  → which classes are absent / weak?
2. capture  → shoot those classes; add crops for them
3. synth    → multiply the new crops
4. merge    → rebuild the training set
5. train    → same recipe, more data
6. eval     → per-class recall; confusion matrix; FP causes
7. tune     → re-derive thresholds
8. validate → python scripts/validate_panel_inspector.py
9. goto 1
```

Read the diagnostics, don't just watch mAP:

| Symptom in `eval` | What it means | What to do |
|---|---|---|
| Low recall on one class | not enough examples, or too little variety | capture more of that class, from more manufacturers |
| High `class_confusion` between two classes | the two are genuinely similar, or labelling is inconsistent | re-check the labelling guide; both classes are in a `CONFUSABLE_GROUPS` entry so the gate can arbitrate |
| High `spurious_detection` | the model fires on background | more negative-heavy images; raise that class's threshold via `tune` |
| High `localisation` | boxes are loose | tighten labelling; check `imgsz` is not too small |
| Many *Unknown Industrial Component* in production | the model is honestly uncertain | those crops are exactly the next training batch — collect them |

Stopping condition: on a held-out set of complete Madkour panels never seen in
training, per-class precision and recall meet the deployment requirement, **and**
the bill-of-materials count matches the as-built drawing. Not "mAP looks good".

---

## 7. Reproducing the pipeline validation

```bash
python scripts/validate_panel_inspector.py --images 25 --json validation.json
```

Measures the wiring false-positive removal, the post-processing gate against the
old logic, the ONNX label decoding fix, and the panel-type rule base. It
validates the *pipeline* on exact ground truth. It does not measure real-world
accuracy — the harness says so in its own output, and so does the audit.
