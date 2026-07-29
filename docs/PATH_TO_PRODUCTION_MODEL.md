# Path to a Production Electrical Component Detection Model

What it takes to get from the current state to **mAP50 > 85%, recall > 80%** on real
industrial panels — the numbers, the plan, and the two things that are not code
problems.

---

## Where this stands

| | |
|---|---|
| Pipeline | Complete and measured end to end (see [`AUDIT_v5.2.0.md`](AUDIT_v5.2.0.md)) |
| Training recipe | Validated — see the recipe experiment below |
| Class scope | `core15` profile implemented, matching the 15 priority classes |
| **Real annotated panel images available** | **~3,500 public, ~5,300 instances, 6 classes viable** |
| **Required for mAP50 > 85% on 15 classes** | **3,300–5,600 images, 10,500–18,000 instances** |

The gap is data, and it is roughly a factor of 2–3 in images and a factor of 2–3.5 in
instances. Every number above is computed, not asserted:

```bash
python -m training.electrical.cli scope --list --name core15   # requirement
python -m training.electrical.cli plan                          # what exists publicly
python -m training.electrical.cli gap --root data/final          # the shortfall
```

---

## 1. Reduce the class scope first — this is the highest-leverage change

mAP is a **mean over classes**. Training 54 classes on thin data averages to a number
no confidence threshold can rescue; training 15 classes with several hundred instances
each is a model that works. The `core15` profile implements exactly the requested list:

```bash
python -m training.electrical.cli scope --list --name core15
```

| # | Class | Taxonomy id |
|---|---|---|
| 0 | MCB | `mcb` |
| 1 | MCCB | `mccb` |
| 2 | Contactor | `contactor` |
| 3 | Relay | `relay` |
| 4 | PLC | `plc` |
| 5 | Terminal block | `terminal_block` |
| 6 | Fuse | `fuse` |
| 7 | Power supply | `power_supply` |
| 8 | Transformer | `transformer` |
| 9 | VFD | `vfd` |
| 10 | Busbar | `busbar` |
| 11 | Cable duct | `wire_duct` |
| 12 | Emergency stop | `emergency_stop` |
| 13 | Switch | `selector_switch` |
| 14 | Indicator lamp | `indicator_lamp` |

**"Switch" was read as `selector_switch`** — the taxonomy distinguishes selector,
changeover and Ethernet switches, and the brief lists it beside emergency stop and
indicator lamp, which are the other fascia-mounted operator devices. If a transfer
switch was meant, use `core18`.

`core18` adds the three most valuable extras, appended so a `core15` checkpoint
fine-tunes onto it rather than needing a retrain:

- **`overload_relay`** — always bolted under a contactor, and the
  contactor-without-overload check is one of the report's most useful findings.
- **`din_rail`** — cheap to label, enables row inference.
- **`circuit_breaker`** — an honest home for breakers whose family cannot be read.

### Applying a profile

```bash
python -m training.electrical.cli scope --name core15 \
    --src data/final --dst data/core15
```

This filters the dataset **and remaps every label index to 0..14**. That remapping is
the step that must not be skipped: a dataset filtered to 15 classes but still carrying
54-class indices trains a 15-class head against indices scattered up to 53. The loss
falls, the run looks healthy, and every prediction is garbage. It is silent in every
trainer, which is why it has its own tests.

Two behaviours worth knowing:

- **Images left with no in-profile boxes are kept as negatives.** An image containing
  only out-of-profile devices teaches the detector not to fire on them. Cap them at
  ~10–15% of the training set; beyond that they suppress genuine detections.
  `--drop-empty` removes them.
- **`--only-present`** narrows the profile to classes that actually have data. An
  absent class contributes a zero to the mAP mean and nothing to the model, so a
  15-class mAP with 7 empty classes misleads in both directions.

A profile bundle needs **no runtime change**: the recogniser reads its label space from
`classes.json` and canonicalises through the taxonomy, so a 15-class model still
returns canonical ids, and anything it is unsure of still becomes
`unknown_industrial_component`.

---

## 2. How much data, exactly

```bash
python -m training.electrical.cli scope --list --name core15 --target-map 0.85
```

Per-class instance requirements. Bands rather than points, because the real figure
depends on intra-class visual variety — how many manufacturers, how many framings —
which is a property of the capture programme, not of the model:

| Target mAP50 | Instances per class | 15-class total | Images | What it means |
|---|---|---|---|---|
| 0.50 | 150–300 | 2,250–4,500 | ~700–1,400 | A usable demonstrator; obvious devices in good light |
| 0.70 | 300–600 | 4,500–9,000 | ~1,400–2,800 | Useful in production, with a human reviewing output |
| **0.85** | **700–1,200** | **10,500–18,000** | **~3,300–5,600** | **The stated target; drives a report without per-image review** |
| 0.92 | 1,500–2,500 | 22,500–37,500 | ~7,000–11,700 | Diminishing returns; needs manufacturer-level coverage |

The image count is **not** total instances ÷ boxes per image. A panel photograph yields
~12 boxes but only ~4 *distinct* classes — an image full of MCBs contributes nothing to
the VFD count. That co-occurrence factor is why 10,500 instances needs ~3,300 images
rather than ~875.

Assumptions, stated so they can be argued with:

- ~12 labelled boxes per panel photograph at row framing.
- ~4 distinct profile classes visible per frame.
- **At least three manufacturers per class.** Manufacturer invariance is learned from
  examples; a model trained on one brand fails on the next.
- **Split by panel, not by image**, or the measured mAP is inflated and the target is
  met on paper only.

Your own instinct of "minimum 5000, prefer 10000+" is right, and independently
corroborated by this estimate.

---

## 3. Where the data comes from

### Public sources: ~3,500 images, and that is the ceiling

Every verified public source is in the registry with real locators, licences and
per-class counts (`cli plan`). The measured totals:

- **~3,500 usable images**, ~5,300 annotated instances across all sources combined.
- **6 classes** reach the 300-instance reliability bar: `mcb`, `contactor`, `relay`,
  `fuse`, `plc`, `timer_relay`.
- **6 of the 15 priority classes have zero public instances**: `vfd`, `power_supply`,
  `busbar`, `din_rail`, `wire_duct`, `emergency_stop`.

That is not a search that was given up on. Roboflow Universe was searched across
multiple angles, GitHub returned nothing usable (a Java circuit-breaker library, a
Brawl Stars mod, a switchgear-symbol printing repo), Open Images has no industrial
electrical classes at all, and no Kaggle dataset was found with boxed panel devices.

**Public data gets you to roughly the 0.50 band on 6 classes. It cannot reach 0.85 on
15.** No code changes that.

### Two blockers that need you, not me

1. **`ROBOFLOW_API_KEY` is not set, and the Roboflow connector's token has expired.**
   Until it is re-authorized (claude.ai connector settings, or `/mcp` in an interactive
   session) even the ~3,500 public images cannot be downloaded from here. With a key:
   ```bash
   export ROBOFLOW_API_KEY=...
   python -m training.electrical.cli download --all --dst data/raw
   ```
2. **`control-panel-azure/control-panels` has no generated version** — the single best
   public source of modular DIN-rail instances (~703 MCB, ~211 RCCB boxes). Fork it at
   app.roboflow.com, generate a version, then:
   ```bash
   python -m training.electrical.cli download --sources rf_control_panels_azure \
       --locator <your-workspace>/control-panels/1 --dst data/raw
   ```

### The capture programme is the actual answer

~2,000–4,000 photographs of real Madkour panels, which is 2–4 weeks of one engineer
with a phone plus labelling. Everything needed to make that efficient exists:

```bash
python -m training.electrical.cli plan | jq '.custom_collection'   # capture protocol
python -m training.electrical.cli gap --priority-only | jq '.what_to_collect'
python -m training.electrical.cli labelguide                        # labelling rules
python -m training.electrical.cli autolabel --images captures/ --out data/prelabelled
```

Auto-annotation with SAM2 box refinement makes a labeller *correct* boxes instead of
drawing them — a 3–5× speed-up, validated on real weights (a loose box refined to
within 5 px of ground truth, 56% area reduction). The review loop is at
`/api/annotations`.

**Capture priority**, cheapest-first by instances-per-hour:

| Priority | Classes | Why |
|---|---|---|
| 1 | `terminal_block`, `wire_duct`, `din_rail`, `mcb` | Present in nearly every panel, many instances per photograph |
| 2 | `contactor`, `relay`, `overload_relay`, `fuse` | Every motor starter; dense in MCC panels |
| 3 | `plc`, `power_supply`, `transformer` | One or two per panel — needs many panels |
| 4 | `emergency_stop`, `selector_switch`, `indicator_lamp` | Fascia devices; photograph doors deliberately |
| 5 | `vfd`, `busbar`, `mccb` | Fewest per panel; needs drive cabinets and incomer sections specifically |

---

## 4. Training recipe

Once the data exists:

```bash
# quality gate, then profile, then split by panel
python -m training.electrical.cli quality --root data/merged --dst data/clean
python -m training.electrical.cli dedup   --root data/clean  --dst data/dedup
python -m training.electrical.cli scope   --name core15 --src data/dedup --dst data/core15
python -m training.electrical.cli split   --src data/core15 --dst data/final --groups groups.json

# pick the architecture on accuracy AND latency
python -m training.electrical.cli bench --data data/final/dataset.yaml --root data/final \
    --archs yolo11s yolo11m yolo12s rtdetr-l --epochs 60 --device 0

# tune, then train the winner properly
python -m training.electrical.cli hpo   --data data/final/dataset.yaml --root data/final \
    --arch yolo11m --trials 20 --epochs 20 --device 0
python -m training.electrical.cli train --data data/final/dataset.yaml \
    --arch yolo11m --epochs 150 --imgsz 960 --batch 16 --device 0

# evaluate, export with evidence, install
python -m training.electrical.cli eval --root data/final --backend industrial_ultralytics \
    --params '{"weights":"runs/electrical/yolo11m/weights/best.pt"}' > dist/eval.json
python -m training.electrical.cli tune --root data/final --backend industrial_ultralytics \
    --params '{"weights":".../best.pt"}' --min-precision 0.9
python -m training.electrical.cli export --weights .../best.pt --out dist/model \
    --imgsz 960 --eval-json dist/eval.json --install
```

**150 epochs, not 100**, at `imgsz=960`: a modular device in a wide cabinet shot is a
handful of pixels, and this dataset size converges slowly. Watch `results.csv` and stop
when val mAP plateaus for ~25 epochs — `patience=25` does it automatically.

**Augmentation is already tuned for panels rather than copied from COCO**, and the two
non-obvious choices matter:

- `fliplr=0.0`, `flipud=0.0` — panels are gravity-oriented and device markings are
  directional. A mirrored nameplate is not a real thing. HPO holds these at 0 by
  default for the same reason; a search maximising val mAP would turn flip on because
  it looks like free augmentation.
- `close_mosaic=10` — mosaic helps small-object recall but destroys the row/layout
  context the model should learn, so it is disabled for the final epochs.

### Recipe validation

To separate "is the recipe capable of high mAP?" from "is there enough data?", the
recipe was run on a dataset engineered to have adequate instance counts — 600 train /
120 val images, 8 classes at ~970 instances each, squarely inside the 700–1,200 band
the table above says is needed for 0.85.

Results are recorded in [`AUDIT_v5.2.0.md`](AUDIT_v5.2.0.md). Read them for what they
are: **the recipe and the class-reduction mechanism measured on synthetic imagery.**
They say nothing about real-world accuracy, because the images are procedurally
composited. What they do establish is whether the bottleneck is the pipeline or the
data — and that is the question worth answering before spending four weeks
photographing panels.

---

## 5. Acceptance criteria

Do not ship on headline mAP. Gate on:

- [ ] **Per-class recall ≥ 0.80 for every class you will act on.** The mean hides a
      class at 0.3.
- [ ] **Per-class precision ≥ 0.90 for safety-relevant findings.** A false alarm costs
      an engineer a site visit; `cli tune --min-precision 0.9` derives thresholds for it.
- [ ] **`classes_absent_from_val` is empty** in the split report. A class with no
      validation instances is unmeasured, whatever the mAP says.
- [ ] **`cli dedup` reports 0 cross-split groups.** Otherwise the metrics are inflated.
- [ ] **Bill-of-materials count accuracy** against the as-built drawing, on a held-out
      set of complete panels never seen in training. This is the acceptance test that
      matters, because it is the product.
- [ ] **Unknown rate < 20%** on field imagery. A higher rate means the training
      distribution does not match deployment.

`cli eval` produces per-class precision/recall/F1, a confusion matrix keyed by device
name, and FP/FN cause analysis. `cli export --eval-json` puts all of it in the bundle as
`metrics.json` plus `artifacts/`, so the deployed model carries its own evidence.

---

## Summary

Three things stand between the current state and a production model:

1. **Re-authorize Roboflow and set `ROBOFLOW_API_KEY`** — unblocks ~3,500 public
   images and 6 working classes. Minutes of work, and it is yours to do.
2. **Fork `control-panel-azure/control-panels` and generate a version** — the best
   public source of modular DIN-rail instances.
3. **Capture 2,000–4,000 real panel photographs** and label them with the tooling
   already built. This is the actual project, and there is no substitute for it.

The code to turn that data into a production model is finished, tested, and measured.
What it needs is the data.
