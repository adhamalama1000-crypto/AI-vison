# Domain Transfer: Synthetic → Real Industrial Panels

The class-reduction experiment proved the recipe works. It did not produce a shippable
model, and this document is about the difference between those two statements.

## What the synthetic result does and does not mean

The 8-class run reaches high mAP on procedurally-rendered panels. That number validates
the *pipeline*: the label space is consistent, the splitter is not leaking, the trainer
converges, the exporter round-trips, and the API serves what the model produces. Those
were open questions and they are now closed.

It says nothing about a photograph of a real cabinet. Procedural renders have flat
shading, hard edges, a constant-colour background, and no dust, specular highlights,
cable shadows, motion blur, sensor noise, or depth-of-field falloff. A model that scores
0.85 on them has learned "dark rectangle on grey". Real MCBs are photographed in
torchlight, at an angle, behind a wire duct, half-occluded by a loom.

**The synthetic model is not production ready and must not be deployed.** Measure the
gap before anything else — `cli domain-gap` exists to put a number on it.

## The one rule

**Validate on real images only.**

Fine-tuning on real data while validating on a synthetic/real mix produces a rising
metric and a model that has not improved. The synthetic validation images are easy, they
dominate the mean, and the number goes up while real-world performance does not. Every
tool here enforces this: `build_mixed` takes validation and test *exclusively* from the
real root, by construction, and refuses to run if the real dataset has no `val` split.

## Why the obvious plan might be wrong

The intuitive route is: pretrain on synthetic, fine-tune on real. It is **frequently
worse than fine-tuning straight from COCO**, and the reason is worth understanding
rather than working around.

A COCO-pretrained backbone has seen millions of real photographs. Its early layers
encode real optics — soft shadows, specular highlights, sensor noise, defocus. A
backbone pretrained on a few hundred procedural renders has instead learned features
tuned to flat fills and hard synthetic edges, and fine-tuning has to *unlearn* those
before it can learn anything useful. Sim-to-real pretraining pays off when the synthetic
data is photorealistic and abundant — domain-randomised renders in the tens of
thousands. Flat-shaded procedural panels are neither.

So the comparison is run, not assumed. `cli transfer` trains each candidate and ranks
them on the same real-only validation split, **through the production path, at each
strategy's own best operating point**:

| Strategy | What it is |
|---|---|
| `synth_only` | Trained on synthetic alone, evaluated on real. **The zero-transfer baseline** — what shipping the synthetic model would actually deliver. Expected to lose badly; included because that loss is the number justifying the real-data programme. |
| `real_only` | COCO → real. **The control.** If it wins, the synthetic data contributed nothing and the pipeline is simpler without it. |
| `coco_to_synth_to_real` | Synthetic pretrain, then real fine-tune. The intuitive plan. Included because it *might* win. |
| `coco_to_real` | The same route as `real_only`, named explicitly for when someone says "just train on real data". |
| `mixed` | Synthetic images mixed into the real training set at a capped fraction (default 30%), validated on real only. Often the best when real data is scarce: the synthetic images add shape and layout variety without getting a vote on what a photograph looks like. |

The default is the three-arm validation: `synth_only`, `real_only`,
`coco_to_synth_to_real`. **The winner becomes the base model.** Do not expand the class
set until this comparison has been run — a wider label space on top of an unvalidated
transfer story just makes the result harder to attribute.

Each strategy is scored at its own best threshold rather than at one fixed value.
Comparing at a fixed threshold ranks models on where the default happened to land
relative to their calibration, not on which one detects better — a well-calibrated model
whose scores sit low would lose to a worse one for no real reason.

The ranking reports `synthetic_data_helped` as an explicit boolean, and it requires the
winner to beat the control by **more than 0.02 mAP@50** before it says yes. With a small
real validation set anything under that is noise, and calling noise an improvement is
how a pipeline acquires a stage that costs GPU hours and buys nothing.

## Workflow

Both datasets must already share one label space. Run `cli scope` on each with the same
profile first, or the class indices mean different things and the merge is silently
wrong — the model trains on mislabelled data and the metrics look fine.

```bash
# 0. Put both datasets on the same label space (core8 first, per the staged plan).
python -m training.electrical.cli scope --src data/real_raw  --dst data/real  --profile core8
python -m training.electrical.cli scope --src data/synth_raw --dst data/synth --profile core8

# 1. Measure the gap the synthetic model actually has. This is the baseline every
#    later claim is measured against.
python -m training.electrical.cli domain-gap \
    --weights runs/electrical/core8_recipe/weights/best.pt \
    --synth data/synth --real data/real \
    --out reports/domain_gap.json

# 2. Build a mixed training set with REAL-ONLY validation.
python -m training.electrical.cli mix \
    --real data/real --synth data/synth --dst data/transfer_mixed \
    --synth-fraction 0.3

# 3a. Fine-tune one route (staged: freeze the backbone, then unfreeze at half the LR).
python -m training.electrical.cli finetune \
    --data data/transfer_mixed/dataset.yaml \
    --init-from runs/electrical/core8_recipe/weights/best.pt \
    --arch yolo11s --epochs 60 --device 0 \
    --out reports/finetune.json

# 3b. Or compare every route and keep the winner.
python -m training.electrical.cli transfer \
    --real data/real --synth data/synth \
    --work-dir runs/electrical/transfer \
    --strategies real_only coco_to_synth_to_real mixed \
    --arch yolo11s --epochs 40 --device 0 \
    --out reports/transfer_comparison.json

# 4. Export the winner and verify the label space is the one it was trained on.
python -m training.electrical.cli export \
    --weights runs/electrical/transfer/.../best.pt \
    --data data/transfer_mixed/dataset.yaml --out dist/model
python -m training.electrical.cli verify --bundle dist/model
```

### Why fine-tuning is staged

`finetune` defaults to freeze-then-unfreeze: the backbone is frozen for the first third
of the epoch budget so the detection head adapts to the new domain without the
pretrained features being destroyed by early large gradients, then everything unfreezes
at half the learning rate. On a small real dataset this is meaningfully better than full
fine-tuning from step one, which tends to wash out the pretrained features before the
head is producing useful gradients.

`--lr0` defaults to 0.002, well below the from-scratch 0.01. Fine-tuning at a
training-scale learning rate is the most common way to destroy a good checkpoint.

## Blocker: no real images are reachable from this environment

The machinery above is complete, tested and runnable. It has not been *run on real data*,
because there is currently no way to obtain real data here. Every acquisition route was
checked, not assumed:

| Route | State |
|---|---|
| Roboflow MCP | `token expired` — needs interactive re-authorization, unavailable in a non-interactive session |
| Roboflow REST API | no `ROBOFLOW_API_KEY` in the environment or `.env`; `api.roboflow.com` and `universe.roboflow.com` are both unreachable through the proxy (curl returns `000`) |
| Kaggle | no `~/.kaggle/kaggle.json` token |
| Open Images (FiftyOne) | `storage.googleapis.com` is reachable, but Open Images has no electrical-panel component classes |

**To unblock, one of these is needed:**

1. Re-authorize the Roboflow connector, **or**
2. Set `ROBOFLOW_API_KEY` in the gitignored `.env` — never pasted into chat — *and* allow
   `api.roboflow.com` in the environment's network policy, **or**
3. Provide your own panel photographs. This is the better option regardless: your own
   cabinets are the actual target distribution, and public switchgear sets are mostly
   product photography rather than installed panels.

Once any of those is in place the workflow below runs unchanged.

## Real-data requirements

The gap closes with real images and nothing else. Per-class instance counts needed on
**real** data, from `profiles.requirement_estimate`:

| Target mAP@50 | Instances per class | 8 classes | 15 classes |
|---|---|---|---|
| 0.50 | 150–300 | 1,200–2,400 | 2,250–4,500 |
| 0.70 | 300–600 | 2,400–4,800 | 4,500–9,000 |
| **0.85** | **700–1,200** | **5,600–9,600** | **10,500–18,000** |
| 0.92 | 1,500–2,500 | 12,000–20,000 | 22,500–37,500 |

At roughly 12 instances per panel photograph, the 0.85 target on 15 classes needs about
**3,300–5,600 real images**. Validation needs at least 50 real images per class before a
per-class AP means anything; below that, a 5-point mAP difference between two strategies
is not a difference.

## Staged class expansion

Expand only after the previous stage holds up on real validation data. Adding classes
before the current set converges spreads a fixed amount of data thinner and the mean
drops for reasons that have nothing to do with the new classes.

The path is `core8 → core15 → core18`, and each profile is a **strict positional prefix**
of the next:

| Profile | Indices | Classes |
|---|---|---|
| `core8` | 0–7 | mcb, mccb, contactor, relay, plc, terminal_block, power_supply, vfd |
| `core15` | + 8–14 | fuse, transformer, busbar, wire_duct, emergency_stop, selector_switch, indicator_lamp |
| `core18` | + 15–17 | overload_relay, din_rail, circuit_breaker |

That prefix property is the point, not a coincidence. A detection head is positional, so
because indices 0–7 keep their meaning, a core8 checkpoint fine-tunes onto core15 and
only the seven new classes start from nothing:

```bash
# See what the step involves before running it.
python -m training.electrical.cli expand --from core8 --to core15

# Then run it, carrying the core8 head forward.
python -m training.electrical.cli expand --from core8 --to core15 \
    --data data/core15/dataset.yaml \
    --init-from runs/electrical/core8_recipe/weights/best.pt \
    --epochs 60 --device 0
```

`expand` **refuses** to carry a checkpoint across a profile change that is not
index-stable — `core8 → full`, for instance, because `full` follows taxonomy order and
does not begin with the core8 classes. Reusing a head there produces a model that loads
without complaint and predicts the wrong labels, so it exits non-zero rather than
warning. Drop `--init-from` to train a fresh head instead.

Expect the existing classes to dip for the first few epochs after an expansion while the
widened head settles. That is normal; what matters is whether they recover to their
previous level by the end of the run. If they do not, the new classes are too thin and
should be held back.

`core8` pins the exact label order the existing checkpoint was trained against, and a
test asserts it. `CLASS_ORDER` and the profiles are **append-only** for the same reason:
inserting a class in the middle silently invalidates every checkpoint trained on it.

> One historical note, since it affects any checkpoint from before this change: `core15`
> was reordered so that `core8` is its prefix. `vfd` moved from profile index 9 to 7.
> This was safe only because no `core15` model had been trained yet. Any future reorder
> would not be.

## Acceptance is decided on the production path

`cli accept` is the gate. It evaluates through the registered component backend —
same preprocessing, same `decode_floor`, same per-class thresholds, same NMS and
cross-class suppression, same geometric plausibility gate as `POST /api/panel/analyze` —
and sweeps the operating point so the threshold is a measurement rather than a default.

```bash
python -m training.electrical.cli accept \
    --weights runs/electrical/<run>/weights/best.pt \
    --root data/real --split val --imgsz 640 \
    --confs 0.01 0.03 0.05 0.10 0.20 \
    --objective f1 --max-fp-per-image 1.0 \
    --target-map50 0.85 \
    --out reports/acceptance.json --csv reports/sweep.csv
```

It reports, per operating point: production mAP@50, mAP@50-95, precision, recall, F1,
TP/FP/FN, **false positives per image**, **false negatives per image**, and the number of
predictions that were demoted to unknown. Then it picks the best point against
`--objective` subject to any constraints, and prints per-class AP at that point.

`--target-map50` makes the run **exit non-zero when the target is unmet**, so a CI job
cannot pass by ignoring the shortfall.

### Sweeping "confidence" means moving three gates, not one

There are three thresholds in series and the highest one does the cutting:

| Gate | Production default | Effect |
|---|---|---|
| `decode_floor` | 0.10 | Nothing below it is decoded at all |
| `unknown_floor` | 0.18 | Below it nothing is kept, not even as unknown |
| per-class `thresholds` | taxonomy `min_conf` | Below it a box is demoted to unknown rather than named |

Lowering only `decode_floor` to 0.01 therefore measures nothing — `unknown_floor` at
0.18 still discards everything beneath it, and 0.01/0.03/0.05/0.10 come back identical.
That reads as a plateau in the model and is really a misconfigured harness.
`production_params` moves all three together and pins `strictness` to 1.0, which is what
a single "conf = 0.05" operating point has to mean. The per-class shape from the taxonomy
is deliberately flattened during a sweep so the axis is one-dimensional; recover it
afterwards with `cli tune`, which fits a threshold per class.

### Why the FP count includes unknowns

Lowering the threshold does not only add true positives. It adds boxes the gate cannot
name, accepted as `unknown_industrial_component` — which matches no ground-truth class
and so scores as a false positive. That is the honest behaviour (the alternative is
inventing a label), but it means `fp_per_image` at a low threshold is partly unknowns
rather than misclassifications. Every row reports `unknown_predictions` separately so the
two causes stay distinguishable.

### Choosing an objective

F1 is the default, and it is the wrong choice if you have not thought about it. For an
inspection report the errors are not symmetric: a false positive puts a device on a
report that is not in the panel, costing an engineer a site visit to disprove, while a
false negative is a gap a human reviewer can still catch. If that asymmetry matters, say
so with `--min-precision` or `--max-fp-per-image` rather than trusting F1 to encode it.
When no operating point satisfies the constraints, the report says so explicitly and
flags the returned point as not acceptable rather than dropping the constraint silently.

### What the first production-path run actually caught

The first time the sweep ran, production recall came back at 0.245 against a training
recall of 0.95. That ratio was the clue: 0.245 ≈ 2/8, and `core8` shares exactly two
index positions with the 54-class taxonomy (`mcb` at 0, `mccb` at 1).

`load_ground_truth` was reading YOLO label indices through `taxonomy.CLASS_ORDER`
unconditionally. For a profile-scoped dataset, index 2 is `contactor`; in the taxonomy it
is `acb`. So six of core8's eight classes were being compared against the wrong names.
Nothing errored — labels loaded, evaluation ran, and the score came out low in a way that
looked exactly like a weak model.

Worth being precise about the blast radius: **this was an evaluation bug, not a serving
bug.** Both serving paths were already correct — the Ultralytics backend reads `names`
from the checkpoint, and the ONNX backend reads `classes.json` from the bundle. Live
traffic would have been labelled correctly. What the bug would have done is make every
acceptance decision wrong, by understating the model and misattributing the cause.

`load_ground_truth` now reads the dataset's own `names` block via
`dataset_label_space()` and falls back to the taxonomy only when a dataset declares no
label space of its own. This is the class of defect that only a production-path
evaluation surfaces, and it is the argument for making that path the acceptance gate.

## Two different mAP numbers, and which one to believe

A checkpoint has two legitimate mAP values, and they can differ enormously:

- **Training-time mAP** — what Ultralytics prints each epoch and writes to `results.csv`.
  Computed by its own validator at a confidence floor of ~0.001, which counts detections
  no production system would ever surface.
- **Served mAP** — what `cli eval` and `cli domain-gap` report, because they go through
  the registered production backend. That decodes at `conf=0.10` and then applies the
  per-class gates, so it measures what the API will actually return.

Measured on the same checkpoint mid-run: training-time mAP@50 **0.63** with recall 0.92,
served mAP@50 **0.151** with recall 0.16. Nothing is broken — the model's true positives
were mostly scoring below 0.10 at that point, so the production gate discarded them.

Both numbers matter, and they answer different questions. Training-time mAP tells you
whether learning is progressing. **Served mAP is the only one that describes the
product**, and it is the one to quote in an acceptance decision. A report that cites the
training number as evidence of production readiness is overstating the model, which is
why `cli eval` deliberately routes through the real backend rather than calling the
trainer's validator.

If served recall is far below training recall at the end of a run, the fix is more or
better data — not lowering the gate. Lowering it converts the shortfall into false
positives, which in an inspection report is worse than a miss.

## What "done" looks like

- `domain-gap` measured and recorded *before* fine-tuning, so the improvement is
  attributable.
- The winning strategy chosen by `cli transfer` on real-only validation, not by
  intuition.
- mAP@50 and recall reported on **real** validation images, with per-class AP, the
  confusion matrix, and FP/FN counts.
- `cli verify` passing on the exported bundle, with the label space matching the profile
  the model was trained against.
- Unknown gating intact: a component the model is not confident about comes back as
  `unknown_industrial_component`, and a panel it cannot assess comes back with risk
  `unknown` — never `low`.
