# Audit & redesign — Madkour AI Panel Inspector

Why the previous system reported hundreds of wires and recognised no components,
what was rebuilt, and what is now measured rather than asserted.

Reproduce every number in this document with:

```bash
python scripts/validate_panel_inspector.py --images 25 --json validation.json
python -m pytest tests -q
```

---

## 1. Root cause

The reported behaviour — *"detects hundreds of fake wires, cannot recognise a
single component"* — was not a tuning problem. It was the exact, predictable
output of the configuration that shipped. Four independent defects compounded.

### 1.1 The default configuration guaranteed the symptom

`rtsp_backend/ai/manager.py` shipped these defaults:

```python
"components": ("onnx_components", ...),   # requires_weights = True
"wires":      ("advanced_wires",  ...),   # requires_weights = False
```

`onnx_components` needs a trained checkpoint. No checkpoint existed anywhere in
the repository, so the component task loaded, reported `weights_missing`, and
returned an empty list — forever. `advanced_wires` needs no weights, so it ran on
every frame and returned whatever its classical tracer found.

The result: **components = 0, wires = hundreds.** "The AI behaves as though every
object inside the panel is simply a wire" was a literal, accurate description of
the shipped configuration.

### 1.2 The wire tracer cannot distinguish a conductor from a cabinet seam

`rtsp_backend/panels/wire_detector.py` (707 lines) segments by HSV/LAB colour,
adaptive-thresholds, skeletonises and Hough-transforms the **whole frame**. Every
one of those operators fires on high-contrast elongated structure. A control panel
is made almost entirely of high-contrast elongated structure: DIN rails, cable
duct lips, device housing edges, label borders, busbars, panel seams, shadow
boundaries and window reflections. Meanwhile the actual conductors are mostly
inside the ducting and invisible.

Measured on 25 synthetic panels that contain **zero** conductors
(`scripts/validate_panel_inspector.py`, experiment A):

| Backend | False "wires" | Mean / image | Worst image | Precision |
|---|---|---|---|---|
| `advanced_wires` (was the default) | 715 | 28.6 | 57 | 0.00 |
| `classical_wires` | 4 494 | 179.8 | 200 (the internal cap) | 0.00 |
| Redesigned inspection path | **0** | 0.0 | 0 | n/a — disabled |

Ground truth is zero, so precision is exactly zero. There is no threshold that
fixes this: the detector is measuring the wrong thing.

`ClassicalWireAnalyzer` also truncated at `lines[:200]`, so the true count on a
detailed photograph was higher than reported — the cap hid the scale of it.

### 1.3 The component decoder shifted every class label

`rtsp_backend/ai/detectors.py` chose its output format from the raw column count:

```python
if row.shape[0] >= 85:      # "YOLOv5: x,y,w,h,obj,80cls"
    ...                      # obj * class score
else:                        # "YOLOv8: x,y,w,h,80cls"
    cls_scores = row[4:]     # <-- includes objectness for a v5 export
```

`>= 85` is only true for an **80-class COCO** model. A YOLOv5 export of any
electrical class set has `4 + 1 + nc` columns — 32 for the old 27-class list, 58
for today's 53-class taxonomy — all below 85. So it took the YOLOv8 branch,
`argmax`'d over a slice that begins with the objectness column, and objectness
(typically ~0.95) won almost every time.

Measured (experiment C), 53 classes / 58 columns, 60 predictions:

| Decoder | Correct labels | Label accuracy |
|---|---|---|
| Old column-count heuristic | 1 / 60 | **0.017** |
| New declared-class-count logic | 60 / 60 | **1.000** |

Every component would have been reported as class 0 — even after a successful
training run. This one defect alone made the training pipeline pointless.

### 1.4 `models/components/labels.txt` contained the numbers 0–9

The shipped label file was ten lines: `0`, `1`, … `9`. `_load_labels()` preferred
it over the class list, so had a model ever loaded, its components would have been
named `"0"`…`"9"`. Deleted; replaced by `models/components/classes.json`, which is
generated from the taxonomy and pinned by a test.

### 1.5 Post-processing had one filter, and it was the wrong one

The only gate was a single global `conf=0.25`, followed by one **class-agnostic**
NMS across all classes at once.

- A global threshold cannot express that a large obvious VFD and a 20-pixel
  indicator lamp need different evidence.
- There was no geometric check, so a 900×5 pixel sliver could be reported as a
  PLC.
- Class-agnostic NMS actively destroys correct detections in a panel: an overload
  relay is *bolted underneath* its contactor and overlaps it heavily, so one of
  the two was always suppressed. Same for a CT around a busbar, or a relay in its
  socket.
- There was no honest-uncertainty path. Every activation above 0.25 became a
  confident named component.

### 1.6 No panel-level understanding existed at all

Nothing in the repository inferred panel type, panel function, expected bill of
materials, or component relationships. `panel_svc.analyze` counted labels and
drew boxes. There was no ontology: the 27-string `ELECTRICAL_CLASSES` list
carried no knowledge of what a contactor *is*, so nothing downstream could reason
about it.

### 1.7 Secondary defects found and fixed

| Defect | Impact | Fix |
|---|---|---|
| Per-row Python loop over ~8 400 decoder outputs | ~100× slower than needed on every frame | Vectorised NumPy decode |
| Decoded boxes never clipped to the frame | Overlay boxes ran off-image | `sanitise()` clips and drops degenerate boxes |
| `np.squeeze` on the detector output | A **single**-detection output collapsed to 1-D and was silently dropped — a panel with one recognised device returned nothing | Batch dims stripped explicitly; 1-D reshaped |
| No RT-DETR support despite it being advertised | Two-tensor exports returned nothing | `decode_rtdetr()` handles both layouts and raw-vs-sigmoid logits |
| `labels.txt` of integers preferred over the class list | Components named `"0"`…`"9"` | Numeric-only files rejected with a warning |

---

## 2. What was rebuilt

New package `rtsp_backend/electrical/`:

| Module | Responsibility |
|---|---|
| `taxonomy.py` | 53-class domain knowledge base. Per class: engineering function, panel role, electrical domain, mounting style, aspect-ratio and relative-area priors, dataset aliases, zero-shot prompts, per-class confidence threshold, companion devices |
| `postprocess.py` | The six-stage suppression cascade with per-stage drop accounting |
| `recognizer.py` | Inference backends: trained ONNX / Ultralytics, zero-shot OWLv2 / Grounding DINO / Florence-2, and a corroboration-weighted ensemble |
| `nameplate.py` | 100+ manufacturer part-number signatures → manufacturer, product family, and a cross-check against the detector's class |
| `expert.py` | Per-component engineering record; context-sensitive purpose; bill of materials; row layout description |
| `panel_type.py` | 12 panel archetypes as weighted evidence rules; application inference; expected-BOM gap analysis; maintenance observations |
| `inspector.py` | The engine that composes the above into a result, an overlay and a report |
| `metrics.py` | Precision / recall / F1 / AP / mAP@50 / mAP@50-95, confusion matrix with background row+column, FP cause analysis, FN analysis, per-class threshold optimiser, model comparison |

New training stack `training/electrical/`:

| Module | Responsibility |
|---|---|
| `datasets.py` | Curated public-source registry with per-source label maps; YOLO label-space remapping onto the taxonomy; multi-dataset merge; trainability/coverage report; the Madkour field-capture protocol |
| `synthetic.py` | Labelled data generation: composition from real device crops (the useful mode) and procedural stand-ins (pipeline validation); lighting, perspective, rotation, occlusion, dust, shadow, reflection, blur, JPEG artefacts |
| `train.py` | Ultralytics driver for YOLOv11 / YOLOv8 / RT-DETR / YOLOv12-if-available, panel-specific augmentation recipe, ONNX export with a pinned class map, and the benchmark that ranks architectures on measured mAP |
| `cli.py` | `plan · synth · remap · merge · analyse · train · bench · eval · tune` |

### 2.1 The suppression cascade

Ordered, and every stage records why it dropped a box:

1. **Sanitise** — clip to frame, drop degenerate/non-finite/zero-score boxes.
2. **Per-class NMS** — within each class only, so stacked devices survive.
3. **Cross-class dedupe** — resolve a double claim on the same device, but *only*
   between genuinely confusable classes (mcb/mccb/acb, relay/timer/safety…).
   Overlap between unrelated classes is preserved, because in a real panel
   devices are physically stacked and nested.
4. **Geometric plausibility** — reject boxes whose aspect ratio or relative area
   cannot be that device, using the taxonomy priors.
5. **Confidence gate** — per-class threshold. Above it, the class is asserted.
   Between the threshold and the floor, the detection is kept but relabelled
   **Unknown Industrial Component**. Below the floor it is dropped. The system
   never guesses.
6. **Row grouping** — cluster into DIN-rail rows and return results in reading
   order.

Measured (experiment B): 25 panels, 935 ground-truth devices, 1 769 raw
candidates from a detector simulator that reproduces the three real
false-positive modes.

| | Precision | Recall | F1 | mAP@50 | mAP@50-95 | False positives |
|---|---|---|---|---|---|---|
| Old logic (global conf + class-agnostic NMS) | 0.591 | 0.782 | 0.673 | 0.762 | 0.554 | 505 |
| New cascade | **0.916** | 0.766 | **0.834** | 0.750 | 0.544 | **66** |
| Δ | +0.324 | −0.016 | +0.161 | −0.012 | −0.010 | **−86.9 %** |

Drops by cause: implausible aspect ratio 409, same-class NMS 207, implausible
too large 121, degenerate box 83, duplicate class claim 56, too small 3, below
floor 2.

**The two regressions are real and are the intended trade.** Recall falls 1.6
points and mAP@50 falls 1.2 points, because mAP integrates the whole
precision–recall curve *including* the low-confidence tail that the gate removes,
and because a handful of genuine devices sit near a prior boundary. In exchange,
precision at the operating point rises 32 points and false positives fall by a
factor of 7.6. For industrial inspection that is the correct direction — as the
brief puts it, better to correctly identify 20 real components than to
incorrectly detect 300 fake ones. The `strictness` parameter moves the whole
operating point if a deployment wants it elsewhere, and
`training/electrical/cli.py tune` derives the per-class thresholds from a real
validation set instead of leaving them at their hand-set defaults.

### 2.2 Panel understanding

Panel type comes from weighted evidence over the component inventory, with
keystone requirements, diminishing returns per additional device of the same
class, and explicit counter-evidence. Every verdict carries the evidence that
produced it, and the classifier refuses when the evidence is thin.

Measured (experiment D): 12 hand-written inventories drawn from panel-engineering
practice, one per archetype — **12/12 top-1 correct**. Four deliberately
ambiguous inventories — **4/4 honestly refused** as `unclassified` rather than
guessed.

That measures whether the rule base encodes the right reasoning. It does not
measure field accuracy, which depends on the detector feeding it.

---

## 3. Wiring detection: removed, and why it stays removed

Per the brief's highest-priority instruction, wiring detection is gone from the
product path:

- `manager.py` default for the `wires` task is `null_wires`, and the task is
  disabled.
- A **startup migration** rewrites any persisted `advanced_wires` /
  `classical_wires` selection to `null_wires` and disables it, so an existing
  installation stops producing phantom wires on upgrade without operator action.
  The migration is reported in `GET /api/ai/models`.
- `panel_svc.analyze` has no wire stage. The result carries
  `wire_analysis: {enabled: false, reason: ...}` so the UI and the report state
  the decision rather than showing a silent zero.
- `panels/template.py` (reference-panel learning) no longer traces wires unless
  `RTSP_ENABLE_WIRE_TRACING=1` or an explicit `wire_params={"enabled": True}` is
  passed.
- Both tracers remain **registered and flagged `experimental`** with a warning
  string surfaced in the backend catalogue. They are kept for research
  reproducibility, not for use.

Wire analysis returns when there is a trained, quantitatively validated
instance-segmentation model behind it — not before.

---

## 4. What is proven, and what is not

Being precise about this is the point.

**Proven by measurement in this repository**

- Wiring false positives: 715 / 4 494 → **0** (experiment A).
- Gate false positives −86.9 %, precision +0.324, F1 +0.161 (experiment B).
- ONNX label accuracy 0.017 → **1.000** (experiment C).
- Panel-type reasoning 12/12, honest refusal 4/4 (experiment D).
- 340 tests pass, including the pre-existing face-recognition suite unchanged.

**Not proven, and not claimed**

- **Recognition accuracy on real Madkour panels.** No trained checkpoint exists
  and none could be produced here: this environment has no GPU, no access to
  dataset hosts (`huggingface.co` and `universe.roboflow.com` are unreachable),
  and no labelled photographs of Madkour panels. The recogniser therefore reports
  `weights_missing` and returns **zero components** — honestly — until a model is
  supplied. Nothing is fabricated to fill the gap.
- **Zero-shot performance.** The OWLv2 / Grounding DINO / Florence-2 backends are
  implemented against the taxonomy's prompts and fail with a precise reason here
  because the model hub is unreachable. On a connected host they are the fastest
  route to non-zero recognition without a custom dataset, but their accuracy on
  panel imagery is unmeasured.
- **Synthetic-data results are not field results.** The procedural generator
  validates the pipeline on exact ground truth. It is not photorealistic and a
  model trained only on it will not generalise. Every generated dataset carries
  a `warning` field saying so, and `manifest.source` distinguishes procedural
  from real-crop-composed data.

**The path to closing the gap** is in `training/electrical/README.md`: build a
crop library from real device photographs, multiply it with
`compose_from_crops`, bootstrap the common modular classes from public data,
capture real Madkour panels under the documented protocol, then
`train → bench → eval → tune` and re-run the validation harness. `datasets.py
coverage_report()` names in advance which classes will and will not work.

---

## 5. Files changed

**New** — `rtsp_backend/electrical/*` (8 modules), `training/electrical/*`
(4 modules), `scripts/validate_panel_inspector.py`,
`models/components/classes.json`, `docs/AUDIT_PANEL_INSPECTOR.md`,
`training/electrical/README.md`, tests
`test_electrical_{taxonomy,postprocess,intelligence,recognizer,training}.py`.

**Rewritten** — `rtsp_backend/panel_svc.py` (component-centric, no wires).

**Modified** — `ai/manager.py` (defaults + startup migration),
`ai/registry.py` (deprecated/experimental flags), `ai/components.py` and
`ai/wires.py` (flagged, with the reasons), `panels/template.py` (wire tracing
opt-in), `api/analysis.py` (taxonomy / panel-type / nameplate endpoints),
`api/panels.py` (richer summary), `inspection_svc.py` (structured model flag),
plus the frontend rename and the new Panel Inspector page.

**Deleted** — `models/components/labels.txt` (the numeric placeholder; a test
now prevents its return).
