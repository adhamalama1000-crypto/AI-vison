# Training pipeline

A complete, tested image-classifier training pipeline that exports an ONNX model
and verifies the exported model reloads and runs under **ONNX Runtime** — the
same engine the backend uses for inference.

## Install

```bash
pip install -r requirements-train.txt
```

(The runtime backend does **not** need these; it only needs `onnxruntime`, which
is already in `requirements.txt`, to load an exported model.)

## Run the self-test (no dataset required)

```bash
python -m training.train
```

This trains on scikit-learn's `load_digits` dataset — 1,797 **real** 8×8
handwritten-digit images across 10 classes — and prints per-split metrics, e.g.:

```
[data] sklearn load_digits (self-test) — 1797 real 8x8 images, 10 classes
[split] train=1149 val=288 test=360
[train] acc=1.0000 loss=0.0025 P=1.0000 R=1.0000 F1=1.0000
[val  ] acc=0.9861 loss=0.0496 P=0.9879 R=0.9860 F1=0.9864
[test ] acc=0.9778 loss=0.0703 P=0.9778 R=0.9775 F1=0.9775
[onnx] exported -> models/components/classifier.onnx (70338 bytes)
[onnx] reload check: 100.0% predictions match (OK)
```

The digits dataset is a **demonstration** so the whole pipeline can be exercised
and validated with real data when you don't yet have your own — nothing is
fabricated, and it is always labelled as the self-test source.

## Train on your own data

Lay images out as one folder per class:

```
dataset/
  circuit_breaker/  img001.jpg  img002.jpg ...
  contactor/        ...
  relay/            ...
```

Then:

```bash
python -m training.train --data dataset --out models/components/classifier.onnx --image-size 32
```

## What the pipeline does

1. **Dataset loading** — image-folder (grayscale, resized, normalised to `[0,1]`)
   or the digits self-test (`training/dataset.py`).
2. **Split** — stratified train / validation / test (64 / 16 / 20).
3. **Train** — an `MLPClassifier` (scikit-learn).
4. **Metrics** — accuracy, loss (cross-entropy / log-loss), and macro
   precision, recall, and F1 on **all three** splits.
5. **ONNX export** — via `skl2onnx`, written next to a `labels.txt` and a
   `*.report.json` metrics sidecar.
6. **Reload verification** — the exported file is reloaded with `onnxruntime`
   and its predictions are asserted to match the trained model (≥ 99%).

## Relationship to the backend

The exported `.onnx` is a real model that ONNX Runtime loads and runs — the test
`tests/test_training.py::test_exported_onnx_runs_under_onnxruntime` proves this
with the backend's engine.

This trains an image **classifier**. The backend's *component-detection* backend
expects an object-**detection** model (YOLO-style output: boxes + classes),
which is a different output shape. To activate detection, drop a detector
trained in that format into `models/detection/` or `models/components/` and
select it on the AI Models page — see `models/README.md`. The classifier here is
a faithful, end-to-end demonstration of the train → export → reload cycle, run
and verified on real data.
