"""Tests for the standalone training pipeline (training/)."""

from __future__ import annotations

import json
import os

import numpy as np


def test_self_test_pipeline_trains_and_exports(tmp_path):
    from training.train import train

    out = str(tmp_path / "models" / "classifier.onnx")
    report = train(data_dir=None, out_path=out, max_iter=80, verbose=False)

    # every split reports the full metric set
    for split in ("train", "val", "test"):
        m = report["metrics"][split]
        assert set(m) == {"accuracy", "loss", "precision", "recall", "f1", "n"}
        assert 0.0 <= m["accuracy"] <= 1.0
        assert m["loss"] >= 0.0

    # the pipeline actually learns on real data
    assert report["metrics"]["test"]["accuracy"] > 0.85
    assert report["metrics"]["test"]["f1"] > 0.85

    # ONNX artefact exists and reloads with matching predictions (backend engine)
    assert os.path.isfile(out)
    assert report["onnx"]["bytes"] > 0
    assert report["onnx"]["verification"]["ok"] is True
    assert report["onnx"]["verification"]["match_rate"] >= 0.99

    # sidecar report + labels written next to the model
    assert os.path.isfile(os.path.splitext(out)[0] + ".report.json")
    labels = (tmp_path / "models" / "labels.txt").read_text().split()
    assert labels == [str(i) for i in range(10)]


def test_exported_onnx_runs_under_onnxruntime(tmp_path):
    import onnxruntime as ort
    from training.train import train

    out = str(tmp_path / "m" / "clf.onnx")
    train(data_dir=None, out_path=out, max_iter=60, verbose=False)

    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    # 64 features (8x8 digits); a zero vector must produce a valid class prediction
    pred = sess.run(None, {name: np.zeros((1, 64), dtype=np.float32)})[0]
    pred = np.asarray(pred).reshape(-1)
    assert pred.shape == (1,)
    assert 0 <= int(pred[0]) <= 9


def test_image_folder_loader_roundtrip(tmp_path):
    import cv2
    from sklearn.datasets import load_digits

    from training.dataset import load_image_folder
    from training.train import train

    # write real digit images to disk as an image-folder dataset (3 classes)
    digits = load_digits()
    root = tmp_path / "dataset"
    per_class = 40
    counts = {0: 0, 1: 0, 2: 0}
    for img, label in zip(digits.images, digits.target):
        label = int(label)
        if label in counts and counts[label] < per_class:
            d = root / f"class_{label}"
            d.mkdir(parents=True, exist_ok=True)
            # scale 0..16 -> 0..255 and save as PNG
            arr = (img / 16.0 * 255).astype(np.uint8)
            cv2.imwrite(str(d / f"{counts[label]}.png"), arr)
            counts[label] += 1

    ds = load_image_folder(str(root), image_size=8)
    assert ds.class_names == ["class_0", "class_1", "class_2"]
    assert ds.n_classes == 3
    assert ds.X.shape[1] == 64
    assert ds.X.min() >= 0.0 and ds.X.max() <= 1.0
    assert len(ds.y) == per_class * 3

    # and the folder dataset trains + exports through the same pipeline
    out = str(tmp_path / "out" / "folder.onnx")
    report = train(data_dir=str(root), out_path=out, image_size=8, max_iter=80, verbose=False)
    assert report["onnx"]["verification"]["ok"] is True
    assert report["metrics"]["test"]["accuracy"] > 0.7  # real, learnable signal
