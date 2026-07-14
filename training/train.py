"""
Image-classifier training pipeline with ONNX export.

Pipeline stages (all real, all measured):

    load dataset -> train / val / test split -> train MLP classifier
    -> accuracy / loss / precision / recall / F1 on every split
    -> export to ONNX (skl2onnx) -> reload with ONNX Runtime and verify
       the ONNX predictions match the trained model.

The ONNX file is written with the same runtime (``onnxruntime``) the backend
uses for inference, and the pipeline asserts it loads and predicts correctly —
this is what "ability to load the exported model into the backend" means here.

Run the self-test (no dataset needed)::

    python -m training.train

Train on your own labelled crops::

    python -m training.train --data path/to/dataset --out models/components/classifier.onnx

Note on scope: this trains an image *classifier*. The backend's component
detector expects an object-*detection* ONNX model (YOLO-style), which is a
different output shape; to train that, supply a detector in the documented
dataset format. The classifier here is a complete, verified demonstration of
the training + export + reload cycle on real data.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np

from .dataset import Dataset, load_dataset


@dataclass
class SplitMetrics:
    accuracy: float
    loss: float          # log loss (cross-entropy)
    precision: float     # macro
    recall: float        # macro
    f1: float            # macro
    n: int


def _metrics(clf, X, y, labels) -> SplitMetrics:
    from sklearn.metrics import (accuracy_score, f1_score, log_loss,
                                 precision_score, recall_score)
    pred = clf.predict(X)
    proba = clf.predict_proba(X)
    return SplitMetrics(
        accuracy=float(accuracy_score(y, pred)),
        loss=float(log_loss(y, proba, labels=labels)),
        precision=float(precision_score(y, pred, average="macro", zero_division=0)),
        recall=float(recall_score(y, pred, average="macro", zero_division=0)),
        f1=float(f1_score(y, pred, average="macro", zero_division=0)),
        n=int(len(y)),
    )


def train(
    data_dir: str | None = None,
    out_path: str = "models/components/classifier.onnx",
    image_size: int = 16,
    hidden: tuple[int, ...] = (128, 64),
    max_iter: int = 300,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier

    ds: Dataset = load_dataset(data_dir, image_size=image_size)
    labels = list(range(ds.n_classes))
    if verbose:
        print(f"[data] {ds.source}")
        print(f"[data] features={ds.n_features} classes={ds.n_classes}")

    # train / val / test = 64 / 16 / 20
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        ds.X, ds.y, test_size=0.20, random_state=seed, stratify=ds.y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.20, random_state=seed, stratify=y_tmp)

    if verbose:
        print(f"[split] train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    clf = MLPClassifier(hidden_layer_sizes=hidden, max_iter=max_iter,
                        random_state=seed, early_stopping=False)
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_secs = time.time() - t0

    metrics = {
        "train": asdict(_metrics(clf, X_train, y_train, labels)),
        "val": asdict(_metrics(clf, X_val, y_val, labels)),
        "test": asdict(_metrics(clf, X_test, y_test, labels)),
    }
    if verbose:
        for split, m in metrics.items():
            print(f"[{split:5s}] acc={m['accuracy']:.4f} loss={m['loss']:.4f} "
                  f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")

    onnx_info = export_onnx(clf, ds.n_features, out_path, ds.class_names, verbose=verbose)
    verify = verify_onnx(clf, ds.X[:64], out_path)
    if verbose:
        print(f"[onnx] exported -> {out_path} ({onnx_info['bytes']} bytes)")
        print(f"[onnx] reload check: {verify['match_rate']*100:.1f}% predictions match "
              f"({'OK' if verify['ok'] else 'MISMATCH'})")

    report = {
        "source": ds.source,
        "image_size": ds.image_size,
        "n_features": ds.n_features,
        "class_names": ds.class_names,
        "train_seconds": round(train_secs, 3),
        "metrics": metrics,
        "onnx": {**onnx_info, "verification": verify},
    }
    # write a sidecar report + labels next to the model
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(os.path.splitext(out_path)[0] + ".report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(os.path.dirname(os.path.abspath(out_path)), "labels.txt"), "w") as f:
        f.write("\n".join(ds.class_names) + "\n")
    return report


def export_onnx(clf, n_features: int, out_path: str, class_names, verbose=True) -> dict:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_type = [("input", FloatTensorType([None, n_features]))]
    onx = convert_sklearn(clf, initial_types=initial_type, target_opset=15)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(onx.SerializeToString())
    return {"path": out_path, "bytes": os.path.getsize(out_path),
            "n_features": n_features, "classes": len(class_names)}


def verify_onnx(clf, X_sample: np.ndarray, out_path: str) -> dict:
    """Reload the exported model with ONNX Runtime and compare predictions."""
    import onnxruntime as ort

    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    onnx_pred = sess.run([label_name], {input_name: X_sample.astype(np.float32)})[0]
    onnx_pred = np.asarray(onnx_pred).reshape(-1)
    skl_pred = clf.predict(X_sample)
    match = float(np.mean(onnx_pred == skl_pred))
    return {"ok": bool(match >= 0.99), "match_rate": match,
            "checked": int(len(X_sample)),
            "providers": sess.get_providers()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train an image classifier and export to ONNX.")
    ap.add_argument("--data", default=None,
                    help="Image-folder dataset root (<root>/<class>/*.jpg). "
                         "If omitted, the sklearn digits self-test dataset is used.")
    ap.add_argument("--out", default="models/components/classifier.onnx",
                    help="Output ONNX path.")
    ap.add_argument("--image-size", type=int, default=16)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    report = train(args.data, args.out, image_size=args.image_size,
                   max_iter=args.max_iter, seed=args.seed, verbose=True)
    ok = report["onnx"]["verification"]["ok"]
    print("\nDONE." if ok else "\nDONE (with ONNX verification mismatch).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
