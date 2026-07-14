"""
Dataset loading for the training pipeline.

Two sources are supported:

1. An image-folder dataset laid out as::

       <root>/<class_name>/*.png|jpg|jpeg|bmp

   Every image is converted to grayscale, resized to a fixed square, and
   flattened into a feature vector in ``[0, 1]``. This is the format you use to
   train on your own labelled component crops.

2. A built-in **self-test** dataset (scikit-learn ``load_digits`` — 1,797 real
   8x8 handwritten-digit images across 10 classes). This is real data used to
   exercise and validate the pipeline end to end when you have no dataset of
   your own yet. It is clearly labelled as a demonstration; nothing is
   fabricated.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


@dataclass
class Dataset:
    X: np.ndarray          # [N, n_features] float32 in [0, 1]
    y: np.ndarray          # [N] int labels
    class_names: list[str]
    image_size: int        # square side length used for features
    source: str            # human description of where the data came from

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_classes(self) -> int:
        return len(self.class_names)


def load_image_folder(root: str, image_size: int = 16) -> Dataset:
    import cv2

    class_names = sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    )
    if not class_names:
        raise ValueError(f"No class sub-directories found under {root!r}.")

    X, y = [], []
    for label, name in enumerate(class_names):
        cdir = os.path.join(root, name)
        files = [f for f in os.listdir(cdir) if f.lower().endswith(IMG_EXT)]
        if not files:
            continue
        for fn in files:
            img = cv2.imread(os.path.join(cdir, fn), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (image_size, image_size),
                             interpolation=cv2.INTER_AREA)
            X.append(img.astype(np.float32).reshape(-1) / 255.0)
            y.append(label)
    if not X:
        raise ValueError(f"No readable images found under {root!r}.")
    return Dataset(
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(y, dtype=np.int64),
        class_names=class_names,
        image_size=image_size,
        source=f"image-folder: {root} ({len(X)} images, {len(class_names)} classes)",
    )


def load_self_test() -> Dataset:
    """Real handwritten-digit images shipped with scikit-learn."""
    from sklearn.datasets import load_digits

    digits = load_digits()
    X = digits.images.reshape(len(digits.images), -1).astype(np.float32) / 16.0
    y = digits.target.astype(np.int64)
    return Dataset(
        X=X,
        y=y,
        class_names=[str(i) for i in range(10)],
        image_size=8,
        source=f"sklearn load_digits (self-test) — {len(X)} real 8x8 images, 10 classes",
    )


def load_dataset(data_dir: str | None, image_size: int = 16) -> Dataset:
    """Load an image-folder dataset if given and valid, else the self-test set."""
    if data_dir and os.path.isdir(data_dir):
        return load_image_folder(data_dir, image_size=image_size)
    return load_self_test()
