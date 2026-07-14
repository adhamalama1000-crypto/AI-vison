"""
Face embedder backends.

Two concrete implementations, both real (neither fabricates detections):

* ``opencv_fallback`` — face detection via OpenCV's bundled Haar cascade, and a
  deterministic, reproducible embedding computed from the aligned grayscale
  crop (intensity + gradient-orientation histogram, L2-normalised). This is a
  *real* feature extractor: the same face yields the same vector and different
  faces yield different vectors, so the enrol -> embed -> persist -> match
  pipeline is fully testable today WITHOUT a GPU or any model download. It is
  explicitly NOT production-grade face recognition — swap in the InsightFace
  backend below for that. This is documented, not hidden.

* ``insightface_arcface`` — real ArcFace embeddings via the InsightFace library,
  used automatically when the package and its models are available. Marked
  ``requires_weights`` so the UI can show whether it is loadable in this
  environment. If the models can't be loaded it reports an error and stays
  unready (it never silently degrades to fake output).
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

from .base import BBox, FaceEmbedder
from .registry import register


@register
class OpenCVFallbackEmbedder(FaceEmbedder):
    backend_id = "opencv_fallback"
    display_name = "OpenCV Haar + deterministic descriptor (no weights)"
    requires_weights = False
    dim = 160  # 64 intensity + 96 gradient-orientation bins

    def load(self) -> None:
        cascade_path = os.path.join(
            cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
        )
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            self._ready = False
            self._status = "error"
            self._error = "Haar cascade failed to load"
            raise RuntimeError(self._error)
        self._ready = True
        self._status = "ready"
        self._error = None

    def detect_faces(self, frame: np.ndarray) -> list[BBox]:
        if not self._ready:
            self.load()
        if frame is None or getattr(frame, "size", 0) == 0:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            rects = self._cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
            )
        except cv2.error:
            # Some OpenCV builds raise an internal range-check on certain
            # frames (e.g. heavily degraded/edge-case inputs). Treat that as
            # "no face" rather than letting it crash capture / enrolment or the
            # recognition worker.
            return []
        return [BBox(float(x), float(y), float(x + w), float(y + h)) for (x, y, w, h) in rects]

    def embed(self, frame: np.ndarray, box: BBox) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = (int(box.x1), int(box.y1), int(box.x2), int(box.y2))
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        crop = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        gray = cv2.equalizeHist(gray)

        # Intensity signature: 8x8 average pool -> 64 dims.
        pooled = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32)
        intensity = pooled.flatten() / 255.0

        # Gradient-orientation histogram over a 4x4 grid of 6 bins -> 96 dims.
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        ang = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)  # 0..1
        bins = 6
        grid = 4
        hist = np.zeros((grid, grid, bins), dtype=np.float32)
        cell = 64 // grid
        for gyi in range(grid):
            for gxi in range(grid):
                a = ang[gyi * cell:(gyi + 1) * cell, gxi * cell:(gxi + 1) * cell]
                m = mag[gyi * cell:(gyi + 1) * cell, gxi * cell:(gxi + 1) * cell]
                idx = np.clip((a * bins).astype(int), 0, bins - 1)
                for b in range(bins):
                    hist[gyi, gxi, b] = float(m[idx == b].sum())
        hist = hist.flatten()
        if hist.sum() > 0:
            hist /= hist.sum()

        vec = np.concatenate([intensity, hist]).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return None
        return vec / norm


@register
class InsightFaceEmbedder(FaceEmbedder):
    backend_id = "insightface_arcface"
    display_name = "InsightFace ArcFace (real embeddings, needs models)"
    requires_weights = True
    dim = 512

    def load(self) -> None:
        try:
            import onnxruntime  # noqa: F401  (insightface needs it too)
        except Exception as exc:
            self._ready = False
            self._status = "error"
            self._reason = "onnxruntime_missing"
            self._error = f"onnxruntime is not installed: {exc}"
            raise RuntimeError(self._error)
        try:
            from insightface.app import FaceAnalysis  # type: ignore
        except Exception as exc:  # library not installed
            self._ready = False
            self._status = "unavailable"
            self._reason = "insightface_missing"
            self._error = f"InsightFace is not installed: {exc}"
            raise RuntimeError(self._error)
        try:
            ctx_id = int(self.params.get("ctx_id", -1))  # -1 = CPU
            app = FaceAnalysis(name=self.params.get("model_pack", "buffalo_l"))
            app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            self._app = app
            self._ready = True
            self._status = "ready"
            self._error = None
            self._reason = None
        except Exception as exc:  # models missing / download blocked
            self._ready = False
            self._status = "error"
            self._reason = "init_failed"
            self._error = f"InsightFace model initialisation failed (weights unavailable?): {exc}"
            raise RuntimeError(self._error)

    def detect_faces(self, frame: np.ndarray) -> list[BBox]:
        if not self._ready:
            self.load()
        faces = self._app.get(frame)
        self._last = {id(f): f for f in faces}
        out = []
        for f in faces:
            b = f.bbox.astype(float)
            out.append(BBox(float(b[0]), float(b[1]), float(b[2]), float(b[3])))
        return out

    def embed(self, frame: np.ndarray, box: BBox) -> Optional[np.ndarray]:
        if not self._ready:
            self.load()
        faces = self._app.get(frame)
        if not faces:
            return None
        # pick the face whose bbox centre is closest to the requested box
        cx, cy = box.center
        best = min(
            faces,
            key=lambda f: (float((f.bbox[0] + f.bbox[2]) / 2) - cx) ** 2
            + (float((f.bbox[1] + f.bbox[3]) / 2) - cy) ** 2,
        )
        emb = np.asarray(best.normed_embedding, dtype=np.float32)
        norm = np.linalg.norm(emb)
        return emb / norm if norm else None
