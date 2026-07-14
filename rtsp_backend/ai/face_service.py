"""
Face recognition service.

Ties an embedder backend to the persistent embedding store in SQLite:

* enrolment: detect a face in an image, embed it, store the vector against an
  employee (multiple images/vectors per employee raise accuracy).
* recognition: for each face in a frame, embed it and match by cosine
  similarity against all stored vectors; above the threshold -> that employee,
  otherwise "Unknown Person".

The embedding cache is rebuilt automatically whenever employees/images change,
so a newly enrolled employee is recognised immediately on the next frame.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np

from .base import BBox, Detection, FaceEmbedder


def _blur_score(gray: np.ndarray) -> float:
    """Variance of the Laplacian — higher means sharper. Used to reject blur."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class FaceRecognitionService:
    def __init__(self, db, embedder: FaceEmbedder, threshold: float = 0.5,
                 min_blur: float = 40.0, min_face_size: int = 24,
                 min_recog_blur: float = 12.0, nms_iou: float = 0.4,
                 topk_vote: int = 3) -> None:
        self.db = db
        self.embedder = embedder
        self.threshold = threshold
        # Minimum Laplacian-variance sharpness for a face crop to be enrollable.
        self.min_blur = min_blur
        # Reject faces smaller than this (px, min side) — ignores tiny faces.
        self.min_face_size = min_face_size
        # Blur floor for *recognition* (looser than enrolment); blurrier faces
        # are still reported but flagged low-quality and never matched.
        self.min_recog_blur = min_recog_blur
        # IoU for de-duplicating overlapping face boxes from the detector.
        self.nms_iou = nms_iou
        # Number of nearest stored vectors that vote on identity (stability).
        self.topk_vote = max(1, int(topk_vote))
        self._lock = threading.RLock()
        self._matrix: Optional[np.ndarray] = None      # [N, dim] unit vectors
        self._meta: list[tuple[int, str]] = []         # (employee_id, name) per row
        self._loaded_for = ""                          # embedder id the cache was built for
        self.reload_cache()

    # -- cache -------------------------------------------------------------

    def reload_cache(self) -> None:
        """Rebuild the in-memory embedding matrix from the DB."""
        with self._lock:
            rows = self.db.query(
                "SELECT fe.employee_id, fe.vector, fe.dim, e.full_name "
                "FROM face_embeddings fe JOIN employees e ON e.id = fe.employee_id "
                "WHERE fe.embedder = ?",
                (self.embedder.backend_id,),
            )
            vecs, meta = [], []
            for r in rows:
                v = np.frombuffer(r["vector"], dtype=np.float32)
                if v.shape[0] != r["dim"]:
                    continue
                n = np.linalg.norm(v)
                if n == 0:
                    continue
                vecs.append(v / n)
                meta.append((r["employee_id"], r["full_name"]))
            self._matrix = np.vstack(vecs) if vecs else None
            self._meta = meta
            self._loaded_for = self.embedder.backend_id

    @property
    def enrolled_vectors(self) -> int:
        with self._lock:
            return 0 if self._matrix is None else int(self._matrix.shape[0])

    # -- validation --------------------------------------------------------

    def validate_frame(self, image_bgr: np.ndarray) -> dict:
        """
        Inspect an image for enrolment suitability without storing anything.

        Returns face count, the primary face box, its sharpness, and a clear
        ``ok`` / ``reason`` verdict so the UI can accept, warn, or reject before
        the user commits to saving.
        """
        if not self.embedder.ready:
            self.embedder.load()
        boxes = self.embedder.detect_faces(image_bgr)
        result = {
            "faces": len(boxes),
            "ok": False,
            "reason": None,
            "blur_score": None,
            "min_blur": self.min_blur,
            "bbox": None,
            "multiple_faces": len(boxes) > 1,
        }
        if not boxes:
            result["reason"] = "no_face_detected"
            return result
        box = max(boxes, key=lambda b: (b.x2 - b.x1) * (b.y2 - b.y1))
        result["bbox"] = [round(v, 1) for v in box.as_list()]
        x1, y1, x2, y2 = (max(0, int(box.x1)), max(0, int(box.y1)),
                          int(box.x2), int(box.y2))
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            result["reason"] = "face_crop_empty"
            return result
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = _blur_score(gray)
        result["blur_score"] = round(blur, 1)
        if blur < self.min_blur:
            result["reason"] = "blurry"
            return result
        result["ok"] = True
        if result["multiple_faces"]:
            result["reason"] = "multiple_faces_warning"
        return result

    # -- enrolment ---------------------------------------------------------

    def enroll_image(
        self, employee_id: int, image_bgr: np.ndarray, image_id: Optional[int] = None
    ) -> dict:
        """Validate, then detect + embed the primary face and store the vector."""
        verdict = self.validate_frame(image_bgr)
        if not verdict["ok"]:
            # multiple faces is only a warning, not a hard rejection
            if verdict["reason"] != "multiple_faces_warning":
                return {"ok": False, "reason": verdict["reason"],
                        "faces": verdict["faces"], "blur_score": verdict["blur_score"]}
        if not self.embedder.ready:
            self.embedder.load()
        boxes = self.embedder.detect_faces(image_bgr)
        if not boxes:
            return {"ok": False, "reason": "no_face_detected"}
        box = max(boxes, key=lambda b: (b.x2 - b.x1) * (b.y2 - b.y1))
        vec = self.embedder.embed(image_bgr, box)
        if vec is None:
            return {"ok": False, "reason": "embedding_failed"}
        vec = vec.astype(np.float32)
        self.db.insert(
            "INSERT INTO face_embeddings(employee_id, image_id, embedder, dim, vector, created_at) "
            "VALUES(?,?,?,?,?,?)",
            (employee_id, image_id, self.embedder.backend_id, int(vec.shape[0]),
             vec.tobytes(), time.time()),
        )
        self.reload_cache()
        return {"ok": True, "dim": int(vec.shape[0]),
                "bbox": [round(v, 1) for v in box.as_list()],
                "faces": verdict["faces"], "blur_score": verdict["blur_score"],
                "multiple_faces": verdict["multiple_faces"]}

    # -- quality helpers ---------------------------------------------------

    @staticmethod
    def _nms_boxes(boxes: list[BBox], iou_thr: float) -> list[BBox]:
        """Drop duplicate/overlapping face boxes, keeping the largest."""
        if len(boxes) <= 1:
            return boxes
        areas = [(b, (b.x2 - b.x1) * (b.y2 - b.y1)) for b in boxes]
        areas.sort(key=lambda ba: ba[1], reverse=True)
        kept: list[BBox] = []
        for b, _ in areas:
            drop = False
            for k in kept:
                ix1, iy1 = max(b.x1, k.x1), max(b.y1, k.y1)
                ix2, iy2 = min(b.x2, k.x2), min(b.y2, k.y2)
                iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
                inter = iw * ih
                ua = (b.x2 - b.x1) * (b.y2 - b.y1) + (k.x2 - k.x1) * (k.y2 - k.y1) - inter
                if ua > 0 and inter / ua >= iou_thr:
                    drop = True
                    break
            if not drop:
                kept.append(b)
        return kept

    def _face_quality(self, frame_bgr: np.ndarray, box: BBox) -> tuple:
        """Return (quality 0..1, blur_score, min_side_px) for a face crop."""
        x1, y1 = max(0, int(box.x1)), max(0, int(box.y1))
        x2, y2 = int(box.x2), int(box.y2)
        crop = frame_bgr[y1:y2, x1:x2]
        min_side = min(x2 - x1, y2 - y1)
        if crop.size == 0 or min_side <= 0:
            return 0.0, 0.0, max(0, min_side)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = _blur_score(gray)
        # size score saturates at ~3x the min size; blur score saturates at 200
        size_q = min(1.0, min_side / (self.min_face_size * 3.0))
        blur_q = min(1.0, blur / 200.0)
        quality = round(0.5 * size_q + 0.5 * blur_q, 3)
        return quality, round(blur, 1), int(min_side)

    # -- recognition -------------------------------------------------------

    def recognize_frame(self, frame_bgr: np.ndarray) -> list[Detection]:
        """
        Return a Detection per face with identity, confidence and quality.

        Improvements over a naive matcher:
        * duplicate/overlapping detector boxes are merged (NMS);
        * tiny faces (below ``min_face_size``) are ignored entirely;
        * each face gets a 0..1 quality score (size + sharpness); faces below
          the recognition blur floor are reported but never matched (avoids
          identifying a smudge as an employee);
        * identity uses top-k nearest-vector voting across an employee's
          multiple stored embeddings, which is far more stable than a single
          argmax when people move / turn.
        """
        if not self.embedder.ready:
            self.embedder.load()
        if self._loaded_for != self.embedder.backend_id:
            self.reload_cache()

        boxes = self.embedder.detect_faces(frame_bgr)
        boxes = self._nms_boxes(boxes, self.nms_iou)

        results: list[Detection] = []
        for box in boxes:
            quality, blur, min_side = self._face_quality(frame_bgr, box)
            # ignore tiny faces outright
            if min_side < self.min_face_size:
                continue

            extra = {"quality": quality, "blur": blur, "min_side": min_side}
            identity, emp_id, score = "Unknown Person", None, 0.0

            too_blurry = blur < self.min_recog_blur
            if not too_blurry:
                vec = self.embedder.embed(frame_bgr, box)
                if vec is not None:
                    with self._lock:
                        matrix, meta = self._matrix, self._meta
                    if matrix is not None:
                        n = np.linalg.norm(vec)
                        if n:
                            sims = matrix @ (vec / n)
                            emp_id, identity, score = self._vote(sims, meta)
            else:
                extra["skipped"] = "blurry"

            results.append(
                Detection(
                    label=identity,
                    confidence=round(float(score), 4),
                    bbox=box,
                    kind="face",
                    identity=None if emp_id is None else identity,
                    employee_id=emp_id,
                    extra=extra,
                )
            )
        return results

    def _vote(self, sims: np.ndarray, meta: list) -> tuple:
        """
        Top-k nearest-vector voting. Consider the k most similar stored vectors;
        the employee with the highest summed similarity (only counting vectors
        above threshold) wins. Falls back to plain best-match if nothing clears
        the threshold. Returns (employee_id|None, name, score).
        """
        k = min(self.topk_vote, sims.shape[0])
        top_idx = np.argsort(sims)[::-1][:k]
        best_j = int(top_idx[0])
        best_sim = float(sims[best_j])

        tally: dict = {}
        for j in top_idx:
            s = float(sims[j])
            if s < self.threshold:
                continue
            emp_id, name = meta[int(j)]
            cur = tally.get(emp_id, (0.0, name, 0.0))
            tally[emp_id] = (cur[0] + s, name, max(cur[2], s))

        if tally:
            emp_id = max(tally, key=lambda e: tally[e][0])
            _, name, top_score = tally[emp_id]
            return emp_id, name, top_score
        # nothing cleared the threshold: unknown, but report the best similarity
        return None, "Unknown Person", best_sim
