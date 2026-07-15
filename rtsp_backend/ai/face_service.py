"""
Face recognition service — enrolment, quality gating and identity matching.

Pipeline (production): SCRFD detection -> ArcFace 512-d embedding -> cosine
similarity against every enrolled employee embedding -> threshold + margin
gate. Matching is served by :class:`~rtsp_backend.ai.face_index.FaceIndex`
(FAISS when available, otherwise a single vectorised BLAS matmul), so a frame is
never scored with a Python loop over the database.

Design priorities (in order):

1. **Never guess.** A face whose best similarity is below the threshold, or that
   does not clear the identity margin over the runner-up employee, is reported
   as ``Unknown Employee`` — never the closest employee. This directly targets a
   low False Acceptance Rate (a stranger must not become an employee).
2. **Quality first.** Blurry, tiny, badly-lit, or (at enrolment) multi-face
   captures are rejected with a clear, machine-readable reason.
3. **Multiple samples per employee.** Each employee stores many embeddings; a
   query is matched either against per-employee *centroids* (``average`` policy)
   or the single *nearest* stored sample (``nearest`` policy) — configurable,
   both anti-FAR-gated by the same threshold + margin.

The in-memory index is rebuilt automatically whenever employees/images change,
so a newly enrolled employee is recognised on the very next frame.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Optional

import cv2
import numpy as np

from .base import BBox, Detection, FaceEmbedder
from .face_index import FaceIndex

# Machine reason code -> human message (the UI shows these verbatim).
QUALITY_MESSAGES: dict[str, str] = {
    "ok": "Face OK",
    "no_face_detected": "No face detected",
    "face_crop_empty": "No face detected",
    "blurry": "Face too blurry",
    "motion_blur": "Motion blur — hold still",
    "face_too_small": "Face too small",
    "multiple_faces": "Multiple faces",
    "overexposed": "Bad lighting — too bright",
    "underexposed": "Bad lighting — too dark",
    "bad_lighting": "Bad lighting",
    "low_detection_confidence": "Face not clear enough",
    "embedding_failed": "Could not read face features",
}


def _blur_score(gray: np.ndarray) -> float:
    """Variance of the Laplacian — higher means sharper. Used to reject blur."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class FaceRecognitionService:
    def __init__(self, db, embedder: FaceEmbedder, threshold: float = 0.65,
                 min_blur: float = 45.0, min_face_size: int = 50,
                 min_recog_blur: float = 25.0, nms_iou: float = 0.4,
                 topk_vote: int = 5, match_policy: str = "average",
                 margin: float = 0.05, min_det_score: float = 0.5,
                 enroll_min_face_size: int = 80,
                 dark_mean: float = 40.0, bright_mean: float = 215.0,
                 use_faiss: bool = True) -> None:
        self.db = db
        self.embedder = embedder
        # Cosine-similarity acceptance threshold (anti-FAR). Default 0.65 gives a
        # very low false-acceptance rate for ArcFace embeddings.
        self.threshold = float(threshold)
        # Best employee must beat the runner-up employee by this margin to be
        # accepted — prevents confusing two similar employees.
        self.margin = float(margin)
        # "average" -> match per-employee centroid; "nearest" -> best single sample.
        self.match_policy = match_policy if match_policy in ("average", "nearest") else "average"
        # Minimum detector confidence (SCRFD) for a face to be matched at all.
        self.min_det_score = float(min_det_score)
        # Quality floors.
        self.min_blur = float(min_blur)                     # enrolment sharpness
        self.min_face_size = int(min_face_size)             # recognition min side px
        self.enroll_min_face_size = int(enroll_min_face_size)
        self.min_recog_blur = float(min_recog_blur)         # recognition sharpness
        self.dark_mean = float(dark_mean)                   # underexposure bound
        self.bright_mean = float(bright_mean)               # overexposure bound
        self.nms_iou = float(nms_iou)
        self.topk_vote = max(1, int(topk_vote))
        self.use_faiss = bool(use_faiss)

        self._lock = threading.RLock()
        self._index = FaceIndex(use_faiss=self.use_faiss)   # over ALL sample vectors
        self._row_emp: np.ndarray = np.empty(0, dtype=np.int64)     # employee_id per row
        self._row_empidx: np.ndarray = np.empty(0, dtype=np.int64)  # 0..E-1 per row
        self._emp_ids: list[int] = []                       # ordered unique employee ids
        self._emp_name: dict[int, str] = {}
        self._centroids: Optional[np.ndarray] = None        # [E, dim] unit centroids
        self._loaded_for = ""
        self.reload_cache()

    # -- cache -------------------------------------------------------------

    def reload_cache(self) -> None:
        """Rebuild the in-memory index + centroids from the DB."""
        with self._lock:
            rows = self.db.query(
                "SELECT fe.employee_id, fe.vector, fe.dim, e.full_name "
                "FROM face_embeddings fe JOIN employees e ON e.id = fe.employee_id "
                "WHERE fe.embedder = ?",
                (self.embedder.backend_id,),
            )
            vecs: list[np.ndarray] = []
            row_emp: list[int] = []
            name_map: dict[int, str] = {}
            for r in rows:
                v = np.frombuffer(r["vector"], dtype=np.float32)
                if v.shape[0] != r["dim"]:
                    continue
                n = np.linalg.norm(v)
                if n == 0:
                    continue
                vecs.append(v / n)
                row_emp.append(int(r["employee_id"]))
                name_map[int(r["employee_id"])] = r["full_name"]

            if vecs:
                matrix = np.vstack(vecs).astype(np.float32)
                emp_arr = np.asarray(row_emp, dtype=np.int64)
                emp_ids = sorted(set(row_emp))
                id_to_idx = {e: i for i, e in enumerate(emp_ids)}
                row_empidx = np.asarray([id_to_idx[e] for e in row_emp], dtype=np.int64)
                # per-employee unit centroids
                centroids = np.zeros((len(emp_ids), matrix.shape[1]), dtype=np.float32)
                for e, i in id_to_idx.items():
                    mean = matrix[emp_arr == e].mean(axis=0)
                    nn = np.linalg.norm(mean)
                    centroids[i] = mean / nn if nn else mean
            else:
                matrix = None
                emp_arr = np.empty(0, dtype=np.int64)
                emp_ids = []
                row_empidx = np.empty(0, dtype=np.int64)
                centroids = None

            self._index.build(matrix)
            self._row_emp = emp_arr
            self._row_empidx = row_empidx
            self._emp_ids = emp_ids
            self._emp_name = name_map
            self._centroids = centroids
            self._loaded_for = self.embedder.backend_id

    @property
    def enrolled_vectors(self) -> int:
        with self._lock:
            return self._index.size

    @property
    def enrolled_employees(self) -> int:
        with self._lock:
            return len(self._emp_ids)

    @property
    def index_backend(self) -> str:
        return self._index.backend

    def config(self) -> dict:
        """Current tunables (surfaced to the API / frontend)."""
        return {
            "threshold": self.threshold,
            "margin": self.margin,
            "match_policy": self.match_policy,
            "min_det_score": self.min_det_score,
            "min_blur": self.min_blur,
            "min_recog_blur": self.min_recog_blur,
            "min_face_size": self.min_face_size,
            "enroll_min_face_size": self.enroll_min_face_size,
            "dark_mean": self.dark_mean,
            "bright_mean": self.bright_mean,
            "topk_vote": self.topk_vote,
            "index_backend": self.index_backend,
            "faiss_available": self._index.faiss_available,
            "embedder": self.embedder.backend_id,
            "embedding_dim": getattr(self.embedder, "dim", None),
            "enrolled_vectors": self.enrolled_vectors,
            "enrolled_employees": self.enrolled_employees,
        }

    # -- quality -----------------------------------------------------------

    def _crop(self, frame: np.ndarray, box: BBox):
        x1, y1 = max(0, int(box.x1)), max(0, int(box.y1))
        x2, y2 = int(box.x2), int(box.y2)
        return frame[y1:y2, x1:x2], (x2 - x1), (y2 - y1)

    def assess_quality(self, frame: np.ndarray, box: BBox,
                       det_score: Optional[float] = None) -> dict:
        """Return a rich quality report for a face crop.

        Keys: quality (0..1), blur, min_side, brightness, over_frac, under_frac,
        det_score, reason (None if acceptable for recognition).
        """
        crop, w, h = self._crop(frame, box)
        min_side = int(min(w, h))
        rep = {"quality": 0.0, "blur": 0.0, "min_side": max(0, min_side),
               "brightness": None, "over_frac": None, "under_frac": None,
               "det_score": None if det_score is None else round(float(det_score), 4),
               "reason": None}
        if crop.size == 0 or min_side <= 0:
            rep["reason"] = "face_crop_empty"
            return rep
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = _blur_score(gray)
        brightness = float(gray.mean())
        over_frac = float((gray > 245).mean())
        under_frac = float((gray < 12).mean())
        rep.update(blur=round(blur, 1), brightness=round(brightness, 1),
                   over_frac=round(over_frac, 3), under_frac=round(under_frac, 3))

        if min_side < self.min_face_size:
            rep["reason"] = "face_too_small"
        elif det_score is not None and det_score < self.min_det_score:
            rep["reason"] = "low_detection_confidence"
        elif blur < self.min_recog_blur:
            rep["reason"] = "blurry"
        elif brightness > self.bright_mean or over_frac > 0.35:
            rep["reason"] = "overexposed"
        elif brightness < self.dark_mean or under_frac > 0.5:
            rep["reason"] = "underexposed"

        # 0..1 quality score (size, sharpness, exposure centredness, det score).
        size_q = min(1.0, min_side / (self.min_face_size * 3.0))
        blur_q = min(1.0, blur / 200.0)
        # exposure quality peaks at mid brightness (~128) and falls off to edges
        expo_q = max(0.0, 1.0 - abs(brightness - 128.0) / 128.0)
        det_q = 1.0 if det_score is None else float(min(1.0, max(0.0, det_score)))
        rep["quality"] = round(0.3 * size_q + 0.3 * blur_q + 0.2 * expo_q + 0.2 * det_q, 3)
        return rep

    # -- validation (enrolment suitability) --------------------------------

    def validate_frame(self, image_bgr: np.ndarray) -> dict:
        """Inspect an image for enrolment suitability without storing anything."""
        if not self.embedder.ready:
            self.embedder.load()
        faces = self.embedder.detect_and_embed(image_bgr)
        result = {
            "faces": len(faces), "ok": False, "reason": None,
            "blur_score": None, "quality": None, "brightness": None,
            "det_score": None, "min_blur": self.min_blur, "bbox": None,
            "multiple_faces": len(faces) > 1,
        }
        if not faces:
            result["reason"] = "no_face_detected"
            result["message"] = QUALITY_MESSAGES["no_face_detected"]
            return result
        if len(faces) > 1:
            # Enrolment must capture exactly one person — reject multi-face frames.
            result["reason"] = "multiple_faces"
            result["message"] = QUALITY_MESSAGES["multiple_faces"]
            return result

        box, det_score, _emb = faces[0]
        rep = self.assess_quality(image_bgr, box, det_score)
        result["bbox"] = [round(v, 1) for v in box.as_list()]
        result["blur_score"] = rep["blur"]
        result["quality"] = rep["quality"]
        result["brightness"] = rep["brightness"]
        result["det_score"] = rep["det_score"]
        # stricter face-size floor for enrolment than for live recognition
        reason = rep["reason"]
        if reason is None and rep["min_side"] < self.enroll_min_face_size:
            reason = "face_too_small"
        if reason is None and rep["blur"] < self.min_blur:
            reason = "blurry"
        if reason:
            result["reason"] = reason
            result["message"] = QUALITY_MESSAGES.get(reason, reason)
            return result
        result["ok"] = True
        result["message"] = QUALITY_MESSAGES["ok"]
        return result

    # -- enrolment ---------------------------------------------------------

    def enroll_image(self, employee_id: int, image_bgr: np.ndarray,
                     image_id: Optional[int] = None) -> dict:
        """Validate, then embed the single face and store the vector + metadata."""
        verdict = self.validate_frame(image_bgr)
        if not verdict["ok"]:
            return {"ok": False, "reason": verdict["reason"],
                    "message": verdict.get("message"),
                    "faces": verdict["faces"], "blur_score": verdict["blur_score"]}
        if not self.embedder.ready:
            self.embedder.load()
        faces = self.embedder.detect_and_embed(image_bgr)
        if not faces:
            return {"ok": False, "reason": "no_face_detected",
                    "message": QUALITY_MESSAGES["no_face_detected"]}
        # exactly one face guaranteed by validate_frame; use it
        box, det_score, vec = faces[0]
        if vec is None:
            return {"ok": False, "reason": "embedding_failed",
                    "message": QUALITY_MESSAGES["embedding_failed"]}
        rep = self.assess_quality(image_bgr, box, det_score)
        vec = vec.astype(np.float32)
        meta = {
            "bbox": [round(v, 1) for v in box.as_list()],
            "blur": rep["blur"], "brightness": rep["brightness"],
            "det_score": rep["det_score"], "min_side": rep["min_side"],
            "model_pack": getattr(self.embedder, "params", {}).get("model_pack"),
        }
        self.db.insert(
            "INSERT INTO face_embeddings(employee_id, image_id, embedder, dim, "
            "vector, quality, meta, created_at) VALUES(?,?,?,?,?,?,?,?)",
            (employee_id, image_id, self.embedder.backend_id, int(vec.shape[0]),
             vec.tobytes(), float(rep["quality"]), json.dumps(meta), time.time()),
        )
        self.reload_cache()
        return {"ok": True, "dim": int(vec.shape[0]),
                "bbox": meta["bbox"], "faces": 1,
                "blur_score": rep["blur"], "quality": rep["quality"],
                "det_score": rep["det_score"], "multiple_faces": False}

    def retrain_employee(self, employee_id: int, data_dir: str) -> dict:
        """Recompute this employee's embeddings from their stored images with the
        current embedder. Existing vectors for this embedder are replaced."""
        import os
        imgs = self.db.query(
            "SELECT id, path FROM employee_images WHERE employee_id=?", (employee_id,))
        if not self.embedder.ready:
            self.embedder.load()
        # drop old vectors for this embedder only (keep other embedders' data)
        self.db.execute(
            "DELETE FROM face_embeddings WHERE employee_id=? AND embedder=?",
            (employee_id, self.embedder.backend_id))
        results = []
        enrolled = 0
        for im in imgs:
            path = os.path.join(data_dir, im["path"])
            img = cv2.imread(path)
            if img is None:
                results.append({"image_id": im["id"], "ok": False, "reason": "image_missing"})
                continue
            res = self.enroll_image(employee_id, img, im["id"])
            if res.get("ok"):
                enrolled += 1
            results.append({"image_id": im["id"], **res})
        self.reload_cache()
        return {"employee_id": employee_id, "images": len(imgs),
                "enrolled": enrolled, "results": results,
                "embedder": self.embedder.backend_id}

    # -- quality helpers ---------------------------------------------------

    @staticmethod
    def _nms_boxes(boxes: list[BBox], iou_thr: float) -> list[int]:
        """Return indices of boxes to keep (drop overlaps, keep the largest)."""
        if len(boxes) <= 1:
            return list(range(len(boxes)))
        order = sorted(range(len(boxes)),
                       key=lambda i: (boxes[i].x2 - boxes[i].x1) * (boxes[i].y2 - boxes[i].y1),
                       reverse=True)
        kept: list[int] = []
        for i in order:
            b = boxes[i]
            drop = False
            for j in kept:
                k = boxes[j]
                ix1, iy1 = max(b.x1, k.x1), max(b.y1, k.y1)
                ix2, iy2 = min(b.x2, k.x2), min(b.y2, k.y2)
                iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
                inter = iw * ih
                ua = ((b.x2 - b.x1) * (b.y2 - b.y1)
                      + (k.x2 - k.x1) * (k.y2 - k.y1) - inter)
                if ua > 0 and inter / ua >= iou_thr:
                    drop = True
                    break
            if not drop:
                kept.append(i)
        return kept

    # -- matching ----------------------------------------------------------

    def _employee_scores(self, vec: np.ndarray) -> np.ndarray:
        """Per-employee similarity score aligned to ``self._emp_ids``.

        ``average`` policy -> cosine to each employee's centroid.
        ``nearest`` policy -> max cosine to any of the employee's samples.
        """
        if self.match_policy == "average" and self._centroids is not None:
            return (self._centroids @ vec).astype(np.float32)
        # nearest: reduce all-sample sims to a per-employee max (vectorised)
        sims = self._index.all_sims(vec)
        scores = np.full(len(self._emp_ids), -1.0, dtype=np.float32)
        if sims.size:
            np.maximum.at(scores, self._row_empidx, sims)
        return scores

    def match(self, vec: np.ndarray) -> dict:
        """Resolve identity for a single unit embedding.

        Returns dict with employee_id (None if unknown), name, similarity,
        margin, runner similarity, accepted flag.
        """
        with self._lock:
            if not self._emp_ids:
                return {"employee_id": None, "name": "Unknown Employee",
                        "similarity": 0.0, "margin": 0.0, "runner": 0.0,
                        "accepted": False}
            scores = self._employee_scores(vec)
            order = np.argsort(scores)[::-1]
            best_i = int(order[0])
            best_score = float(scores[best_i])
            runner = float(scores[int(order[1])]) if len(order) > 1 else -1.0
            best_emp = self._emp_ids[best_i]
            name = self._emp_name.get(best_emp, f"#{best_emp}")

        margin = best_score - runner
        accepted = (best_score >= self.threshold) and (margin >= self.margin)
        if accepted:
            return {"employee_id": best_emp, "name": name,
                    "similarity": best_score, "margin": margin,
                    "runner": runner, "accepted": True}
        return {"employee_id": None, "name": "Unknown Employee",
                "similarity": best_score, "margin": margin,
                "runner": runner, "accepted": False,
                "closest_employee_id": best_emp, "closest_name": name}

    # -- recognition -------------------------------------------------------

    def recognize_frame(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Return a Detection per face with identity, similarity and quality.

        A face is only ever labelled with an employee when its embedding clears
        BOTH the similarity threshold and the identity margin; otherwise it is
        reported as ``Unknown Employee`` (never the closest employee).
        """
        if not self.embedder.ready:
            self.embedder.load()
        if self._loaded_for != self.embedder.backend_id:
            self.reload_cache()

        faces = self.embedder.detect_and_embed(frame_bgr)
        boxes = [f[0] for f in faces]
        keep = set(self._nms_boxes(boxes, self.nms_iou))

        results: list[Detection] = []
        for i, (box, det_score, vec) in enumerate(faces):
            if i not in keep:
                continue
            rep = self.assess_quality(frame_bgr, box, det_score)
            # ignore tiny faces outright (not reported)
            if rep["min_side"] < self.min_face_size:
                continue

            extra = {
                "quality": rep["quality"], "blur": rep["blur"],
                "min_side": rep["min_side"], "brightness": rep["brightness"],
                "det_score": rep["det_score"], "similarity": 0.0,
                "similarity_pct": 0.0, "margin": 0.0, "confidence": 0.0,
                "status": "unknown", "reason": None, "message": None,
            }
            identity, emp_id, score = "Unknown Employee", None, 0.0

            gate_reason = rep["reason"]  # blurry / bad lighting / low det conf
            if vec is None:
                gate_reason = gate_reason or "embedding_failed"

            if gate_reason is None:
                m = self.match(vec)
                score = float(m["similarity"])
                extra["margin"] = round(float(m["margin"]), 4)
                extra["runner"] = round(float(m["runner"]), 4)
                if m["accepted"]:
                    emp_id = m["employee_id"]
                    identity = m["name"]
                    extra["status"] = "employee"
                else:
                    extra["closest_name"] = m.get("closest_name")
            else:
                extra["reason"] = gate_reason
                extra["message"] = QUALITY_MESSAGES.get(gate_reason, gate_reason)

            sim_pct = round(max(0.0, score) * 100.0, 1)
            # confidence: how decisively the match clears the threshold (0..1)
            denom = max(1e-6, 1.0 - self.threshold)
            conf = max(0.0, min(1.0, (score - self.threshold) / denom)) if emp_id else 0.0
            extra["similarity"] = round(float(score), 4)
            extra["similarity_pct"] = sim_pct
            extra["confidence"] = round(float(conf), 3)

            results.append(
                Detection(
                    label=identity,
                    confidence=round(max(0.0, float(score)), 4),
                    bbox=box,
                    kind="face",
                    identity=None if emp_id is None else identity,
                    employee_id=emp_id,
                    extra=extra,
                )
            )
        return results
