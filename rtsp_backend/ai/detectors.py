"""
Generic object-detection backends.

* ``onnx_yolo`` — a real YOLOv5/YOLOv8-style ONNX inference pipeline
  (letterbox preprocess, forward pass via onnxruntime, decode + NMS). It
  activates automatically when a ``.onnx`` file is present in the models
  directory (``models/detection/*.onnx`` or a path given in params). If no
  weights are present it reports ``requires_weights`` and stays unready — it
  never invents boxes.

* ``null`` — always returns no detections. Used when detection is enabled for
  wiring/UI purposes but no model is desired. Honest empty output.

Dropping a trained ``.onnx`` (COCO, or a custom electrical model exported to
ONNX) into ``models/detection/`` and selecting this backend is all that is
needed to get real detections — no code changes.
"""

from __future__ import annotations

import glob
import os
from typing import Optional

import numpy as np

from .base import BBox, Detection, Detector
from .registry import register

# 80 COCO class names — the default label set for standard YOLO exports.
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def _letterbox(img: np.ndarray, new_shape: int = 640):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    import cv2

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
    top, left = (new_shape - nh) // 2, (new_shape - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


class _OnnxDetectorBase(Detector):
    """Shared ONNX YOLO inference used by object + component detectors."""

    requires_weights = True
    default_subdir = "detection"
    class_names = COCO_CLASSES

    def _find_weights(self) -> Optional[str]:
        p = self.params.get("weights")
        if p and os.path.exists(p):
            return p
        models_dir = self.params.get("models_dir", "models")
        pattern = os.path.join(models_dir, self.default_subdir, "*.onnx")
        found = sorted(glob.glob(pattern))
        return found[0] if found else None

    def load(self) -> None:
        weights = self._find_weights()
        if not weights:
            self._ready = False
            self._status = "no_weights"
            self._reason = "weights_missing"
            self._error = (
                f"No .onnx weights found. Drop a model into "
                f"models/{self.default_subdir}/ to activate this backend."
            )
            raise RuntimeError(self._error)
        try:
            import onnxruntime as ort
        except Exception as exc:
            self._ready = False
            self._status = "error"
            self._reason = "onnxruntime_missing"
            self._error = f"onnxruntime is not installed: {exc}"
            raise RuntimeError(self._error)
        try:
            available = ort.get_available_providers()
            want_gpu = str(self.params.get("device", "cpu")).lower() in ("gpu", "cuda")
            providers = ["CPUExecutionProvider"]
            self._reason = None
            if want_gpu:
                if "CUDAExecutionProvider" in available:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                else:
                    # honest: GPU asked for but not available; run on CPU and say so
                    self._reason = "cuda_unavailable"
            self._sess = ort.InferenceSession(weights, providers=providers)
            self._active_providers = self._sess.get_providers()
            self._input = self._sess.get_inputs()[0].name
            shape = self._sess.get_inputs()[0].shape
            self._imgsz = int(shape[2]) if isinstance(shape[2], int) else 640
            self._weights_path = weights
            self._ready = True
            self._status = "ready"
            self._error = (
                "GPU requested but CUDA is unavailable in this environment; "
                "running on CPU." if self._reason == "cuda_unavailable" else None
            )
        except Exception as exc:
            self._ready = False
            self._status = "error"
            self._reason = "init_failed"
            self._error = f"failed to load ONNX model: {exc}"
            raise RuntimeError(self._error)

    def infer(self, frame: np.ndarray) -> list[Detection]:
        if not self._ready:
            self.load()
        conf_thr = float(self.params.get("conf", 0.25))
        iou_thr = float(self.params.get("iou", 0.45))
        img, r, dw, dh = _letterbox(frame, self._imgsz)
        blob = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, 0)
        out = self._sess.run(None, {self._input: blob})[0]
        out = np.squeeze(out)
        # Normalise to [N, ...] with rows = detections.
        if out.ndim == 2 and out.shape[0] < out.shape[1]:
            out = out.transpose()  # YOLOv8 [84, 8400] -> [8400, 84]
        boxes, scores, cls_ids = [], [], []
        for row in out:
            if row.shape[0] >= 85:  # YOLOv5: x,y,w,h,obj,80cls
                obj = row[4]
                cls_scores = row[5:]
                cid = int(np.argmax(cls_scores))
                score = float(obj * cls_scores[cid])
            else:                    # YOLOv8: x,y,w,h,80cls
                cls_scores = row[4:]
                cid = int(np.argmax(cls_scores))
                score = float(cls_scores[cid])
            if score < conf_thr:
                continue
            cx, cy, bw, bh = row[0], row[1], row[2], row[3]
            x1 = (cx - bw / 2 - dw) / r
            y1 = (cy - bh / 2 - dh) / r
            x2 = (cx + bw / 2 - dw) / r
            y2 = (cy + bh / 2 - dh) / r
            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            cls_ids.append(cid)
        if not boxes:
            return []
        boxes_np = np.array(boxes, dtype=np.float32)
        scores_np = np.array(scores, dtype=np.float32)
        keep = _nms(boxes_np, scores_np, iou_thr)
        dets = []
        for i in keep:
            cid = cls_ids[i]
            name = self.class_names[cid] if cid < len(self.class_names) else str(cid)
            b = boxes_np[i]
            dets.append(
                Detection(
                    label=name,
                    confidence=float(scores_np[i]),
                    bbox=BBox(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                    kind=self.task if self.task == "components" else "object",
                )
            )
        return dets


@register
class OnnxYoloDetector(_OnnxDetectorBase):
    backend_id = "onnx_yolo"
    task = "detection"
    display_name = "ONNX YOLO object detector (COCO, needs weights)"
    default_subdir = "detection"
    class_names = COCO_CLASSES


@register
class NullDetector(Detector):
    backend_id = "null"
    task = "detection"
    display_name = "Disabled (no detections)"
    requires_weights = False

    def load(self) -> None:
        self._ready = True
        self._status = "ready"

    def infer(self, frame: np.ndarray) -> list[Detection]:
        return []
