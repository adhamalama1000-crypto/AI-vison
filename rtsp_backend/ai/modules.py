"""
Additional AI detection modules.

Every backend here reuses the *real* shared ONNX inference engine
(:class:`rtsp_backend.ai.detectors._OnnxDetectorBase`) — the same
letterbox → forward-pass → decode → NMS pipeline that already powers object and
component detection. Consistent with the rest of the platform, none of these
fabricate detections:

* When a trained ``.onnx`` model is dropped into the module's ``models/<task>/``
  directory the backend runs genuine inference against it and produces real
  boxes/labels.
* Until then it reports ``weights_missing`` and returns nothing — the module
  shows as "model unavailable" in the UI rather than inventing results.

To activate a module: train / obtain a detector for its classes, export to ONNX,
drop it in ``models/<task>/`` (optionally with a ``labels.txt``), and enable the
task. No other code changes are required.

Fall detection additionally ships a *classical, clearly-labelled heuristic*
backend (person-box aspect ratio) as an opt-in baseline — the same philosophy as
the OpenCV face fallback: a real, transparent method that is explicitly not a
trained model, offered alongside the ONNX backend.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .base import BBox, Detection, Detector
from .detectors import _OnnxDetectorBase
from .registry import register


def _labels_from(models_dir: str, subdir: str, default: list[str]) -> list[str]:
    path = os.path.join(models_dir, subdir, "labels.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            names = [ln.strip() for ln in fh if ln.strip()]
        if names:
            return names
    return default


class _EventOnnxDetector(_OnnxDetectorBase):
    """Shared base for the new single-model ONNX detection modules."""

    requires_weights = True
    default_labels: list[str] = []

    def load(self) -> None:
        models_dir = self.params.get("models_dir", "models")
        self.class_names = (
            self.params.get("labels")
            or _labels_from(models_dir, self.default_subdir, self.default_labels)
        )
        super().load()


# --------------------------------------------------------------------------- #
# Fire / smoke / explosion
# --------------------------------------------------------------------------- #

@register
class FireDetector(_EventOnnxDetector):
    backend_id = "onnx_fire"
    task = "fire"
    display_name = "ONNX fire/smoke detector (needs trained weights)"
    default_subdir = "fire"
    default_labels = ["fire", "smoke", "explosion"]


@register
class NullFire(Detector):
    backend_id = "null"
    task = "fire"
    display_name = "Disabled (no fire detection)"
    requires_weights = False

    def load(self) -> None:
        self._ready, self._status = True, "ready"

    def infer(self, frame):
        return []


# --------------------------------------------------------------------------- #
# Weapon
# --------------------------------------------------------------------------- #

@register
class WeaponDetector(_EventOnnxDetector):
    backend_id = "onnx_weapon"
    task = "weapon"
    display_name = "ONNX weapon detector (gun/rifle/knife, needs weights)"
    default_subdir = "weapon"
    default_labels = ["gun", "pistol", "rifle", "knife"]


@register
class NullWeapon(Detector):
    backend_id = "null"
    task = "weapon"
    display_name = "Disabled (no weapon detection)"
    requires_weights = False

    def load(self) -> None:
        self._ready, self._status = True, "ready"

    def infer(self, frame):
        return []


# --------------------------------------------------------------------------- #
# PPE  (helmet / vest / gloves / goggles, and their "no_*" violation classes)
# --------------------------------------------------------------------------- #

@register
class PPEDetector(_EventOnnxDetector):
    backend_id = "onnx_ppe"
    task = "ppe"
    display_name = "ONNX PPE detector (helmet/vest/gloves/goggles, needs weights)"
    default_subdir = "ppe"
    default_labels = [
        "helmet", "no_helmet", "vest", "no_vest",
        "gloves", "no_gloves", "goggles", "no_goggles", "person",
    ]


@register
class NullPPE(Detector):
    backend_id = "null"
    task = "ppe"
    display_name = "Disabled (no PPE detection)"
    requires_weights = False

    def load(self) -> None:
        self._ready, self._status = True, "ready"

    def infer(self, frame):
        return []


# --------------------------------------------------------------------------- #
# Human detection (person / entry / exit / crowd counting)
# --------------------------------------------------------------------------- #

@register
class HumanDetector(_EventOnnxDetector):
    backend_id = "onnx_human"
    task = "human"
    display_name = "ONNX person detector (COCO or custom, needs weights)"
    default_subdir = "human"
    # A standard COCO YOLO export works out of the box for people; we filter to
    # the person class at inference time.
    default_labels = _OnnxDetectorBase.class_names

    def infer(self, frame: np.ndarray) -> list[Detection]:
        dets = super().infer(frame)
        # keep only people; other COCO classes belong to the object/vehicle tasks
        return [d for d in dets if d.label == "person"]


@register
class NullHuman(Detector):
    backend_id = "null"
    task = "human"
    display_name = "Disabled (no human detection)"
    requires_weights = False

    def load(self) -> None:
        self._ready, self._status = True, "ready"

    def infer(self, frame):
        return []


# --------------------------------------------------------------------------- #
# Vehicle detection (car / truck / bus / motorcycle)
# --------------------------------------------------------------------------- #

_VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}


@register
class VehicleDetector(_EventOnnxDetector):
    backend_id = "onnx_vehicle"
    task = "vehicle"
    display_name = "ONNX vehicle detector (COCO or custom, needs weights)"
    default_subdir = "vehicle"
    default_labels = _OnnxDetectorBase.class_names

    def infer(self, frame: np.ndarray) -> list[Detection]:
        dets = super().infer(frame)
        return [d for d in dets if d.label in _VEHICLE_CLASSES]


@register
class NullVehicle(Detector):
    backend_id = "null"
    task = "vehicle"
    display_name = "Disabled (no vehicle detection)"
    requires_weights = False

    def load(self) -> None:
        self._ready, self._status = True, "ready"

    def infer(self, frame):
        return []


# --------------------------------------------------------------------------- #
# Violence detection  (fighting / assault / aggression)
# --------------------------------------------------------------------------- #

@register
class ViolenceDetector(_EventOnnxDetector):
    backend_id = "onnx_violence"
    task = "violence"
    display_name = "ONNX violence detector (needs trained weights)"
    default_subdir = "violence"
    default_labels = ["violence", "fighting", "assault"]


@register
class NullViolence(Detector):
    backend_id = "null"
    task = "violence"
    display_name = "Disabled (no violence detection)"
    requires_weights = False

    def load(self) -> None:
        self._ready, self._status = True, "ready"

    def infer(self, frame):
        return []


# --------------------------------------------------------------------------- #
# Fall detection
# --------------------------------------------------------------------------- #

@register
class FallDetector(_EventOnnxDetector):
    backend_id = "onnx_fall"
    task = "fall"
    display_name = "ONNX fall detector (needs trained weights)"
    default_subdir = "fall"
    default_labels = ["standing", "sitting", "falling", "lying"]


@register
class HeuristicFallDetector(Detector):
    """
    Classical fall heuristic (opt-in, explicitly NOT a trained model).

    Given person boxes from an upstream person detector, flags a "fall" when a
    person's bounding box becomes markedly wider than tall (aspect ratio) for
    the current frame — a well-known lightweight baseline. It is transparent and
    tunable, and is offered the same way the OpenCV face fallback is: a real,
    documented method to use when no trained model is available. It does not
    fabricate detections — it only classifies boxes actually produced by a
    detector, and returns nothing when given none.
    """

    backend_id = "heuristic_fall"
    task = "fall"
    display_name = "Aspect-ratio fall heuristic (no weights, baseline only)"
    requires_weights = False

    def load(self) -> None:
        self._ready, self._status = True, "ready"

    def analyze_people(self, people: list[Detection]) -> list[Detection]:
        ratio = float(self.params.get("fall_aspect_ratio", 1.3))
        out: list[Detection] = []
        for p in people:
            w = p.bbox.x2 - p.bbox.x1
            h = p.bbox.y2 - p.bbox.y1
            if h <= 0:
                continue
            if (w / h) >= ratio:
                out.append(Detection(
                    label="falling", confidence=p.confidence, bbox=p.bbox,
                    kind="object", extra={"method": "aspect_ratio", "w_h": round(w / h, 2)},
                ))
        return out

    def infer(self, frame: np.ndarray) -> list[Detection]:
        # No standalone image inference; this backend needs person boxes.
        return []


@register
class NullFall(Detector):
    backend_id = "null"
    task = "fall"
    display_name = "Disabled (no fall detection)"
    requires_weights = False

    def load(self) -> None:
        self._ready, self._status = True, "ready"

    def infer(self, frame):
        return []
