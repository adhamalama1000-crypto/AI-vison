"""
Core AI abstractions.

Every model backend implements one of these interfaces and registers itself in
:mod:`rtsp_backend.ai.registry`. The :class:`~rtsp_backend.ai.manager.AIModelManager`
then enables/disables/selects backends at runtime — the UI and the rest of the
system only ever talk to these interfaces, never to a concrete model. This is
what lets a real InsightFace / YOLO / component / wire model be dropped in later
without touching the API or the frontend.

Nothing here fabricates results: a backend with no usable weights returns an
empty list, and the pipeline records "no detections", never invented boxes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def as_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass
class Detection:
    """A single detected object/face/component."""

    label: str
    confidence: float
    bbox: BBox
    kind: str = "object"                 # object|face|component
    identity: Optional[str] = None       # e.g. recognised employee name
    employee_id: Optional[int] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "bbox": [round(v, 1) for v in self.bbox.as_list()],
            "kind": self.kind,
            "identity": self.identity,
            "employee_id": self.employee_id,
            "extra": self.extra,
        }


@dataclass
class Wire:
    wire_uid: str
    start: tuple[float, float]
    end: tuple[float, float]
    color: Optional[str] = None
    status: str = "ok"                   # ok|broken|disconnected|missing|loose|incorrect
    from_component: Optional[int] = None
    to_component: Optional[int] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "wire_uid": self.wire_uid,
            "start": [round(v, 1) for v in self.start],
            "end": [round(v, 1) for v in self.end],
            "color": self.color,
            "status": self.status,
            "from_component": self.from_component,
            "to_component": self.to_component,
            "extra": self.extra,
        }


class ModelBackend:
    """Base class for every pluggable model backend."""

    #: unique id used by the registry and persisted in model_config.backend
    backend_id: str = "base"
    #: which task this backend serves: detection|face|components|wires
    task: str = "detection"
    #: human-readable name for the UI
    display_name: str = "Base backend"

    def __init__(self, **params: Any) -> None:
        self.params = params
        self._ready = False
        self._status = "unloaded"
        self._error: Optional[str] = None
        self._reason: Optional[str] = None   # machine code, e.g. weights_missing
        self._loading = False

    def load(self) -> None:
        """Load weights / initialise. Must set _ready + _status. Idempotent."""
        self._ready = True
        self._status = "ready"

    def unload(self) -> None:
        self._ready = False
        self._status = "unloaded"

    @property
    def ready(self) -> bool:
        return self._ready

    def status(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "task": self.task,
            "display_name": self.display_name,
            "ready": self._ready,
            "status": self._status,
            "error": self._error,
            "reason": self._reason,
            "loading": self._loading,
            "params": self.params,
            "requires_weights": getattr(self, "requires_weights", False),
        }


class Detector(ModelBackend):
    task = "detection"

    def infer(self, frame: np.ndarray) -> list[Detection]:  # pragma: no cover - interface
        raise NotImplementedError


class FaceEmbedder(ModelBackend):
    task = "face"
    dim: int = 128

    def detect_faces(self, frame: np.ndarray) -> list[BBox]:
        """Return face bounding boxes in the frame."""
        raise NotImplementedError

    def embed(self, frame: np.ndarray, box: BBox) -> Optional[np.ndarray]:
        """Return a unit-norm float32 embedding for the face crop, or None."""
        raise NotImplementedError

    def detect_and_embed(
        self, frame: np.ndarray
    ) -> list[tuple[BBox, Optional[float], Optional[np.ndarray]]]:
        """Detect every face and embed it in one pass.

        Returns ``(bbox, det_score, embedding)`` per face. ``det_score`` is the
        detector's confidence for that face (``None`` for detectors that don't
        expose one). The default implementation composes :meth:`detect_faces`
        and :meth:`embed`; real backends (e.g. InsightFace) override it to run
        detection + recognition together and return the genuine det score.
        """
        out: list[tuple[BBox, Optional[float], Optional[np.ndarray]]] = []
        for box in self.detect_faces(frame):
            out.append((box, None, self.embed(frame, box)))
        return out


class ComponentDetector(ModelBackend):
    task = "components"

    def infer(self, frame: np.ndarray) -> list[Detection]:  # pragma: no cover
        raise NotImplementedError


class WireAnalyzer(ModelBackend):
    task = "wires"

    def analyze(
        self, frame: np.ndarray, components: list[Detection]
    ) -> list[Wire]:  # pragma: no cover
        raise NotImplementedError
