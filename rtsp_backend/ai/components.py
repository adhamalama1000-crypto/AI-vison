"""
Electrical-component detection backends.

The full inference pipeline (preprocess, ONNX forward pass, decode, NMS,
labelling, visualisation, DB persistence, API, and UI) is implemented and
tested with the shared ONNX engine. What does NOT exist is a *trained* model
for electrical panel components — no suitable public pretrained model covers
MCB / MCCB / contactors / relays / PLC modules / busbars / VFDs etc.

Therefore:
* ``onnx_components`` loads any ``.onnx`` dropped into ``models/components/``
  and runs real inference against it, mapping class indices through
  ``ELECTRICAL_CLASSES`` (override via a ``labels`` param or a labels.txt).
  Until such a model is trained/exported, this backend reports ``no_weights``
  and returns nothing — never fabricated components.
* ``null_components`` is the honest disabled state.

To enable real component detection later: train/obtain a detector on the
electrical classes below, export to ONNX, and drop it into
``models/components/``. No other code changes are required.
"""

from __future__ import annotations

import os
from typing import Optional

from .base import ComponentDetector
from .detectors import _OnnxDetectorBase
from .registry import register

ELECTRICAL_CLASSES = [
    "circuit_breaker", "mcb", "mccb", "contactor", "relay", "terminal_block",
    "plc_module", "busbar", "power_supply", "fuse", "vfd", "push_button",
    "indicator_lamp", "emergency_stop", "current_transformer",
    "voltage_transformer", "sensor", "industrial_connector",
]


def _load_labels(models_dir: str) -> list[str]:
    path = os.path.join(models_dir, "components", "labels.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            names = [ln.strip() for ln in fh if ln.strip()]
        if names:
            return names
    return ELECTRICAL_CLASSES


@register
class OnnxComponentDetector(_OnnxDetectorBase, ComponentDetector):
    backend_id = "onnx_components"
    task = "components"
    display_name = "ONNX electrical-component detector (needs trained weights)"
    default_subdir = "components"
    requires_weights = True

    def load(self) -> None:
        models_dir = self.params.get("models_dir", "models")
        self.class_names = self.params.get("labels") or _load_labels(models_dir)
        super().load()


@register
class NullComponentDetector(ComponentDetector):
    backend_id = "null_components"
    task = "components"
    display_name = "Disabled (no component detection)"
    requires_weights = False

    def load(self) -> None:
        self._ready = True
        self._status = "ready"

    def infer(self, frame):
        return []
