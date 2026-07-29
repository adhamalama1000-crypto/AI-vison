"""
Madkour AI Panel Inspector — industrial electrical intelligence.

The subsystem that recognises industrial electrical components, understands what
a control panel is *for*, and reports it like an engineer would.

Layout::

    taxonomy.py      the domain knowledge base: 50+ component classes with
                     engineering function, geometric priors and zero-shot prompts
    postprocess.py   the false-positive suppression + honest-unknown gate
    recognizer.py    inference backends (trained ONNX / Ultralytics, and
                     zero-shot OWLv2 / Grounding DINO / Florence-2, plus fusion)
    nameplate.py     manufacturer + part-number identification from OCR text
    expert.py        per-component expert annotation and bill of materials
    panel_type.py    panel-type / function inference, missing-component and
                     maintenance reasoning
    risk.py          aggregates findings into one auditable risk level, and
                     refuses to score when there is no basis for one
    inspector.py     the engine that composes all of the above into a report
    metrics.py       precision / recall / F1 / mAP / confusion matrix /
                     FP-FN analysis / threshold optimisation

Importing this package registers the component-recognition backends with
:mod:`rtsp_backend.ai.registry`.
"""

from __future__ import annotations

from . import taxonomy  # noqa: F401
from . import postprocess  # noqa: F401
from . import nameplate  # noqa: F401
from . import panel_type  # noqa: F401
from . import risk  # noqa: F401
from . import expert  # noqa: F401
from . import metrics  # noqa: F401
from . import recognizer  # noqa: F401  (registers backends)
from . import inspector  # noqa: F401

__all__ = ["taxonomy", "postprocess", "nameplate", "panel_type", "risk",
           "expert", "metrics", "recognizer", "inspector"]
