"""
AI Model Manager.

Owns one selected backend per task (detection, face, components, wires),
persists the choice + params + thresholds to the ``model_config`` table, and
tracks live metrics (fps, inference time, resource usage). Switching a model at
runtime is a matter of ``select(task, backend_id)`` — no server restart.

The manager is the single object the API/pipeline talk to; it hides which
concrete backend is active behind the interfaces in :mod:`.base`.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Optional

# import backend modules so their @register side-effects run
from . import components as _components  # noqa: F401
from . import detectors as _detectors  # noqa: F401
from . import embedders as _embedders  # noqa: F401
from . import modules as _modules  # noqa: F401
from . import wires as _wires  # noqa: F401
from . import registry
from .base import ModelBackend
from .face_service import FaceRecognitionService

try:
    import psutil  # optional, for resource metrics
except Exception:  # pragma: no cover
    psutil = None

# Core tasks plus the additional detection modules. Each has its own runtime
# toggle, selected backend, metrics, and events.
TASKS = (
    "detection", "face", "components", "wires",
    "fire", "violence", "fall", "weapon", "ppe", "human", "vehicle",
)

DEFAULTS = {
    "detection": ("onnx_yolo", {"conf": 0.25, "iou": 0.45, "device": "cpu"}),
    "face": ("opencv_fallback", {"threshold": 0.5, "min_blur": 40.0,
                                  "min_face_size": 24, "min_recog_blur": 12.0,
                                  "topk_vote": 3, "device": "cpu"}),
    "components": ("onnx_components", {"conf": 0.25, "iou": 0.45, "device": "cpu"}),
    "wires": ("classical_wires", {"min_wire_len": 40, "device": "cpu"}),
    "fire": ("onnx_fire", {"conf": 0.35, "iou": 0.45, "device": "cpu"}),
    "violence": ("onnx_violence", {"conf": 0.5, "iou": 0.45, "device": "cpu"}),
    "fall": ("onnx_fall", {"conf": 0.4, "iou": 0.45, "device": "cpu"}),
    "weapon": ("onnx_weapon", {"conf": 0.4, "iou": 0.45, "device": "cpu"}),
    "ppe": ("onnx_ppe", {"conf": 0.35, "iou": 0.45, "device": "cpu"}),
    "human": ("onnx_human", {"conf": 0.3, "iou": 0.45, "device": "cpu"}),
    "vehicle": ("onnx_vehicle", {"conf": 0.3, "iou": 0.45, "device": "cpu"}),
}


class TaskState:
    def __init__(self, task: str) -> None:
        self.task = task
        self.backend: Optional[ModelBackend] = None
        self.backend_id: Optional[str] = None
        self.enabled = False
        self.params: dict[str, Any] = {}
        self.infer_times: deque[float] = deque(maxlen=60)  # ms
        self.frame_times: deque[float] = deque(maxlen=60)  # monotonic
        self.last_error: Optional[str] = None

    def record(self, infer_ms: float) -> None:
        self.infer_times.append(infer_ms)
        self.frame_times.append(time.monotonic())

    def fps(self) -> float:
        if len(self.frame_times) < 2:
            return 0.0
        span = self.frame_times[-1] - self.frame_times[0]
        return round((len(self.frame_times) - 1) / span, 2) if span > 0 else 0.0

    def avg_infer_ms(self) -> Optional[float]:
        return round(sum(self.infer_times) / len(self.infer_times), 2) if self.infer_times else None


class AIModelManager:
    def __init__(self, db, models_dir: str = "models") -> None:
        self.db = db
        self.models_dir = models_dir
        self._lock = threading.RLock()
        self._tasks: dict[str, TaskState] = {t: TaskState(t) for t in TASKS}
        self.face_service: Optional[FaceRecognitionService] = None
        self._restore()

    # -- persistence -------------------------------------------------------

    def _restore(self) -> None:
        for task in TASKS:
            row = self.db.query_one(
                "SELECT backend, enabled, params FROM model_config WHERE name=?",
                (task,),
            )
            if row:
                import json
                backend_id = row["backend"]
                enabled = bool(row["enabled"])
                params = json.loads(row["params"]) if row["params"] else {}
            else:
                backend_id, params = DEFAULTS[task]
                # Face recognition is enabled out of the box: an enrolled
                # employee must be recognised immediately, with no manual
                # model toggle and no server restart. Other tasks stay opt-in.
                enabled = task == "face"
                self._persist(task, backend_id, enabled, params)
            try:
                self._instantiate(task, backend_id, params)
                self._tasks[task].enabled = enabled
                # Attempt to load every backend at startup so status is accurate
                # and any present pretrained weights are picked up automatically.
                # Weightless/optional backends fail fast and report a precise reason.
                self._try_load(task)
            except Exception as exc:  # keep booting even if one backend is broken
                self._tasks[task].last_error = str(exc)

    def _persist(self, task: str, backend_id: str, enabled: bool, params: dict) -> None:
        import json
        self.db.execute(
            "INSERT INTO model_config(name, backend, enabled, params, updated_at) "
            "VALUES(?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET "
            "backend=excluded.backend, enabled=excluded.enabled, "
            "params=excluded.params, updated_at=excluded.updated_at",
            (task, backend_id, 1 if enabled else 0, json.dumps(params), time.time()),
        )

    # -- backend lifecycle -------------------------------------------------

    def _instantiate(self, task: str, backend_id: str, params: dict) -> None:
        cls = registry.get(task, backend_id)
        params = dict(params)
        params.setdefault("models_dir", self.models_dir)
        backend = cls(**params)
        st = self._tasks[task]
        st.backend = backend
        st.backend_id = backend_id
        st.params = params
        if task == "face":
            threshold = float(params.get("threshold", 0.5))
            min_blur = float(params.get("min_blur", 40.0))
            fkw = dict(
                min_face_size=int(params.get("min_face_size", 24)),
                min_recog_blur=float(params.get("min_recog_blur", 12.0)),
                topk_vote=int(params.get("topk_vote", 3)),
            )
            if self.face_service is None:
                self.face_service = FaceRecognitionService(
                    self.db, backend, threshold, min_blur=min_blur, **fkw)
            else:
                self.face_service.embedder = backend
                self.face_service.threshold = threshold
                self.face_service.min_blur = min_blur
                self.face_service.min_face_size = fkw["min_face_size"]
                self.face_service.min_recog_blur = fkw["min_recog_blur"]
                self.face_service.topk_vote = fkw["topk_vote"]
                self.face_service.reload_cache()

    def _try_load(self, task: str) -> None:
        st = self._tasks[task]
        try:
            if st.backend is not None:
                st.backend._loading = True
                st.backend.load()
                st.backend._loading = False
                st.last_error = None
        except Exception as exc:
            if st.backend is not None:
                st.backend._loading = False
            st.last_error = str(exc)

    # -- public control ----------------------------------------------------

    def select(self, task: str, backend_id: str, params: Optional[dict] = None) -> dict:
        if task not in TASKS:
            raise KeyError(f"unknown task '{task}'")
        with self._lock:
            st = self._tasks[task]
            merged = dict(st.params)
            if params:
                merged.update(params)
            # drop internal keys that shouldn't be re-persisted verbatim
            merged.pop("models_dir", None)
            self._instantiate(task, backend_id, merged)
            self._try_load(task)  # load now so status reflects reality
            self._persist(task, backend_id, st.enabled, merged)
            return self.task_status(task)

    def set_enabled(self, task: str, enabled: bool) -> dict:
        with self._lock:
            st = self._tasks[task]
            st.enabled = enabled
            # Always (re)load on enable; on disable keep the backend loaded so its
            # status stays visible, but is_enabled() gates it out of the pipeline.
            self._try_load(task)
            self._persist(task, st.backend_id, enabled,
                          {k: v for k, v in st.params.items() if k != "models_dir"})
            return self.task_status(task)

    def ensure_enabled(self, task: str) -> bool:
        """Enable a task if it isn't already, and report whether it is now on.

        Called right after a successful face enrolment so recognition starts
        working immediately — without the operator having to visit the AI
        Models page or restart the backend. Idempotent and safe to call often.
        """
        st = self._tasks.get(task)
        if st is None:
            return False
        if not st.enabled:
            self.set_enabled(task, True)
        return self.is_enabled(task)

    def update_params(self, task: str, params: dict) -> dict:
        with self._lock:
            st = self._tasks[task]
            merged = dict(st.params)
            merged.update(params)
            self._instantiate(task, st.backend_id, merged)
            self._try_load(task)
            self._persist(task, st.backend_id, st.enabled,
                          {k: v for k, v in merged.items() if k != "models_dir"})
            return self.task_status(task)

    # -- status / metrics --------------------------------------------------

    def is_enabled(self, task: str) -> bool:
        st = self._tasks.get(task)
        return bool(st and st.enabled and st.backend is not None and st.backend.ready)

    def backend(self, task: str) -> Optional[ModelBackend]:
        st = self._tasks.get(task)
        return st.backend if st else None

    def _running(self, st: "TaskState") -> bool:
        """True if the task has produced inference within the last few seconds."""
        if not st.frame_times:
            return False
        return (time.monotonic() - st.frame_times[-1]) < 5.0

    def _state(self, st: "TaskState") -> str:
        """Map a task to the UI state vocabulary.

        loaded | not_loaded | loading | error | running | disabled
        Problems (missing weights/deps/init failure) surface even when the task
        is disabled, so nothing ever fails silently.
        """
        b = st.backend
        if b is None:
            return "not_loaded"
        s = b.status()
        if s.get("loading"):
            return "loading"
        reason = s.get("reason")
        if not s.get("ready"):
            if reason == "weights_missing":
                return "not_loaded"
            if reason in ("onnxruntime_missing", "insightface_missing", "init_failed"):
                return "error"
            if s.get("status") == "error" or s.get("error"):
                return "error"
        if not st.enabled:
            return "disabled"
        if s.get("ready"):
            return "running" if self._running(st) else "loaded"
        return "not_loaded"

    def task_status(self, task: str) -> dict:
        st = self._tasks[task]
        info = st.backend.status() if st.backend else {"backend_id": st.backend_id}
        return {
            "task": task,
            "enabled": st.enabled,
            "selected_backend": st.backend_id,
            "state": self._state(st),
            "reason": info.get("reason"),
            "detail": info.get("error"),
            "backend": info,
            "available_backends": registry.catalog().get(task, []),
            "metrics": {
                "fps": st.fps(),
                "avg_inference_ms": st.avg_infer_ms(),
            },
            "last_error": st.last_error,
        }

    def resource_metrics(self) -> dict:
        data = {"cpu_percent": None, "ram_percent": None, "ram_used_mb": None,
                "gpu_percent": None, "gpu_mem_mb": None, "gpu_available": False}
        if psutil is not None:
            try:
                data["cpu_percent"] = psutil.cpu_percent(interval=None)
                vm = psutil.virtual_memory()
                data["ram_percent"] = vm.percent
                data["ram_used_mb"] = round(vm.used / (1024 * 1024), 1)
            except Exception:
                pass
        # GPU metrics require a CUDA stack that isn't present in this environment;
        # reported as unavailable rather than fabricated.
        return data

    def full_status(self) -> dict:
        return {
            "tasks": {t: self.task_status(t) for t in TASKS},
            "resources": self.resource_metrics(),
            "catalog": registry.catalog(),
        }

    def record_infer(self, task: str, infer_ms: float) -> None:
        st = self._tasks.get(task)
        if st:
            st.record(infer_ms)
