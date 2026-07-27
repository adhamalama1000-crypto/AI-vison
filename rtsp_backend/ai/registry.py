"""
Plugin registry for AI model backends.

Backends register a *factory* (class) under a task. The manager instantiates the
selected backend per task. New models are added by writing a class and calling
``register(MyBackend)`` — no changes to the API layer or the UI are required.
"""

from __future__ import annotations

from typing import Callable, Type

from .base import ModelBackend

# task -> {backend_id -> factory}
_REGISTRY: dict[str, dict[str, Type[ModelBackend]]] = {}


def register(cls: Type[ModelBackend]) -> Type[ModelBackend]:
    """Class decorator / function to register a backend."""
    _REGISTRY.setdefault(cls.task, {})[cls.backend_id] = cls
    return cls


def get(task: str, backend_id: str) -> Type[ModelBackend]:
    try:
        return _REGISTRY[task][backend_id]
    except KeyError as exc:
        raise KeyError(f"No backend '{backend_id}' registered for task '{task}'") from exc


def list_backends(task: str) -> list[Type[ModelBackend]]:
    return list(_REGISTRY.get(task, {}).values())


def all_tasks() -> list[str]:
    return list(_REGISTRY.keys())


def catalog() -> dict[str, list[dict]]:
    """UI-friendly listing of every registered backend, grouped by task."""
    out: dict[str, list[dict]] = {}
    for task, backends in _REGISTRY.items():
        out[task] = [
            {
                "backend_id": cls.backend_id,
                "display_name": cls.display_name,
                "requires_weights": getattr(cls, "requires_weights", False),
                "deprecated": getattr(cls, "deprecated", False),
                "experimental": getattr(cls, "experimental", False),
                "warning": getattr(cls, "warning", None),
            }
            for cls in backends.values()
        ]
    return out
