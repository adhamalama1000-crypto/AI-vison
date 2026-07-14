"""AI subsystem: pluggable model backends, registry, manager, and pipeline."""

from .manager import AIModelManager, TASKS
from .pipeline import AIPipeline

__all__ = ["AIModelManager", "AIPipeline", "TASKS"]
