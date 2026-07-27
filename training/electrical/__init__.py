"""Industrial component detector: dataset acquisition, synthesis, training,
benchmarking and threshold tuning.

Entry point: ``python -m training.electrical.cli --help``.
Procedure and the continuous-improvement loop: ``training/electrical/README.md``.
"""

from __future__ import annotations

__all__ = ["datasets", "synthetic", "train", "cli"]
