"""Industrial component detector: dataset acquisition, synthesis, splitting,
auto-annotation, training, benchmarking, threshold tuning and export.

Modules, in the order the pipeline uses them::

    datasets   verified public source registry, label remapping, merging,
               coverage analysis and the dataset-shortfall report
    download   the fetchers (Roboflow / Kaggle / URL) and layout normalisation
    split      80/10/10 splitting grouped by capture so metrics do not leak
    autolabel  model-assisted pre-labelling plus the human annotation guide
    synthetic  crop-composited dataset multiplication
    train      training, architecture benchmarking and evaluation
    export     best.pt / best.onnx / labels.txt bundles, verification, install

Entry point: ``python -m training.electrical.cli --help``.
Procedure and the continuous-improvement loop: ``training/electrical/README.md``.
Full walkthrough: ``docs/ELECTRICAL_MODEL_TRAINING.md``.
"""

from __future__ import annotations

__all__ = ["datasets", "download", "split", "autolabel", "synthetic", "train",
           "export", "cli"]
