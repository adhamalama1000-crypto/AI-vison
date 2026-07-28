"""
Export a trained checkpoint into a deployable bundle.

The brief asks for ``best.pt``, ``best.onnx`` and ``labels.txt``. Producing those
three files is easy; producing them so that the *runtime* cannot mislabel is the
actual job, and this project has already been burned by exactly that: an earlier
pipeline shipped a ``labels.txt`` containing the literal lines ``0``…``9``, so
every detection came back named ``"0"``. See
:func:`rtsp_backend.electrical.recognizer.load_class_map`.

So a bundle here is:

``best.pt``      the Ultralytics checkpoint (GPU training, further fine-tuning)
``best.onnx``    the portable graph the CPU runtime loads
``labels.txt``   one canonical class id per line, in training index order
``classes.json`` the authoritative label order plus taxonomy version — this is
                 what the runtime prefers, with ``labels.txt`` as the fallback
``model_card.json``  what was trained, on what, scoring what, and what it cannot do

:func:`verify_bundle` then re-reads the bundle the way the runtime will and
checks that the ONNX graph's output width actually matches the label count. A
class-count mismatch between graph and labels is the failure that shifts every
label by one and is invisible until someone notices a contactor being called a
relay — so it is a hard error here, at export time, where it is cheap.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Callable, Optional, Sequence

from rtsp_backend.electrical import taxonomy as tax

BUNDLE_FILES = ("best.pt", "best.onnx", "labels.txt", "classes.json",
                "model_card.json")

#: Runtime install target. Dropping a bundle here is all the backend needs.
DEFAULT_INSTALL_DIR = os.path.join("models", "components")


def write_labels(directory: str,
                 classes: Sequence[str] = tax.CLASS_ORDER) -> str:
    """Write ``labels.txt``: one canonical class id per line, in index order."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "labels.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(classes) + "\n")
    return path


#: Kept verbatim at the top of every ``classes.json`` so anyone who opens the
#: file — including whoever is debugging a mislabelled detection at 2am — learns
#: the append-only rule from the file itself.
CLASSES_JSON_COMMENT = (
    "Canonical class order for the industrial component detector. This file is "
    "the authoritative label map: the recogniser reads it instead of guessing, "
    "and training/electrical writes it next to every exported checkpoint. "
    "APPEND ONLY - reordering invalidates existing weights."
)


def write_classes_json(directory: str,
                       classes: Sequence[str] = tax.CLASS_ORDER,
                       taxonomy_version: str = "5.1") -> str:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "classes.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"_comment": CLASSES_JSON_COMMENT,
                   "taxonomy_version": taxonomy_version,
                   "class_count": len(classes),
                   "classes": list(classes),
                   "display_names": {c: tax.display_name(c) for c in classes}},
                  fh, indent=2)
    return path


def onnx_output_classes(onnx_path: str) -> Optional[int]:
    """Infer the class count from an ONNX detection head's output shape.

    Ultralytics detect exports have output ``(1, 4 + nc, anchors)``. Returns
    ``None`` when the shape is dynamic or unreadable rather than guessing — a
    wrong guess here would produce a spurious hard error at export.
    """
    try:
        import onnx  # type: ignore
    except ImportError:
        return None
    try:
        model = onnx.load(onnx_path)
        out = model.graph.output[0]
        dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
    except Exception:
        return None
    if len(dims) != 3:
        return None
    # (1, 4+nc, anchors) — the class axis is whichever of dims[1]/dims[2] is the
    # smaller; anchor counts are in the thousands, 4+nc is in the tens.
    candidates = [d for d in dims[1:] if d > 4]
    if not candidates:
        return None
    return min(candidates) - 4


def export_bundle(weights: str, out_dir: str,
                  imgsz: int = 960,
                  opset: int = 12,
                  simplify: bool = True,
                  half: bool = False,
                  dynamic: bool = False,
                  classes: Sequence[str] = tax.CLASS_ORDER,
                  metadata: Optional[dict] = None,
                  log: Optional[Callable[[str], None]] = None) -> dict:
    """Build a complete deployable bundle from an Ultralytics ``.pt`` checkpoint.

    ONNX export needs ``ultralytics`` (and therefore torch) installed; when it is
    not, the label files and model card are still written and the ONNX step is
    reported as ``skipped`` with the reason. That is deliberate — a bundle missing
    its ONNX is useful for GPU deployment and for fine-tuning, and pretending the
    export happened would be worse than saying it did not.
    """
    say = log or (lambda m: None)
    result: dict = {"status": "exported", "out_dir": out_dir,
                    "weights_source": weights, "files": {}, "warnings": []}

    if not os.path.exists(weights):
        return {"status": "failed", "reason": f"checkpoint not found: {weights}"}
    os.makedirs(out_dir, exist_ok=True)

    pt_dst = os.path.join(out_dir, "best.pt")
    if os.path.abspath(weights) != os.path.abspath(pt_dst):
        shutil.copy2(weights, pt_dst)
    result["files"]["best.pt"] = pt_dst
    say(f"best.pt -> {pt_dst}")

    result["files"]["labels.txt"] = write_labels(out_dir, classes)
    result["files"]["classes.json"] = write_classes_json(out_dir, classes)
    say(f"labels.txt / classes.json -> {len(classes)} class(es)")

    # -- ONNX ------------------------------------------------------------
    onnx_path: Optional[str] = None
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        result["onnx"] = {"status": "skipped",
                          "reason": f"ultralytics is not installed ({exc}); "
                                    f"pip install ultralytics to export ONNX"}
        result["warnings"].append(
            "no best.onnx in this bundle — the CPU ONNX runtime backend cannot "
            "load it. Export on a machine with ultralytics installed.")
    else:
        try:
            say(f"exporting ONNX at {imgsz}px, opset {opset}")
            exported = YOLO(pt_dst).export(
                format="onnx", imgsz=imgsz, opset=opset, simplify=simplify,
                half=half, dynamic=dynamic)
            src = str(exported) if exported else ""
            if src and os.path.exists(src):
                onnx_path = os.path.join(out_dir, "best.onnx")
                if os.path.abspath(src) != os.path.abspath(onnx_path):
                    shutil.move(src, onnx_path)
                result["files"]["best.onnx"] = onnx_path
                result["onnx"] = {"status": "exported", "imgsz": imgsz,
                                  "opset": opset, "simplify": simplify,
                                  "half": half, "dynamic": dynamic}
                say(f"best.onnx -> {onnx_path}")
            else:
                result["onnx"] = {"status": "failed",
                                  "reason": "ultralytics export returned no path"}
        except Exception as exc:
            result["onnx"] = {"status": "failed",
                              "reason": f"{type(exc).__name__}: {exc}"}
            result["warnings"].append(f"ONNX export failed: {exc}")

    # -- model card ------------------------------------------------------
    card = {
        "name": "Madkour industrial electrical component detector",
        "taxonomy_version": "5.1",
        "class_count": len(classes),
        "classes": list(classes),
        "input": {"imgsz": imgsz, "layout": "BGR uint8 image, letterboxed",
                  "note": "the runtime handles letterboxing and normalisation"},
        "output": {"format": "bounding box (xyxy, pixels), confidence, class",
                   "note": "post-processed by rtsp_backend.electrical.postprocess, "
                           "which applies per-class thresholds and geometric "
                           "plausibility gating and demotes anything it cannot "
                           "confirm to 'unknown_industrial_component'"},
        "limitations": [
            "Accuracy is bounded by the training data. Run "
            "training.electrical.datasets.requirements_report() against the "
            "training set and treat any class listed as weak/untrainable as "
            "unvalidated, whatever the headline mAP says.",
            "Classes with no validation instances are excluded from mAP entirely "
            "— check the split report's classes_absent_from_val.",
            "Trained on visible-light imagery; thermal input is out of "
            "distribution.",
        ],
        "provenance": metadata or {},
    }
    card_path = os.path.join(out_dir, "model_card.json")
    with open(card_path, "w", encoding="utf-8") as fh:
        json.dump(card, fh, indent=2)
    result["files"]["model_card.json"] = card_path

    result["verification"] = verify_bundle(out_dir)
    if not result["verification"]["ok"]:
        result["warnings"].extend(result["verification"]["problems"])
    return result


def verify_bundle(bundle_dir: str) -> dict:
    """Re-read a bundle the way the runtime will, and report any mismatch."""
    problems: list[str] = []
    info: dict = {"bundle_dir": bundle_dir, "present": {}, "ok": True}

    for name in BUNDLE_FILES:
        path = os.path.join(bundle_dir, name)
        info["present"][name] = os.path.exists(path)

    classes: list[str] = []
    cj = os.path.join(bundle_dir, "classes.json")
    if os.path.exists(cj):
        try:
            with open(cj, "r", encoding="utf-8") as fh:
                classes = [str(c) for c in (json.load(fh) or {}).get("classes", [])]
        except (OSError, ValueError) as exc:
            problems.append(f"classes.json is unreadable: {exc}")
    else:
        problems.append("classes.json is missing — the runtime would fall back "
                        "to labels.txt and then to a hardcoded list")

    lt = os.path.join(bundle_dir, "labels.txt")
    labels: list[str] = []
    if os.path.exists(lt):
        with open(lt, "r", encoding="utf-8") as fh:
            labels = [ln.strip() for ln in fh if ln.strip()]
        # This is the exact bug that shipped once already.
        if labels and all(ln.lstrip("-").isdigit() for ln in labels):
            problems.append(
                "labels.txt contains only integers. The runtime rejects it as "
                "not-real-names, but it must never be written this way — every "
                "detection would be labelled with a bare index.")
        if classes and labels and labels != classes:
            problems.append(
                f"labels.txt and classes.json disagree "
                f"({len(labels)} vs {len(classes)} entries, or different order). "
                f"They must be byte-identical in content and order.")
    else:
        problems.append("labels.txt is missing")

    expected = list(classes or labels)
    if expected and list(expected) != list(tax.CLASS_ORDER):
        # Not fatal — an older bundle legitimately has fewer classes because
        # CLASS_ORDER is append-only — but it must be visible.
        if list(expected) == list(tax.CLASS_ORDER[:len(expected)]):
            info["taxonomy_note"] = (
                f"bundle has {len(expected)} classes, the current taxonomy has "
                f"{len(tax.CLASS_ORDER)}. The bundle is a valid prefix, so "
                f"existing indices are correct and the newer classes simply "
                f"cannot be detected by this checkpoint.")
        else:
            problems.append(
                f"bundle label order does not match the taxonomy and is not a "
                f"prefix of it. Indices would be misinterpreted at runtime. "
                f"Bundle: {expected[:5]}... Taxonomy: "
                f"{list(tax.CLASS_ORDER[:5])}...")

    onnx_path = os.path.join(bundle_dir, "best.onnx")
    if os.path.exists(onnx_path) and expected:
        nc = onnx_output_classes(onnx_path)
        info["onnx_class_count"] = nc
        if nc is not None and nc != len(expected):
            problems.append(
                f"best.onnx has a {nc}-class head but the bundle declares "
                f"{len(expected)} labels. Every label would be shifted. Re-export "
                f"from a checkpoint trained on this label space.")

    info["classes"] = len(expected)
    info["problems"] = problems
    info["ok"] = not problems
    return info


def install_bundle(bundle_dir: str,
                   install_dir: str = DEFAULT_INSTALL_DIR,
                   log: Optional[Callable[[str], None]] = None) -> dict:
    """Copy a verified bundle into ``models/components/`` for the backend.

    Refuses to install a bundle that fails verification: a mislabelled model in
    production is worse than no model, because the platform reports confident
    wrong component names instead of an honest "no weights".
    """
    say = log or (lambda m: None)
    check = verify_bundle(bundle_dir)
    if not check["ok"]:
        return {"status": "refused",
                "reason": "bundle failed verification; refusing to install a "
                          "model that would mislabel detections",
                "problems": check["problems"]}
    os.makedirs(install_dir, exist_ok=True)
    copied = []
    for name in BUNDLE_FILES:
        src = os.path.join(bundle_dir, name)
        if os.path.exists(src):
            dst = os.path.join(install_dir, name)
            shutil.copy2(src, dst)
            copied.append(dst)
            say(f"installed {dst}")
    return {
        "status": "installed", "install_dir": install_dir, "files": copied,
        "verification": check,
        "next_step": (
            "Select the 'industrial_onnx' components backend (POST "
            "/api/ai/models/components with backend_id=industrial_onnx), or "
            "'industrial_ultralytics' to run best.pt on a GPU. The backend picks "
            "the bundle up on next load."),
    }


def tensorrt_instructions(bundle_dir: str = DEFAULT_INSTALL_DIR,
                          imgsz: int = 960) -> dict:
    """How to build a TensorRT engine from this bundle, and the caveats.

    Deliberately instructions rather than an implementation: a TensorRT engine is
    tied to the exact GPU architecture, driver and TensorRT version of the machine
    that builds it, so it must be built on the deployment host. Producing one here
    and shipping it would give a file that fails to load in production.
    """
    return {
        "why_not_automated": (
            "A TensorRT engine is not portable. It is compiled for one GPU "
            "compute capability, one TensorRT version and one driver, so it must "
            "be built on the deployment machine. Building it in CI and shipping "
            "the .engine file produces a binary that fails to deserialise on any "
            "other host."),
        "prerequisites": [
            "NVIDIA GPU with a driver matching the CUDA build",
            "pip install ultralytics tensorrt onnx onnxruntime-gpu",
            "nvidia-smi must report the GPU inside the container if using Docker "
            "(--gpus all, plus the NVIDIA Container Toolkit)",
        ],
        "build_from_pt": (
            f"yolo export model={os.path.join(bundle_dir, 'best.pt')} "
            f"format=engine imgsz={imgsz} half=True device=0"),
        "build_from_onnx": (
            f"trtexec --onnx={os.path.join(bundle_dir, 'best.onnx')} "
            f"--saveEngine={os.path.join(bundle_dir, 'best.engine')} "
            f"--fp16 --workspace=4096"),
        "verify": (
            "Run training.electrical.cli eval against the engine before trusting "
            "it. FP16 conversion changes outputs slightly and INT8 changes them "
            "materially; re-measure mAP rather than assuming parity with the "
            "ONNX model."),
        "runtime": (
            "Point the components backend at the .engine via the "
            "'industrial_ultralytics' backend's weights param. Keep best.onnx "
            "installed as the CPU fallback so the platform still works if the "
            "GPU is unavailable."),
        "expected_gain": (
            "Typically 2–4× over ONNX Runtime CPU on the same image size for a "
            "yolo11s at 960px. Measure on your hardware; the ratio depends "
            "heavily on batch size and whether the GPU is shared with other "
            "inference."),
    }


__all__ = ["BUNDLE_FILES", "DEFAULT_INSTALL_DIR", "CLASSES_JSON_COMMENT",
           "write_labels", "write_classes_json", "onnx_output_classes",
           "export_bundle", "verify_bundle", "install_bundle",
           "tensorrt_instructions"]
