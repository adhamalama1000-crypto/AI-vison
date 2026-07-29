"""
Industrial component recognition backends.

Replaces ``rtsp_backend.ai.components.OnnxComponentDetector``, which had three
defects that made it unusable even *with* a trained model:

1. **Broken class mapping.** It inferred the export format from the raw column
   count (``row.shape[0] >= 85``). That test only holds for an 80-class COCO
   model. A YOLOv5 export of any electrical class set has ``4 + 1 + nc``
   columns — 32 for the old 27-class list, 58 for today's taxonomy — all below
   85, so it took the YOLOv8 branch, read the objectness column as class 0, and
   shifted **every** label by one. On top of that ``models/components/
   labels.txt`` shipped as the literal lines ``0``–``9``, so labels came out as
   the strings ``"0"``…``"9"``.
2. **Class-agnostic NMS.** One NMS across all classes at once suppresses an
   overload relay bolted under a contactor, and keeps cross-class duplicates.
3. **A single global confidence threshold** and nothing else — no geometric
   sanity check, no honest "unknown" path.

This module fixes all three: the export format is determined from the *declared
class count*, decoding is vectorised (no Python loop over 8400 rows), labels are
resolved through the taxonomy, and every output goes through the
:mod:`.postprocess` cascade.

Three families of backend are provided:

``industrial_onnx``
    A trained detector exported to ONNX (YOLOv8/v11 or YOLOv5 layout, and
    RT-DETR). This is the production path once a model is trained. Reports
    ``weights_missing`` and returns nothing when no checkpoint is present —
    never a fabricated box.

``industrial_ultralytics``
    Direct Ultralytics inference (``.pt`` checkpoints, YOLOv8/v11/RT-DETR)
    without an ONNX export step. Useful immediately after training.

``openvocab_owlv2`` / ``openvocab_grounding_dino`` / ``openvocab_florence2``
    Open-vocabulary, text-conditioned detectors driven by the taxonomy's
    engineer-phrased prompts. These need **no custom dataset** — the realistic
    path to recognising panel components before a bespoke model exists. They
    require ``transformers`` plus the model weights; where those are not
    available the backend reports exactly which is missing.

``industrial_ensemble``
    Fuses several of the above (score-weighted, per class) so a zero-shot model
    can cover classes a small trained model has not learned yet.

Every backend implements :meth:`recognize`, returning
:class:`~.postprocess.Candidate` objects with raw scores, and the
:class:`~rtsp_backend.ai.base.ComponentDetector` ``infer`` interface so the
existing pipeline, API and UI work unchanged.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from typing import Any, Optional, Sequence

import numpy as np

from ..ai.base import BBox, ComponentDetector, Detection
from ..ai.registry import register
from . import postprocess as pp
from . import taxonomy as tax

_log = logging.getLogger("rtsp_backend.electrical.recognizer")

DEFAULT_SUBDIR = "components"


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------

def letterbox(img: np.ndarray, size: int = 640, pad_value: int = 114
              ) -> tuple[np.ndarray, float, float, float]:
    """Resize keeping aspect ratio and pad to a square. Returns (img, r, dw, dh).

    Never upscales beyond 1.0 — upscaling a small crop adds no information and
    changes the effective object scale the model was trained at.
    """
    import cv2

    h, w = img.shape[:2]
    r = min(size / max(1, h), size / max(1, w))
    nh, nw = max(1, int(round(h * r))), max(1, int(round(w * r)))
    interp = cv2.INTER_AREA if r < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    dh, dw = (size - nh) // 2, (size - nw) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized
    return canvas, float(r), float(dw), float(dh)


def to_blob(img: np.ndarray) -> np.ndarray:
    """HWC BGR uint8 → NCHW RGB float32 in [0, 1]."""
    rgb = np.ascontiguousarray(img[:, :, ::-1])
    blob = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return blob[None, ...]


# --------------------------------------------------------------------------
# class map loading
# --------------------------------------------------------------------------

def load_class_map(models_dir: str, subdir: str = DEFAULT_SUBDIR,
                   explicit: Optional[Sequence[str]] = None
                   ) -> tuple[list[str], str]:
    """Resolve the model's class list. Returns ``(names, source)``.

    Order of preference:

    1. an explicit ``labels`` param;
    2. ``classes.json`` — the file :mod:`training.electrical` writes, holding
       the exact ordered class list the checkpoint was trained with;
    3. ``labels.txt`` — one name per line, but **only if it looks like real
       labels**. A file of bare integers (as shipped) is rejected outright
       rather than silently producing labels ``"0"``, ``"1"``, …;
    4. the taxonomy's canonical :data:`~.taxonomy.CLASS_ORDER`.
    """
    if explicit:
        return [str(x) for x in explicit], "params"

    base = os.path.join(models_dir, subdir)
    js = os.path.join(base, "classes.json")
    if os.path.exists(js):
        try:
            with open(js, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            names = data.get("classes") if isinstance(data, dict) else data
            if isinstance(names, list) and names:
                return [str(n) for n in names], "classes.json"
        except Exception as exc:  # pragma: no cover - corrupt file
            _log.warning("classes.json unreadable (%s); falling back", exc)

    txt = os.path.join(base, "labels.txt")
    if os.path.exists(txt):
        try:
            with open(txt, "r", encoding="utf-8") as fh:
                names = [ln.strip() for ln in fh if ln.strip()]
        except Exception:
            names = []
        # Reject a placeholder/index-only file — it is worse than no file.
        if names and not all(n.lstrip("-").isdigit() for n in names):
            return names, "labels.txt"
        if names:
            _log.warning(
                "%s contains only numeric placeholders — ignoring it and using "
                "the canonical taxonomy order instead.", txt)

    return list(tax.CLASS_ORDER), "taxonomy"


def resolve_names(names: Sequence[str]) -> tuple[list[Optional[str]], list[str]]:
    """Map raw model label names onto canonical taxonomy ids.

    Returns ``(canonical_or_none_per_index, unmapped_names)``. A name the
    taxonomy does not know stays ``None``; its detections are reported as
    ``unknown_industrial_component`` rather than under a label the rest of the
    system cannot reason about.
    """
    canon: list[Optional[str]] = []
    unmapped: list[str] = []
    for n in names:
        cid = tax.resolve(n)
        canon.append(cid)
        if cid is None:
            unmapped.append(str(n))
    return canon, unmapped


# --------------------------------------------------------------------------
# output decoding
# --------------------------------------------------------------------------

def decode_yolo(out: np.ndarray, num_classes: int, conf_thr: float,
                r: float, dw: float, dh: float
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised decode of a YOLOv5 / YOLOv8-v11 head.

    Format is decided from the **declared class count**, not a magic column
    threshold — this is the fix for the label-shift bug:

    * ``cols == num_classes + 5`` → YOLOv5 (cx, cy, w, h, obj, classes…)
    * ``cols == num_classes + 4`` → YOLOv8/v11 (cx, cy, w, h, classes…)

    Coordinates are un-letterboxed back to original image pixels.
    """
    arr = np.asarray(out, dtype=np.float32)
    # Drop leading batch dimensions only. A blanket np.squeeze would collapse a
    # single-detection output such as (1, 4+nc, 1) to 1-D and silently lose it —
    # which is exactly the case of a panel with one recognised component.
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    while arr.ndim > 2 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        return _empty_decode()

    # Orient so rows are predictions. Ultralytics ONNX emits (4+nc, N); the
    # reference implementations emit (N, 4+nc). Decide from the declared class
    # count rather than from which dimension happens to be larger.
    widths = (num_classes + 4, num_classes + 5)
    if arr.shape[1] not in widths and arr.shape[0] in widths:
        arr = arr.T

    cols = arr.shape[1]
    if cols == num_classes + 5:
        obj = arr[:, 4]
        cls_scores = arr[:, 5:] * obj[:, None]
    elif cols == num_classes + 4:
        cls_scores = arr[:, 4:]
    elif cols > 5:
        # Unexpected width: assume the trailing columns are class scores and
        # log it — better than silently shifting every label.
        _log.warning(
            "detector output has %d columns for %d declared classes; assuming "
            "an anchor-free head with %d classes. Verify models/%s/classes.json.",
            cols, num_classes, cols - 4, DEFAULT_SUBDIR)
        cls_scores = arr[:, 4:]
    else:
        return _empty_decode()

    if cls_scores.size == 0:
        return _empty_decode()
    cls_ids = cls_scores.argmax(axis=1)
    scores = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]
    keep = scores >= conf_thr
    if not np.any(keep):
        return _empty_decode()

    xywh = arr[keep, :4]
    scores = scores[keep]
    cls_ids = cls_ids[keep]

    cx, cy, bw, bh = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    r = r if r > 0 else 1.0
    x1 = (cx - bw / 2.0 - dw) / r
    y1 = (cy - bh / 2.0 - dh) / r
    x2 = (cx + bw / 2.0 - dw) / r
    y2 = (cy + bh / 2.0 - dh) / r
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    return boxes.astype(np.float32), scores.astype(np.float32), cls_ids.astype(np.int32)


def decode_rtdetr(outputs: Sequence[np.ndarray], num_classes: int,
                  conf_thr: float, img_w: float, img_h: float
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode an RT-DETR export.

    RT-DETR emits normalised ``cxcywh`` boxes and per-query class logits with no
    objectness and no NMS requirement. Ultralytics' RT-DETR ONNX export already
    concatenates them as ``(1, N, 4 + nc)`` with sigmoid-activated scores, which
    :func:`decode_yolo` handles; this path covers the two-tensor exports
    (``boxes``, ``logits``) produced by the reference implementation.
    """
    if len(outputs) < 2:
        return _empty_decode()

    def _as_2d(x):
        x = np.asarray(x, dtype=np.float32)
        while x.ndim > 2 and x.shape[0] == 1:
            x = x[0]
        return x.reshape(1, -1) if x.ndim == 1 else x

    a, b = _as_2d(outputs[0]), _as_2d(outputs[1])
    if a.ndim != 2 or b.ndim != 2:
        return _empty_decode()
    boxes_n, logits = (a, b) if a.shape[-1] == 4 else (b, a)
    if boxes_n.shape[-1] != 4 or logits.shape[-1] < 1:
        return _empty_decode()

    # Reference RT-DETR emits raw logits; apply sigmoid when values leave [0, 1].
    if logits.min() < 0.0 or logits.max() > 1.0:
        logits = 1.0 / (1.0 + np.exp(-logits))
    cls_ids = logits.argmax(axis=1)
    scores = logits[np.arange(logits.shape[0]), cls_ids]
    keep = scores >= conf_thr
    if not np.any(keep):
        return _empty_decode()
    bn = boxes_n[keep]
    cx, cy, bw, bh = bn[:, 0] * img_w, bn[:, 1] * img_h, bn[:, 2] * img_w, bn[:, 3] * img_h
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    return (boxes.astype(np.float32), scores[keep].astype(np.float32),
            cls_ids[keep].astype(np.int32))


def _empty_decode() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
            np.zeros((0,), np.int32))


# --------------------------------------------------------------------------
# base class
# --------------------------------------------------------------------------

class IndustrialRecognizer(ComponentDetector):
    """Shared behaviour: gating config, candidate → Detection adaptation."""

    task = "components"
    requires_weights = True

    def gate_config(self) -> pp.GateConfig:
        cfg = pp.GateConfig()
        if "strictness" in self.params:
            cfg.strictness = float(self.params["strictness"])
        if "nms_iou" in self.params:
            cfg.nms_iou = float(self.params["nms_iou"])
        if "unknown_floor" in self.params:
            cfg.unknown_floor = float(self.params["unknown_floor"])
        if "max_detections" in self.params:
            cfg.max_detections = int(self.params["max_detections"])
        if "check_plausibility" in self.params:
            cfg.check_plausibility = bool(self.params["check_plausibility"])
        overrides = self.params.get("thresholds")
        if isinstance(overrides, dict):
            cfg.thresholds = {**cfg.thresholds, **{
                str(k): float(v) for k, v in overrides.items()}}
        return cfg

    # -- subclass contract -------------------------------------------------

    def raw_candidates(self, frame: np.ndarray) -> list[pp.Candidate]:
        raise NotImplementedError

    def raw_candidates_batch(self, frames: Sequence[np.ndarray]
                             ) -> list[list[pp.Candidate]]:
        """Batched raw inference. Override where the backend genuinely batches.

        The default is a sequential loop, which is correct but gains nothing. That
        is deliberate: a subclass that cannot batch inherits working behaviour
        rather than a wrong answer, and :attr:`supports_true_batching` tells the
        caller which it got, so a throughput claim is never made on the strength of
        a fake batch.
        """
        return [self.raw_candidates(f) for f in frames]

    #: Whether :meth:`raw_candidates_batch` is a real batched forward pass.
    supports_true_batching: bool = False

    # -- public ------------------------------------------------------------

    def recognize(self, frame: np.ndarray) -> pp.GateResult:
        """Full recognition: raw inference through the post-processing cascade."""
        if not self._ready:
            self.load()
        cands = self.raw_candidates(frame)
        return pp.run(cands, frame.shape[:2], self.gate_config())

    def recognize_batch(self, frames: Sequence[np.ndarray],
                        batch_size: int = 8) -> list[pp.GateResult]:
        """Recognise several frames, batching the forward pass where possible.

        Returns one :class:`~rtsp_backend.electrical.postprocess.GateResult` per
        input frame, in order. Post-processing stays per-frame because the
        geometric plausibility gate is relative to each image's own dimensions —
        batching that would apply one panel's geometry to another's boxes.

        Not used by the RTSP path on purpose: panel inspection is not a real-time
        problem (a cabinet does not change between frames), so buffering frames to
        fill a batch would add latency for no accuracy gain. This exists for
        folder-scale work — re-scoring an archive, or an auto-annotation pass over
        a capture batch.
        """
        if not self._ready:
            self.load()
        if not frames:
            return []
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        cfg = self.gate_config()
        out: list[pp.GateResult] = []
        for start in range(0, len(frames), batch_size):
            chunk = list(frames[start:start + batch_size])
            try:
                per_frame = self.raw_candidates_batch(chunk)
            except Exception:
                # A batched path that fails must not lose the whole chunk; fall
                # back to per-frame inference so the caller still gets results.
                _log.exception("batched inference failed; falling back to "
                               "per-frame for this chunk")
                per_frame = [self.raw_candidates(f) for f in chunk]
            if len(per_frame) != len(chunk):
                raise RuntimeError(
                    f"{type(self).__name__}.raw_candidates_batch returned "
                    f"{len(per_frame)} result(s) for {len(chunk)} frame(s); "
                    f"results would be misattributed to the wrong images")
            for frame, cands in zip(chunk, per_frame):
                out.append(pp.run(cands, frame.shape[:2], cfg))
        return out

    def infer_batch(self, frames: Sequence[np.ndarray],
                    batch_size: int = 8) -> list[list[Detection]]:
        """Batched :class:`ComponentDetector` interface."""
        return [self._to_detections(r)
                for r in self.recognize_batch(frames, batch_size)]

    def _to_detections(self, result: pp.GateResult) -> list[Detection]:
        out: list[Detection] = []
        for c in result.accepted:
            sp = tax.spec(c.class_id)
            out.append(Detection(
                label=sp.name, confidence=float(c.score),
                bbox=BBox(*[float(v) for v in c.box]), kind="component",
                extra={"class_id": c.class_id, "category": sp.category,
                       "source": c.source, **dict(c.extra)},
            ))
        self._last_diagnostics = result.diagnostics.to_dict()
        return out

    def infer(self, frame: np.ndarray) -> list[Detection]:
        """:class:`ComponentDetector` interface used by the live pipeline."""
        return self._to_detections(self.recognize(frame))

    def status(self) -> dict:
        st = super().status()
        st["supports_true_batching"] = bool(self.supports_true_batching)
        st["diagnostics"] = getattr(self, "_last_diagnostics", None)
        st["class_count"] = len(getattr(self, "class_names", ()) or ())
        st["class_source"] = getattr(self, "class_source", None)
        st["unmapped_classes"] = getattr(self, "unmapped_classes", [])
        return st


# --------------------------------------------------------------------------
# ONNX backend
# --------------------------------------------------------------------------

@register
class OnnxIndustrialRecognizer(IndustrialRecognizer):
    """Trained industrial component detector, ONNX runtime."""

    backend_id = "industrial_onnx"
    display_name = "Industrial component detector — ONNX (trained weights)"
    default_subdir = DEFAULT_SUBDIR

    def _find_weights(self) -> Optional[str]:
        p = self.params.get("weights")
        if p and os.path.exists(p):
            return str(p)
        models_dir = self.params.get("models_dir", "models")
        found = sorted(glob.glob(os.path.join(models_dir, self.default_subdir,
                                             "*.onnx")))
        return found[0] if found else None

    def load(self) -> None:
        models_dir = self.params.get("models_dir", "models")
        names, source = load_class_map(models_dir, self.default_subdir,
                                       self.params.get("labels"))
        self.class_names = names
        self.class_source = source
        self.canonical, self.unmapped_classes = resolve_names(names)

        weights = self._find_weights()
        if not weights:
            self._ready = False
            self._status = "no_weights"
            self._reason = "weights_missing"
            self._error = (
                f"No trained component detector found. Export one to "
                f"models/{self.default_subdir}/*.onnx (see "
                f"training/electrical/README.md). No components are reported "
                f"until then — nothing is fabricated.")
            raise RuntimeError(self._error)

        try:
            import onnxruntime as ort
        except Exception as exc:
            self._ready = False
            self._status = "error"
            self._reason = "onnxruntime_missing"
            self._error = f"onnxruntime is not installed: {exc}"
            raise RuntimeError(self._error)

        try:
            available = ort.get_available_providers()
            want_gpu = str(self.params.get("device", "cpu")).lower() in ("gpu", "cuda")
            providers = ["CPUExecutionProvider"]
            self._reason = None
            note = None
            if want_gpu:
                if "CUDAExecutionProvider" in available:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                else:
                    self._reason = "cuda_unavailable"
                    note = ("GPU requested but CUDA is unavailable here; "
                            "running on CPU.")
            self._sess = ort.InferenceSession(weights, providers=providers)
            self._input = self._sess.get_inputs()[0].name
            shape = self._sess.get_inputs()[0].shape
            self._imgsz = int(self.params.get("imgsz") or
                              (shape[2] if isinstance(shape[2], int) else 640))
            self._weights_path = weights
            self._n_outputs = len(self._sess.get_outputs())
            self._ready = True
            self._status = "ready"
            self._error = note
        except Exception as exc:
            self._ready = False
            self._status = "error"
            self._reason = "init_failed"
            self._error = f"failed to load ONNX model: {exc}"
            raise RuntimeError(self._error)

    def raw_candidates(self, frame: np.ndarray) -> list[pp.Candidate]:
        # Decode at a low floor and let the gate decide — a per-class threshold
        # cannot be applied before the class is known.
        floor = float(self.params.get("decode_floor", 0.10))
        img, r, dw, dh = letterbox(frame, self._imgsz)
        outs = self._sess.run(None, {self._input: to_blob(img)})
        nc = len(self.class_names)

        boxes, scores, cls_ids = decode_yolo(outs[0], nc, floor, r, dw, dh)
        if boxes.shape[0] == 0 and self._n_outputs >= 2:
            boxes, scores, cls_ids = decode_rtdetr(
                outs, nc, floor, float(frame.shape[1]), float(frame.shape[0]))

        cands: list[pp.Candidate] = []
        for b, s, cid in zip(boxes, scores, cls_ids):
            raw = self.class_names[cid] if cid < len(self.class_names) else str(cid)
            canon = (self.canonical[cid] if cid < len(self.canonical) else None)
            cands.append(pp.Candidate(
                class_id=canon or tax.UNKNOWN_COMPONENT_ID,
                score=float(s),
                box=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                source=self.backend_id, raw_label=raw,
                extra=({} if canon else
                       {"note": f"model class '{raw}' is not in the taxonomy"}),
            ))
        return cands


# --------------------------------------------------------------------------
# Ultralytics backend (.pt checkpoints)
# --------------------------------------------------------------------------

@register
class UltralyticsIndustrialRecognizer(IndustrialRecognizer):
    """Trained detector loaded directly by Ultralytics (YOLOv8/v11, RT-DETR)."""

    backend_id = "industrial_ultralytics"
    display_name = "Industrial component detector — Ultralytics (.pt weights)"
    default_subdir = DEFAULT_SUBDIR

    def load(self) -> None:
        models_dir = self.params.get("models_dir", "models")
        weights = self.params.get("weights")
        if not (weights and os.path.exists(str(weights))):
            found = sorted(glob.glob(os.path.join(models_dir,
                                                  self.default_subdir, "*.pt")))
            weights = found[0] if found else None
        if not weights:
            self._ready = False
            self._status = "no_weights"
            self._reason = "weights_missing"
            self._error = (f"No .pt checkpoint in models/{self.default_subdir}/. "
                           f"Train one with training/electrical.")
            raise RuntimeError(self._error)
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:
            self._ready = False
            self._status = "error"
            self._reason = "ultralytics_missing"
            self._error = (f"ultralytics is not installed ({exc}). "
                           f"pip install ultralytics")
            raise RuntimeError(self._error)
        try:
            self._model = YOLO(str(weights))
            names = self._model.names
            if isinstance(names, dict):
                self.class_names = [names[k] for k in sorted(names)]
            else:
                self.class_names = list(names)
            self.class_source = "checkpoint"
            self.canonical, self.unmapped_classes = resolve_names(self.class_names)
            self._weights_path = str(weights)
            self._ready = True
            self._status = "ready"
            self._error = None
            self._reason = None
        except Exception as exc:
            self._ready = False
            self._status = "error"
            self._reason = "init_failed"
            self._error = f"failed to load checkpoint: {exc}"
            raise RuntimeError(self._error)

    #: Ultralytics batches a list source in one forward pass, so this is real.
    supports_true_batching = True

    def _predict(self, source):
        floor = float(self.params.get("decode_floor", 0.10))
        return self._model.predict(
            source=source, conf=floor, iou=0.7, verbose=False,
            imgsz=int(self.params.get("imgsz", 640)),
            device=("cuda" if str(self.params.get("device", "cpu")).lower()
                    in ("gpu", "cuda") else "cpu"),
        )

    def _decode(self, result) -> list[pp.Candidate]:
        """One Ultralytics Result → candidates. Shared by single and batched paths
        so the two can never decode differently."""
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        cands: list[pp.Candidate] = []
        for b, s, cid in zip(xyxy, conf, cls):
            raw = (self.class_names[cid] if cid < len(self.class_names)
                   else str(cid))
            canon = (self.canonical[cid] if cid < len(self.canonical) else None)
            cands.append(pp.Candidate(
                class_id=canon or tax.UNKNOWN_COMPONENT_ID,
                score=float(s), box=tuple(float(v) for v in b),
                source=self.backend_id, raw_label=raw))
        return cands

    def raw_candidates(self, frame: np.ndarray) -> list[pp.Candidate]:
        cands: list[pp.Candidate] = []
        for r in self._predict(frame) or []:
            cands.extend(self._decode(r))
        return cands

    def raw_candidates_batch(self, frames: Sequence[np.ndarray]
                             ) -> list[list[pp.Candidate]]:
        """One forward pass over the whole chunk.

        Ultralytics returns one Result per input, in order. That ordering is the
        entire correctness requirement here — a mismatch would attribute one
        panel's detections to another image — so the count is checked and a
        mismatch falls back to per-frame inference rather than guessing the
        pairing.
        """
        results = self._predict(list(frames)) or []
        if len(results) != len(frames):
            _log.warning(
                "ultralytics returned %d result(s) for %d frame(s); falling back "
                "to per-frame inference so detections are not misattributed",
                len(results), len(frames))
            return [self.raw_candidates(f) for f in frames]
        return [self._decode(r) for r in results]


# --------------------------------------------------------------------------
# Open-vocabulary backends
# --------------------------------------------------------------------------

class _OpenVocabBase(IndustrialRecognizer):
    """Shared plumbing for text-conditioned detectors.

    These recognise components from a natural-language description instead of a
    trained class head, which is what makes recognition possible *before* a
    bespoke dataset exists. The prompts come from the taxonomy, so adding a
    class to the taxonomy immediately extends zero-shot coverage.
    """

    requires_weights = False        # weights are pulled by transformers
    hf_model_id = ""
    #: prompts are batched to keep peak memory bounded
    prompt_batch = 24

    def prompt_pairs(self) -> tuple[tuple[str, str], ...]:
        only = self.params.get("classes")
        pairs = tax.flat_prompts()
        if only:
            wanted = {str(c) for c in only}
            pairs = tuple(p for p in pairs if p[1] in wanted)
        return pairs

    def _fail(self, reason: str, message: str) -> None:
        self._ready = False
        self._status = "error"
        self._reason = reason
        self._error = message
        raise RuntimeError(message)

    def _load_transformers(self):
        try:
            import torch  # type: ignore  # noqa: F401
            import transformers  # type: ignore
            return transformers
        except Exception as exc:
            self._fail("transformers_missing",
                       f"Open-vocabulary detection needs torch + transformers "
                       f"({exc}). pip install -r requirements-openvocab.txt")


@register
class Owlv2Recognizer(_OpenVocabBase):
    """OWLv2 zero-shot detection driven by taxonomy prompts."""

    backend_id = "openvocab_owlv2"
    display_name = "Open-vocabulary OWLv2 (zero-shot, no dataset needed)"
    hf_model_id = "google/owlv2-base-patch16-ensemble"

    def load(self) -> None:
        transformers = self._load_transformers()
        model_id = str(self.params.get("model_id") or self.hf_model_id)
        try:
            self._processor = transformers.AutoProcessor.from_pretrained(model_id)
            self._model = transformers.Owlv2ForObjectDetection.from_pretrained(model_id)
            self._model.eval()
        except Exception as exc:
            self._fail("weights_unavailable",
                       f"could not obtain OWLv2 weights '{model_id}': {exc}. "
                       f"Pre-download the model on a host with access to the "
                       f"model hub, or set model_id to a local directory.")
        self.class_names = [cid for _, cid in self.prompt_pairs()]
        self.class_source = "taxonomy_prompts"
        self.unmapped_classes = []
        self._ready = True
        self._status = "ready"
        self._error = None
        self._reason = None

    def raw_candidates(self, frame: np.ndarray) -> list[pp.Candidate]:
        import torch  # type: ignore

        pairs = self.prompt_pairs()
        if not pairs:
            return []
        floor = float(self.params.get("decode_floor", 0.08))
        rgb = frame[:, :, ::-1]
        h, w = frame.shape[:2]
        out: list[pp.Candidate] = []
        for start in range(0, len(pairs), self.prompt_batch):
            chunk = pairs[start:start + self.prompt_batch]
            queries = [p for p, _ in chunk]
            inputs = self._processor(text=[queries], images=rgb,
                                     return_tensors="pt")
            with torch.no_grad():
                res = self._model(**inputs)
            post = self._processor.post_process_grounded_object_detection(
                outputs=res, target_sizes=torch.tensor([[h, w]]),
                threshold=floor)
            for item in post:
                for box, score, lab in zip(item["boxes"], item["scores"],
                                           item["labels"]):
                    li = int(lab)
                    if li >= len(chunk):
                        continue
                    prompt, cid = chunk[li]
                    b = [float(v) for v in box.tolist()]
                    out.append(pp.Candidate(
                        class_id=cid, score=float(score),
                        box=(b[0], b[1], b[2], b[3]),
                        source=self.backend_id, raw_label=prompt,
                        extra={"prompt": prompt}))
        return out


@register
class GroundingDinoRecognizer(_OpenVocabBase):
    """Grounding DINO zero-shot detection driven by taxonomy prompts."""

    backend_id = "openvocab_grounding_dino"
    display_name = "Open-vocabulary Grounding DINO (zero-shot)"
    hf_model_id = "IDEA-Research/grounding-dino-base"
    prompt_batch = 12

    def load(self) -> None:
        transformers = self._load_transformers()
        model_id = str(self.params.get("model_id") or self.hf_model_id)
        try:
            self._processor = transformers.AutoProcessor.from_pretrained(model_id)
            self._model = transformers.AutoModelForZeroShotObjectDetection.\
                from_pretrained(model_id)
            self._model.eval()
        except Exception as exc:
            self._fail("weights_unavailable",
                       f"could not obtain Grounding DINO weights '{model_id}': "
                       f"{exc}. Pre-download or point model_id at a local path.")
        self.class_names = [cid for _, cid in self.prompt_pairs()]
        self.class_source = "taxonomy_prompts"
        self.unmapped_classes = []
        self._ready = True
        self._status = "ready"
        self._error = None
        self._reason = None

    def raw_candidates(self, frame: np.ndarray) -> list[pp.Candidate]:
        import torch  # type: ignore

        pairs = self.prompt_pairs()
        if not pairs:
            return []
        floor = float(self.params.get("decode_floor", 0.20))
        text_floor = float(self.params.get("text_threshold", 0.20))
        rgb = frame[:, :, ::-1]
        h, w = frame.shape[:2]
        out: list[pp.Candidate] = []
        for start in range(0, len(pairs), self.prompt_batch):
            chunk = pairs[start:start + self.prompt_batch]
            # Grounding DINO expects a single '.'-separated caption.
            caption = ". ".join(p for p, _ in chunk) + "."
            inputs = self._processor(images=rgb, text=caption,
                                     return_tensors="pt")
            with torch.no_grad():
                res = self._model(**inputs)
            post = self._processor.post_process_grounded_object_detection(
                res, inputs.input_ids, threshold=floor,
                text_threshold=text_floor, target_sizes=[(h, w)])
            for item in post:
                for box, score, phrase in zip(item["boxes"], item["scores"],
                                              item.get("labels", [])):
                    cid = _match_phrase(str(phrase), chunk)
                    if cid is None:
                        continue
                    b = [float(v) for v in box.tolist()]
                    out.append(pp.Candidate(
                        class_id=cid, score=float(score),
                        box=(b[0], b[1], b[2], b[3]),
                        source=self.backend_id, raw_label=str(phrase),
                        extra={"phrase": str(phrase)}))
        return out


def _match_phrase(phrase: str, chunk: Sequence[tuple[str, str]]) -> Optional[str]:
    """Map a Grounding DINO phrase back to a taxonomy class.

    The model returns the matched sub-phrase, not the whole prompt, so we score
    by token overlap and fall back to the taxonomy resolver.
    """
    ph = phrase.lower().strip()
    if not ph:
        return None
    best, best_score = None, 0.0
    ph_tokens = set(ph.split())
    for prompt, cid in chunk:
        p_tokens = set(prompt.lower().split())
        if not p_tokens:
            continue
        overlap = len(ph_tokens & p_tokens) / len(ph_tokens | p_tokens)
        if overlap > best_score:
            best, best_score = cid, overlap
    if best is not None and best_score >= 0.15:
        return best
    return tax.resolve(ph)


@register
class Florence2Recognizer(_OpenVocabBase):
    """Florence-2 open-vocabulary detection (``<OPEN_VOCABULARY_DETECTION>``)."""

    backend_id = "openvocab_florence2"
    display_name = "Open-vocabulary Florence-2 (zero-shot)"
    hf_model_id = "microsoft/Florence-2-base"
    prompt_batch = 1

    def load(self) -> None:
        transformers = self._load_transformers()
        model_id = str(self.params.get("model_id") or self.hf_model_id)
        try:
            self._processor = transformers.AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True)
            self._model = transformers.AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True)
            self._model.eval()
        except Exception as exc:
            self._fail("weights_unavailable",
                       f"could not obtain Florence-2 weights '{model_id}': "
                       f"{exc}.")
        self.class_names = [cid for _, cid in self.prompt_pairs()]
        self.class_source = "taxonomy_prompts"
        self.unmapped_classes = []
        self._ready = True
        self._status = "ready"
        self._error = None
        self._reason = None

    def raw_candidates(self, frame: np.ndarray) -> list[pp.Candidate]:
        import torch  # type: ignore
        from PIL import Image  # type: ignore

        pairs = self.prompt_pairs()
        if not pairs:
            return []
        pil = Image.fromarray(frame[:, :, ::-1])
        task = "<OPEN_VOCABULARY_DETECTION>"
        # Florence-2 accepts one phrase per call; query one prompt per class to
        # keep the number of forward passes proportional to the taxonomy, not to
        # the (much larger) prompt list.
        seen: set[str] = set()
        out: list[pp.Candidate] = []
        for prompt, cid in pairs:
            if cid in seen:
                continue
            seen.add(cid)
            inputs = self._processor(text=task + prompt, images=pil,
                                     return_tensors="pt")
            with torch.no_grad():
                ids = self._model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=512, num_beams=3, do_sample=False)
            text = self._processor.batch_decode(ids, skip_special_tokens=False)[0]
            parsed = self._processor.post_process_generation(
                text, task=task, image_size=(pil.width, pil.height))
            payload = parsed.get(task, {}) or {}
            for box in payload.get("bboxes", []) or []:
                b = [float(v) for v in box]
                out.append(pp.Candidate(
                    class_id=cid,
                    # Florence-2 returns no calibrated score. Reporting a fixed
                    # nominal value would be dishonest, so it is marked as
                    # uncalibrated and enters the gate at the unknown floor,
                    # meaning these become "Unknown Industrial Component"
                    # unless corroborated by another backend in an ensemble.
                    score=float(self.params.get("nominal_score", 0.30)),
                    box=(b[0], b[1], b[2], b[3]),
                    source=self.backend_id, raw_label=prompt,
                    extra={"score_uncalibrated": True, "prompt": prompt}))
        return out


# --------------------------------------------------------------------------
# Ensemble
# --------------------------------------------------------------------------

@register
class EnsembleRecognizer(IndustrialRecognizer):
    """Fuse several recognisers, weighting each backend's scores.

    Useful in the realistic intermediate state: a small trained model that is
    accurate on the handful of classes it has seen, plus a zero-shot model that
    covers the long tail. Agreement between two independent backends on the same
    box is strong evidence, so overlapping candidates are score-boosted before
    the gate runs.
    """

    backend_id = "industrial_ensemble"
    display_name = "Industrial ensemble (trained + zero-shot fusion)"
    requires_weights = False

    def load(self) -> None:
        from ..ai import registry as reg

        members = self.params.get("members") or ["industrial_onnx",
                                                 "openvocab_owlv2"]
        weights = self.params.get("member_weights") or {}
        self._members: list[tuple[IndustrialRecognizer, float]] = []
        errors: list[str] = []
        base_params = {k: v for k, v in self.params.items()
                       if k not in ("members", "member_weights")}
        for bid in members:
            try:
                cls = reg.get("components", str(bid))
                inst = cls(**base_params)
                inst.load()
                self._members.append((inst, float(weights.get(bid, 1.0))))
            except Exception as exc:
                errors.append(f"{bid}: {exc}")
        if not self._members:
            self._ready = False
            self._status = "error"
            self._reason = "no_members_loaded"
            self._error = "no ensemble member loaded — " + "; ".join(errors)
            raise RuntimeError(self._error)
        self._ready = True
        self._status = "ready"
        self._reason = None
        self._error = ("; ".join(errors) or None)
        self.class_names = list(tax.CLASS_ORDER)
        self.class_source = "ensemble"
        self.unmapped_classes = []

    def raw_candidates(self, frame: np.ndarray) -> list[pp.Candidate]:
        pooled: list[pp.Candidate] = []
        for member, weight in self._members:
            try:
                cands = member.raw_candidates(frame)
            except Exception as exc:
                _log.warning("ensemble member %s failed: %s",
                             member.backend_id, exc)
                continue
            for c in cands:
                pooled.append(pp.Candidate(
                    class_id=c.class_id,
                    score=float(min(1.0, c.score * weight)),
                    box=c.box, source=c.source, raw_label=c.raw_label,
                    extra=dict(c.extra)))
        return fuse(pooled, agreement_iou=float(
            self.params.get("agreement_iou", 0.55)))


def fuse(cands: Sequence[pp.Candidate], agreement_iou: float = 0.55
         ) -> list[pp.Candidate]:
    """Boost candidates that two different backends agree on.

    Agreement is only counted across *distinct sources* — two boxes from the
    same model are a duplicate, not corroboration.
    """
    out: list[pp.Candidate] = []
    for i, c in enumerate(cands):
        agreeing = {
            other.source for j, other in enumerate(cands)
            if j != i and other.source != c.source
            and other.class_id == c.class_id
            and pp.iou(c.box, other.box) >= agreement_iou
        }
        if agreeing:
            boosted = min(0.99, c.score + (1.0 - c.score) * 0.35 * len(agreeing))
            extra = dict(c.extra)
            extra["corroborated_by"] = sorted(agreeing)
            out.append(pp.Candidate(c.class_id, boosted, c.box, c.source,
                                    c.raw_label, extra))
        else:
            out.append(c)
    return out


# --------------------------------------------------------------------------
# Honest disabled state
# --------------------------------------------------------------------------

@register
class NullIndustrialRecognizer(IndustrialRecognizer):
    backend_id = "industrial_disabled"
    display_name = "Disabled (no component recognition)"
    requires_weights = False

    def load(self) -> None:
        self._ready = True
        self._status = "ready"
        self._reason = None
        self._error = None
        self.class_names = []
        self.class_source = "none"
        self.unmapped_classes = []

    def raw_candidates(self, frame: np.ndarray) -> list[pp.Candidate]:
        return []


__all__ = [
    "letterbox", "to_blob", "load_class_map", "resolve_names", "decode_yolo",
    "decode_rtdetr", "IndustrialRecognizer", "OnnxIndustrialRecognizer",
    "UltralyticsIndustrialRecognizer", "Owlv2Recognizer",
    "GroundingDinoRecognizer", "Florence2Recognizer", "EnsembleRecognizer",
    "NullIndustrialRecognizer", "fuse",
]
