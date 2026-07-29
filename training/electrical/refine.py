"""
Box refinement with SAM2 — turning loose open-vocabulary boxes into tight ones.

Why this is not decoration. Open-vocabulary detectors (OWLv2, Grounding DINO,
Florence-2) are trained on natural images with caption supervision, so their boxes
are *approximately* right: for a contactor on a DIN rail they routinely include a
strip of rail, the neighbouring module, and the wire loom leaving the terminals.
A labeller then spends as long dragging box edges as they would have spent drawing
the box from scratch, which destroys the 3–5× speed-up that justifies
auto-annotation at all.

SAM2 fixes exactly that and nothing else. It is not a detector — it has no idea
what a contactor is and never will. It is a *promptable segmenter*: given a rough
box, it returns the mask of the dominant object inside it. The tight bounding box of
that mask is the box the labeller wanted. Detection stays the open-vocabulary
model's job; localisation becomes SAM2's.

Model preference, best first:

``sam2.1_b.pt`` / ``sam2_b.pt``
    SAM2 via Ultralytics. Best masks, and no extra dependency beyond the
    ``ultralytics`` this project already uses for training.
``sam_b.pt`` / ``mobile_sam.pt``
    SAM 1 / MobileSAM, also via Ultralytics. Noticeably faster, slightly looser.
``facebook/sam2-hiera-*``
    The reference SAM2 implementation, if ``sam2`` is installed directly.

Guards, because a refinement can be worse than the original
-----------------------------------------------------------
SAM2 fails in predictable ways on panel imagery: given a box around a modular
device it sometimes segments the *whole DIN rail row* (they are visually
continuous), sometimes just the toggle lever, and sometimes the shadow. So every
refinement is checked before being accepted:

* it must not grow the box beyond :data:`MAX_GROWTH` — catches "segmented the whole
  row";
* it must not shrink below :data:`MIN_AREA_RATIO` — catches "segmented only the
  lever";
* its centre must not drift more than :data:`MAX_CENTRE_DRIFT` of the box diagonal —
  catches "segmented the neighbour";
* it must stay a plausible aspect ratio for the class, using the geometric priors
  already in :mod:`rtsp_backend.electrical.taxonomy`.

A refinement that fails any guard is **rejected and the original box kept**, and the
reason is counted. The report gives the accept rate and the mean IoU shift, so the
question "is SAM2 actually helping here?" has a number rather than an opinion. If
the accept rate is low, that is worth knowing before you run it over 500 images.

With no SAM available this module is a no-op that says so. It never invents a mask.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from rtsp_backend.electrical import taxonomy as tax

#: Candidate checkpoints, best first. Ultralytics downloads these on first use.
SAM_CANDIDATES: tuple[str, ...] = (
    "sam2.1_b.pt", "sam2_b.pt", "sam2.1_t.pt", "sam2_t.pt",
    "sam_b.pt", "mobile_sam.pt",
)

#: A refined box may not exceed this multiple of the original area. Above it, SAM
#: has almost certainly latched onto the whole DIN-rail row rather than the device.
MAX_GROWTH = 1.6
#: ...nor fall below this fraction of it, which means it found only a sub-part
#: (the toggle lever, one terminal, a label).
MIN_AREA_RATIO = 0.35
#: Centre drift, as a fraction of the original box diagonal. Beyond this the mask
#: belongs to a different object — usually the adjacent module.
MAX_CENTRE_DRIFT = 0.35
#: Slack applied to the taxonomy's aspect-ratio prior, since a prior is a sanity
#: band rather than a specification.
ASPECT_SLACK = 1.5


@dataclass
class RefinedBox:
    class_id: str
    original: tuple[float, float, float, float]
    refined: tuple[float, float, float, float]
    accepted: bool
    reason: str = "ok"
    iou: float = 1.0

    @property
    def box(self) -> tuple[float, float, float, float]:
        """The box to actually use — refined when accepted, original otherwise."""
        return self.refined if self.accepted else self.original

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "original": [round(v, 1) for v in self.original],
            "refined": [round(v, 1) for v in self.refined],
            "accepted": self.accepted, "reason": self.reason,
            "iou": round(self.iou, 4),
        }


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


class SamRefiner:
    """Loads a SAM/SAM2 checkpoint and refines boxes with it.

    Construction never raises. Check :attr:`ready` and read :attr:`reason` — a
    missing SAM must degrade auto-annotation to "loose boxes, stated plainly",
    never to a crash mid-run over a 500-image batch.
    """

    def __init__(self, weights: Optional[str] = None,
                 device: str = "cpu",
                 candidates: Sequence[str] = SAM_CANDIDATES,
                 log: Optional[Callable[[str], None]] = None) -> None:
        self.device = device
        self._say = log or (lambda m: None)
        self.ready = False
        self.reason = "not loaded"
        self.backend: Optional[str] = None
        self.weights: Optional[str] = None
        self._model = None
        self._predictor = None
        self._load(weights, candidates)

    # -- loading ---------------------------------------------------------
    def _load(self, weights: Optional[str],
              candidates: Sequence[str]) -> None:
        if self._load_ultralytics(weights, candidates):
            return
        if self._load_native_sam2(weights):
            return
        self.reason = (
            "no SAM/SAM2 backend available. Install one to tighten "
            "auto-annotation boxes:\n"
            "  pip install ultralytics      # then SAM2 downloads on first use\n"
            "  pip install 'sam2 @ git+https://github.com/facebookresearch/sam2'\n"
            "Auto-annotation still works without it — the boxes are just the "
            "detector's own, which are looser and cost the labeller more time.")
        self._say(self.reason)

    def _load_ultralytics(self, weights: Optional[str],
                          candidates: Sequence[str]) -> bool:
        try:
            from ultralytics import SAM  # type: ignore
        except Exception as exc:
            self._say(f"ultralytics SAM unavailable: {exc}")
            return False
        tried: list[str] = []
        for cand in ([weights] if weights else []) + list(candidates):
            if not cand:
                continue
            tried.append(cand)
            try:
                # A local path is used as-is; a bare name is fetched by
                # Ultralytics into its weights cache.
                self._model = SAM(cand)
                self.ready = True
                self.backend = "ultralytics_sam"
                self.weights = cand
                self.reason = "ok"
                self._say(f"SAM backend: ultralytics {cand}")
                return True
            except Exception as exc:
                self._say(f"  {cand} unavailable: {exc}")
        self._say(f"no usable Ultralytics SAM checkpoint among {tried}")
        return False

    def _load_native_sam2(self, weights: Optional[str]) -> bool:
        try:
            import torch  # type: ignore  # noqa: F401
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
        except Exception as exc:
            self._say(f"native sam2 unavailable: {exc}")
            return False
        model_id = weights or os.environ.get(
            "SAM2_MODEL_ID", "facebook/sam2.1-hiera-base-plus")
        try:
            self._predictor = SAM2ImagePredictor.from_pretrained(model_id)
            self.ready = True
            self.backend = "sam2_native"
            self.weights = model_id
            self.reason = "ok"
            self._say(f"SAM backend: native sam2 {model_id}")
            return True
        except Exception as exc:
            self._say(f"could not load native sam2 '{model_id}': {exc}")
            return False

    # -- inference -------------------------------------------------------
    def _masks_for_boxes(self, image, boxes: Sequence[Sequence[float]]):
        """Return one mask per input box, or ``None`` when SAM produced none."""
        import numpy as np

        if self._model is not None:
            results = self._model(image, bboxes=[list(b[:4]) for b in boxes],
                                  device=self.device, verbose=False)
            out = []
            for res in results or []:
                masks = getattr(res, "masks", None)
                if masks is None or getattr(masks, "data", None) is None:
                    continue
                for m in masks.data:
                    arr = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
                    out.append(arr.astype(bool))
            return out or None

        if self._predictor is not None:
            import torch

            rgb = image[..., ::-1]
            with torch.inference_mode():
                self._predictor.set_image(rgb)
                masks, scores, _ = self._predictor.predict(
                    box=np.asarray([list(b[:4]) for b in boxes],
                                   dtype="float32"),
                    multimask_output=False)
            arr = np.asarray(masks)
            if arr.ndim == 4:          # (n, 1, H, W)
                arr = arr[:, 0]
            elif arr.ndim == 2:        # single box, (H, W)
                arr = arr[None]
            return [a.astype(bool) for a in arr]

        return None

    @staticmethod
    def _mask_box(mask) -> Optional[tuple[float, float, float, float]]:
        import numpy as np

        ys, xs = np.where(mask)
        if ys.size == 0:
            return None
        return (float(xs.min()), float(ys.min()),
                float(xs.max()) + 1.0, float(ys.max()) + 1.0)

    # -- guards ----------------------------------------------------------
    @staticmethod
    def _check(original: Sequence[float], refined: Sequence[float],
               class_id: str) -> tuple[bool, str]:
        ox1, oy1, ox2, oy2 = original[:4]
        rx1, ry1, rx2, ry2 = refined[:4]
        ow, oh = ox2 - ox1, oy2 - oy1
        rw, rh = rx2 - rx1, ry2 - ry1
        if rw <= 1.0 or rh <= 1.0:
            return False, "degenerate_mask"
        o_area, r_area = ow * oh, rw * rh
        if o_area <= 0:
            return False, "degenerate_original"

        ratio = r_area / o_area
        if ratio > MAX_GROWTH:
            # Usually the whole DIN-rail row: modular devices are visually
            # continuous, so SAM happily segments all of them as one object.
            return False, "grew_beyond_limit"
        if ratio < MIN_AREA_RATIO:
            return False, "collapsed_to_subpart"

        diag = (ow ** 2 + oh ** 2) ** 0.5
        drift = (((rx1 + rx2) / 2 - (ox1 + ox2) / 2) ** 2
                 + ((ry1 + ry2) / 2 - (oy1 + oy2) / 2) ** 2) ** 0.5
        if diag > 0 and drift / diag > MAX_CENTRE_DRIFT:
            return False, "centre_drifted"

        spec = tax.SPECS.get(class_id)
        if spec is not None and rh > 0:
            lo, hi = spec.aspect_ratio
            aspect = rw / rh
            if not (lo / ASPECT_SLACK <= aspect <= hi * ASPECT_SLACK):
                # The taxonomy's geometric prior already knows an MCB is not
                # 8:1 wide. Reuse it rather than inventing a second rule set.
                return False, "aspect_implausible"
        return True, "ok"

    def refine(self, image, boxes: Sequence[Sequence[float]],
               class_ids: Sequence[str]) -> list[RefinedBox]:
        """Refine every box for one image. Returns one entry per input box."""
        results = [RefinedBox(class_id=cid, original=tuple(b[:4]),
                              refined=tuple(b[:4]), accepted=False,
                              reason="sam_unavailable", iou=1.0)
                   for b, cid in zip(boxes, class_ids)]
        if not self.ready or not boxes:
            return results
        try:
            masks = self._masks_for_boxes(image, boxes)
        except Exception as exc:
            for r in results:
                r.reason = f"sam_error: {type(exc).__name__}: {exc}"
            return results
        if not masks:
            for r in results:
                r.reason = "no_mask_returned"
            return results
        if len(masks) != len(boxes):
            # SAM returned a different number of masks than boxes prompted. Rather
            # than pairing them by position and silently mis-assigning masks to
            # devices, refuse the whole image.
            for r in results:
                r.reason = (f"mask_count_mismatch ({len(masks)} masks for "
                            f"{len(boxes)} boxes)")
            return results

        h, w = image.shape[:2]
        for r, mask in zip(results, masks):
            mb = self._mask_box(mask)
            if mb is None:
                r.reason = "empty_mask"
                continue
            clipped = (max(0.0, mb[0]), max(0.0, mb[1]),
                       min(float(w), mb[2]), min(float(h), mb[3]))
            ok, reason = self._check(r.original, clipped, r.class_id)
            r.refined = clipped
            r.accepted = ok
            r.reason = reason
            r.iou = _iou(r.original, clipped)
        return results


def refine_summary(all_boxes: Sequence[RefinedBox]) -> dict:
    """Aggregate accept rate and IoU shift — is SAM actually helping?"""
    if not all_boxes:
        return {"boxes": 0, "accepted": 0, "accept_rate": None,
                "note": "no boxes were refined"}
    accepted = [b for b in all_boxes if b.accepted]
    reasons = Counter(b.reason for b in all_boxes if not b.accepted)
    mean_iou = (sum(b.iou for b in accepted) / len(accepted)
                if accepted else None)
    tightened = [b for b in accepted
                 if _area(b.refined) < _area(b.original)]
    return {
        "boxes": len(all_boxes),
        "accepted": len(accepted),
        "accept_rate": round(len(accepted) / len(all_boxes), 4),
        "rejected_reasons": dict(reasons.most_common()),
        "mean_iou_original_vs_refined": (round(mean_iou, 4)
                                         if mean_iou is not None else None),
        "boxes_tightened": len(tightened),
        "mean_area_reduction": (
            round(1.0 - sum(_area(b.refined) / max(_area(b.original), 1e-9)
                            for b in tightened) / len(tightened), 4)
            if tightened else None),
        "interpretation": _interpret(len(accepted), len(all_boxes), mean_iou),
    }


def _area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _interpret(accepted: int, total: int, mean_iou: Optional[float]) -> str:
    if not total:
        return "nothing to interpret"
    rate = accepted / total
    if rate < 0.25:
        return (f"Only {rate:.0%} of refinements passed the guards. SAM is not "
                f"helping on this imagery — check the rejected_reasons: mostly "
                f"'grew_beyond_limit' means it is segmenting whole DIN-rail rows "
                f"instead of individual devices, which happens when devices are "
                f"tightly packed and visually continuous. Consider running without "
                f"refinement rather than paying the inference cost for nothing.")
    if mean_iou is not None and mean_iou > 0.95:
        return (f"{rate:.0%} accepted but mean IoU with the original is "
                f"{mean_iou:.2f} — the boxes were already tight, so refinement is "
                f"changing almost nothing. Skip it and save the compute.")
    return (f"{rate:.0%} of boxes were refined"
            + (f", mean IoU {mean_iou:.2f} against the original" if mean_iou
               else "")
            + ". Refinement is doing real work; inspect a few crops to confirm "
              "the direction is correct before trusting it over a large batch.")


__all__ = ["SAM_CANDIDATES", "MAX_GROWTH", "MIN_AREA_RATIO",
           "MAX_CENTRE_DRIFT", "ASPECT_SLACK", "RefinedBox", "SamRefiner",
           "refine_summary"]
