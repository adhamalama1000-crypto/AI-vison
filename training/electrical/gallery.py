"""False-positive and false-negative galleries.

A confusion matrix says *that* the detector confuses a contactor for a relay. It
cannot say whether the cause is a genuinely ambiguous device, a mislabelled
ground-truth box, a box drawn around two adjacent devices, or a class the model
has simply never seen enough of. That question is answered by looking at the
pixels, and looking at the pixels needs the failing crops pulled out and ordered
by how much they matter.

Two galleries, ordered on different principles because the failures differ:

* **False positives** are ranked by descending confidence. A confident mistake is
  the damaging kind — it is the one that reaches an inspection report — while a
  0.41 false positive just above the threshold is noise the operator ignores.
  They are also grouped by the cause :func:`~rtsp_backend.electrical.metrics.match`
  assigns: ``class_confusion`` (landed on a real device, wrong name),
  ``localisation`` (right area, box too loose to count) and
  ``spurious_detection`` (nothing there at all). Those three want different
  fixes — more per-class data, better box quality, and more negatives
  respectively — so mixing them in one pile hides the remedy.
* **False negatives** are ranked by descending box area. A missed device filling
  a tenth of the frame is a different failure from a missed 12-pixel terminal
  block, and the large ones are both more embarrassing and more diagnostic.

Each gallery writes individual annotated crops plus a contact sheet, because the
crops are what you inspect and the contact sheet is what you put in a report.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional, Sequence

from rtsp_backend.electrical import metrics as em
from rtsp_backend.electrical import taxonomy as tax

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

#: BGR colours. Red for a false positive, amber for a miss — the convention used
#: by the annotated frames the platform already serves.
FP_COLOUR = (60, 60, 220)
FN_COLOUR = (40, 170, 240)


def find_image(image_dir: str, image_id: str) -> Optional[str]:
    """Locate an image by stem, whatever extension it was written with."""
    for ext in IMAGE_EXTS:
        p = os.path.join(image_dir, image_id + ext)
        if os.path.exists(p):
            return p
    return None


def _crop(img, box: Sequence[float], pad_frac: float = 0.55,
          min_side: int = 64):
    """Crop around a box with context, clipped to the image.

    Context matters for judging a detection: a bare 30x80 crop of a moulded case
    is unidentifiable even to a person, while the same crop with the neighbouring
    devices and the rail visible usually is not.
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    pad_x, pad_y = bw * pad_frac, bh * pad_frac
    # Widen a very small box so the crop is inspectable rather than a few pixels.
    if bw + 2 * pad_x < min_side:
        pad_x = (min_side - bw) / 2.0
    if bh + 2 * pad_y < min_side:
        pad_y = (min_side - bh) / 2.0
    cx1 = max(0, int(round(x1 - pad_x)))
    cy1 = max(0, int(round(y1 - pad_y)))
    cx2 = min(w, int(round(x2 + pad_x)))
    cy2 = min(h, int(round(y2 + pad_y)))
    if cx2 <= cx1 or cy2 <= cy1:
        return None, (0, 0)
    return img[cy1:cy2, cx1:cx2].copy(), (cx1, cy1)


def _annotate(crop, box: Sequence[float], origin, colour, label: str):
    """Draw the box (in crop coordinates) and a caption above it."""
    import cv2

    ox, oy = origin
    x1 = int(round(float(box[0]) - ox))
    y1 = int(round(float(box[1]) - oy))
    x2 = int(round(float(box[2]) - ox))
    y2 = int(round(float(box[3]) - oy))
    cv2.rectangle(crop, (x1, y1), (x2, y2), colour, 2)
    if label:
        cv2.putText(crop, label, (max(2, x1), max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)
    return crop


def _contact_sheet(tiles: Sequence, captions: Sequence[str], out_path: str,
                   cols: int = 6, tile: int = 200, caption_h: int = 22) -> bool:
    """Grid the crops into one reviewable image."""
    import cv2
    import numpy as np

    if not tiles:
        return False
    cols = max(1, int(cols))
    rows = (len(tiles) + cols - 1) // cols
    cell_h = tile + caption_h
    sheet = np.full((rows * cell_h, cols * tile, 3), 32, np.uint8)

    for i, (crop, cap) in enumerate(zip(tiles, captions)):
        r, c = divmod(i, cols)
        if crop is None or crop.size == 0:
            continue
        ch, cw = crop.shape[:2]
        scale = min(tile / max(1, cw), tile / max(1, ch))
        rw, rh = max(1, int(cw * scale)), max(1, int(ch * scale))
        resized = cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_AREA)
        y0 = r * cell_h + (tile - rh) // 2
        x0 = c * tile + (tile - rw) // 2
        sheet[y0:y0 + rh, x0:x0 + rw] = resized
        cv2.putText(sheet, cap[:34], (c * tile + 3, r * cell_h + tile + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (235, 235, 235), 1,
                    cv2.LINE_AA)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, sheet)
    return True


def write_galleries(gts: Sequence[Mapping], preds: Sequence[Mapping],
                    image_dir: str, out_dir: str,
                    iou_thr: float = em.DEFAULT_IOU,
                    top: int = 48, cols: int = 6,
                    log=None) -> dict:
    """Render the false-positive and false-negative galleries.

    ``preds`` must be the **asserted** production detections — the same list the
    production metrics were computed from — so that what the gallery shows is
    what the reported numbers counted.
    """
    import cv2

    say = log or (lambda m: None)
    m = em.match(gts, preds, iou_thr)
    os.makedirs(out_dir, exist_ok=True)
    result: dict = {"out_dir": out_dir, "iou_threshold": iou_thr,
                    "false_positives": {}, "false_negatives": {}}

    # -- false positives: most confident first, grouped by cause -----------
    fps = sorted(m.false_positives, key=lambda f: -float(f.get("score", 0.0)))
    fp_dir = os.path.join(out_dir, "false_positives")
    tiles, caps, written = [], [], []
    by_cause: dict[str, int] = {}
    for rank, f in enumerate(fps[:top], 1):
        by_cause[f["cause"]] = by_cause.get(f["cause"], 0) + 1
        path = find_image(image_dir, str(f["image_id"]))
        if not path:
            continue
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        crop, origin = _crop(img, f["box"])
        if crop is None:
            continue
        name = tax.display_name(str(f["class_id"]))
        label = f"{f['class_id']} {float(f['score']):.2f}"
        _annotate(crop, f["box"], origin, FP_COLOUR, label)
        sub = os.path.join(fp_dir, str(f["cause"]))
        os.makedirs(sub, exist_ok=True)
        fn = f"{rank:03d}_{f['class_id']}_{float(f['score']):.2f}_{f['image_id']}.jpg"
        cv2.imwrite(os.path.join(sub, fn), crop)
        written.append(os.path.join(str(f["cause"]), fn))
        tiles.append(crop)
        caps.append(f"{f['class_id']} {float(f['score']):.2f} {f['cause'][:9]}")
    sheet = os.path.join(out_dir, "fp_gallery.png")
    result["false_positives"] = {
        "total": len(m.false_positives),
        "rendered": len(written),
        "by_cause_rendered": by_cause,
        "by_cause_all": _tally(m.false_positives, "cause"),
        "worst_classes": [
            {"class_id": c, "name": tax.display_name(c), "false_positives": n}
            for c, n in list(_tally(m.false_positives, "class_id").items())[:5]],
        "contact_sheet": sheet if _contact_sheet(tiles, caps, sheet, cols) else None,
        "files": written,
    }
    say(f"false-positive gallery: {len(written)} of {len(m.false_positives)} rendered")

    # -- false negatives: biggest misses first ------------------------------
    fns = sorted(m.false_negatives,
                 key=lambda t: -_area(t[2]))
    fn_dir = os.path.join(out_dir, "false_negatives")
    tiles, caps, written = [], [], []
    by_class: dict[str, int] = {}
    for rank, (image_id, cid, box) in enumerate(fns[:top], 1):
        by_class[cid] = by_class.get(cid, 0) + 1
        path = find_image(image_dir, str(image_id))
        if not path:
            continue
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        crop, origin = _crop(img, box)
        if crop is None:
            continue
        _annotate(crop, box, origin, FN_COLOUR, f"missed {cid}")
        sub = os.path.join(fn_dir, str(cid))
        os.makedirs(sub, exist_ok=True)
        fn = f"{rank:03d}_{cid}_{image_id}.jpg"
        cv2.imwrite(os.path.join(sub, fn), crop)
        written.append(os.path.join(str(cid), fn))
        tiles.append(crop)
        caps.append(f"missed {cid}")
    sheet = os.path.join(out_dir, "fn_gallery.png")
    result["false_negatives"] = {
        "total": len(m.false_negatives),
        "rendered": len(written),
        "by_class_rendered": by_class,
        "by_class_all": _tally_fn(m.false_negatives),
        "contact_sheet": sheet if _contact_sheet(tiles, caps, sheet, cols) else None,
        "files": written,
    }
    say(f"false-negative gallery: {len(written)} of {len(m.false_negatives)} rendered")
    return result


def _area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = (float(v) for v in box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _tally(items: Sequence[Mapping], key: str) -> dict:
    out: dict[str, int] = {}
    for i in items:
        out[str(i.get(key))] = out.get(str(i.get(key)), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _tally_fn(fns: Sequence) -> dict:
    out: dict[str, int] = {}
    for _, cid, _ in fns:
        out[str(cid)] = out.get(str(cid), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


__all__ = ["write_galleries", "find_image"]
