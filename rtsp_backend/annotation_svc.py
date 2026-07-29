"""
Annotation review — the human-correction half of auto-annotation.

:mod:`training.electrical.autolabel` produces YOLO pre-labels and a manifest that
ranks images worst-first. Those pre-labels are **not** ground truth, and training on
un-reviewed output teaches the model its own mistakes. This module is what turns them
into ground truth: it serves a review batch, records a human's verdict per box, and
re-exports corrected YOLO labels.

Scope, deliberately narrow
--------------------------
This reviews *existing* boxes: accept, reject, or change the class. It does not draw
new boxes from scratch, and it should not — Roboflow, CVAT and Label Studio all do
that far better than a bespoke canvas would, they all import the YOLO tree that
``autolabel`` already writes, and the docs recommend them for exactly that. What they
do *not* do well is the specific loop this platform needs: work a
confidence-ordered queue of machine predictions, with the taxonomy's own class list
and the project's labelling rules to hand.

So the division is: **triage and reclassify here, redraw in a labelling tool.** An
image whose boxes are all wrong gets flagged `needs_redraw`, and the export lists
those separately so they can be sent to a real annotation tool rather than fudged.

State
-----
Verdicts live in the ``annotation_reviews`` table keyed by (batch, filename, box
index), so a review survives a restart and two people can work the same batch
without one overwriting the other's rows. The YOLO label files on disk are only
rewritten by :func:`export_batch`, which means an in-progress review never corrupts
the dataset it is reviewing.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from collections import Counter
from typing import Optional, Sequence

from .electrical import taxonomy as tax

#: Per-box verdicts a reviewer can record.
VERDICTS: tuple[str, ...] = ("accepted", "rejected", "reclassified")

#: Per-image states.
IMAGE_STATES: tuple[str, ...] = ("pending", "reviewed", "needs_redraw", "skipped")

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def ensure_schema(db) -> None:
    """Create the review tables. Idempotent, so it is safe on every startup."""
    db.execute(
        "CREATE TABLE IF NOT EXISTS annotation_batches("
        " name TEXT PRIMARY KEY,"
        " root TEXT NOT NULL,"
        " split TEXT NOT NULL DEFAULT 'train',"
        " created_at REAL NOT NULL,"
        " manifest TEXT)")
    db.execute(
        "CREATE TABLE IF NOT EXISTS annotation_reviews("
        " batch TEXT NOT NULL,"
        " filename TEXT NOT NULL,"
        " box_index INTEGER NOT NULL,"
        " verdict TEXT NOT NULL,"
        " class_id TEXT,"
        " reviewer TEXT,"
        " updated_at REAL NOT NULL,"
        " PRIMARY KEY (batch, filename, box_index))")
    db.execute(
        "CREATE TABLE IF NOT EXISTS annotation_image_state("
        " batch TEXT NOT NULL,"
        " filename TEXT NOT NULL,"
        " state TEXT NOT NULL,"
        " note TEXT,"
        " reviewer TEXT,"
        " updated_at REAL NOT NULL,"
        " PRIMARY KEY (batch, filename))")


# --------------------------------------------------------------------------
# batches
# --------------------------------------------------------------------------

def register_batch(db, name: str, root: str, split: str = "train") -> dict:
    """Register an autolabel output directory as a reviewable batch."""
    if not os.path.isdir(os.path.join(root, "images", split)):
        raise FileNotFoundError(
            f"{root} does not look like an autolabel output — expected "
            f"images/{split}/. Run 'python -m training.electrical.cli autolabel "
            f"--images <dir> --out {root}' first.")
    manifest = {}
    manifest_path = os.path.join(root, "autolabel_manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
        except (OSError, json.JSONDecodeError):
            manifest = {}
    db.execute(
        "INSERT OR REPLACE INTO annotation_batches(name,root,split,created_at,"
        "manifest) VALUES(?,?,?,?,?)",
        (name, os.path.abspath(root), split, time.time(),
         json.dumps(manifest)))
    return batch_detail(db, name)


def list_batches(db) -> list[dict]:
    rows = db.query("SELECT * FROM annotation_batches ORDER BY created_at DESC")
    out = []
    for r in rows:
        d = dict(r)
        d.pop("manifest", None)
        d.update(_progress(db, d["name"]))
        out.append(d)
    return out


def _progress(db, batch: str) -> dict:
    total = len(_image_files(db, batch))
    states = Counter(
        str(r["state"]) for r in
        db.query("SELECT state FROM annotation_image_state WHERE batch=?",
                 (batch,)))
    done = sum(v for k, v in states.items() if k != "pending")
    return {
        "images": total,
        "reviewed": int(states.get("reviewed", 0)),
        "needs_redraw": int(states.get("needs_redraw", 0)),
        "skipped": int(states.get("skipped", 0)),
        "remaining": max(0, total - done),
        "progress": round(done / total, 4) if total else 0.0,
    }


def _batch_row(db, batch: str) -> dict:
    rows = db.query("SELECT * FROM annotation_batches WHERE name=?", (batch,))
    if not rows:
        raise KeyError(f"no annotation batch named {batch!r}")
    return dict(rows[0])


def _image_files(db, batch: str) -> list[str]:
    row = _batch_row(db, batch)
    img_dir = os.path.join(row["root"], "images", row["split"])
    if not os.path.isdir(img_dir):
        return []
    return [f for f in sorted(os.listdir(img_dir))
            if f.lower().endswith(IMAGE_EXTS)]


def batch_detail(db, batch: str) -> dict:
    row = _batch_row(db, batch)
    manifest = {}
    if row.get("manifest"):
        try:
            manifest = json.loads(row["manifest"])
        except (TypeError, json.JSONDecodeError):
            manifest = {}
    return {
        "name": row["name"], "root": row["root"], "split": row["split"],
        "created_at": row["created_at"],
        **_progress(db, batch),
        "backend": manifest.get("backend"),
        "thresholds": manifest.get("thresholds"),
        "box_refinement": manifest.get("box_refinement"),
        "by_verdict": manifest.get("by_verdict"),
        "note": manifest.get("note"),
        "guidance": (
            "These are PRE-LABELS from a model, not ground truth. Accept, reject or "
            "reclassify each box. If an image's boxes are wrong in ways that need "
            "redrawing, mark it 'needs_redraw' — the export lists those separately "
            "so they can go to a real annotation tool rather than being fudged."),
    }


# --------------------------------------------------------------------------
# per-image review
# --------------------------------------------------------------------------

def _read_yolo(path: str) -> list[tuple[int, float, float, float, float]]:
    rows: list[tuple[int, float, float, float, float]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                rows.append((int(float(parts[0])),
                             *(float(v) for v in parts[1:5])))
            except ValueError:
                continue
    return rows


def review_queue(db, batch: str, state: Optional[str] = "pending",
                 limit: int = 100) -> dict:
    """The filenames to work, worst-first.

    Ordering comes from the autolabel manifest's review queue where one exists —
    uncertain images first, then lowest-confidence — because those are both the most
    likely to be wrong and the most informative to fix.
    """
    row = _batch_row(db, batch)
    manifest = json.loads(row["manifest"]) if row.get("manifest") else {}
    files = _image_files(db, batch)
    ranked = [f for f in (manifest.get("review_queue") or []) if f in files]
    ranked += [f for f in files if f not in set(ranked)]

    states = {str(r["filename"]): str(r["state"]) for r in db.query(
        "SELECT filename,state FROM annotation_image_state WHERE batch=?",
        (batch,))}
    out = []
    for fn in ranked:
        current = states.get(fn, "pending")
        if state and current != state:
            continue
        out.append({"filename": fn, "state": current})
        if len(out) >= limit:
            break
    return {"batch": batch, "filter": state, "count": len(out), "items": out,
            **_progress(db, batch)}


def image_detail(db, batch: str, filename: str) -> dict:
    """One image's boxes, with any recorded verdicts merged in."""
    row = _batch_row(db, batch)
    files = set(_image_files(db, batch))
    if filename not in files:
        raise KeyError(f"{filename!r} is not in batch {batch!r}")

    stem = os.path.splitext(filename)[0]
    label_path = os.path.join(row["root"], "labels", row["split"],
                              stem + ".txt")
    image_path = os.path.join(row["root"], "images", row["split"], filename)

    width = height = None
    try:
        import cv2
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is not None:
            height, width = img.shape[:2]
    except Exception:
        pass

    verdicts = {int(r["box_index"]): dict(r) for r in db.query(
        "SELECT * FROM annotation_reviews WHERE batch=? AND filename=?",
        (batch, filename))}

    classes = list(tax.CLASS_ORDER)
    boxes = []
    for i, (cls, cx, cy, bw, bh) in enumerate(_read_yolo(label_path)):
        original = classes[cls] if 0 <= cls < len(classes) else None
        v = verdicts.get(i) or {}
        boxes.append({
            "index": i,
            "class_id": v.get("class_id") or original,
            "original_class_id": original,
            "name": tax.display_name(v.get("class_id") or original or ""),
            "norm": {"cx": round(cx, 6), "cy": round(cy, 6),
                     "w": round(bw, 6), "h": round(bh, 6)},
            # Absolute pixels too, so a client does not have to know the convention.
            "bbox": ([round((cx - bw / 2) * width, 1),
                      round((cy - bh / 2) * height, 1),
                      round((cx + bw / 2) * width, 1),
                      round((cy + bh / 2) * height, 1)]
                     if width and height else None),
            "verdict": v.get("verdict"),
            "unclassified": False,
        })

    # Unclassified boxes live in a sidecar rather than the YOLO file, because
    # `unknown_industrial_component` has no class index by design. They are appended
    # after the classified boxes and share the same index space, so a reviewer
    # classifies them through the ordinary `reclassified` verdict and the export
    # picks them up. These are the highest-value boxes in the batch — they mark
    # exactly where the model is blind.
    sidecar = os.path.join(row["root"], "labels", row["split"],
                           stem + ".unclassified.json")
    unclassified_offset = len(boxes)
    if os.path.exists(sidecar):
        try:
            with open(sidecar, "r", encoding="utf-8") as fh:
                payload = json.load(fh) or {}
        except (OSError, json.JSONDecodeError):
            payload = {}
        for j, entry in enumerate(payload.get("boxes") or []):
            i = unclassified_offset + j
            v = verdicts.get(i) or {}
            norm = entry.get("norm") or {}
            boxes.append({
                "index": i,
                "class_id": v.get("class_id"),
                "original_class_id": None,
                "name": (tax.display_name(v["class_id"])
                         if v.get("class_id") else "Unclassified"),
                "norm": norm,
                "bbox": entry.get("bbox"),
                "confidence": entry.get("confidence"),
                "verdict": v.get("verdict"),
                "unclassified": True,
            })

    state_rows = db.query(
        "SELECT * FROM annotation_image_state WHERE batch=? AND filename=?",
        (batch, filename))
    state = dict(state_rows[0]) if state_rows else {"state": "pending"}

    return {
        "batch": batch, "filename": filename,
        "image_url": f"/api/annotations/{batch}/images/{filename}",
        "width": width, "height": height,
        "boxes": boxes,
        "state": state.get("state", "pending"),
        "note": state.get("note"),
        "unclassified_boxes": len([b for b in boxes if b["unclassified"]]),
        "labelling_rules": _rules_reminder(),
    }


def _rules_reminder() -> list[str]:
    """The rules a reviewer gets wrong most often, on screen where it matters."""
    return [
        "Contactor + overload relay = TWO boxes.",
        "Terminal blocks: one box per contiguous STRIP, never per pole.",
        "A 3-pole MCB with a common toggle is ONE box; three 1-pole MCBs are THREE.",
        "An illuminated push button is a PUSH BUTTON, not an indicator lamp.",
        "An emergency stop is never a push button.",
        f"Cannot identify it? Leave it as '{tax.UNKNOWN_COMPONENT_ID}'. Never guess "
        f"— a wrong confident label is worse than an honest unknown.",
    ]


def record_boxes(db, batch: str, filename: str, boxes: Sequence[dict],
                 reviewer: Optional[str] = None) -> dict:
    """Record per-box verdicts. Validates every class id against the taxonomy."""
    _batch_row(db, batch)
    now = time.time()
    valid = set(tax.CLASS_ORDER) | {tax.UNKNOWN_COMPONENT_ID}
    written = 0
    for b in boxes:
        verdict = str(b.get("verdict") or "").strip()
        if verdict not in VERDICTS:
            raise ValueError(
                f"verdict must be one of {VERDICTS}, got {verdict!r}")
        class_id = b.get("class_id")
        if verdict == "reclassified":
            if class_id not in valid:
                # Refusing an unknown class id here is the whole point: a typo would
                # otherwise become a label the trainer silently ignores.
                raise ValueError(
                    f"reclassified box needs a valid taxonomy class id, got "
                    f"{class_id!r}. See GET /api/panel/classes.")
        try:
            index = int(b.get("index"))
        except (TypeError, ValueError):
            raise ValueError(f"box index must be an integer, got {b.get('index')!r}")
        db.execute(
            "INSERT OR REPLACE INTO annotation_reviews(batch,filename,box_index,"
            "verdict,class_id,reviewer,updated_at) VALUES(?,?,?,?,?,?,?)",
            (batch, filename, index, verdict,
             class_id if verdict == "reclassified" else None, reviewer, now))
        written += 1
    return {"batch": batch, "filename": filename, "boxes_recorded": written}


def set_image_state(db, batch: str, filename: str, state: str,
                    note: Optional[str] = None,
                    reviewer: Optional[str] = None) -> dict:
    _batch_row(db, batch)
    if state not in IMAGE_STATES:
        raise ValueError(f"state must be one of {IMAGE_STATES}, got {state!r}")
    db.execute(
        "INSERT OR REPLACE INTO annotation_image_state(batch,filename,state,note,"
        "reviewer,updated_at) VALUES(?,?,?,?,?,?)",
        (batch, filename, state, note, reviewer, time.time()))
    return {"batch": batch, "filename": filename, "state": state,
            **_progress(db, batch)}


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------

def export_batch(db, batch: str, dst_root: str,
                 include_unreviewed: bool = False) -> dict:
    """Write corrected YOLO labels for the reviewed images.

    Rejected boxes are dropped, reclassified boxes get their new index, accepted
    boxes pass through. Images marked ``needs_redraw`` are **excluded and listed**,
    because their boxes are wrong in ways this interface cannot fix and shipping them
    would put known-bad labels into the training set.

    ``include_unreviewed=False`` by default: an un-reviewed image still holds raw
    model output, and the entire point of the review step is that such output is not
    ground truth.
    """
    row = _batch_row(db, batch)
    src_root, split = row["root"], row["split"]
    index_of = tax.class_index()

    states = {str(r["filename"]): str(r["state"]) for r in db.query(
        "SELECT filename,state FROM annotation_image_state WHERE batch=?",
        (batch,))}
    verdicts: dict[str, dict[int, dict]] = {}
    for r in db.query("SELECT * FROM annotation_reviews WHERE batch=?", (batch,)):
        verdicts.setdefault(str(r["filename"]), {})[int(r["box_index"])] = dict(r)

    os.makedirs(os.path.join(dst_root, "images", split), exist_ok=True)
    os.makedirs(os.path.join(dst_root, "labels", split), exist_ok=True)

    exported = 0
    skipped_redraw: list[str] = []
    skipped_unreviewed: list[str] = []
    kept_boxes = 0
    dropped_boxes = 0
    reclassified_boxes = 0
    unclassified_promoted = 0
    unclassified_unresolved = 0
    per_class: Counter = Counter()

    for fn in _image_files(db, batch):
        state = states.get(fn, "pending")
        if state == "needs_redraw":
            skipped_redraw.append(fn)
            continue
        if state in ("pending", "skipped") and not include_unreviewed:
            skipped_unreviewed.append(fn)
            continue

        stem = os.path.splitext(fn)[0]
        rows = _read_yolo(os.path.join(src_root, "labels", split, stem + ".txt"))
        box_verdicts = verdicts.get(fn, {})
        inv = list(tax.CLASS_ORDER)
        lines: list[str] = []
        for i, (cls, cx, cy, bw, bh) in enumerate(rows):
            v = box_verdicts.get(i)
            verdict = (v or {}).get("verdict")
            if verdict == "rejected":
                dropped_boxes += 1
                continue
            new_cls = cls
            if verdict == "reclassified" and (v or {}).get("class_id"):
                target = index_of.get(str(v["class_id"]))
                if target is None:
                    # Should be impossible — record_boxes validates — but a hand-
                    # edited database must not silently emit a bad index.
                    dropped_boxes += 1
                    continue
                new_cls = target
                reclassified_boxes += 1
            lines.append(f"{new_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            kept_boxes += 1
            if 0 <= new_cls < len(inv):
                per_class[inv[new_cls]] += 1

        # Sidecar boxes enter the labels only once a human has given them a real
        # class. An unclassified box with no verdict is simply not exported —
        # writing it would need a class index that does not exist.
        sidecar = os.path.join(src_root, "labels", split,
                               stem + ".unclassified.json")
        if os.path.exists(sidecar):
            try:
                with open(sidecar, "r", encoding="utf-8") as fh:
                    payload = json.load(fh) or {}
            except (OSError, json.JSONDecodeError):
                payload = {}
            entries = payload.get("boxes") or []
            for j, entry in enumerate(entries):
                v = box_verdicts.get(len(rows) + j)
                if not v or v.get("verdict") != "reclassified" \
                        or not v.get("class_id"):
                    unclassified_unresolved += 1
                    continue
                target = index_of.get(str(v["class_id"]))
                if target is None:
                    unclassified_unresolved += 1
                    continue
                norm = entry.get("norm") or {}
                try:
                    cx, cy = float(norm["cx"]), float(norm["cy"])
                    bw, bh = float(norm["w"]), float(norm["h"])
                except (KeyError, TypeError, ValueError):
                    unclassified_unresolved += 1
                    continue
                lines.append(f"{target} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                kept_boxes += 1
                unclassified_promoted += 1
                if 0 <= target < len(inv):
                    per_class[inv[target]] += 1

        shutil.copy2(os.path.join(src_root, "images", split, fn),
                     os.path.join(dst_root, "images", split, fn))
        with open(os.path.join(dst_root, "labels", split, stem + ".txt"), "w",
                  encoding="utf-8") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))
        exported += 1

    try:
        from training.electrical import datasets as tds
        tds.write_dataset_yaml(dst_root)
    except Exception:
        # The training package is optional at runtime; the YOLO tree is still valid
        # without dataset.yaml and 'cli split' regenerates it.
        pass

    warnings: list[str] = []
    if skipped_redraw:
        warnings.append(
            f"{len(skipped_redraw)} image(s) marked 'needs_redraw' were EXCLUDED. "
            f"Their boxes are wrong in ways this interface cannot fix — send them to "
            f"a labelling tool (Roboflow / CVAT / Label Studio all import the "
            f"original YOLO tree) rather than training on them.")
    if skipped_unreviewed:
        warnings.append(
            f"{len(skipped_unreviewed)} image(s) are still un-reviewed and were "
            f"excluded. Un-reviewed labels are raw model output, not ground truth; "
            f"training on them teaches the model its own mistakes.")
    if include_unreviewed:
        warnings.append(
            "include_unreviewed was set, so RAW MODEL OUTPUT is in this export. Do "
            "not treat the result as ground truth.")
    if unclassified_unresolved:
        warnings.append(
            f"{unclassified_unresolved} unclassified box(es) were left without a "
            f"class and are NOT in the export — there is no valid label index for "
            f"'unclassified'. These are the boxes that mark where the model is "
            f"blind, so they are the most valuable ones to work through: classify "
            f"them in the review interface and re-export.")

    return {
        "status": "exported",
        "batch": batch, "dst_root": dst_root,
        "images_exported": exported,
        "boxes_kept": kept_boxes,
        "boxes_dropped": dropped_boxes,
        "boxes_reclassified": reclassified_boxes,
        "unclassified_promoted": unclassified_promoted,
        "unclassified_still_unresolved": unclassified_unresolved,
        "instances_per_class": dict(per_class.most_common()),
        "skipped_needs_redraw": skipped_redraw,
        "skipped_unreviewed": len(skipped_unreviewed),
        "warnings": warnings,
        "next_step": (
            f"python -m training.electrical.cli split --src {dst_root} "
            f"--dst data/final   # then analyse / gap / train"),
    }


__all__ = [
    "VERDICTS", "IMAGE_STATES", "ensure_schema", "register_batch",
    "list_batches", "batch_detail", "review_queue", "image_detail",
    "record_boxes", "set_image_state", "export_batch",
]
