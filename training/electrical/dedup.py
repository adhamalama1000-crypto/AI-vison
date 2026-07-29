"""
Near-duplicate detection and removal for merged YOLO datasets.

Why this exists, concretely: :data:`training.electrical.datasets.SOURCES` contains
``rf_switchgear_varsha`` (723 images, 24 classes) and ``rf_switchgear_potholes``
(464 images, 17 classes) whose class lists and ~30-instances-per-class signatures
match almost exactly. They are very probably the same photographs republished. Merge
both and the *same image* can land in train and in val, at which point validation is
scoring memorisation and the reported mAP is fiction.

Roboflow exports make this worse: a generated version with augmentation baked in
contains several transformed copies of every original, all with different filenames
and hashes.

So this module answers two questions:

1. **Which images are duplicates of each other?** Exact duplicates by content hash,
   and near-duplicates by perceptual hash within a Hamming-distance threshold.
2. **Do any duplicates straddle a split?** That is the case that corrupts metrics,
   and it is reported separately and prominently from ordinary redundancy.

Perceptual hashing
------------------
Uses ``imagehash`` (already a runtime dependency for the image-comparison feature)
when it is importable. When it is not, :func:`dhash` and :func:`ahash` here are
self-contained NumPy implementations of the same algorithms — this module never
silently skips deduplication because an optional library is missing.

dHash (difference hash) is the primary signal: it compares adjacent pixel
gradients, so it is robust to brightness and contrast changes and to mild
compression, which is exactly the variation between a Roboflow original and its
augmented copies. aHash (average hash) is computed alongside as a cheap
disagreement check — two images that dHash calls identical but aHash calls
different are usually a genuine pair photographed under different exposure, which
is worth keeping rather than dropping.

What it deliberately does not do
--------------------------------
It does not delete anything from the source dataset. :func:`deduplicate` writes a
new dataset, and :func:`analyse_duplicates` is read-only, so a wrong threshold
costs a re-run rather than data. Nor does it merge labels between duplicates: if two
copies of one image carry different annotations, that is a labelling
inconsistency the human needs to see, and it is reported as a conflict instead of
being silently resolved by picking one.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from . import datasets as ds

#: Hamming distance at or below which two 64-bit perceptual hashes are called
#: near-duplicates. 0 is bit-identical; 5 catches augmented copies and re-encodes;
#: above ~12 it starts collapsing genuinely different panels of the same model,
#: which loses real training signal. 5 is conservative on purpose — the cost of a
#: missed duplicate is a slightly inflated metric, the cost of a false duplicate is
#: deleted data.
DEFAULT_THRESHOLD = 5

#: Hash side length. 8 gives the standard 64-bit hash.
HASH_SIZE = 8


# --------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------

def _to_gray_float(image) -> "object":
    import numpy as np

    if image.ndim == 3:
        # Rec. 601 luma, matching PIL's 'L' conversion so our fallback and
        # imagehash agree on what "grayscale" means.
        b, g, r = image[..., 0], image[..., 1], image[..., 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
    else:
        gray = image.astype("float64")
    return np.asarray(gray, dtype="float64")


def _resize(gray, width: int, height: int):
    import numpy as np

    try:
        import cv2
        return cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
    except ImportError:  # pragma: no cover - cv2 is a runtime dependency
        # Nearest-neighbour box sampling. Adequate for hashing; never reached in
        # a normal install.
        h, w = gray.shape[:2]
        ys = (np.arange(height) * h // height).clip(0, h - 1)
        xs = (np.arange(width) * w // width).clip(0, w - 1)
        return gray[np.ix_(ys, xs)]


def _bits_to_int(bits) -> int:
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bool(bit))
    return value


def dhash(image, hash_size: int = HASH_SIZE) -> int:
    """Difference hash: horizontal gradient direction of a downscaled image.

    Robust to brightness, contrast and compression changes; sensitive to structural
    change. This is the primary duplicate signal.
    """
    gray = _to_gray_float(image)
    small = _resize(gray, hash_size + 1, hash_size)
    return _bits_to_int(small[:, 1:] > small[:, :-1])


def ahash(image, hash_size: int = HASH_SIZE) -> int:
    """Average hash: which pixels are above the mean of a downscaled image."""
    gray = _to_gray_float(image)
    small = _resize(gray, hash_size, hash_size)
    return _bits_to_int(small > small.mean())


def hamming(a: int, b: int) -> int:
    return int(bin(a ^ b).count("1"))


def content_hash(path: str, chunk: int = 1 << 20) -> str:
    """SHA-256 of the file bytes — catches byte-identical re-uploads cheaply."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _imagehash_backend():
    """Return ``(dhash_fn, ahash_fn, name)`` preferring the imagehash library."""
    try:
        import imagehash  # type: ignore
        from PIL import Image  # type: ignore

        def _d(path_or_img):
            with Image.open(path_or_img) as im:
                return int(str(imagehash.dhash(im, hash_size=HASH_SIZE)), 16)

        def _a(path_or_img):
            with Image.open(path_or_img) as im:
                return int(str(imagehash.average_hash(im, hash_size=HASH_SIZE)), 16)

        return _d, _a, "imagehash"
    except Exception:
        return None, None, "builtin"


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------

@dataclass
class ImageRecord:
    split: str
    filename: str
    path: str
    dhash: int
    ahash: int
    sha256: str
    label_signature: tuple = ()

    @property
    def stem(self) -> str:
        return os.path.splitext(self.filename)[0]


@dataclass
class DuplicateGroup:
    """A set of images judged to be the same picture."""

    members: list[ImageRecord] = field(default_factory=list)
    kind: str = "near"                    # exact | near
    max_distance: int = 0

    @property
    def splits(self) -> set:
        return {m.split for m in self.members}

    @property
    def crosses_splits(self) -> bool:
        return len(self.splits) > 1

    @property
    def label_conflict(self) -> bool:
        """True when duplicates disagree about their annotations."""
        return len({m.label_signature for m in self.members}) > 1

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "max_distance": self.max_distance,
            "crosses_splits": self.crosses_splits,
            "splits": sorted(self.splits),
            "label_conflict": self.label_conflict,
            "members": [{"split": m.split, "filename": m.filename}
                        for m in self.members],
        }


# --------------------------------------------------------------------------
# scanning
# --------------------------------------------------------------------------

def _label_signature(root: str, split: str, stem: str) -> tuple:
    """A comparable summary of an image's annotations.

    Rounded to 2 decimals so trivial float noise between re-exports of the same
    labels does not read as a conflict, but a genuinely different box does.
    """
    path = os.path.join(root, "labels", split, stem + ".txt")
    if not os.path.exists(path):
        return ()
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                rows.append((int(float(parts[0])),
                             *(round(float(v), 2) for v in parts[1:5])))
            except ValueError:
                continue
    return tuple(sorted(rows))


def scan_dataset(root: str, splits: Sequence[str] = ("train", "val", "test"),
                 log: Optional[Callable[[str], None]] = None
                 ) -> tuple[list[ImageRecord], list[str]]:
    """Hash every image in a YOLO dataset. Returns ``(records, unreadable)``."""
    say = log or (lambda m: None)
    import cv2

    d_fn, a_fn, backend = _imagehash_backend()
    say(f"hashing with the '{backend}' backend")

    records: list[ImageRecord] = []
    unreadable: list[str] = []
    for split in splits:
        img_dir = os.path.join(root, "images", split)
        if not os.path.isdir(img_dir):
            continue
        files = [f for f in sorted(os.listdir(img_dir))
                 if f.lower().endswith(ds.IMAGE_EXTS)]
        for n, fn in enumerate(files, 1):
            path = os.path.join(img_dir, fn)
            try:
                if d_fn is not None:
                    dh, ah = d_fn(path), a_fn(path)
                else:
                    img = cv2.imread(path, cv2.IMREAD_COLOR)
                    if img is None:
                        unreadable.append(f"{split}/{fn}")
                        continue
                    dh, ah = dhash(img), ahash(img)
                sha = content_hash(path)
            except Exception as exc:
                say(f"  cannot hash {split}/{fn}: {exc}")
                unreadable.append(f"{split}/{fn}")
                continue
            records.append(ImageRecord(
                split=split, filename=fn, path=path, dhash=dh, ahash=ah,
                sha256=sha,
                label_signature=_label_signature(
                    root, split, os.path.splitext(fn)[0])))
            if n % 200 == 0:
                say(f"  {split}: {n}/{len(files)}")
    say(f"hashed {len(records)} image(s)")
    return records, unreadable


def find_duplicate_groups(records: Sequence[ImageRecord],
                          threshold: int = DEFAULT_THRESHOLD,
                          require_ahash_agreement: bool = True
                          ) -> list[DuplicateGroup]:
    """Group records into sets of duplicate images.

    Exact (byte-identical) duplicates are grouped first and cheaply. The remaining
    records are compared by perceptual hash using a BK-tree-free bucketed scan:
    records are indexed by the high bits of their dHash so that only plausible
    candidates are compared, which keeps this usable on tens of thousands of images
    instead of quadratic over all of them.

    ``require_ahash_agreement`` guards against dHash's one weakness — two different
    photographs of the same device model can share a gradient signature. Requiring
    aHash to also be close means a genuine pair shot under different exposure is
    kept rather than deleted.
    """
    adjacency: defaultdict[int, set] = defaultdict(set)
    exact_pairs: set[tuple] = set()

    # -- exact duplicates -------------------------------------------------
    # These become edges rather than finished groups. An earlier version emitted
    # exact groups immediately and excluded their members from the near-duplicate
    # pass, which silently defeated the whole point: if A(train) and B(train) are
    # byte-identical and C(val) is a re-compressed copy of A, excluding A and B
    # left C unlinked and the cross-split leak went unreported.
    by_sha: defaultdict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_sha[r.sha256].append(i)
    for idxs in by_sha.values():
        if len(idxs) < 2:
            continue
        first = idxs[0]
        for other in idxs[1:]:
            adjacency[first].add(other)
            adjacency[other].add(first)
            exact_pairs.add((min(first, other), max(first, other)))

    # -- near duplicates --------------------------------------------------
    # Bucket on 16-bit slices of the dHash. Two hashes within a small Hamming
    # distance usually agree on most bits of at least one slice, so this keeps
    # recall high without a quadratic sweep over the whole dataset.
    buckets: defaultdict[int, list[int]] = defaultdict(list)
    for i in range(len(records)):
        h = records[i].dhash
        for shift in (0, 16, 32, 48):
            buckets[(shift << 16) | ((h >> shift) & 0xFFFF)].append(i)

    seen_pairs: set[tuple] = set()
    for idxs in buckets.values():
        if len(idxs) < 2:
            continue
        for a_pos in range(len(idxs)):
            for b_pos in range(a_pos + 1, len(idxs)):
                i, j = idxs[a_pos], idxs[b_pos]
                key = (i, j) if i < j else (j, i)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                d = hamming(records[i].dhash, records[j].dhash)
                if d > threshold:
                    continue
                if require_ahash_agreement:
                    # Allow aHash a wider band: it is the noisier of the two.
                    if hamming(records[i].ahash, records[j].ahash) > threshold * 2:
                        continue
                adjacency[i].add(j)
                adjacency[j].add(i)

    # -- one connected-components pass over both relations ----------------
    groups: list[DuplicateGroup] = []
    visited: set[int] = set()
    for start in sorted(adjacency):
        if start in visited:
            continue
        stack, component = [start], []
        visited.add(start)
        while stack:
            node = stack.pop()
            component.append(node)
            for nb in adjacency[node]:
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        if len(component) < 2:
            continue
        component.sort()
        worst = 0
        all_exact = True
        for a_pos in range(len(component)):
            for b_pos in range(a_pos + 1, len(component)):
                i, j = component[a_pos], component[b_pos]
                worst = max(worst, hamming(records[i].dhash, records[j].dhash))
                if (i, j) not in exact_pairs:
                    all_exact = False
        groups.append(DuplicateGroup(
            members=[records[i] for i in component],
            kind="exact" if all_exact else "near",
            max_distance=worst))

    return groups


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

#: Splits ranked by which copy of a duplicate to keep. Keeping the training copy
#: and dropping the val/test copy is the only safe direction: it shrinks the
#: training set slightly but guarantees the evaluation set is unseen. Dropping the
#: train copy instead would leave the model evaluated on an image it trained on.
_KEEP_PRIORITY = {"train": 0, "val": 1, "test": 2}


def analyse_duplicates(root: str, threshold: int = DEFAULT_THRESHOLD,
                       splits: Sequence[str] = ("train", "val", "test"),
                       log: Optional[Callable[[str], None]] = None) -> dict:
    """Read-only duplicate report for a dataset."""
    say = log or (lambda m: None)
    records, unreadable = scan_dataset(root, splits, log=say)
    if not records:
        return {"status": "skipped",
                "reason": f"no images found under {root}/images/<split>",
                "unreadable": unreadable}

    groups = find_duplicate_groups(records, threshold)
    cross = [g for g in groups if g.crosses_splits]
    conflicts = [g for g in groups if g.label_conflict]
    redundant = sum(len(g.members) - 1 for g in groups)

    per_split_dupes: Counter = Counter()
    for g in groups:
        for m in g.members[1:]:
            per_split_dupes[m.split] += 1

    leaked_pairs = []
    for g in cross:
        for m in g.members:
            leaked_pairs.append({"split": m.split, "filename": m.filename,
                                 "kind": g.kind,
                                 "distance": g.max_distance})

    return {
        "status": "analysed",
        "root": root,
        "images": len(records),
        "unreadable": unreadable,
        "threshold": threshold,
        "duplicate_groups": len(groups),
        "exact_groups": len([g for g in groups if g.kind == "exact"]),
        "near_groups": len([g for g in groups if g.kind == "near"]),
        "redundant_images": redundant,
        "redundancy_fraction": round(redundant / len(records), 4),
        "cross_split_groups": len(cross),
        "cross_split_images": len(leaked_pairs),
        "label_conflict_groups": len(conflicts),
        "duplicates_per_split": dict(per_split_dupes),
        "groups": [g.to_dict() for g in groups[:200]],
        "groups_truncated": max(0, len(groups) - 200),
        "verdict": _verdict(len(records), redundant, cross, conflicts),
    }


def _verdict(total: int, redundant: int, cross: Sequence[DuplicateGroup],
             conflicts: Sequence[DuplicateGroup]) -> str:
    parts = []
    if not redundant:
        parts.append("No duplicates found.")
    else:
        parts.append(
            f"{redundant} of {total} image(s) ({redundant / total:.1%}) are "
            f"redundant copies.")
    if cross:
        parts.append(
            f"{len(cross)} duplicate group(s) STRADDLE SPLITS — the same "
            f"photograph appears in more than one of train/val/test, so every "
            f"validation metric from this dataset is inflated. Deduplicate before "
            f"believing any number it produces.")
    elif redundant:
        parts.append(
            "None of them straddle a split, so metrics are not corrupted; the "
            "redundancy only wastes training time and skews class balance.")
    if conflicts:
        parts.append(
            f"{len(conflicts)} group(s) contain duplicates whose LABELS DISAGREE. "
            f"That is a labelling inconsistency, not a duplication problem — "
            f"review those images by hand; deduplication would silently pick one "
            f"annotation over the other.")
    return " ".join(parts)


# --------------------------------------------------------------------------
# removal
# --------------------------------------------------------------------------

def deduplicate(src_root: str, dst_root: str,
                threshold: int = DEFAULT_THRESHOLD,
                splits: Sequence[str] = ("train", "val", "test"),
                keep: str = "train_first",
                symlink: bool = False,
                drop_label_conflicts: bool = False,
                log: Optional[Callable[[str], None]] = None) -> dict:
    """Write a deduplicated copy of a dataset.

    ``keep``:

    ``train_first``
        Default. From each duplicate group keep the copy already in the split
        closest to ``train``, and drop the rest. This is the only direction that is
        safe for evaluation: it can shrink the val/test set but guarantees nothing
        in them was trained on.
    ``first``
        Keep whichever copy was scanned first. Use only when you know the groups do
        not straddle splits.

    Groups whose labels disagree are **kept in full** by default and listed, because
    picking one annotation over another silently discards a human disagreement.
    Pass ``drop_label_conflicts=True`` to deduplicate them anyway.
    """
    say = log or (lambda m: None)
    if keep not in ("train_first", "first"):
        raise ValueError(f"keep must be 'train_first' or 'first', got {keep!r}")

    records, unreadable = scan_dataset(src_root, splits, log=say)
    if not records:
        raise ValueError(f"no images found under {src_root}/images/<split>")
    groups = find_duplicate_groups(records, threshold)

    drop: set[tuple] = set()
    kept_conflicts = 0
    for g in groups:
        if g.label_conflict and not drop_label_conflicts:
            kept_conflicts += 1
            continue
        members = list(g.members)
        if keep == "train_first":
            members.sort(key=lambda m: (_KEEP_PRIORITY.get(m.split, 9),
                                        m.filename))
        for m in members[1:]:
            drop.add((m.split, m.filename))

    for split in splits:
        os.makedirs(os.path.join(dst_root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(dst_root, "labels", split), exist_ok=True)

    copied: Counter = Counter()
    dropped: Counter = Counter()
    for r in records:
        if (r.split, r.filename) in drop:
            dropped[r.split] += 1
            continue
        d_img = os.path.join(dst_root, "images", r.split, r.filename)
        if symlink:
            if os.path.lexists(d_img):
                os.remove(d_img)
            os.symlink(os.path.abspath(r.path), d_img)
        else:
            shutil.copy2(r.path, d_img)
        s_lbl = os.path.join(src_root, "labels", r.split, r.stem + ".txt")
        d_lbl = os.path.join(dst_root, "labels", r.split, r.stem + ".txt")
        if os.path.exists(s_lbl):
            shutil.copy2(s_lbl, d_lbl)
        else:
            open(d_lbl, "w", encoding="utf-8").close()
        copied[r.split] += 1

    ds.write_dataset_yaml(dst_root)

    report = {
        "status": "deduplicated",
        "src_root": src_root, "dst_root": dst_root,
        "threshold": threshold, "keep": keep,
        "images_in": len(records),
        "images_out": int(sum(copied.values())),
        "images_dropped": int(sum(dropped.values())),
        "dropped_per_split": dict(dropped),
        "kept_per_split": dict(copied),
        "duplicate_groups": len(groups),
        "cross_split_groups_resolved": len(
            [g for g in groups
             if g.crosses_splits and not (g.label_conflict
                                          and not drop_label_conflicts)]),
        "label_conflict_groups_kept": kept_conflicts,
        "unreadable": unreadable,
        "warnings": [],
    }
    if kept_conflicts:
        report["warnings"].append(
            f"{kept_conflicts} duplicate group(s) had disagreeing labels and were "
            f"KEPT IN FULL rather than deduplicated. Review them by hand — one of "
            f"the two annotations is wrong. Re-run with drop_label_conflicts=True "
            f"to remove them anyway.")
    remaining_cross = [g for g in groups
                       if g.crosses_splits and g.label_conflict
                       and not drop_label_conflicts]
    if remaining_cross:
        report["warnings"].append(
            f"{len(remaining_cross)} of those conflicted group(s) also straddle "
            f"splits, so train/val leakage REMAINS in the output. Resolve the "
            f"label conflicts, then re-run.")

    say(f"kept {report['images_out']}, dropped {report['images_dropped']}")
    for w in report["warnings"]:
        say(f"warning: {w}")
    return report


__all__ = [
    "DEFAULT_THRESHOLD", "HASH_SIZE", "ImageRecord", "DuplicateGroup",
    "dhash", "ahash", "hamming", "content_hash", "scan_dataset",
    "find_duplicate_groups", "analyse_duplicates", "deduplicate",
]
