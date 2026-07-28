"""
Dataset splitting — 80 / 10 / 10, grouped so the metrics do not lie.

The brief asks for an 80/10/10 train/val/test split. The naive implementation
shuffles image files and slices the list, and it is the single most effective way
to produce an industrial detector that reports mAP@50 of 0.95 and then fails on
site. Panel photography is taken in bursts: the same cabinet is shot from five
angles, the same DIN rail row appears in a wide shot and two close-ups, and a
Roboflow export may already contain augmented copies of one original. Slice that
at random and near-duplicate images land in both train and val, so validation is
measuring memorisation.

:func:`split_dataset` therefore splits by **group**, not by image. The group key
is derived from the filename by :func:`group_key` — which strips the source
prefix, Roboflow's augmentation suffixes (``_jpg.rf.<hash>``), and any trailing
frame/shot counter — or supplied explicitly via a ``groups.json`` mapping when
the capture programme recorded which panel each photograph came from. Every image
sharing a group key goes to exactly one split.

Group assignment is then done greedily against per-class quotas rather than
purely at random, because with 20–50 instances of the rarer classes a random
split routinely leaves a class entirely absent from validation, and a class with
no validation instances silently drops out of mAP.

Everything is deterministic for a given seed.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from typing import Callable, Mapping, Optional, Sequence

from rtsp_backend.electrical import taxonomy as tax

from . import datasets as ds

DEFAULT_RATIOS = (0.8, 0.1, 0.1)

#: Roboflow appends ``.rf.<32-hex>`` (and often ``_jpg``/``_png``) to every
#: exported image, and augmented copies of one original share the stem before it.
_RF_SUFFIX = re.compile(r"[._](jpe?g|png|bmp|webp)?[._]*rf[._][0-9a-f]{8,}$",
                        re.IGNORECASE)
#: Trailing capture counters: ``panel12_003``, ``IMG-4821 (2)``, ``shot-7``.
#:
#: The separator (or the parentheses) is mandatory, and that is the whole point.
#: An earlier version allowed a bare digit run, which stripped the *identifier*
#: rather than the counter — ``panel12`` became ``panel``, so every panel in the
#: dataset collapsed into one group and the entire set landed in a single split.
#: Only strip a number that is visibly separated from the name.
_COUNTER_SUFFIX = re.compile(r"([ _-]+\(?\d{1,4}\)?|\(\d{1,4}\))$")
#: Roboflow augmentation verbs that appear in exported filenames.
_AUG_TOKENS = re.compile(
    r"[._-](aug|augmented|flip|fliph|flipv|rot(ate)?[-_]?\d*|bright(ness)?|"
    r"blur|noise|mosaic|crop|shear|exposure|hue|sat(uration)?)\d*$",
    re.IGNORECASE)


def group_key(filename: str, source_prefixes: Sequence[str] = ()) -> str:
    """Derive the capture-group key for an image filename.

    Two images that are different framings, crops or augmentations of the same
    physical panel should return the same key. This is a heuristic on filenames —
    it cannot see the pixels — so a capture programme that records the true panel
    id in a ``groups.json`` will always beat it. It is, however, dramatically
    better than treating every file as independent.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    for prefix in source_prefixes:
        if prefix and stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    stem = _RF_SUFFIX.sub("", stem)
    # Augmentation verbs and counters can stack: 'panel3_rot90_002'.
    for _ in range(3):
        new = _AUG_TOKENS.sub("", stem)
        new = _COUNTER_SUFFIX.sub("", new)
        if new == stem or not new:
            break
        stem = new
    return stem.strip("._- ").lower() or os.path.splitext(
        os.path.basename(filename))[0].lower()


def _iter_split_files(root: str, splits: Sequence[str]) -> list[tuple[str, str]]:
    """Return ``(split, image_filename)`` for every image in the dataset."""
    out: list[tuple[str, str]] = []
    for split in splits:
        img_dir = os.path.join(root, "images", split)
        if not os.path.isdir(img_dir):
            continue
        for fn in sorted(os.listdir(img_dir)):
            if fn.lower().endswith(ds.IMAGE_EXTS):
                out.append((split, fn))
    return out


def _classes_in(label_path: str, inv: Mapping[int, str]) -> Counter:
    counts: Counter = Counter()
    if not os.path.exists(label_path):
        return counts
    with open(label_path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cid = inv[int(float(parts[0]))]
            except (ValueError, KeyError):
                continue
            counts[cid] += 1
    return counts


def analyse_groups(root: str, splits: Sequence[str] = ("train", "val", "test"),
                   groups_json: Optional[str] = None,
                   source_prefixes: Sequence[str] = ()) -> dict:
    """Group the dataset and report how leaky the current split is.

    ``leaking_groups`` is the number of capture groups that currently appear in
    more than one split — i.e. the amount of train/val contamination in the
    dataset as it stands.
    """
    inv = {v: k for k, v in tax.class_index().items()}
    explicit: dict[str, str] = {}
    if groups_json and os.path.exists(groups_json):
        with open(groups_json, "r", encoding="utf-8") as fh:
            explicit = {str(k): str(v) for k, v in (json.load(fh) or {}).items()}

    members: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    group_classes: defaultdict[str, Counter] = defaultdict(Counter)
    group_splits: defaultdict[str, set] = defaultdict(set)

    for split, fn in _iter_split_files(root, splits):
        stem = os.path.splitext(fn)[0]
        key = explicit.get(fn) or explicit.get(stem) or group_key(
            fn, source_prefixes)
        members[key].append((split, fn))
        group_splits[key].add(split)
        group_classes[key] += _classes_in(
            os.path.join(root, "labels", split, stem + ".txt"), inv)

    leaking = [k for k, s in group_splits.items() if len(s) > 1]
    return {
        "root": root,
        "images": sum(len(v) for v in members.values()),
        "groups": len(members),
        "mean_images_per_group": (round(sum(len(v) for v in members.values())
                                        / len(members), 2) if members else 0),
        "largest_groups": [
            {"group": k, "images": len(v)}
            for k, v in sorted(members.items(), key=lambda kv: -len(kv[1]))[:10]
        ],
        "leaking_groups": len(leaking),
        "leaking_examples": sorted(leaking)[:10],
        "_members": dict(members),
        "_group_classes": {k: dict(v) for k, v in group_classes.items()},
    }


def assign_groups(group_classes: Mapping[str, Mapping[str, int]],
                  ratios: Sequence[float] = DEFAULT_RATIOS,
                  seed: int = 1234) -> dict[str, str]:
    """Assign each group to train/val/test, balancing per-class instances.

    Groups are processed rarest-class-first so that a class with 30 instances
    gets represented in val and test *before* the abundant classes soak up the
    quota. Within that ordering the choice is the split furthest below its target
    share of that class, which keeps every split's class mix close to the whole
    dataset's without ever splitting a group.
    """
    rng = random.Random(seed)
    names = ["train", "val", "test"]
    total = float(sum(ratios)) or 1.0
    share = {n: r / total for n, r in zip(names, ratios)}

    overall: Counter = Counter()
    for counts in group_classes.values():
        overall.update(counts)

    # Rarity of a group = the rarest class it contains. Sorting ascending on that
    # puts the groups carrying scarce classes first, where the quota logic can
    # still place them deliberately.
    def rarity(key: str) -> tuple[int, str]:
        counts = group_classes.get(key) or {}
        if not counts:
            return (10 ** 9, key)
        return (min(overall[c] for c in counts), key)

    keys = sorted(group_classes, key=rarity)
    rng.shuffle(keys)                      # tie-break deterministically
    keys.sort(key=rarity)

    assigned: dict[str, str] = {}
    got: dict[str, Counter] = {n: Counter() for n in names}
    n_groups: Counter = Counter()

    for key in keys:
        counts = group_classes.get(key) or {}
        if not counts:
            # An image with no labels carries no class signal; place it purely by
            # group-count quota so negatives are spread across splits too.
            best = min(names, key=lambda n: (n_groups[n] / max(share[n], 1e-9)))
            assigned[key] = best
            n_groups[best] += 1
            continue
        # Deficit = how far this split is below its target share, weighted by
        # how scarce each class is (1/sqrt(total)) so rare classes dominate.
        def deficit(n: str) -> float:
            d = 0.0
            for cid, k in counts.items():
                target = overall[cid] * share[n]
                have = got[n][cid]
                weight = 1.0 / max(overall[cid], 1) ** 0.5
                d += (target - have) * weight
            return d
        best = max(names, key=lambda n: (deficit(n), -n_groups[n]))
        assigned[key] = best
        got[best].update(counts)
        n_groups[best] += 1

    return assigned


def split_dataset(src_root: str, dst_root: str,
                  ratios: Sequence[float] = DEFAULT_RATIOS,
                  seed: int = 1234,
                  groups_json: Optional[str] = None,
                  source_prefixes: Sequence[str] = (),
                  symlink: bool = False,
                  log: Optional[Callable[[str], None]] = None) -> dict:
    """Re-split a YOLO dataset into train/val/test without leaking a group.

    Reads whatever splits ``src_root`` already has, pools every image, groups
    them, and writes a fresh ``dst_root`` with the requested ratios. The source
    is left untouched.
    """
    say = log or (lambda m: None)
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {tuple(ratios)} "
                         f"= {sum(ratios)}")

    grouped = analyse_groups(src_root, groups_json=groups_json,
                             source_prefixes=source_prefixes)
    members = grouped.pop("_members")
    group_classes = grouped.pop("_group_classes")
    if not members:
        raise ValueError(f"no images found under {src_root}/images/<split>")

    say(f"{grouped['images']} image(s) in {grouped['groups']} capture group(s)")
    if grouped["leaking_groups"]:
        say(f"the input split leaks {grouped['leaking_groups']} group(s) across "
            f"splits — re-splitting fixes that")

    assignment = assign_groups(group_classes, ratios, seed)

    inv = {v: k for k, v in tax.class_index().items()}
    per_split_images: Counter = Counter()
    per_split_classes: dict[str, Counter] = {
        n: Counter() for n in ("train", "val", "test")}

    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(dst_root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(dst_root, "labels", split), exist_ok=True)

    # Splitting pools every input split into one namespace, so a dataset that
    # uses the same filename in train and val (common: a source re-exported with
    # a stable naming scheme) would have one copy silently overwrite the other and
    # lose both its image and its labels. Track what has been written and
    # disambiguate instead.
    written: set[str] = set()
    collisions: list[str] = []

    for key, files in members.items():
        target = assignment[key]
        for src_split, fn in files:
            stem, ext = os.path.splitext(fn)
            if stem in written:
                collisions.append(fn)
                stem = f"{stem}__{src_split}"
                if stem in written:
                    stem = f"{stem}_{len(written)}"
            written.add(stem)
            s_img = os.path.join(src_root, "images", src_split, fn)
            s_lbl = os.path.join(src_root, "labels", src_split,
                                 os.path.splitext(fn)[0] + ".txt")
            d_img = os.path.join(dst_root, "images", target, stem + ext)
            d_lbl = os.path.join(dst_root, "labels", target, stem + ".txt")
            if symlink:
                if os.path.lexists(d_img):
                    os.remove(d_img)
                os.symlink(os.path.abspath(s_img), d_img)
            else:
                shutil.copy2(s_img, d_img)
            if os.path.exists(s_lbl):
                shutil.copy2(s_lbl, d_lbl)
            else:
                open(d_lbl, "w", encoding="utf-8").close()
            per_split_images[target] += 1
            per_split_classes[target] += _classes_in(d_lbl, inv)

    ds.write_dataset_yaml(dst_root)

    # A class present in train but absent from val cannot be validated, and its
    # absence is invisible in the headline mAP. Name it.
    train_c, val_c, test_c = (per_split_classes["train"],
                              per_split_classes["val"],
                              per_split_classes["test"])
    unvalidatable = sorted(c for c in train_c if not val_c.get(c))
    untestable = sorted(c for c in train_c if not test_c.get(c))
    only_val = sorted(c for c in val_c if not train_c.get(c))

    total_img = sum(per_split_images.values()) or 1
    report = {
        "src_root": src_root, "dst_root": dst_root,
        "ratios_requested": list(ratios),
        "ratios_achieved": {n: round(per_split_images[n] / total_img, 4)
                            for n in ("train", "val", "test")},
        "images_per_split": dict(per_split_images),
        "groups": grouped["groups"],
        "groups_per_split": dict(Counter(assignment.values())),
        "input_leaking_groups": grouped["leaking_groups"],
        "leaking_groups": 0,
        "instances_per_split": {n: dict(sorted(c.items(), key=lambda kv: -kv[1]))
                                for n, c in per_split_classes.items()},
        "classes_absent_from_val": unvalidatable,
        "classes_absent_from_test": untestable,
        "classes_absent_from_train": only_val,
        "seed": seed,
        "grouping": ("explicit groups.json" if groups_json
                     else "filename heuristic (group_key)"),
        "filename_collisions_renamed": len(collisions),
        "warnings": [],
    }
    if collisions:
        report["warnings"].append(
            f"{len(collisions)} filename(s) appeared in more than one input "
            f"split and were renamed rather than overwritten (e.g. "
            f"{collisions[0]}). Nothing was lost, but a source that reuses "
            f"filenames across splits is worth checking for duplicated imagery.")
    if unvalidatable:
        report["warnings"].append(
            f"{len(unvalidatable)} class(es) have training instances but NONE in "
            f"val, so their accuracy is unmeasured and they are excluded from "
            f"mAP: {', '.join(unvalidatable)}. This is a data-volume problem, "
            f"not a split problem — those classes need more instances.")
    if only_val:
        report["warnings"].append(
            f"{len(only_val)} class(es) appear only in val/test and were never "
            f"trained: {', '.join(only_val)}. Expect zero recall for them.")
    if grouped["groups"] < 20:
        report["warnings"].append(
            f"only {grouped['groups']} capture group(s) — an 80/10/10 split of "
            f"so few independent scenes gives a validation set too small to "
            f"trust. Collect more distinct panels before believing the metrics.")

    say(f"train/val/test = {per_split_images['train']}/{per_split_images['val']}"
        f"/{per_split_images['test']} image(s)")
    for w in report["warnings"]:
        say(f"warning: {w}")
    return report


__all__ = ["DEFAULT_RATIOS", "group_key", "analyse_groups", "assign_groups",
           "split_dataset"]
