"""
Focused class profiles — training a subset of the taxonomy well, instead of all of it badly.

:data:`~rtsp_backend.electrical.taxonomy.CLASS_ORDER` has 54 classes because that is
what a real panel contains. It is the right *inference* label space and the wrong
*training* label space for a first production model, for a reason worth stating
plainly: mAP is a mean over classes, so 54 classes with 30 instances each averages to a
number no threshold can rescue, while 15 classes with 400 instances each is a model
that works.

A profile is a named subset. :func:`apply` filters a YOLO dataset to a profile's
classes and **remaps the indices to 0..N-1**, which is the part that must be right —
a filtered dataset that keeps the original 54-class indices trains a 54-class head on
15 classes' worth of data, which is the failure this module exists to prevent.

Why this needs no runtime change
--------------------------------
The recogniser resolves its label space from ``classes.json``
(:func:`rtsp_backend.electrical.recognizer.load_class_map`) and canonicalises the names
through the taxonomy resolver. So a bundle exported from a 15-class profile carries a
15-entry ``classes.json``, the runtime reads it, and every detection still comes back
with a canonical taxonomy id. The 54-class taxonomy remains the inference vocabulary;
the model simply cannot produce the 39 classes it was not trained on, which is exactly
correct — and anything it is unsure of still becomes
``unknown_industrial_component`` rather than a guess.

Growing a profile later
-----------------------
Profiles are **append-only for the same reason CLASS_ORDER is**: a checkpoint's head is
positional. Adding a class to the end of a profile keeps every existing index valid and
lets an older checkpoint be fine-tuned rather than retrained. Inserting one in the
middle silently relabels everything. :func:`validate` enforces the append-only
relationship between a profile and any bundle already trained on it.
"""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

from rtsp_backend.electrical import taxonomy as tax

from . import datasets as ds


@dataclass(frozen=True)
class ClassProfile:
    """A named, ordered subset of the taxonomy to train against."""

    name: str
    #: Canonical taxonomy class ids, in training index order. APPEND ONLY.
    classes: tuple[str, ...]
    rationale: str = ""
    #: Classes deliberately excluded, and why — so the omission is a decision on the
    #: record rather than something that looks like an oversight.
    excluded_notes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = [c for c in self.classes if c not in tax.SPECS]
        if unknown:
            raise ValueError(
                f"profile {self.name!r} references classes that are not in the "
                f"taxonomy: {unknown}")
        if len(set(self.classes)) != len(self.classes):
            dupes = [c for c, n in Counter(self.classes).items() if n > 1]
            raise ValueError(f"profile {self.name!r} repeats classes: {dupes}")

    @property
    def class_count(self) -> int:
        return len(self.classes)

    def index_of(self) -> dict[str, int]:
        """class id -> profile index (0..N-1)."""
        return {cid: i for i, cid in enumerate(self.classes)}

    def taxonomy_to_profile(self) -> dict[int, int]:
        """canonical taxonomy index -> profile index, for label rewriting."""
        canon = tax.class_index()
        return {canon[cid]: i for i, cid in enumerate(self.classes)
                if cid in canon}

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "class_count": self.class_count,
            "classes": list(self.classes),
            "display_names": {c: tax.display_name(c) for c in self.classes},
            "rationale": self.rationale,
            "excluded_notes": dict(self.excluded_notes),
        }


#: The brief's 15 priority classes, mapped onto canonical taxonomy ids.
#:
#: "Switch" in the brief is ambiguous — the taxonomy distinguishes ``selector_switch``
#: (panel-fascia operator device), ``changeover_switch`` (transfer switch) and
#: ``ethernet_switch`` (network). It is read here as ``selector_switch``, because the
#: brief lists it alongside emergency stop and indicator lamp, which are the other
#: fascia-mounted operator devices. If a transfer switch was meant, use ``core18``,
#: which includes it explicitly.
CORE15 = ClassProfile(
    name="core15",
    classes=(
        "mcb", "mccb", "contactor", "relay", "plc", "terminal_block", "fuse",
        "power_supply", "transformer", "vfd", "busbar", "wire_duct",
        "emergency_stop", "selector_switch", "indicator_lamp",
    ),
    rationale=(
        "The 15 devices that account for the large majority of what is physically "
        "present in an LV industrial panel, and for nearly all of what an inspection "
        "report needs to say. Chosen to make mAP a meaningful number: mAP is a mean "
        "over classes, so training 54 classes on thin data averages to something no "
        "threshold can rescue, whereas 15 classes with several hundred instances each "
        "is a model that works. Grow the profile once these are reliable."),
    excluded_notes={
        "overload_relay": "Excluded from core15 to match the brief exactly, but it is "
                          "the single most valuable addition: it is always bolted "
                          "under a contactor, and the contactor-without-overload "
                          "check is one of the report's most useful findings. Present "
                          "in core18.",
        "din_rail": "Structural rather than a device. Useful for row inference and "
                    "cheap to label. Present in core18.",
        "circuit_breaker": "The generic 'type unspecified' breaker. Excluded here so "
                           "the model is forced to commit to MCB or MCCB; include it "
                           "(core18) if your imagery genuinely contains breakers whose "
                           "family cannot be determined.",
        "acb": "Air circuit breakers appear almost exclusively in incomer sections and "
               "are rare per panel — hard to reach a usable instance count early.",
        "current_transformer": "Small, visually similar to a cable gland, and usually "
                               "partly occluded by the conductor passing through it. "
                               "Add it once the easier classes are solid.",
        "cooling_fan": "Mounted on the enclosure wall rather than the back plate, so "
                       "it is frequently out of frame at row framing.",
    },
)

#: core15 plus the three highest-value additions. Appended, so a core15 checkpoint can
#: be fine-tuned onto core18 without invalidating its existing head indices.
CORE18 = ClassProfile(
    name="core18",
    classes=CORE15.classes + ("overload_relay", "din_rail", "circuit_breaker"),
    rationale=(
        "core15 plus the overload relay (the contactor-without-overload check is one "
        "of the report's most useful findings), the DIN rail (cheap to label, enables "
        "row inference) and the generic circuit breaker (an honest home for breakers "
        "whose family genuinely cannot be read). A strict superset of core15 in the "
        "same order, so a core15 checkpoint fine-tunes onto it."),
)

#: Every taxonomy class. The eventual target, not a sensible starting point.
FULL = ClassProfile(
    name="full",
    classes=tuple(tax.CLASS_ORDER),
    rationale=(
        "The complete inference vocabulary. Only worth training once the core "
        "profiles are reliable and the long tail has real instance counts — before "
        "that, the rare classes contribute noise to the mean and nothing to the "
        "product."),
)

PROFILES: dict[str, ClassProfile] = {p.name: p for p in (CORE15, CORE18, FULL)}

DEFAULT_PROFILE = "core15"


def get(name: str) -> ClassProfile:
    try:
        return PROFILES[name]
    except KeyError:
        raise KeyError(
            f"unknown profile {name!r}; known profiles: "
            f"{', '.join(sorted(PROFILES))}") from None


def list_profiles() -> dict:
    return {
        "default": DEFAULT_PROFILE,
        "profiles": [p.to_dict() for p in PROFILES.values()],
        "note": (
            "A profile is the TRAINING label space. The 54-class taxonomy stays the "
            "inference vocabulary: a profile bundle ships a matching classes.json, "
            "the runtime reads it, and detections still carry canonical taxonomy ids. "
            "The model simply cannot emit the classes it was not trained on, and "
            "anything it is unsure of still becomes "
            f"'{tax.UNKNOWN_COMPONENT_ID}' rather than a guess."),
    }


# --------------------------------------------------------------------------
# dataset filtering
# --------------------------------------------------------------------------

def write_profile_yaml(root: str, profile: ClassProfile) -> str:
    """Write ``dataset.yaml`` + ``classes.json`` for a profile-filtered dataset."""
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "dataset.yaml")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# Class profile '{profile.name}' — a SUBSET of "
                 f"rtsp_backend.electrical.taxonomy.CLASS_ORDER.\n")
        fh.write(f"# Indices here are 0..{profile.class_count - 1} and are NOT "
                 f"taxonomy indices.\n")
        fh.write(f"path: {os.path.abspath(root)}\n")
        fh.write("train: images/train\nval: images/val\n")
        if os.path.isdir(os.path.join(root, "images", "test")):
            fh.write("test: images/test\n")
        fh.write(f"nc: {profile.class_count}\nnames:\n")
        for i, cid in enumerate(profile.classes):
            fh.write(f"  {i}: {cid}\n")

    with open(os.path.join(root, "classes.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "_comment": (
                f"Class profile '{profile.name}'. This is the model's label order and "
                f"the runtime reads it directly. APPEND ONLY — reordering invalidates "
                f"any checkpoint trained against it."),
            "profile": profile.name,
            "taxonomy_version": "5.1",
            "class_count": profile.class_count,
            "classes": list(profile.classes),
            "display_names": {c: tax.display_name(c) for c in profile.classes},
        }, fh, indent=2)
    return path


def present_classes(root: str, profile: ClassProfile,
                    splits: Sequence[str] = ("train", "val", "test"),
                    min_instances: int = 1) -> tuple[str, ...]:
    """Which of a profile's classes actually have instances in a dataset.

    Order is preserved from the profile, so the derived subset stays a *subsequence*
    of it — which keeps the append-only reasoning intact and means a checkpoint
    trained on the subset can be compared against one trained on the full profile.
    """
    counts = Counter()
    inv = {i: c for c, i in tax.class_index().items()}
    wanted = set(profile.classes)
    for split in splits:
        lbl_dir = os.path.join(root, "labels", split)
        if not os.path.isdir(lbl_dir):
            continue
        for fn in sorted(os.listdir(lbl_dir)):
            if not fn.endswith(".txt"):
                continue
            with open(os.path.join(lbl_dir, fn), "r", encoding="utf-8") as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    try:
                        cid = inv.get(int(float(parts[0])))
                    except ValueError:
                        continue
                    if cid in wanted:
                        counts[cid] += 1
    return tuple(c for c in profile.classes
                 if counts.get(c, 0) >= min_instances)


def derive(profile: ClassProfile, classes: Sequence[str],
           suffix: str = "present") -> ClassProfile:
    """A profile narrowed to ``classes``, preserving the parent's order.

    Used to train on what a dataset actually contains rather than declaring classes
    the data cannot support. An absent class does not merely fail to learn — it
    contributes a zero to the mAP mean, so a 15-class profile with 7 empty classes
    caps the reportable mAP at 8/15 no matter how well the rest train. Reporting that
    number as the model's accuracy would be misleading in both directions.
    """
    kept = tuple(c for c in profile.classes if c in set(classes))
    return ClassProfile(
        name=f"{profile.name}_{suffix}",
        classes=kept,
        rationale=(
            f"Derived from '{profile.name}' ({profile.class_count} classes), narrowed "
            f"to the {len(kept)} class(es) with instances in the dataset it was "
            f"derived from. A subsequence of the parent, so order and append-only "
            f"reasoning are preserved. The {profile.class_count - len(kept)} omitted "
            f"class(es) had no data: training them would add zeros to the mAP mean "
            f"and nothing to the model."),
        excluded_notes={
            c: f"no instances in the source dataset; omitted so it does not dilute "
               f"the mAP mean"
            for c in profile.classes if c not in set(kept)},
    )


def apply(src_root: str, dst_root: str, profile: ClassProfile,
          splits: Sequence[str] = ("train", "val", "test"),
          drop_empty: bool = False,
          symlink: bool = False,
          log: Optional[Callable[[str], None]] = None) -> dict:
    """Filter a canonically-labelled YOLO dataset down to one profile.

    Boxes whose class is outside the profile are dropped and counted. Indices are
    rewritten to the profile's 0..N-1 space — the step that must not be skipped, since
    a filtered dataset carrying the original taxonomy indices would train an
    N-class head against indices scattered up to 53.

    ``drop_empty`` also removes images left with no boxes. Off by default: an image
    that contained only out-of-profile devices is a genuine **negative** for this
    profile, and negatives are how a detector learns not to fire on the classes it was
    not trained for. Keep some.
    """
    say = log or (lambda m: None)
    remap = profile.taxonomy_to_profile()
    inv_canon = {i: c for c, i in tax.class_index().items()}

    stats = {
        "profile": profile.name,
        "src_root": src_root, "dst_root": dst_root,
        "images_in": 0, "images_out": 0, "images_emptied": 0,
        "images_dropped_empty": 0,
        "instances_kept": 0, "instances_dropped": 0,
        "dropped_by_class": Counter(), "per_class": Counter(),
        "per_split": {},
    }

    for split in splits:
        img_dir = os.path.join(src_root, "images", split)
        if not os.path.isdir(img_dir):
            continue
        d_img = os.path.join(dst_root, "images", split)
        d_lbl = os.path.join(dst_root, "labels", split)
        os.makedirs(d_img, exist_ok=True)
        os.makedirs(d_lbl, exist_ok=True)
        kept_here = 0

        for fn in sorted(os.listdir(img_dir)):
            if not fn.lower().endswith(ds.IMAGE_EXTS):
                continue
            stats["images_in"] += 1
            stem, ext = os.path.splitext(fn)
            src_lbl = os.path.join(src_root, "labels", split, stem + ".txt")
            lines: list[str] = []
            if os.path.exists(src_lbl):
                with open(src_lbl, "r", encoding="utf-8") as fh:
                    for line in fh:
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        try:
                            canon_idx = int(float(parts[0]))
                        except ValueError:
                            continue
                        if canon_idx not in remap:
                            stats["instances_dropped"] += 1
                            stats["dropped_by_class"][
                                inv_canon.get(canon_idx, str(canon_idx))] += 1
                            continue
                        new_idx = remap[canon_idx]
                        lines.append(" ".join([str(new_idx)] + parts[1:5]))
                        stats["instances_kept"] += 1
                        stats["per_class"][profile.classes[new_idx]] += 1

            if not lines:
                stats["images_emptied"] += 1
                if drop_empty:
                    stats["images_dropped_empty"] += 1
                    continue

            if symlink:
                dst = os.path.join(d_img, fn)
                if os.path.lexists(dst):
                    os.remove(dst)
                os.symlink(os.path.abspath(os.path.join(img_dir, fn)), dst)
            else:
                shutil.copy2(os.path.join(img_dir, fn),
                             os.path.join(d_img, fn))
            with open(os.path.join(d_lbl, stem + ".txt"), "w",
                      encoding="utf-8") as fh:
                fh.write("\n".join(lines) + ("\n" if lines else ""))
            stats["images_out"] += 1
            kept_here += 1
        stats["per_split"][split] = kept_here
        say(f"  {split}: {kept_here} image(s)")

    write_profile_yaml(dst_root, profile)
    stats["dropped_by_class"] = dict(
        sorted(stats["dropped_by_class"].items(), key=lambda kv: -kv[1]))
    stats["per_class"] = dict(
        sorted(stats["per_class"].items(), key=lambda kv: -kv[1]))
    stats["dataset_yaml"] = os.path.join(dst_root, "dataset.yaml")
    stats["absent_classes"] = [c for c in profile.classes
                               if c not in stats["per_class"]]
    stats["warnings"] = _apply_warnings(stats, profile)

    say(f"{stats['images_out']} image(s), {stats['instances_kept']} instance(s) "
        f"across {len(stats['per_class'])}/{profile.class_count} profile class(es)")
    for w in stats["warnings"]:
        say(f"warning: {w}")
    return stats


def _apply_warnings(stats: Mapping, profile: ClassProfile) -> list[str]:
    out: list[str] = []
    if stats["absent_classes"]:
        out.append(
            f"{len(stats['absent_classes'])} profile class(es) have NO instances in "
            f"this dataset and cannot be learned: "
            f"{', '.join(stats['absent_classes'])}. They will also drag the mAP mean "
            f"down to no purpose — either collect data for them or drop them from the "
            f"profile.")
    if stats["images_emptied"] and not stats["images_dropped_empty"]:
        out.append(
            f"{stats['images_emptied']} image(s) have no in-profile boxes and were "
            f"kept as negatives. That is usually right — negatives teach the detector "
            f"not to fire on out-of-profile devices — but cap them at roughly 10-15% "
            f"of the training set, or they start suppressing genuine detections.")
    counts = stats["per_class"]
    if counts:
        least = min(counts.values())
        most = max(counts.values())
        if least < ds.MIN_INSTANCES_TRAINABLE:
            thin = [c for c, n in counts.items()
                    if n < ds.MIN_INSTANCES_TRAINABLE]
            out.append(
                f"{len(thin)} class(es) are below the "
                f"{ds.MIN_INSTANCES_TRAINABLE}-instance trainability floor: "
                f"{', '.join(thin)}. Reducing the profile further is better than "
                f"training a class on 20 examples.")
        if most / max(least, 1) >= 20:
            out.append(
                f"class balance within the profile is {most / max(least, 1):.0f}:1 — "
                f"the loss will be dominated by the common classes.")
    return out


def validate(profile: ClassProfile, bundle_classes: Sequence[str]) -> dict:
    """Check a trained bundle's label space against a profile.

    A bundle must be the profile exactly, or an append-only **prefix** of it (an older
    checkpoint trained before the profile grew). Anything else means the indices would
    be misinterpreted at runtime.
    """
    got = list(bundle_classes)
    want = list(profile.classes)
    problems: list[str] = []
    note = None

    if got == want:
        note = f"bundle matches profile '{profile.name}' exactly."
    elif got == want[:len(got)]:
        note = (f"bundle has {len(got)} of the profile's {len(want)} classes and is a "
                f"valid prefix, so its indices are correct — it simply cannot detect "
                f"the {len(want) - len(got)} class(es) added later "
                f"({', '.join(want[len(got):])}). Fine-tune rather than retrain.")
    else:
        first_diff = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b),
                          min(len(got), len(want)))
        problems.append(
            f"bundle label order diverges from profile '{profile.name}' at index "
            f"{first_diff} (bundle has "
            f"{got[first_diff] if first_diff < len(got) else '<end>'!r}, profile has "
            f"{want[first_diff] if first_diff < len(want) else '<end>'!r}). Every "
            f"detection from index {first_diff} on would be mislabelled. The profile "
            f"was reordered or edited in the middle — profiles are append-only.")

    return {"ok": not problems, "profile": profile.name,
            "bundle_class_count": len(got), "profile_class_count": len(want),
            "problems": problems, "note": note}


def requirement_estimate(profile: ClassProfile,
                         target_map50: float = 0.85) -> dict:
    """What it takes to reach a target mAP on this profile, and where that comes from.

    These are working figures from detection fine-tuning practice, not a guarantee —
    stated as a band, with the assumptions visible, because a single confident number
    here would be false precision.
    """
    # Instances per class needed, by target. Bands rather than points: the real figure
    # depends on intra-class visual variety (how many manufacturers, how many framings),
    # which is a property of the capture programme rather than of the model.
    bands = [
        (0.50, 150, 300, "a usable demonstrator; obvious devices in good light"),
        (0.70, 300, 600, "useful in production for the common classes, with a human "
                         "reviewing the output"),
        (0.85, 700, 1200, "the stated target; reliable enough to drive a report "
                          "without per-image review"),
        (0.92, 1500, 2500, "diminishing returns territory; usually needs "
                           "manufacturer-level coverage of every class"),
    ]
    chosen = min(bands, key=lambda b: abs(b[0] - target_map50))
    _tgt, lo, hi, meaning = chosen

    n = profile.class_count
    inst_lo, inst_hi = lo * n, hi * n
    # Real panel photography at row framing yields ~12 labelled boxes per image, but
    # they are not evenly spread across classes — an image full of MCBs contributes
    # nothing to the VFD count. The effective yield per class is far lower, so image
    # counts are derived from the per-class requirement and an assumed co-occurrence
    # factor rather than from the raw instance total.
    boxes_per_image = ds.INSTANCES_PER_IMAGE_ESTIMATE
    classes_per_image = 4.0          # distinct profile classes visible in one frame
    images_lo = int(round(inst_lo / (boxes_per_image * classes_per_image / n)))
    images_hi = int(round(inst_hi / (boxes_per_image * classes_per_image / n)))

    return {
        "profile": profile.name,
        "class_count": n,
        "target_map50": target_map50,
        "meaning": meaning,
        "instances_per_class": {"low": lo, "high": hi},
        "total_instances": {"low": inst_lo, "high": inst_hi},
        "images": {"low": images_lo, "high": images_hi},
        "bands": [
            {"map50": t, "instances_per_class_low": a,
             "instances_per_class_high": b, "meaning": m}
            for t, a, b, m in bands
        ],
        "assumptions": [
            f"~{boxes_per_image:g} labelled boxes per panel photograph at row framing.",
            f"~{classes_per_image:g} distinct profile classes visible per frame — an "
            f"image full of MCBs contributes nothing to the VFD count, which is why "
            f"the image requirement is far higher than total_instances / "
            f"boxes_per_image.",
            "At least three manufacturers per class. Manufacturer invariance is "
            "learned from examples; a model trained on one brand fails on the next.",
            "Split by panel, not by image, or the measured mAP is inflated and the "
            "target is met on paper only.",
        ],
        "caveat": (
            "These bands describe what the DATA has to be. They assume the training "
            "recipe is already sound — which for this repository is measured rather "
            "than assumed, see the recipe-validation run in docs/AUDIT_v5.2.0.md. No "
            "hyperparameter search substitutes for instances: if a class has 30 "
            "examples, no setting makes it work."),
    }


__all__ = [
    "ClassProfile", "CORE15", "CORE18", "FULL", "PROFILES", "DEFAULT_PROFILE",
    "get", "list_profiles", "write_profile_yaml", "apply", "validate",
    "requirement_estimate",
]
