"""
Dataset acquisition and unification for industrial component detection.

The single biggest reason the previous system could not recognise a contactor is
that no model was ever trained, and no model was ever trained because there was
no dataset. This module is the plan for getting one, executed as code.

Three problems it solves:

1. **Where to get data.** :data:`SOURCES` is a curated registry of public
   datasets that contain industrial electrical components, each with its licence,
   the classes it actually covers, and a mapping from its label names onto the
   canonical taxonomy. Nothing is downloaded implicitly — you supply an API key
   or a local copy, and :func:`plan` tells you exactly what you will get.

2. **Merging incompatible label spaces.** Public sets disagree on everything:
   ``"breaker"`` vs ``"MCB"`` vs ``"circuit_breaker_1p"``. :func:`remap_yolo_dataset`
   rewrites any YOLO-format dataset onto :data:`~rtsp_backend.electrical.taxonomy.CLASS_ORDER`
   using each source's declared mapping plus the taxonomy resolver, and *drops*
   (with a count) any class it cannot map rather than guessing. :func:`merge`
   then unions several remapped datasets into one training set.

3. **Knowing whether the result is usable.** :func:`analyse_dataset` reports
   per-class instance counts, images per class, box-size distribution and the
   long tail. :func:`coverage_report` compares that against the taxonomy and
   names the classes that will *not* work yet, so nobody is surprised when the
   model cannot find an ACB it never saw.

Where public data is insufficient — which, for most of this taxonomy, it is —
:func:`custom_collection_plan` emits the concrete capture protocol for building
a proprietary Madkour dataset, and
:mod:`training.electrical.synthetic` multiplies a small crop library into a
large labelled set.
"""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

from rtsp_backend.electrical import taxonomy as tax

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True)
class DatasetSource:
    """A public dataset that contains industrial electrical components."""

    key: str
    name: str
    #: "roboflow" | "kaggle" | "url" | "manual"
    kind: str
    #: Where to get it. Roboflow entries are ``workspace/project/version``.
    locator: str
    licence: str
    #: Taxonomy classes this source realistically contributes.
    provides: tuple[str, ...]
    #: source label -> taxonomy id. Labels absent here fall back to the resolver.
    label_map: Mapping[str, str] = field(default_factory=dict)
    notes: str = ""
    #: Rough instance count, for planning only.
    approx_instances: Optional[int] = None

    def cli_hint(self) -> str:
        if self.kind == "roboflow":
            return (f"roboflow download (needs ROBOFLOW_API_KEY): "
                    f"rf.workspace('{self.locator.split('/')[0]}')"
                    f".project('{self.locator.split('/')[1]}')"
                    f".version({self.locator.split('/')[-1]})"
                    f".download('yolov8')")
        if self.kind == "kaggle":
            return f"kaggle datasets download -d {self.locator}"
        if self.kind == "url":
            return f"curl -L -o dataset.zip {self.locator}"
        return "manual acquisition — see notes"


#: Curated source registry.
#:
#: These are *starting points*, not a solved dataset problem. Public coverage of
#: this taxonomy is thin and skewed heavily toward modular breakers and
#: terminal blocks; drives, ACBs, safety relays, meters and instrument
#: transformers are effectively absent. Treat public data as a way to bootstrap
#: the common classes and plan to capture the rest (see
#: :func:`custom_collection_plan`). Locators are subject to change upstream —
#: `plan()` verifies nothing, it only tells you what to fetch and how.
SOURCES: tuple[DatasetSource, ...] = (
    DatasetSource(
        key="rf_electrical_panel_components",
        name="Roboflow Universe — electrical panel component detection sets",
        kind="roboflow", locator="<workspace>/<project>/<version>",
        licence="per-project (mostly CC BY 4.0) — verify before commercial use",
        provides=("mcb", "mccb", "contactor", "relay", "terminal_block",
                  "power_supply", "plc", "fuse_holder"),
        label_map={"breaker": "mcb", "circuit breaker": "mcb",
                   "mould case circuit breaker": "mccb",
                   "magnetic contactor": "contactor",
                   "aux relay": "relay", "terminal": "terminal_block",
                   "smps": "power_supply", "controller": "plc"},
        notes="Search Roboflow Universe for 'electrical panel', 'control panel "
              "components', 'switchgear', 'MCB detection'. Several small "
              "community projects exist; individually they are too small, "
              "merged they cover the common modular classes. Set the concrete "
              "workspace/project/version in your own config before downloading.",
        approx_instances=None,
    ),
    DatasetSource(
        key="rf_switchgear",
        name="Roboflow Universe — switchgear / distribution board sets",
        kind="roboflow", locator="<workspace>/<project>/<version>",
        licence="per-project — verify",
        provides=("mcb", "rccb", "rcbo", "busbar", "neutral_bar", "earth_bar",
                  "din_rail"),
        label_map={"rcd": "rccb", "elcb": "rccb", "bus bar": "busbar",
                   "earth": "earth_bar", "neutral": "neutral_bar",
                   "rail": "din_rail"},
        notes="Distribution-board photographs are the best-covered category in "
              "public data; useful for the modular protection classes and for "
              "learning the DIN-rail row structure.",
    ),
    DatasetSource(
        key="rf_ppe_industrial_context",
        name="Roboflow Universe — industrial cabinet / machine-room imagery",
        kind="roboflow", locator="<workspace>/<project>/<version>",
        licence="per-project — verify",
        provides=("cooling_fan", "wire_duct", "cable_gland", "hmi",
                  "push_button", "emergency_stop", "indicator_lamp"),
        label_map={"estop": "emergency_stop", "e-stop": "emergency_stop",
                   "hmi screen": "hmi", "pilot lamp": "indicator_lamp",
                   "trunking": "wire_duct"},
        notes="Operator-device classes appear in HMI/panel-door datasets. "
              "Emergency stops in particular are well represented because of "
              "safety-compliance projects.",
    ),
    DatasetSource(
        key="vendor_catalogue_crops",
        name="Manufacturer catalogue product photography",
        kind="manual", locator="vendor product pages / catalogue PDFs",
        licence="COPYRIGHTED — obtain written permission before training on it",
        provides=tuple(tax.CLASS_ORDER),
        notes="Every manufacturer publishes clean studio photographs of every "
              "product, labelled with the exact part number. As a *crop "
              "library* for training.electrical.synthetic.compose_from_crops "
              "this is the highest-value source per unit of effort: it covers "
              "the whole taxonomy including the classes public datasets miss, "
              "and it comes with ground-truth part numbers for the nameplate "
              "catalogue. It is also copyrighted — clear it with the vendor or "
              "your legal team first. Studio crops alone under-represent real "
              "lighting and occlusion, which is precisely what the synthetic "
              "compositor adds.",
    ),
    DatasetSource(
        key="madkour_field_capture",
        name="Madkour field capture programme (proprietary)",
        kind="manual", locator="internal capture — see custom_collection_plan()",
        licence="proprietary — owned by Madkour",
        provides=tuple(tax.CLASS_ORDER),
        notes="The only source that matches the deployment distribution: real "
              "Madkour panels, real cabinets, real lighting, real dirt. This is "
              "what production accuracy ultimately depends on. Everything else "
              "is a bootstrap.",
    ),
)

SOURCE_INDEX: dict[str, DatasetSource] = {s.key: s for s in SOURCES}


def plan(keys: Optional[Sequence[str]] = None) -> dict:
    """What each selected source contributes, and what remains uncovered."""
    chosen = [SOURCE_INDEX[k] for k in (keys or list(SOURCE_INDEX))
              if k in SOURCE_INDEX]
    # A "manual" source (vendor catalogues, field capture) nominally provides
    # every class, but only after somebody photographs and labels it. Counting
    # that as coverage would turn this report into wishful thinking, so
    # downloadable and manual coverage are reported separately.
    downloadable: set[str] = set()
    manual: set[str] = set()
    for s in chosen:
        (downloadable if s.kind != "manual" else manual).update(s.provides)
    uncovered = [c for c in tax.CLASS_ORDER if c not in downloadable]
    manual_only = [c for c in uncovered if c in manual]
    nowhere = [c for c in uncovered if c not in manual]

    return {
        "sources": [
            {"key": s.key, "name": s.name, "kind": s.kind,
             "locator": s.locator, "licence": s.licence,
             "provides": list(s.provides), "notes": s.notes,
             "how": s.cli_hint()}
            for s in chosen
        ],
        "classes_from_downloadable_sources": sorted(downloadable),
        "classes_needing_manual_capture": manual_only,
        "classes_with_no_source": nowhere,
        "downloadable_coverage_fraction": round(
            len(downloadable) / max(1, len(tax.CLASS_ORDER)), 3),
        "verdict": (
            f"{len(downloadable)} of {len(tax.CLASS_ORDER)} classes have a "
            f"downloadable public source; {len(manual_only)} depend on vendor "
            f"catalogue crops or the Madkour field-capture programme. Public "
            f"data alone will not cover this taxonomy — bootstrap the common "
            f"modular classes from it, build a crop library for the rest, and "
            f"capture real Madkour panels for anything that must work in "
            f"production."
            if uncovered else
            "Every taxonomy class has a downloadable source."),
    }


# --------------------------------------------------------------------------
# YOLO dataset remapping / merging
# --------------------------------------------------------------------------

def read_yolo_names(dataset_yaml: str) -> list[str]:
    """Read the ``names`` list from a YOLO ``dataset.yaml`` (dict or list form)."""
    import yaml  # PyYAML is already a runtime dependency

    with open(dataset_yaml, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    names = data.get("names")
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
    if isinstance(names, list):
        return [str(n) for n in names]
    raise ValueError(f"{dataset_yaml}: no usable 'names' entry")


def build_index_map(source_names: Sequence[str],
                    label_map: Optional[Mapping[str, str]] = None
                    ) -> tuple[dict[int, int], list[str]]:
    """Map source class indices onto canonical taxonomy indices.

    Returns ``(index_map, unmapped_names)``. A source class that cannot be
    resolved is *omitted* from the map, and its instances are dropped with a
    count — silently folding an unknown class into a known one is how label noise
    gets baked into a model.
    """
    lm = {str(k).strip().lower(): v for k, v in (label_map or {}).items()}
    canon_idx = tax.class_index()
    out: dict[int, int] = {}
    unmapped: list[str] = []
    for i, name in enumerate(source_names):
        explicit = lm.get(str(name).strip().lower())
        cid = explicit or tax.resolve(name)
        if cid and cid in canon_idx:
            out[i] = canon_idx[cid]
        else:
            unmapped.append(str(name))
    return out, unmapped


def remap_yolo_dataset(src_root: str, dst_root: str,
                       source_names: Sequence[str],
                       label_map: Optional[Mapping[str, str]] = None,
                       splits: Sequence[str] = ("train", "val", "test"),
                       copy_images: bool = True,
                       prefix: str = "") -> dict:
    """Rewrite a YOLO dataset onto the canonical label space."""
    index_map, unmapped = build_index_map(source_names, label_map)
    stats = {"images": 0, "instances_kept": 0, "instances_dropped": 0,
             "dropped_by_class": Counter(), "unmapped_source_classes": unmapped,
             "per_class": Counter()}
    inv = {v: k for k, v in tax.class_index().items()}

    for split in splits:
        s_img = os.path.join(src_root, "images", split)
        s_lbl = os.path.join(src_root, "labels", split)
        if not os.path.isdir(s_img):
            continue
        d_img = os.path.join(dst_root, "images", split)
        d_lbl = os.path.join(dst_root, "labels", split)
        os.makedirs(d_img, exist_ok=True)
        os.makedirs(d_lbl, exist_ok=True)

        for fn in sorted(os.listdir(s_img)):
            if not fn.lower().endswith(IMAGE_EXTS):
                continue
            stem, ext = os.path.splitext(fn)
            out_stem = f"{prefix}{stem}" if prefix else stem
            lbl_path = os.path.join(s_lbl, stem + ".txt")
            kept_lines: list[str] = []
            if os.path.exists(lbl_path):
                with open(lbl_path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        try:
                            src_cls = int(float(parts[0]))
                        except ValueError:
                            continue
                        if src_cls not in index_map:
                            stats["instances_dropped"] += 1
                            name = (source_names[src_cls]
                                    if src_cls < len(source_names) else str(src_cls))
                            stats["dropped_by_class"][name] += 1
                            continue
                        new_cls = index_map[src_cls]
                        kept_lines.append(" ".join([str(new_cls)] + parts[1:5]))
                        stats["instances_kept"] += 1
                        stats["per_class"][inv[new_cls]] += 1
            with open(os.path.join(d_lbl, out_stem + ".txt"), "w",
                      encoding="utf-8") as fh:
                fh.write("\n".join(kept_lines) + ("\n" if kept_lines else ""))
            dst_img = os.path.join(d_img, out_stem + ext)
            if copy_images:
                shutil.copy2(os.path.join(s_img, fn), dst_img)
            else:
                if os.path.lexists(dst_img):
                    os.remove(dst_img)
                os.symlink(os.path.abspath(os.path.join(s_img, fn)), dst_img)
            stats["images"] += 1

    stats["dropped_by_class"] = dict(stats["dropped_by_class"])
    stats["per_class"] = dict(stats["per_class"])
    write_dataset_yaml(dst_root)
    return stats


def merge(roots: Sequence[str], dst_root: str,
          splits: Sequence[str] = ("train", "val", "test"),
          copy_images: bool = True) -> dict:
    """Union several already-remapped datasets into one.

    File names are prefixed per source so identically-named images from
    different datasets cannot silently overwrite one another — a real and easy
    way to lose half a dataset.
    """
    totals = {"images": 0, "instances": 0, "per_class": Counter(),
              "per_source": {}}
    for n, root in enumerate(roots):
        src_stats = {"images": 0, "instances": 0}
        for split in splits:
            s_img = os.path.join(root, "images", split)
            s_lbl = os.path.join(root, "labels", split)
            if not os.path.isdir(s_img):
                continue
            d_img = os.path.join(dst_root, "images", split)
            d_lbl = os.path.join(dst_root, "labels", split)
            os.makedirs(d_img, exist_ok=True)
            os.makedirs(d_lbl, exist_ok=True)
            for fn in sorted(os.listdir(s_img)):
                if not fn.lower().endswith(IMAGE_EXTS):
                    continue
                stem, ext = os.path.splitext(fn)
                out_stem = f"s{n}_{stem}"
                if copy_images:
                    shutil.copy2(os.path.join(s_img, fn),
                                 os.path.join(d_img, out_stem + ext))
                else:
                    dst = os.path.join(d_img, out_stem + ext)
                    if os.path.lexists(dst):
                        os.remove(dst)
                    os.symlink(os.path.abspath(os.path.join(s_img, fn)), dst)
                lbl = os.path.join(s_lbl, stem + ".txt")
                lines: list[str] = []
                if os.path.exists(lbl):
                    with open(lbl, "r", encoding="utf-8") as fh:
                        lines = [ln.strip() for ln in fh if ln.strip()]
                with open(os.path.join(d_lbl, out_stem + ".txt"), "w",
                          encoding="utf-8") as fh:
                    fh.write("\n".join(lines) + ("\n" if lines else ""))
                inv = {v: k for k, v in tax.class_index().items()}
                for ln in lines:
                    try:
                        totals["per_class"][inv[int(float(ln.split()[0]))]] += 1
                    except (ValueError, KeyError, IndexError):
                        continue
                totals["images"] += 1
                totals["instances"] += len(lines)
                src_stats["images"] += 1
                src_stats["instances"] += len(lines)
        totals["per_source"][root] = src_stats
    totals["per_class"] = dict(sorted(totals["per_class"].items(),
                                      key=lambda kv: -kv[1]))
    write_dataset_yaml(dst_root)
    return totals


def write_dataset_yaml(root: str) -> str:
    idx = tax.class_index()
    path = os.path.join(root, "dataset.yaml")
    os.makedirs(root, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Canonical label space — rtsp_backend.electrical.taxonomy\n")
        fh.write(f"path: {os.path.abspath(root)}\n")
        fh.write("train: images/train\nval: images/val\n")
        if os.path.isdir(os.path.join(root, "images", "test")):
            fh.write("test: images/test\n")
        fh.write(f"nc: {len(idx)}\nnames:\n")
        for cid, i in sorted(idx.items(), key=lambda kv: kv[1]):
            fh.write(f"  {i}: {cid}\n")
    with open(os.path.join(root, "classes.json"), "w", encoding="utf-8") as fh:
        json.dump({"classes": list(tax.CLASS_ORDER)}, fh, indent=2)
    return path


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------

def analyse_dataset(root: str, splits: Sequence[str] = ("train", "val", "test")
                    ) -> dict:
    """Per-class instance/image counts and box-size distribution."""
    inv = {v: k for k, v in tax.class_index().items()}
    per_class: Counter = Counter()
    images_with: defaultdict[str, set] = defaultdict(set)
    sizes: defaultdict[str, list[float]] = defaultdict(list)
    n_images = 0
    per_split: dict[str, int] = {}

    for split in splits:
        lbl_dir = os.path.join(root, "labels", split)
        if not os.path.isdir(lbl_dir):
            continue
        count = 0
        for fn in sorted(os.listdir(lbl_dir)):
            if not fn.endswith(".txt"):
                continue
            count += 1
            n_images += 1
            with open(os.path.join(lbl_dir, fn), "r", encoding="utf-8") as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    try:
                        cid = inv[int(float(parts[0]))]
                        w, h = float(parts[3]), float(parts[4])
                    except (ValueError, KeyError):
                        continue
                    per_class[cid] += 1
                    images_with[cid].add(f"{split}/{fn}")
                    sizes[cid].append(w * h)
        per_split[split] = count

    rows = []
    for cid, n in per_class.most_common():
        areas = sizes[cid]
        rows.append({
            "class_id": cid, "name": tax.display_name(cid), "instances": n,
            "images": len(images_with[cid]),
            "mean_rel_area": round(sum(areas) / len(areas), 6) if areas else None,
            "min_rel_area": round(min(areas), 6) if areas else None,
            "max_rel_area": round(max(areas), 6) if areas else None,
        })
    return {"root": root, "images": n_images, "images_per_split": per_split,
            "instances": int(sum(per_class.values())), "per_class": rows}


#: Below this many instances a class will not train usefully; below the warn
#: level it will train but stay unreliable. These are working rules of thumb for
#: detection fine-tuning, not guarantees.
MIN_INSTANCES_TRAINABLE = 50
MIN_INSTANCES_RELIABLE = 300


def coverage_report(analysis: Mapping) -> dict:
    """Name the classes that will and will not work, before training starts."""
    counts = {r["class_id"]: r["instances"] for r in analysis.get("per_class", [])}
    reliable, weak, untrainable, absent = [], [], [], []
    for cid in tax.CLASS_ORDER:
        n = counts.get(cid, 0)
        entry = {"class_id": cid, "name": tax.display_name(cid), "instances": n}
        if n == 0:
            absent.append(entry)
        elif n < MIN_INSTANCES_TRAINABLE:
            untrainable.append(entry)
        elif n < MIN_INSTANCES_RELIABLE:
            weak.append(entry)
        else:
            reliable.append(entry)
    return {
        "reliable": reliable, "weak": weak, "untrainable": untrainable,
        "absent": absent,
        "thresholds": {"trainable": MIN_INSTANCES_TRAINABLE,
                       "reliable": MIN_INSTANCES_RELIABLE},
        "summary": (
            f"{len(reliable)} class(es) have enough data to be reliable, "
            f"{len(weak)} will train but stay weak, "
            f"{len(untrainable)} have too few instances, "
            f"{len(absent)} are absent entirely. Detections for anything other "
            f"than the reliable set should be expected to fall through to "
            f"'Unknown Industrial Component'."),
    }


def custom_collection_plan() -> dict:
    """The capture protocol for building the proprietary Madkour dataset.

    Written as a checklist an engineer with a phone camera can execute on site.
    """
    return {
        "objective": (
            f"{MIN_INSTANCES_RELIABLE}+ labelled instances per class that must "
            f"work in production, captured under deployment conditions."),
        "capture_protocol": [
            "Photograph every panel at three framings: whole cabinet with the "
            "door open, each device row filling the frame, and a close-up of "
            "every device nameplate. The row framing is what the model will see "
            "in service; the nameplate close-ups train and validate the "
            "part-number reader.",
            "Vary the camera angle deliberately: straight on, and roughly ±30° "
            "horizontally and vertically. A model trained only on square-on "
            "photographs fails the moment an inspector stands to one side.",
            "Capture in the lighting that actually exists — overhead fluorescent, "
            "torch, flash, and backlit through the cabinet window. Do not "
            "correct or normalise it; that variation is the training signal.",
            "Include the panels that look bad: dusty, oil-filmed, with cable "
            "bundles crossing devices, faded labels, mixed manufacturers, "
            "retrofitted devices. Clean panels alone produce a fragile model.",
            "Photograph the same device family from several manufacturers. "
            "Manufacturer-invariance has to be learned from examples; there is "
            "no shortcut.",
            "Record the ground-truth bill of materials per panel from the as-built "
            "drawing. It validates counts and panel-type inference independently "
            "of the boxes, and costs almost nothing to collect at capture time.",
        ],
        "labelling_rules": [
            "One box per physical device, tight to the housing including "
            "terminals but excluding wiring.",
            "An overload relay bolted under a contactor is TWO boxes, not one — "
            "they are separately replaceable devices and separately reported.",
            "Terminal blocks: label contiguous strips as one box per strip, not "
            "per pole. Per-pole labelling produces hundreds of boxes per image "
            "and destroys the class balance.",
            "Structural items (DIN rail, cable duct, busbar) get one box per "
            "continuous run.",
            "If a labeller cannot identify a device with certainty, label it "
            f"'{tax.UNKNOWN_COMPONENT_ID}'. A wrong label is worse than an "
            "honest unknown, and the unknowns become the next capture list.",
            "Two labellers on a 10% sample; measure agreement. Below ~0.85 IoU "
            "agreement the labelling guide needs work, not the model.",
        ],
        "split_policy": (
            "Split by PANEL, never by image. Multiple framings of the same "
            "cabinet in both train and val leaks and inflates every metric — "
            "this is the most common way an industrial detector is reported as "
            "excellent and then fails on site."),
        "multiplication": (
            "Crop every labelled device into a per-class crop library, then use "
            "training.electrical.synthetic.compose_from_crops to multiply it "
            "with real appearance and synthetic arrangement, lighting, "
            "perspective, occlusion and dirt."),
        "acceptance": (
            "A held-out set of complete Madkour panels never seen in training, "
            "scored with rtsp_backend.electrical.metrics: per-class recall and "
            "precision, plus bill-of-materials count accuracy against the "
            "as-built drawing."),
    }


__all__ = [
    "DatasetSource", "SOURCES", "SOURCE_INDEX", "plan", "read_yolo_names",
    "build_index_map", "remap_yolo_dataset", "merge", "write_dataset_yaml",
    "analyse_dataset", "coverage_report", "custom_collection_plan",
    "MIN_INSTANCES_TRAINABLE", "MIN_INSTANCES_RELIABLE",
]
