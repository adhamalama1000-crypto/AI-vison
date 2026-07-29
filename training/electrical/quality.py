"""
Dataset quality inspection — the checks that decide whether training is worth starting.

:mod:`training.electrical.dedup` handles duplicates. This handles everything else that
makes a merged dataset quietly unfit: files that will not decode, labels that will not
parse, boxes outside the image, images too blurred or too dark for a device to be
identifiable in the first place, and a class balance so skewed that the loss is
dominated by one class.

The reason to run this *before* training rather than debug it after: every one of these
problems is silent. YOLO trainers skip an unreadable image with a warning nobody reads,
clamp an out-of-range box without comment, and train perfectly happily on a label file
whose class index does not exist in the label space. The result is a model that is worse
than it should be for reasons that never appear in the metrics.

Severity model
--------------
``fatal``
    The file cannot be used at all — unreadable image, unparseable label, class index
    outside the label space. These are quarantined by :func:`clean`.
``warning``
    Usable but suspect — very small, very blurred, very dark or blown-out, extreme
    aspect ratio, a box covering almost the whole frame. Kept by default, because
    "low quality" is a judgement about the deployment and a dim photograph of a real
    panel is exactly the input the model must survive.
``info``
    Worth knowing — an unlabelled image (a legitimate negative), a tiny box.

The blur and exposure thresholds are deliberately permissive. Field panel photography
is *badly lit by nature*: torch-lit, backlit through a cabinet window, flash-blown on
one side. Filtering aggressively on image statistics would throw away the hardest and
most valuable training examples and leave a model that only works in a showroom. These
checks exist to catch a lens cap or a motion-blurred write-off, not to curate.

Nothing is deleted. :func:`clean` writes a new dataset and moves rejects to a
``quarantine/`` directory with a reason file, so a wrong threshold costs a re-run.
"""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from rtsp_backend.electrical import taxonomy as tax

from . import datasets as ds

#: Variance of the Laplacian below which an image is considered badly blurred.
#: Deliberately low — a genuinely sharp panel photograph scores in the hundreds, and
#: this catches write-offs rather than merely soft images.
BLUR_VARIANCE_FLOOR = 25.0

#: Mean luminance bounds. Outside these the frame is nearly black or nearly white and
#: no device is identifiable in it.
DARK_MEAN_FLOOR = 18.0
BRIGHT_MEAN_CEILING = 240.0

#: Fraction of pixels at the extremes above which the frame is clipped past use.
CLIPPED_FRACTION_CEILING = 0.75

#: Smallest usable image edge, pixels. Below this a modular device is a few pixels
#: across even at row framing.
MIN_EDGE_PX = 160

#: Image aspect ratios outside this band are usually a stitched panorama or a crop
#: strip rather than a photograph, and letterboxing them wastes most of the tensor.
ASPECT_BAND = (0.25, 4.0)

#: A single box covering more than this fraction of the frame is usually a
#: whole-cabinet box mistakenly labelled as a device.
BOX_AREA_CEILING = 0.85
#: ...and below this it is a handful of pixels and cannot train anything.
BOX_AREA_FLOOR = 1e-5

#: Class-balance ratio (most common : least common, among present classes) above
#: which the loss is dominated by the majority classes.
IMBALANCE_RATIO_WARN = 20.0


@dataclass
class Issue:
    split: str
    filename: str
    code: str
    severity: str            # fatal | warning | info
    detail: str

    def to_dict(self) -> dict:
        return {"split": self.split, "filename": self.filename,
                "code": self.code, "severity": self.severity,
                "detail": self.detail}


@dataclass
class QualityReport:
    root: str
    images: int = 0
    labels: int = 0
    instances: int = 0
    issues: list = field(default_factory=list)
    per_code: dict = field(default_factory=dict)
    per_severity: dict = field(default_factory=dict)
    fatal_files: list = field(default_factory=list)
    class_balance: dict = field(default_factory=dict)
    verdict: str = ""
    recommendations: list = field(default_factory=list)

    def to_dict(self, max_issues: int = 500) -> dict:
        return {
            "root": self.root,
            "images": self.images,
            "labels": self.labels,
            "instances": self.instances,
            "per_severity": self.per_severity,
            "per_code": self.per_code,
            "fatal_files": self.fatal_files[:max_issues],
            "fatal_count": len(self.fatal_files),
            "issues": [i.to_dict() for i in self.issues[:max_issues]],
            "issues_truncated": max(0, len(self.issues) - max_issues),
            "class_balance": self.class_balance,
            "verdict": self.verdict,
            "recommendations": list(self.recommendations),
            "thresholds": {
                "blur_variance_floor": BLUR_VARIANCE_FLOOR,
                "dark_mean_floor": DARK_MEAN_FLOOR,
                "bright_mean_ceiling": BRIGHT_MEAN_CEILING,
                "min_edge_px": MIN_EDGE_PX,
                "box_area_ceiling": BOX_AREA_CEILING,
                "imbalance_ratio_warn": IMBALANCE_RATIO_WARN,
            },
        }


# --------------------------------------------------------------------------
# per-file checks
# --------------------------------------------------------------------------

def check_image(path: str) -> tuple[Optional[tuple], list[tuple]]:
    """Inspect one image. Returns ``((h, w), issues)`` or ``(None, issues)``.

    ``None`` dimensions mean the file is unusable; the caller quarantines it.
    """
    import cv2

    issues: list[tuple] = []
    if os.path.getsize(path) == 0:
        return None, [("empty_file", "fatal", "file is zero bytes")]

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None, [("unreadable", "fatal",
                       "cv2 could not decode this file — truncated download or "
                       "wrong extension")]
    h, w = img.shape[:2]
    if h < 2 or w < 2:
        return None, [("degenerate", "fatal", f"{w}x{h} is not an image")]

    if min(h, w) < MIN_EDGE_PX:
        issues.append(("small_image", "warning",
                       f"{w}x{h}: shortest edge under {MIN_EDGE_PX}px, so a modular "
                       f"device is only a few pixels across"))
    aspect = w / h
    if not (ASPECT_BAND[0] <= aspect <= ASPECT_BAND[1]):
        issues.append(("extreme_aspect", "warning",
                       f"aspect {aspect:.2f} outside {ASPECT_BAND} — probably a "
                       f"panorama or crop strip; letterboxing wastes the tensor"))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    if mean < DARK_MEAN_FLOOR:
        issues.append(("too_dark", "warning",
                       f"mean luminance {mean:.1f} — nearly black; no device is "
                       f"identifiable"))
    elif mean > BRIGHT_MEAN_CEILING:
        issues.append(("too_bright", "warning",
                       f"mean luminance {mean:.1f} — blown out"))

    clipped = float(((gray <= 2) | (gray >= 253)).mean())
    if clipped > CLIPPED_FRACTION_CEILING:
        issues.append(("clipped", "warning",
                       f"{clipped:.0%} of pixels at the extremes — the frame is "
                       f"clipped past use"))

    # Variance of the Laplacian: the standard sharpness proxy. Low variance means
    # no edges, which for a panel full of rectangular devices means motion blur or
    # a defocused lens.
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur < BLUR_VARIANCE_FLOOR:
        issues.append(("blurred", "warning",
                       f"Laplacian variance {blur:.1f} — badly blurred or "
                       f"defocused"))

    if float(gray.std()) < 3.0:
        issues.append(("featureless", "warning",
                       "almost no pixel variation — lens cap, blank wall, or a "
                       "solid-colour frame"))
    return (h, w), issues


def check_labels(path: str, class_count: int) -> tuple[list[tuple], list[tuple]]:
    """Parse one YOLO label file. Returns ``(rows, issues)``.

    ``rows`` is ``[(class_index, cx, cy, w, h), ...]`` for the rows that parsed.
    """
    issues: list[tuple] = []
    if not os.path.exists(path):
        return [], [("no_label_file", "info",
                     "no label file — treated as a negative example, which is "
                     "legitimate, but check it was intended")]
    rows: list[tuple] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except (OSError, UnicodeDecodeError) as exc:
        return [], [("label_unreadable", "fatal", f"{exc}")]

    non_empty = [ln for ln in lines if ln.strip()]
    if not non_empty:
        issues.append(("empty_annotation", "info",
                       "label file exists but is empty — a negative example"))

    for n, line in enumerate(non_empty, 1):
        parts = line.split()
        if len(parts) < 5:
            issues.append(("malformed_row", "fatal",
                           f"line {n}: expected 5 fields, got {len(parts)}"))
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
        except ValueError:
            issues.append(("unparseable_row", "fatal",
                           f"line {n}: non-numeric values"))
            continue
        if not (0 <= cls < class_count):
            # This is the one that silently ruins a model: the trainer accepts it
            # and the class index means nothing.
            issues.append(("class_out_of_range", "fatal",
                           f"line {n}: class index {cls} outside the "
                           f"{class_count}-class label space"))
            continue
        if not all(0.0 <= v <= 1.0 for v in (cx, cy)):
            issues.append(("centre_out_of_range", "fatal",
                           f"line {n}: centre ({cx:.3f}, {cy:.3f}) outside [0,1] — "
                           f"labels are not normalised"))
            continue
        if bw <= 0 or bh <= 0:
            issues.append(("degenerate_box", "fatal",
                           f"line {n}: box {bw:.4f}x{bh:.4f} has no area"))
            continue
        if bw > 1.0 or bh > 1.0:
            issues.append(("box_exceeds_image", "fatal",
                           f"line {n}: box {bw:.3f}x{bh:.3f} is larger than the "
                           f"image"))
            continue

        area = bw * bh
        if area > BOX_AREA_CEILING:
            issues.append(("box_covers_frame", "warning",
                           f"line {n}: box covers {area:.0%} of the frame — usually "
                           f"a whole-cabinet box mislabelled as a device"))
        elif area < BOX_AREA_FLOOR:
            issues.append(("box_too_small", "info",
                           f"line {n}: box covers {area:.2e} of the frame — too few "
                           f"pixels to learn from"))
        # A box whose edges fall outside the frame was drawn on a different crop.
        if (cx - bw / 2) < -0.01 or (cx + bw / 2) > 1.01 \
                or (cy - bh / 2) < -0.01 or (cy + bh / 2) > 1.01:
            issues.append(("box_outside_frame", "warning",
                           f"line {n}: box extends past the image edge — it was "
                           f"probably drawn on a different crop of this image"))
        rows.append((cls, cx, cy, bw, bh))
    return rows, issues


# --------------------------------------------------------------------------
# dataset scan
# --------------------------------------------------------------------------

def inspect(root: str, splits: Sequence[str] = ("train", "val", "test"),
            classes: Sequence[str] = tax.CLASS_ORDER,
            check_pixels: bool = True,
            log: Optional[Callable[[str], None]] = None) -> QualityReport:
    """Inspect every image and label in a YOLO dataset.

    ``check_pixels=False`` skips decoding (fast structural check only) — useful on a
    very large dataset where you only want the label problems.
    """
    say = log or (lambda m: None)
    report = QualityReport(root=root)
    inv = {i: c for i, c in enumerate(classes)}
    per_class: Counter = Counter()
    images_with: defaultdict[str, set] = defaultdict(set)
    fatal: set[str] = set()

    for split in splits:
        img_dir = os.path.join(root, "images", split)
        if not os.path.isdir(img_dir):
            continue
        files = [f for f in sorted(os.listdir(img_dir))
                 if f.lower().endswith(ds.IMAGE_EXTS)]
        say(f"{split}: {len(files)} image(s)")
        for n, fn in enumerate(files, 1):
            stem = os.path.splitext(fn)[0]
            img_path = os.path.join(img_dir, fn)
            report.images += 1

            dims: Optional[tuple] = None
            if check_pixels:
                dims, img_issues = check_image(img_path)
            else:
                img_issues = []
            for code, severity, detail in img_issues:
                report.issues.append(Issue(split, fn, code, severity, detail))
                if severity == "fatal":
                    fatal.add(f"{split}/{fn}")
            if check_pixels and dims is None:
                continue                     # unusable; label check is moot

            lbl_path = os.path.join(root, "labels", split, stem + ".txt")
            if os.path.exists(lbl_path):
                report.labels += 1
            rows, lbl_issues = check_labels(lbl_path, len(classes))
            for code, severity, detail in lbl_issues:
                report.issues.append(Issue(split, fn, code, severity, detail))
                if severity == "fatal":
                    fatal.add(f"{split}/{fn}")
            report.instances += len(rows)
            for cls, *_ in rows:
                cid = inv.get(cls)
                if cid:
                    per_class[cid] += 1
                    images_with[cid].add(f"{split}/{fn}")
            if n % 500 == 0:
                say(f"  {split}: {n}/{len(files)}")

    report.fatal_files = sorted(fatal)
    report.per_code = dict(Counter(i.code for i in report.issues).most_common())
    report.per_severity = dict(Counter(i.severity for i in report.issues))

    # -- class balance ---------------------------------------------------
    if per_class:
        counts = sorted(per_class.items(), key=lambda kv: -kv[1])
        most, least = counts[0][1], counts[-1][1]
        ratio = most / max(least, 1)
        report.class_balance = {
            "classes_present": len(counts),
            "classes_absent": len(classes) - len(counts),
            "instances_per_class": dict(counts),
            "images_per_class": {c: len(images_with[c]) for c, _ in counts},
            "most_common": {"class_id": counts[0][0], "instances": most},
            "least_common": {"class_id": counts[-1][0], "instances": least},
            "imbalance_ratio": round(ratio, 2),
            # >= so that a ratio of exactly IMBALANCE_RATIO_WARN fires. A threshold
            # named "warn at 20" that stays silent at 20:1 is a trap.
            "imbalanced": ratio >= IMBALANCE_RATIO_WARN,
        }
    else:
        report.class_balance = {"classes_present": 0,
                                "classes_absent": len(classes),
                                "instances_per_class": {}}

    report.verdict, report.recommendations = _verdict(report)
    say(report.verdict)
    return report


def _verdict(report: QualityReport) -> tuple[str, list[str]]:
    fatal_n = len(report.fatal_files)
    warn_n = report.per_severity.get("warning", 0)
    parts: list[str] = []
    recs: list[str] = []

    if not report.images:
        return ("No images found — nothing to inspect.",
                ["Check the dataset root; expected images/<split>/."])

    if fatal_n:
        parts.append(
            f"{fatal_n} of {report.images} file(s) ({fatal_n / report.images:.1%}) "
            f"are UNUSABLE and must be removed before training.")
        recs.append(
            "Run with --dst to write a cleaned dataset; rejects are moved to "
            "quarantine/ with a reason file, not deleted.")
    else:
        parts.append(f"All {report.images} file(s) are structurally usable.")

    if report.per_code.get("class_out_of_range"):
        recs.append(
            f"{report.per_code['class_out_of_range']} label row(s) reference a class "
            f"index outside the label space. This is the failure that silently ruins "
            f"a model — the trainer accepts it and the index means nothing. It "
            f"almost always means a dataset was merged without remapping; re-run "
            f"'cli remap' on the offending source.")
    if report.per_code.get("centre_out_of_range"):
        recs.append(
            "Some labels are not normalised to [0,1] — they are probably in absolute "
            "pixels. Convert them before merging.")
    if report.per_code.get("box_covers_frame"):
        recs.append(
            f"{report.per_code['box_covers_frame']} box(es) cover most of the frame. "
            f"Check whether a whole-cabinet box was labelled as a device; that "
            f"teaches the model that any large region is a component.")
    if report.per_code.get("no_label_file", 0) + \
            report.per_code.get("empty_annotation", 0) > report.images * 0.3:
        recs.append(
            "Over 30% of images have no annotations. If those are intentional "
            "negatives, cap them at roughly 10% of the training set — beyond that "
            "they suppress genuine detections. If they are not intentional, the "
            "labels did not come across in the merge.")

    if warn_n:
        parts.append(
            f"{warn_n} quality warning(s) — kept by default, because a dim or soft "
            f"photograph of a real panel is exactly the input the model must "
            f"survive. Review them before filtering.")

    balance = report.class_balance
    if balance.get("imbalanced"):
        parts.append(
            f"Class balance is skewed {balance['imbalance_ratio']:.0f}:1 "
            f"({balance['most_common']['class_id']} "
            f"{balance['most_common']['instances']} vs "
            f"{balance['least_common']['class_id']} "
            f"{balance['least_common']['instances']}).")
        recs.append(
            f"With a {balance['imbalance_ratio']:.0f}:1 imbalance the loss is "
            f"dominated by the majority classes and the rare ones will barely "
            f"train. Collect more of the rare classes rather than downsampling the "
            f"common ones — you need the instances, not a prettier ratio.")
    if balance.get("classes_absent"):
        recs.append(
            f"{balance['classes_absent']} taxonomy class(es) have no instances at "
            f"all. Run 'cli gap' for the exact shortfall and where to capture each.")

    return " ".join(parts), recs


# --------------------------------------------------------------------------
# cleaning
# --------------------------------------------------------------------------

def clean(src_root: str, dst_root: str,
          splits: Sequence[str] = ("train", "val", "test"),
          classes: Sequence[str] = tax.CLASS_ORDER,
          drop_warnings: Sequence[str] = (),
          quarantine: bool = True,
          log: Optional[Callable[[str], None]] = None) -> dict:
    """Write a cleaned dataset, quarantining what cannot be used.

    Fatal files are always excluded. ``drop_warnings`` additionally excludes files
    carrying the named warning codes (e.g. ``("blurred", "too_dark")``) — off by
    default, deliberately: filtering on image statistics throws away the hardest and
    most valuable field examples and leaves a model that only works in a showroom.

    Nothing is deleted from the source. Rejects are copied to
    ``<dst_root>/quarantine/<split>/`` with a ``_reasons.json`` beside them.
    """
    say = log or (lambda m: None)
    report = inspect(src_root, splits, classes, log=say)

    by_file: defaultdict[str, list[Issue]] = defaultdict(list)
    for issue in report.issues:
        by_file[f"{issue.split}/{issue.filename}"].append(issue)

    drop_set = set(drop_warnings)
    rejected: dict[str, list[dict]] = {}
    for key, issues in by_file.items():
        reasons = [i for i in issues
                   if i.severity == "fatal" or i.code in drop_set]
        if reasons:
            rejected[key] = [i.to_dict() for i in reasons]

    for split in splits:
        os.makedirs(os.path.join(dst_root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(dst_root, "labels", split), exist_ok=True)

    kept: Counter = Counter()
    dropped: Counter = Counter()
    for split in splits:
        img_dir = os.path.join(src_root, "images", split)
        if not os.path.isdir(img_dir):
            continue
        for fn in sorted(os.listdir(img_dir)):
            if not fn.lower().endswith(ds.IMAGE_EXTS):
                continue
            key = f"{split}/{fn}"
            stem = os.path.splitext(fn)[0]
            s_img = os.path.join(img_dir, fn)
            s_lbl = os.path.join(src_root, "labels", split, stem + ".txt")
            if key in rejected:
                dropped[split] += 1
                if quarantine:
                    q_dir = os.path.join(dst_root, "quarantine", split)
                    os.makedirs(q_dir, exist_ok=True)
                    try:
                        shutil.copy2(s_img, os.path.join(q_dir, fn))
                    except OSError:
                        pass            # an unreadable file may also be uncopyable
                    if os.path.exists(s_lbl):
                        shutil.copy2(s_lbl, os.path.join(q_dir, stem + ".txt"))
                continue
            shutil.copy2(s_img, os.path.join(dst_root, "images", split, fn))
            d_lbl = os.path.join(dst_root, "labels", split, stem + ".txt")
            if os.path.exists(s_lbl):
                shutil.copy2(s_lbl, d_lbl)
            else:
                open(d_lbl, "w", encoding="utf-8").close()
            kept[split] += 1

    if quarantine and rejected:
        q_root = os.path.join(dst_root, "quarantine")
        os.makedirs(q_root, exist_ok=True)
        with open(os.path.join(q_root, "_reasons.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(rejected, fh, indent=2)

    ds.write_dataset_yaml(dst_root)
    out = {
        "status": "cleaned",
        "src_root": src_root, "dst_root": dst_root,
        "images_in": report.images,
        "images_kept": int(sum(kept.values())),
        "images_dropped": int(sum(dropped.values())),
        "kept_per_split": dict(kept),
        "dropped_per_split": dict(dropped),
        "dropped_warning_codes": sorted(drop_set),
        "quarantine": (os.path.join(dst_root, "quarantine")
                       if quarantine and rejected else None),
        "quality_report": report.to_dict(),
    }
    say(f"kept {out['images_kept']}, dropped {out['images_dropped']}")
    if not drop_set:
        say("only unusable files were dropped; quality warnings were kept. Pass "
            "--drop-warnings <code>... to filter on image statistics too, but "
            "consider that a dim panel photograph is real deployment input.")
    return out


__all__ = [
    "BLUR_VARIANCE_FLOOR", "DARK_MEAN_FLOOR", "BRIGHT_MEAN_CEILING",
    "MIN_EDGE_PX", "ASPECT_BAND", "BOX_AREA_CEILING", "BOX_AREA_FLOOR",
    "IMBALANCE_RATIO_WARN", "Issue", "QualityReport", "check_image",
    "check_labels", "inspect", "clean",
]
