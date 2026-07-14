"""
Dataset ingestion, type detection, and validation (Part 2).

Given a directory of uploaded/extracted data this module figures out what kind
of dataset it is (YOLO / COCO / Pascal VOC / image-classification folders /
loose images / videos / mixed) and runs a real validation pass over it:
missing labels, corrupt/unreadable images, class distribution and imbalance,
plus a human-readable summary. Nothing is fabricated — every count comes from
walking the actual files.
"""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from typing import Optional

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #

def safe_extract_zip(zip_path: str, dest_dir: str) -> None:
    """Extract a zip, guarding against path-traversal (zip-slip)."""
    os.makedirs(dest_dir, exist_ok=True)
    base = os.path.abspath(dest_dir)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            target = os.path.abspath(os.path.join(dest_dir, member))
            if not (target == base or target.startswith(base + os.sep)):
                raise ValueError(f"unsafe path in archive: {member}")
        zf.extractall(dest_dir)


def _walk_files(root: str) -> list[str]:
    out: list[str] = []
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            out.append(os.path.join(dirpath, f))
    return out


def _ext(path: str) -> str:
    return os.path.splitext(path)[1].lower()


# --------------------------------------------------------------------------- #
# type detection
# --------------------------------------------------------------------------- #

def detect_kind(root: str) -> str:
    files = _walk_files(root)
    lower = [f.lower() for f in files]
    imgs = [f for f in files if _ext(f) in IMAGE_EXTS]
    vids = [f for f in files if _ext(f) in VIDEO_EXTS]

    # COCO: a json file with the coco keys
    for f in files:
        if _ext(f) == ".json":
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    obj = json.load(fh)
                if isinstance(obj, dict) and "images" in obj and "annotations" in obj:
                    return "coco"
            except Exception:
                pass

    # YOLO: data.yaml / dataset.yaml, or images + parallel .txt labels
    if any(os.path.basename(l) in ("data.yaml", "dataset.yaml", "data.yml") for l in lower):
        return "yolo"
    txts = [f for f in files if _ext(f) == ".txt"
            and os.path.basename(f).lower() not in ("classes.txt", "readme.txt")]
    if imgs and txts and _looks_like_yolo(imgs, txts):
        return "yolo"

    # Pascal VOC: .xml annotation files with <annotation><object>
    xmls = [f for f in files if _ext(f) == ".xml"]
    if xmls and _looks_like_voc(xmls):
        return "voc"

    # classification: images grouped into class subfolders directly under root
    if imgs and _looks_like_classification(root):
        return "classification"

    if imgs and not vids:
        return "images"
    if vids and not imgs:
        return "videos"
    if imgs and vids:
        return "mixed"
    return "unknown"


def _stem_set(paths: list[str]) -> set[str]:
    return {os.path.splitext(os.path.basename(p))[0] for p in paths}


def _looks_like_yolo(imgs: list[str], txts: list[str]) -> bool:
    img_stems = _stem_set(imgs)
    txt_stems = _stem_set(txts)
    return len(img_stems & txt_stems) >= max(1, int(0.3 * len(img_stems)))


def _looks_like_voc(xmls: list[str]) -> bool:
    for x in xmls[:20]:
        try:
            root = ET.parse(x).getroot()
            if root.tag == "annotation" and root.find("object") is not None:
                return True
        except Exception:
            continue
    return False


def _looks_like_classification(root: str) -> bool:
    # each immediate subdirectory holds images -> folder name is the class
    try:
        subdirs = [d for d in os.listdir(root)
                   if os.path.isdir(os.path.join(root, d))]
    except OSError:
        return False
    if len(subdirs) < 2:
        return False
    class_like = 0
    for d in subdirs:
        p = os.path.join(root, d)
        has_img = any(_ext(f) in IMAGE_EXTS for f in os.listdir(p)
                      if os.path.isfile(os.path.join(p, f)))
        if has_img:
            class_like += 1
    return class_like >= 2


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #

def _read_yolo_classes(root: str) -> Optional[list[str]]:
    for name in ("classes.txt", "obj.names"):
        for f in _walk_files(root):
            if os.path.basename(f).lower() == name:
                with open(f, "r", encoding="utf-8") as fh:
                    names = [ln.strip() for ln in fh if ln.strip()]
                if names:
                    return names
    # data.yaml names:
    for f in _walk_files(root):
        if os.path.basename(f).lower() in ("data.yaml", "dataset.yaml", "data.yml"):
            try:
                import yaml
                with open(f, "r", encoding="utf-8") as fh:
                    obj = yaml.safe_load(fh) or {}
                names = obj.get("names")
                if isinstance(names, dict):
                    return [names[k] for k in sorted(names, key=lambda x: int(x))]
                if isinstance(names, list):
                    return names
            except Exception:
                pass
    return None


def validate(root: str, kind: Optional[str] = None) -> dict:
    kind = kind or detect_kind(root)
    files = _walk_files(root)
    imgs = [f for f in files if _ext(f) in IMAGE_EXTS]
    vids = [f for f in files if _ext(f) in VIDEO_EXTS]

    report: dict = {
        "kind": kind,
        "n_images": len(imgs),
        "n_videos": len(vids),
        "n_labels": 0,
        "classes": [],
        "class_counts": {},
        "missing_labels": [],
        "corrupt_images": [],
        "warnings": [],
        "errors": [],
    }

    # corrupt-image scan (cap the number decoded so huge sets stay responsive)
    scan = imgs[:2000]
    for p in scan:
        img = cv2.imread(p)
        if img is None or getattr(img, "size", 0) == 0:
            report["corrupt_images"].append(os.path.relpath(p, root))
    if len(imgs) > len(scan):
        report["warnings"].append(
            f"only scanned {len(scan)}/{len(imgs)} images for corruption")

    class_counts: Counter = Counter()

    if kind == "yolo":
        names = _read_yolo_classes(root)
        txts = {os.path.splitext(os.path.basename(f))[0]: f for f in files
                if _ext(f) == ".txt"
                and os.path.basename(f).lower() not in ("classes.txt", "readme.txt")}
        labelled = 0
        for p in imgs:
            stem = os.path.splitext(os.path.basename(p))[0]
            lp = txts.get(stem)
            if not lp:
                report["missing_labels"].append(os.path.relpath(p, root))
                continue
            labelled += 1
            try:
                with open(lp, "r", encoding="utf-8") as fh:
                    for ln in fh:
                        parts = ln.split()
                        if parts:
                            class_counts[int(float(parts[0]))] += 1
            except Exception:
                report["warnings"].append(f"unreadable label: {os.path.basename(lp)}")
        report["n_labels"] = labelled
        if names:
            report["classes"] = names
            report["class_counts"] = {
                (names[i] if i < len(names) else str(i)): c
                for i, c in sorted(class_counts.items())}
        else:
            report["class_counts"] = {str(i): c for i, c in sorted(class_counts.items())}
            report["classes"] = list(report["class_counts"].keys())

    elif kind == "coco":
        for f in files:
            if _ext(f) != ".json":
                continue
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    obj = json.load(fh)
            except Exception:
                continue
            if not (isinstance(obj, dict) and "annotations" in obj):
                continue
            cats = {c["id"]: c.get("name", str(c["id"]))
                    for c in obj.get("categories", [])}
            report["classes"] = list(cats.values())
            report["n_labels"] = len(obj.get("annotations", []))
            for a in obj.get("annotations", []):
                class_counts[cats.get(a.get("category_id"), str(a.get("category_id")))] += 1
            report["class_counts"] = dict(class_counts)
            break

    elif kind == "voc":
        xmls = [f for f in files if _ext(f) == ".xml"]
        report["n_labels"] = len(xmls)
        img_stems = _stem_set(imgs)
        for x in xmls:
            try:
                r = ET.parse(x).getroot()
                for obj in r.findall("object"):
                    name = obj.findtext("name", default="unknown")
                    class_counts[name] += 1
            except Exception:
                report["warnings"].append(f"unreadable xml: {os.path.basename(x)}")
        xml_stems = _stem_set(xmls)
        for p in imgs:
            if os.path.splitext(os.path.basename(p))[0] not in xml_stems:
                report["missing_labels"].append(os.path.relpath(p, root))
        report["class_counts"] = dict(class_counts)
        report["classes"] = list(class_counts.keys())

    elif kind == "classification":
        subdirs = [d for d in os.listdir(root)
                   if os.path.isdir(os.path.join(root, d))]
        for d in sorted(subdirs):
            p = os.path.join(root, d)
            n = sum(1 for f in _walk_files(p) if _ext(f) in IMAGE_EXTS)
            if n:
                class_counts[d] = n
        report["classes"] = list(class_counts.keys())
        report["class_counts"] = dict(class_counts)
        report["n_labels"] = sum(class_counts.values())

    # class imbalance
    counts = list(report["class_counts"].values())
    if counts:
        mx, mn = max(counts), max(1, min(counts))
        ratio = round(mx / mn, 2)
        report["imbalance_ratio"] = ratio
        if ratio >= 10:
            report["warnings"].append(
                f"severe class imbalance (ratio {ratio}:1) — consider balancing/augmentation")
        elif ratio >= 3:
            report["warnings"].append(f"class imbalance (ratio {ratio}:1)")
    else:
        report["imbalance_ratio"] = None

    if report["missing_labels"]:
        report["warnings"].append(
            f"{len(report['missing_labels'])} image(s) have no label")
    if report["corrupt_images"]:
        report["errors"].append(
            f"{len(report['corrupt_images'])} corrupt/unreadable image(s)")

    # verdict
    report["n_classes"] = len(report["classes"])
    if report["n_images"] == 0 and report["n_videos"] == 0:
        report["ok"] = False
        report["errors"].append("no images or videos found")
    elif kind in ("yolo", "coco", "voc") and report["n_labels"] == 0:
        report["ok"] = False
        report["errors"].append("annotation dataset but no labels parsed")
    else:
        report["ok"] = len(report["errors"]) == 0
    # trim long lists for storage
    report["missing_labels"] = report["missing_labels"][:100]
    report["corrupt_images"] = report["corrupt_images"][:100]
    return report
