"""
Dataset acquisition — the part that actually fetches bytes.

:mod:`training.electrical.datasets` says *what* to get and how each source's
labels map onto the canonical taxonomy. This module does the fetching, the
layout normalisation and the remapping, so that

    python -m training.electrical.cli download --all --dst data/raw

leaves you with one canonically-labelled YOLO dataset per source, ready for
``merge``.

Three fetchers, all real:

``roboflow``
    Downloads a generated version through the Roboflow export API. Uses the
    ``roboflow`` SDK when it is installed and falls back to the REST export
    endpoint with ``requests`` otherwise, so the only hard requirement is a
    ``ROBOFLOW_API_KEY``.
``kaggle``
    Shells out to the ``kaggle`` CLI, which is the only supported way to
    authenticate against Kaggle. Reports the exact ``~/.kaggle/kaggle.json``
    remedy when credentials are missing.
``url``
    Plain HTTPS download of a zip/tar archive.

Everything here fails loudly and specifically. A missing API key, an
un-versioned Roboflow project, an export format the upstream project does not
publish — each produces a named error with the fix, never a silent empty
dataset that looks like a successful download until training reports 0 images.

Layout normalisation
--------------------
Public YOLO exports use ``<split>/images`` + ``<split>/labels`` with a
``data.yaml``; this project's canonical layout is ``images/<split>`` +
``labels/<split>`` with ``dataset.yaml``. :func:`normalise_yolo_layout` converts
between them and canonicalises Roboflow's ``valid`` to ``val``. It also handles
flat datasets (all images in one directory) by treating them as a single
``train`` split for the splitter to divide later.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
import zipfile
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from . import datasets as ds

#: Roboflow export formats to try, in order. All three emit the same YOLO
#: ``class cx cy w h`` text format; they differ only in the ``data.yaml`` and in
#: which of them a given upstream project has published. Trying several is the
#: difference between "works" and "404 on half the registry".
ROBOFLOW_FORMATS: tuple[str, ...] = ("yolov11", "yolov9", "yolov8",
                                     "yolov5pytorch")

#: Roboflow's split directory names → ours.
SPLIT_ALIASES = {"train": "train", "valid": "val", "validation": "val",
                 "val": "val", "test": "test"}


class DownloadError(RuntimeError):
    """A fetch failed for a reason the caller should be told verbatim."""


@dataclass
class DownloadResult:
    key: str
    status: str                      # downloaded | skipped | failed
    reason: Optional[str] = None
    raw_dir: Optional[str] = None
    dataset_dir: Optional[str] = None
    source_names: list[str] = field(default_factory=list)
    remap_stats: dict = field(default_factory=dict)
    licence: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "key": self.key, "status": self.status, "reason": self.reason,
            "raw_dir": self.raw_dir, "dataset_dir": self.dataset_dir,
            "source_classes": self.source_names,
            "remap": self.remap_stats, "licence": self.licence,
        }


# --------------------------------------------------------------------------
# archive helpers
# --------------------------------------------------------------------------

def _extract(archive: str, dest: str) -> None:
    os.makedirs(dest, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            _safe_extract_zip(zf, dest)
        return
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            _safe_extract_tar(tf, dest)
        return
    raise DownloadError(f"{archive} is neither a zip nor a tar archive")


def _is_within(base: str, target: str) -> bool:
    base = os.path.abspath(base)
    target = os.path.abspath(target)
    return target == base or target.startswith(base + os.sep)


def _safe_extract_zip(zf: zipfile.ZipFile, dest: str) -> None:
    # Archive members are attacker-controlled in the general case (these are
    # third-party downloads), so absolute paths and ../ traversal are rejected
    # rather than trusted.
    for member in zf.namelist():
        if not _is_within(dest, os.path.join(dest, member)):
            raise DownloadError(f"unsafe path in archive: {member}")
    zf.extractall(dest)


def _safe_extract_tar(tf: tarfile.TarFile, dest: str) -> None:
    for member in tf.getmembers():
        if member.issym() or member.islnk():
            raise DownloadError(f"archive contains a link member: {member.name}")
        if not _is_within(dest, os.path.join(dest, member.name)):
            raise DownloadError(f"unsafe path in archive: {member.name}")
    tf.extractall(dest)


def _http_download(url: str, dest_path: str,
                   log: Optional[Callable[[str], None]] = None) -> str:
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - requests is a hard dep
        raise DownloadError(f"requests is required to download: {exc}") from exc

    say = log or (lambda m: None)
    say(f"GET {url.split('?')[0]}")
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as resp:
        if resp.status_code != 200:
            raise DownloadError(
                f"HTTP {resp.status_code} for {url.split('?')[0]}")
        total = int(resp.headers.get("content-length") or 0)
        done = 0
        with open(dest_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                fh.write(chunk)
                done += len(chunk)
                if total:
                    say(f"  {done / 1e6:.1f}/{total / 1e6:.1f} MB")
    return dest_path


# --------------------------------------------------------------------------
# fetchers
# --------------------------------------------------------------------------

def fetch_roboflow(locator: str, dest: str,
                   api_key: Optional[str] = None,
                   formats: Sequence[str] = ROBOFLOW_FORMATS,
                   log: Optional[Callable[[str], None]] = None) -> str:
    """Download a Roboflow Universe version into ``dest``.

    ``locator`` is ``workspace/project/version``. A two-part locator means the
    upstream project has no generated version, which cannot be downloaded — that
    is reported as such, with the fork-and-generate remedy, because it is a real
    and common state for community projects.
    """
    say = log or (lambda m: None)
    parts = [p for p in locator.split("/") if p]
    if len(parts) < 3:
        raise DownloadError(
            f"'{locator}' has no version component. The upstream Roboflow "
            f"project has no generated version, so there is nothing to "
            f"download. Fork it into your own workspace at "
            f"https://app.roboflow.com, generate a version (any preprocessing; "
            f"no augmentation needed — this pipeline augments at training "
            f"time), then set the locator to "
            f"'<your-workspace>/{parts[-1] if parts else 'project'}/1'.")
    workspace, project, version = parts[0], parts[1], parts[2]

    key = api_key or os.environ.get("ROBOFLOW_API_KEY") or ""
    if not key:
        raise DownloadError(
            "ROBOFLOW_API_KEY is not set. Create a free account at "
            "https://app.roboflow.com, copy the private API key from Settings "
            "→ API, and export it: export ROBOFLOW_API_KEY=...  (store it in a "
            "gitignored .env, never in the repository).")

    os.makedirs(dest, exist_ok=True)

    # Preferred path: the official SDK, which handles export generation waits.
    try:
        from roboflow import Roboflow  # type: ignore

        say(f"roboflow SDK: {workspace}/{project} v{version}")
        rf = Roboflow(api_key=key)
        proj = rf.workspace(workspace).project(project)
        ver = proj.version(int(version))
        last_exc: Optional[Exception] = None
        for fmt in formats:
            try:
                say(f"  export format '{fmt}'")
                ver.download(fmt, location=dest, overwrite=True)
                return dest
            except Exception as exc:
                last_exc = exc
                say(f"  '{fmt}' unavailable: {exc}")
        raise DownloadError(
            f"none of the export formats {list(formats)} are published for "
            f"{locator}: {last_exc}")
    except ImportError:
        say("roboflow SDK not installed — using the REST export API")

    # Fallback: the REST export endpoint. Same result, one fewer dependency.
    try:
        import requests
    except ImportError as exc:  # pragma: no cover
        raise DownloadError(
            "either the 'roboflow' SDK or 'requests' is required "
            f"({exc})") from exc

    errors: list[str] = []
    for fmt in formats:
        url = (f"https://api.roboflow.com/{workspace}/{project}/{version}/{fmt}"
               f"?api_key={key}")
        say(f"  export format '{fmt}'")
        try:
            resp = requests.get(url, timeout=120)
        except Exception as exc:
            errors.append(f"{fmt}: {exc}")
            continue
        if resp.status_code != 200:
            errors.append(f"{fmt}: HTTP {resp.status_code}")
            continue
        try:
            payload = resp.json()
        except ValueError:
            errors.append(f"{fmt}: response was not JSON")
            continue
        link = (payload.get("export") or {}).get("link")
        if not link:
            errors.append(f"{fmt}: no export link in response")
            continue
        archive = os.path.join(dest, f"{project}-{version}-{fmt}.zip")
        _http_download(link, archive, log=say)
        _extract(archive, dest)
        os.remove(archive)
        return dest

    raise DownloadError(
        f"could not export {locator} in any of {list(formats)}: "
        + "; ".join(errors)
        + ". If every format returned 401/403 the API key lacks access to this "
          "workspace; if they returned 404 the version does not exist.")


def fetch_kaggle(locator: str, dest: str,
                 log: Optional[Callable[[str], None]] = None) -> str:
    """Download a Kaggle dataset via the official CLI."""
    say = log or (lambda m: None)
    if "<" in locator:
        raise DownloadError(
            f"'{locator}' is a placeholder, not a dataset slug. Pass the real "
            f"one with --locator <owner>/<dataset-slug>. No Kaggle dataset was "
            f"verified to add boxed panel-device instances, which is why the "
            f"registry does not name one.")
    if shutil.which("kaggle") is None:
        raise DownloadError(
            "the 'kaggle' CLI is not installed: pip install kaggle, then place "
            "your API token at ~/.kaggle/kaggle.json (Kaggle → Account → "
            "Create New API Token) and chmod 600 it.")
    os.makedirs(dest, exist_ok=True)
    say(f"kaggle datasets download -d {locator}")
    proc = subprocess.run(
        ["kaggle", "datasets", "download", "-d", locator, "-p", dest,
         "--unzip"],
        capture_output=True, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        if "401" in err or "credentials" in err.lower():
            raise DownloadError(
                f"Kaggle authentication failed. Put a valid token at "
                f"~/.kaggle/kaggle.json and chmod 600 it. Raw error: {err}")
        raise DownloadError(f"kaggle CLI failed (exit {proc.returncode}): {err}")
    return dest


def fetch_url(locator: str, dest: str,
              log: Optional[Callable[[str], None]] = None) -> str:
    """Download and unpack an archive from a plain URL."""
    if "<" in locator or not locator.lower().startswith(("http://", "https://")):
        raise DownloadError(f"'{locator}' is not a downloadable URL")
    os.makedirs(dest, exist_ok=True)
    archive = os.path.join(dest, os.path.basename(locator.split("?")[0])
                           or "dataset.zip")
    _http_download(locator, archive, log=log)
    _extract(archive, dest)
    os.remove(archive)
    return dest


# --------------------------------------------------------------------------
# layout normalisation
# --------------------------------------------------------------------------

def find_dataset_root(root: str) -> Optional[str]:
    """Locate the directory that actually holds the YOLO tree.

    Archives commonly unpack into a single wrapper directory, sometimes nested
    twice. Rather than guessing, this walks down looking for the two shapes that
    matter: ``<split>/images`` (Roboflow) or ``images/<split>`` (ours).
    """
    for base, dirs, _files in os.walk(root):
        names = set(dirs)
        if names & set(SPLIT_ALIASES):
            for split in names & set(SPLIT_ALIASES):
                if os.path.isdir(os.path.join(base, split, "images")):
                    return base
        if "images" in names:
            img = os.path.join(base, "images")
            sub = set(os.listdir(img)) if os.path.isdir(img) else set()
            if sub & set(SPLIT_ALIASES):
                return base
            # flat: images/ holds image files directly
            if any(f.lower().endswith(ds.IMAGE_EXTS) for f in sub):
                return base
    return None


def find_data_yaml(root: str) -> Optional[str]:
    for name in ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml"):
        for base, _dirs, files in os.walk(root):
            if name in files:
                return os.path.join(base, name)
    return None


def normalise_yolo_layout(src_root: str, dst_root: str,
                          log: Optional[Callable[[str], None]] = None) -> dict:
    """Convert any common YOLO layout into ``images/<split>`` + ``labels/<split>``.

    An image with no corresponding label file gets an empty label file, not a
    dropped image: in detection, "this panel photograph contains no labelled
    devices" is a legitimate negative example, whereas silently discarding it
    changes the dataset behind your back. Images that appear to be unannotated
    en masse are reported in the returned ``images_without_labels`` count so the
    difference between "negatives" and "you downloaded an unlabelled set" stays
    visible.
    """
    say = log or (lambda m: None)
    base = find_dataset_root(src_root) or src_root
    stats = {"root": base, "splits": {}, "images": 0,
             "images_without_labels": 0}

    def copy_pair(img_dir: str, lbl_dir: str, split: str) -> int:
        d_img = os.path.join(dst_root, "images", split)
        d_lbl = os.path.join(dst_root, "labels", split)
        os.makedirs(d_img, exist_ok=True)
        os.makedirs(d_lbl, exist_ok=True)
        n = 0
        for fn in sorted(os.listdir(img_dir)):
            if not fn.lower().endswith(ds.IMAGE_EXTS):
                continue
            stem = os.path.splitext(fn)[0]
            shutil.copy2(os.path.join(img_dir, fn), os.path.join(d_img, fn))
            src_lbl = os.path.join(lbl_dir, stem + ".txt")
            dst_lbl = os.path.join(d_lbl, stem + ".txt")
            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, dst_lbl)
            else:
                open(dst_lbl, "w", encoding="utf-8").close()
                stats["images_without_labels"] += 1
            n += 1
        return n

    # Shape 1: <split>/images + <split>/labels  (Roboflow exports)
    found = False
    for raw_split, split in SPLIT_ALIASES.items():
        img_dir = os.path.join(base, raw_split, "images")
        lbl_dir = os.path.join(base, raw_split, "labels")
        if os.path.isdir(img_dir):
            n = copy_pair(img_dir, lbl_dir, split)
            stats["splits"][split] = stats["splits"].get(split, 0) + n
            stats["images"] += n
            found = True
            say(f"  {raw_split} -> {split}: {n} image(s)")

    # Shape 2: images/<split> + labels/<split>  (already ours)
    if not found and os.path.isdir(os.path.join(base, "images")):
        img_base = os.path.join(base, "images")
        entries = sorted(os.listdir(img_base))
        subsplits = [e for e in entries
                     if e in SPLIT_ALIASES
                     and os.path.isdir(os.path.join(img_base, e))]
        if subsplits:
            for raw_split in subsplits:
                split = SPLIT_ALIASES[raw_split]
                n = copy_pair(os.path.join(img_base, raw_split),
                              os.path.join(base, "labels", raw_split), split)
                stats["splits"][split] = stats["splits"].get(split, 0) + n
                stats["images"] += n
                found = True
                say(f"  {raw_split} -> {split}: {n} image(s)")
        elif any(f.lower().endswith(ds.IMAGE_EXTS) for f in entries):
            # Shape 3: flat. Everything becomes 'train'; the splitter divides it.
            n = copy_pair(img_base, os.path.join(base, "labels"), "train")
            stats["splits"]["train"] = n
            stats["images"] += n
            found = True
            say(f"  flat layout -> train: {n} image(s)")

    if not found:
        raise DownloadError(
            f"no recognisable YOLO layout under {src_root}. Expected "
            f"'<split>/images' or 'images/<split>' or a flat 'images/' "
            f"directory.")
    return stats


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------

def download_source(key: str, dst_root: str,
                    locator: Optional[str] = None,
                    api_key: Optional[str] = None,
                    remap: bool = True,
                    keep_raw: bool = False,
                    log: Optional[Callable[[str], None]] = None
                    ) -> DownloadResult:
    """Fetch one registry source, normalise it, and remap it onto the taxonomy.

    Returns a result object in every case — a failed download is reported, never
    raised, so a batch download of twelve sources is not aborted by one dead
    upstream project.
    """
    say = log or (lambda m: None)
    src = ds.SOURCE_INDEX.get(key)
    if src is None:
        return DownloadResult(key, "failed",
                              f"unknown source key '{key}'; known keys: "
                              f"{', '.join(sorted(ds.SOURCE_INDEX))}")
    if not src.usable:
        return DownloadResult(key, "skipped",
                              f"excluded by the registry: {src.excluded_reason}",
                              licence=src.licence)
    if src.kind == "manual":
        return DownloadResult(
            key, "skipped",
            "manual source — nothing to download. See "
            "training.electrical.datasets.custom_collection_plan().",
            licence=src.licence)

    loc = locator or src.locator
    raw_dir = os.path.join(dst_root, "_raw", key)
    out_dir = os.path.join(dst_root, key)
    say(f"=== {key} ({src.kind}: {loc}) ===")
    say(f"    licence: {src.licence}")

    try:
        if src.kind == "roboflow":
            fetch_roboflow(loc, raw_dir, api_key=api_key, log=say)
        elif src.kind == "kaggle":
            fetch_kaggle(loc, raw_dir, log=say)
        elif src.kind == "url":
            fetch_url(loc, raw_dir, log=say)
        else:
            return DownloadResult(key, "failed",
                                  f"no fetcher for kind '{src.kind}'")
    except DownloadError as exc:
        return DownloadResult(key, "failed", str(exc), raw_dir=raw_dir,
                              licence=src.licence)
    except Exception as exc:
        return DownloadResult(key, "failed", f"{type(exc).__name__}: {exc}",
                              raw_dir=raw_dir, licence=src.licence)

    # The source's own class order comes from its data.yaml. Without it the
    # class indices in the label files are meaningless, so this is fatal rather
    # than something to guess around.
    yaml_path = find_data_yaml(raw_dir)
    if not yaml_path:
        return DownloadResult(
            key, "failed",
            f"no data.yaml found under {raw_dir}; the label indices cannot be "
            f"interpreted without the source class order.",
            raw_dir=raw_dir, licence=src.licence)
    try:
        source_names = ds.read_yolo_names(yaml_path)
    except Exception as exc:
        return DownloadResult(key, "failed",
                              f"could not read class names from {yaml_path}: "
                              f"{exc}",
                              raw_dir=raw_dir, licence=src.licence)
    say(f"    {len(source_names)} source class(es)")

    normalised = os.path.join(dst_root, "_norm", key)
    try:
        if os.path.isdir(normalised):
            shutil.rmtree(normalised)
        norm_stats = normalise_yolo_layout(raw_dir, normalised, log=say)
    except DownloadError as exc:
        return DownloadResult(key, "failed", str(exc), raw_dir=raw_dir,
                              source_names=source_names, licence=src.licence)

    if not remap:
        return DownloadResult(key, "downloaded", None, raw_dir, normalised,
                              source_names, norm_stats, src.licence)

    stats = ds.remap_yolo_dataset(
        normalised, out_dir, source_names, dict(src.label_map),
        prefix=f"{key}_")
    stats["normalisation"] = norm_stats
    kept, dropped = stats["instances_kept"], stats["instances_dropped"]
    say(f"    remapped: {kept} instance(s) kept, {dropped} dropped")
    if stats["unmapped_source_classes"]:
        say(f"    unmapped source classes (instances dropped, not guessed): "
            f"{stats['unmapped_source_classes']}")

    if not keep_raw:
        shutil.rmtree(raw_dir, ignore_errors=True)
        shutil.rmtree(normalised, ignore_errors=True)

    return DownloadResult(key, "downloaded", None,
                          raw_dir if keep_raw else None, out_dir,
                          source_names, stats, src.licence)


def download_all(dst_root: str, keys: Optional[Sequence[str]] = None,
                 api_key: Optional[str] = None, keep_raw: bool = False,
                 log: Optional[Callable[[str], None]] = None) -> dict:
    """Fetch several sources and summarise what landed and what did not."""
    say = log or (lambda m: None)
    chosen = list(keys) if keys else [
        s.key for s in ds.SOURCES if s.usable and s.kind != "manual"]
    results = [download_source(k, dst_root, api_key=api_key,
                               keep_raw=keep_raw, log=say) for k in chosen]

    ok = [r for r in results if r.status == "downloaded"]
    licences = sorted({r.licence for r in ok if r.licence})
    manifest = {
        "dst_root": dst_root,
        "results": [r.to_dict() for r in results],
        "downloaded": [r.key for r in ok],
        "skipped": [r.key for r in results if r.status == "skipped"],
        "failed": [r.key for r in results if r.status == "failed"],
        "dataset_dirs": [r.dataset_dir for r in ok if r.dataset_dir],
        "instances_kept": sum(r.remap_stats.get("instances_kept", 0) for r in ok),
        "instances_dropped": sum(r.remap_stats.get("instances_dropped", 0)
                                 for r in ok),
        "licences": licences,
        "attribution_note": (
            "Every source above is redistributed under its own licence "
            f"({', '.join(licences) or 'see per-source licence'}). CC BY 4.0 "
            "requires attribution: keep this manifest with the trained model "
            "and credit each dataset in the model card."),
        "next_step": (
            "python -m training.electrical.cli merge --roots "
            + " ".join(r.dataset_dir for r in ok if r.dataset_dir)
            + " --dst data/merged") if ok else
            "Nothing downloaded — read the 'failed' reasons above.",
    }
    os.makedirs(dst_root, exist_ok=True)
    with open(os.path.join(dst_root, "download_manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


__all__ = [
    "DownloadError", "DownloadResult", "ROBOFLOW_FORMATS", "SPLIT_ALIASES",
    "fetch_roboflow", "fetch_kaggle", "fetch_url", "find_dataset_root",
    "find_data_yaml", "normalise_yolo_layout", "download_source",
    "download_all",
]
