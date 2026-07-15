"""
Fetch a small, real, multi-identity face dataset for evaluation.

Pulls a subset of the canonical **Labeled Faces in the Wild (LFW)** benchmark
(real photographs of public figures, many images per person, varied pose /
lighting / expression) from a public mirror. The images are the genuine LFW
funneled crops — not synthetic stand-ins — so FAR/FRR/EER measured on them are
meaningful. Files are cached locally (git-ignored) so a re-run is instant.

Two disjoint identity groups are fetched:
* KNOWN     — enrolled as employees (gallery + probes of the same people).
* STRANGERS — never enrolled; used as impostor probes to measure FAR.

Usage:
    python -m scripts.fetch_lfw_eval            # download into the cache
    from scripts.fetch_lfw_eval import load_dataset
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

# Public LFW mirror (git-LFS); media endpoint serves the real image bytes and is
# reachable where the raw GitHub release host is not.
_REPO = "alessiosavi/tensorflow-face-recognition"
_REF = "62626eba92eed7fb21f4354cf38fa66c6240fb83"
_BASE = f"https://media.githubusercontent.com/media/{_REPO}/{_REF}/lfw"

CACHE_DIR = os.environ.get(
    "AIVISION_EVAL_DIR", os.path.expanduser("~/.cache/aivision_eval/lfw"))

# Identities with plenty of images. KNOWN people are enrolled; STRANGERS are not.
KNOWN = [
    "George_W_Bush", "Colin_Powell", "Tony_Blair", "Donald_Rumsfeld",
    "Gerhard_Schroeder", "Ariel_Sharon", "Hugo_Chavez", "Junichiro_Koizumi",
    "Jean_Chretien", "John_Ashcroft",
]
STRANGERS = [
    "Serena_Williams", "Vladimir_Putin", "Jacques_Chirac",
    "Luiz_Inacio_Lula_da_Silva", "Arnold_Schwarzenegger", "Jennifer_Capriati",
]

# how many images to try per identity (sequential _0001.._000N, gaps tolerated)
MAX_PER_KNOWN = 30
MAX_PER_STRANGER = 12


def _download_one(name: str, idx: int, timeout: float = 60.0) -> Optional[str]:
    import urllib.request

    fname = f"{name}_{idx:04d}.jpg"
    dest_dir = os.path.join(CACHE_DIR, name)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, fname)
    if os.path.isfile(dest) and os.path.getsize(dest) > 1000:
        return dest
    url = f"{_BASE}/{name}/{fname}"
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler(urllib.request.getproxies()))
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "aivision-eval"})
        with opener.open(req, timeout=timeout) as resp:
            data = resp.read()
    except Exception:
        return None
    if len(data) < 1000:  # 404 page / empty
        return None
    with open(dest, "wb") as fh:
        fh.write(data)
    return dest


def _fetch_identity(name: str, max_n: int) -> list[str]:
    paths: list[str] = []
    misses = 0
    for idx in range(1, max_n + 1):
        p = _download_one(name, idx)
        if p:
            paths.append(p)
            misses = 0
        else:
            misses += 1
            if misses >= 5 and idx > 3:  # sequence exhausted
                break
    return paths


def download(verbose: bool = True) -> dict[str, list[str]]:
    """Download the subset; return ``{identity: [image_paths]}``."""
    result: dict[str, list[str]] = {}
    jobs = [(n, MAX_PER_KNOWN) for n in KNOWN] + \
           [(n, MAX_PER_STRANGER) for n in STRANGERS]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_fetch_identity, n, m): n for n, m in jobs}
        for fut in futs:
            name = futs[fut]
            paths = fut.result()
            if paths:
                result[name] = paths
            if verbose:
                print(f"  {name}: {len(paths)} images")
    return result


def load_dataset(verbose: bool = False) -> dict[str, list["object"]]:
    """Load the dataset as ``{identity: [BGR ndarray, ...]}`` (downloads if
    the cache is empty). Returns an empty dict if nothing could be fetched."""
    import cv2

    paths = download(verbose=verbose)
    out: dict[str, list] = {}
    for name, files in paths.items():
        imgs = []
        for f in files:
            img = cv2.imread(f)
            if img is not None:
                imgs.append(img)
        if imgs:
            out[name] = imgs
    return out


if __name__ == "__main__":
    print(f"Fetching LFW evaluation subset into {CACHE_DIR} …")
    data = download(verbose=True)
    n_imgs = sum(len(v) for v in data.values())
    print(f"Done: {len(data)} identities, {n_imgs} images.")
