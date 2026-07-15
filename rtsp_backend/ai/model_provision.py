"""
Real face-model provisioning (SCRFD detection + ArcFace recognition).

The production face pipeline uses InsightFace model packs (SCRFD for detection,
ArcFace for 512-d recognition embeddings). InsightFace can download these packs
itself from its GitHub release, which works in most deployments. This module is
a robust, verifiable provisioner that:

* knows the exact SHA-256 and byte size of every ONNX file in each supported
  pack, so a downloaded file is *cryptographically verified* to be the genuine
  InsightFace weight before it is ever used — a corrupted or substituted file is
  rejected, never silently loaded;
* fetches each file from an ordered list of mirrors (respecting the environment
  ``HTTPS_PROXY`` and CA bundle), so it still works in locked-down networks
  where the raw GitHub release host is unreachable;
* never fabricates a model — if no source yields a byte-exact file it reports a
  precise error and the caller surfaces it (no fallback to a weak descriptor).

Only the ``detection`` and ``recognition`` ONNX files are provisioned, because
the face service uses ``allowed_modules=["detection", "recognition"]`` — the
landmark / gender-age models in the full pack are not needed and not fetched.
"""

from __future__ import annotations

import hashlib
import logging
import os
import ssl
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("rtsp_backend.ai.provision")


@dataclass
class ModelFile:
    """One ONNX file in a pack, with the sources to fetch it from and its hash."""

    name: str                       # filename inside the pack dir, e.g. det_10g.onnx
    role: str                       # "detection" | "recognition"
    sha256: str                     # canonical InsightFace weight hash (integrity)
    size: int                       # expected byte size
    urls: list[str] = field(default_factory=list)


# Ordered sources. media.githubusercontent.com serves git-LFS content and is the
# most widely reachable mirror; the official InsightFace release zip is not a
# per-file URL, so InsightFace's own downloader (tried separately by the
# embedder) covers the canonical path. Every file is SHA-256 verified after
# download regardless of which source served it, so a bad mirror cannot poison
# the model store.
MODEL_PACKS: dict[str, dict[str, ModelFile]] = {
    "buffalo_l": {
        "detection": ModelFile(
            name="det_10g.onnx", role="detection",
            sha256="5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
            size=16923827,
            urls=[
                "https://media.githubusercontent.com/media/dxcanh/face_swap/7286c3798f98b4fa37a368820cde03c7780d02b0/buffalo_l/det_10g.onnx",
                "https://media.githubusercontent.com/media/M00nWol/solidhaven-ai/772abae8d93c9d246400309b7b7f47a9d94743f5/model/insightface/models/buffalo_l/det_10g.onnx",
            ],
        ),
        "recognition": ModelFile(
            name="w600k_r50.onnx", role="recognition",
            sha256="4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
            size=174383860,
            urls=[
                "https://media.githubusercontent.com/media/mantzaris/Tagasaurus/b76c922f0e575b0ac7b18fb2846ba885d148e900/models/buffalo_l/w600k_r50.onnx",
                "https://media.githubusercontent.com/media/kumar-kiran-24/Automated-Attendance-System/ae44cfa71f3cdffcff07eb715ae40e53f38bd64e/models/models/buffalo_l/w600k_r50.onnx",
                "https://media.githubusercontent.com/media/M00nWol/solidhaven-ai/772abae8d93c9d246400309b7b7f47a9d94743f5/model/insightface/models/buffalo_l/w600k_r50.onnx",
            ],
        ),
    },
    "buffalo_s": {
        "detection": ModelFile(
            name="det_500m.onnx", role="detection",
            sha256="5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a",
            size=2524817,
            urls=[
                "https://media.githubusercontent.com/media/goyal705/FaceFindModel/941f6ce65f07ef08747e7d113f9d8a90e69e3652/app/core/buffalo_s/models/buffalo_s/det_500m.onnx",
                "https://media.githubusercontent.com/media/ankitkr9900/Real-Time-Attendance/85aaaaad71021aaac3b716044445fe9c74d1b9b1/Insightface_Model/Models/buffalo_s/det_500m.onnx",
            ],
        ),
        "recognition": ModelFile(
            name="w600k_mbf.onnx", role="recognition",
            sha256="9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f",
            size=13616099,
            urls=[
                "https://media.githubusercontent.com/media/JupiterMetaLabs/Face-Auth-SDK/68481243b71e423ec940d91cc96bbe31b85715eb/assets/models/w600k_mbf.onnx",
                "https://media.githubusercontent.com/media/goyal705/FaceFindModel/941f6ce65f07ef08747e7d113f9d8a90e69e3652/app/core/buffalo_s/models/buffalo_s/w600k_mbf.onnx",
            ],
        ),
    },
}

DEFAULT_PACK = "buffalo_l"

# Where the CA bundle lives when the environment routes HTTPS through a
# TLS-terminating proxy. Checked in order; falls back to the system default.
_CA_ENV_VARS = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "PIP_CERT")
_CA_FALLBACK = "/root/.ccr/ca-bundle.crt"


def _ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    cafile: Optional[str] = None
    for var in _CA_ENV_VARS:
        p = os.environ.get(var)
        if p and os.path.isfile(p):
            cafile = p
            break
    if cafile is None and os.path.isfile(_CA_FALLBACK):
        cafile = _CA_FALLBACK
    if cafile:
        try:
            ctx.load_verify_locations(cafile)
        except Exception:  # keep system defaults if the bundle is unreadable
            pass
    return ctx


def _opener() -> urllib.request.OpenerDirector:
    # ProxyHandler with no args reads *_PROXY env vars (HTTPS_PROXY, etc.).
    handlers: list = [urllib.request.ProxyHandler(urllib.request.getproxies())]
    handlers.append(urllib.request.HTTPSHandler(context=_ssl_context()))
    return urllib.request.build_opener(*handlers)


def _sha256_of(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def file_ok(path: str, spec: ModelFile) -> bool:
    """True iff ``path`` exists, has the expected size and the exact SHA-256."""
    try:
        if not os.path.isfile(path) or os.path.getsize(path) != spec.size:
            return False
        return _sha256_of(path) == spec.sha256
    except OSError:
        return False


def _download(url: str, dest: str, timeout: float = 300.0) -> None:
    """Stream ``url`` to ``dest`` (atomic via a .part temp file)."""
    tmp = dest + ".part"
    opener = _opener()
    req = urllib.request.Request(url, headers={"User-Agent": "ai-vision/face-provision"})
    with opener.open(req, timeout=timeout) as resp, open(tmp, "wb") as out:
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            out.write(block)
    os.replace(tmp, dest)


def ensure_model_file(spec: ModelFile, dest_dir: str,
                      timeout: float = 300.0) -> dict:
    """Ensure a single verified model file exists in ``dest_dir``.

    Returns a status dict; ``ok`` is True only when the file is present AND its
    SHA-256 matches the canonical InsightFace weight.
    """
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, spec.name)
    if file_ok(dest, spec):
        return {"ok": True, "name": spec.name, "role": spec.role,
                "source": "cache", "verified": True}

    errors: list[str] = []
    for url in spec.urls:
        try:
            _download(url, dest, timeout=timeout)
        except Exception as exc:  # try the next mirror
            errors.append(f"{url.split('/')[2]}: {type(exc).__name__}: {exc}")
            continue
        if file_ok(dest, spec):
            return {"ok": True, "name": spec.name, "role": spec.role,
                    "source": url, "verified": True}
        # Downloaded but hash/size mismatch — do not keep a suspect file.
        actual = None
        try:
            actual = _sha256_of(dest)
        except OSError:
            pass
        try:
            os.remove(dest)
        except OSError:
            pass
        errors.append(f"{url.split('/')[2]}: sha256 mismatch (got {actual})")

    return {"ok": False, "name": spec.name, "role": spec.role, "verified": False,
            "errors": errors}


def ensure_model_pack(pack: str, root: str, timeout: float = 300.0) -> dict:
    """Ensure the detection+recognition ONNX files of ``pack`` are present under
    ``<root>/models/<pack>/`` and byte-exact. Returns a structured status.

    ``root`` is the InsightFace root; FaceAnalysis(name=pack, root=root) then
    finds the files with no network access of its own.
    """
    if pack not in MODEL_PACKS:
        return {"ok": False, "pack": pack, "error": f"unknown model pack '{pack}'",
                "available": list(MODEL_PACKS)}
    dest_dir = os.path.join(root, "models", pack)
    t0 = time.monotonic()
    files: dict[str, dict] = {}
    ok = True
    for role, spec in MODEL_PACKS[pack].items():
        res = ensure_model_file(spec, dest_dir, timeout=timeout)
        files[spec.name] = res
        ok = ok and res["ok"]
    status = {
        "ok": ok, "pack": pack, "dir": dest_dir, "files": files,
        "elapsed_s": round(time.monotonic() - t0, 2),
    }
    if not ok:
        status["error"] = (
            f"could not provision a verified '{pack}' pack; every mirror failed "
            "or returned a file whose SHA-256 did not match the official "
            "InsightFace weight")
    return status
