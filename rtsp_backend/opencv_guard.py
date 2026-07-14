"""
OpenCV install guard.

A recurring foot-gun: installing ``insightface`` (or various other packages)
pulls in the GUI ``opencv-python`` wheel, which then coexists with the server's
``opencv-python-headless``. The two ship the same ``cv2`` module and clobber
each other, producing a broken import where APIs like ``cv2.CascadeClassifier``
and ``cv2.data`` silently disappear — surfacing later as::

    AttributeError: module 'cv2' has no attribute 'CascadeClassifier'

This module detects that condition, emits a clear warning with the one-line fix,
and (only when explicitly allowed) repairs the install automatically.
"""

from __future__ import annotations

import logging
import subprocess
import sys

log = logging.getLogger("rtsp_backend")

FIX_COMMAND = (
    'pip uninstall -y opencv-python opencv-python-headless && '
    'pip install --force-reinstall "opencv-python-headless>=4.9,<5"'
)


def _installed_opencv_packages() -> list[str]:
    """Return which opencv distributions pip currently has installed."""
    found = []
    try:
        from importlib import metadata
        names = {d.metadata["Name"].lower() for d in metadata.distributions()
                 if d.metadata and d.metadata.get("Name")}
        for pkg in ("opencv-python", "opencv-python-headless",
                    "opencv-contrib-python", "opencv-contrib-python-headless"):
            if pkg in names:
                found.append(pkg)
    except Exception:
        pass
    return found


def opencv_is_healthy() -> bool:
    """True if cv2 imports and exposes the legacy detection API we rely on."""
    try:
        import cv2  # noqa: F401
        return hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data")
    except Exception:
        return False


def diagnose() -> dict:
    pkgs = _installed_opencv_packages()
    gui = [p for p in pkgs if not p.endswith("headless")]
    return {
        "healthy": opencv_is_healthy(),
        "packages": pkgs,
        "conflict": len(pkgs) > 1 or bool(gui),
        "fix": FIX_COMMAND,
    }


def repair_opencv() -> bool:
    """Uninstall the GUI opencv wheels and (re)install headless. Returns True if
    cv2 is healthy afterwards. Time-bounded; used only when auto-repair is on."""
    try:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y",
                        "opencv-python", "opencv-contrib-python"], timeout=180)
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                        "--force-reinstall", "opencv-python-headless>=4.9,<5"],
                       timeout=300)
    except Exception as exc:
        log.warning("OpenCV auto-repair failed: %s", exc)
        return False
    # cv2 is already imported in this process; a reinstall won't take effect
    # until restart, so report based on package state, not the live module.
    d = diagnose()
    return not d["conflict"]


def check_and_warn(auto_repair: bool = False) -> dict:
    """Log a clear warning (and optionally repair) if the OpenCV install is
    broken/conflicted. Safe to call at startup; never raises."""
    d = diagnose()
    if d["healthy"] and not d["conflict"]:
        return d
    if d["conflict"]:
        log.warning(
            "Conflicting OpenCV packages installed (%s). This breaks cv2 "
            "(e.g. missing CascadeClassifier). Fix with:\n    %s",
            ", ".join(d["packages"]) or "unknown", FIX_COMMAND)
    if not d["healthy"]:
        log.warning("cv2 is missing expected APIs (CascadeClassifier/data). "
                    "Fix with:\n    %s", FIX_COMMAND)
    if auto_repair:
        log.info("Attempting automatic OpenCV repair (RTSP_ALLOW_AUTO_INSTALL)…")
        if repair_opencv():
            log.info("OpenCV repaired — restart the process for it to take effect.")
    return d
