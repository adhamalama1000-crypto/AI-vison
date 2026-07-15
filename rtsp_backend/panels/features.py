"""
Panel feature embedding & registration.

Two jobs:

* **Embedding** — a compact, storable fingerprint of a reference panel image so
  a panel can be recognised and roughly matched later. We use ORB keypoints +
  descriptors (rotation/scale robust, weights-free) plus a coarse colour /
  gradient histogram as a global descriptor. Descriptors are stored base64 in
  SQLite so registration survives a restart.

* **Alignment** — estimate the homography that maps an *observed* panel image
  into the *reference* image's coordinate frame, using ORB feature matches +
  RANSAC. This is what lets the comparison stage say "the contactor moved" or
  "this wire is missing" in the reference's coordinates, instead of being
  fooled by camera pose / zoom differences.

Nothing here needs a GPU or trained weights.
"""

from __future__ import annotations

import base64
from typing import Any, Optional

import cv2
import numpy as np

_MAX_FEATURES = 1200


def _orb(nfeatures: int = _MAX_FEATURES):
    return cv2.ORB_create(nfeatures=nfeatures, scaleFactor=1.2, nlevels=8)


def color_histogram(image_bgr: np.ndarray, bins: int = 8) -> list[float]:
    """Normalised 3-channel HSV histogram — a cheap global descriptor."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [bins, bins, bins],
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return [float(v) for v in hist]


def extract_features(image_bgr: np.ndarray) -> dict[str, Any]:
    """Compute a storable feature embedding for a reference image."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    orb = _orb()
    kps, desc = orb.detectAndCompute(gray, None)
    out: dict[str, Any] = {
        "width": int(image_bgr.shape[1]),
        "height": int(image_bgr.shape[0]),
        "n_keypoints": int(len(kps) if kps else 0),
        "histogram": color_histogram(image_bgr),
        "descriptor_type": "ORB",
    }
    if desc is not None and len(desc):
        out["descriptors_b64"] = base64.b64encode(
            np.ascontiguousarray(desc, dtype=np.uint8).tobytes()).decode("ascii")
        out["descriptors_shape"] = [int(desc.shape[0]), int(desc.shape[1])]
        pts = np.array([kp.pt for kp in kps], dtype=np.float32)
        out["keypoints_b64"] = base64.b64encode(
            np.ascontiguousarray(pts).tobytes()).decode("ascii")
        out["keypoints_shape"] = [int(pts.shape[0]), 2]
    return out


def _decode(features: dict) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Reconstruct (keypoints Nx2, descriptors NxD) from a stored embedding."""
    desc = kps = None
    if features.get("descriptors_b64") and features.get("descriptors_shape"):
        raw = base64.b64decode(features["descriptors_b64"])
        shape = tuple(features["descriptors_shape"])
        desc = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
    if features.get("keypoints_b64") and features.get("keypoints_shape"):
        raw = base64.b64decode(features["keypoints_b64"])
        shape = tuple(features["keypoints_shape"])
        kps = np.frombuffer(raw, dtype=np.float32).reshape(shape)
    return kps, desc


def histogram_similarity(a: list[float], b: list[float]) -> float:
    """Correlation of two colour histograms in [0,1] (clamped)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    sim = float(cv2.compareHist(va, vb, cv2.HISTCMP_CORREL))
    return max(0.0, min(1.0, (sim + 1.0) / 2.0))


def align(ref_features: dict, observed_bgr: np.ndarray,
          min_matches: int = 12) -> dict[str, Any]:
    """Estimate the homography mapping observed -> reference coordinates.

    Returns ``{"ok", "homography" (3x3 list | None), "n_matches",
    "inliers", "match_ratio", "note"}``. Never raises.
    """
    result: dict[str, Any] = {"ok": False, "homography": None, "n_matches": 0,
                              "inliers": 0, "match_ratio": 0.0, "note": None}
    ref_kps, ref_desc = _decode(ref_features)
    if ref_desc is None or ref_kps is None:
        result["note"] = "reference has no stored ORB descriptors"
        return result
    try:
        gray = cv2.cvtColor(observed_bgr, cv2.COLOR_BGR2GRAY)
        obs_kps_cv, obs_desc = _orb().detectAndCompute(gray, None)
        if obs_desc is None or len(obs_desc) < min_matches:
            result["note"] = "too few features in observed image"
            return result
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        knn = bf.knnMatch(obs_desc, ref_desc, k=2)
        good = []
        for pair in knn:
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                good.append(pair[0])
        result["n_matches"] = len(good)
        if len(good) < min_matches:
            result["note"] = f"only {len(good)} good matches (< {min_matches})"
            return result
        src = np.array([obs_kps_cv[m.queryIdx].pt for m in good], dtype=np.float32).reshape(-1, 1, 2)
        dst = np.array([ref_kps[m.trainIdx] for m in good], dtype=np.float32).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            result["note"] = "homography estimation failed"
            return result
        inliers = int(mask.sum()) if mask is not None else 0
        result.update({
            "ok": True, "homography": H.tolist(), "inliers": inliers,
            "match_ratio": round(inliers / max(1, len(good)), 3),
        })
        return result
    except Exception as exc:  # never break inspection over alignment
        result["note"] = f"alignment error: {exc}"
        return result


def warp_point(H: Optional[list], x: float, y: float) -> tuple[float, float]:
    """Map a point through a homography (identity if H is None)."""
    if H is None:
        return (x, y)
    M = np.array(H, dtype=np.float64)
    v = M @ np.array([x, y, 1.0])
    if abs(v[2]) < 1e-9:
        return (x, y)
    return (float(v[0] / v[2]), float(v[1] / v[2]))
