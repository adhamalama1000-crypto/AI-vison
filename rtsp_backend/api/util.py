"""Shared context and helpers for the API routers."""

from __future__ import annotations

import base64
import binascii
import os
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from ..errors import RTSPBackendError


class BadImageError(RTSPBackendError):
    status_code = 400
    code = "bad_image"


@dataclass
class Context:
    db: object
    manager: object          # CameraManager
    ai: object               # AIModelManager
    pipeline: object         # AIPipeline
    bus: object              # EventBus
    data_dir: str


def decode_image(data: str) -> np.ndarray:
    """Decode a base64 (optionally data-URL) image string to a BGR ndarray."""
    if not isinstance(data, str) or len(data) < 8:
        raise BadImageError("Empty or invalid image data.")
    if data.startswith("data:"):
        _, _, data = data.partition(",")
    try:
        raw = base64.b64decode(data, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise BadImageError(f"Invalid base64 image: {exc}")
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise BadImageError("Could not decode image bytes as an image.")
    return img


def encode_jpeg_b64(img: np.ndarray, quality: int = 85) -> str:
    """Encode a BGR image to a base64 data URL (used to preview candidates)."""
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise BadImageError("Failed to encode image.")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def save_image(data_dir: str, subdir: str, img: np.ndarray, prefix: str = "img") -> str:
    rel = f"{subdir}/{prefix}_{int(time.time()*1000)}.jpg"
    path = os.path.join(data_dir, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise BadImageError("Failed to encode image for storage.")
    with open(path, "wb") as fh:
        fh.write(buf.tobytes())
    return rel


def employee_dict(row) -> dict:
    return {
        "id": row["id"],
        "employee_code": row["employee_code"],
        "full_name": row["full_name"],
        "department": row["department"],
        "job_title": row["job_title"],
        "profile_image": row["profile_image"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
