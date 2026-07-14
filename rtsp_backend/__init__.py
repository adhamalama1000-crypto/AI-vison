"""RTSP-only camera backend for industrial deployments."""

from .app import build_app
from .config import CameraConfig, Settings, load_settings

__version__ = "3.1.4"

__all__ = ["build_app", "load_settings", "Settings", "CameraConfig", "__version__"]
