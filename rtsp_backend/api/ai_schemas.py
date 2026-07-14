"""Request models for the AI / employee / events API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class EmployeeCreate(BaseModel):
    full_name: str = Field(min_length=1)
    employee_code: Optional[str] = None
    department: Optional[str] = None
    job_title: Optional[str] = None


class EmployeeUpdate(BaseModel):
    full_name: Optional[str] = Field(None, min_length=1)
    employee_code: Optional[str] = None
    department: Optional[str] = None
    job_title: Optional[str] = None


class ImageUpload(BaseModel):
    """Base64-encoded image (data URL or raw base64)."""

    image: str = Field(min_length=8)
    make_profile: bool = False


class CaptureRequest(BaseModel):
    """Capture the current frame from a camera and enroll it."""

    camera_id: Optional[str] = None
    make_profile: bool = False


class RegisterFromCaptures(BaseModel):
    """Create an employee and enrol one or more faces captured from the live
    RTSP stream in a single atomic call. Each image is a base64 data URL that
    already passed live validation on the client. If not a single image yields
    a usable face the whole registration is rolled back, so an employee is
    never created without a recognisable face."""

    full_name: str = Field(min_length=1)
    employee_code: Optional[str] = None
    department: Optional[str] = None
    job_title: Optional[str] = None
    images: list[str] = Field(min_length=1)


class ModelSelect(BaseModel):
    backend_id: str = Field(min_length=1)
    params: Optional[dict[str, Any]] = None


class ModelEnable(BaseModel):
    enabled: bool


class ModelParams(BaseModel):
    params: dict[str, Any]


class SettingSet(BaseModel):
    value: Any
