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


class FaceConfig(BaseModel):
    """Tunable face-recognition parameters, all optional (partial update)."""

    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    margin: Optional[float] = Field(None, ge=0.0, le=1.0)
    match_policy: Optional[str] = Field(None, pattern="^(average|nearest)$")
    min_det_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    min_blur: Optional[float] = Field(None, ge=0.0)
    min_recog_blur: Optional[float] = Field(None, ge=0.0)
    min_face_size: Optional[int] = Field(None, ge=1, le=2000)
    enroll_min_face_size: Optional[int] = Field(None, ge=1, le=2000)
    det_size: Optional[int] = Field(None, ge=128, le=1280)
    model_pack: Optional[str] = Field(None, pattern="^(buffalo_l|buffalo_s)$")
