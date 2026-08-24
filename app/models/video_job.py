from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
from app.models.job import JobType, JobTier, JobStatus

class VideoJobParams(BaseModel):
    sceneScript: str = Field(..., description="User scene text script describing camera motion, action, and environment")
    images: List[str] = Field(default_factory=list, description="Optional reference image URLs (0 to 3 images)")

class GenerateVideoJobRequest(BaseModel):
    jobType: JobType = Field(default=JobType.VIDEO_GEN)
    tier: JobTier = Field(default=JobTier.FAST)
    duration: int = Field(default=10, description="Video duration in seconds: 10 or 15")
    aspectRatio: str = Field(default="16:9", description="Aspect ratio: 16:9, 9:16, 1:1")
    params: VideoJobParams

    @field_validator("duration")
    @classmethod
    def validate_duration(cls, v: int) -> int:
        if v not in (10, 15):
            raise ValueError("Video duration must be either 10 or 15 seconds")
        return v

    @field_validator("aspectRatio")
    @classmethod
    def validate_aspect_ratio(cls, v: str) -> str:
        if v not in ("16:9", "9:16", "1:1"):
            raise ValueError("Aspect ratio must be one of '16:9', '9:16', '1:1'")
        return v

class GenerateVideoJobResponse(BaseModel):
    jobId: str
    status: JobStatus = JobStatus.PENDING
    cost: int
    duration: int
    createdAt: str

class LipSyncJobParams(BaseModel):
    videoUrl: Optional[str] = Field(default=None, description="Uploaded sample video asset URL")
    imageUrl: Optional[str] = Field(default=None, description="Uploaded sample AI character image asset URL")
    paragraphText: str = Field(..., description="Paragraph of text for character to speak with lip sync")
    voice: Optional[str] = Field(default="en-US-GuyNeural", description="Voice selection: 'en-US-GuyNeural' (male) or 'en-US-JennyNeural' (female)")

class GenerateLipSyncJobRequest(BaseModel):
    jobType: JobType = Field(default=JobType.LIP_SYNC)
    tier: JobTier = Field(default=JobTier.FAST)
    params: LipSyncJobParams


class VideoJobDocument(BaseModel):
    jobId: str
    userId: str
    type: JobType = JobType.VIDEO_GEN
    tier: JobTier = JobTier.FAST
    status: JobStatus = JobStatus.PENDING
    cost: int
    duration: int
    aspectRatio: str
    params: VideoJobParams
    structuredJson: Optional[Dict[str, Any]] = None
    processedPrompt: Optional[Dict[str, Any]] = None
    videoUrl: Optional[str] = None
    error: Optional[str] = None
    createdAt: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
