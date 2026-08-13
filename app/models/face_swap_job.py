from datetime import datetime, timezone
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from app.models.job import JobType, JobTier, JobStatus

class FaceSwapJobParams(BaseModel):
    targetVideoUrl: str = Field(..., description="URL of the target/base video to replace faces in")
    sourceImageUrl: str = Field(..., description="URL of the character face photo to insert")
    title: Optional[str] = Field(default=None, description="Optional title or label for the swap job")

class GenerateFaceSwapJobRequest(BaseModel):
    jobType: JobType = Field(default=JobType.VIDEO_FACE_SWAP)
    tier: JobTier = Field(default=JobTier.FAST)
    params: FaceSwapJobParams

class GenerateFaceSwapJobResponse(BaseModel):
    jobId: str
    status: JobStatus = JobStatus.PENDING
    cost: int = 30
    createdAt: str

class FaceSwapJobDocument(BaseModel):
    jobId: str
    userId: str
    type: JobType = JobType.VIDEO_FACE_SWAP
    tier: JobTier = JobTier.FAST
    status: JobStatus = JobStatus.PENDING
    cost: int = 30
    params: FaceSwapJobParams
    videoUrl: Optional[str] = None
    error: Optional[str] = None
    createdAt: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
