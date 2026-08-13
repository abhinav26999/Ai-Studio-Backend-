from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class JobType(str, Enum):
    IMAGE_GEN = "IMAGE_GEN"
    MESH_GEN = "MESH_GEN"
    BG_REMOVAL = "BG_REMOVAL"
    THEME_CHANGE = "THEME_CHANGE"
    VIDEO_GEN = "VIDEO_GEN"
    VIDEO_FACE_SWAP = "VIDEO_FACE_SWAP"


class JobTier(str, Enum):
    FAST = "FAST"

class JobStatus(str, Enum):
    PENDING = "pending"
    DEDUCTING_CREDITS = "deducting_credits"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

class JobParams(BaseModel):
    userPrompt: Optional[str] = None
    themeId: Optional[int] = None
    imageUrl: Optional[str] = None

class GenerateJobRequest(BaseModel):
    jobType: JobType
    tier: JobTier = JobTier.FAST
    params: JobParams

class GenerateJobResponse(BaseModel):
    jobId: str
    status: JobStatus = JobStatus.PENDING
    cost: int
    createdAt: str

class JobDocument(BaseModel):
    jobId: str
    userId: str
    type: JobType
    tier: JobTier = JobTier.FAST
    status: JobStatus = JobStatus.PENDING
    cost: int
    params: JobParams
    outputUrl: Optional[str] = None
    meshUrl: Optional[str] = None
    error: Optional[str] = None
    createdAt: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
