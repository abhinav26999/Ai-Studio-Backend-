from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from app.models.job import JobStatus


class InterviewQuestion(BaseModel):
    id: str = Field(..., description="Unique ID for question, e.g. Q1, Q2")
    tag: Optional[str] = Field(default=None, description="Topic or category tag, e.g. introduction, technicalHTML")
    text: str = Field(..., description="Text paragraph for the AI avatar to speak")
    voice: Optional[str] = Field(default="en-US-GuyNeural", description="TTS voice identifier")


class CandidateInfo(BaseModel):
    name: Optional[str] = Field(default="Candidate", description="Full name of candidate")
    email: Optional[str] = Field(default=None, description="Email address")
    difficulty: Optional[str] = Field(default="easy", description="Interview difficulty level")


class GenerateInterviewSequenceRequest(BaseModel):
    avatarMediaUrl: str = Field(..., description="Base avatar video URL or high-res photo URL")
    candidate: Optional[CandidateInfo] = Field(default_factory=CandidateInfo)
    questions: List[InterviewQuestion] = Field(..., min_length=1, description="List of interview questions in sequence")
    generateIdleLoop: bool = Field(default=True, description="Whether to extract/generate a seamless silent idle breathing loop for intervals")


class InterviewSegmentMetadata(BaseModel):
    segmentIndex: int
    questionId: str
    tag: Optional[str] = None
    questionText: str
    durationSeconds: float
    videoUrl: Optional[str] = None
    status: str = "completed"
    error: Optional[str] = None


class InterviewJobDocument(BaseModel):
    interviewId: str
    userId: str
    status: JobStatus = JobStatus.PENDING
    candidate: Optional[Dict[str, Any]] = None
    totalQuestions: int
    completedQuestions: int = 0
    idleLoopVideoUrl: Optional[str] = None
    segments: List[Dict[str, Any]] = []
    totalDurationSeconds: float = 0.0
    createdAt: str
    updatedAt: Optional[str] = None
    error: Optional[str] = None


class GenerateInterviewSequenceResponse(BaseModel):
    interviewId: str
    status: JobStatus = JobStatus.PENDING
    totalQuestions: int
    createdAt: str
