import uuid
import logging
import asyncio
from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime, timezone
from app.models.interview_job import (
    GenerateInterviewSequenceRequest,
    GenerateInterviewSequenceResponse,
)
from app.models.job import JobStatus
from app.services.interview_service import InterviewService
from app.dependencies.auth import get_current_user_id
from app.dependencies.rate_limiter import enforce_rate_limit
from app.db.firebase import get_db

def get_utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/interview", tags=["interview-lipsync"])


@router.post("/generate-sequence", response_model=GenerateInterviewSequenceResponse, status_code=status.HTTP_201_CREATED)
async def generate_interview_sequence(
    request: GenerateInterviewSequenceRequest,
    user_id: str = Depends(get_current_user_id)
):
    """
    Dedicated Multi-Question Interactive Interview Lip-Sync Generation Endpoint (Option A):
    1. Authenticates user and enforces rate limit.
    2. Validates question sequence.
    3. Creates initial Firestore document in 'interviews/{interviewId}'.
    4. Dispatches background worker to render all questions and optional idle loop.
    5. Returns interviewId immediately for realtime progress tracking.
    """
    enforce_rate_limit(user_id)

    if not request.avatarMediaUrl or not request.avatarMediaUrl.strip():
        raise HTTPException(status_code=400, detail="avatarMediaUrl is required")
    if not request.questions or len(request.questions) == 0:
        raise HTTPException(status_code=400, detail="At least one interview question is required")

    interview_id = f"iv_{uuid.uuid4().hex[:12]}"
    created_at = get_utc_now_iso()

    db = get_db()
    doc_data = {
        "interviewId": interview_id,
        "userId": user_id,
        "status": JobStatus.PENDING.value,
        "candidate": request.candidate.model_dump() if request.candidate else {},
        "totalQuestions": len(request.questions),
        "completedQuestions": 0,
        "idleLoopVideoUrl": None,
        "segments": [],
        "totalDurationSeconds": 0.0,
        "createdAt": created_at,
        "updatedAt": created_at,
        "error": None
    }
    db.collection("interviews").document(interview_id).set(doc_data)
    logger.info(f"Created interview document {interview_id} for user {user_id} with {len(request.questions)} questions")

    # Dispatch background worker
    asyncio.create_task(
        InterviewService.process_interview_job(
            interview_id=interview_id,
            user_id=user_id,
            request_data=request.model_dump()
        )
    )

    return GenerateInterviewSequenceResponse(
        interviewId=interview_id,
        status=JobStatus.PENDING,
        totalQuestions=len(request.questions),
        createdAt=created_at
    )


@router.get("/{interview_id}", status_code=status.HTTP_200_OK)
async def get_interview_status(interview_id: str):
    """
    Polls the interview generation status and returns full segments manifest when completed.
    """
    db = get_db()
    doc = db.collection("interviews").document(interview_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail=f"Interview {interview_id} not found")

    return doc.to_dict()
