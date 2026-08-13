import logging
from fastapi import APIRouter, Depends, status
from app.models.video_job import GenerateVideoJobRequest, GenerateVideoJobResponse
from app.services.video_service import create_and_enqueue_video_job
from app.services.job_service import get_job_by_id
from app.dependencies.auth import get_current_user_id
from app.dependencies.rate_limiter import enforce_rate_limit

logger = logging.getLogger(__name__)

router = APIRouter(tags=["video-jobs"])

@router.post("/v1/generateVideoJob", response_model=GenerateVideoJobResponse, status_code=status.HTTP_201_CREATED)
async def generate_video_job(
    request: GenerateVideoJobRequest,
    user_id: str = Depends(get_current_user_id)
):
    """
    Fast Tier Video Generation Endpoint:
    1. Authenticates request user.
    2. Enforces rate limiting per user.
    3. Atomically deducts Fast Tier video credits (40 credits for 10s, 50 credits for 15s).
    4. Runs Gemini 2.5 Flash / Structurer + Wan 2.2 Prompt Template Expansion Engine.
    5. Writes video job document to Firestore.
    6. Enqueues task to Celery / GPU worker queue.
    7. Returns immediately with jobId and status='pending'.
    """
    enforce_rate_limit(user_id)
    result = await create_and_enqueue_video_job(user_id=user_id, request=request)
    return GenerateVideoJobResponse(**result)

@router.get("/v1/videoJob/{job_id}", status_code=status.HTTP_200_OK)
async def get_video_job_status(job_id: str):
    """
    Poll video job execution status.
    Statuses: pending -> queued -> processing -> completed / error
    When completed, returns videoUrl asset URL.
    """
    return get_job_by_id(job_id)
