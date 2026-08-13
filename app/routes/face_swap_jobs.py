import logging
from fastapi import APIRouter, Depends, status
from app.models.face_swap_job import GenerateFaceSwapJobRequest, GenerateFaceSwapJobResponse
from app.services.face_swap_service import create_and_enqueue_face_swap_job
from app.services.job_service import get_job_by_id
from app.dependencies.auth import get_current_user_id
from app.dependencies.rate_limiter import enforce_rate_limit

logger = logging.getLogger(__name__)

router = APIRouter(tags=["face-swap-jobs"])

@router.post("/v1/generateFaceSwapJob", response_model=GenerateFaceSwapJobResponse, status_code=status.HTTP_201_CREATED)
async def generate_face_swap_job(
    request: GenerateFaceSwapJobRequest,
    user_id: str = Depends(get_current_user_id)
):
    """
    AI Video Face Swap Endpoint:
    1. Authenticates user and enforces rate limit.
    2. Atomically deducts 30 credits.
    3. Writes job document to Firestore.
    4. Enqueues task to Celery worker queue.
    5. Returns immediately with jobId and status='pending'.
    """
    enforce_rate_limit(user_id)
    result = await create_and_enqueue_face_swap_job(user_id=user_id, request=request)
    return GenerateFaceSwapJobResponse(**result)

@router.get("/v1/faceSwapJob/{job_id}", status_code=status.HTTP_200_OK)
async def get_face_swap_job_status(job_id: str):
    """
    Poll video face swap job execution status.
    Statuses: pending -> processing -> completed / error
    When completed, returns videoUrl of the swapped video asset.
    """
    return get_job_by_id(job_id)
