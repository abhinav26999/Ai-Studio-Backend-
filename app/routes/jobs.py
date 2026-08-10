import logging
from typing import Optional
from fastapi import APIRouter, Depends, Query, status
from app.models.job import GenerateJobRequest, GenerateJobResponse
from app.services.job_service import create_and_enqueue_job, get_job_by_id, list_user_jobs
from app.dependencies.auth import get_current_user_id
from app.dependencies.rate_limiter import enforce_rate_limit

logger = logging.getLogger(__name__)

router = APIRouter(tags=["jobs"])

@router.post("/v1/generateJob", response_model=GenerateJobResponse, status_code=status.HTTP_201_CREATED)
async def generate_job(
    request: GenerateJobRequest,
    user_id: str = Depends(get_current_user_id)
):
    """
    Primary Fast Tier Generation Endpoint:
    1. Authenticates request user.
    2. Enforces sliding window rate-limiting (15 requests/min per user).
    3. Atomically deducts Fast Tier credits (3 for IMAGE_GEN/THEME_CHANGE/BG_REMOVAL, 5 for MESH_GEN).
    4. Routes prompt (O(1) static lookup for themeId, Qwen 2.5 7B LLM call for userPrompt).
    5. Writes job document to Firestore.
    6. Enqueues job to Celery / Redis worker queue.
    7. Returns immediately with jobId and status='pending'.
    """
    enforce_rate_limit(user_id)
    result = await create_and_enqueue_job(user_id=user_id, request=request)
    return GenerateJobResponse(**result)

@router.get("/job/{job_id}", status_code=status.HTTP_200_OK)
async def get_job_status(job_id: str):
    """
    Poll job execution status:
    Statuses: pending -> queued -> processing -> completed / error
    When completed, returns outputUrl or meshUrl.
    """
    job_data = get_job_by_id(job_id)
    return job_data

@router.get("/jobs", status_code=status.HTTP_200_OK)
async def list_jobs(
    user_id: str = Depends(get_current_user_id),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """
    Fetch paginated job generation history for authenticated user.
    """
    return list_user_jobs(user_id=user_id, limit=limit, offset=offset)
