import uuid
import logging
from typing import Any
from datetime import datetime, timezone
from fastapi import HTTPException

from app.config import settings
from app.db.firebase import get_db, firestore
from app.models.job import JobStatus, JobType, JobTier
from app.models.video_job import GenerateVideoJobRequest
from app.services.wallet_service import check_and_deduct_credits
from app.services.video_prompt_engine import process_video_job_prompt

logger = logging.getLogger(__name__)

def get_utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def get_video_job_cost(duration: int) -> int:
    """Fast Tier Video Credit Cost Rules: 10s = 40 credits, 15s = 50 credits."""
    if duration == 15:
        return settings.CREDIT_COST_VIDEO_15S
    return settings.CREDIT_COST_VIDEO_10S

def enqueue_video_job_task(job_id: str, user_id: str, job_type: str, tier: str, duration: int, payload_data: dict):
    """Enqueue video task to Celery / Redis worker."""
    if settings.ENABLE_CELERY:
        try:
            from app.workers.video_celery_worker import process_async_video_job
            process_async_video_job.delay(
                job_id=job_id,
                user_id=user_id,
                job_type=job_type,
                tier=tier,
                duration=duration,
                payload_data=payload_data
            )
            logger.info(f"Successfully enqueued video task {job_id} to Celery queue")
            return
        except Exception as err:
            logger.warning(f"Failed to enqueue to Celery queue ({str(err)}). Falling back to direct worker.")

    # In-memory execution fallback if Celery is disabled
    try:
        from app.workers.video_celery_worker import process_async_video_job
        process_async_video_job(
            job_id=job_id,
            user_id=user_id,
            job_type=job_type,
            tier=tier,
            duration=duration,
            payload_data=payload_data
        )
    except Exception as err:
        logger.error(f"In-memory video worker execution failed for {job_id}: {str(err)}")

async def create_and_enqueue_video_job(user_id: str, request: GenerateVideoJobRequest) -> dict:
    """
    Video Job Orchestrator:
    1. Validate input parameters (sceneScript required).
    2. Calculate cost (40 credits for 10s, 50 credits for 15s).
    3. Perform atomic credit deduction.
    4. Process prompt via Gemini Flash Structurer & Wan 2.2 Prompt Expansion Engine.
    5. Write jobs/{jobId} to Firestore.
    6. Enqueue task for Celery / GPU worker.
    7. Return response immediately to client.
    """
    if not request.params.sceneScript or not request.params.sceneScript.strip():
        raise HTTPException(
            status_code=400,
            detail="Video generation parameters must specify a non-empty 'sceneScript'."
        )

    job_id = f"vjob_{uuid.uuid4().hex[:12]}"
    cost = get_video_job_cost(request.duration)
    created_at = get_utc_now_iso()

    # Step 1: Deduct credits atomically from Firebase Wallet
    check_and_deduct_credits(user_id=user_id, cost=cost, job_id=job_id)

    # Step 2: Gemini Flash Structuring & Prompt Expansion Engine
    processed_prompt = await process_video_job_prompt(
        params_dict=request.params.model_dump(),
        duration=request.duration,
        aspect_ratio=request.aspectRatio
    )

    # Step 3: Create Firestore job document
    job_doc_data = {
        "jobId": job_id,
        "userId": user_id,
        "type": request.jobType.value if hasattr(request.jobType, 'value') else str(request.jobType),
        "tier": request.tier.value if hasattr(request.tier, 'value') else str(request.tier),
        "duration": request.duration,
        "aspectRatio": request.aspectRatio,
        "status": JobStatus.PENDING.value,
        "cost": cost,
        "params": request.params.model_dump(),
        "processedPrompt": processed_prompt,
        "videoUrl": None,
        "outputUrl": None,
        "error": None,
        "createdAt": created_at
    }

    db = get_db()
    db.collection("jobs").document(job_id).set(job_doc_data)
    logger.info(f"Created video job document {job_id} for user {user_id} with cost {cost} credits ({request.duration}s)")

    # Step 4: Enqueue task for background processing
    enqueue_video_job_task(
        job_id=job_id,
        user_id=user_id,
        job_type=request.jobType.value if hasattr(request.jobType, 'value') else str(request.jobType),
        tier=request.tier.value if hasattr(request.tier, 'value') else str(request.tier),
        duration=request.duration,
        payload_data={
            "params": request.params.model_dump(),
            "processedPrompt": processed_prompt
        }
    )

    return {
        "jobId": job_id,
        "status": JobStatus.PENDING.value,
        "cost": cost,
        "duration": request.duration,
        "createdAt": created_at
    }


async def create_and_enqueue_lip_sync_job(user_id: str, request: Any) -> dict:
    """
    Lip-Sync Job Orchestrator (100% Free / Zero-Cost):
    1. Validate input parameters (videoUrl and paragraphText required).
    2. Deduct fixed 50 wallet credits atomically.
    3. Write jobs/{jobId} to Firestore.
    4. Enqueue background Lip-Sync task.
    5. Return immediate response with jobId.
    """
    params = request.params
    input_media_url = (getattr(params, "imageUrl", None) or getattr(params, "videoUrl", None) or "").strip()
    if not input_media_url:
        raise HTTPException(status_code=400, detail="Lip-Sync parameters must specify a non-empty 'imageUrl' or 'videoUrl'.")
    if not params.paragraphText or not params.paragraphText.strip():
        raise HTTPException(status_code=400, detail="Lip-Sync parameters must specify a non-empty 'paragraphText'.")

    job_id = f"lsjob_{uuid.uuid4().hex[:12]}"
    cost = settings.CREDIT_COST_LIP_SYNC
    created_at = get_utc_now_iso()

    # Step 1: Deduct 50 credits atomically from Firebase Wallet
    check_and_deduct_credits(user_id=user_id, cost=cost, job_id=job_id)

    # Step 2: Create Firestore job document
    job_doc_data = {
        "jobId": job_id,
        "userId": user_id,
        "type": JobType.LIP_SYNC.value,
        "tier": request.tier.value if hasattr(request.tier, 'value') else str(request.tier),
        "status": JobStatus.PENDING.value,
        "cost": cost,
        "params": params.model_dump(),
        "outputUrl": None,
        "error": None,
        "createdAt": created_at
    }

    db = get_db()
    db.collection("jobs").document(job_id).set(job_doc_data)
    logger.info(f"Created Lip-Sync job document {job_id} for user {user_id} with cost {cost} credits")

    # Step 3: Enqueue task for background processing via LipSyncProcessor
    from app.services.lip_sync_processor import LipSyncProcessor
    import asyncio
    asyncio.create_task(LipSyncProcessor.process_job(job_id=job_id, user_id=user_id, params=params.model_dump()))

    return {
        "jobId": job_id,
        "status": JobStatus.PENDING.value,
        "cost": cost,
        "createdAt": created_at
    }

