import uuid
import logging
from datetime import datetime, timezone
from app.models.face_swap_job import GenerateFaceSwapJobRequest, FaceSwapJobDocument
from app.services.wallet_service import check_and_deduct_credits
from app.db.firebase import get_db

logger = logging.getLogger(__name__)

COST_FACE_SWAP = 30  # 30 credits per video face swap

def get_face_swap_cost() -> int:
    return COST_FACE_SWAP

async def create_and_enqueue_face_swap_job(user_id: str, request: GenerateFaceSwapJobRequest) -> dict:
    """
    Creates and enqueues a video face swap job:
    1. Generates unique job ID.
    2. Atomically deducts 30 credits from user wallet.
    3. Saves initial job document in Firestore 'jobs' collection.
    4. Pushes job task to Celery worker queue.
    """
    job_id = f"fsjob_{uuid.uuid4().hex[:12]}"
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    logger.info(f"Creating Face Swap job={job_id} for user={user_id} (Cost={COST_FACE_SWAP} credits)")

    # 1. Deduct credits
    check_and_deduct_credits(
        user_id=user_id,
        cost=COST_FACE_SWAP,
        job_id=job_id
    )


    # 2. Write Job Document to Firestore
    job_doc = FaceSwapJobDocument(
        jobId=job_id,
        userId=user_id,
        params=request.params,
        status="pending",
        cost=COST_FACE_SWAP,
        createdAt=now_iso
    )
    db = get_db()
    db.collection("jobs").document(job_id).set(job_doc.model_dump(), merge=True)

    # 3. Enqueue to Celery background worker
    from app.workers.video_celery_worker import process_async_face_swap_job
    process_async_face_swap_job.delay(
        job_id=job_id,
        user_id=user_id,
        target_video_url=request.params.targetVideoUrl,
        source_image_url=request.params.sourceImageUrl,
        cost=COST_FACE_SWAP
    )

    return {
        "jobId": job_id,
        "status": "pending",
        "cost": COST_FACE_SWAP,
        "createdAt": now_iso
    }
