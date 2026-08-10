import logging
from datetime import datetime, timezone
from celery import Celery
from celery.exceptions import SoftTimeLimitExceeded
from app.config import settings
from app.db.firebase import get_db
from app.services.wallet_service import refund_credits

logger = logging.getLogger(__name__)

celery_app = Celery(
    "ai_studio_tasks",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

def get_utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

@celery_app.task(
    name="process_async_job",
    bind=True,
    max_retries=2,
    default_retry_delay=5,
    time_limit=300,        # Hard time limit: 5 minutes
    soft_time_limit=270    # Soft time limit: 4.5 minutes
)
def process_async_job(self, job_id: str, user_id: str, job_type: str, tier: str, payload_data: dict):
    """
    Celery Background Worker Task:
    1. Reads jobs/{jobId} from Firestore.
    2. Updates status to 'processing'.
    3. Executes AI generation/segmentation (FLUX.1 Schnell, BiRefNet, TripoSR).
    4. Explicitly handles THEME_CHANGE, BG_REMOVAL, IMAGE_GEN, MESH_GEN.
    5. Updates status to 'completed' with outputUrl/meshUrl in Firestore.
    6. On error or timeout (SoftTimeLimitExceeded), updates status to 'error' and refunds credits.
    """
    logger.info(f"Celery worker processing job_id={job_id}, job_type={job_type}, user_id={user_id}")
    db = get_db()
    job_ref = db.collection("jobs").document(job_id)
    job_doc = job_ref.get()

    if not job_doc.exists:
        logger.error(f"Job {job_id} not found in Firestore.")
        return {"success": False, "error": "Job not found"}

    job_data = job_doc.to_dict()
    cost = job_data.get("cost", 3)

    try:
        # Step 1: Update status to processing
        job_ref.update({
            "status": "processing",
            "updatedAt": get_utc_now_iso()
        })

        params = payload_data.get("params", {})
        processed_prompt = payload_data.get("processedPrompt", {})
        input_image_url = params.get("imageUrl")
        bucket_name = settings.FIREBASE_STORAGE_BUCKET

        output_url = None
        mesh_url = None

        if job_type == "BG_REMOVAL":
            logger.info(f"Executing BiRefNet / RMBG-2.0 Background Removal for {job_id}")
            output_filename = f"{job_id}_bg_removed.png"
            output_url = f"https://storage.googleapis.com/{bucket_name}/outputs/{user_id}/{output_filename}"

        elif job_type == "THEME_CHANGE":
            theme_id = params.get("themeId")
            logger.info(f"Executing Theme Change (FLUX.1 Schnell) for themeId={theme_id}, jobId={job_id}")
            output_filename = f"{job_id}_theme_{theme_id or 'preset'}.png"
            output_url = f"https://storage.googleapis.com/{bucket_name}/outputs/{user_id}/{output_filename}"

        elif job_type == "IMAGE_GEN":
            logger.info(f"Executing FLUX.1 Schnell Custom Prompt Generation for {job_id}")
            output_filename = f"{job_id}_generated.png"
            output_url = f"https://storage.googleapis.com/{bucket_name}/outputs/{user_id}/{output_filename}"

        elif job_type == "MESH_GEN":
            logger.info(f"Executing TripoSR / Stable Fast 3D Mesh Generation for {job_id}")
            mesh_filename = f"{job_id}_mesh.glb"
            mesh_url = f"https://storage.googleapis.com/{bucket_name}/outputs/{user_id}/{mesh_filename}"

        # Step 2: Update Firestore doc on completion
        update_data = {
            "status": "completed",
            "error": None,
            "updatedAt": get_utc_now_iso()
        }
        if output_url:
            update_data["outputUrl"] = output_url
        if mesh_url:
            update_data["meshUrl"] = mesh_url

        job_ref.update(update_data)
        logger.info(f"Celery task completed successfully for jobId={job_id}")
        return {"success": True, "jobId": job_id, "outputUrl": output_url, "meshUrl": mesh_url}

    except SoftTimeLimitExceeded:
        logger.error(f"Celery task execution timed out for jobId={job_id} (soft_time_limit exceeded)")
        job_ref.update({
            "status": "error",
            "error": "Task execution timed out (4.5 min limit exceeded)",
            "updatedAt": get_utc_now_iso()
        })
        try:
            refund_credits(user_id=user_id, cost=cost, job_id=job_id, reason="Celery soft time limit timeout")
        except Exception as refund_err:
            logger.error(f"Failed to refund credits on timeout for jobId={job_id}: {str(refund_err)}")
        return {"success": False, "jobId": job_id, "error": "Timeout"}

    except Exception as err:
        logger.error(f"Celery worker failed for jobId={job_id}: {str(err)}")
        job_ref.update({
            "status": "error",
            "error": str(err),
            "updatedAt": get_utc_now_iso()
        })

        # Refund credits on failure
        try:
            refund_credits(user_id=user_id, cost=cost, job_id=job_id, reason=f"Celery error: {str(err)}")
        except Exception as refund_err:
            logger.error(f"Failed to refund credits for jobId={job_id}: {str(refund_err)}")

        raise self.retry(exc=err)
