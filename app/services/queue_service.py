import logging
from datetime import datetime, timezone
from app.config import settings
from app.db.firebase import get_db
from app.workers.celery_worker import process_async_job

logger = logging.getLogger(__name__)

def get_utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def enqueue_job_task(job_id: str, user_id: str, job_type: str, tier: str, payload_data: dict) -> bool:
    """
    Enqueue task into Celery / Redis background worker queue.
    """
    db = get_db()
    job_ref = db.collection("jobs").document(job_id)

    try:
        async_res = process_async_job.delay(
            job_id=job_id,
            user_id=user_id,
            job_type=job_type,
            tier=tier,
            payload_data=payload_data
        )
        logger.info(f"Successfully enqueued job {job_id} to Celery / Redis queue (task_id={async_res.id})")
        job_ref.update({
            "status": "queued",
            "updatedAt": get_utc_now_iso()
        })
        return True
    except Exception as err:
        logger.error(f"Error enqueuing job {job_id} to Celery/Redis: {str(err)}")
        job_ref.update({
            "status": "queued",
            "updatedAt": get_utc_now_iso()
        })
        return False
