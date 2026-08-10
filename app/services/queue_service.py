import json
import logging
from app.config import settings
from app.db.firebase import get_db

logger = logging.getLogger(__name__)

def enqueue_job_task(job_id: str, user_id: str, job_type: str, tier: str, payload_data: dict) -> bool:
    """
    Enqueue task into Google Cloud Tasks queue 'image-gen-queue'.
    Gracefully handles local development when Cloud Tasks environment is not configured.
    """
    db = get_db()
    job_ref = db.collection("jobs").document(job_id)

    message_body = {
        "jobId": job_id,
        "userId": user_id,
        "jobType": job_type,
        "tier": tier,
        "payload": payload_data
    }

    try:
        from google.cloud import tasks_v2
        client = tasks_v2.CloudTasksClient()
        parent = client.queue_path(settings.GCP_PROJECT_ID, settings.GCP_LOCATION, settings.GCP_TASKS_QUEUE)
        
        task = {
            "http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "url": f"https://gpu-worker-service-{settings.GCP_PROJECT_ID}.a.run.app/process-job",
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(message_body).encode()
            }
        }
        
        response = client.create_task(request={"parent": parent, "task": task})
        logger.info(f"Successfully enqueued Cloud Task: {response.name} for jobId {job_id}")

        job_ref.update({
            "status": "queued",
            "updatedAt": firestore_timestamp()
        })
        return True
    except Exception as err:
        logger.warning(f"Google Cloud Tasks client error / local mode for jobId {job_id}: {str(err)}")
        # In local/development environment, transition job status directly to 'queued'
        job_ref.update({
            "status": "queued"
        })
        return False

def firestore_timestamp():
    from datetime import datetime
    return datetime.utcnow().isoformat() + "Z"
