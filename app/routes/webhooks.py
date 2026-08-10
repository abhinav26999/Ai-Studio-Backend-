import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from app.db.firebase import get_db
from app.services.wallet_service import refund_credits

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook", tags=["webhooks"])

class JobCompletedWebhookRequest(BaseModel):
    jobId: str
    outputUrl: Optional[str] = None
    meshUrl: Optional[str] = None
    status: str = "completed"

class JobFailedWebhookRequest(BaseModel):
    jobId: str
    error: str
    status: str = "error"

@router.post("/job-completed", status_code=status.HTTP_200_OK)
async def webhook_job_completed(payload: JobCompletedWebhookRequest):
    """
    Called by Cloud Run GPU workers on job completion.
    Updates jobs/{jobId} in Firestore: status="completed", outputUrl/meshUrl set.
    """
    db = get_db()
    job_ref = db.collection("jobs").document(payload.jobId)
    job_doc = job_ref.get()

    if not job_doc.exists:
        raise HTTPException(status_code=404, detail=f"Job {payload.jobId} not found")

    update_data = {
        "status": "completed",
        "error": None
    }
    if payload.outputUrl:
        update_data["outputUrl"] = payload.outputUrl
    if payload.meshUrl:
        update_data["meshUrl"] = payload.meshUrl

    job_ref.update(update_data)
    logger.info(f"Webhook job-completed processed for jobId: {payload.jobId}")
    return {"success": True, "jobId": payload.jobId, "status": "completed"}

@router.post("/job-failed", status_code=status.HTTP_200_OK)
async def webhook_job_failed(payload: JobFailedWebhookRequest):
    """
    Called by Cloud Run GPU workers on job failure.
    Updates status="error", records error reason, and refunds deducted credits atomically.
    """
    db = get_db()
    job_ref = db.collection("jobs").document(payload.jobId)
    job_doc = job_ref.get()

    if not job_doc.exists:
        raise HTTPException(status_code=404, detail=f"Job {payload.jobId} not found")

    job_data = job_doc.to_dict()
    user_id = job_data.get("userId")
    cost = job_data.get("cost", 0)

    # Update Firestore job document
    job_ref.update({
        "status": "error",
        "error": payload.error
    })

    # Refund credits to user wallet atomically
    if user_id and cost > 0:
        try:
            refund_credits(user_id=user_id, cost=cost, job_id=payload.jobId, reason=payload.error)
            logger.info(f"Refunded {cost} credits to user {user_id} due to job {payload.jobId} failure.")
        except Exception as err:
            logger.error(f"Error executing refund for job {payload.jobId}: {str(err)}")

    return {"success": True, "jobId": payload.jobId, "status": "error", "refunded": cost}
