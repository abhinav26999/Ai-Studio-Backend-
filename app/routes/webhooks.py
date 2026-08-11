import logging
from typing import Optional
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from app.db.firebase import get_db
from app.services.wallet_service import refund_credits

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook", tags=["webhooks"])

def get_utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

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
    Also creates an entry in the 'gallery' Firestore collection.
    """
    db = get_db()
    job_ref = db.collection("jobs").document(payload.jobId)
    job_doc = job_ref.get()

    if not job_doc.exists:
        raise HTTPException(status_code=404, detail=f"Job {payload.jobId} not found")

    job_data = job_doc.to_dict()
    user_id = job_data.get("userId")
    job_type = job_data.get("type")
    tier = job_data.get("tier", "FAST")
    processed_prompt = job_data.get("processedPrompt", {})

    update_data = {
        "status": "completed",
        "error": None,
        "updatedAt": get_utc_now_iso()
    }
    if payload.outputUrl:
        update_data["outputUrl"] = payload.outputUrl
    if payload.meshUrl:
        update_data["meshUrl"] = payload.meshUrl

    job_ref.update(update_data)

    # Write entry to Firestore 'gallery' collection
    final_url = payload.outputUrl or payload.meshUrl
    if final_url and user_id:
        gallery_ref = db.collection("gallery").document(payload.jobId)
        gallery_data = {
            "galleryId": payload.jobId,
            "jobId": payload.jobId,
            "userId": user_id,
            "jobType": job_type,
            "tier": tier,
            "imageUrl": final_url,
            "processedPrompt": processed_prompt,
            "createdAt": get_utc_now_iso()
        }
        gallery_ref.set(gallery_data)
        logger.info(f"Webhook created gallery doc for jobId: {payload.jobId}")

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
        "error": payload.error,
        "updatedAt": get_utc_now_iso()
    })

    # Refund credits to user wallet atomically
    if user_id and cost > 0:
        try:
            refund_credits(user_id=user_id, cost=cost, job_id=payload.jobId, reason=payload.error)
            logger.info(f"Refunded {cost} credits to user {user_id} due to job {payload.jobId} failure.")
        except Exception as err:
            logger.error(f"Error executing refund for job {payload.jobId}: {str(err)}")

    return {"success": True, "jobId": payload.jobId, "status": "error", "refunded": cost}
