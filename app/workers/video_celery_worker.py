import logging
from datetime import datetime, timezone
from app.config import settings
from app.db.firebase import get_db
from app.services.wallet_service import refund_credits
from app.services.wan_video_processor import generate_wan_video, upload_video_file_to_storage
from app.workers.celery_worker import celery_app

logger = logging.getLogger(__name__)

def get_utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

@celery_app.task(
    name="process_async_video_job",
    bind=True,
    max_retries=2,
    default_retry_delay=5,
    time_limit=180,        # Hard time limit: 3 minutes
    soft_time_limit=150    # Soft time limit: 2.5 minutes
)
def process_async_video_job(self, job_id: str, user_id: str, job_type: str, tier: str, duration: int, payload_data: dict):
    """
    Celery Video Task Worker:
    1. Reads jobs/{jobId} status.
    2. Updates status to 'processing'.
    3. Runs Wan 2.2 Fast Video model inference on self-hosted GPU worker.
    4. Uploads video MP4 blob to Firebase Storage outputs/{userId}/{jobId}/video.mp4.
    5. Writes completed video record into Firestore 'video_gallery' collection.
    6. Updates status to 'completed' with videoUrl in Firestore.
    7. On error/timeout, refunds credits (40 for 10s, 50 for 15s) and updates status to 'error'.
    """
    logger.info(f"Video Celery worker processing job_id={job_id}, duration={duration}s, user_id={user_id}")
    db = get_db()
    job_ref = db.collection("jobs").document(job_id)

    cost = settings.CREDIT_COST_VIDEO_10S if duration == 10 else settings.CREDIT_COST_VIDEO_15S

    try:
        # Step 1: Update status to processing
        job_ref.set({
            "jobId": job_id,
            "userId": user_id,
            "jobType": job_type,
            "tier": tier,
            "duration": duration,
            "status": "processing",
            "updatedAt": get_utc_now_iso()
        }, merge=True)

        params = payload_data.get("params", {}) if isinstance(payload_data, dict) else {}
        processed_prompt = payload_data.get("processedPrompt", {}) if isinstance(payload_data, dict) else {}
        input_images = params.get("images", [])

        # Step 2: Run Wan 2.2 Fast Tier Video model execution
        video_bytes = generate_wan_video(
            expanded_prompt_data=processed_prompt,
            input_image_urls=input_images
        )

        # Step 3: Upload output video MP4 asset to Firebase Storage
        video_url = upload_video_file_to_storage(
            user_id=user_id,
            job_id=job_id,
            filename="video.mp4",
            content_bytes=video_bytes
        )

        # Step 4: Update Firestore jobs doc on completion
        update_data = {
            "status": "completed",
            "videoUrl": video_url,
            "outputUrl": video_url,
            "error": None,
            "updatedAt": get_utc_now_iso()
        }
        job_ref.set(update_data, merge=True)

        # Step 5: Write record into Firestore 'video_gallery' collection
        gallery_ref = db.collection("video_gallery").document(job_id)
        gallery_data = {
            "galleryId": job_id,
            "jobId": job_id,
            "userId": user_id,
            "jobType": job_type,
            "tier": tier,
            "duration": duration,
            "videoUrl": video_url,
            "processedPrompt": processed_prompt,
            "createdAt": get_utc_now_iso()
        }
        gallery_ref.set(gallery_data, merge=True)
        logger.info(f"Created video gallery item in Firestore 'video_gallery/{job_id}'")

        logger.info(f"Video Celery task completed successfully for jobId={job_id}")
        return {"success": True, "jobId": job_id, "videoUrl": video_url}

    except Exception as err:
        logger.error(f"Video Celery worker failed for jobId={job_id}: {str(err)}")
        job_ref.set({
            "status": "error",
            "error": str(err),
            "updatedAt": get_utc_now_iso()
        }, merge=True)

        # Refund credits on failure
        try:
            refund_credits(user_id=user_id, cost=cost, job_id=job_id, reason=f"Video Celery error: {str(err)}")
            logger.info(f"Successfully refunded {cost} credits for failed video jobId={job_id}")
        except Exception as refund_err:
            logger.error(f"Failed to refund credits for video jobId={job_id}: {str(refund_err)}")

        raise self.retry(exc=err) if hasattr(self, 'retry') else err


@celery_app.task(
    name="process_async_face_swap_job",
    bind=True,
    max_retries=1,
    default_retry_delay=5,
    time_limit=300,        # Hard time limit: 5 minutes
    soft_time_limit=270    # Soft time limit: 4.5 minutes
)
def process_async_face_swap_job(self, job_id: str, user_id: str, target_video_url: str, source_image_url: str, cost: int = 30):
    """
    Celery Task Worker for Video Face Swap:
    1. Downloads target video and source character photo.
    2. Runs InsightFace + INSwapper to replace face in each frame.
    3. Uploads output MP4 to Firebase Storage.
    4. Updates Firestore jobs doc to 'completed' with videoUrl.
    5. On error, refunds credits and marks 'error'.
    """
    from app.services.face_swap_processor import swap_faces_in_video

    logger.info(f"Face Swap Celery worker processing job_id={job_id}, user_id={user_id}")
    db = get_db()
    job_ref = db.collection("jobs").document(job_id)

    try:
        # Step 1: Update status to processing
        job_ref.set({
            "jobId": job_id,
            "userId": user_id,
            "jobType": "VIDEO_FACE_SWAP",
            "status": "processing",
            "updatedAt": get_utc_now_iso()
        }, merge=True)

        # Step 2: Execute Face Swap
        video_url = swap_faces_in_video(
            target_video_url=target_video_url,
            source_image_url=source_image_url,
            user_id=user_id,
            job_id=job_id
        )

        # Step 3: Update Firestore doc
        job_ref.set({
            "status": "completed",
            "videoUrl": video_url,
            "outputUrl": video_url,
            "error": None,
            "updatedAt": get_utc_now_iso()
        }, merge=True)

        # Step 4: Write to video gallery
        gallery_ref = db.collection("video_gallery").document(job_id)
        gallery_ref.set({
            "galleryId": job_id,
            "jobId": job_id,
            "userId": user_id,
            "jobType": "VIDEO_FACE_SWAP",
            "videoUrl": video_url,
            "createdAt": get_utc_now_iso()
        }, merge=True)

        logger.info(f"Face Swap Celery task completed successfully for jobId={job_id}")
        return {"success": True, "jobId": job_id, "videoUrl": video_url}

    except Exception as err:
        logger.error(f"Face Swap Celery worker failed for jobId={job_id}: {str(err)}")
        job_ref.set({
            "status": "error",
            "error": str(err),
            "updatedAt": get_utc_now_iso()
        }, merge=True)

        try:
            refund_credits(user_id=user_id, cost=cost, job_id=job_id, reason=f"Face Swap error: {str(err)}")
            logger.info(f"Successfully refunded {cost} credits for failed face swap jobId={job_id}")
        except Exception as refund_err:
            logger.error(f"Failed to refund credits: {str(refund_err)}")

        raise self.retry(exc=err) if hasattr(self, 'retry') else err

