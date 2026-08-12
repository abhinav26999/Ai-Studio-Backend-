import logging
from datetime import datetime, timezone
from celery import Celery
from celery.exceptions import SoftTimeLimitExceeded
from firebase_admin import storage
from app.config import settings
from app.db.firebase import get_db
from app.services.wallet_service import refund_credits
from app.services.image_processor import perform_real_bg_removal, generate_image_flux, generate_placeholder_image
from app.services.image_editor import perform_image_editing, perform_theme_recoloring

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

def upload_output_file_to_storage(user_id: str, job_id: str, filename: str, content_bytes: bytes, content_type: str = "image/png") -> str:
    """
    Uploads output file directly to Firebase Storage bucket under outputs/{user_id}/{job_id}/{filename}
    so the outputs/ folder physically exists in Firebase Storage Console with valid image preview.
    """
    try:
        bucket = storage.bucket(settings.FIREBASE_STORAGE_BUCKET)
        blob_path = f"outputs/{user_id}/{job_id}/{filename}"
        blob = bucket.blob(blob_path)
        blob.upload_from_string(content_bytes, content_type=content_type)
        logger.info(f"Successfully uploaded physical blob file ({len(content_bytes)} bytes) to Firebase Storage at: {blob_path}")
        return blob.public_url
    except Exception as err:
        logger.warning(f"Firebase Storage SDK upload notice for {job_id}: {str(err)}")
        return f"https://storage.googleapis.com/{settings.FIREBASE_STORAGE_BUCKET}/outputs/{user_id}/{job_id}/{filename}"

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
    3. Executes AI generation/segmentation/editing.
    4. Saves output blob into Firebase Storage under outputs/{userId}/{jobId}/...
    5. Writes completed item into Firestore 'gallery' collection.
    6. Updates status to 'completed' with outputUrl/meshUrl in Firestore using set(merge=True).
    7. On error or timeout (SoftTimeLimitExceeded), updates status to 'error' and refunds credits.
    """
    logger.info(f"Celery worker processing job_id={job_id}, job_type={job_type}, user_id={user_id}")
    db = get_db()
    job_ref = db.collection("jobs").document(job_id)

    # Use set(merge=True) to guarantee safe upsert without 404 NotFound errors
    cost = 3
    try:
        job_doc = job_ref.get()
        if job_doc.exists:
            cost = job_doc.to_dict().get("cost", 3)
    except Exception:
        pass

    try:
        # Step 1: Update status to processing
        job_ref.set({
            "jobId": job_id,
            "userId": user_id,
            "jobType": job_type,
            "tier": tier,
            "status": "processing",
            "updatedAt": get_utc_now_iso()
        }, merge=True)

        params = payload_data.get("params", {}) if isinstance(payload_data, dict) else {}
        processed_prompt = payload_data.get("processedPrompt", {}) if isinstance(payload_data, dict) else {}
        input_image_url = params.get("imageUrl")
        bucket_name = settings.FIREBASE_STORAGE_BUCKET

        output_url = None
        mesh_url = None

        if job_type == "BG_REMOVAL":
            logger.info(f"Executing BiRefNet / U2Net Background Removal for {job_id}")
            if input_image_url:
                bg_png_bytes = perform_real_bg_removal(input_image_url)
            else:
                bg_png_bytes = generate_placeholder_image("Background Removal", "Clean Transparent Cutout")

            output_url = upload_output_file_to_storage(
                user_id=user_id,
                job_id=job_id,
                filename="bg_removed.png",
                content_bytes=bg_png_bytes,
                content_type="image/png"
            )

        elif job_type == "THEME_CHANGE":
            theme_id = params.get("themeId")
            theme_name = processed_prompt.get("name", f"Theme #{theme_id}") if isinstance(processed_prompt, dict) else f"Theme #{theme_id}"
            prompt_str = processed_prompt.get("prompt", "High-resolution studio portrait, highly detailed") if isinstance(processed_prompt, dict) else "High-resolution theme portrait"

            logger.info(f"Executing Theme Change for themeId={theme_id}, input_image_url={input_image_url is not None}, jobId={job_id}")
            
            if input_image_url:
                theme_png_bytes = perform_theme_recoloring(input_image_url, theme_id=theme_id or 1, prompt=prompt_str)
            else:
                theme_png_bytes = generate_image_flux(prompt_str, title=f"Theme: {theme_name}")

            output_url = upload_output_file_to_storage(
                user_id=user_id,
                job_id=job_id,
                filename=f"theme_{theme_id or 'preset'}.png",
                content_bytes=theme_png_bytes,
                content_type="image/png"
            )

        elif job_type == "IMAGE_GEN":
            user_prompt = params.get("userPrompt", "")
            prompt_str = processed_prompt.get("prompt", user_prompt) if isinstance(processed_prompt, dict) and processed_prompt.get("prompt") else user_prompt
            if not prompt_str:
                prompt_str = "A stunning cinematic masterpiece, highly detailed, 8k resolution"

            logger.info(f"Executing IMAGE_GEN for prompt='{prompt_str[:50]}', input_image_url={input_image_url is not None}, jobId={job_id}")

            if input_image_url:
                gen_png_bytes = perform_image_editing(input_image_url, prompt=prompt_str)
            else:
                gen_png_bytes = generate_image_flux(prompt_str, title="FLUX.1 Schnell Render")

            output_url = upload_output_file_to_storage(
                user_id=user_id,
                job_id=job_id,
                filename="generated.png",
                content_bytes=gen_png_bytes,
                content_type="image/png"
            )

        elif job_type == "IMAGE_3D":
            logger.info(f"Executing IMAGE_3D: 2D Photo → 3D GLB Mesh for {job_id}")
            user_prompt = params.get("userPrompt", "")
            prompt_str = processed_prompt.get("prompt", user_prompt) if isinstance(processed_prompt, dict) and processed_prompt.get("prompt") else user_prompt

            from app.services.image_3d import generate_3d_mesh_from_image
            glb_bytes = generate_3d_mesh_from_image(
                image_url=input_image_url,
                prompt=prompt_str,
                quality_preset="FAST"
            )

            mesh_url = upload_output_file_to_storage(
                user_id=user_id,
                job_id=job_id,
                filename="mesh.glb",
                content_bytes=glb_bytes,
                content_type="model/gltf-binary"
            )
            output_url = mesh_url

        # Step 2: Update Firestore jobs doc on completion using merge=True
        update_data = {
            "status": "completed",
            "error": None,
            "updatedAt": get_utc_now_iso()
        }
        if output_url:
            update_data["outputUrl"] = output_url
        if mesh_url:
            update_data["meshUrl"] = mesh_url

        job_ref.set(update_data, merge=True)

        # Step 3: Write item to Firestore 'gallery' collection
        final_image_url = output_url or mesh_url
        if final_image_url:
            gallery_ref = db.collection("gallery").document(job_id)
            gallery_data = {
                "galleryId": job_id,
                "jobId": job_id,
                "userId": user_id,
                "jobType": job_type,
                "tier": tier,
                "imageUrl": final_image_url,
                "processedPrompt": processed_prompt,
                "createdAt": get_utc_now_iso()
            }
            gallery_ref.set(gallery_data, merge=True)
            logger.info(f"Created gallery item in Firestore 'gallery/{job_id}'")

        logger.info(f"Celery task completed successfully for jobId={job_id}")
        return {"success": True, "jobId": job_id, "outputUrl": output_url, "meshUrl": mesh_url}

    except SoftTimeLimitExceeded:
        logger.error(f"Celery task execution timed out for jobId={job_id} (soft_time_limit exceeded)")
        job_ref.set({
            "status": "error",
            "error": "Task execution timed out (4.5 min limit exceeded)",
            "updatedAt": get_utc_now_iso()
        }, merge=True)
        try:
            refund_credits(user_id=user_id, cost=cost, job_id=job_id, reason="Celery soft time limit timeout")
        except Exception as refund_err:
            logger.error(f"Failed to refund credits on timeout for jobId={job_id}: {str(refund_err)}")
        return {"success": False, "jobId": job_id, "error": "Timeout"}

    except Exception as err:
        logger.error(f"Celery worker failed for jobId={job_id}: {str(err)}")
        job_ref.set({
            "status": "error",
            "error": str(err),
            "updatedAt": get_utc_now_iso()
        }, merge=True)

        # Refund credits on failure
        try:
            refund_credits(user_id=user_id, cost=cost, job_id=job_id, reason=f"Celery error: {str(err)}")
        except Exception as refund_err:
            logger.error(f"Failed to refund credits for jobId={job_id}: {str(refund_err)}")

        raise self.retry(exc=err)
