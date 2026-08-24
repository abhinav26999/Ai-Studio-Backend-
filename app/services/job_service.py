import uuid
import logging
from datetime import datetime, timezone
from fastapi import HTTPException
from app.config import settings
from app.db.firebase import get_db, firestore
from app.models.job import JobType, GenerateJobRequest, JobDocument, JobStatus
from app.services.wallet_service import check_and_deduct_credits
from app.services.prompt_service import process_job_prompt
from app.services.queue_service import enqueue_job_task

logger = logging.getLogger(__name__)

def get_utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def get_job_cost(job_type: JobType) -> int:
    """Map Fast Tier credit costs per contract."""
    if job_type == JobType.IMAGE_GEN:
        return settings.CREDIT_COST_IMAGE_GEN
    elif job_type == JobType.THEME_CHANGE:
        return settings.CREDIT_COST_THEME_CHANGE
    elif job_type == JobType.BG_REMOVAL:
        return settings.CREDIT_COST_BG_REMOVAL
    elif job_type == JobType.MESH_GEN:
        return settings.CREDIT_COST_MESH_GEN
    elif job_type == JobType.LIP_SYNC:
        return settings.CREDIT_COST_LIP_SYNC
    return 3


async def create_and_enqueue_job(user_id: str, request: GenerateJobRequest) -> dict:
    """
    Core job orchestrator:
    1. Validate input parameters (either userPrompt or themeId required).
    2. Calculate cost based on JobType.
    3. Perform atomic credit deduction.
    4. Process prompt (static lookup for themeId, Qwen 2.5 7B for userPrompt).
    5. Write jobs/{jobId} to Firestore.
    6. Enqueue task for Celery / Redis GPU worker processing.
    7. Return immediately to caller.
    """
    if (
        request.params.themeId is None 
        and not request.params.userPrompt 
        and not request.params.imageUrl 
        and request.jobType != JobType.BG_REMOVAL
    ):
        raise HTTPException(
            status_code=400,
            detail="Job parameters must specify either 'themeId' (preset path), 'userPrompt' (flexible path), or 'imageUrl'."
        )

    job_id = f"job_{uuid.uuid4().hex[:12]}"
    cost = get_job_cost(request.jobType)
    created_at = get_utc_now_iso()

    # Step 1: Deduct credits atomically
    check_and_deduct_credits(user_id=user_id, cost=cost, job_id=job_id)

    # Step 2: Route prompt
    processed_prompt = await process_job_prompt(request.params.dict(), job_type=request.jobType.value)

    # Step 3: Create Firestore job document
    job_doc_data = {
        "jobId": job_id,
        "userId": user_id,
        "type": request.jobType.value,
        "tier": request.tier.value,
        "status": JobStatus.PENDING.value,
        "cost": cost,
        "params": request.params.dict(),
        "processedPrompt": processed_prompt,
        "outputUrl": None,
        "meshUrl": None,
        "error": None,
        "createdAt": created_at
    }

    db = get_db()
    db.collection("jobs").document(job_id).set(job_doc_data)
    logger.info(f"Created job doc {job_id} for user {user_id} with cost {cost}")

    # Step 4: Enqueue task
    enqueue_job_task(
        job_id=job_id,
        user_id=user_id,
        job_type=request.jobType.value,
        tier=request.tier.value,
        payload_data={
            "params": request.params.dict(),
            "processedPrompt": processed_prompt
        }
    )

    return {
        "jobId": job_id,
        "status": JobStatus.PENDING.value,
        "cost": cost,
        "createdAt": created_at
    }

def get_job_by_id(job_id: str) -> dict:
    """Retrieve job by ID from Firestore."""
    db = get_db()
    doc = db.collection("jobs").document(job_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return doc.to_dict()

def list_user_jobs(user_id: str, limit: int = 20, offset: int = 0) -> dict:
    """Query user jobs from Firestore with in-memory sorting to avoid composite index requirements."""
    db = get_db()
    query = db.collection("jobs").where(filter=firestore.FieldFilter("userId", "==", user_id))
    docs = query.stream()
    jobs_list = [doc.to_dict() for doc in docs]
    jobs_list.sort(key=lambda x: x.get("createdAt", ""), reverse=True)
    paginated_list = jobs_list[offset:offset+limit]
    return {
        "jobs": paginated_list,
        "limit": limit,
        "offset": offset,
        "count": len(paginated_list),
        "total": len(jobs_list)
    }


