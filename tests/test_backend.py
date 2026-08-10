import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.config import settings
from app.services.prompt_service import load_theme_prompt, STATIC_THEMES
from app.workers.celery_worker import process_async_job, celery_app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["firebaseProjectId"] == "ai-studio-637ab"
    assert data["huggingFaceConfigured"] is True

def test_static_theme_lookup():
    preset = load_theme_prompt(1)
    assert preset["id"] == 1
    assert preset["name"] == "Cyberpunk Neon"
    assert "Cyberpunk city" in preset["prompt"]

def test_generate_job_schema_validation():
    # Missing required params should return 400
    response = client.post("/v1/generateJob", json={
        "jobType": "IMAGE_GEN",
        "tier": "FAST",
        "params": {}
    }, headers={"Authorization": "Bearer test_user_12345"})
    assert response.status_code == 400

def test_wallet_endpoint():
    response = client.get("/wallet/test_user_12345")
    assert response.status_code == 200
    data = response.json()
    assert "balance" in data
    assert data["userId"] == "test_user_12345"

def test_webhook_endpoints_schema():
    # Test job-completed webhook model
    completed_res = client.post("/webhook/job-completed", json={
        "jobId": "non_existent_job",
        "outputUrl": "https://storage.googleapis.com/bucket/output.png"
    })
    assert completed_res.status_code == 404

    # Test job-failed webhook model
    failed_res = client.post("/webhook/job-failed", json={
        "jobId": "non_existent_job",
        "error": "GPU worker timeout"
    })
    assert failed_res.status_code == 404

def test_celery_worker_task_registration():
    assert "process_async_job" in celery_app.tasks
