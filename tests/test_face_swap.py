import pytest
from app.models.face_swap_job import GenerateFaceSwapJobRequest, FaceSwapJobParams, FaceSwapJobDocument
from app.services.face_swap_service import get_face_swap_cost
from app.main import app

def test_face_swap_credit_cost():
    """Verify face swap cost is 30 credits."""
    assert get_face_swap_cost() == 30

def test_face_swap_request_validation():
    """Verify GenerateFaceSwapJobRequest schema validation."""
    valid_req = GenerateFaceSwapJobRequest(
        params=FaceSwapJobParams(
            targetVideoUrl="https://example.com/dance.mp4",
            sourceImageUrl="https://example.com/face.jpg",
            title="Dance Face Swap"
        )
    )
    assert valid_req.params.targetVideoUrl == "https://example.com/dance.mp4"
    assert valid_req.params.sourceImageUrl == "https://example.com/face.jpg"
    assert valid_req.params.title == "Dance Face Swap"

def test_face_swap_document_defaults():
    """Verify FaceSwapJobDocument structure and default status."""
    doc = FaceSwapJobDocument(
        jobId="fsjob_test123",
        userId="user_abc",
        params=FaceSwapJobParams(
            targetVideoUrl="https://example.com/dance.mp4",
            sourceImageUrl="https://example.com/face.jpg"
        )
    )
    assert doc.jobId == "fsjob_test123"
    assert doc.status == "pending"
    assert doc.cost == 30
    assert doc.type == "VIDEO_FACE_SWAP"

def test_face_swap_routes_registered():
    """Verify face swap routes exist in FastAPI route table."""
    routes = [route.path for route in app.routes]
    assert "/v1/generateFaceSwapJob" in routes
    assert "/v1/faceSwapJob/{job_id}" in routes

def test_models_loaded_from_cache():
    """Verify local insightface models load successfully."""
    from app.services.face_swap_processor import get_face_analysis, get_inswapper
    app_detector = get_face_analysis()
    swapper = get_inswapper()
    assert app_detector is not None
    assert swapper is not None
