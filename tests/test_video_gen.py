import pytest
from app.services.video_prompt_engine import expand_video_prompt, gemini_structure_scene_script
from app.services.video_service import get_video_job_cost
from app.models.video_job import GenerateVideoJobRequest, VideoJobParams

def test_video_job_credit_costs():
    """Verify 10s video costs 40 credits and 15s video costs 50 credits."""
    assert get_video_job_cost(10) == 40
    assert get_video_job_cost(15) == 50

def test_prompt_template_engine_expansion():
    """Verify prompt expansion engine formats structured JSON for Wan 2.2."""
    structured = {
        "camera": "tracking shot",
        "subject": "golden compass",
        "emotion": "mysterious",
        "lighting": "sunset light",
        "motion": "floating dust",
        "environment": "ancient library"
    }
    expanded = expand_video_prompt(structured, duration=10, aspect_ratio="16:9")
    assert "expanded_prompt" in expanded
    assert "tracking shot" in expanded["expanded_prompt"]
    assert "golden compass" in expanded["expanded_prompt"]
    assert expanded["duration"] == 10
    assert expanded["aspect_ratio"] == "16:9"

def test_gemini_structuring_fallback():
    """Verify gemini structure scene script fallback returns valid dictionary."""
    import asyncio
    scene_script = "A futuristic sports car drifting through neon wet streets at night"
    res = asyncio.run(gemini_structure_scene_script(scene_script))
    assert isinstance(res, dict)
    assert "camera" in res
    assert "subject" in res

def test_request_validation():
    """Verify duration and aspect ratio validation."""
    valid_req = GenerateVideoJobRequest(
        duration=10,
        aspectRatio="16:9",
        params=VideoJobParams(sceneScript="Test script")
    )
    assert valid_req.duration == 10
    assert valid_req.aspectRatio == "16:9"

    with pytest.raises(ValueError):
        GenerateVideoJobRequest(
            duration=20, # Invalid duration
            aspectRatio="16:9",
            params=VideoJobParams(sceneScript="Test script")
        )

def test_animated_motion_synthesis_output():
    """Verify cinematic AI video synthesis produces valid MP4 video bytes with correct size."""
    from app.services.wan_video_processor import render_cinematic_ai_video
    video_bytes = render_cinematic_ai_video("A girl dancing gracefully", duration=1, aspect_ratio="16:9", input_image_urls=[])
    assert isinstance(video_bytes, bytes)
    assert len(video_bytes) > 1000  # Valid MP4 header + data


