import json
import logging
import httpx
from typing import Dict, Any, Optional
from app.config import settings

logger = logging.getLogger(__name__)

GEMINI_VIDEO_STRUCTURING_PROMPT = (
    "You are a expert cinematic AI video prompt engineer. "
    "Your job is to analyze the user's raw scene script and extract video metadata into a strict JSON payload.\n"
    "Respond with ONLY a raw JSON object containing these exact keys:\n"
    "{\n"
    '  "camera": "<camera movement e.g. tracking shot, slow panning, orbital, static>",\n'
    '  "subject": "<main character, item, or focal point of the scene>",\n'
    '  "emotion": "<mood or atmosphere e.g. happy, mysterious, dramatic, uplifting>",\n'
    '  "lighting": "<lighting style e.g. sunset golden hour, neon moonlight, soft studio light>",\n'
    '  "motion": "<movement dynamics e.g. floating particles, wind in hair, slow motion walking>",\n'
    '  "environment": "<setting or backdrop details>"\n'
    "}\n"
    "Do not include markdown triple backticks (```json). Output raw JSON string only."
)

async def gemini_structure_scene_script(scene_script: str) -> Dict[str, Any]:
    """
    Step 1: Gemini 2.5 Flash / Qwen 2.5 7B Structuring Step
    Converts raw user text into structured JSON payload.
    """
    if not scene_script or not scene_script.strip():
        return {
            "camera": "smooth tracking shot",
            "subject": "cinematic scene",
            "emotion": "dramatic",
            "lighting": "natural lighting",
            "motion": "subtle motion",
            "environment": "detailed backdrop"
        }

    # If HF_TOKEN or Gemini API is available, invoke LLM for structuring
    if settings.HF_TOKEN:
        url = f"{settings.HF_ROUTER_URL.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.HF_TOKEN}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": settings.QWEN_MODEL_ID,
            "messages": [
                {"role": "system", "content": GEMINI_VIDEO_STRUCTURING_PROMPT},
                {"role": "user", "content": scene_script}
            ],
            "temperature": 0.2,
            "max_tokens": 250
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.post(url, headers=headers, json=payload)
                if res.status_code == 200:
                    raw_content = res.json()["choices"][0]["message"]["content"].strip()
                    if raw_content.startswith("```"):
                        lines = raw_content.splitlines()
                        raw_content = "\n".join([line for line in lines if not line.startswith("```")]).strip()
                    parsed = json.loads(raw_content)
                    logger.info(f"Successfully structured scene script via LLM: {parsed}")
                    return parsed
        except Exception as err:
            logger.warning(f"Structuring LLM call failed: {str(err)}. Using intelligent fallback parser.")

    # Fallback structuring logic
    return {
        "camera": "smooth tracking shot",
        "subject": scene_script.strip(),
        "emotion": "cinematic",
        "lighting": "vibrant lighting",
        "motion": "fluid dynamic motion",
        "environment": "photorealistic environment"
    }

def expand_video_prompt(structured_json: Dict[str, Any], duration: int = 10, aspect_ratio: str = "16:9") -> Dict[str, Any]:
    """
    Step 2: Prompt Template Engine
    Expands structured JSON into a full production prompt optimized for Wan 2.2 Fast Tier.
    """
    camera = structured_json.get("camera", "smooth tracking shot")
    subject = structured_json.get("subject", "subject")
    emotion = structured_json.get("emotion", "cinematic")
    lighting = structured_json.get("lighting", "natural light")
    motion = structured_json.get("motion", "fluid movement")
    environment = structured_json.get("environment", "scenic background")

    # Production Prompt Template for Wan 2.2
    production_prompt = (
        f"{camera} of {subject} in {environment}, "
        f"{lighting}, {emotion} tone, {motion}, "
        f"masterpiece quality, 8k resolution, photorealistic, Wan 2.2 fast video generation"
    )

    negative_prompt = (
        "blurry, flickering, distorted faces, low frame rate, glitching, static image, "
        "oversaturated, morphing artifacts, low quality video"
    )

    return {
        "structured": structured_json,
        "expanded_prompt": production_prompt,
        "negative_prompt": negative_prompt,
        "model_target": settings.WAN_MODEL_ID,
        "duration": duration,
        "aspect_ratio": aspect_ratio,
        "fps": 24,
        "motion_bucket": 127
    }

async def process_video_job_prompt(params_dict: dict, duration: int = 10, aspect_ratio: str = "16:9") -> Dict[str, Any]:
    """
    Main entry point for Video Prompt Processing:
    1. Runs Gemini Flash / Structurer to produce structured JSON.
    2. Runs Prompt Template Engine to generate full Wan 2.2 expanded prompt.
    """
    scene_script = params_dict.get("sceneScript", "")
    structured_json = await gemini_structure_scene_script(scene_script)
    expanded_payload = expand_video_prompt(structured_json, duration=duration, aspect_ratio=aspect_ratio)
    return expanded_payload
