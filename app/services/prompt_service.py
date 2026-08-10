import json
import logging
import httpx
from typing import Dict, Any
from app.config import settings

logger = logging.getLogger(__name__)

# Static Theme Preset Library (O(1) lookup - 0 LLM Cost)
STATIC_THEMES: Dict[int, Dict[str, Any]] = {
    1: {
        "id": 1,
        "name": "Cyberpunk Neon",
        "prompt": "Cyberpunk city street at night, neon signs, wet asphalt reflections, futuristic atmosphere, highly detailed, 8k resolution",
        "negative_prompt": "blurry, low quality, daytime, realistic human distortion",
        "style_tags": ["cyberpunk", "neon", "futuristic"]
    },
    2: {
        "id": 2,
        "name": "Minimalist Studio Product",
        "prompt": "Professional studio product photo, clean pastel background, soft directional lighting, minimal aesthetic, sharp focus, 8k",
        "negative_prompt": "cluttered, shadows, noise, oversaturated",
        "style_tags": ["minimalist", "studio", "clean"]
    },
    3: {
        "id": 3,
        "name": "Fantasy Cinematic",
        "prompt": "Epic fantasy landscape, majestic mountains, glowing ethereal flora, dramatic golden hour lighting, octane render, masterpiece",
        "negative_prompt": "modern structures, power lines, text, watermark",
        "style_tags": ["fantasy", "cinematic", "epic"]
    },
    4: {
        "id": 4,
        "name": "Anime Watercolor",
        "prompt": "Vibrant anime art style, soft watercolor textures, scenic sky with fluffy clouds, high quality illustration",
        "negative_prompt": "3d render, photo, realistic, ugly, deformed",
        "style_tags": ["anime", "watercolor", "illustration"]
    },
    5: {
        "id": 5,
        "name": "3D Isometric Low Poly",
        "prompt": "Cute isometric 3d scene, low poly style, vibrant color palette, blender render, clean lighting, 4k",
        "negative_prompt": "photorealistic, noisy, flat 2d",
        "style_tags": ["3d", "isometric", "lowpoly"]
    }
}

def load_theme_prompt(theme_id: int) -> Dict[str, Any]:
    """O(1) static lookup for theme prompt without calling LLM."""
    if theme_id in STATIC_THEMES:
        return STATIC_THEMES[theme_id]
    # Fallback for unknown theme ID
    return {
        "id": theme_id,
        "name": f"Theme #{theme_id}",
        "prompt": f"Professional artistic rendering in theme style {theme_id}, high resolution",
        "negative_prompt": "low quality, blurry",
        "style_tags": ["preset"]
    }

async def qwen_parse_user_prompt(user_prompt: str) -> Dict[str, Any]:
    """
    Call Hugging Face Serverless Router API for Qwen/Qwen2.5-7B-Instruct
    to transform raw user text into an optimized, safe image generation prompt.
    """
    if not settings.HF_TOKEN:
        logger.warning("HF_TOKEN is not set. Returning basic structured prompt fallback.")
        return {
            "prompt": user_prompt,
            "negative_prompt": "blurry, low quality, distorted",
            "style_tags": ["raw"]
        }

    url = f"{settings.HF_ROUTER_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.HF_TOKEN}",
        "Content-Type": "application/json"
    }

    system_instructions = (
        "You are an expert AI prompt engineer for image models like FLUX.1 Schnell. "
        "Your job is to take raw user input and output ONLY a valid JSON object matching this schema:\n"
        '{"prompt": "<enhanced detailed positive prompt>", "negative_prompt": "<negative terms>", "style_tags": ["tag1", "tag2"]}\n'
        "Do not include markdown block syntax like ```json. Output raw JSON only."
    )

    payload = {
        "model": settings.QWEN_MODEL_ID,
        "messages": [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 300
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code != 200:
                logger.error(f"Hugging Face Router API returned status {response.status_code}: {response.text}")
                return {
                    "prompt": user_prompt,
                    "negative_prompt": "blurry, low quality",
                    "style_tags": ["custom"]
                }
            
            res_data = response.json()
            raw_content = res_data["choices"][0]["message"]["content"].strip()
            
            # Clean possible markdown fence code blocks
            if raw_content.startswith("```"):
                lines = raw_content.splitlines()
                raw_content = "\n".join([line for line in lines if not line.startswith("```")]).strip()
            
            parsed = json.loads(raw_content)
            return {
                "prompt": parsed.get("prompt", user_prompt),
                "negative_prompt": parsed.get("negative_prompt", "blurry, low quality"),
                "style_tags": parsed.get("style_tags", ["ai-enhanced"])
            }
    except Exception as err:
        logger.error(f"Failed to process prompt via Qwen: {str(err)}")
        return {
            "prompt": user_prompt,
            "negative_prompt": "blurry, low quality",
            "style_tags": ["fallback"]
        }

async def process_job_prompt(params_dict: dict) -> Dict[str, Any]:
    """
    Route prompt parsing per Task B2:
    - themeId -> static lookup (O(1), $0 LLM cost)
    - userPrompt -> Qwen 2.5 7B LLM call
    """
    theme_id = params_dict.get("themeId")
    user_prompt = params_dict.get("userPrompt")

    if theme_id is not None:
        logger.info(f"Routing job prompt via static Theme Preset Library (themeId={theme_id})")
        return load_theme_prompt(theme_id)
    elif user_prompt:
        logger.info(f"Routing user prompt via Qwen 2.5 7B Instruct")
        return await qwen_parse_user_prompt(user_prompt)
    else:
        raise ValueError("Job parameters must include either 'themeId' or 'userPrompt'")
