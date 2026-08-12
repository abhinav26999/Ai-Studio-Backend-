import io
import urllib.parse
import logging
import httpx
import numpy as np
from PIL import Image
import cv2

logger = logging.getLogger(__name__)

COLOR_MAP_HSV = {
    "black": "black",
    "white": "white",
    "gray": "gray",
    "grey": "gray",
    "green": 60,
    "emerald": 65,
    "lime": 45,
    "blue": 120,
    "navy": 115,
    "sky blue": 100,
    "purple": 140,
    "violet": 145,
    "magenta": 150,
    "pink": 160,
    "red": 0,
    "crimson": 175,
    "orange": 15,
    "yellow": 30,
    "gold": 25,
    "cyan": 90,
    "teal": 85,
    "brown": 10,
}

COLOR_KEYS_ORDERED = [
    "sky blue", "emerald", "lime", "navy", "magenta", "violet", "crimson",
    "green", "blue", "purple", "pink", "red", "yellow", "orange", "gold",
    "cyan", "teal", "brown", "black", "white", "gray", "grey"
]

def download_image_bytes(image_url: str) -> bytes:
    """Downloads image bytes safely via Firebase Storage SDK or HTTP GET fallback."""
    try:
        from firebase_admin import storage
        from app.config import settings
        bucket = storage.bucket(settings.FIREBASE_STORAGE_BUCKET)

        blob_path = None
        if "/o/" in image_url:
            path_part = image_url.split("/o/")[1].split("?")[0]
            blob_path = urllib.parse.unquote(path_part)
        elif settings.FIREBASE_STORAGE_BUCKET in image_url:
            path_part = image_url.split(settings.FIREBASE_STORAGE_BUCKET)[1].lstrip("/")
            blob_path = urllib.parse.unquote(path_part)

        if blob_path:
            blob = bucket.blob(blob_path)
            if blob.exists():
                logger.info(f"Downloaded image bytes directly via Firebase Storage SDK for: {blob_path}")
                return blob.download_as_bytes()
    except Exception as err:
        logger.warning(f"Firebase Storage SDK download notice: {str(err)}")

    try:
        resp = httpx.get(image_url, timeout=20.0, follow_redirects=True)
        if resp.status_code == 200:
            return resp.content
    except Exception as err:
        logger.error(f"HTTP GET download error: {str(err)}")

    raise ValueError(f"Could not download image bytes for URL: {image_url}")

def optimize_user_prompt(prompt: str) -> str:
    """Optimizes short user prompts for AI image generation."""
    if not prompt:
        return "A cinematic masterpiece, highly detailed, 8k resolution"
    p = prompt.strip()
    p_lower = p.lower()

    if not any(w in p_lower for w in ["detailed", "resolution", "8k", "masterpiece", "photography"]):
        return f"{p}, highly detailed, cinematic lighting, 8k resolution, photorealistic masterpiece"

    return p

def clean_background_prompt(prompt: str) -> str:
    """Strips person/subject nouns from prompt so generated backgrounds contain zero duplicate people."""
    if not prompt:
        return "sleek modern architectural interior background, elegant ambient studio lighting"
    p = prompt.lower()
    for noun in ["a beautiful girl", "beautiful girl", "a girl", "girl", "a boy", "boy", "a man", "man", "a woman", "woman", "person", "human", "model", "standing", "sitting", "dancing", "wearing"]:
        p = p.replace(noun, "")
    p = p.strip(" ,.-")
    if not p:
        p = "sleek modern architectural wall background"
    return f"{p}, empty architectural scene, clean background photography, studio lighting"

def is_wall_or_bg_recolor_intent(prompt: str) -> bool:
    """Checks if user prompt specifically requests background/wall color change."""
    if not prompt:
        return False
    p = prompt.lower()
    if "wall" in p or "background" in p or "backdrop" in p or "room color" in p or "wall color" in p or "bg color" in p:
        if "color" in p or "colour" in p or "change" in p or "recolor" in p or "to " in p or "in " in p or any(c in p for c in COLOR_KEYS_ORDERED):
            return True
    return False

def is_garment_recolor_intent(prompt: str) -> bool:
    """Checks if user prompt specifically requests outfit/garment color change."""
    if not prompt:
        return False
    p = prompt.lower()
    if "dress" in p or "shirt" in p or "top" in p or "hoodie" in p or "jacket" in p or "outfit" in p or "clothes" in p or "cloth" in p:
        if "color" in p or "colour" in p or "change" in p or "recolor" in p or "to " in p or "in " in p:
            return True
    return False

def parse_target_color(prompt: str):
    """
    Parses target color and dark/light modifier from prompt string.
    Returns tuple: (target_color_or_hue, modifier_str).
    """
    if not prompt:
        return 60, "normal"
    prompt_lower = prompt.lower()

    is_dark = "dark" in prompt_lower or "deep" in prompt_lower
    is_light = "light" in prompt_lower or "bright" in prompt_lower
    modifier = "dark" if is_dark else ("light" if is_light else "normal")

    for color_name in COLOR_KEYS_ORDERED:
        if color_name in prompt_lower:
            target = COLOR_MAP_HSV[color_name]
            return target, modifier

    return 60, modifier

def perform_background_wall_recoloring(input_bytes: bytes, prompt: str) -> bytes:
    """
    PERFECT WALL & BACKGROUND RECOLORING ENGINE
    Recolors ONLY the background wall to requested target color and tone (dark green, light blue, black, etc.)
    while keeping the foreground person 100% UNTOUCHED.
    """
    try:
        from app.services.onnx_bg_remover import remove_background_onnx
        logger.info(f"Executing Background/Wall Recoloring for prompt: '{prompt}'...")

        orig_pil = Image.open(io.BytesIO(input_bytes)).convert("RGB")
        img_np = np.array(orig_pil)

        cutout_bytes = remove_background_onnx(input_bytes)
        cutout_pil = Image.open(io.BytesIO(cutout_bytes)).convert("RGBA")
        subject_mask = np.array(cutout_pil.split()[3])

        bg_mask = cv2.bitwise_not(subject_mask)
        bg_mask = cv2.GaussianBlur(bg_mask, (11, 11), 0)

        target_color, modifier = parse_target_color(prompt)
        logger.info(f"Parsed target wall color '{target_color}' with modifier '{modifier}' for prompt: '{prompt}'")

        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        hsv_edited = hsv.copy()

        if target_color == "black":
            hsv_edited[:, :, 1] = np.where(bg_mask > 20, 10, hsv_edited[:, :, 1])
            hsv_edited[:, :, 2] = np.where(bg_mask > 20, np.clip(hsv_edited[:, :, 2].astype(int) * 0.2, 10, 50), hsv_edited[:, :, 2])
        elif target_color == "white":
            hsv_edited[:, :, 1] = np.where(bg_mask > 20, 10, hsv_edited[:, :, 1])
            hsv_edited[:, :, 2] = np.where(bg_mask > 20, 240, hsv_edited[:, :, 2])
        elif target_color == "gray":
            hsv_edited[:, :, 1] = np.where(bg_mask > 20, 15, hsv_edited[:, :, 1])
            hsv_edited[:, :, 2] = np.where(bg_mask > 20, 128, hsv_edited[:, :, 2])
        else:
            target_hue = int(target_color) if isinstance(target_color, (int, float)) else 60
            hsv_edited[:, :, 0] = np.where(bg_mask > 20, target_hue, hsv_edited[:, :, 0])

            if modifier == "dark":
                # Rich Dark Forest / Navy Tone
                hsv_edited[:, :, 1] = np.where(bg_mask > 20, np.clip(hsv_edited[:, :, 1].astype(int) + 60, 140, 255), hsv_edited[:, :, 1])
                hsv_edited[:, :, 2] = np.where(bg_mask > 20, np.clip(hsv_edited[:, :, 2].astype(float) * 0.4, 15, 80), hsv_edited[:, :, 2])
            elif modifier == "light":
                # Soft Light Tone
                hsv_edited[:, :, 1] = np.where(bg_mask > 20, np.clip(hsv_edited[:, :, 1].astype(int) * 0.6, 20, 110), hsv_edited[:, :, 1])
                hsv_edited[:, :, 2] = np.where(bg_mask > 20, np.clip(hsv_edited[:, :, 2].astype(int) * 1.3 + 50, 170, 255), hsv_edited[:, :, 2])
            else:
                # Standard Vibrant Tone
                hsv_edited[:, :, 1] = np.where(bg_mask > 20, np.clip(hsv_edited[:, :, 1].astype(int) + 40, 50, 255), hsv_edited[:, :, 1])

        result_rgb = cv2.cvtColor(hsv_edited, cv2.COLOR_HSV2RGB)
        result_pil = Image.fromarray(result_rgb)

        result_rgba = result_pil.convert("RGBA")
        final_composite = Image.alpha_composite(result_rgba, cutout_pil).convert("RGB")

        buffer = io.BytesIO()
        final_composite.save(buffer, format="PNG")
        output_bytes = buffer.getvalue()
        logger.info(f"Successfully generated background wall recolored image ({len(output_bytes)} bytes)")
        return output_bytes

    except Exception as err:
        logger.error(f"Wall recoloring error: {str(err)}")
        return perform_face_preserving_scene_change(input_bytes, prompt)

def perform_face_preserving_scene_change(input_bytes: bytes, prompt: str) -> bytes:
    """
    FACE & IDENTITY PRESERVING SCENE SUBSTITUTION ENGINE
    """
    try:
        from app.services.onnx_bg_remover import remove_background_onnx
        from app.services.private_sd_pipeline import generate_private_sd_image

        logger.info("Extracting face-preserving person cutout via ONNX U2Net...")
        cutout_bytes = remove_background_onnx(input_bytes)
        person_cutout = Image.open(io.BytesIO(cutout_bytes)).convert("RGBA")
        cutout_w, cutout_h = person_cutout.size

        clean_bg_p = clean_background_prompt(prompt)
        logger.info(f"Generating clean prompt background via Private SD-Turbo for: '{clean_bg_p}'...")
        bg_bytes = generate_private_sd_image(clean_bg_p, num_inference_steps=2)

        bg_img = Image.open(io.BytesIO(bg_bytes)).convert("RGBA")
        bg_img = bg_img.resize((cutout_w, cutout_h), Image.Resampling.LANCZOS)

        final_composite = Image.alpha_composite(bg_img, person_cutout)
        final_rgb = final_composite.convert("RGB")

        buffer = io.BytesIO()
        final_rgb.save(buffer, format="PNG")
        output_bytes = buffer.getvalue()
        logger.info(f"Successfully generated face-preserving composite image ({len(output_bytes)} bytes)")
        return output_bytes

    except Exception as err:
        logger.error(f"Face-preserving scene change notice: {str(err)}")
        from app.services.private_sd_pipeline import generate_private_sd_image
        return generate_private_sd_image(optimize_user_prompt(prompt))

def perform_image_editing(image_url: str, prompt: str) -> bytes:
    """
    UNIVERSAL 4-ROUTE SOTA AI IMAGE EDITOR
    Route 1: Background / Wall Recoloring (e.g. 'Change the wall color to dark green').
    Route 2: Garment / Outfit Recoloring (e.g. 'change dress color to black').
    Route 3: Scene / Pose Environment Substitution (e.g. 'standing on a beach').
    Route 4: Text-to-Image Fallback.
    """
    enhanced_prompt = optimize_user_prompt(prompt)

    try:
        logger.info(f"Downloading source image for image editing from: {image_url}")
        input_bytes = download_image_bytes(image_url)
    except Exception as download_err:
        logger.warning(f"Could not download input image ({str(download_err)}). Fallback to Private SD Txt2Img for prompt: '{enhanced_prompt}'")
        from app.services.private_sd_pipeline import generate_private_sd_image
        return generate_private_sd_image(enhanced_prompt)

    try:
        # Route 1: Background or Wall Recoloring Prompt
        if is_wall_or_bg_recolor_intent(prompt):
            logger.info(f"Prompt '{prompt}' detected as Background/Wall Recolor. Routing to Wall Recoloring Engine...")
            return perform_background_wall_recoloring(input_bytes=input_bytes, prompt=prompt)

        # Route 2: Garment or Clothing Recoloring Prompt
        if is_garment_recolor_intent(prompt):
            logger.info(f"Prompt '{prompt}' detected as Garment Recolor. Routing to Garment Recoloring Segmenter...")
            orig_pil = Image.open(io.BytesIO(input_bytes))
            has_alpha = "A" in orig_pil.getbands()

            if has_alpha:
                rgba_img = orig_pil.convert("RGBA")
                alpha_channel = rgba_img.split()[3]
                rgb_img = rgba_img.convert("RGB")
            else:
                rgb_img = orig_pil.convert("RGB")
                alpha_channel = None

            img_np = np.array(rgb_img)
            h, w, _ = img_np.shape

            try:
                from app.services.onnx_bg_remover import remove_background_onnx
                cutout_bytes = remove_background_onnx(input_bytes)
                cutout_pil = Image.open(io.BytesIO(cutout_bytes)).convert("RGBA")
                subject_mask_np = np.array(cutout_pil.split()[3])
            except Exception as bg_err:
                logger.warning(f"Subject cutout mask notice: {str(bg_err)}. Using full mask fallback.")
                subject_mask_np = np.full((h, w), 255, dtype=np.uint8)

            nonzero_y, _ = np.nonzero(subject_mask_np > 20)
            if len(nonzero_y) > 0:
                person_top = int(np.min(nonzero_y))
                person_bottom = int(np.max(nonzero_y))
                person_height = person_bottom - person_top
                head_bottom = person_top + int(person_height * 0.45)
            else:
                head_bottom = int(h * 0.45)

            exclusion_mask = np.zeros((h, w), dtype=np.uint8)
            exclusion_mask[0:head_bottom, :] = 255

            ycrcb = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
            skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
            skin_mask = cv2.dilate(skin_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
            exclusion_mask = cv2.bitwise_or(exclusion_mask, skin_mask)

            hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
            sat = hsv[:, :, 1]
            val = hsv[:, :, 2]

            target_color, modifier = parse_target_color(prompt)
            logger.info(f"Parsed target garment color '{target_color}' with modifier '{modifier}' for prompt: '{prompt}'")

            garment_mask = cv2.bitwise_and(subject_mask_np, cv2.inRange(sat, 20, 255))
            garment_mask = cv2.bitwise_and(garment_mask, cv2.inRange(val, 20, 255))
            garment_mask = cv2.bitwise_and(garment_mask, cv2.bitwise_not(exclusion_mask))

            close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
            garment_mask = cv2.morphologyEx(garment_mask, cv2.MORPH_CLOSE, close_kernel)
            garment_mask = cv2.dilate(garment_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
            garment_mask = cv2.GaussianBlur(garment_mask, (11, 11), 0)

            hsv_edited = hsv.copy()

            if target_color == "black":
                hsv_edited[:, :, 1] = np.where(garment_mask > 20, 10, hsv_edited[:, :, 1])
                hsv_edited[:, :, 2] = np.where(garment_mask > 20, np.clip(hsv_edited[:, :, 2].astype(int) * 0.15, 10, 45), hsv_edited[:, :, 2])
            elif target_color == "white":
                hsv_edited[:, :, 1] = np.where(garment_mask > 20, 10, hsv_edited[:, :, 1])
                hsv_edited[:, :, 2] = np.where(garment_mask > 20, np.clip(hsv_edited[:, :, 2].astype(int) * 1.5 + 180, 200, 255), hsv_edited[:, :, 2])
            elif target_color == "gray":
                hsv_edited[:, :, 1] = np.where(garment_mask > 20, 15, hsv_edited[:, :, 1])
                hsv_edited[:, :, 2] = np.where(garment_mask > 20, 128, hsv_edited[:, :, 2])
            else:
                target_hue = int(target_color) if isinstance(target_color, (int, float)) else 60
                hsv_edited[:, :, 0] = np.where(garment_mask > 20, target_hue, hsv_edited[:, :, 0])

                if modifier == "dark":
                    hsv_edited[:, :, 1] = np.where(garment_mask > 20, np.clip(hsv_edited[:, :, 1].astype(int) + 60, 140, 255), hsv_edited[:, :, 1])
                    hsv_edited[:, :, 2] = np.where(garment_mask > 20, np.clip(hsv_edited[:, :, 2].astype(float) * 0.4, 15, 80), hsv_edited[:, :, 2])
                else:
                    hsv_edited[:, :, 1] = np.where(garment_mask > 20, np.clip(hsv_edited[:, :, 1].astype(int) + 35, 40, 255), hsv_edited[:, :, 1])

            result_np = cv2.cvtColor(hsv_edited, cv2.COLOR_HSV2RGB)
            result_img = Image.fromarray(result_np)

            if alpha_channel:
                result_img = result_img.convert("RGBA")
                result_img.putalpha(alpha_channel)

            buffer = io.BytesIO()
            result_img.save(buffer, format="PNG")
            output_bytes = buffer.getvalue()
            logger.info(f"Successfully generated universal edited image ({len(output_bytes)} bytes)")
            return output_bytes

        # Route 3: Scene / Environment Substitution
        logger.info(f"Prompt '{prompt}' detected as Scene/Environment edit. Routing to Face-Preserving Scene Replacement Engine...")
        return perform_face_preserving_scene_change(input_bytes=input_bytes, prompt=prompt)

    except Exception as err:
        logger.error(f"Universal image editing fallback: {str(err)}")
        from app.services.private_sd_pipeline import generate_private_sd_image
        return generate_private_sd_image(enhanced_prompt)

def perform_theme_recoloring(image_url: str, theme_id: int = 1, prompt: str = "") -> bytes:
    """Universal Theme Style Transfer Engine."""
    enhanced_prompt = optimize_user_prompt(prompt)
    try:
        logger.info(f"Downloading source image for theme recoloring from: {image_url}")
        input_bytes = download_image_bytes(image_url)
    except Exception as download_err:
        logger.warning(f"Theme recoloring download notice: {str(download_err)}. Fallback to Private SD Txt2Img")
        from app.services.private_sd_pipeline import generate_private_sd_image
        return generate_private_sd_image(enhanced_prompt)

    try:
        orig_pil = Image.open(io.BytesIO(input_bytes))
        has_alpha = "A" in orig_pil.getbands()

        if has_alpha:
            rgba_img = orig_pil.convert("RGBA")
            alpha_channel = rgba_img.split()[3]
            rgb_img = rgba_img.convert("RGB")
        else:
            rgb_img = orig_pil.convert("RGB")
            alpha_channel = None

        img_np = np.array(rgb_img)
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)

        if theme_id == 1 or "blue" in prompt.lower() or "cyberpunk" in prompt.lower() or "neon" in prompt.lower():
            hsv[:, :, 0] = (hsv[:, :, 0] + 90) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
        elif theme_id == 2 or "vintage" in prompt.lower() or "amber" in prompt.lower():
            hsv[:, :, 0] = np.clip(hsv[:, :, 0] * 0.4 + 15, 0, 180)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.8, 0, 255)
        else:
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0, 255)

        result_np = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        result_img = Image.fromarray(result_np)

        if alpha_channel:
            result_img = result_img.convert("RGBA")
            result_img.putalpha(alpha_channel)

        buffer = io.BytesIO()
        result_img.save(buffer, format="PNG")
        output_bytes = buffer.getvalue()
        logger.info(f"Successfully generated theme-recolored image ({len(output_bytes)} bytes)")
        return output_bytes

    except Exception as err:
        logger.error(f"Theme recoloring error for {image_url}: {str(err)}")
        from app.services.private_sd_pipeline import generate_private_sd_image
        return generate_private_sd_image(enhanced_prompt)
