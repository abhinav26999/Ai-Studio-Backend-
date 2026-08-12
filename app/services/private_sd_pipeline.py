import io
import logging
import warnings
import torch
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    import diffusers.logging as diffusers_logging
    import huggingface_hub.utils as hf_logging
    diffusers_logging.set_verbosity_error()
    hf_logging.set_verbosity_error()
except Exception:
    pass

logger = logging.getLogger(__name__)

# Global singleton pipeline instances
_PRIVATE_SD_TXT2IMG_PIPE = None
_PRIVATE_SD_IMG2IMG_PIPE = None

def get_private_sd_device_and_dtype():
    """Detects optimal device and dtype for PyTorch execution."""
    if torch.cuda.is_available():
        return "cuda", torch.float16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float32
    else:
        return "cpu", torch.float32

def get_private_sd_txt2img_pipeline():
    """Lazy loads private SD-Turbo Text-to-Image pipeline."""
    global _PRIVATE_SD_TXT2IMG_PIPE
    if _PRIVATE_SD_TXT2IMG_PIPE is not None:
        return _PRIVATE_SD_TXT2IMG_PIPE

    try:
        from diffusers import AutoPipelineForText2Image
        device, torch_dtype = get_private_sd_device_and_dtype()
        logger.info(f"Loading private SD-Turbo Txt2Img model weights on device='{device}'...")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sd-turbo",
                torch_dtype=torch_dtype,
                variant="fp16" if device == "cuda" else None,
                use_safetensors=True,
                safety_checker=None
            )
            pipe = pipe.to(device)
            pipe.set_progress_bar_config(disable=True)

        if device == "cuda":
            pipe.enable_attention_slicing()

        _PRIVATE_SD_TXT2IMG_PIPE = pipe
        return _PRIVATE_SD_TXT2IMG_PIPE
    except Exception as err:
        logger.error(f"Failed to load Txt2Img pipeline: {str(err)}")
        return None

def get_private_sd_img2img_pipeline():
    """Lazy loads private SD-Turbo Image-to-Image AI transformation pipeline."""
    global _PRIVATE_SD_IMG2IMG_PIPE
    if _PRIVATE_SD_IMG2IMG_PIPE is not None:
        return _PRIVATE_SD_IMG2IMG_PIPE

    try:
        from diffusers import AutoPipelineForImage2Image
        device, torch_dtype = get_private_sd_device_and_dtype()
        logger.info(f"Loading private SD-Turbo Img2Img AI transformation model weights on device='{device}'...")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe = AutoPipelineForImage2Image.from_pretrained(
                "stabilityai/sd-turbo",
                torch_dtype=torch_dtype,
                variant="fp16" if device == "cuda" else None,
                use_safetensors=True,
                safety_checker=None
            )
            pipe = pipe.to(device)
            pipe.set_progress_bar_config(disable=True)

        if device == "cuda":
            pipe.enable_attention_slicing()

        _PRIVATE_SD_IMG2IMG_PIPE = pipe
        return _PRIVATE_SD_IMG2IMG_PIPE
    except Exception as err:
        logger.error(f"Failed to load Img2Img pipeline: {str(err)}")
        return None

def generate_private_sd_image(prompt: str, num_inference_steps: int = 1) -> bytes:
    """Generates 1024x1024 text-to-image AI PNG bytes."""
    pipe = get_private_sd_txt2img_pipeline()
    if pipe is None:
        raise RuntimeError("Private SD-Turbo Txt2Img pipeline is unavailable.")

    logger.info(f"Executing Private SD-Turbo Txt2Img for prompt: '{prompt[:60]}...'")
    with torch.inference_mode():
        image = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=0.0).images[0]

    if image.size != (1024, 1024):
        image = image.resize((1024, 1024), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

def generate_private_sd_img2img(input_image_bytes: bytes, prompt: str, strength: float = 0.55, num_inference_steps: int = 2) -> bytes:
    """
    Executes AI Image-to-Image Transformation (e.g., 'Make girl stand against a wall', 'eating a burger').
    Transforms subject pose, action, outfit, and background guided by user prompt!
    """
    pipe = get_private_sd_img2img_pipeline()
    if pipe is None:
        raise RuntimeError("Private SD-Turbo Img2Img pipeline is unavailable.")

    logger.info(f"Executing Private AI Img2Img Transformation for prompt: '{prompt[:60]}...'")
    init_img = Image.open(io.BytesIO(input_image_bytes)).convert("RGB")
    init_img = init_img.resize((512, 512), Image.Resampling.LANCZOS)

    with torch.inference_mode():
        edited_img = pipe(
            prompt=prompt,
            image=init_img,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=0.0
        ).images[0]

    edited_img = edited_img.resize((1024, 1024), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    edited_img.save(buffer, format="PNG")
    output_bytes = buffer.getvalue()
    logger.info(f"Successfully generated AI Img2Img transformation ({len(output_bytes)} bytes)")
    return output_bytes
