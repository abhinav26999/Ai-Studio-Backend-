import io
import sys
import os
import warnings
warnings.filterwarnings("ignore")
import logging
import torch
import numpy as np
from PIL import Image

# Add vendor/TripoSR to Python path
vendor_triposr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../vendor/TripoSR"))
if vendor_triposr_path not in sys.path:
    sys.path.insert(0, vendor_triposr_path)

from tsr.system import TSR
from tsr.utils import resize_foreground, to_gradio_3d_orientation

logger = logging.getLogger(__name__)

_triposr_model = None

def get_triposr_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def get_triposr_model() -> TSR:
    global _triposr_model
    if _triposr_model is None:
        device = get_triposr_device()
        logger.info(f"Loading TripoSR 3D AI model on device='{device}'...")
        _triposr_model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt"
        )
        _triposr_model.renderer.set_chunk_size(0)
        _triposr_model.to(device)
        logger.info("TripoSR 3D AI model loaded successfully!")
    return _triposr_model

def generate_3d_mesh_triposr(image_pil: Image.Image, foreground_ratio: float = 0.88, mc_resolution: int = 512) -> bytes:
    """
    Generates a high-quality 3D GLB mesh from a 2D image using the self-hosted TripoSR model.
    """
    device = get_triposr_device()
    model = get_triposr_model()

    # Step 1: Ensure RGBA & preprocess foreground scaling
    image_rgba = image_pil.convert("RGBA")
    processed_image = resize_foreground(image_rgba, foreground_ratio)

    # Prepare 3-channel input as expected by TripoSR
    img_np = np.array(processed_image).astype(np.float32) / 255.0
    img_rgb = img_np[:, :, :3] * img_np[:, :, 3:4] + (1 - img_np[:, :, 3:4]) * 0.5
    input_pil = Image.fromarray((img_rgb * 255.0).astype(np.uint8))

    # Step 2: Run TripoSR neural model inference
    logger.info("Running TripoSR neural 3D reconstruction...")
    with torch.no_grad():
        scene_codes = model([input_pil], device=device)

    # Step 3: Extract 3D Mesh surface
    logger.info(f"Extracting 3D surface geometry via TripoSR (resolution={mc_resolution})...")
    meshes = model.extract_mesh(scene_codes, True, resolution=mc_resolution)
    mesh = meshes[0]

    # Step 4: Fix 3D mesh orientation (stand upright facing front along X/Z axis)
    mesh = to_gradio_3d_orientation(mesh)

    # Step 5: Export to GLB bytes
    buffer = io.BytesIO()
    mesh.export(buffer, file_type="glb")
    glb_bytes = buffer.getvalue()
    logger.info(f"Successfully generated TripoSR 3D GLB Mesh ({len(glb_bytes)} bytes)")
    return glb_bytes

