import os
import io
import time
import logging
import tempfile
import urllib.request
import httpx
import numpy as np
from PIL import Image
import imageio
from firebase_admin import storage
from app.config import settings

logger = logging.getLogger(__name__)

_face_analysis = None
_inswapper = None

def get_face_analysis():
    """Lazy load FaceAnalysis detector (buffalo_l)."""
    global _face_analysis
    if _face_analysis is not None:
        return _face_analysis

    import insightface
    from insightface.app import FaceAnalysis

    logger.info("Initializing InsightFace FaceAnalysis (buffalo_l)...")
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    _face_analysis = app
    logger.info("FaceAnalysis initialized successfully!")
    return _face_analysis


def get_inswapper():
    """Lazy load INSwapper model (inswapper_128.onnx)."""
    global _inswapper
    if _inswapper is not None:
        return _inswapper

    import insightface

    model_dir = os.path.expanduser("~/.insightface/models")
    os.makedirs(model_dir, exist_ok=True)
    swapper_path = os.path.join(model_dir, "inswapper_128.onnx")

    if not os.path.exists(swapper_path):
        url = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
        logger.info(f"Downloading inswapper_128.onnx to {swapper_path}...")
        urllib.request.urlretrieve(url, swapper_path)
        logger.info("Downloaded inswapper_128.onnx successfully!")

    logger.info(f"Loading INSwapper model from {swapper_path}...")
    swapper = insightface.model_zoo.get_model(swapper_path, download=False, providers=["CPUExecutionProvider"])
    _inswapper = swapper
    logger.info("INSwapper model loaded successfully!")
    return _inswapper


MIXKIT_FALLBACK_MAP = {
    "41220": "templates/dance/hip_hop_dance.mp4",
    "41235": "templates/dance/shuffle_dance.mp4",
    "41215": "templates/dance/ballet_spin.mp4",
    "41225": "templates/dance/breakdance_power.mp4",
    "41230": "templates/dance/kpop_stage.mp4",
    "41240": "templates/dance/salsa_fiesta.mp4",
}

DEFAULT_DANCE_TEMPLATE_BLOB = "templates/dance/hip_hop_dance.mp4"


def download_file_to_temp(url: str, suffix: str = ".tmp") -> str:
    """Download a remote URL or Firebase Storage blob to a temporary local file."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name

    bucket_name = settings.FIREBASE_STORAGE_BUCKET
    bucket = storage.bucket(bucket_name) if bucket_name else None

    # 1. Check if this is a known Mixkit URL and redirect to our reliable Firebase template
    if "mixkit" in url:
        for key, blob_target in MIXKIT_FALLBACK_MAP.items():
            if key in url:
                try:
                    blob = bucket.blob(blob_target)
                    blob.download_to_filename(tmp_path)
                    logger.info(f"Mapped Mixkit URL to Firebase template: {blob_target}")
                    return tmp_path
                except Exception as map_err:
                    logger.warning(f"Failed downloading mapped template {blob_target}: {map_err}")

    # 2. Check if URL points to our Firebase Storage bucket
    if bucket and bucket_name in url:
        try:
            import urllib.parse
            if "/o/" in url:
                blob_name = url.split("/o/")[1].split("?")[0]
                blob_name = urllib.parse.unquote(blob_name)
            elif f"{bucket_name}/" in url:
                blob_name = url.split(f"{bucket_name}/")[1].split("?")[0]
                blob_name = urllib.parse.unquote(blob_name)
            else:
                blob_name = None

            if blob_name:
                blob = bucket.blob(blob_name)
                blob.download_to_filename(tmp_path)
                logger.info(f"Downloaded Firebase Storage blob directly via Admin SDK: {blob_name}")
                return tmp_path
        except Exception as fb_err:
            logger.warning(f"Direct Firebase SDK download notice: {fb_err}. Falling back to HTTP.")

    # 3. Standard HTTP download with browser headers
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "*/*",
    }
    try:
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            with open(tmp_path, "wb") as f:
                f.write(resp.content)
        return tmp_path
    except Exception as http_err:
        logger.warning(f"HTTP download failed for {url}: {http_err}")
        # If it's a video file, fallback to default template rather than crashing
        if suffix == ".mp4" and bucket:
            try:
                blob = bucket.blob(DEFAULT_DANCE_TEMPLATE_BLOB)
                blob.download_to_filename(tmp_path)
                logger.info(f"Fell back to default Firebase dance template: {DEFAULT_DANCE_TEMPLATE_BLOB}")
                return tmp_path
            except Exception as fb_fallback_err:
                logger.error(f"Fallback to default template failed: {fb_fallback_err}")
        raise




def swap_faces_in_video(target_video_url: str, source_image_url: str, user_id: str, job_id: str) -> str:
    """
    Core AI Video Face Swap Pipeline:
    1. Downloads target video and source character photo.
    2. Detects source face embedding.
    3. Iterates over video frames and swaps target faces.
    4. Re-encodes MP4 and uploads to Firebase Storage.
    """
    logger.info(f"Starting Video Face Swap job={job_id} for user={user_id}")
    start_time = time.time()

    # Load models
    app = get_face_analysis()
    swapper = get_inswapper()

    # 1. Download source image & extract source face
    source_img_path = None
    target_video_path = None
    output_video_path = None

    try:
        source_img_path = download_file_to_temp(source_image_url, suffix=".jpg")
        source_pil = Image.open(source_img_path).convert("RGB")
        source_rgb = np.array(source_pil)
        # InsightFace expects BGR
        source_bgr = source_rgb[:, :, ::-1]

        source_faces = app.get(source_bgr)
        if not source_faces or len(source_faces) == 0:
            raise ValueError("No face detected in the provided source image. Please upload a clear photo of a face.")

        # Take largest detected face in source image
        source_face = max(source_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        logger.info(f"Source face detected successfully (bbox: {source_face.bbox})")

        # 2. Download target video
        target_video_path = download_file_to_temp(target_video_url, suffix=".mp4")

        # 3. Read video frames and swap faces
        reader = imageio.get_reader(target_video_path)
        fps = reader.get_meta_data().get("fps", 24)
        if fps <= 0 or fps > 60:
            fps = 24

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_out:
            output_video_path = tmp_out.name

        writer = imageio.get_writer(
            output_video_path,
            fps=fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
            ffmpeg_params=["-preset", "ultrafast", "-crf", "22"]
        )

        frame_count = 0
        swapped_count = 0

        for frame_rgb in reader:
            frame_count += 1
            # Convert RGB to BGR for InsightFace
            frame_bgr = frame_rgb[:, :, ::-1].copy()

            target_faces = app.get(frame_bgr)
            if target_faces and len(target_faces) > 0:
                for target_face in target_faces:
                    frame_bgr = swapper.get(frame_bgr, target_face, source_face, paste_back=True)
                    swapped_count += 1

            # Convert BGR back to RGB for imageio writer
            swapped_rgb = frame_bgr[:, :, ::-1]
            writer.append_data(swapped_rgb)

        reader.close()
        writer.close()

        elapsed = time.time() - start_time
        logger.info(f"Face Swap completed: {frame_count} frames processed ({swapped_count} face swaps) in {elapsed:.1f}s")

        # 4. Read final video bytes and upload to Firebase Storage
        with open(output_video_path, "rb") as vf:
            video_bytes = vf.read()

        import uuid
        import urllib.parse

        token = str(uuid.uuid4())
        bucket = storage.bucket(settings.FIREBASE_STORAGE_BUCKET)
        blob_path = f"outputs/{user_id}/{job_id}/faceswap_video.mp4"
        blob = bucket.blob(blob_path)
        blob.metadata = {"firebaseStorageDownloadTokens": token}
        blob.upload_from_string(video_bytes, content_type="video/mp4")
        logger.info(f"Uploaded swapped video ({len(video_bytes)} bytes) to Firebase Storage: {blob_path}")

        bucket_name = settings.FIREBASE_STORAGE_BUCKET
        encoded_path = urllib.parse.quote(blob_path, safe="")
        download_url = f"https://firebasestorage.googleapis.com/v0/b/{bucket_name}/o/{encoded_path}?alt=media&token={token}"
        return download_url

    finally:
        # Cleanup temporary files
        for p in [source_img_path, target_video_path, output_video_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
