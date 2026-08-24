"""
SadTalker Integration Module
Generates full talking avatar with:
- Facial expressions driven by audio
- Natural head pose movement
- Eye blinks
- Lip sync (via Wav2Lip inside SadTalker)
"""
import os
import sys
import logging
import subprocess
import tempfile

from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

# SadTalker source code path (bundled with project)
SADTALKER_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "sadtalker_src")

# All SadTalker checkpoint files from vinthony/SadTalker on HuggingFace
SADTALKER_CHECKPOINT_FILES = [
    "auido2exp_00300-model.pth",
    "auido2pose_00140-model.pth",
    "mapping_00109-model.pth.tar",
    "mapping_00229-model.pth.tar",
    "epoch_20.pth",
    "wav2lip.pth",
    "shape_predictor_68_face_landmarks.dat",
    "hub/checkpoints/s3fd-619a316812.pth",
]

# BFM Fitting files
SADTALKER_BFM_FILES = [
    "BFM_Fitting/01_MorphableModel.mat",
    "BFM_Fitting/BFM09_model_info.mat",
    "BFM_Fitting/BFM_exp_idx.mat",
    "BFM_Fitting/BFM_front_idx.mat",
    "BFM_Fitting/Exp_Pca.bin",
    "BFM_Fitting/facemodel_info.mat",
    "BFM_Fitting/select_vertex_id.mat",
    "BFM_Fitting/similarity_Lm3D_all.mat",
    "BFM_Fitting/std_exp.txt",
]

def ensure_sadtalker_checkpoints() -> str:
    """Downloads SadTalker checkpoints and returns the checkpoint directory."""
    cached_snapshot = "/Users/aeologic/.cache/huggingface/hub/models--vinthony--SadTalker/snapshots/4aedd064359e623398a2d73eb8c253ebb2bd516c"
    if os.path.exists(os.path.join(cached_snapshot, "auido2exp_00300-model.pth")):
        return cached_snapshot

    import huggingface_hub
    checkpoint_dir = None
    for filename in SADTALKER_CHECKPOINT_FILES + SADTALKER_BFM_FILES:
        try:
            local_path = hf_hub_download(repo_id="vinthony/SadTalker", filename=filename)
            if checkpoint_dir is None and filename == "auido2exp_00300-model.pth":
                checkpoint_dir = os.path.dirname(local_path)
                logger.info(f"SadTalker checkpoint dir: {checkpoint_dir}")
        except Exception as e:
            logger.warning(f"Could not download {filename}: {e}")

    return checkpoint_dir or cached_snapshot

def run_sadtalker(
    image_path: str,
    audio_path: str,
    output_path: str,
    pose_style: int = 0,
    size: int = 256,
    preprocess: str = "crop",
) -> str:
    """
    Runs SadTalker to generate a full talking avatar video with:
    - Audio-driven facial expressions
    - Natural head pose movement
    - Eye blinks
    - Lip sync

    Args:
        image_path: Path to source portrait image
        audio_path: Path to audio file (mp3/wav)
        output_path: Path for output video (mp4)
        pose_style: Head pose style (0-45)
        size: Output size (256 or 512)
        preprocess: 'crop' or 'full'

    Returns:
        Path to generated video
    """
    logger.info(f"Starting SadTalker generation for: {image_path}")

    # Ensure checkpoints are downloaded
    checkpoint_dir = ensure_sadtalker_checkpoints()
    if checkpoint_dir is None:
        raise RuntimeError("Failed to download SadTalker checkpoints")

    # BFM config dir is inside SadTalker src
    config_dir = os.path.join(SADTALKER_SRC, "src", "config")

    with tempfile.TemporaryDirectory() as result_dir:
        # Build the inference command using SadTalker's inference.py
        cmd = [
            sys.executable,
            os.path.join(SADTALKER_SRC, "inference.py"),
            "--driven_audio", audio_path,
            "--source_image", image_path,
            "--checkpoint_dir", checkpoint_dir,
            "--result_dir", result_dir,
            "--pose_style", str(pose_style),
            "--size", str(size),
            "--preprocess", preprocess,
            "--old_version",  # Use .pth checkpoints from vinthony/SadTalker
            "--batch_size", "1",
            "--cpu",
        ]

        logger.info(f"Running SadTalker: {' '.join(cmd)}")

        env = os.environ.copy()
        env["PYTHONPATH"] = SADTALKER_SRC + ":" + env.get("PYTHONPATH", "")

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=SADTALKER_SRC,
        )

        stdout = result.stdout.decode("utf-8", errors="ignore")
        stderr = result.stderr.decode("utf-8", errors="ignore")

        logger.info(f"SadTalker stdout: {stdout[-500:]}")
        if result.returncode != 0:
            logger.error(f"SadTalker stderr: {stderr[-1000:]}")
            raise RuntimeError(f"SadTalker failed (code {result.returncode}): {stderr[-300:]}")

        # Find generated video in result_dir
        import glob
        video_files = glob.glob(os.path.join(result_dir, "**", "*.mp4"), recursive=True)
        if not video_files:
            raise RuntimeError(f"SadTalker finished but no video found in {result_dir}")

        # Copy to desired output path
        import shutil
        shutil.copy2(video_files[0], output_path)

        logger.info(f"✅ SadTalker video saved to: {output_path} ({os.path.getsize(output_path):,} bytes)")
        return output_path
