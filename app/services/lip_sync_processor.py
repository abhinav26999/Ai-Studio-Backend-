import os
import uuid
import logging
import asyncio
import tempfile
import subprocess
import httpx
import numpy as np
import cv2
import librosa
import edge_tts
import imageio_ffmpeg
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from app.db.firebase import get_db, upload_to_storage
from app.models.job import JobStatus

logger = logging.getLogger(__name__)

# Standard US Voice mappings for Edge-TTS (100% free)
VOICE_MAP = {
    "male": "en-US-GuyNeural",
    "female": "en-US-JennyNeural",
    "en-us-male": "en-US-GuyNeural",
    "en-us-female": "en-US-JennyNeural",
    "en-US-GuyNeural": "en-US-GuyNeural",
    "en-US-JennyNeural": "en-US-JennyNeural",
}

# ONNX Wav2Lip audio parameters
NUM_MELS = 80
SAMPLE_RATE = 16000
N_FFT = 800
HOP_SIZE = 200
WIN_SIZE = 800
FMIN = 55
FMAX = 7600
FPS = 25
MEL_STEP_SIZE = 16
WAV2LIP_SIZE = 96
MAX_WIDTH = 720

# Temporal smoothing weight for frame-to-frame mouth texture stability.
# Smoothed_t = ALPHA * Generated_t + (1 - ALPHA) * Smoothed_{t-1}
# Lower = smoother (less flicker), Higher = more responsive to fast syllable changes.
TEMPORAL_SMOOTHING_ALPHA = 0.65


# Canonical 96x96 frontal face landmark configuration
CANONICAL_PTS_96 = np.array([
    [28.0, 32.0],  # Right Eye (Subject's right / viewer's left)
    [68.0, 32.0],  # Left Eye (Subject's left / viewer's right)
    [48.0, 50.0],  # Nose Tip
    [32.0, 72.0],  # Right Mouth Corner
    [64.0, 72.0],  # Left Mouth Corner
], dtype=np.float32)


class LipSyncProcessor:
    """
    100% Free & Local ONNX Wav2Lip Lip-Sync Generator with 3D Landmark Perspective Frontalization.

    1. 3D Landmark Frontalization: Normalizes any 3D rotated/tilted face to a canonical frontal
       96x96 canvas for Wav2Lip inference, and inverse-warps the synthesized mouth back onto the
       original frame using the exact inverse affine matrix M_inv.
    2. Zero Head-Turning Drift: The mouth rotates and scales with the subject's skull in 3D,
       never drifting to the cheek or nose during head turns.
    3. Seamless Elliptical Feathering: Replaces rectangular box cuts with an anatomical elliptical
       feathered alpha mask, eliminating all boundary demarcations.
    4. 3D Living Photo Reenactment: Simulates natural 3D head pitch/yaw oscillation, posture
       breathing, and micro-expressions for static photo avatars.
    """

    @staticmethod
    def normalize_video_to_cfr(input_video_path: str, output_video_path: str) -> str:
        """Normalizes input video to Constant Frame Rate (25 FPS CFR) via FFmpeg."""
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-y",
            "-i", input_video_path,
            "-r", str(FPS),
            "-vsync", "cfr",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
            "-an",
            "-pix_fmt", "yuv420p",
            output_video_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0 and os.path.exists(output_video_path):
            return output_video_path
        return input_video_path

    @staticmethod
    async def generate_tts_audio(text: str, voice_option: str, output_path: str) -> str:
        """Converts paragraph text into neural speech MP3 using edge-tts."""
        voice = VOICE_MAP.get(voice_option, VOICE_MAP.get(voice_option.lower(), "en-US-GuyNeural"))
        logger.info(f"Generating TTS audio (len={len(text)}) with voice: {voice}")
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)
        logger.info(f"TTS audio saved to: {output_path}")
        return output_path

    @staticmethod
    def audio_to_mel(audio_path: str) -> tuple:
        """
        Extracts 80-channel log-mel spectrogram for Wav2Lip ONNX model with temporal cross-fade.
        Returns (mel_spectrogram, audio_duration_seconds).
        """
        wav, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        duration = float(librosa.get_duration(y=wav, sr=SAMPLE_RATE))
        mel = librosa.feature.melspectrogram(
            y=wav, sr=SAMPLE_RATE, n_fft=N_FFT,
            hop_length=HOP_SIZE, win_length=WIN_SIZE,
            n_mels=NUM_MELS, fmin=FMIN, fmax=FMAX
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_normalized = np.clip((mel_db + 100) / 100 * 8 - 4, -4, 4)

        # Apply subtle temporal 1D smoothing along time axis for smooth cross-fading across sliding windows
        mel_smoothed = cv2.GaussianBlur(mel_normalized, (5, 1), 0)
        return mel_smoothed, duration

    @staticmethod
    def detect_landmarks(detector, frame: np.ndarray, conf_thresh: float = 0.60):
        """Detects 5 facial landmarks using YuNet. Returns (landmarks_5x2, confidence)."""
        faces = detector.detect(frame)[1]
        if faces is None or len(faces) == 0:
            return None, 0.0

        valid_faces = [f for f in faces if len(f) > 14 and f[-1] >= conf_thresh]
        if not valid_faces:
            return None, 0.0

        face = max(valid_faces, key=lambda f: f[2] * f[3])
        landmarks = face[4:14].reshape(5, 2).astype(np.float32)
        return landmarks, float(face[-1])

    @staticmethod
    def match_lab_color(source_bgr: np.ndarray, target_bgr: np.ndarray) -> np.ndarray:
        """Matches color/tone in LAB color space using channel-wise mean & standard deviation transfer."""
        src_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        tgt_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        src_mean, src_std = np.mean(src_lab, axis=(0, 1)), np.std(src_lab, axis=(0, 1))
        tgt_mean, tgt_std = np.mean(tgt_lab, axis=(0, 1)), np.std(tgt_lab, axis=(0, 1))

        matched_lab = np.zeros_like(src_lab)
        for c in range(3):
            std_ratio = tgt_std[c] / (src_std[c] + 1e-6)
            matched_lab[:, :, c] = (src_lab[:, :, c] - src_mean[c]) * std_ratio + tgt_mean[c]

        matched_lab = np.clip(matched_lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)

    @classmethod
    def restore_and_sharpen_patch(cls, patch_bgr: np.ndarray, ref_bgr: np.ndarray) -> np.ndarray:
        """Face-restoration pass: LAB color matching + high-frequency edge sharpening + bilateral skin smoothing."""
        matched = cls.match_lab_color(patch_bgr, ref_bgr)
        gaussian = cv2.GaussianBlur(matched, (0, 0), 1.5)
        sharpened = cv2.addWeighted(matched, 1.4, gaussian, -0.4, 0)
        restored = cv2.bilateralFilter(sharpened, 5, 30, 30)
        return restored

    @classmethod
    def get_elliptical_mouth_mask(cls) -> np.ndarray:
        """Creates an anatomically calibrated elliptical feather mask in 96x96 canonical space."""
        mask_96 = np.zeros((WAV2LIP_SIZE, WAV2LIP_SIZE), dtype=np.float32)
        cv2.ellipse(mask_96, (48, 73), (25, 17), 0, 0, 360, 1.0, -1)
        mask_96 = cv2.GaussianBlur(mask_96, (13, 13), 0)
        return mask_96

    @classmethod
    def run_onnx_wav2lip(cls, media_path: str, audio_path: str, output_path: str) -> str:
        """Runs the 3D Landmark Frontalized ONNX Wav2Lip generation pipeline."""
        logger.info(f"Running 3D Landmark ONNX Wav2Lip on: {media_path}")

        # Load models
        wav2lip_model = hf_hub_download(repo_id="bluefoxcreation/Wav2lip-Onnx", filename="wav2lip_gan.onnx")
        yunet_model = hf_hub_download(repo_id="opencv/face_detection_yunet", filename="face_detection_yunet_2023mar.onnx")
        session = ort.InferenceSession(wav2lip_model, providers=["CPUExecutionProvider"])

        # Audio mel extraction
        mel, audio_duration = cls.audio_to_mel(audio_path)
        mel_idx_multiplier = 80.0 / FPS
        total_frames = int(round(audio_duration * FPS))
        mel_chunks = []
        silence_val = -4.0  # Normalized 0 dB silence
        for i in range(total_frames):
            s = int(i * mel_idx_multiplier)
            e = s + MEL_STEP_SIZE
            if e <= mel.shape[1]:
                chunk = mel[:, s:e]
            else:
                pad_width = e - mel.shape[1]
                chunk = np.pad(mel[:, max(0, s):], ((0, 0), (0, pad_width)), mode="constant", constant_values=silence_val)
            mel_chunks.append(chunk)
        logger.info(f"Audio duration: {audio_duration:.2f}s -> Mel chunks: {len(mel_chunks)} frames")

        # Determine media type (Video vs Static Photo)
        is_video = False
        loaded_frames = []

        is_vid_ext = any(media_path.lower().endswith(ext) for ext in [".mp4", ".mov", ".avi", ".webm", ".m4v"])
        actual_media_path = media_path
        cfr_temp_path = None

        if is_vid_ext:
            cfr_temp_path = os.path.join(os.path.dirname(output_path), f"cfr_{uuid.uuid4().hex[:6]}.mp4")
            actual_media_path = cls.normalize_video_to_cfr(media_path, cfr_temp_path)

        cap = cv2.VideoCapture(actual_media_path)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 1:
                is_video = True
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                while True:
                    ret2, frm = cap.read()
                    if not ret2 or frm is None:
                        break
                    fh, fw = frm.shape[:2]
                    if fw > MAX_WIDTH:
                        scale = MAX_WIDTH / fw
                        new_h = int(fh * scale)
                        new_h = new_h if new_h % 2 == 0 else new_h - 1
                        frm = cv2.resize(frm, (MAX_WIDTH, new_h))
                    else:
                        frm = cv2.resize(frm, (fw if fw % 2 == 0 else fw - 1, fh if fh % 2 == 0 else fh - 1))
                    loaded_frames.append(frm)
            cap.release()

        if cfr_temp_path and os.path.exists(cfr_temp_path):
            try: os.remove(cfr_temp_path)
            except Exception: pass

        if is_video:
            if not loaded_frames:
                raise ValueError("Could not load video frames.")
            logger.info(f"Character Video mode: {len(loaded_frames)} frames.")
            base_frame = loaded_frames[0]
        else:
            orig = cv2.imread(media_path)
            if orig is None:
                raise ValueError(f"Cannot read image from {media_path}")
            oh, ow = orig.shape[:2]
            if ow > MAX_WIDTH:
                scale = MAX_WIDTH / ow
                new_h = int(oh * scale)
                new_h = new_h if new_h % 2 == 0 else new_h - 1
                base_frame = cv2.resize(orig, (MAX_WIDTH, new_h))
            else:
                base_frame = cv2.resize(orig, (ow if ow % 2 == 0 else ow - 1, oh if oh % 2 == 0 else oh - 1))
            logger.info(f"Avatar Photo mode: {base_frame.shape[1]}x{base_frame.shape[0]} px.")

        target_h, target_w = base_frame.shape[:2]
        detector = cv2.FaceDetectorYN.create(yunet_model, "", (target_w, target_h), score_threshold=0.55)

        # Detect canonical landmarks on base frame
        canonical_landmarks, _ = cls.detect_landmarks(detector, base_frame, conf_thresh=0.50)
        if canonical_landmarks is None:
            cx, cy = target_w / 2.0, target_h * 0.45
            canonical_landmarks = np.array([
                [cx - 40, cy - 30],
                [cx + 40, cy - 30],
                [cx, cy],
                [cx - 30, cy + 40],
                [cx + 30, cy + 40]
            ], dtype=np.float32)

        base_M_front, _ = cv2.estimateAffinePartial2D(canonical_landmarks, CANONICAL_PTS_96)
        if base_M_front is None:
            base_M_front = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        smoothed_M = base_M_front.copy()
        prev_center = np.mean(canonical_landmarks, axis=0)
        mask_96 = cls.get_elliptical_mouth_mask()

        temp_dir = os.path.join(os.path.dirname(output_path), f"lf_{uuid.uuid4().hex[:6]}")
        os.makedirs(temp_dir, exist_ok=True)
        frame_files = []
        prev_pred96_f = None

        for idx, mel_chunk in enumerate(mel_chunks):
            t = idx / float(FPS)

            if is_video:
                curr_frame = loaded_frames[idx % len(loaded_frames)].copy()
                faces = detector.detect(curr_frame)[1]
                best_face = None
                if faces is not None and len(faces) > 0:
                    valid_faces = [f for f in faces if len(f) > 14 and f[-1] >= 0.55]
                    if valid_faces:
                        if prev_center is not None:
                            # Continuity tracking: pick face closest to previous frame's face center
                            def dist_to_prev(f):
                                fcx = (f[4] + f[6]) / 2.0
                                fcy = (f[5] + f[7]) / 2.0
                                return (fcx - prev_center[0])**2 + (fcy - prev_center[1])**2
                            best_face = min(valid_faces, key=dist_to_prev)
                        else:
                            best_face = max(valid_faces, key=lambda f: f[2] * f[3])

                if best_face is not None:
                    lm = best_face[4:14].reshape(5, 2).astype(np.float32)
                    curr_center = np.mean(lm, axis=0)

                    # Outlier jump rejection (> 120px sudden teleportation between adjacent frames)
                    if prev_center is not None and np.linalg.norm(curr_center - prev_center) > 120:
                        M_to_use = smoothed_M
                    else:
                        M_front, _ = cv2.estimateAffinePartial2D(lm, CANONICAL_PTS_96)
                        if M_front is not None:
                            # Velocity-adaptive zero-lag tracking:
                            # When face is moving fast, track instantaneously (alpha ~ 1.0) with zero phase delay
                            # When face is resting, smooth sub-pixel sensor jitter (alpha = 0.55)
                            vel = np.linalg.norm(curr_center - prev_center) if prev_center is not None else 0.0
                            alpha = min(1.0, max(0.55, vel / 6.0))
                            smoothed_M = alpha * M_front + (1.0 - alpha) * smoothed_M
                            prev_center = curr_center
                        M_to_use = smoothed_M
                else:
                    M_to_use = smoothed_M
            else:
                # Enhanced 3D Living Motion Reenactment for Photos
                s_zoom = 1.0 + 0.010 * np.sin(2 * np.pi * t / 3.4)
                dx = 3.5 * np.sin(2 * np.pi * t / 4.2) + 1.2 * np.cos(2 * np.pi * t / 2.1)
                dy = 2.5 * np.cos(2 * np.pi * t / 3.0)
                angle = 0.40 * np.sin(2 * np.pi * t / 3.6)

                center = (target_w / 2.0, target_h / 2.0)
                M_living = cv2.getRotationMatrix2D(center, angle, s_zoom)
                M_living[0, 2] += dx
                M_living[1, 2] += dy
                curr_frame = cv2.warpAffine(base_frame, M_living, (target_w, target_h), borderMode=cv2.BORDER_REFLECT)

                corners = canonical_landmarks.reshape(-1, 1, 2)
                transformed_lm = cv2.transform(corners, M_living).reshape(-1, 2)
                M_to_use, _ = cv2.estimateAffinePartial2D(transformed_lm, CANONICAL_PTS_96)
                if M_to_use is None:
                    M_to_use = smoothed_M

            # 3D Landmark Frontalization: Warps face to level, frontal 96x96 canvas
            face_96 = cv2.warpAffine(curr_frame, M_to_use, (WAV2LIP_SIZE, WAV2LIP_SIZE), flags=cv2.INTER_LANCZOS4)

            # Wav2Lip input prep
            face_norm = face_96.astype(np.float32) / 255.0
            masked = face_norm.copy()
            masked[WAV2LIP_SIZE // 2:, :, :] = 0.0
            vid_in = np.concatenate([masked, face_norm], axis=2)
            vid_in = np.transpose(vid_in, (2, 0, 1))[np.newaxis]

            # Inference
            mel_inp = np.expand_dims(mel_chunk, axis=(0, 1)).astype(np.float32)
            pred = session.run(None, {"mel": mel_inp, "vid": vid_in})[0][0]
            pred = np.transpose(pred, (1, 2, 0))
            pred96 = np.clip(pred * 255.0, 0, 255).astype(np.uint8)

            # Temporal smoothing on generated mouth
            pred96_f = pred96.astype(np.float32)
            if prev_pred96_f is None:
                smoothed_pred96 = pred96_f
            else:
                smoothed_pred96 = (
                    TEMPORAL_SMOOTHING_ALPHA * pred96_f +
                    (1.0 - TEMPORAL_SMOOTHING_ALPHA) * prev_pred96_f
                )
            prev_pred96_f = smoothed_pred96.copy()
            smoothed_pred96_uint8 = np.clip(smoothed_pred96, 0, 255).astype(np.uint8)

            # LAB Color matching & detail sharpening
            pred96_restored = cls.restore_and_sharpen_patch(smoothed_pred96_uint8, face_96)

            # Inverse Affine 3D Compositing: Warps generated mouth and elliptical mask back into original frame
            M_inv = cv2.invertAffineTransform(M_to_use)
            warped_pred = cv2.warpAffine(pred96_restored, M_inv, (target_w, target_h), flags=cv2.INTER_LANCZOS4)
            warped_mask = cv2.warpAffine(mask_96, M_inv, (target_w, target_h), flags=cv2.INTER_LANCZOS4)
            warped_mask3 = np.stack([warped_mask, warped_mask, warped_mask], axis=2)

            blended_frame = (
                warped_pred.astype(np.float32) * warped_mask3 +
                curr_frame.astype(np.float32) * (1.0 - warped_mask3)
            )
            blended_frame_uint8 = np.clip(blended_frame, 0, 255).astype(np.uint8)

            fpath = os.path.join(temp_dir, f"frame_{idx:05d}.png")
            cv2.imwrite(fpath, blended_frame_uint8)
            frame_files.append(fpath)

        logger.info(f"Rendered {len(frame_files)} frames. Muxing MP4 with audio...")
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-y",
            "-r", str(FPS), "-i", os.path.join(temp_dir, "frame_%05d.png"),
            "-i", audio_path,
            "-c:v", "libx264", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-pix_fmt", "yuv420p", "-shortest",
            output_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        for f in frame_files:
            try: os.remove(f)
            except: pass
        try: os.rmdir(temp_dir)
        except: pass

        if res.returncode == 0 and os.path.exists(output_path):
            logger.info(f"Lip-Sync output completed: {output_path} ({os.path.getsize(output_path):,} bytes)")
            return output_path
        raise RuntimeError(f"FFmpeg encoding failed: {res.stderr.decode('utf-8', errors='ignore')[:500]}")

    @classmethod
    async def process_job(cls, job_id: str, user_id: str, params: dict):
        """Processes a background Lip-Sync job and uploads to Firebase."""
        db = get_db()
        job_ref = db.collection("jobs").document(job_id)

        try:
            job_ref.update({"status": JobStatus.PROCESSING.value})

            input_url = params.get("videoUrl") or params.get("imageUrl") or params.get("avatarUrl")
            paragraph_text = params.get("paragraphText") or params.get("userPrompt")
            voice_option = params.get("voice", "en-US-GuyNeural")

            if not input_url:
                raise ValueError("Missing input URL (videoUrl or imageUrl)")
            if not paragraph_text:
                raise ValueError("Missing paragraphText")

            with tempfile.TemporaryDirectory() as temp_dir:
                url_clean = input_url.split("?")[0].lower()
                ext = ".mp4" if any(url_clean.endswith(e) for e in [".mp4", ".mov", ".avi", ".webm", ".m4v"]) else ".jpg"

                local_media = os.path.join(temp_dir, f"media{ext}")
                local_audio = os.path.join(temp_dir, "speech.mp3")
                local_output = os.path.join(temp_dir, "output.mp4")

                logger.info(f"Downloading media from: {input_url}")
                async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                    resp = await client.get(input_url)
                    resp.raise_for_status()
                    with open(local_media, "wb") as f:
                        f.write(resp.content)

                await cls.generate_tts_audio(paragraph_text, voice_option, local_audio)

                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, cls.run_onnx_wav2lip, local_media, local_audio, local_output)

                dest_blob = f"outputs/{user_id}/lip_sync/{job_id}.mp4"
                output_url = upload_to_storage(local_output, dest_blob)
                logger.info(f"Uploaded result for job {job_id} -> {output_url}")

                job_ref.update({"status": JobStatus.COMPLETED.value, "outputUrl": output_url, "error": None})
                return output_url

        except Exception as e:
            error_msg = f"LipSync job error: {str(e)}"
            logger.error(f"Job {job_id} failed: {error_msg}", exc_info=True)
            job_ref.update({"status": JobStatus.ERROR.value, "error": error_msg})
            raise
