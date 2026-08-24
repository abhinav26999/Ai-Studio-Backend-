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


class LipSyncProcessor:
    """
    100% Free & Local ONNX Wav2Lip Lip-Sync Generator with Enhanced Spatial Locking & Temporal Stability.

    1. Temporal Mouth Smoothing: Per-pixel EMA filter (alpha=0.65) on raw Wav2Lip mouth output
       to eliminate frame-to-frame texture flickering/jitter.
    2. Stable Mask Seams: Precomputes normalized Gaussian blend mask to prevent blend boundary shimmer.
    3. Cross-Faded Mel Windowing: Applies temporal Gaussian cross-fade on acoustic features to eliminate
       abrupt frame-boundary spectral discontinuities.
    4. CFR 25 FPS Normalization: Automatically converts Variable Frame Rate (VFR) inputs to Constant Frame
       Rate (CFR 25.0 fps) to eliminate video stutter.
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
    def detect_anatomical_face_xywh(detector, frame: np.ndarray, target_w: int, target_h: int, conf_thresh: float = 0.65):
        """
        Uses YuNet 5-point landmarks to compute an anatomically aligned face box in (x, y, w, h).
        Returns None if no face detected or confidence < conf_thresh.
        """
        faces = detector.detect(frame)[1]
        if faces is None or len(faces) == 0:
            return None

        # Filter by confidence threshold (score is at face[14])
        valid_faces = [f for f in faces if len(f) > 14 and f[-1] >= conf_thresh]
        if not valid_faces:
            return None

        # Pick largest face by area
        face = max(valid_faces, key=lambda f: f[2] * f[3])
        landmarks = face[4:14].reshape(5, 2).astype(float)
        re, le, nose, rm, lm = landmarks

        eye_center = (re + le) / 2.0
        mouth_center = (rm + lm) / 2.0
        d_em = max(20.0, float(np.linalg.norm(mouth_center - eye_center)))
        face_center_x = (eye_center[0] + mouth_center[0]) / 2.0

        top = max(0.0, eye_center[1] - 0.70 * d_em)
        bottom = min(float(target_h), mouth_center[1] + 0.55 * d_em)
        left = max(0.0, face_center_x - 0.80 * d_em)
        right = min(float(target_w), face_center_x + 0.80 * d_em)

        bx = float(left)
        by = float(top)
        bw = max(10.0, float(right - left))
        bh = max(10.0, float(bottom - top))
        return [bx, by, bw, bh]

    @staticmethod
    def transform_box_affine(box_xywh: list, M: np.ndarray, target_w: int, target_h: int) -> list:
        """Transforms a bounding box's 4 corners with affine matrix M and returns new (x, y, w, h)."""
        bx, by, bw, bh = box_xywh
        corners = np.array([
            [bx, by],
            [bx + bw, by],
            [bx + bw, by + bh],
            [bx, by + bh]
        ], dtype=np.float32).reshape(-1, 1, 2)

        transformed = cv2.transform(corners, M).reshape(-1, 2)
        x1 = max(0.0, float(np.min(transformed[:, 0])))
        y1 = max(0.0, float(np.min(transformed[:, 1])))
        x2 = min(float(target_w), float(np.max(transformed[:, 0])))
        y2 = min(float(target_h), float(np.max(transformed[:, 1])))
        return [x1, y1, max(10.0, x2 - x1), max(10.0, y2 - y1)]

    @staticmethod
    def crop_and_pad_square(frame: np.ndarray, box_xywh: list):
        """
        Pads box to a 1:1 square centered on the face box with safe border reflection.
        Returns (square_crop, square_coords_tuple, pad_info).
        """
        h, w = frame.shape[:2]
        bx, by, bw, bh = box_xywh
        side = int(round(max(bw, bh)))
        cx = bx + bw / 2.0
        cy = by + bh / 2.0

        sq_x1 = int(round(cx - side / 2.0))
        sq_y1 = int(round(cy - side / 2.0))
        sq_x2 = sq_x1 + side
        sq_y2 = sq_y1 + side

        pad_top = max(0, -sq_y1)
        pad_bottom = max(0, sq_y2 - h)
        pad_left = max(0, -sq_x1)
        pad_right = max(0, sq_x2 - w)

        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            padded_frame = cv2.copyMakeBorder(
                frame, pad_top, pad_bottom, pad_left, pad_right,
                borderType=cv2.BORDER_REFLECT
            )
            crop = padded_frame[sq_y1 + pad_top:sq_y2 + pad_top, sq_x1 + pad_left:sq_x2 + pad_left]
        else:
            padded_frame = frame
            crop = frame[sq_y1:sq_y2, sq_x1:sq_x2]

        return crop, (sq_x1, sq_y1, sq_x2, sq_y2), (pad_top, pad_bottom, pad_left, pad_right)

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
        """
        Face-restoration & detail recovery pass after resizing to side x side.
        Performs LAB-space color transfer and high-frequency edge/texture sharpening.
        """
        # 1. LAB-Space Histogram Color Matching
        matched = cls.match_lab_color(patch_bgr, ref_bgr)

        # 2. Unsharp detail sharpening on mouth/teeth texture
        gaussian = cv2.GaussianBlur(matched, (0, 0), 2.0)
        sharpened = cv2.addWeighted(matched, 1.45, gaussian, -0.45, 0)

        # 3. Bilateral smoothing to maintain skin smoothness while preserving crisp lips/teeth
        restored = cv2.bilateralFilter(sharpened, 5, 40, 40)
        return restored

    @classmethod
    def get_stable_feather_mask(cls, side: int) -> np.ndarray:
        """Computes a stable, calibrated 3-channel Gaussian feather blend mask for the lower face."""
        mask = np.zeros((side, side), dtype=np.float32)
        m_start = int(side * 0.50)
        mask[m_start:, :] = 1.0
        feather_x = max(1, int(side * 0.12))
        for i in range(feather_x):
            mask[:, i] *= (i / feather_x)
            mask[:, side - 1 - i] *= (i / feather_x)
        feather_y = max(1, int(side * 0.10))
        for j in range(feather_y):
            mask[side - 1 - j, :] *= (j / feather_y)
        k = max(15, int(side * 0.15) | 1)
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        return np.stack([mask, mask, mask], axis=2)

    @classmethod
    def run_onnx_wav2lip(cls, media_path: str, audio_path: str, output_path: str) -> str:
        """Runs the ONNX Wav2Lip generation pipeline."""
        logger.info(f"Running ONNX Wav2Lip on: {media_path}")

        # Load models
        wav2lip_model = hf_hub_download(repo_id="bluefoxcreation/Wav2lip-Onnx", filename="wav2lip_gan.onnx")
        yunet_model = hf_hub_download(repo_id="opencv/face_detection_yunet", filename="face_detection_yunet_2023mar.onnx")
        session = ort.InferenceSession(wav2lip_model, providers=["CPUExecutionProvider"])

        # Fix 5: Audio mel chunk extraction matching exact audio duration * FPS
        mel, audio_duration = cls.audio_to_mel(audio_path)
        mel_idx_multiplier = 80.0 / FPS
        total_frames = int(round(audio_duration * FPS))
        mel_chunks = []
        for i in range(total_frames):
            s = int(i * mel_idx_multiplier)
            e = s + MEL_STEP_SIZE
            if e <= mel.shape[1]:
                chunk = mel[:, s:e]
            else:
                pad_width = e - mel.shape[1]
                chunk = np.pad(mel[:, max(0, s):], ((0, 0), (0, pad_width)), mode="edge")
            mel_chunks.append(chunk)
        logger.info(f"Audio duration: {audio_duration:.2f}s -> Mel chunks: {len(mel_chunks)} frames")

        # Determine media type (Video vs Static Photo)
        is_video = False
        loaded_frames = []

        # Check if media is video and normalize to constant 25 FPS (CFR)
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

        # Clean up temporary CFR video if created
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
        detector = cv2.FaceDetectorYN.create(yunet_model, "", (target_w, target_h), score_threshold=0.65)

        # Canonical face detection on base frame
        canonical_box = cls.detect_anatomical_face_xywh(detector, base_frame, target_w, target_h, conf_thresh=0.50)
        if canonical_box is None:
            canonical_box = [target_w * 0.20, target_h * 0.20, target_w * 0.60, target_h * 0.45]

        # Fix 2: EMA smoothed (x, y, w, h) tracking across all 4 parameters
        smoothed_xywh = list(canonical_box)
        temp_dir = os.path.join(os.path.dirname(output_path), f"lf_{uuid.uuid4().hex[:6]}")
        os.makedirs(temp_dir, exist_ok=True)
        frame_files = []

        # Fix 1: Temporal smoothing state for raw generated mouth patch
        prev_pred96_f = None

        for idx, mel_chunk in enumerate(mel_chunks):
            t = idx / float(FPS)

            if is_video:
                curr_frame = loaded_frames[idx % len(loaded_frames)].copy()
                # Fix 3: Gated per-frame detection with confidence threshold (conf >= 0.65)
                det_xywh = cls.detect_anatomical_face_xywh(detector, curr_frame, target_w, target_h, conf_thresh=0.65)
                if det_xywh is not None:
                    # Fix 2: Apply EMA on all 4 parameters (x, y, w, h)
                    for c in range(4):
                        smoothed_xywh[c] = 0.70 * smoothed_xywh[c] + 0.30 * det_xywh[c]
                box_to_use = smoothed_xywh
            else:
                # Fix 1: Warp image with Living Motion transform M(t)
                s_zoom = 1.0 + 0.008 * np.sin(2 * np.pi * t / 3.5)
                dx = 3.5 * np.sin(2 * np.pi * t / 4.0) + 1.5 * np.cos(2 * np.pi * t / 2.0)
                dy = 2.0 * np.cos(2 * np.pi * t / 3.0)
                angle = 0.35 * np.sin(2 * np.pi * t / 3.8)

                center = (target_w / 2.0, target_h / 2.0)
                M = cv2.getRotationMatrix2D(center, angle, s_zoom)
                M[0, 2] += dx
                M[1, 2] += dy
                curr_frame = cv2.warpAffine(base_frame, M, (target_w, target_h), borderMode=cv2.BORDER_REFLECT)

                # Fix 1: Transform canonical box corners with exact same matrix M(t)
                box_to_use = cls.transform_box_affine(canonical_box, M, target_w, target_h)

            # Fix 4: Pad box to 1:1 square crop to eliminate Wav2Lip aspect ratio distortion
            sq_crop, (sq_x1, sq_y1, sq_x2, sq_y2), (pt, pb, pl, pr) = cls.crop_and_pad_square(curr_frame, box_to_use)
            side = sq_crop.shape[0]

            if side < 10:
                fpath = os.path.join(temp_dir, f"frame_{idx:05d}.png")
                cv2.imwrite(fpath, curr_frame)
                frame_files.append(fpath)
                continue

            # Wav2Lip input prep (96x96 square)
            face_96 = cv2.resize(sq_crop, (WAV2LIP_SIZE, WAV2LIP_SIZE))
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

            # Fix 1: Temporal EMA smoothing on raw generated mouth patch (per-pixel float32)
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

            # Fix 4: Inverse resize back to (side, side)
            pred_sq = cv2.resize(smoothed_pred96_uint8, (side, side), interpolation=cv2.INTER_LANCZOS4)

            # High-Resolution Face & Detail Restoration Pass (LAB color transfer + high-frequency sharpening)
            restored_sq = cls.restore_and_sharpen_patch(pred_sq, sq_crop)

            # Fix 2: Stable, pre-calculated Gaussian feather blend mask (lower 50%)
            mask3 = cls.get_stable_feather_mask(side)

            blended_sq = restored_sq.astype(np.float32) * mask3 + sq_crop.astype(np.float32) * (1.0 - mask3)
            blended_sq_uint8 = np.clip(blended_sq, 0, 255).astype(np.uint8)

            # Fix 4: Exact inverse pasting back into frame (accounting for safe border padding)
            if pt > 0 or pb > 0 or pl > 0 or pr > 0:
                unpadded_h = sq_crop.shape[0] - pt - pb
                unpadded_w = sq_crop.shape[1] - pl - pr
                valid_patch = blended_sq_uint8[pt:pt + unpadded_h, pl:pl + unpadded_w]
                curr_frame[max(0, sq_y1):min(target_h, sq_y2), max(0, sq_x1):min(target_w, sq_x2)] = valid_patch
            else:
                curr_frame[sq_y1:sq_y2, sq_x1:sq_x2] = blended_sq_uint8

            fpath = os.path.join(temp_dir, f"frame_{idx:05d}.png")
            cv2.imwrite(fpath, curr_frame)
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
