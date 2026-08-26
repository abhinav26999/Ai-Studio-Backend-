import os
import uuid
import logging
import asyncio
import tempfile
import subprocess
import httpx
import cv2
import numpy as np
import imageio_ffmpeg
from datetime import datetime, timezone
from typing import Dict, Any, List
from app.db.firebase import get_db, upload_to_storage
from app.models.job import JobStatus
from app.services.lip_sync_processor import LipSyncProcessor

def get_utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

logger = logging.getLogger(__name__)


class InterviewService:
    """
    Dedicated Service for Multi-Question Interactive Interview Lip-Sync (Option A).
    Orchestrates sequential question video generation, idle loop extraction,
    and structured playback manifest compilation.
    """

    @staticmethod
    def generate_idle_loop(media_path: str, output_path: str, duration_sec: float = 3.0, fps: int = 25) -> str:
        """
        Creates a seamless, silent looping video of the avatar breathing/resting
        for the client app to play while waiting for the candidate to answer.
        """
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        is_video = any(media_path.lower().endswith(ext) for ext in [".mp4", ".mov", ".avi", ".webm", ".m4v"])

        if is_video:
            # Load first resting frame from the video
            cap = cv2.VideoCapture(media_path)
            ret, img = cap.read()
            cap.release()
            if not ret or img is None:
                raise ValueError(f"Cannot read frame from video: {media_path}")
        else:
            img = cv2.imread(media_path)
            if img is None:
                raise ValueError(f"Cannot read avatar image for idle loop: {media_path}")

        h, w = img.shape[:2]
        total_frames = int(duration_sec * fps)
        temp_dir = os.path.join(os.path.dirname(output_path), f"idle_{uuid.uuid4().hex[:6]}")
        os.makedirs(temp_dir, exist_ok=True)

        for i in range(total_frames):
            t = i / float(fps)
            # Gentle sinusoidal breathing motion
            s_zoom = 1.0 + 0.008 * np.sin(2 * np.pi * t / duration_sec)
            dx = 1.5 * np.sin(2 * np.pi * t / duration_sec)
            dy = 1.2 * np.cos(2 * np.pi * t / duration_sec)
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 0.2 * np.sin(2 * np.pi * t / duration_sec), s_zoom)
            M[0, 2] += dx
            M[1, 2] += dy
            frame = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            cv2.imwrite(os.path.join(temp_dir, f"idle_{i:04d}.png"), frame)

        cmd = [
            ffmpeg_exe, "-y",
            "-r", str(fps),
            "-i", os.path.join(temp_dir, "idle_%04d.png"),
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            "-an",
            output_path
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        for f in os.listdir(temp_dir):
            try: os.remove(os.path.join(temp_dir, f))
            except Exception: pass
        try: os.rmdir(temp_dir)
        except Exception: pass

        return output_path

    @classmethod
    async def process_interview_job(cls, interview_id: str, user_id: str, request_data: Dict[str, Any]):
        """
        Background orchestrator for rendering all question clips in an interview.
        Updates Firestore with real-time per-question progress.
        """
        db = get_db()
        interview_ref = db.collection("interviews").document(interview_id)

        try:
            interview_ref.update({
                "status": JobStatus.PROCESSING.value,
                "updatedAt": get_utc_now_iso()
            })

            avatar_url = request_data["avatarMediaUrl"]
            questions = request_data["questions"]
            generate_idle = request_data.get("generateIdleLoop", True)

            with tempfile.TemporaryDirectory() as temp_dir:
                # 1. Download Base Avatar Media
                url_clean = avatar_url.split("?")[0].lower()
                ext = ".mp4" if any(url_clean.endswith(e) for e in [".mp4", ".mov", ".avi", ".webm", ".m4v"]) else ".jpg"
                local_avatar = os.path.join(temp_dir, f"base_avatar{ext}")

                logger.info(f"[{interview_id}] Downloading base avatar from {avatar_url}")
                async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                    resp = await client.get(avatar_url)
                    resp.raise_for_status()
                    with open(local_avatar, "wb") as f:
                        f.write(resp.content)

                # 2. Generate and upload Idle Loop if requested
                idle_loop_url = None
                if generate_idle:
                    local_idle = os.path.join(temp_dir, "idle_loop.mp4")
                    cls.generate_idle_loop(local_avatar, local_idle)
                    idle_blob = f"outputs/{user_id}/interviews/{interview_id}/idle_loop.mp4"
                    idle_loop_url = upload_to_storage(local_idle, idle_blob)
                    interview_ref.update({
                        "idleLoopVideoUrl": idle_loop_url,
                        "updatedAt": get_utc_now_iso()
                    })
                    logger.info(f"[{interview_id}] Uploaded Idle Loop: {idle_loop_url}")

                # 3. Process Questions Sequentially
                segments_metadata: List[Dict[str, Any]] = []
                total_duration = 0.0
                loop = asyncio.get_event_loop()

                for idx, q in enumerate(questions):
                    q_id = q.get("id", f"Q{idx+1}")
                    q_tag = q.get("tag")
                    q_text = q.get("text", "")
                    q_voice = q.get("voice", "en-US-GuyNeural")

                    logger.info(f"[{interview_id}] Processing Question {idx+1}/{len(questions)}: {q_id} ({q_tag})")

                    local_speech = os.path.join(temp_dir, f"speech_{idx:03d}.mp3")
                    local_q_output = os.path.join(temp_dir, f"video_{idx:03d}.mp4")

                    # Step A: Synthesize TTS
                    await LipSyncProcessor.generate_tts_audio(q_text, q_voice, local_speech)

                    # Step B: Render Lip-Sync Video Segment
                    await loop.run_in_executor(
                        None,
                        LipSyncProcessor.run_onnx_wav2lip,
                        local_avatar,
                        local_speech,
                        local_q_output
                    )

                    # Step C: Upload Question Video to Storage
                    q_blob = f"outputs/{user_id}/interviews/{interview_id}/segments/q_{idx+1:02d}_{q_id}.mp4"
                    video_url = upload_to_storage(local_q_output, q_blob)

                    # Step D: Measure duration
                    _, duration_sec = LipSyncProcessor.audio_to_mel(local_speech)
                    total_duration += duration_sec

                    segment_entry = {
                        "segmentIndex": idx + 1,
                        "questionId": q_id,
                        "tag": q_tag,
                        "questionText": q_text,
                        "durationSeconds": round(duration_sec, 2),
                        "videoUrl": video_url,
                        "status": "completed"
                    }
                    segments_metadata.append(segment_entry)

                    # Update progressive status in Firestore
                    interview_ref.update({
                        "completedQuestions": idx + 1,
                        "segments": segments_metadata,
                        "totalDurationSeconds": round(total_duration, 2),
                        "updatedAt": get_utc_now_iso()
                    })

                    # Cleanup per-question local temp files
                    try: os.remove(local_speech)
                    except Exception: pass
                    try: os.remove(local_q_output)
                    except Exception: pass

                # 4. Finalize Interview Document
                interview_ref.update({
                    "status": JobStatus.COMPLETED.value,
                    "segments": segments_metadata,
                    "totalDurationSeconds": round(total_duration, 2),
                    "updatedAt": get_utc_now_iso(),
                    "error": None
                })
                logger.info(f"[{interview_id}] Interview generation completed: {len(segments_metadata)} questions ({total_duration:.1f}s)")

        except Exception as e:
            error_msg = f"Interview generation failed: {str(e)}"
            logger.error(f"[{interview_id}] Error: {error_msg}", exc_info=True)
            interview_ref.update({
                "status": JobStatus.ERROR.value,
                "error": error_msg,
                "updatedAt": get_utc_now_iso()
            })
            raise
