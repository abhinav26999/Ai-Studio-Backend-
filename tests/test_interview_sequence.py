import os
import pytest
import tempfile
import asyncio
from app.models.interview_job import (
    InterviewQuestion,
    GenerateInterviewSequenceRequest,
    CandidateInfo
)
from app.services.interview_service import InterviewService
from app.services.lip_sync_processor import LipSyncProcessor

# 12 Akash Kumar Solanki Interview Questions
EXAMPLE_QUESTIONS = [
    InterviewQuestion(
        id="Q1",
        tag="introduction",
        text=(
            "Hey Akash Kumar Solanki, I’m Aeologic Bot, the interviewer representative from Aeologic Technologies. "
            "It’s a pleasure to connect with you today. To begin, I would love to learn more about you. "
            "Could you please introduce yourself and share your professional journey so far? "
            "I’d also be interested in hearing about the key skills, technical expertise, and industry knowledge "
            "you have gained through your experience over the years, along with the projects or achievements you are most proud of."
        ),
        voice="en-US-GuyNeural"
    ),
    InterviewQuestion(
        id="Q2",
        tag="technicalHTML",
        text="Can you explain what HTML5 is and how it differs from previous versions of HTML?",
        voice="en-US-GuyNeural"
    ),
    InterviewQuestion(
        id="Q3",
        tag="technicalCSS",
        text="What are some of the new features introduced in CSS3?",
        voice="en-US-GuyNeural"
    ),
    InterviewQuestion(
        id="Q4",
        tag="technicalJavaScript",
        text="How would you describe the role of JavaScript in front-end development?",
        voice="en-US-GuyNeural"
    ),
    InterviewQuestion(
        id="Q5",
        tag="technicalReact.js",
        text="What is React.js and why is it popular for building user interfaces?",
        voice="en-US-GuyNeural"
    ),
    InterviewQuestion(
        id="Q6",
        tag="technicalReact.js",
        text="Can you explain the concept of reusable components in React?",
        voice="en-US-GuyNeural"
    ),
    InterviewQuestion(
        id="Q7",
        tag="technicalResponsiveDesign",
        text="What is responsive design and why is it important in web development?",
        voice="en-US-GuyNeural"
    ),
    InterviewQuestion(
        id="Q8",
        tag="technicalCSSFrameworks",
        text="How do you use Bootstrap or Tailwind CSS to create responsive layouts?",
        voice="en-US-GuyNeural"
    ),
    InterviewQuestion(
        id="Q9",
        tag="technicalAPIs",
        text="What is a REST API, and how would you integrate it into a React application?",
        voice="en-US-GuyNeural"
    ),
    InterviewQuestion(
        id="Q10",
        tag="technicalJSON",
        text="Can you explain what JSON is and how it is used in web applications?",
        voice="en-US-GuyNeural"
    ),
    InterviewQuestion(
        id="Q11",
        tag="technicalCrossBrowserCompatibility",
        text="How do you ensure cross-browser compatibility when developing web applications?",
        voice="en-US-GuyNeural"
    ),
    InterviewQuestion(
        id="Q12",
        tag="technicalDebuggingAndTesting",
        text="What tools do you use for debugging and testing your applications?",
        voice="en-US-GuyNeural"
    ),
]


def test_interview_request_model_validation():
    """Validates that the 12-question payload parses cleanly into Pydantic models."""
    req = GenerateInterviewSequenceRequest(
        avatarMediaUrl="https://storage.googleapis.com/test-bucket/avatar.jpg",
        candidate=CandidateInfo(
            name="Akash Kumar Solanki",
            email="akashsolanki1292001@gmail.com",
            difficulty="easy"
        ),
        questions=EXAMPLE_QUESTIONS,
        generateIdleLoop=True
    )
    assert len(req.questions) == 12
    assert req.candidate.name == "Akash Kumar Solanki"
    assert req.questions[0].id == "Q1"
    assert req.questions[11].id == "Q12"


def test_tts_generation_and_duration():
    """Tests generating TTS audio for Q2 & computing its mel duration."""
    async def _runner():
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = os.path.join(temp_dir, "q2_audio.mp3")
            await LipSyncProcessor.generate_tts_audio(
                EXAMPLE_QUESTIONS[1].text,
                "en-US-GuyNeural",
                audio_path
            )
            assert os.path.exists(audio_path)
            assert os.path.getsize(audio_path) > 1000

            mel, duration = LipSyncProcessor.audio_to_mel(audio_path)
            assert duration > 1.0
            assert mel.shape[0] == 80

    asyncio.run(_runner())


def test_idle_loop_generation():
    """Tests generating a 2.5s idle breathing loop video from a local test image."""
    test_img_path = "test_clean_green.png"
    if os.path.exists(test_img_path):
        with tempfile.TemporaryDirectory() as temp_dir:
            idle_output = os.path.join(temp_dir, "idle_test.mp4")
            result = InterviewService.generate_idle_loop(test_img_path, idle_output, duration_sec=2.0)
            assert os.path.exists(result)
            assert os.path.getsize(result) > 5000
