"""
Test version of FastAPI app with mocked music generation.

This tests the entire API flow without requiring working ML dependencies.
"""

import asyncio
import os
import uuid
import numpy as np
import wave
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from musicgen.infrastructure.config.config import config
from musicgen.infrastructure.monitoring.logging import setup_logging
from musicgen.infrastructure.monitoring.metrics import metrics

# Setup logging
setup_logging()


class GenerationRequest(BaseModel):
    """Request model for music generation."""

    prompt: str = Field(..., description="Text description of the music to generate")
    duration: float = Field(default=30.0, ge=1.0, le=600.0, description="Duration in seconds")
    model: str = Field(default="facebook/musicgen-small", description="Model to use")


class GenerationResponse(BaseModel):
    """Response model for music generation."""

    job_id: str
    status: str
    message: str


class JobStatus(BaseModel):
    """Job status response."""

    job_id: str
    status: str
    progress: float = 0.0
    message: str = ""
    audio_url: str = None
    error: str = None


# Background task storage
_jobs: dict[str, JobStatus] = {}


def generate_test_audio(duration: float, filename: str) -> None:
    """Generate a test audio file (beep sound)."""
    sample_rate = 22050
    samples = int(sample_rate * duration)

    # Generate a simple sine wave
    t = np.linspace(0, duration, samples)
    frequency = 440  # A4 note
    audio_data = np.sin(2 * np.pi * frequency * t) * 0.5

    # Convert to 16-bit integers
    audio_data = (audio_data * 32767).astype(np.int16)

    # Save as WAV file
    with wave.open(filename, "w") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())


async def mock_generate_music_task(job_id: str, request: GenerationRequest):
    """Mock background task for music generation."""
    try:
        # Update job status
        _jobs[job_id].status = "processing"
        _jobs[job_id].progress = 0.1
        _jobs[job_id].message = "Loading model..."

        # Simulate model loading
        await asyncio.sleep(1)

        _jobs[job_id].progress = 0.3
        _jobs[job_id].message = "Generating music..."

        # Simulate music generation
        await asyncio.sleep(2)

        _jobs[job_id].progress = 0.8
        _jobs[job_id].message = "Saving audio..."

        # Generate test audio
        output_dir = config.OUTPUT_DIR
        if not output_dir.startswith("/app/"):
            output_dir = os.path.join(os.getcwd(), "outputs")

        output_path = os.path.join(output_dir, f"{job_id}.wav")

        # Create the test audio file
        generate_test_audio(request.duration, output_path)

        # Update job status
        _jobs[job_id].status = "completed"
        _jobs[job_id].progress = 1.0
        _jobs[job_id].message = "Generation completed successfully"
        _jobs[job_id].audio_url = f"/audio/{job_id}.wav"

        print(f"✓ Mock music generation completed for job {job_id}")

    except Exception as e:
        print(f"✗ Mock music generation failed for job {job_id}: {e}")
        _jobs[job_id].status = "failed"
        _jobs[job_id].error = str(e)
        _jobs[job_id].message = f"Generation failed: {e}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    print("🚀 Starting Test MusicGen API")

    # Create output directory
    output_dir = config.OUTPUT_DIR
    if not output_dir.startswith("/app/"):
        output_dir = os.path.join(os.getcwd(), "outputs")

    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")

    yield

    print("🛑 Shutting down Test MusicGen API")


# Create FastAPI app
app = FastAPI(
    title="MusicGen Test API",
    description="Test version with mocked music generation",
    version="2.0.1-test",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "musicgen-test-api",
        "version": "2.0.1-test",
        "mode": "mock",
    }


@app.post("/generate", response_model=GenerationResponse)
async def generate_music(request: GenerationRequest, background_tasks: BackgroundTasks):
    """Generate music from text prompt (mocked)."""
    try:
        # Create job ID
        job_id = str(uuid.uuid4())

        # Initialize job status
        _jobs[job_id] = JobStatus(
            job_id=job_id, status="queued", message="Job queued for processing"
        )

        # Add background task
        background_tasks.add_task(mock_generate_music_task, job_id, request)

        print(f"🎵 Mock music generation job {job_id} queued")

        return GenerationResponse(
            job_id=job_id, status="queued", message="Mock music generation job queued successfully"
        )

    except Exception as e:
        print(f"❌ Failed to queue mock music generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue generation: {e}",
        )


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get job status."""
    if job_id not in _jobs:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    return _jobs[job_id]


@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """Serve generated audio files."""
    output_dir = config.OUTPUT_DIR
    if not output_dir.startswith("/app/"):
        output_dir = os.path.join(os.getcwd(), "outputs")

    file_path = os.path.join(output_dir, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Audio file not found")

    return FileResponse(file_path, media_type="audio/wav", filename=filename)


@app.get("/models")
async def list_models():
    """List available models."""
    return {
        "models": [
            {
                "name": "facebook/musicgen-small",
                "description": "Small model (300M parameters) - MOCKED",
                "memory_usage": "2GB",
                "quality": "Good",
                "mode": "mock",
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn

    print("🧪 Starting Test MusicGen API Server")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
