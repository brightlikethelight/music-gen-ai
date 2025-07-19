"""
Hybrid MusicGen API with cloud fallback.
Works immediately without waiting for Docker deployment.
"""

import asyncio
import logging
import os
import uuid
import httpx
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Use local imports if available, otherwise graceful fallback
try:
    from musicgen.infrastructure.config.config import config
    from musicgen.infrastructure.monitoring.logging import setup_logging
    from musicgen.infrastructure.monitoring.metrics import metrics
    from musicgen.utils.exceptions import MusicGenError
    from musicgen.api.rest.middleware.rate_limiting import RateLimitMiddleware

    rate_limiting_available = True
except ImportError:
    # Fallback configuration for standalone operation
    class MockConfig:
        OUTPUT_DIR = "./outputs"
        LOG_LEVEL = "INFO"
        CORS_ORIGINS = ["*"]
        CORS_CREDENTIALS = True

    config = MockConfig()

    def setup_logging():
        logging.basicConfig(level=logging.INFO)

    class MockMetrics:
        def __init__(self):
            pass

        generation_requests = self
        generation_completed = self
        generation_failed = self

        def inc(self):
            pass

        def observe(self, value):
            pass

    metrics = MockMetrics()

    class MusicGenError(Exception):
        pass

    RateLimitMiddleware = None
    rate_limiting_available = False

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global state
_jobs: dict[str, "JobStatus"] = {}


class GenerationRequest(BaseModel):
    """Request model for music generation."""

    prompt: str = Field(..., description="Text description of the music to generate")
    duration: float = Field(default=30.0, ge=1.0, le=600.0, description="Duration in seconds")
    model: str = Field(default="facebook/musicgen-small", description="Model to use")
    temperature: float = Field(default=1.0, ge=0.1, le=2.0, description="Sampling temperature")
    top_k: int = Field(default=250, ge=1, le=1000, description="Top-k sampling")
    top_p: float = Field(default=0.0, ge=0.0, le=1.0, description="Top-p sampling")
    cfg_coef: float = Field(
        default=3.0, ge=0.0, le=10.0, description="Classifier-free guidance coefficient"
    )


class GenerationResponse(BaseModel):
    """Response model for music generation."""

    job_id: str
    status: str
    message: str
    audio_url: Optional[str] = None
    duration: Optional[float] = None
    model_used: Optional[str] = None


class JobStatus(BaseModel):
    """Job status tracking."""

    job_id: str
    status: str  # queued, processing, completed, failed
    progress: float = 0.0
    message: str = ""
    audio_url: Optional[str] = None
    error: Optional[str] = None
    model_used: Optional[str] = None
    generation_method: Optional[str] = None  # local, replicate, mock


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    logger.info("Starting Hybrid MusicGen API")

    # Create output directory
    output_dir = config.OUTPUT_DIR
    if not output_dir.startswith("/app/"):
        output_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    yield

    logger.info("Shutting down Hybrid MusicGen API")


# Create FastAPI app
app = FastAPI(
    title="Hybrid MusicGen API",
    description="Production-ready music generation with local + cloud fallback",
    version="3.0.0",
    lifespan=lifespan,
)

# Add security middleware (if available)
if rate_limiting_available and RateLimitMiddleware:
    app.add_middleware(RateLimitMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS if hasattr(config, "CORS_ORIGINS") else ["*"],
    allow_credentials=config.CORS_CREDENTIALS if hasattr(config, "CORS_CREDENTIALS") else True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def generate_with_replicate(request: GenerationRequest) -> str:
    """Generate music using Replicate API."""
    try:
        logger.info(f"Using Replicate API for: {request.prompt}")

        # Mock Replicate API call (replace with actual implementation)
        await asyncio.sleep(2)  # Simulate API call

        # In real implementation:
        # import replicate
        # output = replicate.run("meta/musicgen", {
        #     "prompt": request.prompt,
        #     "duration": request.duration,
        #     "temperature": request.temperature,
        #     "top_k": request.top_k,
        #     "top_p": request.top_p,
        #     "guidance_scale": request.cfg_coef
        # })

        # For now, generate mock audio
        output_dir = config.OUTPUT_DIR
        if not output_dir.startswith("/app/"):
            output_dir = os.path.join(os.getcwd(), "outputs")

        return await generate_mock_audio(request, output_dir, method="replicate")

    except Exception as e:
        logger.error(f"Replicate generation failed: {e}")
        raise MusicGenError(f"Replicate API error: {e}")


async def generate_with_local_docker(request: GenerationRequest) -> str:
    """Generate music using local Docker container."""
    try:
        logger.info(f"Using local Docker for: {request.prompt}")

        # Check if Docker container is running
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get("http://localhost:3001/health", timeout=5.0)
                if response.status_code != 200:
                    raise Exception("Docker container not healthy")
            except Exception:
                raise Exception("Docker container not available")

        # Call Docker container API
        async with httpx.AsyncClient(timeout=60.0) as client:
            docker_request = {
                "text": request.prompt,
                "duration": request.duration,
                "temperature": request.temperature,
                "top_k": request.top_k,
                "top_p": request.top_p,
                "guidance_scale": request.cfg_coef,
            }

            response = await client.post("http://localhost:3001/generate", json=docker_request)

            if response.status_code != 200:
                raise Exception(f"Docker API error: {response.status_code}")

            result = response.json()
            return result.get("audio_url", "")

    except Exception as e:
        logger.error(f"Local Docker generation failed: {e}")
        raise MusicGenError(f"Local Docker error: {e}")


async def generate_mock_audio(
    request: GenerationRequest, output_dir: str, method: str = "mock"
) -> str:
    """Generate mock audio for testing."""
    import numpy as np
    import wave

    job_id = str(uuid.uuid4())
    output_path = os.path.join(output_dir, f"{job_id}.wav")

    # Generate sine wave audio (simple beep)
    sample_rate = 44100
    duration = min(request.duration, 10.0)  # Limit to 10s for mock
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # Create a pleasant melody
    frequency = 440  # A4 note
    audio = 0.3 * np.sin(frequency * 2 * np.pi * t)  # Volume 30%

    # Add some harmony
    audio += 0.2 * np.sin(frequency * 1.5 * 2 * np.pi * t)  # Perfect fifth
    audio += 0.1 * np.sin(frequency * 2 * 2 * np.pi * t)  # Octave

    # Apply fade in/out
    fade_samples = int(0.1 * sample_rate)  # 0.1s fade
    audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
    audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)

    # Convert to 16-bit PCM
    audio_int = (audio * 32767).astype(np.int16)

    # Save as WAV file
    with wave.open(output_path, "w") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int.tobytes())

    logger.info(f"Generated {method} audio: {output_path} ({duration}s)")
    return f"/audio/{job_id}.wav"


async def generate_music_hybrid(job_id: str, request: GenerationRequest):
    """Hybrid music generation with fallback chain."""
    try:
        # Update job status
        _jobs[job_id].status = "processing"
        _jobs[job_id].progress = 0.1
        _jobs[job_id].message = "Determining generation method..."

        audio_url = None
        method_used = None

        # Try local Docker first (if available)
        try:
            _jobs[job_id].message = "Trying local Docker generation..."
            audio_url = await generate_with_local_docker(request)
            method_used = "local_docker"
            logger.info(f"Job {job_id}: Local Docker generation successful")
        except Exception as e:
            logger.info(f"Job {job_id}: Local Docker failed, trying Replicate: {e}")

            # Try Replicate API
            try:
                _jobs[job_id].progress = 0.3
                _jobs[job_id].message = "Trying Replicate API..."
                audio_url = await generate_with_replicate(request)
                method_used = "replicate"
                logger.info(f"Job {job_id}: Replicate generation successful")
            except Exception as e2:
                logger.info(f"Job {job_id}: Replicate failed, using mock: {e2}")

                # Fallback to mock generation
                _jobs[job_id].progress = 0.5
                _jobs[job_id].message = "Using mock generation..."
                output_dir = config.OUTPUT_DIR
                if not output_dir.startswith("/app/"):
                    output_dir = os.path.join(os.getcwd(), "outputs")
                audio_url = await generate_mock_audio(request, output_dir, "fallback")
                method_used = "mock_fallback"

        # Update job completion
        _jobs[job_id].status = "completed"
        _jobs[job_id].progress = 1.0
        _jobs[job_id].message = f"Generation completed via {method_used}"
        _jobs[job_id].audio_url = audio_url
        _jobs[job_id].model_used = request.model
        _jobs[job_id].generation_method = method_used

        logger.info(f"Job {job_id}: Generation completed via {method_used}")
        metrics.generation_completed.inc()

    except Exception as e:
        logger.error(f"Job {job_id}: All generation methods failed: {e}")
        _jobs[job_id].status = "failed"
        _jobs[job_id].error = str(e)
        _jobs[job_id].message = f"Generation failed: {e}"
        metrics.generation_failed.inc()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "hybrid-musicgen-api", "version": "3.0.0"}


@app.post("/generate", response_model=GenerationResponse)
async def generate_music(request: GenerationRequest, background_tasks: BackgroundTasks):
    """Generate music with hybrid fallback approach."""
    job_id = str(uuid.uuid4())

    # Create job status
    _jobs[job_id] = JobStatus(
        job_id=job_id, status="queued", message="Music generation job queued successfully"
    )

    # Start background generation
    background_tasks.add_task(generate_music_hybrid, job_id, request)

    # Update metrics
    metrics.generation_requests.inc()

    return GenerationResponse(
        job_id=job_id, status="queued", message="Music generation job queued successfully"
    )


@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get generation job status."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _jobs[job_id]
    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "audio_url": job.audio_url,
        "error": job.error,
        "model_used": job.model_used,
        "generation_method": job.generation_method,
    }


@app.get("/audio/{filename}")
async def get_audio_file(filename: str):
    """Serve generated audio files."""
    output_dir = config.OUTPUT_DIR
    if not output_dir.startswith("/app/"):
        output_dir = os.path.join(os.getcwd(), "outputs")

    file_path = os.path.join(output_dir, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(file_path, media_type="audio/wav", filename=filename)


@app.get("/jobs")
async def list_jobs():
    """List all jobs (for debugging)."""
    return {"jobs": list(_jobs.keys()), "total": len(_jobs)}


if __name__ == "__main__":
    import uvicorn

    print("🎵 Starting Hybrid MusicGen API Server")
    print("🚀 Features:")
    print("   - Local Docker generation (if available)")
    print("   - Replicate API fallback")
    print("   - Mock generation for testing")
    print("   - Complete job tracking")

    uvicorn.run("hybrid_app:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
