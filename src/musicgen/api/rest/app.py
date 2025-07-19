"""
FastAPI application for MusicGen API.

Production-ready async endpoints with background tasks.
"""

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from musicgen.infrastructure.config.config import config
from musicgen.infrastructure.monitoring.logging import setup_logging
from musicgen.infrastructure.monitoring.metrics import metrics
from musicgen.utils.exceptions import MusicGenError
from musicgen.api.rest.middleware.rate_limiting import RateLimitMiddleware

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global model storage
_model_cache = {}


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
    """Job status response."""

    job_id: str
    status: str  # "queued", "processing", "completed", "failed"
    progress: float = 0.0
    message: str = ""
    audio_url: Optional[str] = None
    error: Optional[str] = None


# Background task storage
_jobs: dict[str, JobStatus] = {}


async def load_model(model_name: str):
    """Load MusicGen model with caching using transformers library."""
    if model_name not in _model_cache:
        try:
            logger.info(f"Loading model: {model_name}")
            # Import here to avoid startup issues
            from transformers import AutoProcessor, MusicgenForConditionalGeneration
            import torch

            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load processor and model
            processor = AutoProcessor.from_pretrained(model_name)
            model = MusicgenForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            model.to(device)

            # Cache both processor and model
            _model_cache[model_name] = {"model": model, "processor": processor, "device": device}
            logger.info(f"Model loaded successfully: {model_name} on {device}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise MusicGenError(f"Failed to load model: {e}")

    return _model_cache[model_name]


def _generate_music_sync(processor, model, device, request: GenerationRequest):
    """Synchronous music generation function for executor."""
    import torch

    # Process prompt
    inputs = processor(text=[request.prompt], padding=True, return_tensors="pt").to(device)

    # Calculate tokens for duration (256 tokens ≈ 5 seconds)
    max_new_tokens = int(256 * request.duration / 5)

    # Generate audio with transformers
    with torch.no_grad():
        audio_values = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p if request.top_p > 0 else None,
            guidance_scale=request.cfg_coef,
        )

    # Extract audio as numpy array
    audio = audio_values[0, 0].cpu().numpy()
    return audio


async def generate_music_task(job_id: str, request: GenerationRequest):
    """Background task for music generation."""
    try:
        # Update job status
        _jobs[job_id].status = "processing"
        _jobs[job_id].progress = 0.1
        _jobs[job_id].message = "Loading model..."

        # Load model components
        model_cache = await load_model(request.model)
        model = model_cache["model"]
        processor = model_cache["processor"]
        device = model_cache["device"]

        _jobs[job_id].progress = 0.3
        _jobs[job_id].message = "Generating music..."

        # Generate music using transformers approach
        logger.info(f"Generating music for job {job_id} with prompt: {request.prompt}")

        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        audio_data = await loop.run_in_executor(
            None, lambda: _generate_music_sync(processor, model, device, request)
        )

        _jobs[job_id].progress = 0.8
        _jobs[job_id].message = "Saving audio..."

        # Save audio file
        output_dir = config.OUTPUT_DIR
        if not output_dir.startswith("/app/"):
            # Local environment
            output_dir = os.path.join(os.getcwd(), "outputs")

        output_path = os.path.join(output_dir, f"{job_id}.wav")

        # Import here to avoid startup issues
        import torchaudio
        import torch

        # Save the audio (audio_data is already numpy array)
        audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)  # Add batch dimension
        sample_rate = model.config.audio_encoder.sampling_rate
        torchaudio.save(output_path, audio_tensor, sample_rate)

        # Update job status
        _jobs[job_id].status = "completed"
        _jobs[job_id].progress = 1.0
        _jobs[job_id].message = "Generation completed successfully"
        _jobs[job_id].audio_url = f"/audio/{job_id}.wav"

        logger.info(f"Music generation completed for job {job_id}")

        # Update metrics
        metrics.generation_completed.inc()

    except Exception as e:
        logger.error(f"Music generation failed for job {job_id}: {e}")
        _jobs[job_id].status = "failed"
        _jobs[job_id].error = str(e)
        _jobs[job_id].message = f"Generation failed: {e}"

        # Update metrics
        metrics.generation_failed.inc()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    logger.info("Starting MusicGen API")

    # Create output directory using config
    output_dir = config.OUTPUT_DIR
    if output_dir.startswith("/app/"):
        # Docker environment
        output_dir = output_dir
    else:
        # Local environment
        output_dir = os.path.join(os.getcwd(), "outputs")

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Pre-load default model if configured
    if config.MODEL_NAME:
        try:
            await load_model(config.MODEL_NAME)
        except Exception as e:
            logger.warning(f"Failed to pre-load model {config.MODEL_NAME}: {e}")

    yield

    logger.info("Shutting down MusicGen API")


# Create FastAPI app
app = FastAPI(
    title="MusicGen API",
    description="Production-ready AI music generation API",
    version="2.0.1",
    lifespan=lifespan,
)

# Add security middleware
app.add_middleware(RateLimitMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=config.CORS_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "musicgen-api", "version": "2.0.1"}


@app.post("/generate", response_model=GenerationResponse)
async def generate_music(request: GenerationRequest, background_tasks: BackgroundTasks):
    """Generate music from text prompt."""
    try:
        # Create job ID
        job_id = str(uuid.uuid4())

        # Initialize job status
        _jobs[job_id] = JobStatus(
            job_id=job_id, status="queued", message="Job queued for processing"
        )

        # Add background task
        background_tasks.add_task(generate_music_task, job_id, request)

        logger.info(f"Music generation job {job_id} queued")

        # Update metrics
        metrics.generation_requests.inc()

        return GenerationResponse(
            job_id=job_id, status="queued", message="Music generation job queued successfully"
        )

    except Exception as e:
        logger.error(f"Failed to queue music generation: {e}")
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
        # Local environment
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
                "description": "Small model (300M parameters) - Fast generation",
                "memory_usage": "2GB",
                "quality": "Good",
            },
            {
                "name": "facebook/musicgen-medium",
                "description": "Medium model (1.5B parameters) - Balanced performance",
                "memory_usage": "6GB",
                "quality": "Very Good",
            },
            {
                "name": "facebook/musicgen-large",
                "description": "Large model (3.3B parameters) - Best quality",
                "memory_usage": "12GB",
                "quality": "Excellent",
            },
        ]
    }


@app.get("/metrics")
async def get_metrics():
    """Get API metrics."""
    return {
        "generation_requests": metrics.generation_requests._value._value,
        "generation_completed": metrics.generation_completed._value._value,
        "generation_failed": metrics.generation_failed._value._value,
        "active_jobs": len([j for j in _jobs.values() if j.status == "processing"]),
        "total_jobs": len(_jobs),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT, log_level=config.LOG_LEVEL.lower())
