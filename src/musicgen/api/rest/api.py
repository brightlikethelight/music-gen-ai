"""
Simple REST API for music generation.
No complexity, just endpoints that work.
"""

import asyncio
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Lazy imports to speed up module loading for tests
# These will be imported when first used
from ..streaming import list_sessions, websocket_endpoint
from .middleware.rate_limiting import RateLimitMiddleware, rate_limiter


# Lazy loading for heavy ML dependencies
def get_music_generator():
    """Lazy import MusicGenerator to avoid loading heavy ML dependencies."""
    from ...core.generator import MusicGenerator

    return MusicGenerator


def get_prompt_engineer():
    """Lazy import PromptEngineer."""
    from ...core.prompt import PromptEngineer

    return PromptEngineer


def get_batch_processor():
    """Lazy import BatchProcessor."""
    from ...services.batch import BatchProcessor

    return BatchProcessor


# Global executor for background tasks
_executor = ThreadPoolExecutor(max_workers=2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle with startup and shutdown events."""
    # Startup - only preload if not in test mode
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        try:
            generator = get_generator()
            print(f"✓ Model loaded: {generator.model_name}")
        except Exception as e:
            print(f"Warning: Failed to preload model: {e}")

    yield

    # Shutdown
    _executor.shutdown(wait=True)


# Create app
app = FastAPI(
    title="MusicGen API",
    description="Simple API for instrumental music generation",
    version="2.0.0",
    lifespan=lifespan,
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Rate Limiting for security
app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)

# Global generator instance (lazy loaded)
_generator = None


def get_generator():
    """Get or create generator instance."""
    global _generator
    if _generator is None:
        MusicGenerator = get_music_generator()
        _generator = MusicGenerator()
    return _generator


# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Music description", min_length=3, max_length=500)
    duration: float = Field(30.0, ge=0.1, le=300, description="Duration in seconds")
    temperature: float = Field(1.0, ge=0.1, le=2.0, description="Sampling temperature")
    guidance_scale: float = Field(3.0, ge=1.0, le=10.0, description="Guidance scale")
    format: str = Field("mp3", description="Output format (wav/mp3)")

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """Validate and sanitize prompt."""
        # Remove excessive whitespace
        v = " ".join(v.split())

        # Check for potential injection patterns
        dangerous_patterns = ["<script", "javascript:", "file://", "../", "\\x", "\0"]
        for pattern in dangerous_patterns:
            if pattern.lower() in v.lower():
                raise ValueError(f"Invalid prompt: contains potentially dangerous content")

        # Ensure prompt is music-related (basic check)
        if len(v) < 3:
            raise ValueError("Prompt too short, please provide a meaningful description")

        return v

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        """Validate output format."""
        allowed_formats = ["mp3", "wav"]
        if v.lower() not in allowed_formats:
            raise ValueError(f"Format must be one of: {', '.join(allowed_formats)}")
        return v.lower()

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prompt": "smooth jazz piano with soft drums",
                "duration": 30,
                "temperature": 1.0,
                "guidance_scale": 3.0,
                "format": "mp3",
            }
        }
    )


class GenerateResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: Optional[float] = None
    result_url: Optional[str] = None
    error: Optional[str] = None


class PromptRequest(BaseModel):
    prompt: str

    model_config = ConfigDict(json_schema_extra={"example": {"prompt": "jazz piano"}})


class PromptResponse(BaseModel):
    original: str
    improved: str
    is_valid: bool
    issues: list[str] = []
    variations: list[str] = []


# In-memory job tracking (educational demo only)
jobs = {}


async def generate_music_task(job_id: str, request: GenerateRequest):
    """Background task for music generation."""
    try:
        jobs[job_id] = {"status": "processing", "progress": 0}

        # Get generator
        generator = get_generator()

        # Progress callback
        def progress_callback(percent, message):
            jobs[job_id]["progress"] = percent

        # Generate music
        audio, sample_rate = await asyncio.get_event_loop().run_in_executor(
            _executor,
            lambda: generator.generate(
                request.prompt,
                request.duration,
                request.temperature,
                request.guidance_scale,
                progress_callback,
            ),
        )

        # Save audio
        output_dir = "api_outputs"
        os.makedirs(output_dir, exist_ok=True)

        filename = f"{output_dir}/{job_id}.{request.format}"
        await asyncio.get_event_loop().run_in_executor(
            _executor, lambda: generator.save_audio(audio, sample_rate, filename, request.format)
        )

        # Update job status
        jobs[job_id] = {
            "status": "completed",
            "progress": 100,
            "result_url": f"/download/{job_id}.{request.format}",
        }

    except Exception as e:
        jobs[job_id] = {"status": "failed", "error": str(e)}


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "MusicGen API",
        "version": "2.0.0",
        "endpoints": {
            "generate": "/generate",
            "status": "/status/{job_id}",
            "download": "/download/{filename}",
            "improve-prompt": "/improve-prompt",
            "batch": "/batch",
        },
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Generate music from text prompt.

    Returns a job ID for tracking progress.
    """
    # Create job ID
    job_id = str(uuid.uuid4())

    # Start background task
    background_tasks.add_task(generate_music_task, job_id, request)

    return GenerateResponse(job_id=job_id, status="accepted", message="Generation started")


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    """Get generation job status."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress"),
        result_url=job.get("result_url"),
        error=job.get("error"),
    )


@app.get("/download/{filename}")
async def download(filename: str):
    """Download generated audio file."""
    from pathlib import Path

    # Prevent directory traversal attacks
    output_dir = Path("api_outputs")
    output_dir.mkdir(exist_ok=True)  # Ensure directory exists

    # Safely construct the file path
    safe_filename = Path(filename).name  # This removes any directory components
    file_path = output_dir / safe_filename

    # Verify the resolved path is within our output directory
    try:
        file_path = file_path.resolve(strict=False)
        output_dir = output_dir.resolve(strict=False)
        if not file_path.is_relative_to(output_dir):
            raise HTTPException(status_code=403, detail="Access denied")
    except (ValueError, RuntimeError):
        raise HTTPException(status_code=403, detail="Invalid filename")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        str(file_path),
        media_type="audio/mpeg" if filename.endswith(".mp3") else "audio/wav",
        filename=safe_filename,
    )


@app.post("/improve-prompt", response_model=PromptResponse)
async def improve_prompt(request: PromptRequest):
    """Improve and validate a prompt."""
    PromptEngineer = get_prompt_engineer()
    engineer = PromptEngineer()

    # Validate
    is_valid, issues = engineer.validate_prompt(request.prompt)

    # Improve
    improved = engineer.improve_prompt(request.prompt)

    # Get variations
    variations = engineer.suggest_variations(improved, count=3)

    return PromptResponse(
        original=request.prompt,
        improved=improved,
        is_valid=is_valid,
        issues=issues,
        variations=variations,
    )


@app.post("/batch")
async def batch_process(file: UploadFile = File(...)):
    """
    Process batch generation from CSV file.

    CSV should have columns: prompt, duration, output_file
    """
    # Save uploaded file
    temp_path = f"temp_{uuid.uuid4()}.csv"

    try:
        contents = await file.read()
        with open(temp_path, "wb") as f:
            f.write(contents)

        # Process batch
        BatchProcessor = get_batch_processor()
        processor = BatchProcessor()
        jobs = processor.load_csv(temp_path)

        if not jobs:
            raise HTTPException(status_code=400, detail="No valid jobs in CSV")

        # Start processing (educational demo - simplified implementation)
        job_id = str(uuid.uuid4())

        # For now, return job info
        return {
            "batch_id": job_id,
            "total_jobs": len(jobs),
            "status": "accepted",
            "message": "Batch processing started",
        }

    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        # Check if model can be loaded
        generator = get_generator()
        info = generator.get_info()

        return {"status": "healthy", "model": info["model"], "device": info["device"]}
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "error": str(e)})


# WebSocket endpoint for streaming
@app.websocket("/ws/generate")
async def websocket_generate(websocket):
    """WebSocket endpoint for streaming music generation."""
    await websocket_endpoint(websocket)


@app.get("/streaming/sessions")
async def get_streaming_sessions():
    """Get list of active streaming sessions."""
    return {"sessions": list_sessions()}


def main():
    """Main entry point for API server."""
    import uvicorn

    # Use localhost by default for security, can be overridden by environment variable
    host = os.getenv("API_HOST", "127.0.0.1")
    uvicorn.run("musicgen.api:app", host=host, port=8000, reload=False)


if __name__ == "__main__":
    main()
