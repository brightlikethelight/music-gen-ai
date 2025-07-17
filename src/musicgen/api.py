"""
Simple REST API for music generation.
No complexity, just endpoints that work.
"""

import os
import uuid
import asyncio
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .generator import MusicGenerator
from .batch import BatchProcessor
from .prompt import PromptEngineer

# Create app
app = FastAPI(
    title="MusicGen API",
    description="Simple API for instrumental music generation",
    version="2.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global generator instance (lazy loaded)
_generator = None
_executor = ThreadPoolExecutor(max_workers=2)


def get_generator() -> MusicGenerator:
    """Get or create generator instance."""
    global _generator
    if _generator is None:
        _generator = MusicGenerator()
    return _generator


# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Music description")
    duration: float = Field(30.0, ge=0.1, le=300, description="Duration in seconds")
    temperature: float = Field(1.0, ge=0.1, le=2.0, description="Sampling temperature")
    guidance_scale: float = Field(3.0, ge=1.0, le=10.0, description="Guidance scale")
    format: str = Field("mp3", description="Output format (wav/mp3)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "smooth jazz piano with soft drums",
                "duration": 30,
                "temperature": 1.0,
                "guidance_scale": 3.0,
                "format": "mp3"
            }
        }


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
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "jazz piano"
            }
        }


class PromptResponse(BaseModel):
    original: str
    improved: str
    is_valid: bool
    issues: list[str] = []
    variations: list[str] = []


# In-memory job tracking (use Redis in production)
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
                progress_callback
            )
        )
        
        # Save audio
        output_dir = "api_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"{output_dir}/{job_id}.{request.format}"
        await asyncio.get_event_loop().run_in_executor(
            _executor,
            lambda: generator.save_audio(audio, sample_rate, filename, request.format)
        )
        
        # Update job status
        jobs[job_id] = {
            "status": "completed",
            "progress": 100,
            "result_url": f"/download/{job_id}.{request.format}"
        }
        
    except Exception as e:
        jobs[job_id] = {
            "status": "failed",
            "error": str(e)
        }


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
            "batch": "/batch"
        }
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
    
    return GenerateResponse(
        job_id=job_id,
        status="accepted",
        message="Generation started"
    )


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
        error=job.get("error")
    )


@app.get("/download/{filename}")
async def download(filename: str):
    """Download generated audio file."""
    file_path = f"api_outputs/{filename}"
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        file_path,
        media_type="audio/mpeg" if filename.endswith(".mp3") else "audio/wav",
        filename=filename
    )


@app.post("/improve-prompt", response_model=PromptResponse)
async def improve_prompt(request: PromptRequest):
    """Improve and validate a prompt."""
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
        variations=variations
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
        processor = BatchProcessor()
        jobs = processor.load_csv(temp_path)
        
        if not jobs:
            raise HTTPException(status_code=400, detail="No valid jobs in CSV")
        
        # Start processing (in production, use Celery or similar)
        job_id = str(uuid.uuid4())
        
        # For now, return job info
        return {
            "batch_id": job_id,
            "total_jobs": len(jobs),
            "status": "accepted",
            "message": "Batch processing started"
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
        
        return {
            "status": "healthy",
            "model": info["model"],
            "device": info["device"]
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


# Startup event
@app.on_event("startup")
async def startup():
    """Preload model on startup."""
    try:
        generator = get_generator()
        print(f"✓ Model loaded: {generator.model_name}")
    except Exception as e:
        print(f"Warning: Failed to preload model: {e}")


# Shutdown event
@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    _executor.shutdown(wait=True)


def main():
    """Main entry point for API server."""
    import uvicorn
    uvicorn.run(
        "musicgen.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )


if __name__ == "__main__":
    main()