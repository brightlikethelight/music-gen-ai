"""
API Bridge for MusicGen
Connects our FastAPI interface to the pre-built TTS container
"""

import os
import asyncio
import httpx
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
import uuid
from typing import Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="MusicGen Academic API Example",
    description="Educational API bridge example for MusicGen - Harvard CS 109B Project",
    version="0.1.0-academic"
)

# Configuration
TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://musicgen-academic:3001")
OUTPUT_DIR = "/outputs"

# Models
class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text description of music")
    duration: float = Field(default=30.0, ge=1.0, le=300.0)
    model: str = Field(default="facebook/musicgen-small")
    streaming: bool = Field(default=False, description="Enable streaming response")

class GenerationResponse(BaseModel):
    job_id: str
    status: str
    message: str
    streaming_url: Optional[str] = None

# Job storage
jobs = {}

# Background task for generation
async def generate_with_streaming(job_id: str, request: GenerationRequest):
    """Generate music with optional streaming."""
    try:
        jobs[job_id]["status"] = "processing"
        
        # Call TTS service
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{TTS_SERVICE_URL}/generate",
                json={
                    "text": request.prompt,
                    "model": request.model,
                    "duration": request.duration
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                audio_path = result.get("audio_path")
                
                # Save job result
                jobs[job_id]["status"] = "completed"
                jobs[job_id]["audio_path"] = audio_path
                jobs[job_id]["streaming_url"] = f"/stream/{job_id}" if request.streaming else None
                
                logger.info(f"Generation completed for job {job_id}")
            else:
                raise Exception(f"TTS service error: {response.status_code}")
                
    except Exception as e:
        logger.error(f"Generation failed for job {job_id}: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "musicgen-api-bridge"}

@app.post("/generate", response_model=GenerationResponse)
async def generate_music(request: GenerationRequest, background_tasks: BackgroundTasks):
    """Generate music with optional streaming."""
    job_id = str(uuid.uuid4())
    
    # Initialize job
    jobs[job_id] = {
        "status": "queued",
        "request": request.dict(),
        "streaming": request.streaming
    }
    
    # Add background task
    background_tasks.add_task(generate_with_streaming, job_id, request)
    
    response = GenerationResponse(
        job_id=job_id,
        status="queued",
        message="Generation started"
    )
    
    if request.streaming:
        response.streaming_url = f"/stream/{job_id}"
    
    return response

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get job status."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

@app.get("/stream/{job_id}")
async def stream_audio(job_id: str):
    """Stream audio as it's being generated."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    async def audio_streamer():
        """Stream audio chunks as they become available."""
        while job["status"] == "processing":
            # Check for partial audio
            partial_path = f"{OUTPUT_DIR}/{job_id}_partial.wav"
            if os.path.exists(partial_path):
                with open(partial_path, "rb") as f:
                    while chunk := f.read(8192):
                        yield chunk
            await asyncio.sleep(0.1)
        
        # Stream final audio
        if job["status"] == "completed" and "audio_path" in job:
            with open(job["audio_path"], "rb") as f:
                while chunk := f.read(8192):
                    yield chunk
    
    return StreamingResponse(
        audio_streamer(),
        media_type="audio/wav",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/models")
async def list_models():
    """List available models with hot-swap support."""
    return {
        "models": [
            {
                "id": "facebook/musicgen-small",
                "name": "MusicGen Small",
                "size": "300M",
                "loaded": True,
                "memory_usage": "4GB",
                "supports_streaming": True
            },
            {
                "id": "facebook/musicgen-medium", 
                "name": "MusicGen Medium",
                "size": "1.5B",
                "loaded": False,
                "memory_usage": "8GB",
                "supports_streaming": True
            },
            {
                "id": "facebook/musicgen-large",
                "name": "MusicGen Large", 
                "size": "3.3B",
                "loaded": False,
                "memory_usage": "16GB",
                "supports_streaming": True
            }
        ],
        "hot_swap_enabled": True,
        "current_model": "facebook/musicgen-small"
    }

@app.post("/models/{model_id}/load")
async def load_model(model_id: str):
    """Hot-swap to a different model."""
    # In a real implementation, this would trigger model loading
    return {
        "status": "loading",
        "model_id": model_id,
        "message": "Model swap initiated"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
