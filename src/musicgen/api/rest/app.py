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

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

from musicgen.api.middleware.auth import (
    UserClaims, UserRole, auth_middleware, get_current_user, require_auth
)
from musicgen.api.rest.middleware.rate_limiting import RateLimitMiddleware
from musicgen.infrastructure.config.config import config
from musicgen.infrastructure.monitoring.logging import setup_logging
from musicgen.infrastructure.monitoring.metrics import metrics
from musicgen.utils.exceptions import AuthenticationError, MusicGenError

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


# Auth models
class UserRegistration(BaseModel):
    """User registration request."""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')
    password: str = Field(..., min_length=6)
    full_name: Optional[str] = None


class UserResponse(BaseModel):
    """User response model."""
    user_id: str
    username: str
    email: str
    roles: list[str]
    tier: str
    is_verified: bool


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


# Background task storage
_jobs: dict[str, JobStatus] = {}

# Simple user storage for demo (in production, use a proper database)
_users: dict[str, dict] = {}


async def load_model(model_name: str):
    """Load MusicGen model with caching using transformers library."""
    if model_name not in _model_cache:
        try:
            logger.info(f"Loading model: {model_name}")
            # Import here to avoid startup issues
            import torch
            from transformers import AutoProcessor, MusicgenForConditionalGeneration

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
        import torch
        import torchaudio

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
        metrics.record_generation_request(request.model, "completed")

    except Exception as e:
        logger.error(f"Music generation failed for job {job_id}: {e}")
        _jobs[job_id].status = "failed"
        _jobs[job_id].error = str(e)
        _jobs[job_id].message = f"Generation failed: {e}"

        # Update metrics
        metrics.record_generation_request(request.model, "failed")


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
        metrics.record_generation_request(request.model, "queued")

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
    metrics_summary = metrics.get_metrics_summary()
    return {
        **metrics_summary,
        "active_jobs": len([j for j in _jobs.values() if j.status == "processing"]),
        "total_jobs": len(_jobs),
    }


# Authentication endpoints
@app.post("/auth/register")
async def register_user(user_data: UserRegistration):
    """Register a new user."""
    try:
        # Check if user already exists
        if any(u.get("email") == user_data.email for u in _users.values()):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )
        
        if any(u.get("username") == user_data.username for u in _users.values()):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
        
        # Create new user
        user_id = str(uuid.uuid4())
        user = {
            "user_id": user_id,
            "username": user_data.username,
            "email": user_data.email,
            "password": user_data.password,  # In production, hash this
            "roles": [UserRole.USER.value],
            "tier": "free",
            "is_verified": True,
            "tracks_generated": 0,
            "playlists_count": 0,
        }
        
        _users[user_id] = user
        logger.info(f"User registered: {user_data.username} ({user_data.email})")
        
        # Create tokens for auto-login
        access_token = auth_middleware.create_access_token(
            user_id=user_id,
            email=user_data.email,
            username=user_data.username,
            roles=[UserRole.USER.value],
            tier="free",
            is_verified=True,
        )
        
        refresh_token = auth_middleware.create_refresh_token(user_id)
        
        # Return registration response with tokens and user info
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "user": {
                "user_id": user_id,
                "username": user_data.username,
                "email": user_data.email,
                "roles": [UserRole.USER.value],
                "tier": "free",
                "is_verified": True,
                "tracks_generated": 0,
                "playlists_count": 0
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@app.post("/auth/login")
async def login_user(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login user and return JWT tokens."""
    try:
        # Find user by email (username field contains email)
        user = None
        for u in _users.values():
            if u.get("email") == form_data.username:
                user = u
                break
        
        if not user or user.get("password") != form_data.password:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create tokens
        access_token = auth_middleware.create_access_token(
            user_id=user["user_id"],
            email=user["email"],
            username=user["username"],
            roles=user["roles"],
            tier=user["tier"],
            is_verified=user["is_verified"],
        )
        
        refresh_token = auth_middleware.create_refresh_token(user["user_id"])
        
        logger.info(f"User logged in: {user['username']}")
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "user": {
                "user_id": user["user_id"],
                "username": user["username"],
                "email": user["email"],
                "roles": user["roles"],
                "tier": user["tier"],
                "is_verified": user["is_verified"],
                "tracks_generated": user.get("tracks_generated", 0),
                "playlists_count": user.get("playlists_count", 0)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@app.get("/auth/me")
async def get_current_user_info(current_user: UserClaims = Depends(require_auth)):
    """Get current user information."""
    # Get user from storage to include tracks_generated and playlists_count
    user_data = _users.get(current_user.user_id, {})
    
    return {
        "user_id": current_user.user_id,
        "username": current_user.username,
        "email": current_user.email,
        "roles": [role.value for role in current_user.roles],
        "tier": current_user.tier,
        "is_verified": current_user.is_verified,
        "tracks_generated": user_data.get("tracks_generated", 0),
        "playlists_count": user_data.get("playlists_count", 0)
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT, log_level=config.LOG_LEVEL.lower())
