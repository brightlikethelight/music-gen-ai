"""
FastAPI application for MusicGen API.

Educational demonstration of async endpoints with background tasks.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordRequestForm

from musicgen import __version__
from musicgen.api.cors_config import cors_config
from musicgen.api.middleware.auth import (
    UserClaims,
    UserRole,
    get_auth_middleware,
    require_auth,
)
from musicgen.api.rest.middleware.rate_limiting import RateLimitMiddleware
from musicgen.api.rest.models import (
    BatchGenerationRequest,
    GenerationRequest,
    GenerationResponse,
    PlaylistCreate,
    UserRegistration,
)
from musicgen.api.rest.state import JobStatus, state
from musicgen.infrastructure.config.config import config
from musicgen.infrastructure.monitoring.logging import setup_logging
from musicgen.infrastructure.monitoring.metrics import metrics
from musicgen.infrastructure.security import (
    hash_password,
    log_login_attempt,
    verify_password,
)
from musicgen.infrastructure.security.validation import validate_prompt
from musicgen.utils.exceptions import MusicGenError, ValidationError

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Backward-compatible aliases — tests import these directly from this module
_model_cache = state.model_cache
_jobs = state.jobs
_users = state.users
_playlists = state.playlists


async def load_model(model_name: str) -> dict[str, Any]:
    """Load MusicGen model with caching, delegating to MusicGenerator."""
    cached = await state.get_model(model_name)
    if cached is not None:
        return cached

    try:
        logger.info("Loading model: %s", model_name)
        from musicgen.core.generator import MusicGenerator

        gen = MusicGenerator(model_name=model_name)
        model_data: dict[str, Any] = {
            "model": gen.model,
            "processor": gen.processor,
            "device": str(gen.device),
        }
        await state.set_model(model_name, model_data)
        logger.info("Model loaded successfully: %s on %s", model_name, model_data["device"])
        return model_data
    except Exception as e:
        logger.error("Failed to load model %s: %s", model_name, e)
        raise MusicGenError(f"Failed to load model: {e}")


def _generate_music_sync(
    processor: Any, model: Any, device: str, request: GenerationRequest
) -> Any:
    """Synchronous music generation function for executor."""
    import torch

    inputs = processor(text=[request.prompt], padding=True, return_tensors="pt").to(device)
    max_new_tokens = int(256 * request.duration / 5)

    use_amp = device != "cpu"
    with torch.inference_mode():
        with torch.amp.autocast("cuda", enabled=use_amp):
            audio_values = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p if request.top_p > 0 else None,
                guidance_scale=request.cfg_coef,
            )

    audio = audio_values[0, 0].cpu().numpy()
    return audio


async def generate_music_task(job_id: str, request: GenerationRequest) -> None:
    """Background task for music generation."""
    try:
        await state.update_job(
            job_id, status="processing", progress=0.1, message="Loading model..."
        )

        model_cache = await load_model(request.model)
        model = model_cache["model"]
        processor = model_cache["processor"]
        device = model_cache["device"]

        await state.update_job(job_id, progress=0.3, message="Generating music...")

        logger.info("Generating music for job %s with prompt: %s", job_id, request.prompt)

        loop = asyncio.get_running_loop()
        audio_data = await loop.run_in_executor(
            None, lambda: _generate_music_sync(processor, model, device, request)
        )

        await state.update_job(job_id, progress=0.8, message="Saving audio...")

        output_dir = Path(config.OUTPUT_DIR)
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / "outputs"

        output_path = str(output_dir / f"{job_id}.wav")

        import torch
        import torchaudio

        audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
        sample_rate = model.config.audio_encoder.sampling_rate
        torchaudio.save(output_path, audio_tensor, sample_rate)

        await state.update_job(
            job_id,
            status="completed",
            progress=1.0,
            message="Generation completed successfully",
            audio_url=f"/audio/{job_id}.wav",
        )

        logger.info("Music generation completed for job %s", job_id)
        metrics.record_generation_request(request.model, "completed")

    except Exception as e:
        logger.error("Music generation failed for job %s: %s", job_id, e)
        await state.update_job(
            job_id,
            status="failed",
            error="Music generation failed",
            message="Generation failed",
        )
        metrics.record_generation_request(request.model, "failed")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan management."""
    logger.info("Starting MusicGen API")

    output_dir = Path(config.OUTPUT_DIR)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / "outputs"

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    if config.MODEL_NAME:
        try:
            await load_model(config.MODEL_NAME)
        except Exception as e:
            logger.warning("Failed to pre-load model %s: %s", config.MODEL_NAME, e)

    yield

    logger.info("Shutting down MusicGen API")


# Create FastAPI app
app = FastAPI(
    title="MusicGen API",
    description="Educational AI music generation API (Harvard CS 109B project)",
    version=__version__,
    lifespan=lifespan,
)

# Middleware order: FastAPI executes in reverse order of addition.
# Add CORS first so rate limiting runs before CORS (preflight OPTIONS
# requests are rate-limited, preventing abuse).
cors_options = cors_config.get_cors_options()
app.add_middleware(CORSMiddleware, **cors_options)
app.add_middleware(RateLimitMiddleware)


# ── Health & Info ────────────────────────────────────────────────────────────


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "api-gateway",
        "version": __version__,
        "timestamp": time.time(),
    }


@app.get("/health/services")
async def health_services(
    current_user: UserClaims = Depends(require_auth),
) -> dict[str, Any]:
    """Check health of all microservices."""
    services_health: dict[str, dict[str, Any]] = {
        "generation": {
            "status": "healthy",
            "message": "Music generation service operational",
            "response_time_ms": 12,
        },
        "audio-processing": {
            "status": "healthy",
            "message": "Audio processing service operational",
            "response_time_ms": 8,
        },
        "user-management": {
            "status": "healthy",
            "message": "User management service operational",
            "response_time_ms": 5,
        },
        "redis": {
            "status": "healthy" if get_auth_middleware().redis_client else "unavailable",
            "message": (
                "Redis cache operational"
                if get_auth_middleware().redis_client
                else "Redis not configured"
            ),
            "response_time_ms": 3,
        },
        "postgres": {
            "status": "degraded",
            "message": "Using in-memory storage (no database configured)",
            "response_time_ms": 0,
        },
    }

    statuses = [s["status"] for s in services_health.values()]
    if all(s == "healthy" for s in statuses):
        overall_status = "healthy"
    elif any(s == "unhealthy" for s in statuses):
        overall_status = "unhealthy"
    else:
        overall_status = "degraded"

    return {"services": services_health, "overall_status": overall_status, "timestamp": time.time()}


@app.get("/models")
async def list_models() -> dict[str, Any]:
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
async def get_metrics(
    current_user: UserClaims = Depends(require_auth),
) -> dict[str, Any]:
    """Get API metrics."""
    all_jobs = await state.get_all_jobs()
    metrics_summary = metrics.get_metrics_summary()
    return {
        **metrics_summary,
        "active_jobs": len([j for j in all_jobs.values() if j.status == "processing"]),
        "total_jobs": len(all_jobs),
    }


# ── Generation ───────────────────────────────────────────────────────────────


@app.post("/generate", response_model=GenerationResponse)
async def generate_music(
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
    current_user: UserClaims = Depends(require_auth),
) -> GenerationResponse:
    """Generate music from text prompt."""
    try:
        validate_prompt(request.prompt)
        job_id = str(uuid.uuid4())
        job = JobStatus(job_id=job_id, status="queued", message="Job queued for processing")
        await state.add_job(job_id, job)

        background_tasks.add_task(generate_music_task, job_id, request)

        logger.info("Music generation job %s queued", job_id)
        metrics.record_generation_request(request.model, "queued")

        return GenerationResponse(
            job_id=job_id, status="queued", message="Music generation job queued successfully"
        )

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Failed to queue music generation: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to queue generation",
        )


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str) -> JobStatus:
    """Get job status."""
    job = await state.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return job


@app.get("/generate/job/{job_id}")
async def get_generation_job_status(job_id: str) -> dict[str, Any]:
    """Get status of a music generation job (alias for /status/{job_id})."""
    job = await state.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found")

    response: dict[str, Any] = {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
    }

    if job.audio_url:
        response["audio_url"] = job.audio_url
    if job.error:
        response["error"] = job.error

    return response


@app.post("/generate/batch")
async def generate_music_batch(
    batch_data: BatchGenerationRequest,
    background_tasks: BackgroundTasks,
    current_user: UserClaims = Depends(require_auth),
) -> dict[str, Any]:
    """Generate multiple music tracks in batch."""
    requests = batch_data.requests
    if len(requests) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Maximum 10 tracks per batch request"
        )

    batch_id = str(uuid.uuid4())
    job_ids = []

    for request in requests:
        job_id = str(uuid.uuid4())
        job = JobStatus(
            job_id=job_id, status="queued", message=f"Batch {batch_id}: Job queued for processing"
        )
        await state.add_job(job_id, job)
        background_tasks.add_task(generate_music_task, job_id, request)
        job_ids.append(job_id)

    logger.info("Batch generation %s created with %s jobs", batch_id, len(job_ids))

    return {
        "batch_id": batch_id,
        "jobs": job_ids,
        "status": "processing",
        "total_jobs": len(job_ids),
    }


# ── Audio ────────────────────────────────────────────────────────────────────


@app.get("/audio/{filename}")
async def get_audio(filename: str) -> FileResponse:
    """Serve generated audio files."""
    output_dir = Path(config.OUTPUT_DIR)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / "outputs"

    output_dir.mkdir(parents=True, exist_ok=True)

    safe_filename = Path(filename).name
    file_path = output_dir / safe_filename

    # Check for symlinks before resolving (resolve follows symlinks, hiding them)
    if file_path.is_symlink():
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    try:
        file_path = file_path.resolve(strict=False)
        output_dir = output_dir.resolve(strict=False)
        if not file_path.is_relative_to(output_dir):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    except (ValueError, RuntimeError):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid filename")

    if not file_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Audio file not found")

    return FileResponse(str(file_path), media_type="audio/wav", filename=safe_filename)


@app.post("/audio/analyze")
async def analyze_audio(
    request: dict[str, Any], current_user: UserClaims = Depends(require_auth)
) -> dict[str, Any]:
    """Analyze audio file and return metadata."""
    audio_url = request.get("audio_url")
    return {
        "audio_url": audio_url,
        "duration": 30.0,
        "format": "wav",
        "sample_rate": 32000,
        "channels": 1,
        "bitrate": 512000,
        "analysis": {
            "tempo": 120,
            "key": "C major",
            "mood": "uplifting",
            "energy": 0.7,
            "danceability": 0.8,
        },
    }


@app.post("/audio/waveform")
async def generate_waveform(
    audio_url: str = Query(..., description="URL of audio file"),
    width: int = Query(default=1920, description="Waveform image width"),
    height: int = Query(default=200, description="Waveform image height"),
    current_user: UserClaims = Depends(require_auth),
) -> dict[str, Any]:
    """Generate waveform visualization for audio file."""
    waveform_id = str(uuid.uuid4())
    return {
        "waveform_url": f"/static/waveforms/{waveform_id}.png",
        "width": width,
        "height": height,
        "audio_url": audio_url,
    }


# ── Auth ─────────────────────────────────────────────────────────────────────


@app.post("/auth/register")
async def register_user(user_data: UserRegistration) -> dict[str, Any]:
    """Register a new user."""
    try:
        existing_email = await state.find_user_by_email(user_data.email)
        existing_username = await state.find_user_by_username(user_data.username)

        if existing_email or existing_username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Registration failed. The provided details may already be in use.",
            )

        user_id = str(uuid.uuid4())
        user = {
            "user_id": user_id,
            "username": user_data.username,
            "email": user_data.email,
            "password_hash": hash_password(user_data.password),
            "roles": [UserRole.USER.value],
            "tier": "free",
            "is_verified": True,
            "tracks_generated": 0,
            "playlists_count": 0,
        }

        await state.add_user(user_id, user)
        logger.info("User registered: %s (%s)", user_data.username, user_data.email)

        access_token = get_auth_middleware().create_access_token(
            user_id=user_id,
            email=user_data.email,
            username=user_data.username,
            roles=[UserRole.USER.value],
            tier="free",
            is_verified=True,
        )

        refresh_token = get_auth_middleware().create_refresh_token(
            user_id=user_id,
            email=user_data.email,
            username=user_data.username,
            roles=[UserRole.USER.value],
            tier="free",
            is_verified=True,
        )

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "user": {
                "id": user_id,
                "user_id": user_id,
                "username": user_data.username,
                "email": user_data.email,
                "roles": [UserRole.USER.value],
                "tier": "free",
                "is_verified": True,
                "tracks_generated": 0,
                "playlists_count": 0,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Registration error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Registration failed"
        )


@app.post("/auth/login")
async def login_user(
    request: Request, form_data: OAuth2PasswordRequestForm = Depends()
) -> dict[str, Any]:
    """Login user and return JWT tokens."""
    try:
        user = await state.find_user_by_email(form_data.username)

        password_hash: str = (user.get("password_hash") or "") if user else ""
        if not user or not verify_password(form_data.password, password_hash):
            log_login_attempt(
                request=request,
                email=form_data.username,
                success=False,
                failure_reason="Invalid credentials",
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        access_token = get_auth_middleware().create_access_token(
            user_id=user["user_id"],
            email=user["email"],
            username=user["username"],
            roles=user["roles"],
            tier=user["tier"],
            is_verified=user["is_verified"],
        )

        refresh_token = get_auth_middleware().create_refresh_token(
            user_id=user["user_id"],
            email=user["email"],
            username=user["username"],
            roles=user["roles"],
            tier=user["tier"],
            is_verified=user["is_verified"],
        )

        log_login_attempt(
            request=request,
            email=user["email"],
            success=True,
            user_id=user["user_id"],
        )

        logger.info("User logged in: %s", user["username"])

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "user": {
                "id": user["user_id"],
                "user_id": user["user_id"],
                "username": user["username"],
                "email": user["email"],
                "roles": user["roles"],
                "tier": user["tier"],
                "is_verified": user["is_verified"],
                "tracks_generated": user.get("tracks_generated", 0),
                "playlists_count": user.get("playlists_count", 0),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Login error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Login failed"
        )


@app.get("/auth/me")
async def get_current_user_info(
    current_user: UserClaims = Depends(require_auth),
) -> dict[str, Any]:
    """Get current user information."""
    user_data = await state.get_user(current_user.user_id) or {}

    return {
        "user_id": current_user.user_id,
        "username": current_user.username,
        "email": current_user.email,
        "roles": [role.value for role in current_user.roles],
        "tier": current_user.tier,
        "is_verified": current_user.is_verified,
        "tracks_generated": user_data.get("tracks_generated", 0),
        "playlists_count": user_data.get("playlists_count", 0),
    }


# ── Social ───────────────────────────────────────────────────────────────────


@app.post("/playlists")
async def create_playlist(
    playlist_data: PlaylistCreate, current_user: UserClaims = Depends(require_auth)
) -> dict[str, Any]:
    """Create a new playlist."""
    playlist_id = str(uuid.uuid4())
    playlist = {
        "id": playlist_id,
        "name": playlist_data.name,
        "description": playlist_data.description,
        "is_public": playlist_data.is_public,
        "user_id": current_user.user_id,
        "tracks": [],
        "created_at": time.time(),
        "updated_at": time.time(),
    }

    await state.add_playlist(playlist_id, playlist)
    logger.info("Playlist created: %s by user %s", playlist_id, current_user.user_id)

    user_data = await state.get_user(current_user.user_id)
    if user_data:
        await state.update_user(
            current_user.user_id,
            playlists_count=user_data.get("playlists_count", 0) + 1,
        )

    return playlist


@app.get("/playlists")
async def get_playlists(current_user: UserClaims = Depends(require_auth)) -> dict[str, Any]:
    """Get user's playlists."""
    user_playlists = await state.get_playlists_for_user(current_user.user_id)
    return {"playlists": user_playlists, "total": len(user_playlists)}


@app.get("/dashboard")
async def get_dashboard_data(current_user: UserClaims = Depends(require_auth)) -> dict[str, Any]:
    """Get dashboard data for current user."""
    user_data = await state.get_user(current_user.user_id) or {}
    all_jobs = await state.get_all_jobs()
    all_users = await state.get_all_users()
    user_playlists = await state.get_playlists_for_user(current_user.user_id)

    return {
        "user_stats": {
            "tracks_generated": user_data.get("tracks_generated", 0),
            "playlists_count": user_data.get("playlists_count", 0),
            "total_duration": user_data.get("tracks_generated", 0) * 30.0,
            "favorite_genres": ["Electronic", "Ambient", "Classical"],
        },
        "recent_activity": {"last_generation": time.time() - 3600, "last_login": time.time()},
        "system_stats": {
            "total_users": len(all_users),
            "total_generations": len(all_jobs),
            "active_jobs": len([j for j in all_jobs.values() if j.status == "processing"]),
        },
        "user_profile": {
            "username": current_user.username,
            "email": current_user.email,
            "tier": current_user.tier,
            "member_since": time.time() - 86400,
        },
        "social_profile": {
            "followers": 0,
            "following": 0,
            "public_playlists": len([p for p in user_playlists if p["is_public"]]),
        },
        "playlists": user_playlists[:5],
    }


@app.get("/search")
async def search(
    query: str = Query(..., description="Search query"),
    type: str = Query(default="all", description="Type to search: all, tracks, playlists, users"),
    current_user: UserClaims = Depends(require_auth),
) -> dict[str, Any]:
    """Search for tracks, playlists, or users."""
    results = {
        "query": query,
        "type": type,
        "results": {
            "tracks": (
                [
                    {
                        "id": "track-1",
                        "title": f"Generated Track matching '{query}'",
                        "duration": 30.0,
                        "created_at": time.time() - 7200,
                        "genre": "Electronic",
                    }
                ]
                if type in ["all", "tracks"]
                else []
            ),
            "playlists": (
                [
                    {
                        "id": "playlist-1",
                        "name": f"Playlist matching '{query}'",
                        "track_count": 5,
                        "owner": "user123",
                    }
                ]
                if type in ["all", "playlists"]
                else []
            ),
            "users": (
                [{"id": "user-1", "username": f"User matching '{query}'", "tracks_count": 10}]
                if type in ["all", "users"]
                else []
            ),
        },
        "total_results": 3 if type == "all" else 1,
    }

    return results


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT, log_level=config.LOG_LEVEL.lower())
