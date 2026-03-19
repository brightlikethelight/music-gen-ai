"""
Pydantic request/response models for MusicGen API.
"""

from typing import Any

from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    """Request model for music generation."""

    prompt: str = Field(
        ..., max_length=500, description="Text description of the music to generate"
    )
    duration: float = Field(default=30.0, ge=1.0, le=600.0, description="Duration in seconds")
    model: str = Field(
        default="facebook/musicgen-small",
        pattern=r"^facebook/musicgen-(small|medium|large)$",
        description="Model to use (facebook/musicgen-small, -medium, or -large)",
    )
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
    audio_url: str | None = None


class UserRegistration(BaseModel):
    """User registration request."""

    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r"^[^@]+@[^@]+\.[^@]+$")
    password: str = Field(..., min_length=8)


class PlaylistCreate(BaseModel):
    """Playlist creation request."""

    name: str = Field(..., description="Playlist name")
    description: str = Field(default="", description="Playlist description")
    is_public: bool = Field(default=True, description="Whether playlist is public")


class BatchGenerationRequest(BaseModel):
    """Batch generation request model."""

    requests: list[GenerationRequest] = Field(..., max_length=10)


# ── Response models ──────────────────────────────────────────────────────────


class UserPublic(BaseModel):
    """User profile in auth responses."""

    id: str
    user_id: str
    username: str
    email: str
    roles: list[str]
    tier: str
    is_verified: bool
    tracks_generated: int = 0
    playlists_count: int = 0


class AuthTokenResponse(BaseModel):
    """Response for /auth/register and /auth/login."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: UserPublic


class RefreshTokenRequest(BaseModel):
    """Request for POST /auth/refresh."""

    refresh_token: str = Field(..., description="Refresh token to exchange for new tokens")


class RefreshTokenResponse(BaseModel):
    """Response for POST /auth/refresh."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class LogoutResponse(BaseModel):
    """Response for POST /auth/logout."""

    message: str
    success: bool


class UserProfileResponse(BaseModel):
    """Response for /auth/me."""

    user_id: str
    username: str
    email: str
    roles: list[str]
    tier: str
    is_verified: bool
    tracks_generated: int = 0
    playlists_count: int = 0


class PlaylistResponse(BaseModel):
    """Single playlist object."""

    id: str
    name: str
    description: str
    is_public: bool
    user_id: str
    tracks: list[Any]
    created_at: float
    updated_at: float


class PlaylistListResponse(BaseModel):
    """Response for GET /playlists."""

    playlists: list[PlaylistResponse]
    total: int


class BatchGenerationResponse(BaseModel):
    """Response for POST /generate/batch."""

    batch_id: str
    jobs: list[str]
    status: str
    total_jobs: int
