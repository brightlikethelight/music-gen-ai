"""
Pydantic request/response models for MusicGen API.
"""

from typing import Optional

from pydantic import BaseModel, Field


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


class UserRegistration(BaseModel):
    """User registration request."""

    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r"^[^@]+@[^@]+\.[^@]+$")
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None


class PlaylistCreate(BaseModel):
    """Playlist creation request."""

    name: str = Field(..., description="Playlist name")
    description: str = Field(default="", description="Playlist description")
    is_public: bool = Field(default=True, description="Whether playlist is public")


class BatchGenerationRequest(BaseModel):
    """Batch generation request model."""

    requests: list[GenerationRequest]
    sequential: bool = False
