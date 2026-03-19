"""Custom exceptions for MusicGen Unified.

Provides clear, actionable error messages.
"""

from typing import Any, Dict, Optional


class MusicGenError(Exception):
    """Base exception for all MusicGen errors."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None,
    ):
        super().__init__(message)
        self.details = details or {}
        self.error_code = error_code or self.__class__.__name__


class ModelError(MusicGenError):
    """Errors related to model loading or inference."""

    pass


class GenerationError(MusicGenError):
    """Errors during music generation."""

    pass


class PromptError(MusicGenError):
    """Errors related to prompt validation or processing."""

    pass


class AudioError(MusicGenError):
    """Errors related to audio processing or saving."""

    pass


class AudioProcessingError(AudioError):
    """Errors during audio processing operations."""

    pass


class ConfigError(MusicGenError):
    """Configuration-related errors."""

    pass


class ConfigurationError(ConfigError):
    """Configuration-related errors (alias for backward compatibility)."""

    pass


class ResourceError(MusicGenError):
    """Resource-related errors (memory, disk, etc)."""

    pass


class ValidationError(MusicGenError):
    """Errors related to input validation."""

    pass


class DataLoadingError(MusicGenError):
    """Errors related to data loading operations."""

    pass


class APIError(MusicGenError):
    """Errors related to API operations."""

    pass


class AuthenticationError(MusicGenError):
    """Errors related to authentication."""

    pass


class AuthorizationError(MusicGenError):
    """Errors related to authorization and permissions."""

    pass


# Specific error cases with helpful messages
class PromptTooLongError(PromptError):
    """Prompt exceeds maximum length."""

    def __init__(self, length: int, max_length: int):
        super().__init__(
            f"Prompt length ({length} chars) exceeds maximum ({max_length} chars). "
            f"Please shorten your prompt or split into multiple generations.",
            details={"length": length, "max_length": max_length},
        )


class DurationError(GenerationError):
    """Invalid duration specified."""

    def __init__(self, duration: float, max_duration: float):
        super().__init__(
            f"Duration {duration}s exceeds maximum {max_duration}s. "
            f"Use extended generation for longer pieces or reduce duration.",
            details={"duration": duration, "max_duration": max_duration},
        )


class OutOfMemoryError(ResourceError):
    """Not enough memory for generation."""

    def __init__(self, required_gb: float, available_gb: float):
        super().__init__(
            f"Insufficient memory: {required_gb:.1f}GB required, {available_gb:.1f}GB available. "
            f"Try: 1) Use smaller model, 2) Reduce duration, 3) Close other applications.",
            details={"required_gb": required_gb, "available_gb": available_gb},
        )


class ModelNotFoundError(ModelError):
    """Model files not found."""

    def __init__(self, model_name: str):
        super().__init__(
            f"Model '{model_name}' not found. "
            f"It will be downloaded on first use (requires internet connection). "
            "Available models: facebook/musicgen-small, "
            "facebook/musicgen-medium, facebook/musicgen-large",
            details={"model_name": model_name},
        )


class MP3ConversionError(AudioError):
    """MP3 conversion failed."""

    def __init__(self, reason: str):
        super().__init__(
            f"MP3 conversion failed: {reason}. "
            f"Audio saved as WAV instead. "
            f"To enable MP3: 1) Install ffmpeg, 2) pip install pydub",
            details={"reason": reason},
        )


class VocalRequestError(PromptError):
    """User requested vocals which aren't supported."""

    def __init__(self) -> None:
        super().__init__(
            "MusicGen doesn't support vocals or singing - it generates instrumental music only. "
            "Please remove references to vocals, singing, or lyrics from your prompt."
        )
