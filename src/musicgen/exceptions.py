"""
Custom exceptions for MusicGen Unified.

Provides clear, actionable error messages.
"""


class MusicGenError(Exception):
    """Base exception for all MusicGen errors."""
    pass


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


class ConfigError(MusicGenError):
    """Configuration-related errors."""
    pass


class ResourceError(MusicGenError):
    """Resource-related errors (memory, disk, etc)."""
    pass


# Specific error cases with helpful messages
class PromptTooLongError(PromptError):
    """Prompt exceeds maximum length."""
    def __init__(self, length: int, max_length: int):
        super().__init__(
            f"Prompt length ({length} chars) exceeds maximum ({max_length} chars). "
            f"Please shorten your prompt or split into multiple generations."
        )


class DurationError(GenerationError):
    """Invalid duration specified."""
    def __init__(self, duration: float, max_duration: float):
        super().__init__(
            f"Duration {duration}s exceeds maximum {max_duration}s. "
            f"Use extended generation for longer pieces or reduce duration."
        )


class OutOfMemoryError(ResourceError):
    """Not enough memory for generation."""
    def __init__(self, required_gb: float, available_gb: float):
        super().__init__(
            f"Insufficient memory: {required_gb:.1f}GB required, {available_gb:.1f}GB available. "
            f"Try: 1) Use smaller model, 2) Reduce duration, 3) Close other applications."
        )


class ModelNotFoundError(ModelError):
    """Model files not found."""
    def __init__(self, model_name: str):
        super().__init__(
            f"Model '{model_name}' not found. "
            f"It will be downloaded on first use (requires internet connection). "
            f"Available models: facebook/musicgen-small, facebook/musicgen-medium, facebook/musicgen-large"
        )


class MP3ConversionError(AudioError):
    """MP3 conversion failed."""
    def __init__(self, reason: str):
        super().__init__(
            f"MP3 conversion failed: {reason}. "
            f"Audio saved as WAV instead. "
            f"To enable MP3: 1) Install ffmpeg, 2) pip install pydub"
        )


class VocalRequestError(PromptError):
    """User requested vocals which aren't supported."""
    def __init__(self):
        super().__init__(
            "MusicGen doesn't support vocals or singing - it generates instrumental music only. "
            "Please remove references to vocals, singing, or lyrics from your prompt."
        )


def handle_gpu_error(e: Exception) -> None:
    """Convert GPU errors to helpful messages."""
    error_str = str(e).lower()
    
    if "out of memory" in error_str or "oom" in error_str:
        import torch
        if torch.cuda.is_available():
            free_gb = torch.cuda.mem_get_info()[0] / 1e9
            raise OutOfMemoryError(4.0, free_gb) from e
        else:
            raise ResourceError("GPU out of memory. Try using CPU mode.") from e
    
    elif "cuda" in error_str and "not available" in error_str:
        raise ModelError(
            "CUDA not available. Using CPU mode (will be slower). "
            "For GPU: 1) Install CUDA toolkit, 2) Install PyTorch with CUDA support."
        ) from e
    
    else:
        raise ModelError(f"GPU error: {e}") from e