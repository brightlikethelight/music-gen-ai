"""Custom exceptions for MusicGen Unified.

Provides clear, actionable error messages with professional error handling utilities.
"""

import functools
import logging
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Type, Union


class MusicGenError(Exception):
    """Base exception for all MusicGen errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, error_code: Optional[str] = None):
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


# Specific error cases with helpful messages
class PromptTooLongError(PromptError):
    """Prompt exceeds maximum length."""

    def __init__(self, length: int, max_length: int):
        super().__init__(
            f"Prompt length ({length} chars) exceeds maximum ({max_length} chars). "
            f"Please shorten your prompt or split into multiple generations.",
            details={"length": length, "max_length": max_length}
        )


class DurationError(GenerationError):
    """Invalid duration specified."""

    def __init__(self, duration: float, max_duration: float):
        super().__init__(
            f"Duration {duration}s exceeds maximum {max_duration}s. "
            f"Use extended generation for longer pieces or reduce duration.",
            details={"duration": duration, "max_duration": max_duration}
        )


class OutOfMemoryError(ResourceError):
    """Not enough memory for generation."""

    def __init__(self, required_gb: float, available_gb: float):
        super().__init__(
            f"Insufficient memory: {required_gb:.1f}GB required, {available_gb:.1f}GB available. "
            f"Try: 1) Use smaller model, 2) Reduce duration, 3) Close other applications.",
            details={"required_gb": required_gb, "available_gb": available_gb}
        )


class ModelNotFoundError(ModelError):
    """Model files not found."""

    def __init__(self, model_name: str):
        super().__init__(
            f"Model '{model_name}' not found. "
            f"It will be downloaded on first use (requires internet connection). "
            f"Available models: facebook/musicgen-small, facebook/musicgen-medium, facebook/musicgen-large",
            details={"model_name": model_name}
        )


class MP3ConversionError(AudioError):
    """MP3 conversion failed."""

    def __init__(self, reason: str):
        super().__init__(
            f"MP3 conversion failed: {reason}. "
            f"Audio saved as WAV instead. "
            f"To enable MP3: 1) Install ffmpeg, 2) pip install pydub",
            details={"reason": reason}
        )


class VocalRequestError(PromptError):
    """User requested vocals which aren't supported."""

    def __init__(self):
        super().__init__(
            "MusicGen doesn't support vocals or singing - it generates instrumental music only. "
            "Please remove references to vocals, singing, or lyrics from your prompt."
        )


# Exception handling decorators and utilities

def handle_exceptions(*exception_types: Type[Exception], reraise: bool = True, default_return: Any = None) -> Callable:
    """Decorator to handle specific exception types.
    
    Args:
        exception_types: Exception types to catch
        reraise: Whether to reraise the exception after handling
        default_return: Value to return if exception is caught and not reraised
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception_types as e:
                logging.error(f"Exception in {func.__name__}: {e}")
                if reraise:
                    raise
                return default_return
        return wrapper
    return decorator


def retry_on_error(
    max_attempts: int = 3, 
    delay: float = 1.0, 
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """Decorator to retry function on specific exceptions.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts in seconds
        backoff_factor: Factor to multiply delay by after each attempt
        exceptions: Exception types to retry on
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:  # Last attempt
                        raise
                    logging.warning(f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}")
                    if attempt < max_attempts - 1:  # Don't sleep after last attempt
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
        return wrapper
    return decorator


def validate_input(validator: Callable, error_message: str) -> Callable:
    """Decorator to validate function input.
    
    Args:
        validator: Function that returns True if input is valid
        error_message: Error message to raise if validation fails
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # For simplicity, validate the first argument
            if args and not validator(args[0]):
                raise ValidationError(error_message)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def with_error_context(context: Dict[str, Any]) -> Callable:
    """Decorator to add context to exceptions.
    
    Args:
        context: Dictionary of context information to add to exceptions
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except MusicGenError as e:
                # Add context to existing MusicGen error
                e.details.update(context)
                raise
            except Exception as e:
                # Convert generic exception to MusicGenError with context
                error_details = context.copy()
                error_details.update({
                    "original_exception": str(e),
                    "original_type": type(e).__name__
                })
                raise MusicGenError(
                    f"Unexpected error in {func.__name__}: {e}",
                    details=error_details
                ) from e
        return wrapper
    return decorator


def safe_execute(func: Callable, *args, default_return: Any = None, log_errors: bool = True, **kwargs) -> Any:
    """Safely execute a function, returning default value on error.
    
    Args:
        func: Function to execute
        *args: Arguments to pass to function
        default_return: Value to return on error
        log_errors: Whether to log errors
        **kwargs: Keyword arguments to pass to function
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logging.error(f"Error in safe_execute for {func.__name__}: {e}")
        return default_return


class ErrorRecovery:
    """Context manager for error recovery with cleanup operations."""
    
    def __init__(self, log_errors: bool = True):
        self.log_errors = log_errors
        self.cleanup_functions: List[Callable] = []
    
    def add_cleanup(self, cleanup_func: Callable) -> None:
        """Add a cleanup function to be called on error."""
        self.cleanup_functions.append(cleanup_func)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:  # An exception occurred
            if self.log_errors:
                logging.error(f"Error in ErrorRecovery context: {exc_val}")
            
            # Run cleanup functions in reverse order
            for cleanup_func in reversed(self.cleanup_functions):
                try:
                    cleanup_func()
                except Exception as cleanup_error:
                    if self.log_errors:
                        logging.error(f"Error in cleanup function: {cleanup_error}")
        
        # Don't suppress the original exception
        return False


def format_error_for_user(error: Exception, include_details: bool = False) -> str:
    """Format an error for user-friendly display.
    
    Args:
        error: The exception to format
        include_details: Whether to include detailed information
    """
    if isinstance(error, MusicGenError):
        message = str(error)
        if include_details and error.details:
            details_str = ", ".join(f"{k}: {v}" for k, v in error.details.items())
            message += f" (Details: {details_str})"
        return message
    else:
        return f"An unexpected error occurred: {error}"


def get_error_summary(error: Exception) -> Dict[str, Any]:
    """Get a comprehensive summary of an error.
    
    Args:
        error: The exception to summarize
    """
    summary = {
        "error_type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc()
    }
    
    if isinstance(error, MusicGenError):
        summary["details"] = error.details
        summary["error_code"] = error.error_code
    else:
        summary["details"] = {}
    
    return summary


def handle_gpu_error(e: Exception) -> None:
    """Convert GPU errors to helpful messages."""
    error_str = str(e).lower()

    if "out of memory" in error_str or "oom" in error_str:
        try:
            import torch
            if torch.cuda.is_available():
                free_gb = torch.cuda.mem_get_info()[0] / 1e9
                raise OutOfMemoryError(4.0, free_gb) from e
            else:
                raise ResourceError("GPU out of memory. Try using CPU mode.") from e
        except ImportError:
            raise ResourceError("GPU out of memory. Try using CPU mode.") from e

    elif "cuda" in error_str and "not available" in error_str:
        raise ModelError(
            "CUDA not available. Using CPU mode (will be slower). "
            "For GPU: 1) Install CUDA toolkit, 2) Install PyTorch with CUDA support."
        ) from e

    else:
        raise ModelError(f"GPU error: {e}") from e