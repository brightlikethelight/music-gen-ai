"""
Test utility functions and classes for maximum coverage with minimal dependencies.
"""

import os
import pytest
import tempfile
import logging
from pathlib import Path

# Set environment variables before imports
os.environ["JWT_SECRET_KEY"] = "test-key"
os.environ["MUSICGEN_SKIP_AUTH"] = "1"
os.environ["MUSICGEN_SKIP_MODEL_DOWNLOAD"] = "1"
os.environ["PYTEST_CURRENT_TEST"] = "1"


def test_estimate_memory_usage():
    """Test memory estimation function."""
    from musicgen.utils.helpers import estimate_memory_usage
    
    # Test small model
    estimate = estimate_memory_usage(10.0, "small")
    assert isinstance(estimate, dict)
    assert "model_memory_gb" in estimate
    assert "generation_memory_gb" in estimate
    assert "total_memory_gb" in estimate
    assert estimate["total_memory_gb"] > 0
    
    # Test medium model
    estimate = estimate_memory_usage(20.0, "medium")
    assert estimate["model_memory_gb"] > 0.5
    
    # Test large model 
    estimate = estimate_memory_usage(30.0, "large")
    assert estimate["model_memory_gb"] > 1.5


def test_progress_tracker():
    """Test ProgressTracker class."""
    from musicgen.utils.helpers import ProgressTracker
    
    tracker = ProgressTracker(10, "Testing")
    assert tracker.total == 10
    assert tracker.current == 0
    assert tracker.description == "Testing"
    
    # Test update
    tracker.update(3)
    assert tracker.current == 3
    
    # Test progress info
    progress = tracker.get_progress()
    assert isinstance(progress, dict)
    assert progress["current"] == 3
    assert progress["total"] == 10
    assert progress["percent"] == 30.0
    assert "elapsed" in progress
    assert "remaining" in progress


def test_setup_logging():
    """Test logging setup function."""
    from musicgen.utils.helpers import setup_logging
    
    # Test basic setup
    setup_logging("DEBUG")
    logger = logging.getLogger("test")
    assert logger.isEnabledFor(logging.DEBUG)
    
    # Test with custom format
    setup_logging("INFO", "%(levelname)s: %(message)s")
    assert logger.isEnabledFor(logging.INFO)


def test_error_recovery():
    """Test ErrorRecovery context manager."""
    from musicgen.utils.exceptions import ErrorRecovery
    
    cleanup_called = False
    
    def cleanup_func():
        nonlocal cleanup_called
        cleanup_called = True
    
    # Test normal execution (no exception)
    with ErrorRecovery(log_errors=False) as recovery:
        recovery.add_cleanup(cleanup_func)
        pass
    
    # cleanup should not be called if no exception
    assert cleanup_called == False
    
    # Test with exception
    cleanup_called = False
    try:
        with ErrorRecovery(log_errors=False) as recovery:
            recovery.add_cleanup(cleanup_func)
            raise ValueError("test error")
    except ValueError:
        pass
    
    # cleanup should be called on exception
    assert cleanup_called == True


def test_decorators():
    """Test exception handling decorators."""
    from musicgen.utils.exceptions import handle_exceptions, retry_on_error, validate_input
    
    # Test handle_exceptions
    @handle_exceptions(ValueError, reraise=False, default_return="handled")
    def func_with_error():
        raise ValueError("test")
    
    result = func_with_error()
    assert result == "handled"
    
    # Test validate_input
    def is_positive(x):
        return x > 0
    
    @validate_input(is_positive, "Must be positive")
    def square(x):
        return x * x
    
    assert square(5) == 25
    
    with pytest.raises(Exception):  # ValidationError
        square(-1)


def test_device_selection():
    """Test device selection helper."""
    try:
        from musicgen.utils.helpers import get_device
        import torch
        
        # Test auto device selection
        device = get_device()
        assert isinstance(device, torch.device)
        
        # Test explicit device
        cpu_device = get_device("cpu")
        assert cpu_device.type == "cpu"
        
    except ImportError:
        pytest.skip("PyTorch not available for device testing")


def test_exception_error_codes():
    """Test exception error codes and details."""
    from musicgen.utils.exceptions import (
        MusicGenError, ModelError, GenerationError,
        PromptError, AudioError, ConfigError
    )
    
    # Test base error with custom error code
    error = MusicGenError("test", error_code="CUSTOM_ERROR")
    assert error.error_code == "CUSTOM_ERROR"
    
    # Test inherited errors have correct codes
    model_error = ModelError("model issue")
    assert model_error.error_code == "ModelError"
    
    gen_error = GenerationError("gen issue")
    assert gen_error.error_code == "GenerationError"
    
    prompt_error = PromptError("prompt issue")
    assert prompt_error.error_code == "PromptError"
    
    audio_error = AudioError("audio issue") 
    assert audio_error.error_code == "AudioError"
    
    config_error = ConfigError("config issue")
    assert config_error.error_code == "ConfigError"


def test_with_error_context():
    """Test with_error_context decorator."""
    from musicgen.utils.exceptions import with_error_context, MusicGenError
    
    context = {"operation": "test", "version": "1.0"}
    
    @with_error_context(context)
    def failing_function():
        raise ValueError("original error")
    
    try:
        failing_function()
        assert False, "Should have raised an exception"
    except MusicGenError as e:
        assert e.details["operation"] == "test"
        assert e.details["version"] == "1.0"
        assert e.details["original_exception"] == "original error"
        assert e.details["original_type"] == "ValueError"


def test_get_cache_dir_creation():
    """Test cache directory creation."""
    from musicgen.utils.helpers import get_cache_dir
    
    cache_dir = get_cache_dir()
    assert cache_dir.exists()
    assert cache_dir.is_dir()
    assert "musicgen-unified" in str(cache_dir)


def test_audio_processing_helpers():
    """Test audio processing helper functions."""
    try:
        import numpy as np
        from musicgen.utils.helpers import apply_fade, crossfade_audio
        
        # Create test audio data
        sample_rate = 44100
        audio1 = np.random.randn(sample_rate)  # 1 second
        audio2 = np.random.randn(sample_rate)  # 1 second
        
        # Test apply_fade
        faded = apply_fade(audio1, sample_rate, fade_in=0.1, fade_out=0.1)
        assert faded.shape == audio1.shape
        assert not np.array_equal(faded, audio1)  # Should be modified
        
        # Test crossfade
        crossfaded = crossfade_audio(audio1, audio2, 0.1, sample_rate)
        assert len(crossfaded) > len(audio1)
        assert len(crossfaded) > len(audio2)
        
    except ImportError:
        pytest.skip("NumPy not available for audio processing tests")


def test_handle_gpu_error():
    """Test GPU error handling."""
    from musicgen.utils.exceptions import handle_gpu_error, OutOfMemoryError, ResourceError, ModelError
    
    # Test OOM error
    try:
        oom_error = RuntimeError("CUDA out of memory")
        handle_gpu_error(oom_error)
        assert False, "Should have raised an exception"
    except (OutOfMemoryError, ResourceError):
        pass
    
    # Test CUDA not available error
    try:
        cuda_error = RuntimeError("CUDA not available")
        handle_gpu_error(cuda_error)
        assert False, "Should have raised an exception"  
    except ModelError:
        pass
    
    # Test generic GPU error
    try:
        gpu_error = RuntimeError("Some other GPU error")
        handle_gpu_error(gpu_error)
        assert False, "Should have raised an exception"
    except ModelError:
        pass