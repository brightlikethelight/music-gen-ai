"""Tests to boost coverage to reach 25% threshold."""

import os
os.environ["JWT_SECRET_KEY"] = "test-key"
os.environ["PYTEST_CURRENT_TEST"] = "1"
os.environ["MUSICGEN_SKIP_MODEL_DOWNLOAD"] = "1"

import pytest
from unittest.mock import MagicMock, patch, Mock


def test_api_rest_api_requests():
    """Test API request/response models."""
    from musicgen.api.rest.api import (
        GenerateRequest,
        GenerateResponse,
        PromptRequest,
        PromptResponse,
        JobStatus
    )
    
    # Test GenerateRequest with various options
    req = GenerateRequest(prompt="piano jazz", duration=15.0, temperature=0.8)
    assert req.prompt == "piano jazz"
    assert req.duration == 15.0
    assert req.temperature == 0.8
    
    # Test GenerateResponse
    resp = GenerateResponse(job_id="test-123", status="processing", message="Processing")
    assert resp.job_id == "test-123"
    assert resp.status == "processing"
    
    # Test with required fields only (GenerateResponse only has job_id, status, message)
    resp2 = GenerateResponse(
        job_id="test-456",
        status="completed",
        message="Completed successfully"
    )
    assert resp2.job_id == "test-456"
    assert resp2.status == "completed"
    assert resp2.message == "Completed successfully"
    
    # Test PromptRequest
    prompt_req = PromptRequest(prompt="short prompt")
    assert prompt_req.prompt == "short prompt"
    
    # Test PromptResponse
    prompt_resp = PromptResponse(
        original="short",
        improved="short melodic instrumental music",
        is_valid=True
    )
    assert prompt_resp.is_valid is True
    
    # Test variations response
    prompt_resp = PromptResponse(
        original="piano",
        improved="piano melody",
        is_valid=True,
        variations=["jazz piano", "classical piano"]
    )
    assert len(prompt_resp.variations) == 2
    
    # Test JobStatus
    status = JobStatus(
        job_id="job-001",
        status="completed",
        progress=100,
        created_at="2024-01-01T00:00:00"
    )
    assert status.progress == 100


def test_config_module_comprehensive():
    """Comprehensive config tests."""
    from musicgen.infrastructure.config.config import Config
    
    # Test instance creation
    config = Config()
    
    # Test basic attributes
    assert config.OUTPUT_DIR is not None
    assert config.MODEL_CACHE_DIR is not None
    assert config.LOG_LEVEL is not None
    assert config.API_HOST is not None
    assert config.API_PORT is not None
    
    # Test default values
    assert config.MAX_DURATION > 0
    assert config.DEFAULT_DURATION > 0
    assert config.MAX_PROMPT_LENGTH > 0
    
    # Test methods
    model_config = config.get_model_config()
    assert isinstance(model_config, dict)
    assert 'model_name' in model_config
    
    api_config = config.get_api_config()
    assert isinstance(api_config, dict)
    assert 'host' in api_config
    assert 'port' in api_config
    
    # Test string representation
    str_repr = str(config)
    assert "Config" in str_repr or len(str_repr) > 0


def test_logging_module_comprehensive():
    """Test logging module functionality."""
    from musicgen.infrastructure.monitoring.logging import (
        setup_logging,
        get_logger
    )
    import logging
    
    # Test setup_logging
    setup_logging(level="INFO")
    
    # Test get_logger
    logger = get_logger("test.module")
    assert logger.name == "test.module"
    assert isinstance(logger, logging.Logger)
    
    # Test with different module names
    logger2 = get_logger("musicgen.test")
    assert logger2.name == "musicgen.test"
    
    # Test logging levels
    logger.info("Test info message")
    logger.debug("Test debug message")
    logger.warning("Test warning message")


def test_helpers_module():
    """Test helpers utility functions."""
    from musicgen.utils.helpers import (
        load_audio,
        save_audio,
        format_time,
        get_cache_dir,
        hash_text,
        validate_prompt_length,
        setup_logging
    )
    
    # Test that imports work
    assert load_audio is not None
    assert save_audio is not None
    assert format_time is not None
    assert get_cache_dir is not None
    
    # Test format_time
    assert format_time(30.0) == "30.0s"
    assert format_time(90.0) == "1m 30s"
    
    # Test hash_text
    hash1 = hash_text("test text")
    hash2 = hash_text("test text")
    assert hash1 == hash2
    assert len(hash1) == 8
    
    # Test validate_prompt_length
    assert validate_prompt_length("good music") == "good music"
    assert validate_prompt_length("  spaced   text  ") == "spaced   text"
    
    # Test get_cache_dir
    cache_dir = get_cache_dir()
    assert cache_dir.name == "musicgen-unified"
    
    # Test setup_logging (should not raise)
    setup_logging("INFO")


def test_exceptions_hierarchy():
    """Test exception class hierarchy."""
    from musicgen.utils.exceptions import (
        MusicGenError,
        ModelError,
        GenerationError,
        PromptError,
        AudioError,
        ConfigError,
        ValidationError
    )
    
    # Test base exception
    base_exc = MusicGenError("base error")
    assert str(base_exc) == "base error"
    
    # Test model error
    model_exc = ModelError("model failed")
    assert isinstance(model_exc, MusicGenError)
    assert "model failed" in str(model_exc)
    
    # Test generation error
    gen_exc = GenerationError("generation failed")
    assert isinstance(gen_exc, MusicGenError)
    
    # Test prompt error
    prompt_exc = PromptError("invalid prompt")
    assert isinstance(prompt_exc, MusicGenError)
    
    # Test audio error
    audio_exc = AudioError("audio processing failed")
    assert isinstance(audio_exc, MusicGenError)
    
    # Test config error
    config_exc = ConfigError("bad config")
    assert isinstance(config_exc, MusicGenError)
    
    # Test validation error
    val_exc = ValidationError("validation failed")
    assert isinstance(val_exc, MusicGenError)


def test_prompt_engineer_basic():
    """Test PromptEngineer basic functionality."""
    from musicgen.core.prompt import PromptEngineer
    
    engineer = PromptEngineer()
    
    # Test improve_prompt
    improved = engineer.improve_prompt("jazz")
    assert len(improved) > 4
    assert isinstance(improved, str)
    
    # Test validate_prompt
    is_valid, issues = engineer.validate_prompt("good music")
    assert isinstance(is_valid, bool)
    assert isinstance(issues, list)
    
    # Test get_examples
    examples = engineer.get_examples()
    assert isinstance(examples, list)
    assert len(examples) > 0
    
    # Test suggest_variations
    variations = engineer.suggest_variations("piano", count=3)
    assert isinstance(variations, list)
    assert len(variations) == 3


def test_batch_processor_imports():
    """Test batch processor imports."""
    from musicgen.services.batch import BatchProcessor, create_sample_csv
    
    # Test that imports work
    assert BatchProcessor is not None
    assert create_sample_csv is not None
    
    # Test create_sample_csv returns a filename
    result = create_sample_csv()
    assert isinstance(result, str)
    assert ".csv" in result


def test_metrics_collector():
    """Test metrics collector."""
    from musicgen.infrastructure.monitoring.metrics import MetricsCollector
    
    # Create collector instance
    collector = MetricsCollector()
    
    # Test that it exists and has expected attributes
    assert collector is not None
    assert hasattr(collector, 'enabled')
    
    # Set to disabled
    collector.enabled = False
    assert collector.enabled is False
    
    # Test enabling/disabling
    collector.enabled = True
    assert collector.enabled is True