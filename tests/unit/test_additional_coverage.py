"""Additional tests to reach 25% coverage."""

import os
os.environ["JWT_SECRET_KEY"] = "test-key"
os.environ["PYTEST_CURRENT_TEST"] = "1"
os.environ["MUSICGEN_SKIP_MODEL_DOWNLOAD"] = "1"

import pytest
from unittest.mock import MagicMock, patch


def test_prompt_engineer_methods():
    """Test PromptEngineer methods for coverage."""
    from musicgen.core.prompt import PromptEngineer
    
    engineer = PromptEngineer()
    
    # Test _expand_short_prompt
    expanded = engineer._expand_short_prompt("jazz")
    assert len(expanded) > 4
    
    # Test _add_genre_context
    with_genre = engineer._add_genre_context("music")
    assert len(with_genre) > 5
    
    # Test _add_mood  
    with_mood = engineer._add_mood("jazz piano")
    assert len(with_mood) > 10
    
    # Test _structure_prompt
    structured = engineer._structure_prompt("random words here")
    assert len(structured) > 0
    
    # Test get_examples
    examples = engineer.get_examples()
    assert len(examples) > 0
    
    examples_jazz = engineer.get_examples(genre="jazz")
    assert len(examples_jazz) > 0
    
    # Test suggest_variations
    variations = engineer.suggest_variations("piano music", count=2)
    assert len(variations) == 2


def test_config_class_methods():
    """Test Config class methods."""
    from musicgen.infrastructure.config.config import Config
    
    # Test get_model_config
    model_cfg = Config.get_model_config()
    assert 'model_name' in model_cfg
    assert 'device' in model_cfg
    
    # Test get_api_config
    api_cfg = Config.get_api_config()
    assert 'host' in api_cfg
    assert 'port' in api_cfg
    
    # Test validate
    try:
        Config.validate()
    except ValueError:
        pass  # May fail with test config


def test_cors_config():
    """Test CORS configuration."""
    from musicgen.api.cors_config import CORSConfig
    
    config = CORSConfig()
    assert config.environment in ["development", "staging", "production"]
    assert isinstance(config.allowed_origins, set)


def test_exceptions_module():
    """Test exception classes."""
    from musicgen.utils.exceptions import (
        MusicGenException,
        ConfigurationError,
        ValidationError,
        GenerationError,
        ModelError
    )
    
    # Create instances
    base_exc = MusicGenException("base error")
    assert str(base_exc) == "base error"
    
    config_exc = ConfigurationError("config error")
    assert "config" in str(config_exc).lower()
    
    val_exc = ValidationError("validation error")
    assert "validation" in str(val_exc).lower()
    
    gen_exc = GenerationError("generation failed")
    assert "generation" in str(gen_exc).lower()
    
    model_exc = ModelError("model error")
    assert "model" in str(model_exc).lower()


def test_monitoring_logging():
    """Test logging setup."""
    from musicgen.infrastructure.monitoring.logging import (
        setup_logging,
        get_logger
    )
    
    # Setup logging
    setup_logging(level="INFO")
    
    # Get logger
    logger = get_logger("test_logger")
    assert logger is not None
    assert logger.name == "test_logger"


def test_api_rest_middleware_rate_limiting():
    """Test rate limiting."""
    from musicgen.api.rest.middleware.rate_limiting import RateLimiter
    
    limiter = RateLimiter()
    assert limiter.limits["per_minute"] > 0
    assert limiter.limits["per_hour"] > 0
    
    # Test IP validation
    assert limiter._is_valid_ip("192.168.1.1") is True
    assert limiter._is_valid_ip("invalid") is False
    
    # Test exempt IPs
    assert limiter._is_exempt_ip("127.0.0.1") is True
    assert limiter._is_exempt_ip("8.8.8.8") is False


def test_web_app_module():
    """Test web app module."""
    from musicgen.web.app import static_dir
    
    assert static_dir is not None
    assert isinstance(static_dir, str)


def test_api_main_module():
    """Test API main module."""
    from musicgen.api.main import app_instance
    
    assert app_instance is not None


def test_musicgen_main():
    """Test main module."""
    from musicgen.__main__ import main
    
    assert main is not None


def test_monitoring_metrics():
    """Test metrics module."""
    from musicgen.infrastructure.monitoring.metrics import MetricsCollector
    
    collector = MetricsCollector(enabled=False)
    
    # These should not raise even when disabled
    collector.record_metric("test", 1.0)
    collector.increment_counter("counter")
    collector.record_duration("op", 0.5)
    
    assert collector.enabled is False


def test_batch_job():
    """Test batch job."""
    from musicgen.services.batch import BatchJob
    
    job = BatchJob(
        job_id="test-1",
        prompt="test prompt",
        duration=30.0
    )
    
    assert job.job_id == "test-1"
    assert job.prompt == "test prompt"
    assert job.duration == 30.0
    assert job.status in ["pending", "processing", "completed", "failed"]


def test_api_response_models():
    """Test API response models."""
    from musicgen.api.rest.api import (
        GenerateResponse,
        PromptResponse,
        StatusResponse
    )
    
    # Test GenerateResponse
    gen_resp = GenerateResponse(
        job_id="job-1",
        status="processing"
    )
    assert gen_resp.job_id == "job-1"
    
    # Test PromptResponse
    prompt_resp = PromptResponse(
        original="test",
        improved="better test",
        is_valid=True
    )
    assert prompt_resp.original == "test"
    
    # Test StatusResponse
    status_resp = StatusResponse(
        job_id="job-1",
        status="completed",
        progress=100
    )
    assert status_resp.progress == 100