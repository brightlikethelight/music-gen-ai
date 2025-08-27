"""Final tests to reach 25% coverage threshold."""

import os
os.environ["JWT_SECRET_KEY"] = "test-key"
os.environ["PYTEST_CURRENT_TEST"] = "1"
os.environ["MUSICGEN_SKIP_MODEL_DOWNLOAD"] = "1"

import pytest
from unittest.mock import MagicMock, patch, Mock


def test_prompt_engineer_comprehensive():
    """Comprehensive test of PromptEngineer."""
    from musicgen.core.prompt import PromptEngineer
    
    engineer = PromptEngineer()
    
    # Test all methods
    prompt = "short"
    improved = engineer.improve_prompt(prompt)
    assert len(improved) > len(prompt)
    
    # Test validation
    is_valid, issues = engineer.validate_prompt("")
    assert is_valid is False
    
    is_valid, issues = engineer.validate_prompt("good jazz music")
    assert is_valid is True
    
    # Test examples
    examples = engineer.get_examples()
    assert len(examples) >= 5
    
    examples = engineer.get_examples(genre="electronic")
    assert len(examples) >= 3
    
    # Test variations
    variations = engineer.suggest_variations("piano", count=5)
    assert len(variations) == 5
    
    # Test _replace_or_add_mood
    result = engineer._replace_or_add_mood("happy music", "sad")
    assert "sad" in result.lower() or "happy" not in result.lower()


def test_config_comprehensive():
    """Comprehensive config tests."""
    from musicgen.infrastructure.config.config import Config
    
    # Test all attributes exist
    assert hasattr(Config, 'ENVIRONMENT')
    assert hasattr(Config, 'DEBUG')
    assert hasattr(Config, 'MODEL_NAME')
    assert hasattr(Config, 'DEVICE')
    assert hasattr(Config, 'OPTIMIZE')
    assert hasattr(Config, 'MODEL_CACHE_DIR')
    assert hasattr(Config, 'MAX_DURATION')
    assert hasattr(Config, 'DEFAULT_DURATION')
    assert hasattr(Config, 'MAX_PROMPT_LENGTH')
    assert hasattr(Config, 'API_HOST')
    assert hasattr(Config, 'API_PORT')
    assert hasattr(Config, 'API_WORKERS')
    assert hasattr(Config, 'API_KEY')
    assert hasattr(Config, 'CORS_ORIGINS')
    assert hasattr(Config, 'CORS_CREDENTIALS')
    assert hasattr(Config, 'RATE_LIMIT_ENABLED')
    assert hasattr(Config, 'RATE_LIMIT_PER_MINUTE')
    assert hasattr(Config, 'RATE_LIMIT_PER_HOUR')
    assert hasattr(Config, 'OUTPUT_DIR')
    assert hasattr(Config, 'TEMP_DIR')
    assert hasattr(Config, 'JOB_RETENTION_HOURS')
    assert hasattr(Config, 'BATCH_MAX_WORKERS')
    assert hasattr(Config, 'BATCH_TIMEOUT')
    assert hasattr(Config, 'MAX_BATCH_SIZE')
    assert hasattr(Config, 'LOG_LEVEL')
    assert hasattr(Config, 'LOG_FORMAT')
    assert hasattr(Config, 'SECRET_KEY')
    assert hasattr(Config, 'SECURE_HEADERS')
    
    # Test methods
    model_config = Config.get_model_config()
    assert isinstance(model_config, dict)
    
    api_config = Config.get_api_config()
    assert isinstance(api_config, dict)


def test_cors_config_comprehensive():
    """Comprehensive CORS config tests."""
    from musicgen.api.cors_config import CORSConfig
    
    config = CORSConfig()
    
    # Test attributes
    assert hasattr(config, 'environment')
    assert hasattr(config, 'allowed_origins')
    
    # Test methods if they exist
    if hasattr(config, 'is_origin_allowed'):
        # Test with localhost
        assert config.is_origin_allowed("http://localhost:3000") is True
        
    if hasattr(config, 'get_cors_headers'):
        headers = config.get_cors_headers()
        assert isinstance(headers, dict)


def test_rate_limiter_comprehensive():
    """Comprehensive rate limiter tests."""
    from musicgen.api.rest.middleware.rate_limiting import RateLimiter
    
    limiter = RateLimiter()
    
    # Test configuration
    assert limiter.limits["per_minute"] == 60
    assert limiter.limits["per_hour"] == 1000
    assert limiter.limits["per_day"] == 10000
    
    # Test IP utilities
    assert limiter._is_valid_ip("192.168.1.1") is True
    assert limiter._is_valid_ip("256.256.256.256") is False
    assert limiter._is_valid_ip("not-an-ip") is False
    
    # Test exempt IPs
    assert limiter._is_exempt_ip("127.0.0.1") is True
    assert limiter._is_exempt_ip("localhost") is False
    assert limiter._is_exempt_ip("192.168.1.1") is True
    assert limiter._is_exempt_ip("10.0.0.1") is True
    assert limiter._is_exempt_ip("172.16.0.1") is True
    assert limiter._is_exempt_ip("8.8.8.8") is False
    
    # Test client IP extraction
    request = Mock()
    request.client = Mock(host="192.168.1.100")
    request.headers = {}
    
    ip = limiter._get_client_ip(request)
    assert ip == "192.168.1.100"
    
    # Test with X-Forwarded-For
    request.headers = {"X-Forwarded-For": "203.0.113.1, 192.168.1.1"}
    ip = limiter._get_client_ip(request)
    assert ip == "203.0.113.1"


def test_logging_comprehensive():
    """Comprehensive logging tests."""
    from musicgen.infrastructure.monitoring.logging import (
        setup_logging,
        get_logger,
        JSONFormatter
    )
    
    # Test setup
    setup_logging(level="DEBUG")
    
    # Test get_logger
    logger = get_logger("test.module")
    assert logger.name == "test.module"
    
    # Test JSON formatter
    formatter = JSONFormatter()
    assert formatter is not None
    
    # Create a log record
    import logging
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Test message",
        args=(),
        exc_info=None
    )
    
    formatted = formatter.format(record)
    assert "Test message" in formatted
    assert "INFO" in formatted


def test_exceptions_comprehensive():
    """Comprehensive exception tests."""
    from musicgen.utils.exceptions import (
        MusicGenException,
        ConfigurationError,
        ValidationError,
        GenerationError,
        ModelError,
        APIError,
        AuthenticationError,
        AuthorizationError,
        RateLimitError,
        BatchProcessingError
    )
    
    # Test hierarchy
    assert issubclass(ConfigurationError, MusicGenException)
    assert issubclass(ValidationError, MusicGenException)
    assert issubclass(GenerationError, MusicGenException)
    assert issubclass(ModelError, MusicGenException)
    assert issubclass(APIError, MusicGenException)
    assert issubclass(AuthenticationError, APIError)
    assert issubclass(AuthorizationError, APIError)
    assert issubclass(RateLimitError, APIError)
    assert issubclass(BatchProcessingError, MusicGenException)
    
    # Test creation with different messages
    exc = ConfigurationError("Bad config", details={"key": "value"})
    assert "Bad config" in str(exc)
    
    exc = ValidationError("Invalid input", field="prompt")
    assert "Invalid input" in str(exc)
    
    exc = GenerationError("Gen failed", prompt="test")
    assert "Gen failed" in str(exc)
    
    exc = ModelError("Model error", model="test-model")
    assert "Model error" in str(exc)
    
    exc = AuthenticationError("Auth failed")
    assert "Auth failed" in str(exc)
    
    exc = AuthorizationError("Not authorized")
    assert "Not authorized" in str(exc)
    
    exc = RateLimitError("Too many requests")
    assert "Too many" in str(exc)
    
    exc = BatchProcessingError("Batch failed", job_id="123")
    assert "Batch failed" in str(exc)


def test_api_request_validation():
    """Test API request validation."""
    from musicgen.api.rest.api import GenerateRequest
    
    # Valid request
    req = GenerateRequest(prompt="jazz music", duration=30)
    assert req.prompt == "jazz music"
    assert req.duration == 30
    assert req.temperature == 1.0
    assert req.guidance_scale == 3.0
    assert req.format == "mp3"
    
    # Test validation
    req = GenerateRequest(prompt="rock music", duration=10, format="wav")
    assert req.format == "wav"
    
    # Test prompt validation
    req = GenerateRequest(prompt="  test  music  ")
    assert "  " not in req.prompt  # Should be normalized