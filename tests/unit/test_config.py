"""Tests for configuration module."""

import os
# Set test environment
os.environ["JWT_SECRET_KEY"] = "test-key" 
os.environ["PYTEST_CURRENT_TEST"] = "1"
os.environ["MUSICGEN_SKIP_MODEL_DOWNLOAD"] = "1"

import pytest
from unittest.mock import patch
from musicgen.infrastructure.config.config import Config


class TestConfig:
    """Test configuration management."""
    
    def test_default_config(self):
        """Test default configuration values."""
        from musicgen.infrastructure.config.config import Config
        
        # Environment
        assert config.ENVIRONMENT in ["development", "staging", "production"]
        assert config.DEBUG in [True, False]
        
        # Model settings
        assert config.MODEL_NAME is not None
        assert config.DEVICE in ["cuda", "cpu", "auto"]
        assert config.MODEL_CACHE_DIR is not None
        
        # Generation limits
        assert config.MAX_DURATION > 0
        assert config.DEFAULT_DURATION > 0
        assert config.MAX_PROMPT_LENGTH > 0
        assert config.DEFAULT_DURATION <= config.MAX_DURATION
        
        # API settings
        assert config.API_HOST is not None
        assert config.API_PORT > 0
        assert config.API_WORKERS >= 1
        
        # CORS settings
        assert isinstance(config.CORS_ORIGINS, list)
        assert config.CORS_CREDENTIALS in [True, False]
        
        # Rate limiting
        assert config.RATE_LIMIT_ENABLED in [True, False]
        assert config.RATE_LIMIT_PER_MINUTE > 0
        assert config.RATE_LIMIT_PER_HOUR > 0
        assert config.RATE_LIMIT_PER_HOUR > config.RATE_LIMIT_PER_MINUTE
        
        # Storage
        assert config.OUTPUT_DIR is not None
        assert config.TEMP_DIR is not None
        assert config.JOB_RETENTION_HOURS > 0
        
        # Batch processing
        assert config.BATCH_MAX_WORKERS >= 1
        assert config.BATCH_TIMEOUT > 0
        assert config.MAX_BATCH_SIZE > 0
        
        # Logging
        assert config.LOG_LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert config.LOG_FORMAT is not None
    
    def test_environment_specific_config(self):
        """Test environment-specific configurations."""
        # Test development
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            config = Config()
            assert config.ENVIRONMENT == "development"
            assert config.DEBUG is True
            assert config.LOG_LEVEL in ["DEBUG", "INFO"]
        
        # Test production
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            config = Config()
            assert config.ENVIRONMENT == "production"
            assert config.DEBUG is False
            assert config.LOG_LEVEL in ["WARNING", "ERROR", "CRITICAL"]
        
        # Test staging
        with patch.dict(os.environ, {"ENVIRONMENT": "staging"}):
            config = Config()
            assert config.ENVIRONMENT == "staging"
            assert config.LOG_LEVEL in ["INFO", "WARNING"]
    
    def test_env_var_overrides(self):
        """Test environment variable overrides."""
        custom_values = {
            "MODEL_NAME": "facebook/musicgen-large",
            "API_PORT": "9000",
            "MAX_DURATION": "600",
            "DEFAULT_DURATION": "60",
            "RATE_LIMIT_PER_MINUTE": "30",
            "BATCH_MAX_WORKERS": "8",
            "LOG_LEVEL": "ERROR",
            "OUTPUT_DIR": "/custom/outputs"
        }
        
        with patch.dict(os.environ, custom_values):
            config = Config()
            assert config.MODEL_NAME == "facebook/musicgen-large"
            assert config.API_PORT == 9000
            assert config.MAX_DURATION == 600.0
            assert config.DEFAULT_DURATION == 60.0
            assert config.RATE_LIMIT_PER_MINUTE == 30
            assert config.BATCH_MAX_WORKERS == 8
            assert config.LOG_LEVEL == "ERROR"
            assert config.OUTPUT_DIR == "/custom/outputs"
    
    def test_type_conversions(self):
        """Test type conversions for config values."""
        with patch.dict(os.environ, {
            "API_PORT": "8080",
            "MAX_DURATION": "120.5",
            "DEBUG": "false",
            "RATE_LIMIT_ENABLED": "true",
            "CORS_CREDENTIALS": "False",
            "API_WORKERS": "4"
        }):
            config = Config()
            assert isinstance(config.API_PORT, int)
            assert config.API_PORT == 8080
            assert isinstance(config.MAX_DURATION, float)
            assert config.MAX_DURATION == 120.5
            assert isinstance(config.DEBUG, bool)
            assert config.DEBUG is False
            assert isinstance(config.RATE_LIMIT_ENABLED, bool)
            assert config.RATE_LIMIT_ENABLED is True
            assert isinstance(config.CORS_CREDENTIALS, bool)
            assert config.CORS_CREDENTIALS is False
            assert isinstance(config.API_WORKERS, int)
            assert config.API_WORKERS == 4
    
    def test_cors_origins_parsing(self):
        """Test CORS origins parsing."""
        # Single origin
        with patch.dict(os.environ, {"CORS_ORIGINS": "http://localhost:3000"}):
            config = Config()
            assert config.CORS_ORIGINS == ["http://localhost:3000"]
        
        # Multiple origins
        with patch.dict(os.environ, {"CORS_ORIGINS": "http://localhost:3000,http://example.com,https://app.example.com"}):
            config = Config()
            assert len(config.CORS_ORIGINS) == 3
            assert "http://localhost:3000" in config.CORS_ORIGINS
            assert "http://example.com" in config.CORS_ORIGINS
            assert "https://app.example.com" in config.CORS_ORIGINS
        
        # Wildcard
        with patch.dict(os.environ, {"CORS_ORIGINS": "*"}):
            config = Config()
            assert config.CORS_ORIGINS == ["*"]
    
    def test_model_cache_dir_expansion(self):
        """Test model cache directory path expansion."""
        # Test tilde expansion
        with patch.dict(os.environ, {"MODEL_CACHE_DIR": "~/.cache/test"}):
            config = Config()
            assert "~" not in config.MODEL_CACHE_DIR
            assert config.MODEL_CACHE_DIR.startswith("/")
        
        # Test absolute path
        with patch.dict(os.environ, {"MODEL_CACHE_DIR": "/absolute/path"}):
            config = Config()
            assert config.MODEL_CACHE_DIR == "/absolute/path"
    
    def test_validation_constraints(self):
        """Test configuration validation constraints."""
        config = Config()
        
        # Duration constraints
        assert config.DEFAULT_DURATION <= config.MAX_DURATION
        
        # Rate limit constraints
        assert config.RATE_LIMIT_PER_HOUR > config.RATE_LIMIT_PER_MINUTE
        
        # Worker constraints
        assert config.API_WORKERS >= 1
        assert config.BATCH_MAX_WORKERS >= 1
        
        # Port constraints
        assert 1 <= config.API_PORT <= 65535
        
        # Prompt length constraints
        assert config.MAX_PROMPT_LENGTH > 0
    
    def test_get_config_dict(self):
        """Test getting configuration as dictionary."""
        config = Config()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "ENVIRONMENT" in config_dict
        assert "API_PORT" in config_dict
        assert "MODEL_NAME" in config_dict
        assert "LOG_LEVEL" in config_dict
        
        # Check all important keys are present
        important_keys = [
            "ENVIRONMENT", "DEBUG", "MODEL_NAME", "DEVICE",
            "API_HOST", "API_PORT", "MAX_DURATION", "DEFAULT_DURATION",
            "RATE_LIMIT_ENABLED", "LOG_LEVEL"
        ]
        for key in important_keys:
            assert key in config_dict
    
    def test_config_singleton_pattern(self):
        """Test config follows singleton pattern if implemented."""
        config1 = Config()
        config2 = Config()
        
        # Both should have same values
        assert config1.API_PORT == config2.API_PORT
        assert config1.MODEL_NAME == config2.MODEL_NAME
        assert config1.ENVIRONMENT == config2.ENVIRONMENT
    
    def test_invalid_env_values(self):
        """Test handling of invalid environment values."""
        # Invalid port
        with patch.dict(os.environ, {"API_PORT": "invalid"}):
            config = Config()
            # Should fall back to default
            assert config.API_PORT == 8000  # default value
        
        # Invalid boolean
        with patch.dict(os.environ, {"DEBUG": "maybe"}):
            config = Config()
            # Should interpret as False for safety
            assert config.DEBUG is False
        
        # Invalid number
        with patch.dict(os.environ, {"MAX_DURATION": "not-a-number"}):
            config = Config()
            # Should fall back to default
            assert config.MAX_DURATION == 300.0  # default value
    
    def test_security_settings(self):
        """Test security-related configuration."""
        config = Config()
        
        # Rate limiting should be enabled by default
        assert config.RATE_LIMIT_ENABLED is True
        
        # API key is optional but should be None or string
        assert config.API_KEY is None or isinstance(config.API_KEY, str)
        
        # CORS should have sensible defaults
        if config.ENVIRONMENT == "production":
            # Production should not use wildcard
            assert config.CORS_ORIGINS != ["*"] or config.CORS_CREDENTIALS is False
    
    def test_logging_configuration(self):
        """Test logging configuration."""
        config = Config()
        
        # Log level should be valid
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert config.LOG_LEVEL in valid_levels
        
        # Log format should contain important fields
        assert config.LOG_FORMAT is not None
        assert "%(asctime)s" in config.LOG_FORMAT or "time" in config.LOG_FORMAT.lower()
        assert "%(levelname)s" in config.LOG_FORMAT or "level" in config.LOG_FORMAT.lower()
        assert "%(message)s" in config.LOG_FORMAT or "message" in config.LOG_FORMAT.lower()
    
    def test_batch_processing_config(self):
        """Test batch processing configuration."""
        config = Config()
        
        # Batch size limits
        assert config.MAX_BATCH_SIZE > 0
        assert config.MAX_BATCH_SIZE <= 1000  # Reasonable upper limit
        
        # Batch workers
        assert config.BATCH_MAX_WORKERS >= 1
        assert config.BATCH_MAX_WORKERS <= 32  # Reasonable upper limit
        
        # Batch timeout
        assert config.BATCH_TIMEOUT > 0
        assert config.BATCH_TIMEOUT <= 3600  # Max 1 hour
    
    def test_storage_configuration(self):
        """Test storage configuration."""
        config = Config()
        
        # Output directory
        assert config.OUTPUT_DIR is not None
        assert len(config.OUTPUT_DIR) > 0
        
        # Temp directory
        assert config.TEMP_DIR is not None
        assert len(config.TEMP_DIR) > 0
        
        # Job retention
        assert config.JOB_RETENTION_HOURS > 0
        assert config.JOB_RETENTION_HOURS <= 168  # Max 1 week
    
    def test_device_configuration(self):
        """Test device configuration for model loading."""
        # Test auto device
        with patch.dict(os.environ, {"DEVICE": "auto"}):
            config = Config()
            assert config.DEVICE == "auto"
        
        # Test specific GPU
        with patch.dict(os.environ, {"DEVICE": "cuda:0"}):
            config = Config()
            assert config.DEVICE == "cuda:0"
        
        # Test CPU
        with patch.dict(os.environ, {"DEVICE": "cpu"}):
            config = Config()
            assert config.DEVICE == "cpu"