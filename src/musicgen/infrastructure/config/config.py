"""
Configuration settings for MusicGen Unified.

Environment variables for educational demonstration.
"""

import os
import secrets
import tempfile
from typing import Optional


def _parse_bool(value: str, default: bool = False) -> bool:
    """Parse boolean from environment variable string."""
    if value.lower() in ("true", "1", "yes", "on"):
        return True
    elif value.lower() in ("false", "0", "no", "off"):
        return False
    return default


def _parse_int(value: str, default: int) -> int:
    """Parse integer from environment variable string with fallback."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _parse_float(value: str, default: float) -> float:
    """Parse float from environment variable string with fallback."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


class Config:
    """Application configuration with environment-aware defaults."""

    def __init__(self) -> None:
        """Initialize configuration from environment variables."""
        # Environment
        self.ENVIRONMENT: str = os.environ.get("ENVIRONMENT", "development")
        self.DEBUG: bool = _parse_bool(
            os.environ.get("DEBUG", "true" if self.ENVIRONMENT == "development" else "false")
        )

        # Model settings
        self.MODEL_NAME: str = os.environ.get("MODEL_NAME", "facebook/musicgen-small")
        self.DEVICE: Optional[str] = os.environ.get("DEVICE", "auto")
        self.OPTIMIZE: bool = _parse_bool(os.environ.get("OPTIMIZE", "true"), True)
        cache_dir = os.environ.get("MODEL_CACHE_DIR", "~/.cache/musicgen")
        self.MODEL_CACHE_DIR: str = os.path.expanduser(cache_dir)

        # Generation limits
        self.MAX_DURATION: float = _parse_float(os.environ.get("MAX_DURATION", "300"), 300.0)
        self.DEFAULT_DURATION: float = _parse_float(os.environ.get("DEFAULT_DURATION", "30"), 30.0)
        self.MAX_PROMPT_LENGTH: int = _parse_int(os.environ.get("MAX_PROMPT_LENGTH", "256"), 256)

        # API settings
        self.API_HOST: str = os.environ.get("API_HOST", "127.0.0.1")
        self.API_PORT: int = _parse_int(os.environ.get("API_PORT", "8000"), 8000)
        self.API_WORKERS: int = _parse_int(os.environ.get("API_WORKERS", "1"), 1)
        self.API_KEY: Optional[str] = os.environ.get("API_KEY", None)

        # CORS settings
        self.CORS_ORIGINS: list = os.environ.get("CORS_ORIGINS", "*").split(",")
        self.CORS_CREDENTIALS: bool = _parse_bool(os.environ.get("CORS_CREDENTIALS", "true"), True)

        # Rate limiting
        self.RATE_LIMIT_ENABLED: bool = _parse_bool(
            os.environ.get("RATE_LIMIT_ENABLED", "true"), True
        )
        self.RATE_LIMIT_PER_MINUTE: int = _parse_int(
            os.environ.get("RATE_LIMIT_PER_MINUTE", "60"), 60
        )
        self.RATE_LIMIT_PER_HOUR: int = _parse_int(
            os.environ.get("RATE_LIMIT_PER_HOUR", "1000"), 1000
        )

        # Storage
        self.OUTPUT_DIR: str = os.environ.get("OUTPUT_DIR", "outputs")
        self.TEMP_DIR: str = os.environ.get(
            "TEMP_DIR", os.path.join(tempfile.gettempdir(), "musicgen")
        )
        self.JOB_RETENTION_HOURS: int = _parse_int(os.environ.get("JOB_RETENTION_HOURS", "24"), 24)

        # Batch processing
        self.BATCH_MAX_WORKERS: int = _parse_int(os.environ.get("BATCH_MAX_WORKERS", "4"), 4)
        self.BATCH_TIMEOUT: int = _parse_int(os.environ.get("BATCH_TIMEOUT", "300"), 300)
        self.MAX_BATCH_SIZE: int = _parse_int(os.environ.get("MAX_BATCH_SIZE", "100"), 100)

        # Logging - adjust based on environment
        default_log_level = "DEBUG" if self.ENVIRONMENT == "development" else "INFO"
        if self.ENVIRONMENT == "production":
            default_log_level = "WARNING"
        self.LOG_LEVEL: str = os.environ.get("LOG_LEVEL", default_log_level)
        self.LOG_FORMAT: str = os.environ.get(
            "LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Security
        self.SECRET_KEY: str = os.environ.get("SECRET_KEY", secrets.token_urlsafe(32))
        self.SECURE_HEADERS: bool = _parse_bool(os.environ.get("SECURE_HEADERS", "true"), True)

        # AWS (optional)
        self.AWS_REGION: Optional[str] = os.environ.get("AWS_REGION", None)
        self.S3_BUCKET: Optional[str] = os.environ.get("S3_BUCKET", None)

    def to_dict(self) -> dict:
        """Return configuration as dictionary."""
        return {
            "ENVIRONMENT": self.ENVIRONMENT,
            "DEBUG": self.DEBUG,
            "MODEL_NAME": self.MODEL_NAME,
            "DEVICE": self.DEVICE,
            "OPTIMIZE": self.OPTIMIZE,
            "MODEL_CACHE_DIR": self.MODEL_CACHE_DIR,
            "MAX_DURATION": self.MAX_DURATION,
            "DEFAULT_DURATION": self.DEFAULT_DURATION,
            "MAX_PROMPT_LENGTH": self.MAX_PROMPT_LENGTH,
            "API_HOST": self.API_HOST,
            "API_PORT": self.API_PORT,
            "API_WORKERS": self.API_WORKERS,
            "API_KEY": self.API_KEY,
            "CORS_ORIGINS": self.CORS_ORIGINS,
            "CORS_CREDENTIALS": self.CORS_CREDENTIALS,
            "RATE_LIMIT_ENABLED": self.RATE_LIMIT_ENABLED,
            "RATE_LIMIT_PER_MINUTE": self.RATE_LIMIT_PER_MINUTE,
            "RATE_LIMIT_PER_HOUR": self.RATE_LIMIT_PER_HOUR,
            "OUTPUT_DIR": self.OUTPUT_DIR,
            "TEMP_DIR": self.TEMP_DIR,
            "JOB_RETENTION_HOURS": self.JOB_RETENTION_HOURS,
            "BATCH_MAX_WORKERS": self.BATCH_MAX_WORKERS,
            "BATCH_TIMEOUT": self.BATCH_TIMEOUT,
            "MAX_BATCH_SIZE": self.MAX_BATCH_SIZE,
            "LOG_LEVEL": self.LOG_LEVEL,
            "LOG_FORMAT": self.LOG_FORMAT,
            "SECURE_HEADERS": self.SECURE_HEADERS,
        }

    def validate(self) -> bool:
        """Validate configuration."""
        errors = []

        if self.MAX_DURATION > 600:
            errors.append("MAX_DURATION should not exceed 600 seconds")

        if self.API_KEY and len(self.API_KEY) < 16:
            errors.append("API_KEY should be at least 16 characters")

        if len(self.SECRET_KEY) < 32:
            errors.append("SECRET_KEY should be at least 32 characters for security")

        if errors:
            raise ValueError(f"Configuration errors: {'; '.join(errors)}")

        return True

    def get_model_config(self) -> dict:
        """Get model-specific configuration."""
        return {
            "model_name": self.MODEL_NAME,
            "device": self.DEVICE,
            "optimize": self.OPTIMIZE,
            "cache_dir": self.MODEL_CACHE_DIR,
        }

    def get_api_config(self) -> dict:
        """Get API-specific configuration."""
        return {
            "host": self.API_HOST,
            "port": self.API_PORT,
            "workers": self.API_WORKERS,
            "cors_origins": self.CORS_ORIGINS,
            "rate_limit_enabled": self.RATE_LIMIT_ENABLED,
        }

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT == "development"

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == "production"

    def is_staging(self) -> bool:
        """Check if running in staging environment."""
        return self.ENVIRONMENT == "staging"

    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.ENVIRONMENT == "testing" or os.environ.get("PYTEST_CURRENT_TEST") is not None

    def get_log_level(self) -> str:
        """Get the configured log level."""
        return self.LOG_LEVEL


# Create singleton instance
config = Config()
