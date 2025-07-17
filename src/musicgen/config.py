"""
Configuration settings for MusicGen Unified.

Environment variables for production deployment.
"""

import os
from typing import Optional


class Config:
    """Application configuration."""
    
    # Model settings
    MODEL_NAME: str = os.environ.get("MODEL_NAME", "facebook/musicgen-small")
    DEVICE: Optional[str] = os.environ.get("DEVICE", None)
    OPTIMIZE: bool = os.environ.get("OPTIMIZE", "true").lower() == "true"
    MODEL_CACHE_DIR: str = os.environ.get("MODEL_CACHE_DIR", "~/.cache/musicgen")
    
    # Generation limits
    MAX_DURATION: float = float(os.environ.get("MAX_DURATION", "300"))
    DEFAULT_DURATION: float = float(os.environ.get("DEFAULT_DURATION", "30"))
    MAX_PROMPT_LENGTH: int = int(os.environ.get("MAX_PROMPT_LENGTH", "256"))
    
    # API settings
    API_HOST: str = os.environ.get("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.environ.get("API_PORT", "8000"))
    API_WORKERS: int = int(os.environ.get("API_WORKERS", "1"))
    API_KEY: Optional[str] = os.environ.get("API_KEY", None)
    
    # CORS settings
    CORS_ORIGINS: list = os.environ.get("CORS_ORIGINS", "*").split(",")
    CORS_CREDENTIALS: bool = os.environ.get("CORS_CREDENTIALS", "true").lower() == "true"
    
    # Rate limiting
    RATE_LIMIT_ENABLED: bool = os.environ.get("RATE_LIMIT_ENABLED", "true").lower() == "true"
    RATE_LIMIT_PER_MINUTE: int = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "60"))
    RATE_LIMIT_PER_HOUR: int = int(os.environ.get("RATE_LIMIT_PER_HOUR", "1000"))
    
    # Storage
    OUTPUT_DIR: str = os.environ.get("OUTPUT_DIR", "outputs")
    TEMP_DIR: str = os.environ.get("TEMP_DIR", "/tmp/musicgen")
    JOB_RETENTION_HOURS: int = int(os.environ.get("JOB_RETENTION_HOURS", "24"))
    
    # Batch processing
    BATCH_MAX_WORKERS: int = int(os.environ.get("BATCH_MAX_WORKERS", "4"))
    BATCH_TIMEOUT: int = int(os.environ.get("BATCH_TIMEOUT", "300"))
    
    # Logging
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.environ.get("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Security
    SECRET_KEY: str = os.environ.get("SECRET_KEY", "change-me-in-production")
    SECURE_HEADERS: bool = os.environ.get("SECURE_HEADERS", "true").lower() == "true"
    
    # AWS (optional)
    AWS_REGION: Optional[str] = os.environ.get("AWS_REGION", None)
    S3_BUCKET: Optional[str] = os.environ.get("S3_BUCKET", None)
    
    @classmethod
    def validate(cls):
        """Validate configuration."""
        errors = []
        
        if cls.MAX_DURATION > 600:
            errors.append("MAX_DURATION should not exceed 600 seconds")
        
        if cls.API_KEY and len(cls.API_KEY) < 16:
            errors.append("API_KEY should be at least 16 characters")
        
        if cls.SECRET_KEY == "change-me-in-production":
            import logging
            logging.warning("Using default SECRET_KEY - change in production!")
        
        if errors:
            raise ValueError(f"Configuration errors: {'; '.join(errors)}")
        
        return True
    
    @classmethod
    def get_model_config(cls):
        """Get model-specific configuration."""
        return {
            "model_name": cls.MODEL_NAME,
            "device": cls.DEVICE,
            "optimize": cls.OPTIMIZE,
            "cache_dir": cls.MODEL_CACHE_DIR,
        }
    
    @classmethod
    def get_api_config(cls):
        """Get API-specific configuration."""
        return {
            "host": cls.API_HOST,
            "port": cls.API_PORT,
            "workers": cls.API_WORKERS,
            "cors_origins": cls.CORS_ORIGINS,
            "rate_limit_enabled": cls.RATE_LIMIT_ENABLED,
        }


# Create singleton instance
config = Config()