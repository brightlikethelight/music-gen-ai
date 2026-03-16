"""
Structured logging configuration for MusicGen.

Provides consistent logging across all components with support for different
output formats and log levels based on environment.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, cast

try:
    import structlog

    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Setup structured logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    try:
        # Try to use config if available and no parameters provided
        if level == "INFO" and log_file is None:
            from musicgen.infrastructure.config.config import config

            level = config.LOG_LEVEL
    except ImportError:
        # Use defaults if config not available
        pass

    configure_logging(
        level=level,
        format_type="json" if STRUCTLOG_AVAILABLE else "detailed",
        log_file=log_file,
    )


def configure_logging(
    level: str = "INFO", format_type: str = "detailed", log_file: Optional[str] = None
) -> None:
    """
    Configure application logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        format_type: Format type (minimal, detailed, json)
        log_file: Optional log file path
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure basic logging
    handlers: list[logging.Handler] = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    if format_type == "minimal":
        formatter = logging.Formatter("%(levelname)s: %(message)s")
    elif format_type == "json" and STRUCTLOG_AVAILABLE:
        # Use structlog for JSON formatting
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        formatter = None
    else:
        # Detailed format
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )

    if formatter:
        console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        if formatter:
            file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(level=log_level, handlers=handlers, force=True)

    # Also configure musicgen logger for the package hierarchy
    musicgen_logger = logging.getLogger("musicgen")
    musicgen_logger.setLevel(log_level)
    musicgen_logger.handlers = []  # Clear existing handlers
    for handler in handlers:
        musicgen_logger.addHandler(handler)

    # Silence some noisy libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    if STRUCTLOG_AVAILABLE:
        return cast(logging.Logger, structlog.get_logger(name))
    return logging.getLogger(name)
