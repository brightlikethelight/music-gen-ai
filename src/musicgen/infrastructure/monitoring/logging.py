"""
Structured logging configuration for MusicGen.

Provides consistent logging across all components with support for different
output formats and log levels based on environment.
"""

import logging
import sys
from typing import Dict, Any
from pathlib import Path

try:
    import structlog

    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False


def configure_logging(
    level: str = "INFO", format_type: str = "detailed", log_file: str = None
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
    handlers = []

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

    # Silence some noisy libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    if STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    return logging.getLogger(name)


class ContextualLogger:
    """Logger that maintains context across operations."""

    def __init__(self, name: str):
        self.logger = get_logger(name)
        self.context = {}

    def with_context(self, **kwargs) -> "ContextualLogger":
        """Create a new logger with additional context."""
        new_logger = ContextualLogger(self.logger.name)
        new_logger.context = {**self.context, **kwargs}
        new_logger.logger = self.logger
        return new_logger

    def _log_with_context(self, level: str, message: str, **kwargs):
        """Log message with context."""
        full_context = {**self.context, **kwargs}

        if STRUCTLOG_AVAILABLE:
            getattr(self.logger, level)(message, **full_context)
        else:
            # Format context for standard logging
            context_str = " ".join(f"{k}={v}" for k, v in full_context.items())
            if context_str:
                message = f"{message} | {context_str}"
            getattr(self.logger, level)(message)

    def debug(self, message: str, **kwargs):
        self._log_with_context("debug", message, **kwargs)

    def info(self, message: str, **kwargs):
        self._log_with_context("info", message, **kwargs)

    def warning(self, message: str, **kwargs):
        self._log_with_context("warning", message, **kwargs)

    def error(self, message: str, **kwargs):
        self._log_with_context("error", message, **kwargs)

    def exception(self, message: str, **kwargs):
        if STRUCTLOG_AVAILABLE:
            self.logger.exception(message, **{**self.context, **kwargs})
        else:
            context_str = " ".join(f"{k}={v}" for k, v in {**self.context, **kwargs}.items())
            if context_str:
                message = f"{message} | {context_str}"
            self.logger.exception(message)
