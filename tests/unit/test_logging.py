"""
Tests for musicgen.utils.logging module
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from musicgen.infrastructure.monitoring.logging import (
    get_logger,
    setup_logging,
)


class TestLoggingSetup:
    """Test logging setup functions."""

    def test_setup_logging_default(self):
        """Test default logging setup."""
        # Clear existing handlers and reset level to avoid state leaking from other tests
        logger = logging.getLogger("musicgen")
        logger.handlers = []
        logger.setLevel(logging.WARNING)

        # Prevent config import from overriding level (config.LOG_LEVEL defaults to
        # DEBUG in development, which would override the INFO we're testing for)
        with patch.dict("sys.modules", {"musicgen.infrastructure.config.config": None}):
            setup_logging(level="INFO")

        # Check logger is configured
        assert len(logger.handlers) > 0
        assert logger.level == logging.INFO

    def test_setup_logging_custom_level(self):
        """Test logging setup with custom level."""
        # Clear existing handlers
        logger = logging.getLogger("musicgen")
        logger.handlers = []

        setup_logging(level="DEBUG")

        # Check logger level
        assert logger.level == logging.DEBUG

    def test_setup_logging_with_file(self, tmp_path):
        """Test logging setup with file handler."""
        log_file = tmp_path / "test.log"

        # Clear existing handlers
        logger = logging.getLogger("musicgen")
        logger.handlers = []

        setup_logging(log_file=str(log_file))

        # Check file handler exists
        file_handlers = [h for h in logger.handlers if hasattr(h, "baseFilename")]
        assert len(file_handlers) > 0

    def test_get_logger(self):
        """Test getting logger instance."""
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

        # Should return same instance
        logger2 = get_logger("test.module")
        assert logger is logger2


class TestLoggingIntegration:
    """Test integrated logging scenarios."""

    def test_custom_formatter_logging(self):
        """Test custom formatter functionality."""
        logger = get_logger("test.custom")

        # Create a string stream handler with custom formatter
        import io

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)

        # Use a simple custom formatter since we removed pythonjsonlogger
        class SimpleFormatter(logging.Formatter):
            def format(self, record):
                # Simple key=value format
                base = super().format(record)
                extras = []
                for key, value in record.__dict__.items():
                    if key not in [
                        "name",
                        "msg",
                        "args",
                        "levelname",
                        "levelno",
                        "pathname",
                        "filename",
                        "module",
                        "lineno",
                        "funcName",
                        "created",
                        "msecs",
                        "relativeCreated",
                        "thread",
                        "threadName",
                        "processName",
                        "process",
                        "stack_info",
                        "exc_info",
                        "exc_text",
                        "message",
                    ]:
                        extras.append(f"{key}={value}")
                if extras:
                    return f"{base} | {' '.join(extras)}"
                return base

        formatter = SimpleFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        # Log a message with extra fields
        logger.info("Test message", extra={"user_id": "123", "action": "test"})

        # Check output contains our data
        output = stream.getvalue()
        assert "Test message" in output
        assert "user_id=123" in output
        assert "action=test" in output


class TestFileHandler:
    """Test file handler functionality."""

    def test_file_handler_creation(self, tmp_path):
        """Test creating file handler with setup_logging."""
        log_file = tmp_path / "test.log"

        # Clear existing handlers
        logger = logging.getLogger("musicgen")
        logger.handlers = []

        setup_logging(log_file=str(log_file))

        # Log a message
        test_logger = get_logger("musicgen.test")
        test_logger.info("Test message")

        # Check file was created and contains message
        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message" in content

    def test_exception_logging(self, caplog):
        """Test logging exceptions."""
        logger = get_logger("test.exceptions")

        try:
            raise ValueError("Test error")
        except ValueError:
            logger.exception("An error occurred")

        assert "An error occurred" in caplog.text
        assert "ValueError: Test error" in caplog.text
        assert "Traceback" in caplog.text

    def test_log_aggregation(self):
        """Test aggregating logs from multiple sources."""
        # Create parent logger - note the difference: musicgen vs music_gen
        parent_logger = get_logger("musicgen")

        # Create child loggers
        api_logger = get_logger("musicgen.api")
        model_logger = get_logger("musicgen.models")

        # Capture all logs
        import io

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        parent_logger.addHandler(handler)
        parent_logger.setLevel(logging.DEBUG)

        # Ensure child loggers propagate to parent
        api_logger.setLevel(logging.DEBUG)
        model_logger.setLevel(logging.DEBUG)

        # Log from different modules
        api_logger.info("API request received")
        model_logger.debug("Loading model")
        api_logger.warning("Rate limit approaching")

        output = stream.getvalue()

        assert "API request received" in output
        assert "Loading model" in output
        assert "Rate limit approaching" in output
