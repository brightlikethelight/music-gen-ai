"""
Tests for musicgen.utils.exceptions module
"""

import pytest

# Import all exceptions from the actual module
from musicgen.utils.exceptions import (
    APIError,
    AudioProcessingError,
    ConfigurationError,
    DataLoadingError,
    GenerationError,
    ModelError,
    MusicGenError,
    ResourceError,
    ValidationError,
)


class TestExceptions:
    """Test custom exception classes."""

    def test_base_exception(self):
        """Test base MusicGenError."""
        with pytest.raises(MusicGenError) as exc_info:
            raise MusicGenError("Test error")

        assert str(exc_info.value) == "Test error"
        assert exc_info.value.details == {}
        assert exc_info.value.error_code == "MusicGenError"

    def test_exception_with_details(self):
        """Test exception with details and error code."""
        details = {"key": "value", "code": 123}

        with pytest.raises(ModelError) as exc_info:
            raise ModelError("Model failed", details=details, error_code="MODEL_001")

        assert str(exc_info.value) == "Model failed"
        assert exc_info.value.details == details
        assert exc_info.value.error_code == "MODEL_001"

    def test_all_exception_types(self):
        """Test that all exception types can be raised properly."""
        exception_types = [
            (ModelError, "Model error"),
            (GenerationError, "Generation failed"),
            (AudioProcessingError, "Audio error"),
            (ConfigurationError, "Config error"),
            (ValidationError, "Validation failed"),
            (DataLoadingError, "Data loading error"),
            (ResourceError, "Resource error"),
            (APIError, "API error"),
        ]

        for exc_class, message in exception_types:
            with pytest.raises(exc_class) as exc_info:
                raise exc_class(message)
            assert str(exc_info.value) == message
            assert isinstance(exc_info.value, MusicGenError)


class TestExceptionUsage:
    """Test real-world usage patterns."""

    def test_api_error_chain(self):
        """Test chaining API-related exceptions."""
        try:
            try:
                raise ValidationError("Invalid input")
            except ValidationError as e:
                raise APIError("API request failed") from e
        except APIError as e:
            assert e.__cause__ is not None
            assert isinstance(e.__cause__, ValidationError)

    def test_model_loading_error_flow(self):
        """Test model loading error flow."""

        def load_model(path):
            if not path:
                raise ValidationError("Path cannot be empty")
            if path == "missing":
                raise ModelError(f"Model not found: {path}")
            if path == "corrupt":
                raise ModelError("Model file corrupted", details={"path": path})
            return "model"

        # Test validation
        with pytest.raises(ValidationError):
            load_model("")

        # Test not found
        with pytest.raises(ModelError) as exc_info:
            load_model("missing")
        assert "Model not found" in str(exc_info.value)

        # Test corruption
        with pytest.raises(ModelError) as exc_info:
            load_model("corrupt")
        assert exc_info.value.details["path"] == "corrupt"

        # Test success
        assert load_model("valid") == "model"

    def test_generation_pipeline_errors(self):
        """Test generation pipeline error handling."""

        def generate_audio(prompt, model=None):
            if not prompt:
                raise ValidationError("Prompt cannot be empty")
            if model is None:
                raise ConfigurationError("Model not configured")
            if prompt == "oom":
                raise ResourceError("Out of memory")
            return f"audio_{prompt}"

        # Test various error scenarios
        with pytest.raises(ValidationError):
            generate_audio("")
        with pytest.raises(ConfigurationError):
            generate_audio("test")
        with pytest.raises(ResourceError):
            generate_audio("oom", model="test")

        # Test success
        assert generate_audio("music", model="test") == "audio_music"
