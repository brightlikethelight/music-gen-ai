"""
Basic integration test to demonstrate test framework functionality.
This establishes that integration testing capability exists even if
full integration tests are not yet implemented.
"""

import pytest
from pathlib import Path
import tempfile
import json


@pytest.mark.integration
class TestBasicIntegration:
    """Basic integration tests to establish framework functionality."""

    def test_config_loading_integration(self):
        """Test that configuration can be loaded and used across modules."""
        from musicgen.infrastructure.config.config import Config

        # Test default configuration loading
        config = Config()

        # Verify configuration has expected attributes
        assert hasattr(config, "MODEL_NAME")
        assert hasattr(config, "API_HOST")
        assert hasattr(config, "API_PORT")

        # Verify configuration values are reasonable
        assert config.API_PORT > 0
        assert config.API_PORT < 65536
        assert isinstance(config.API_HOST, str)
        assert isinstance(config.MODEL_NAME, str)

    def test_helpers_integration(self):
        """Test that helper functions work together correctly."""
        from musicgen.utils.helpers import format_time, hash_text, validate_prompt_length

        # Test a workflow using multiple helpers
        prompt = "Generate a 30 second jazz piano piece"
        duration = 30.0

        # Validate and process prompt
        validated_prompt = validate_prompt_length(prompt)
        prompt_hash = hash_text(validated_prompt)
        time_str = format_time(duration)

        # Verify integration
        assert len(prompt_hash) == 8
        assert time_str == "30.0s"
        assert validated_prompt == prompt  # Should not be truncated

    def test_file_operations_integration(self):
        """Test basic file operations work correctly."""
        # Create a temporary file and verify operations
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            test_data = {"prompt": "Test prompt", "duration": 15.0, "model": "test-model"}
            json.dump(test_data, f)
            temp_path = Path(f.name)

        try:
            # Verify file exists and can be read
            assert temp_path.exists()

            # Read and verify contents
            with open(temp_path, "r") as f:
                loaded_data = json.load(f)

            assert loaded_data["prompt"] == "Test prompt"
            assert loaded_data["duration"] == 15.0
            assert loaded_data["model"] == "test-model"

        finally:
            # Clean up
            if temp_path.exists():
                temp_path.unlink()

    def test_exception_handling_integration(self):
        """Test that custom exceptions work correctly across modules."""
        from musicgen.utils.exceptions import MusicGenError, ConfigurationError, ValidationError

        # Test exception hierarchy
        assert issubclass(ConfigurationError, MusicGenError)
        assert issubclass(ValidationError, MusicGenError)

        # Test exception creation
        try:
            raise ValidationError("Test validation error")
        except ValidationError as e:
            assert str(e) == "Test validation error"
            assert isinstance(e, MusicGenError)
            assert isinstance(e, Exception)
