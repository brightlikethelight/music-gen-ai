"""Tests for input validation in API."""

import os

# Set test environment variables before any imports
os.environ["JWT_SECRET_KEY"] = "test-key"
os.environ["PYTEST_CURRENT_TEST"] = "1"
os.environ["MUSICGEN_SKIP_MODEL_DOWNLOAD"] = "1"

import pytest
from pydantic import ValidationError

from musicgen.api.rest.api import GenerateRequest


class TestGenerateRequestValidation:
    """Test input validation for GenerateRequest."""

    def test_valid_request(self):
        """Test valid request passes validation."""
        request = GenerateRequest(
            prompt="smooth jazz piano",
            duration=30.0,
            temperature=1.0,
            guidance_scale=3.0,
            format="mp3",
        )
        assert request.prompt == "smooth jazz piano"
        assert request.duration == 30.0
        assert request.format == "mp3"

    def test_prompt_length_validation(self):
        """Test prompt length validation."""
        # Too short
        with pytest.raises(ValidationError) as exc_info:
            GenerateRequest(prompt="ab")
        assert "at least 3 characters" in str(exc_info.value)

        # Too long
        with pytest.raises(ValidationError) as exc_info:
            GenerateRequest(prompt="x" * 501)
        assert "at most 500 characters" in str(exc_info.value)

    def test_prompt_sanitization(self):
        """Test prompt sanitization removes extra whitespace."""
        request = GenerateRequest(prompt="  smooth   jazz    piano  ")
        assert request.prompt == "smooth jazz piano"

    def test_dangerous_content_detection(self):
        """Test detection of dangerous patterns in prompt."""
        dangerous_prompts = [
            "<script>alert('xss')</script>",
            "javascript:void(0)",
            "file:///etc/passwd",
            "../../../etc/passwd",
            "test\\x00null",
            "test\0null",
        ]

        for dangerous_prompt in dangerous_prompts:
            with pytest.raises(ValidationError) as exc_info:
                GenerateRequest(prompt=dangerous_prompt)
            assert "dangerous content" in str(exc_info.value).lower()

    def test_duration_validation(self):
        """Test duration validation."""
        # Too short
        with pytest.raises(ValidationError) as exc_info:
            GenerateRequest(prompt="test", duration=0.05)
        assert "greater than or equal to 0.1" in str(exc_info.value)

        # Too long
        with pytest.raises(ValidationError) as exc_info:
            GenerateRequest(prompt="test", duration=301)
        assert "less than or equal to 300" in str(exc_info.value)

        # Valid durations
        assert GenerateRequest(prompt="test", duration=0.1).duration == 0.1
        assert GenerateRequest(prompt="test", duration=300).duration == 300

    def test_temperature_validation(self):
        """Test temperature validation."""
        # Too low
        with pytest.raises(ValidationError) as exc_info:
            GenerateRequest(prompt="test", temperature=0.05)
        assert "greater than or equal to 0.1" in str(exc_info.value)

        # Too high
        with pytest.raises(ValidationError) as exc_info:
            GenerateRequest(prompt="test", temperature=2.1)
        assert "less than or equal to 2" in str(exc_info.value)

    def test_guidance_scale_validation(self):
        """Test guidance scale validation."""
        # Too low
        with pytest.raises(ValidationError) as exc_info:
            GenerateRequest(prompt="test", guidance_scale=0.5)
        assert "greater than or equal to 1" in str(exc_info.value)

        # Too high
        with pytest.raises(ValidationError) as exc_info:
            GenerateRequest(prompt="test", guidance_scale=10.1)
        assert "less than or equal to 10" in str(exc_info.value)

    def test_format_validation(self):
        """Test output format validation."""
        # Valid formats
        assert GenerateRequest(prompt="test", format="mp3").format == "mp3"
        assert GenerateRequest(prompt="test", format="wav").format == "wav"
        assert GenerateRequest(prompt="test", format="MP3").format == "mp3"
        assert GenerateRequest(prompt="test", format="WAV").format == "wav"

        # Invalid format
        with pytest.raises(ValidationError) as exc_info:
            GenerateRequest(prompt="test", format="flac")
        assert "Format must be one of: mp3, wav" in str(exc_info.value)

    def test_defaults(self):
        """Test default values."""
        request = GenerateRequest(prompt="test music")
        assert request.duration == 30.0
        assert request.temperature == 1.0
        assert request.guidance_scale == 3.0
        assert request.format == "mp3"

    def test_edge_cases(self):
        """Test edge cases."""
        # Minimum valid prompt
        request = GenerateRequest(prompt="abc")
        assert request.prompt == "abc"

        # Maximum valid prompt
        long_prompt = "x" * 500
        request = GenerateRequest(prompt=long_prompt)
        assert len(request.prompt) == 500

        # Unicode handling
        request = GenerateRequest(prompt="音楽 מוזיקה موسيقى")
        assert request.prompt == "音楽 מוזיקה موسيقى"

        # Special characters (safe ones)
        request = GenerateRequest(prompt="rock & roll! (1980's)")
        assert request.prompt == "rock & roll! (1980's)"
