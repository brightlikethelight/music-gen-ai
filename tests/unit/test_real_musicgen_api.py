"""
Tests for generation API endpoints.
"""

import pytest

# Import API modules - handle missing dependencies gracefully
try:
    from musicgen.api.rest.app import app, GenerationRequest, GenerationResponse
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    app = None
    GenerationRequest = None
    GenerationResponse = None


@pytest.mark.unit
@pytest.mark.skipif(not API_AVAILABLE, reason="API dependencies not available")
class TestGenerationApi:
    """Test cases for generation API endpoints."""

    def test_app_exists(self):
        """Test that app exists."""
        assert app is not None
        assert app.title == "MusicGen API"

    def test_generation_models(self):
        """Test that request/response models exist."""
        if GenerationRequest is not None:
            # Test request model
            request = GenerationRequest(prompt="Generate upbeat music", duration=10.0)
            assert request.prompt == "Generate upbeat music"
            assert request.duration == 10.0

        if GenerationResponse is not None:
            # Test response model validation
            response = GenerationResponse(
                job_id="test-task-123",
                status="completed",
                message="Generation completed"
            )
            assert response.job_id == "test-task-123"
            assert response.status == "completed"
