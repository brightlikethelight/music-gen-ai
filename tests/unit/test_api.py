"""
Unit tests for consolidated API functionality.
"""

from io import BytesIO
from unittest.mock import Mock, patch

import pytest
import torch
from fastapi.testclient import TestClient

# Import API modules - handle missing dependencies gracefully
try:
    from musicgen.api.rest.app import app

    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    app = None


@pytest.mark.unit
@pytest.mark.skipif(not API_AVAILABLE, reason="API dependencies not available")
class TestMainAPI:
    """Test main FastAPI application."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    

    def test_app_exists(self):
        """Test that app exists and has correct metadata."""
        assert app is not None
        assert app.title == "MusicGen API"
        assert app.version == "2.0.1"

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_api_prefix(self, client):
        """Test that API uses correct prefix."""
        # Test that endpoints exist
        response = client.get("/health")
        assert response.status_code == 200

        # Test that root path (may not be implemented)
        response = client.get("/")
        assert response.status_code in [200, 307, 404]  # Redirect, docs, or not found

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200

        data = response.json()
        assert "generation_requests" in data
        assert "generation_completed" in data
        assert "generation_failed" in data
        assert "active_generations" in data
        assert "active_jobs" in data
        assert "total_jobs" in data

        # All counts should be zero initially
        assert data["generation_requests"] == 0
        assert data["generation_completed"] == 0
        assert data["generation_failed"] == 0
        assert data["active_generations"] == 0


@pytest.mark.unit
@pytest.mark.skipif(not API_AVAILABLE, reason="API dependencies not available")
class TestGenerationEndpoint:
    """Test generation API endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    

    @patch("musicgen.api.rest.app.BackgroundTasks.add_task")
    def test_generation_endpoint_validation(self, mock_add_task, client, auth_headers):
        """Test input validation on generation endpoint."""
        # Missing required fields
        response = client.post("/generate", json={}, headers=auth_headers)
        assert response.status_code == 422

        # Invalid duration
        response = client.post("/generate", json={"prompt": "Test", "duration": -1}, headers=auth_headers)
        assert response.status_code == 422

        # Valid minimal request (will fail with 503 if model not loaded)
        response = client.post("/generate", json={"prompt": "Test music"}, headers=auth_headers)
        assert response.status_code in [200, 503]

        # If successful, verify background task was added
        if response.status_code == 200:
            assert mock_add_task.called

    @patch("musicgen.api.rest.app.load_model")
    @patch("torchaudio.save")
    def test_generate_endpoint_mocked(self, mock_save, mock_get_model, client, auth_headers):
        """Test generation endpoint with mocked model."""
        # Mock model
        mock_model = Mock()
        mock_model.generate.return_value = torch.randn(1, 1, 32000)
        mock_model.sample_rate = 32000
        mock_get_model.return_value = mock_model

        # Mock audio save
        mock_save.return_value = "test_output.wav"

        response = client.post("/generate", json={"prompt": "Happy jazz music", "duration": 10.0}, headers=auth_headers)

        # Check response
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "queued"
        assert data["message"] == "Music generation job queued successfully"


@pytest.mark.unit
@pytest.mark.skipif(not API_AVAILABLE, reason="API dependencies not available")
@pytest.mark.skip(reason="Multi-instrument endpoints not implemented yet")
class TestMultiInstrumentEndpoint:
    """Test multi-instrument API endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_multi_instrument_validation(self, client):
        """Test input validation on multi-instrument endpoint."""
        # Missing tracks
        response = client.post("/api/v1/multi-instrument/generate", json={})
        assert response.status_code == 422

        # Empty tracks
        response = client.post("/api/v1/multi-instrument/generate", json={"tracks": []})
        assert response.status_code == 422

        # Valid request structure
        response = client.post(
            "/api/v1/multi-instrument/generate",
            json={
                "tracks": [
                    {"instrument": "piano", "prompt": "Soft piano"},
                    {"instrument": "drums", "prompt": "Jazz drums"},
                ],
                "duration": 30.0,
            },
        )
        # Should be 200 or 503 (if model not loaded)
        assert response.status_code in [200, 503]


@pytest.mark.unit
@pytest.mark.skipif(not API_AVAILABLE, reason="API dependencies not available")
@pytest.mark.skip(reason="Streaming endpoints not implemented yet")
class TestStreamingEndpoint:
    """Test streaming API endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_streaming_session_creation(self, client):
        """Test creating streaming session."""
        response = client.post(
            "/api/v1/stream/session", json={"prompt": "Streaming test", "duration": 10.0}
        )

        # Should be 200 or 503 (if streaming not available)
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "session_id" in data
            assert "websocket_url" in data

    def test_list_sessions(self, client):
        """Test listing active sessions."""
        response = client.get("/api/v1/stream/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert "count" in data
        assert isinstance(data["sessions"], list)


@pytest.mark.unit
class TestAPIHelpers:
    """Test API helper functions and utilities."""

    def test_endpoint_modules_exist(self):
        """Test that all endpoint modules exist."""
        if API_AVAILABLE:
            # Check that the app has the expected routes
            routes = [route.path for route in app.routes]
            assert "/health" in routes
            assert "/generate" in routes
            # Additional endpoints may not be implemented yet

    def test_request_models(self):
        """Test that request/response models are properly defined."""
        if API_AVAILABLE:
            from musicgen.api.rest.app import GenerationRequest, GenerationResponse

            # Test model instantiation with available models
            gen_req = GenerationRequest(prompt="Test")
            assert gen_req.prompt == "Test"
            assert gen_req.duration == 30.0  # default

            # Test response model
            response = GenerationResponse(
                job_id="test-123", status="queued", message="Test message"
            )
            assert response.job_id == "test-123"
            assert response.status == "queued"


if __name__ == "__main__":
    pytest.main([__file__])
