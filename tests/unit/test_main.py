"""
Tests for API main app.
"""

import pytest

# Import API modules - handle missing dependencies gracefully
try:
    from musicgen.api.rest.app import app

    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    app = None


@pytest.mark.unit
@pytest.mark.skipif(not API_AVAILABLE, reason="API dependencies not available")
class TestApp:
    """Test cases for consolidated API app."""

    def test_app_exists(self):
        """Test that FastAPI app exists."""
        assert app is not None
        assert app.title == "MusicGen API"
        assert app.version == "2.0.1"

    def test_endpoints_configured(self):
        """Test that endpoints are configured."""
        # Check that the app has routes
        routes = [route.path for route in app.routes]
        assert "/health" in routes
        assert "/generate" in routes
