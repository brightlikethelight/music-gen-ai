"""
Tests for api/main.py
"""

import pytest

from musicgen.api.app import app
from musicgen.api.endpoints import generation, health, streaming


class TestApp:
    """Test cases for consolidated API app."""

    def test_app_exists(self):
        """Test that FastAPI app exists."""
        assert app is not None
        assert app.title == "Music Gen AI API"

    def test_endpoints_imported(self):
        """Test that all endpoint modules are imported."""
        assert generation is not None
        assert health is not None
        assert streaming is not None
