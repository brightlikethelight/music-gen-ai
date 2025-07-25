"""
Main entry point for MusicGen API.

This module exports the FastAPI app instance for use in tests and deployment.
"""

from .rest.app import app

__all__ = ["app"]
