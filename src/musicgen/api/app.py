"""
Application factory for MusicGen API.

This module provides a factory function to create configured FastAPI instances.
"""

from fastapi import FastAPI

from .rest.app import app as base_app


def create_app() -> FastAPI:
    """
    Create and configure a FastAPI application instance.

    Returns:
        FastAPI: Configured application instance
    """
    # Return the base app which already has CORS configured
    return base_app


__all__ = ["app", "create_app"]

# Export the default app
app = create_app()
