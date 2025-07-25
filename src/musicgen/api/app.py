"""
Application factory for MusicGen API.

This module provides a factory function to create configured FastAPI instances.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .cors_config import cors_config
from .rest.app import app as base_app


def create_app() -> FastAPI:
    """
    Create and configure a FastAPI application instance.
    
    Returns:
        FastAPI: Configured application instance
    """
    # Return the base app which already has CORS configured
    return base_app


# Export the default app
app = create_app()