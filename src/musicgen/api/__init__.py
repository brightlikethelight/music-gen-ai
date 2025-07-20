"""
API layer for MusicGen.

Provides REST and streaming APIs for music generation services.
"""

try:
    from .rest.app import app
except ImportError:
    # Fallback to hybrid app if main app not available
    try:
        from .rest.hybrid_app import app
    except ImportError:
        # Last resort: create a minimal app for CI/CD
        from fastapi import FastAPI

        app = FastAPI(title="MusicGen API", version="2.0.1")

__all__ = ["app"]
