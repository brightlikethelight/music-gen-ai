"""
API layer for MusicGen.

Provides REST and streaming APIs for music generation services.
"""

from .rest.app import app

__all__ = ["app"]
