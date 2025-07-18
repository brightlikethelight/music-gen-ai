"""
Core music generation functionality.

This module contains the fundamental business logic for music generation,
including the main generator engine and audio processing utilities.
"""

from . import generator, prompt
from .generator import MusicGenerator
from .prompt import PromptEngineer

__all__ = ["generator", "prompt", "MusicGenerator", "PromptEngineer"]
