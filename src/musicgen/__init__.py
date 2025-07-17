"""
MusicGen Unified - Clean, focused instrumental music generation.
"""

__version__ = "2.0.1"

from .generator import MusicGenerator
from .batch import BatchProcessor
from .prompt import PromptEngineer

__all__ = ["MusicGenerator", "BatchProcessor", "PromptEngineer"]