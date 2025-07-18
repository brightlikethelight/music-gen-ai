"""
MusicGen Unified - Clean, focused instrumental music generation.
"""

__version__ = "2.0.1"


# Lazy imports to avoid heavy ML dependencies at package level
def __getattr__(name):
    """Lazy import for heavy ML dependencies."""
    if name == "MusicGenerator":
        from .core.generator import MusicGenerator

        return MusicGenerator
    elif name == "BatchProcessor":
        from .services.batch import BatchProcessor

        return BatchProcessor
    elif name == "PromptEngineer":
        from .core.prompt import PromptEngineer

        return PromptEngineer
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["MusicGenerator", "BatchProcessor", "PromptEngineer"]
