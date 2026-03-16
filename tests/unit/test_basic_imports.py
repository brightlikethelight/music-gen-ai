"""
Test basic imports and module structure.

These tests provide quick coverage wins by testing basic imports and module attributes.
"""

import pytest


class TestBasicImports:
    """Test basic module imports and structure."""

    def test_main_module_import(self):
        """Test main musicgen module can be imported."""
        import musicgen

        # Test version is accessible
        assert hasattr(musicgen, "__version__")
        assert isinstance(musicgen.__version__, str)
        assert "." in musicgen.__version__  # Should be semantic version

    def test_main_module_all_exports(self):
        """Test __all__ exports are accessible."""
        import musicgen

        # Test __all__ is defined
        assert hasattr(musicgen, "__all__")
        assert isinstance(musicgen.__all__, list)

        # Test that all items in __all__ are accessible
        for item in musicgen.__all__:
            assert hasattr(musicgen, item)

    def test_utils_module_import(self):
        """Test utils module imports work."""
        from musicgen.utils import GenerationError, MusicGenError

        # Test exception classes are properly defined
        assert issubclass(MusicGenError, Exception)
        assert issubclass(GenerationError, MusicGenError)

    def test_core_module_import(self):
        """Test core module imports work."""
        import musicgen.core

        # Test module has expected attributes
        assert hasattr(musicgen.core, "__all__")
        assert "MusicGenerator" in musicgen.core.__all__
        assert "PromptEngineer" in musicgen.core.__all__

    def test_config_module_import(self):
        """Test config module imports work."""
        from musicgen.infrastructure.config.config import Config

        # Test Config class exists
        assert Config is not None

    def test_main_entry_point(self):
        """Test main entry point module."""
        import musicgen.__main__

        # Test module loads without error
        assert musicgen.__main__ is not None

    def test_api_module_import(self):
        """Test API module imports work."""
        from musicgen.api import app

        # Test app exists
        assert app is not None

    def test_api_main_import(self):
        """Test API app module imports work."""
        import musicgen.api.app

        # Test module has app export
        assert hasattr(musicgen.api.app, "__all__")
        assert "app" in musicgen.api.app.__all__


class TestLazyImports:
    """Test lazy import functionality in main module."""

    def test_lazy_import_musicgenerator(self):
        """Test MusicGenerator lazy import works."""
        import musicgen

        # This should trigger the lazy import
        music_gen_class = musicgen.MusicGenerator
        assert music_gen_class is not None

    def test_lazy_import_prompt_engineer(self):
        """Test PromptEngineer lazy import works."""
        import musicgen

        # This should trigger the lazy import
        prompt_class = musicgen.PromptEngineer
        assert prompt_class is not None

    def test_lazy_import_batch_processor(self):
        """Test BatchProcessor lazy import works."""
        import musicgen

        # This should trigger the lazy import
        batch_class = musicgen.BatchProcessor
        assert batch_class is not None

    def test_lazy_import_invalid_attribute(self):
        """Test lazy import raises AttributeError for invalid attributes."""
        import musicgen

        with pytest.raises(
            AttributeError, match="module 'musicgen' has no attribute 'InvalidClass'"
        ):
            _ = musicgen.InvalidClass


class TestModuleConstants:
    """Test module constants and metadata."""

    def test_version_format(self):
        """Test version follows semantic versioning."""
        import musicgen

        version = musicgen.__version__
        parts = version.split(".")

        # Should have at least major.minor.patch
        assert len(parts) >= 3

        # Major, minor, patch should be numeric
        assert parts[0].isdigit()
        assert parts[1].isdigit()
        assert parts[2].split("-")[0].isdigit()  # Handle pre-release versions

    def test_exception_inheritance(self):
        """Test exception class inheritance chain."""
        from musicgen.utils import GenerationError, MusicGenError

        # Test inheritance chain
        assert issubclass(GenerationError, MusicGenError)
        assert issubclass(MusicGenError, Exception)

        # Test can be instantiated
        base_error = MusicGenError("test")
        assert str(base_error) == "test"

        gen_error = GenerationError("test generation")
        assert str(gen_error) == "test generation"
