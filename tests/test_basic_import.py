"""Basic import test to ensure CI has at least one passing test."""


def test_can_import_musicgen():
    """Test that musicgen package can be imported."""
    import musicgen

    assert musicgen.__version__ == "2.0.1"


def test_can_import_exceptions():
    """Test that exceptions can be imported."""
    from musicgen.utils.exceptions import MusicGenError

    assert MusicGenError is not None


def test_can_import_core_modules():
    """Test that core modules can be imported."""
    from musicgen.core import generator, prompt
    from musicgen.cli import main as cli
    from musicgen.services import batch

    # Should be able to import without errors
    assert generator is not None
    assert cli is not None
    assert batch is not None
    assert prompt is not None
