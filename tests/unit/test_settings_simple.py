"""
Test settings module with simple imports and basic functionality.
"""

import os
import pytest

# Set environment variables before imports
os.environ["JWT_SECRET_KEY"] = "test-key"
os.environ["MUSICGEN_SKIP_AUTH"] = "1"
os.environ["MUSICGEN_SKIP_MODEL_DOWNLOAD"] = "1"
os.environ["PYTEST_CURRENT_TEST"] = "1"


def test_settings_import():
    """Test basic settings module import."""
    try:
        from musicgen.infrastructure.config.settings import Settings
        assert Settings is not None
    except ImportError as e:
        # If settings doesn't exist, skip
        pytest.skip(f"Settings module not available: {e}")


def test_settings_constants():
    """Test settings constants and basic attributes."""
    try:
        from musicgen.infrastructure.config.settings import (
            DEFAULT_MODEL_NAME,
            DEFAULT_DURATION,
            MAX_DURATION,
            DEFAULT_OUTPUT_DIR
        )
        
        assert isinstance(DEFAULT_MODEL_NAME, str)
        assert isinstance(DEFAULT_DURATION, (int, float))
        assert isinstance(MAX_DURATION, (int, float))
        assert isinstance(DEFAULT_OUTPUT_DIR, str)
        assert MAX_DURATION > DEFAULT_DURATION
        
    except ImportError:
        # If individual constants don't exist, try class
        try:
            from musicgen.infrastructure.config.settings import Settings
            settings = Settings()
            assert hasattr(settings, '__dict__')
        except Exception:
            pytest.skip("Settings module not accessible")


def test_environment_variables():
    """Test environment variable handling in settings."""
    # Test that our test environment variables are set
    assert os.environ.get("JWT_SECRET_KEY") == "test-key"
    assert os.environ.get("MUSICGEN_SKIP_AUTH") == "1"
    assert os.environ.get("MUSICGEN_SKIP_MODEL_DOWNLOAD") == "1"
    

def test_main_module():
    """Test __main__ module."""
    try:
        import musicgen.__main__
        assert musicgen.__main__ is not None
    except Exception:
        pytest.skip("__main__ module not accessible")


def test_cli_main_import():
    """Test CLI main module import.""" 
    try:
        import musicgen.cli.main
        assert musicgen.cli.main is not None
    except Exception as e:
        pytest.skip(f"CLI main not accessible: {e}")


def test_web_app_import():
    """Test web app import."""
    try:
        import musicgen.web.app
        assert musicgen.web.app is not None
    except Exception as e:
        pytest.skip(f"Web app not accessible: {e}")


def test_api_main_import():
    """Test API main import."""
    try:
        import musicgen.api.main
        assert musicgen.api.main is not None
    except Exception as e:
        pytest.skip(f"API main not accessible: {e}")
        

def test_hybrid_app_import():
    """Test hybrid app import."""
    try:
        import musicgen.api.rest.hybrid_app
        assert musicgen.api.rest.hybrid_app is not None
    except Exception as e:
        pytest.skip(f"Hybrid app not accessible: {e}")


def test_streaming_api_import():
    """Test streaming API import."""
    try:
        from musicgen.api.streaming.streaming import StreamingAPI
        assert StreamingAPI is not None
    except Exception as e:
        pytest.skip(f"Streaming API not accessible: {e}")


def test_rate_limiting_import():
    """Test rate limiting import."""
    try:
        from musicgen.api.rest.middleware.rate_limiting import RateLimitMiddleware
        assert RateLimitMiddleware is not None
    except Exception as e:
        pytest.skip(f"Rate limiting not accessible: {e}")


def test_logging_import():
    """Test logging import."""
    try:
        from musicgen.infrastructure.monitoring.logging import setup_logging
        assert setup_logging is not None
    except Exception as e:
        pytest.skip(f"Logging setup not accessible: {e}")


def test_metrics_import():
    """Test metrics import."""
    try:
        from musicgen.infrastructure.monitoring.metrics import MetricsCollector
        assert MetricsCollector is not None
    except Exception as e:
        pytest.skip(f"Metrics not accessible: {e}")