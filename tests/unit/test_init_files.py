"""
Test all __init__.py files - these are easy wins for coverage.
"""

import os

import pytest

# Set environment variables before imports
os.environ["JWT_SECRET_KEY"] = "test-key"
os.environ["MUSICGEN_SKIP_REDIS"] = "1"
os.environ["MUSICGEN_SKIP_MODEL_DOWNLOAD"] = "1"
os.environ["PYTEST_CURRENT_TEST"] = "1"

pytestmark = pytest.mark.unit


def test_main_init():
    """Test main package __init__.py"""
    import musicgen

    assert musicgen.__version__ == "2.0.1"

    # Test lazy imports work
    from musicgen import MusicGenerator

    assert MusicGenerator is not None

    from musicgen import BatchProcessor

    assert BatchProcessor is not None

    from musicgen import PromptEngineer

    assert PromptEngineer is not None


def test_api_init():
    """Test api package __init__.py"""
    import musicgen.api

    assert musicgen.api is not None


def test_cli_init():
    """Test cli package __init__.py"""
    import musicgen.cli

    assert musicgen.cli is not None


def test_core_init():
    """Test core package __init__.py"""
    import musicgen.core

    assert musicgen.core is not None


def test_services_init():
    """Test services package __init__.py"""
    import musicgen.services

    assert musicgen.services is not None


def test_utils_init():
    """Test utils package __init__.py"""
    import musicgen.utils

    assert musicgen.utils is not None


def test_infrastructure_init():
    """Test infrastructure package __init__.py"""
    import musicgen.infrastructure

    assert musicgen.infrastructure is not None


def test_monitoring_init():
    """Test monitoring package __init__.py"""
    import musicgen.infrastructure.monitoring

    assert musicgen.infrastructure.monitoring is not None


def test_config_init():
    """Test config package __init__.py"""
    import musicgen.infrastructure.config

    assert musicgen.infrastructure.config is not None


def test_security_init():
    """Test security package __init__.py"""
    import musicgen.infrastructure.security

    assert musicgen.infrastructure.security is not None


def test_web_init():
    """Test web package __init__.py"""
    import musicgen.web

    assert musicgen.web is not None


def test_rest_api_init():
    """Test rest api package __init__.py"""
    import musicgen.api.rest

    assert musicgen.api.rest is not None


def test_middleware_init():
    """Test middleware package __init__.py"""
    import musicgen.api.middleware

    assert musicgen.api.middleware is not None


def test_rest_middleware_init():
    """Test rest middleware package __init__.py"""
    import musicgen.api.rest.middleware

    assert musicgen.api.rest.middleware is not None
