"""Simple tests for configuration module to boost coverage."""

import os
os.environ["JWT_SECRET_KEY"] = "test-key"
os.environ["PYTEST_CURRENT_TEST"] = "1"
os.environ["MUSICGEN_SKIP_MODEL_DOWNLOAD"] = "1"

import pytest
from musicgen.infrastructure.config.config import Config


def test_config_attributes():
    """Test Config class attributes exist."""
    # Check key attributes exist
    assert hasattr(Config, 'ENVIRONMENT')
    assert hasattr(Config, 'MODEL_NAME')
    assert hasattr(Config, 'API_HOST')
    assert hasattr(Config, 'API_PORT')
    assert hasattr(Config, 'LOG_LEVEL')
    assert hasattr(Config, 'MAX_DURATION')
    assert hasattr(Config, 'DEFAULT_DURATION')
    

def test_config_validate():
    """Test config validation method."""
    # Should not raise for default config
    assert Config.validate() is True
    

def test_config_get_model_config():
    """Test getting model configuration."""
    model_config = Config.get_model_config()
    assert isinstance(model_config, dict)
    assert 'model_name' in model_config
    assert 'device' in model_config
    

def test_config_get_api_config():
    """Test getting API configuration."""
    api_config = Config.get_api_config()
    assert isinstance(api_config, dict)
    assert 'host' in api_config
    assert 'port' in api_config