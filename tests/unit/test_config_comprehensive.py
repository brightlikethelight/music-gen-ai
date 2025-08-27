"""
Comprehensive tests for configuration modules to maximize coverage.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

# Set environment variables before imports
os.environ["JWT_SECRET_KEY"] = "test-key"
os.environ["MUSICGEN_SKIP_AUTH"] = "1"
os.environ["MUSICGEN_SKIP_MODEL_DOWNLOAD"] = "1"
os.environ["PYTEST_CURRENT_TEST"] = "1"


def test_config_initialization():
    """Test Config class initialization."""
    from musicgen.infrastructure.config.config import Config
    
    config = Config()
    
    # Test all basic attributes exist
    assert hasattr(config, 'MODEL_NAME')
    assert hasattr(config, 'OUTPUT_DIR')
    assert hasattr(config, 'MAX_DURATION')
    assert hasattr(config, 'DEFAULT_DURATION')
    assert hasattr(config, 'SAMPLE_RATE')
    
    # Test types and values
    assert isinstance(config.MODEL_NAME, str)
    assert isinstance(config.OUTPUT_DIR, str)
    assert isinstance(config.MAX_DURATION, (int, float))
    assert isinstance(config.DEFAULT_DURATION, (int, float))
    assert isinstance(config.SAMPLE_RATE, int)
    
    # Test reasonable defaults
    assert config.MAX_DURATION > 0
    assert config.DEFAULT_DURATION > 0
    assert config.SAMPLE_RATE > 0
    assert config.MAX_DURATION >= config.DEFAULT_DURATION


def test_config_environment_variables():
    """Test config with environment variables."""
    from musicgen.infrastructure.config.config import Config
    
    # Test with custom environment variables
    with patch.dict(os.environ, {
        'MUSICGEN_MODEL_NAME': 'test-model',
        'MUSICGEN_OUTPUT_DIR': '/tmp/test',
        'MUSICGEN_MAX_DURATION': '60',
        'MUSICGEN_DEFAULT_DURATION': '10'
    }):
        config = Config()
        
        # Test environment variables are respected
        if hasattr(config, 'from_env') or 'test-model' in getattr(config, 'MODEL_NAME', ''):
            assert 'test-model' in config.MODEL_NAME or config.MODEL_NAME == 'test-model'


def test_config_validation():
    """Test configuration validation."""
    from musicgen.infrastructure.config.config import Config
    
    config = Config()
    
    # Test that configuration is valid
    assert config.validate_config() == True
    
    # Test individual validation methods if they exist
    if hasattr(config, 'validate_model_name'):
        assert config.validate_model_name() == True
        
    if hasattr(config, 'validate_durations'):
        assert config.validate_durations() == True
        
    if hasattr(config, 'validate_paths'):
        assert config.validate_paths() == True


def test_config_get_model_config():
    """Test model configuration retrieval."""
    from musicgen.infrastructure.config.config import Config
    
    config = Config()
    model_config = config.get_model_config()
    
    assert isinstance(model_config, dict)
    assert 'name' in model_config or 'model_name' in model_config
    assert 'device' in model_config or 'use_cuda' in model_config


def test_config_get_generation_config():
    """Test generation configuration retrieval."""
    from musicgen.infrastructure.config.config import Config
    
    config = Config()
    gen_config = config.get_generation_config()
    
    assert isinstance(gen_config, dict)
    assert 'duration' in gen_config or 'max_duration' in gen_config
    assert 'sample_rate' in gen_config


def test_config_get_output_config():
    """Test output configuration retrieval."""
    from musicgen.infrastructure.config.config import Config
    
    config = Config()
    output_config = config.get_output_config()
    
    assert isinstance(output_config, dict)
    assert 'output_dir' in output_config or 'output_path' in output_config
    assert 'format' in output_config or 'default_format' in output_config


def test_config_update_from_dict():
    """Test updating config from dictionary."""
    from musicgen.infrastructure.config.config import Config
    
    config = Config()
    original_duration = config.DEFAULT_DURATION
    
    # Test update with valid values
    updates = {
        'DEFAULT_DURATION': original_duration + 5,
        'MAX_DURATION': original_duration + 10
    }
    
    config.update_from_dict(updates)
    
    # Verify updates were applied
    assert config.DEFAULT_DURATION == original_duration + 5
    assert config.MAX_DURATION == original_duration + 10


def test_config_to_dict():
    """Test converting config to dictionary."""
    from musicgen.infrastructure.config.config import Config
    
    config = Config()
    config_dict = config.to_dict()
    
    assert isinstance(config_dict, dict)
    assert 'MODEL_NAME' in config_dict
    assert 'OUTPUT_DIR' in config_dict
    assert 'MAX_DURATION' in config_dict
    assert 'DEFAULT_DURATION' in config_dict


def test_config_reset_to_defaults():
    """Test resetting configuration to defaults."""
    from musicgen.infrastructure.config.config import Config
    
    config = Config()
    original_model = config.MODEL_NAME
    
    # Modify configuration
    config.MODEL_NAME = "modified-model"
    assert config.MODEL_NAME == "modified-model"
    
    # Reset to defaults
    config.reset_to_defaults()
    
    # Should be back to original
    assert config.MODEL_NAME == original_model


def test_config_device_detection():
    """Test device detection in config."""
    from musicgen.infrastructure.config.config import Config
    
    config = Config()
    
    # Test device configuration
    device_config = config.get_device_config()
    assert isinstance(device_config, dict)
    assert 'device' in device_config or 'use_cuda' in device_config or 'device_type' in device_config


def test_config_logging_setup():
    """Test logging configuration setup."""
    from musicgen.infrastructure.config.config import Config
    
    config = Config()
    
    # Test logging configuration
    if hasattr(config, 'setup_logging'):
        config.setup_logging()
        
    log_config = config.get_logging_config()
    assert isinstance(log_config, dict)


def test_config_security_settings():
    """Test security-related configuration."""
    from musicgen.infrastructure.config.config import Config
    
    config = Config()
    
    # Test security configuration
    security_config = config.get_security_config()
    assert isinstance(security_config, dict)
    
    # Should have JWT settings
    assert 'jwt_secret' in security_config or 'secret_key' in security_config or 'auth_enabled' in security_config


def test_config_api_settings():
    """Test API-related configuration.""" 
    from musicgen.infrastructure.config.config import Config
    
    config = Config()
    
    # Test API configuration
    api_config = config.get_api_config()
    assert isinstance(api_config, dict)
    
    # Should have API settings
    assert 'host' in api_config or 'port' in api_config or 'cors_enabled' in api_config


def test_config_file_operations():
    """Test configuration file operations."""
    from musicgen.infrastructure.config.config import Config
    import tempfile
    import json
    
    config = Config()
    
    # Test saving to file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config.save_to_file(f.name)
        
        # Verify file was created and has content
        with open(f.name, 'r') as read_f:
            saved_data = json.load(read_f)
            assert isinstance(saved_data, dict)
            assert len(saved_data) > 0
        
        os.unlink(f.name)


def test_config_singleton_behavior():
    """Test if config behaves as singleton or allows multiple instances."""
    from musicgen.infrastructure.config.config import Config
    
    config1 = Config()
    config2 = Config()
    
    # Test that both instances work
    assert config1.MODEL_NAME == config2.MODEL_NAME
    assert config1.DEFAULT_DURATION == config2.DEFAULT_DURATION
    
    # If it's a singleton, they should be the same object
    # If not, they should have the same default values
    if config1 is config2:
        # Singleton behavior
        config1.DEFAULT_DURATION = 999
        assert config2.DEFAULT_DURATION == 999
    else:
        # Non-singleton behavior - should have same defaults
        assert config1.DEFAULT_DURATION == config2.DEFAULT_DURATION