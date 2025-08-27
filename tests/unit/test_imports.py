"""Test all module imports for coverage."""

import os
import sys
import pytest

# Set required environment variables for auth module
os.environ["JWT_SECRET_KEY"] = "test-key"
os.environ["PYTEST_CURRENT_TEST"] = "1"
os.environ["MUSICGEN_SKIP_AUTH"] = "1"
os.environ["MUSICGEN_SKIP_MODEL_DOWNLOAD"] = "1"


class TestImports:
    """Test that all modules can be imported."""
    
    def test_main_import(self):
        """Test main package import."""
        import musicgen
        assert musicgen is not None
    
    def test_core_imports(self):
        """Test core module imports."""
        from musicgen import core
        from musicgen.core import generator
        from musicgen.core import prompt
        assert core is not None
        assert generator is not None
        assert prompt is not None
    
    def test_api_imports(self):
        """Test API module imports."""
        from musicgen import api
        from musicgen.api import middleware
        from musicgen.api import rest
        from musicgen.api import streaming
        assert api is not None
        assert middleware is not None
        assert rest is not None
        assert streaming is not None
    
    def test_services_imports(self):
        """Test services module imports."""
        from musicgen import services
        from musicgen.services import batch
        assert services is not None
        assert batch is not None
    
    def test_utils_imports(self):
        """Test utils module imports."""
        from musicgen import utils
        from musicgen.utils import exceptions
        from musicgen.utils import helpers
        assert utils is not None
        assert exceptions is not None
        assert helpers is not None
    
    def test_infrastructure_imports(self):
        """Test infrastructure module imports."""
        from musicgen import infrastructure
        from musicgen.infrastructure import config
        from musicgen.infrastructure import monitoring
        assert infrastructure is not None
        assert config is not None
        assert monitoring is not None
    
    def test_cli_imports(self):
        """Test CLI module imports."""
        from musicgen import cli
        from musicgen.cli import main
        assert cli is not None
        assert main is not None
    
    def test_web_imports(self):
        """Test web module imports."""
        from musicgen import web
        from musicgen.web import app
        assert web is not None
        assert app is not None


class TestMainModule:
    """Test __main__ module functionality."""
    
    def test_main_module_exists(self):
        """Test that __main__ module exists."""
        from musicgen import __main__
        assert __main__ is not None
    
    def test_version_info(self):
        """Test version information."""
        from musicgen import __version__
        assert __version__ is not None
        assert isinstance(__version__, str)


class TestConfigModule:
    """Test configuration module."""
    
    def test_config_import(self):
        """Test config module import."""
        from musicgen.infrastructure.config import config
        assert config is not None
    
    def test_config_defaults(self):
        """Test default configuration values."""
        from musicgen.infrastructure.config.config import Config
        cfg = Config()
        assert cfg.OUTPUT_DIR is not None
        assert cfg.MODEL_CACHE_DIR is not None
        assert cfg.LOG_LEVEL is not None


class TestCORSConfig:
    """Test CORS configuration."""
    
    def test_cors_import(self):
        """Test CORS config import."""
        from musicgen.api import cors_config
        assert cors_config is not None
    
    def test_cors_manager(self):
        """Test CORS manager creation."""
        from musicgen.api.cors_config import CORSConfig
        config = CORSConfig()
        assert config is not None
        assert config.environment in ["development", "staging", "production"]


class TestAPIDocs:
    """Test API documentation."""
    
    def test_api_app_import(self):
        """Test API app module."""
        from musicgen.api import app
        assert app is not None
    
    def test_api_main_import(self):
        """Test API main module."""
        from musicgen.api import main
        assert main is not None


class TestStreamingAPI:
    """Test streaming API imports."""
    
    def test_streaming_components(self):
        """Test streaming components."""
        from musicgen.api.streaming import websocket_endpoint, list_sessions, streaming_manager
        assert websocket_endpoint is not None
        assert list_sessions is not None
        assert streaming_manager is not None


class TestExceptionClasses:
    """Test exception class definitions."""
    
    def test_exception_hierarchy(self):
        """Test exception class hierarchy."""
        from musicgen.utils.exceptions import (
            MusicGenError,
            ModelError,
            GenerationError,
            PromptError,
            AudioError,
            ConfigError,
            ValidationError,
            APIError,
            AuthenticationError,
            AuthorizationError
        )
        
        # Test that all exceptions are subclasses of MusicGenError
        assert issubclass(ModelError, MusicGenError)
        assert issubclass(GenerationError, MusicGenError)
        assert issubclass(PromptError, MusicGenError)
        assert issubclass(AudioError, MusicGenError)
        assert issubclass(ConfigError, MusicGenError)
        assert issubclass(ValidationError, MusicGenError)
        assert issubclass(APIError, MusicGenError)
        assert issubclass(AuthenticationError, MusicGenError)
        assert issubclass(AuthorizationError, MusicGenError)
    
    def test_specific_exceptions(self):
        """Test specific exception classes."""
        from musicgen.utils.exceptions import (
            PromptTooLongError,
            DurationError,
            OutOfMemoryError,
            ModelNotFoundError,
            MP3ConversionError,
            VocalRequestError
        )
        
        # Test specific exceptions
        err1 = PromptTooLongError(100, 50)
        assert "exceeds maximum" in str(err1)
        
        err2 = DurationError(100.0, 30.0)
        assert "exceeds maximum" in str(err2)
        
        err3 = OutOfMemoryError(8.0, 4.0)
        assert "Insufficient memory" in str(err3)
        
        err4 = ModelNotFoundError("test-model")
        assert "test-model" in str(err4)
        
        err5 = MP3ConversionError("test reason")
        assert "test reason" in str(err5)
        
        err6 = VocalRequestError()
        assert "vocals" in str(err6).lower()