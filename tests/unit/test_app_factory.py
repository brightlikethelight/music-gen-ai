"""
Test app factory and entry point modules.

These tests provide coverage for simple modules with minimal dependencies.
"""

import pytest


class TestAppFactory:
    """Test application factory functionality."""

    def test_create_app_function(self):
        """Test create_app function exists and works."""
        from musicgen.api.app import create_app

        # Test function exists
        assert callable(create_app)

        # Test function returns FastAPI app
        app = create_app()
        assert app is not None
        assert hasattr(app, "title")

    def test_app_export(self):
        """Test default app export."""
        from musicgen.api.app import app

        # Test app is exported
        assert app is not None
        assert hasattr(app, "title")

    def test_api_main_exports(self):
        """Test API main module exports."""
        import musicgen.api.main

        # Test __all__ is properly defined
        assert hasattr(musicgen.api.main, "__all__")
        assert "app" in musicgen.api.main.__all__

        # Test app is accessible
        from musicgen.api.main import app

        assert app is not None


class TestMainEntryPoint:
    """Test main entry point functionality."""

    def test_main_module_exists(self):
        """Test __main__ module can be imported."""
        import musicgen.__main__

        # Test module loads without error
        assert musicgen.__main__ is not None

    def test_main_imports(self):
        """Test main module imports work."""
        # This tests the import statement in __main__.py
        from musicgen.cli.main import main

        # Test main function exists
        assert callable(main)


class TestConfigImports:
    """Test configuration module imports."""

    def test_config_class_import(self):
        """Test Config class can be imported."""
        from musicgen.infrastructure.config.config import Config

        # Test Config class exists
        assert Config is not None
        assert hasattr(Config, "__init__")

    def test_config_instance_creation(self):
        """Test Config instance can be created."""
        from musicgen.infrastructure.config.config import Config

        # Test Config can be instantiated with defaults
        config = Config()
        assert config is not None

    def test_config_environment_methods(self):
        """Test Config environment detection methods."""
        from musicgen.infrastructure.config.config import Config

        config = Config()

        # Test methods exist and return booleans
        assert isinstance(config.is_development(), bool)
        assert isinstance(config.is_production(), bool)
        assert isinstance(config.is_testing(), bool)

    def test_config_log_level_method(self):
        """Test Config log level method."""
        from musicgen.infrastructure.config.config import Config

        config = Config()

        # Test method exists and returns a string
        log_level = config.get_log_level()
        assert isinstance(log_level, str)
        assert log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class TestInfrastructureModules:
    """Test infrastructure module imports."""

    def test_infrastructure_init_import(self):
        """Test infrastructure __init__ module."""
        import musicgen.infrastructure

        # Test module loads without error
        assert musicgen.infrastructure is not None

    def test_config_init_import(self):
        """Test config __init__ module."""
        import musicgen.infrastructure.config

        # Test module loads without error
        assert musicgen.infrastructure.config is not None

    def test_monitoring_init_import(self):
        """Test monitoring __init__ module."""
        import musicgen.infrastructure.monitoring

        # Test module loads without error
        assert musicgen.infrastructure.monitoring is not None

    def test_security_init_import(self):
        """Test security __init__ module."""
        import musicgen.infrastructure.security

        # Test module loads without error
        assert musicgen.infrastructure.security is not None
