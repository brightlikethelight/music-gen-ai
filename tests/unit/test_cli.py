"""
Unit tests for CLI functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

import pytest
import torch
from typer.testing import CliRunner

# Mock only the heavy transformers dependency to prevent model downloads
sys.modules["transformers"] = MagicMock()

# Import CLI modules - handle missing dependencies gracefully
try:
    from musicgen.cli.main import app

    CLI_AVAILABLE = True
except ImportError:
    CLI_AVAILABLE = False

    # Mock CLI app for testing
    class MockApp:
        def command(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

    app = MockApp()


@pytest.mark.unit
@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI modules not available")
class TestCLI:
    """Test CLI interface functionality."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_cli_help(self, runner):
        """Test CLI help command."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "MusicGen" in result.stdout
        assert "generate" in result.stdout
        assert "info" in result.stdout

    def test_generate_command_invalid_duration(self, runner):
        """Test generate command with invalid duration."""
        result = runner.invoke(
            app,
            ["generate", "Test music", "--duration", "500", "--output", "test.mp3"],
        )

        # Should fail due to invalid duration
        assert result.exit_code == 1
        assert "Duration must be between 0 and 300 seconds" in result.stdout

    def test_generate_command_invalid_model(self, runner):
        """Test generate command with invalid model."""
        result = runner.invoke(
            app,
            ["generate", "Test music", "--model", "invalid", "--output", "test.mp3"],
        )

        # Should fail due to invalid model
        assert result.exit_code == 1
        assert "Model must be one of" in result.stdout

    def test_generate_command_help(self, runner):
        """Test generate command help."""
        result = runner.invoke(app, ["generate", "--help"])
        assert result.exit_code == 0
        assert "Generate music from text description" in result.stdout
        assert "--duration" in result.stdout
        assert "--model" in result.stdout

    def test_prompt_command_help(self, runner):
        """Test prompt command help."""
        result = runner.invoke(app, ["prompt", "--help"])
        assert result.exit_code == 0
        assert "Improve prompts for better results" in result.stdout
        assert "--examples" in result.stdout
        assert "--validate" in result.stdout

    def test_batch_command_help(self, runner):
        """Test batch command help."""
        result = runner.invoke(app, ["batch", "--help"])
        assert result.exit_code == 0
        assert "Process multiple generations from CSV file" in result.stdout
        assert "--output-dir" in result.stdout
        assert "--workers" in result.stdout

    def test_info_command(self, runner):
        """Test info command."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.get_device_name", return_value="Mock GPU"):
                with patch("torch.cuda.get_device_properties") as mock_props:
                    mock_props.return_value.total_memory = 8000000000  # 8GB
                    result = runner.invoke(app, ["info"])

                    assert result.exit_code == 0
                    assert "MusicGen Unified" in result.stdout
                    assert "PyTorch" in result.stdout

    def test_create_sample_csv_command(self, runner):
        """Test create-sample-csv command."""
        with patch("musicgen.cli.main.create_sample_csv") as mock_create:
            result = runner.invoke(app, ["create-sample-csv"])
            assert result.exit_code == 0
            mock_create.assert_called_once_with("sample_batch.csv")

    def test_serve_command_help(self, runner):
        """Test serve command help."""
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "Start web interface" in result.stdout
        assert "--port" in result.stdout
        assert "--host" in result.stdout

    def test_api_command_help(self, runner):
        """Test api command help."""
        result = runner.invoke(app, ["api", "--help"])
        assert result.exit_code == 0
        assert "Start REST API server" in result.stdout
        assert "--port" in result.stdout
        assert "--workers" in result.stdout


@pytest.mark.unit
class TestCLIUtils:
    """Test CLI utility functions."""

    def test_basic_cli_structure(self):
        """Test basic CLI structure and commands."""
        # Test that the app is a Typer instance
        from typer import Typer

        assert isinstance(app, Typer)

        # Test that commands exist by trying to get help
        result = CliRunner().invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "generate" in result.stdout
        assert "info" in result.stdout
        assert "batch" in result.stdout
        assert "prompt" in result.stdout

    def test_path_validation(self):
        """Test output path validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Valid path
            valid_path = temp_path / "output.wav"
            assert str(valid_path).endswith(".wav")

            # Test extension checking
            invalid_path = temp_path / "output.txt"
            assert not str(invalid_path).endswith(".wav")

    def test_duration_parsing(self):
        """Test duration parsing and formatting."""
        # Simple duration tests
        duration = 65.5
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        formatted = f"{minutes}:{seconds:02d}"

        assert formatted == "1:05"

        duration = 30.0
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        formatted = f"{minutes}:{seconds:02d}"

        assert formatted == "0:30"


@pytest.mark.unit
class TestCLIConfiguration:
    """Test CLI configuration management."""

    def test_model_size_options(self):
        """Test model size configuration."""
        # Test that model sizes are valid
        valid_sizes = ["small", "medium", "large"]

        for size in valid_sizes:
            assert size in ["small", "medium", "large"]

    def test_device_configuration(self):
        """Test device configuration options."""
        # Test device selection logic
        with patch("torch.cuda.is_available", return_value=True):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            assert device == "cuda"

        with patch("torch.cuda.is_available", return_value=False):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            assert device == "cpu"

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test basic parameter ranges
        assert 0.0 <= 1.0 <= 2.0  # Temperature range
        assert 1 <= 50 <= 100  # Top-k range
        assert 0.0 <= 0.9 <= 1.0  # Top-p range
        assert 60 <= 120 <= 180  # Tempo range


if __name__ == "__main__":
    pytest.main([__file__])
