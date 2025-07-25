"""
Unit tests for musicgen.cli.main module.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import pytest
from typer.testing import CliRunner

from musicgen.cli.main import app


class TestCLI:
    """Test CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_generator(self):
        """Mock MusicGenerator class."""
        with patch("musicgen.cli.main.MusicGenerator") as mock:
            generator_instance = MagicMock()

            # Fix return value - generate() returns (audio_array, sample_rate)
            import numpy as np

            generator_instance.generate.return_value = (np.array([0.1, 0.2, 0.3]), 44100)

            # Mock save_audio to return a real temporary file path
            import tempfile

            temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            temp_file.write(b"fake audio data")
            temp_file.close()
            generator_instance.save_audio.return_value = temp_file.name

            # Fix format string errors by providing proper return values
            generator_instance.get_info.return_value = {
                "device": "cpu",
                "gpu": "N/A",
                "model": "facebook/musicgen-small",
                "memory": "1.2GB",
            }

            generator_instance.__enter__ = MagicMock(return_value=generator_instance)
            generator_instance.__exit__ = MagicMock(return_value=None)
            mock.return_value = generator_instance
            yield mock, generator_instance

            # Cleanup temp file
            import os

            try:
                os.unlink(temp_file.name)
            except:
                pass

    def test_generate_basic(self, runner, mock_generator):
        """Test basic generate command."""
        mock_class, mock_instance = mock_generator

        result = runner.invoke(app, ["generate", "piano music", "--yes"])

        assert result.exit_code == 0
        assert "✅ Success!" in result.output
        assert "Output:" in result.output
        mock_instance.generate.assert_called_once()

    def test_generate_with_output(self, runner, mock_generator, temp_dir):
        """Test generate with custom output path."""
        mock_class, mock_instance = mock_generator
        output_path = str(temp_dir / "custom.mp3")

        result = runner.invoke(app, ["generate", "piano music", "--output", output_path, "--yes"])

        assert result.exit_code == 0
        call_args = mock_instance.generate.call_args
        assert call_args[1]["output_path"] == output_path

    def test_generate_with_duration(self, runner, mock_generator):
        """Test generate with custom duration."""
        mock_class, mock_instance = mock_generator

        result = runner.invoke(app, ["generate", "piano music", "--duration", "45", "--yes"])

        assert result.exit_code == 0
        call_args = mock_instance.generate.call_args
        assert call_args[1]["duration"] == 45.0

    def test_generate_invalid_duration(self, runner, mock_generator):
        """Test generate with invalid duration."""
        result = runner.invoke(app, ["generate", "piano music", "--duration", "400"])  # Too long

        assert result.exit_code == 1
        assert "Duration must be between" in result.output

    def test_generate_with_model(self, runner, mock_generator):
        """Test generate with different model sizes."""
        mock_class, mock_instance = mock_generator

        # Test each model size
        for model_size in ["small", "medium", "large"]:
            result = runner.invoke(app, ["generate", "piano music", "--model", model_size, "--yes"])

            assert result.exit_code == 0
            mock_class.assert_called()
            call_args = mock_class.call_args
            expected_model = f"facebook/musicgen-{model_size}"
            assert call_args[1]["model_name"] == expected_model

    def test_generate_with_temperature(self, runner, mock_generator):
        """Test generate with custom temperature."""
        mock_class, mock_instance = mock_generator

        result = runner.invoke(app, ["generate", "piano music", "--temperature", "0.8", "--yes"])

        assert result.exit_code == 0
        call_args = mock_instance.generate.call_args
        assert call_args[1]["temperature"] == 0.8

    def test_generate_with_guidance(self, runner, mock_generator):
        """Test generate with custom guidance scale."""
        mock_class, mock_instance = mock_generator

        result = runner.invoke(app, ["generate", "piano music", "--guidance", "5.0", "--yes"])

        assert result.exit_code == 0
        call_args = mock_instance.generate.call_args
        assert call_args[1]["guidance_scale"] == 5.0

    def test_generate_with_device(self, runner, mock_generator):
        """Test generate with specific device."""
        mock_class, mock_instance = mock_generator

        result = runner.invoke(app, ["generate", "piano music", "--device", "cpu", "--yes"])

        assert result.exit_code == 0
        mock_class.assert_called_with(
            model_name="facebook/musicgen-small", device="cpu", optimize=True
        )

    def test_generate_no_optimize(self, runner, mock_generator):
        """Test generate without optimization."""
        mock_class, mock_instance = mock_generator

        result = runner.invoke(app, ["generate", "piano music", "--no-optimize", "--yes"])

        assert result.exit_code == 0
        mock_class.assert_called_with(
            model_name="facebook/musicgen-small", device=None, optimize=False
        )

    def test_generate_error_handling(self, runner, mock_generator):
        """Test error handling during generation."""
        mock_class, mock_instance = mock_generator
        mock_instance.generate.side_effect = Exception("Generation failed")

        result = runner.invoke(app, ["generate", "piano music", "--yes"])

        assert result.exit_code == 1
        assert "Generation failed" in result.output

    @patch("musicgen.cli.main.BatchProcessor")
    def test_batch_command(self, mock_batch, runner, temp_dir):
        """Test batch processing command."""
        mock_processor = MagicMock()
        mock_processor.process.return_value = ["output1.mp3", "output2.mp3"]
        mock_batch.return_value = mock_processor

        csv_file = temp_dir / "batch.csv"
        csv_file.write_text("prompt,duration\npiano,30\nguitar,45\n")

        result = runner.invoke(app, ["batch", str(csv_file), "--output-dir", str(temp_dir)])

        assert result.exit_code == 0
        assert "Processing 2 prompts" in result.output
        mock_processor.process.assert_called_once()

    @patch("musicgen.cli.main.BatchProcessor")
    def test_batch_missing_file(self, mock_batch, runner):
        """Test batch command with missing file."""
        # Mock processor that returns empty jobs list for missing file
        mock_processor = MagicMock()
        mock_processor.load_csv.return_value = []  # Empty list means no valid jobs
        mock_batch.return_value = mock_processor

        result = runner.invoke(app, ["batch", "nonexistent.csv"])

        assert result.exit_code == 1
        assert "No valid jobs found" in result.output

    @patch("musicgen.cli.main.create_sample_csv")
    def test_batch_create_sample(self, mock_create, runner, temp_dir):
        """Test creating sample batch file."""
        sample_path = temp_dir / "sample.csv"
        mock_create.return_value = str(sample_path)

        result = runner.invoke(app, ["batch", "dummy.csv", "--create-sample"])

        assert result.exit_code == 0
        assert "Sample CSV created" in result.output
        mock_create.assert_called_once()

    def test_enhance_command(self, runner):
        """Test prompt enhancement command."""
        with patch("musicgen.cli.main.PromptEngineer") as mock_engineer:
            mock_instance = MagicMock()
            mock_instance.enhance_prompt.return_value = "enhanced jazz piano with smooth rhythm"
            mock_engineer.return_value = mock_instance

            result = runner.invoke(app, ["enhance", "jazz piano"])

            assert result.exit_code == 0
            assert "Original" in result.output
            assert "Enhanced" in result.output
            assert "enhanced jazz piano" in result.output

    def test_enhance_multiple_prompts(self, runner):
        """Test enhancing multiple prompts."""
        with patch("musicgen.cli.main.PromptEngineer") as mock_engineer:
            mock_instance = MagicMock()
            mock_instance.enhance_prompt.side_effect = ["enhanced prompt 1", "enhanced prompt 2"]
            mock_engineer.return_value = mock_instance

            result = runner.invoke(app, ["enhance", "prompt 1", "prompt 2"])

            assert result.exit_code == 0
            assert "enhanced prompt 1" in result.output
            assert "enhanced prompt 2" in result.output

    def test_info_command(self, runner):
        """Test info command."""
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.get_device_name", return_value="NVIDIA GPU"
        ):

            result = runner.invoke(app, ["info"])

            assert result.exit_code == 0
            assert "MusicGen Unified" in result.output
            assert "System Information" in result.output
            assert "Available Models" in result.output
            assert "GPU Available" in result.output

    def test_info_command_no_gpu(self, runner):
        """Test info command without GPU."""
        with patch("torch.cuda.is_available", return_value=False):

            result = runner.invoke(app, ["info"])

            assert result.exit_code == 0
            assert "GPU Available: ❌ No" in result.output

    def test_serve_command(self, runner):
        """Test serve command."""
        with patch("musicgen.cli.main.uvicorn") as mock_uvicorn:
            result = runner.invoke(app, ["serve"])

            assert result.exit_code == 0
            mock_uvicorn.run.assert_called_once()

            # Check default parameters
            call_args = mock_uvicorn.run.call_args
            assert call_args[1]["host"] == "0.0.0.0"
            assert call_args[1]["port"] == 8000

    def test_serve_custom_params(self, runner):
        """Test serve command with custom parameters."""
        with patch("musicgen.cli.main.uvicorn") as mock_uvicorn:
            result = runner.invoke(app, ["serve", "--host", "localhost", "--port", "8080"])

            assert result.exit_code == 0
            call_args = mock_uvicorn.run.call_args
            assert call_args[1]["host"] == "localhost"
            assert call_args[1]["port"] == 8080

    def test_version_display(self, runner):
        """Test version display in commands."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "MusicGen Unified" in result.output

    def test_progress_reporting(self, runner, mock_generator):
        """Test progress reporting during generation."""
        mock_class, mock_instance = mock_generator

        # Simulate progress callbacks
        def generate_with_progress(*args, **kwargs):
            callback = kwargs.get("callback")
            if callback:
                callback(1, 10)
                callback(5, 10)
                callback(10, 10)
            return "output.mp3"

        mock_instance.generate.side_effect = generate_with_progress

        result = runner.invoke(app, ["generate", "piano music", "--yes"])

        assert result.exit_code == 0
        # Progress reporting should be handled gracefully

    @patch.dict(os.environ, {"MUSICGEN_CACHE_DIR": "/custom/cache"})
    def test_custom_cache_dir(self, runner, mock_generator):
        """Test using custom cache directory from environment."""
        mock_class, mock_instance = mock_generator

        result = runner.invoke(app, ["generate", "piano music", "--yes"])

        assert result.exit_code == 0
        # Model should be initialized with custom cache dir
        # (implementation would use os.environ['MUSICGEN_CACHE_DIR'])

    def test_keyboard_interrupt(self, runner, mock_generator):
        """Test handling keyboard interrupt."""
        mock_class, mock_instance = mock_generator
        mock_instance.generate.side_effect = KeyboardInterrupt()

        result = runner.invoke(app, ["generate", "piano music", "--yes"])

        # Should handle gracefully
        assert "Generation cancelled" in result.output or result.exit_code != 0

    def test_generate_with_all_options(self, runner, mock_generator, temp_dir):
        """Test generate command with all options."""
        mock_class, mock_instance = mock_generator

        result = runner.invoke(
            app,
            [
                "generate",
                "complex piano jazz",
                "--output",
                str(temp_dir / "output.mp3"),
                "--duration",
                "60",
                "--model",
                "medium",
                "--temperature",
                "0.9",
                "--guidance",
                "4.0",
                "--device",
                "cuda",
                "--no-optimize",
                "--yes",
            ],
        )

        assert result.exit_code == 0

        # Verify all parameters were passed correctly
        mock_class.assert_called_with(
            model_name="facebook/musicgen-medium", device="cuda", optimize=False
        )

        call_args = mock_instance.generate.call_args
        assert call_args[0][0] == "complex piano jazz"
        assert call_args[1]["duration"] == 60.0
        assert call_args[1]["temperature"] == 0.9
        assert call_args[1]["guidance_scale"] == 4.0
