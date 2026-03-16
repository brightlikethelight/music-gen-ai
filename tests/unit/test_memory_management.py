"""Tests for memory management improvements."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from musicgen.core.generator import MusicGenerator


class TestMemoryManagement:
    """Test memory management in generators."""

    @pytest.fixture
    def mock_cuda(self):
        """Mock CUDA availability and functions."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.empty_cache") as mock_empty_cache,
            patch("torch.cuda.memory_allocated", return_value=1e9),
            patch("torch.cuda.memory_reserved", return_value=2e9),
            patch("torch.cuda.get_device_name", return_value="Mock GPU"),
            patch("torch.cuda.get_device_properties") as mock_props,
        ):

            mock_props.return_value = MagicMock(total_memory=8e9)
            yield mock_empty_cache

    def test_generator_cleans_memory_after_generation(self, mock_cuda):
        """Test that GPU memory is cleaned after generation."""
        generator = MusicGenerator()

        # Mock generate should still work with skipped model
        audio, sr = generator.generate("test prompt", duration=5.0)
        assert audio is not None
        assert sr == 32000

    def test_generator_info_includes_memory_cleanup(self, mock_cuda):
        """Test that get_info triggers cleanup when memory is high."""
        generator = MusicGenerator()

        # Mock high memory usage (>80%)
        with patch("torch.cuda.memory_allocated", return_value=7e9):
            info = generator.get_info()

            assert "model" in info
            assert info["device"] == "cuda:0"


class TestGeneratorMemoryInfo:
    """Test memory information reporting."""

    def test_memory_info_without_gpu(self):
        """Test memory info when GPU is not available."""
        with patch("torch.cuda.is_available", return_value=False):
            generator = MusicGenerator()
            info = generator.get_info()

            assert "gpu" not in info
            assert "gpu_memory" not in info
            assert info["device"] == "cpu"

    def test_memory_info_with_gpu(self):
        """Test memory info includes GPU details."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 3090"),
            patch("torch.cuda.memory_allocated", return_value=2e9),
            patch("torch.cuda.memory_reserved", return_value=3e9),
            patch("torch.cuda.get_device_properties") as mock_props,
            patch("torch.cuda.device_count", return_value=1),
            patch("torch.cuda.set_device"),
            patch("torch.cuda.mem_get_info", return_value=(8e9, 8e9)),
        ):

            mock_props.return_value = MagicMock(total_memory=24e9)

            generator = MusicGenerator()
            info = generator.get_info()

            assert info["gpu"] == "NVIDIA RTX 3090"
            assert "gpu_memory" in info
            assert info["gpu_memory"]["total"] == "24.0 GB"
            assert info["gpu_memory"]["allocated"] == "2.0 GB"
            assert info["gpu_memory"]["reserved"] == "3.0 GB"
            assert info["gpu_memory"]["available"] == "21.0 GB"

    def test_memory_warning_when_low(self):
        """Test warning when GPU memory is low."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_name", return_value="NVIDIA GTX 1060"),
            patch("torch.cuda.memory_allocated", return_value=1e9),
            patch("torch.cuda.memory_reserved", return_value=5e9),
            patch("torch.cuda.get_device_properties") as mock_props,
            patch("torch.cuda.device_count", return_value=1),
            patch("torch.cuda.set_device"),
            patch("torch.cuda.mem_get_info", return_value=(8e9, 8e9)),
        ):

            mock_props.return_value = MagicMock(total_memory=6e9)

            generator = MusicGenerator()
            info = generator.get_info()

            # Should have warning when available memory < 2GB
            assert "memory_warning" in info
            assert "Low GPU memory" in info["memory_warning"]

    def test_memory_recommendation_when_highly_allocated(self):
        """Test recommendation when memory is highly allocated."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 3090"),
            patch("torch.cuda.memory_allocated", return_value=20e9),
            patch("torch.cuda.memory_reserved", return_value=22e9),
            patch("torch.cuda.get_device_properties") as mock_props,
            patch("torch.cuda.empty_cache") as mock_empty_cache,
            patch("torch.cuda.device_count", return_value=1),
            patch("torch.cuda.set_device"),
            patch("torch.cuda.mem_get_info", return_value=(8e9, 8e9)),
        ):

            mock_props.return_value = MagicMock(total_memory=24e9)

            generator = MusicGenerator()
            info = generator.get_info()

            # Should have recommendation when allocated > 80% of total
            assert "memory_recommendation" in info
            assert "High GPU memory usage" in info["memory_recommendation"]

            # Should trigger cleanup
            mock_empty_cache.assert_called_once()


class TestExtendedGenerationMemory:
    """Test memory management in extended generation."""

    def test_extended_generation_cleans_memory(self):
        """Test that extended generation cleans memory after completion."""
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.empty_cache"):

            generator = MusicGenerator()

            # Mock generate to avoid actual model usage
            with patch.object(generator, "generate") as mock_gen:
                mock_gen.return_value = (np.random.randn(32000 * 25), 32000)

                # Generate extended audio
                audio, sr = generator.generate_extended(
                    "test prompt", duration=60.0, segment_duration=25.0
                )

                # Should have called generate multiple times
                assert mock_gen.call_count >= 2
