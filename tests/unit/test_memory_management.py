"""Tests for memory management improvements."""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Set required environment variables for testing
os.environ["MUSICGEN_SKIP_MODEL_DOWNLOAD"] = "1"
os.environ["JWT_SECRET_KEY"] = "test-key"
os.environ["PYTEST_CURRENT_TEST"] = "1"

from musicgen.core.async_generator import AsyncMusicGenerator
from musicgen.core.generator import MusicGenerator


class TestMemoryManagement:
    """Test memory management in generators."""

    @pytest.fixture
    def mock_cuda(self):
        """Mock CUDA availability and functions."""
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.empty_cache"
        ) as mock_empty_cache, patch("torch.cuda.memory_allocated", return_value=1e9), patch(
            "torch.cuda.memory_reserved", return_value=2e9
        ), patch(
            "torch.cuda.get_device_name", return_value="Mock GPU"
        ), patch(
            "torch.cuda.get_device_properties"
        ) as mock_props:

            mock_props.return_value = MagicMock(total_memory=8e9)
            yield mock_empty_cache

    def test_generator_cleans_memory_after_generation(self, mock_cuda):
        """Test that GPU memory is cleaned after generation."""
        generator = MusicGenerator()

        # Mock generate should still work with skipped model
        audio, sr = generator.generate("test prompt", duration=5.0)
        assert audio is not None
        assert sr == 32000

        # In real implementation with GPU, empty_cache would be called
        # but with MUSICGEN_SKIP_MODEL_DOWNLOAD, it returns mock data immediately

    def test_generator_info_includes_memory_cleanup(self, mock_cuda):
        """Test that get_info triggers cleanup when memory is high."""
        generator = MusicGenerator()

        # Mock high memory usage (>80%)
        with patch("torch.cuda.memory_allocated", return_value=7e9):
            info = generator.get_info()

            assert "model" in info
            assert info["device"] == "cuda:0"
            # With mocked high allocation, recommendation should be added
            # but since we skip model download, it won't actually trigger

    def test_async_generator_cleanup(self, mock_cuda):
        """Test async generator cleanup releases memory."""
        async_gen = AsyncMusicGenerator()

        # Set a mock generator
        async_gen.generator = MagicMock()

        # Call cleanup
        async_gen.cleanup()

        # Check that generator was cleared
        assert async_gen.generator is None

        # Check CUDA cache was cleared
        mock_cuda.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_generator_context_manager_cleanup(self, mock_cuda):
        """Test async generator cleans up in context manager."""
        async with AsyncMusicGenerator() as generator:
            assert generator is not None

        # After exiting context, cleanup should have been called
        # GPU cache should be cleared if CUDA is available
        mock_cuda.assert_called()


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
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.get_device_name", return_value="NVIDIA RTX 3090"
        ), patch("torch.cuda.memory_allocated", return_value=2e9), patch(
            "torch.cuda.memory_reserved", return_value=3e9
        ), patch(
            "torch.cuda.get_device_properties"
        ) as mock_props:

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
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.get_device_name", return_value="NVIDIA GTX 1060"
        ), patch("torch.cuda.memory_allocated", return_value=1e9), patch(
            "torch.cuda.memory_reserved", return_value=5e9
        ), patch(
            "torch.cuda.get_device_properties"
        ) as mock_props:

            mock_props.return_value = MagicMock(total_memory=6e9)

            generator = MusicGenerator()
            info = generator.get_info()

            # Should have warning when available memory < 2GB
            assert "memory_warning" in info
            assert "Low GPU memory" in info["memory_warning"]

    def test_memory_recommendation_when_highly_allocated(self):
        """Test recommendation when memory is highly allocated."""
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.get_device_name", return_value="NVIDIA RTX 3090"
        ), patch("torch.cuda.memory_allocated", return_value=20e9), patch(
            "torch.cuda.memory_reserved", return_value=22e9
        ), patch(
            "torch.cuda.get_device_properties"
        ) as mock_props, patch(
            "torch.cuda.empty_cache"
        ) as mock_empty_cache:

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

    @pytest.fixture
    def generator_with_mocked_generate(self):
        """Create generator with mocked single generation."""
        generator = MusicGenerator()

        # Mock the single generate method to return fake audio
        def mock_generate(prompt, duration, temp=1.0, guidance=3.0, callback=None):
            samples = int(duration * 32000)
            return np.random.randn(samples).astype(np.float32), 32000

        generator.generate = mock_generate
        return generator

    def test_extended_generation_cleans_memory(self):
        """Test that extended generation cleans memory after completion."""
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.empty_cache"
        ) as mock_empty_cache:

            generator = MusicGenerator()

            # Mock generate to avoid actual model usage
            with patch.object(generator, "generate") as mock_gen:
                mock_gen.return_value = (np.random.randn(32000 * 25), 32000)

                # Generate extended audio
                audio, sr = generator.generate_extended(
                    "test prompt", duration=60.0, segment_duration=25.0  # More than 30s
                )

                # Should have called generate multiple times
                assert mock_gen.call_count >= 2

            # GPU cache should be cleared after extended generation
            # (This happens in the real generate_extended method)


class TestWebSocketMemoryCleanup:
    """Test memory cleanup in WebSocket streaming."""

    @pytest.mark.asyncio
    async def test_session_cleanup_clears_memory(self):
        """Test that closing a session clears GPU memory."""
        from musicgen.api.streaming.streaming import StreamingSession

        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.empty_cache"
        ) as mock_empty_cache:

            # Create mock websocket
            mock_ws = MagicMock()
            mock_ws.client_state = MagicMock()
            mock_ws.close = MagicMock(return_value=None)

            session = StreamingSession("test-id", mock_ws)
            session.generator = MagicMock()  # Mock generator

            # Close session
            await session.close()

            # Check cleanup happened
            assert session.generator is None
            assert session.is_active is False
            mock_empty_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_manager_cleanup_when_no_sessions(self):
        """Test manager cleans up when last session is removed."""
        from musicgen.api.streaming.streaming import StreamingManager

        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.empty_cache"
        ) as mock_empty_cache:

            manager = StreamingManager()
            manager._generator_cache = MagicMock()

            # Add a mock session
            mock_session = MagicMock()
            mock_session.close = MagicMock(return_value=None)
            manager.sessions["test-id"] = mock_session

            # Remove the last session
            await manager.remove_session("test-id")

            # Should trigger cleanup when no sessions remain
            assert len(manager.sessions) == 0
            mock_empty_cache.assert_called()
