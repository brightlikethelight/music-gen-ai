"""
Unit tests for musicgen.utils.helpers module.
"""

import os
import logging
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest
import torch

from musicgen.utils.helpers import (
    load_audio, save_audio, get_device, format_time, get_cache_dir,
    hash_text, estimate_memory_usage, crossfade_audio, apply_fade,
    validate_prompt_length, setup_logging, ProgressTracker
)


class TestAudioHelpers:
    """Test audio-related helper functions."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_audio(self):
        """Create sample audio data."""
        sample_rate = 32000
        duration = 1.0
        samples = int(sample_rate * duration)
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples))
        return audio.astype(np.float32), sample_rate

    @patch('musicgen.utils.helpers.librosa')
    def test_load_audio(self, mock_librosa, temp_dir):
        """Test audio loading."""
        # Mock librosa.load
        expected_audio = np.random.randn(32000).astype(np.float32)
        expected_sr = 32000
        mock_librosa.load.return_value = (expected_audio, expected_sr)
        
        # Test loading
        audio_path = temp_dir / "test.wav"
        audio, sr = load_audio(audio_path)
        
        assert np.array_equal(audio, expected_audio)
        assert sr == expected_sr
        mock_librosa.load.assert_called_once_with(audio_path, sr=32000, mono=True)

    @patch('musicgen.utils.helpers.librosa')
    def test_load_audio_custom_params(self, mock_librosa):
        """Test audio loading with custom parameters."""
        mock_librosa.load.return_value = (np.zeros(44100), 44100)
        
        load_audio("test.wav", target_sr=44100, mono=False)
        
        mock_librosa.load.assert_called_with("test.wav", sr=44100, mono=False)

    @patch('musicgen.utils.helpers.sf.write')
    def test_save_audio_numpy(self, mock_write, sample_audio, temp_dir):
        """Test saving numpy audio."""
        audio, sr = sample_audio
        output_path = temp_dir / "output.wav"
        
        result = save_audio(audio, sr, output_path)
        
        assert result == str(output_path)
        mock_write.assert_called_once()
        
        # Check arguments
        call_args = mock_write.call_args
        assert str(output_path) in str(call_args[0][0])
        assert np.array_equal(call_args[0][1], audio)
        assert call_args[0][2] == sr

    @patch('musicgen.utils.helpers.sf.write')
    def test_save_audio_torch(self, mock_write, sample_audio, temp_dir):
        """Test saving torch tensor audio."""
        audio_np, sr = sample_audio
        audio_torch = torch.from_numpy(audio_np)
        output_path = temp_dir / "output.wav"
        
        result = save_audio(audio_torch, sr, output_path)
        
        assert result == str(output_path)
        mock_write.assert_called_once()
        
        # Should convert to numpy
        call_args = mock_write.call_args
        saved_audio = call_args[0][1]
        assert isinstance(saved_audio, np.ndarray)

    @patch('musicgen.utils.helpers.sf.write')
    def test_save_audio_normalization(self, mock_write, temp_dir):
        """Test audio normalization during save."""
        # Create audio that needs normalization
        audio = np.array([2.0, -2.0, 1.5, -1.5], dtype=np.float32)
        sr = 32000
        output_path = temp_dir / "output.wav"
        
        save_audio(audio, sr, output_path)
        
        # Check audio was clipped
        call_args = mock_write.call_args
        saved_audio = call_args[0][1]
        assert saved_audio.max() <= 1.0
        assert saved_audio.min() >= -1.0

    @patch('musicgen.utils.helpers.sf.write')
    @patch('musicgen.utils.helpers.AudioSegment')
    def test_save_audio_mp3(self, mock_segment, mock_write, sample_audio, temp_dir):
        """Test saving audio as MP3."""
        audio, sr = sample_audio
        output_path = temp_dir / "output.mp3"
        
        # Mock AudioSegment
        mock_audio_seg = MagicMock()
        mock_segment.from_wav.return_value = mock_audio_seg
        
        result = save_audio(audio, sr, output_path, format="mp3")
        
        # Should first save as WAV, then convert
        mock_write.assert_called_once()
        mock_segment.from_wav.assert_called_once()
        mock_audio_seg.export.assert_called_once()
        
        assert result == str(output_path)

    @patch('musicgen.utils.helpers.sf.write')
    @patch('musicgen.utils.helpers.AudioSegment')
    def test_save_audio_mp3_fallback(self, mock_segment, mock_write, sample_audio, temp_dir):
        """Test MP3 conversion fallback."""
        audio, sr = sample_audio
        output_path = temp_dir / "output.mp3"
        
        # Mock conversion failure
        mock_segment.from_wav.side_effect = Exception("Conversion failed")
        
        result = save_audio(audio, sr, output_path, format="mp3")
        
        # Should fall back to WAV
        assert result.endswith(".wav")
        mock_write.assert_called()


class TestDeviceHelpers:
    """Test device-related helper functions."""

    def test_get_device_auto_cuda(self):
        """Test automatic CUDA device selection."""
        with patch('torch.cuda.is_available', return_value=True):
            device = get_device()
            assert device == torch.device("cuda")

    def test_get_device_auto_mps(self):
        """Test automatic MPS device selection."""
        with patch('torch.cuda.is_available', return_value=False), \
             patch('musicgen.utils.helpers.torch.backends.mps.is_available', return_value=True):
            device = get_device()
            assert device == torch.device("mps")

    def test_get_device_auto_cpu(self):
        """Test fallback to CPU."""
        with patch('torch.cuda.is_available', return_value=False), \
             patch.object(torch.backends, 'mps', create=True) as mock_mps:
            mock_mps.is_available.return_value = False
            device = get_device()
            assert device == torch.device("cpu")

    def test_get_device_explicit(self):
        """Test explicit device selection."""
        assert get_device("cpu") == torch.device("cpu")
        assert get_device("cuda") == torch.device("cuda")
        assert get_device("cuda:1") == torch.device("cuda:1")


class TestUtilityHelpers:
    """Test utility helper functions."""

    def test_format_time_seconds(self):
        """Test time formatting for seconds."""
        assert format_time(5.5) == "5.5s"
        assert format_time(30) == "30.0s"
        assert format_time(59.9) == "59.9s"

    def test_format_time_minutes(self):
        """Test time formatting for minutes."""
        assert format_time(60) == "1m 0s"
        assert format_time(90) == "1m 30s"
        assert format_time(125) == "2m 5s"

    def test_format_time_hours(self):
        """Test time formatting for hours."""
        assert format_time(3600) == "1h 0m"
        assert format_time(3665) == "1h 1m"
        assert format_time(7200) == "2h 0m"

    def test_get_cache_dir(self):
        """Test cache directory creation."""
        with patch.object(Path, 'home', return_value=Path('/home/user')):
            cache_dir = get_cache_dir()
            assert str(cache_dir) == str(Path('/home/user/.cache/musicgen-unified'))

    def test_get_cache_dir_exists(self):
        """Test cache directory when it already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(Path, 'home', return_value=Path(tmpdir)):
                cache_dir = get_cache_dir()
                assert cache_dir.exists()
                
                # Call again - should not fail
                cache_dir2 = get_cache_dir()
                assert cache_dir == cache_dir2

    def test_hash_text(self):
        """Test text hashing."""
        # Same text should produce same hash
        hash1 = hash_text("test prompt")
        hash2 = hash_text("test prompt")
        assert hash1 == hash2
        
        # Different text should produce different hash
        hash3 = hash_text("different prompt")
        assert hash1 != hash3
        
        # Hash should be 8 characters
        assert len(hash1) == 8

    def test_hash_text_unicode(self):
        """Test hashing with unicode text."""
        hash_unicode = hash_text("音楽 🎵")
        assert len(hash_unicode) == 8


class TestMemoryEstimation:
    """Test memory estimation functions."""

    def test_estimate_memory_usage_small(self):
        """Test memory estimation for small model."""
        estimate = estimate_memory_usage(30, "small")
        
        assert estimate["model_memory_gb"] == 0.5
        assert estimate["generation_memory_gb"] == 0.3  # 30s / 10 * 0.1
        assert estimate["total_memory_gb"] == 0.8
        assert estimate["recommended_gpu_memory_gb"] == 1.2  # 0.8 * 1.5

    def test_estimate_memory_usage_medium(self):
        """Test memory estimation for medium model."""
        estimate = estimate_memory_usage(60, "medium")
        
        assert estimate["model_memory_gb"] == 1.5
        assert estimate["generation_memory_gb"] == 0.6
        assert estimate["total_memory_gb"] == 2.1
        assert estimate["recommended_gpu_memory_gb"] == 3.15

    def test_estimate_memory_usage_large(self):
        """Test memory estimation for large model."""
        estimate = estimate_memory_usage(120, "large")
        
        assert estimate["model_memory_gb"] == 3.5
        assert estimate["generation_memory_gb"] == 1.2
        assert estimate["total_memory_gb"] == 4.7
        assert estimate["recommended_gpu_memory_gb"] == 7.05

    def test_estimate_memory_usage_unknown_model(self):
        """Test memory estimation with unknown model size."""
        estimate = estimate_memory_usage(30, "unknown")
        
        # Should default to small model size
        assert estimate["model_memory_gb"] == 0.5


class TestAudioProcessing:
    """Test audio processing functions."""

    def test_crossfade_audio_basic(self):
        """Test basic audio crossfading."""
        sr = 32000
        audio1 = np.ones(sr * 2)  # 2 seconds
        audio2 = np.ones(sr * 2) * 0.5  # 2 seconds at half amplitude
        
        result = crossfade_audio(audio1, audio2, 0.5, sr)
        
        # Result should be shorter than concatenation
        assert len(result) < len(audio1) + len(audio2)
        assert len(result) == len(audio1) + len(audio2) - int(0.5 * sr)

    def test_crossfade_audio_short_segments(self):
        """Test crossfading with segments too short for overlap."""
        sr = 32000
        audio1 = np.ones(100)  # Very short
        audio2 = np.ones(100)
        
        result = crossfade_audio(audio1, audio2, 1.0, sr)
        
        # Should just concatenate
        assert len(result) == len(audio1) + len(audio2)

    def test_apply_fade_in(self):
        """Test fade-in application."""
        sr = 32000
        audio = np.ones(sr)  # 1 second
        
        result = apply_fade(audio, sr, fade_in=0.1, fade_out=0.0)
        
        # Check fade-in was applied
        assert result[0] < result[-1]  # Start quieter than end
        assert result[0] < 0.1  # Should start near zero
        assert result[-1] == 1.0  # End unchanged

    def test_apply_fade_out(self):
        """Test fade-out application."""
        sr = 32000
        audio = np.ones(sr)  # 1 second
        
        result = apply_fade(audio, sr, fade_in=0.0, fade_out=0.1)
        
        # Check fade-out was applied
        assert result[0] > result[-1]  # Start louder than end
        assert result[0] == 1.0  # Start unchanged
        assert result[-1] < 0.1  # Should end near zero

    def test_apply_fade_both(self):
        """Test both fade-in and fade-out."""
        sr = 32000
        audio = np.ones(sr)  # 1 second
        
        result = apply_fade(audio, sr, fade_in=0.1, fade_out=0.1)
        
        # Check both fades
        assert result[0] < 0.1  # Fade in
        assert result[-1] < 0.1  # Fade out
        assert result[sr // 2] == 1.0  # Middle unchanged


class TestPromptValidation:
    """Test prompt validation functions."""

    def test_validate_prompt_length_valid(self):
        """Test validation of valid prompts."""
        prompt = "This is a valid prompt"
        result = validate_prompt_length(prompt)
        assert result == prompt

    def test_validate_prompt_length_truncate(self):
        """Test prompt truncation."""
        long_prompt = "a" * 300
        result = validate_prompt_length(long_prompt, max_length=256)
        
        assert len(result) <= 256 + 3  # +3 for "..."
        assert result.endswith("...")

    def test_validate_prompt_length_truncate_word_boundary(self):
        """Test truncation at word boundary."""
        prompt = "This is a very long prompt " * 20
        result = validate_prompt_length(prompt, max_length=50)
        
        assert len(result) <= 53  # 50 + "..."
        assert not result[:-3].endswith(" a")  # Should break at word

    def test_validate_prompt_length_strip(self):
        """Test prompt stripping."""
        prompt = "  prompt with spaces  "
        result = validate_prompt_length(prompt)
        assert result == "prompt with spaces"


class TestLoggingSetup:
    """Test logging setup."""

    @patch('logging.basicConfig')
    def test_setup_logging_default(self, mock_config):
        """Test default logging setup."""
        setup_logging()
        
        mock_config.assert_called_once()
        call_args = mock_config.call_args[1]
        assert call_args['level'] == logging.INFO

    @patch('logging.basicConfig')
    def test_setup_logging_custom_level(self, mock_config):
        """Test custom log level."""
        setup_logging(level="DEBUG")
        
        call_args = mock_config.call_args[1]
        assert call_args['level'] == logging.DEBUG

    @patch('logging.basicConfig')
    def test_setup_logging_custom_format(self, mock_config):
        """Test custom log format."""
        custom_format = "%(levelname)s: %(message)s"
        setup_logging(format=custom_format)
        
        call_args = mock_config.call_args[1]
        assert call_args['format'] == custom_format


class TestProgressTracker:
    """Test ProgressTracker class."""

    def test_progress_tracker_init(self):
        """Test ProgressTracker initialization."""
        tracker = ProgressTracker(100, "Processing")
        
        assert tracker.total == 100
        assert tracker.current == 0
        assert tracker.description == "Processing"

    def test_progress_tracker_update(self):
        """Test progress updates."""
        tracker = ProgressTracker(100)
        
        tracker.update(10)
        assert tracker.current == 10
        
        tracker.update(5)
        assert tracker.current == 15

    def test_progress_tracker_get_progress(self):
        """Test getting progress information."""
        tracker = ProgressTracker(100, "Test")
        tracker.update(25)
        
        progress = tracker.get_progress()
        
        assert progress["current"] == 25
        assert progress["total"] == 100
        assert progress["percent"] == 25.0
        assert progress["description"] == "Test"
        assert "elapsed" in progress
        assert "remaining" in progress

    def test_progress_tracker_zero_total(self):
        """Test progress tracker with zero total."""
        tracker = ProgressTracker(0)
        progress = tracker.get_progress()
        
        assert progress["percent"] == 0
        assert progress["remaining"] == 0

    @patch('time.time')
    def test_progress_tracker_time_estimation(self, mock_time):
        """Test time estimation."""
        # Mock time progression
        mock_time.side_effect = [0, 10]  # Start at 0, then 10 seconds later
        
        tracker = ProgressTracker(100)
        tracker.update(50)  # 50% complete after 10 seconds
        
        progress = tracker.get_progress()
        
        assert progress["elapsed"] == 10
        assert progress["remaining"] == pytest.approx(10, rel=0.1)  # Should estimate 10 more seconds