"""
Unit tests for musicgen.core.generator module.
Tests both skip-mode (env var set) and mock-mode (mocks replacing transformers).
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from musicgen.core.generator import MusicGenerator


class TestMusicGeneratorSkipMode:
    """Tests with MUSICGEN_SKIP_MODEL_DOWNLOAD set (no real model loading)."""

    def test_init_skip_mode(self):
        """Generator initializes with model=None when skip env is set."""
        generator = MusicGenerator()
        assert generator.model is None
        assert generator.processor is None
        assert generator.model_name == "facebook/musicgen-small"

    def test_generate_returns_mock_audio(self):
        """Generate returns random audio in skip mode."""
        generator = MusicGenerator()
        audio, sr = generator.generate("test music", duration=5.0)
        assert isinstance(audio, np.ndarray)
        assert sr == 32000
        assert len(audio) == int(5.0 * 32000)

    def test_generate_empty_prompt_raises(self):
        """Empty prompts raise ValueError regardless of mode."""
        generator = MusicGenerator()
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            generator.generate("")
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            generator.generate("   ")

    def test_get_info_skip_mode(self):
        """get_info returns basic info in skip mode."""
        generator = MusicGenerator()
        info = generator.get_info()
        assert info["model"] == "facebook/musicgen-small"
        assert info["sample_rate"] == 32000

    def test_save_audio_wav(self, tmp_path):
        """save_audio saves WAV files."""
        generator = MusicGenerator()
        audio = np.random.randn(32000).astype(np.float32)
        with patch("musicgen.core.generator.sf") as mock_sf:
            mock_sf.write = MagicMock()
            path = generator.save_audio(audio, 32000, str(tmp_path / "test.wav"))
            assert path.endswith(".wav")
            mock_sf.write.assert_called_once()

    def test_save_audio_mp3_with_pydub(self, tmp_path):
        """save_audio converts to MP3 when pydub is available."""
        generator = MusicGenerator()
        audio = np.random.randn(32000).astype(np.float32)
        with (
            patch("musicgen.core.generator.PYDUB_AVAILABLE", True),
            patch("musicgen.core.generator.AudioSegment") as mock_segment,
            patch("musicgen.core.generator.sf") as mock_sf,
            patch("os.remove"),
        ):
            mock_sf.write = MagicMock()
            mock_audio_segment = MagicMock()
            mock_segment.from_wav.return_value = mock_audio_segment
            path = generator.save_audio(audio, 32000, str(tmp_path / "test.mp3"))
            assert path.endswith(".mp3")

    def test_save_audio_fallback_without_soundfile(self, tmp_path):
        """save_audio falls back to scipy when soundfile unavailable."""
        import musicgen.core.generator as gen_module

        generator = MusicGenerator()
        audio = np.random.randn(32000).astype(np.float32)
        mock_wavfile = MagicMock()
        with (
            patch("musicgen.core.generator.SOUNDFILE_AVAILABLE", False),
            patch.object(gen_module, "wavfile", mock_wavfile, create=True),
        ):
            generator.save_audio(audio, 32000, str(tmp_path / "test.wav"))
            mock_wavfile.write.assert_called_once()


class _MockBatchEncoding(dict):
    """Dict subclass that supports .to() like HuggingFace BatchEncoding."""

    def to(self, device):
        return self


class TestMusicGeneratorMockMode:
    """Tests with mocked transformers (env var removed, mocks injected)."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self, monkeypatch):
        """Remove skip env var and inject mock transformers."""
        monkeypatch.delenv("MUSICGEN_SKIP_MODEL_DOWNLOAD", raising=False)

        # Create mock processor that returns BatchEncoding-like object
        self.mock_processor_instance = MagicMock()
        self.mock_processor_instance.return_value = _MockBatchEncoding(
            {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }
        )

        self.mock_model_instance = MagicMock()
        self.mock_model_instance.to = MagicMock(return_value=self.mock_model_instance)
        self.mock_model_instance.config.audio_encoder.sampling_rate = 32000
        self.mock_model_instance.generate = MagicMock(return_value=torch.randn(1, 1, 32000))

        # Create mock classes
        self.mock_processor_class = MagicMock()
        self.mock_processor_class.from_pretrained = MagicMock(
            return_value=self.mock_processor_instance
        )
        self.mock_model_class = MagicMock()
        self.mock_model_class.from_pretrained = MagicMock(return_value=self.mock_model_instance)

        # Patch module globals
        monkeypatch.setattr("musicgen.core.generator.AutoProcessor", self.mock_processor_class)
        monkeypatch.setattr(
            "musicgen.core.generator.MusicgenForConditionalGeneration",
            self.mock_model_class,
        )

        # Mock soundfile
        mock_sf = MagicMock()
        mock_sf.write = MagicMock()
        monkeypatch.setattr("musicgen.core.generator.sf", mock_sf, raising=False)
        monkeypatch.setattr("musicgen.core.generator.SOUNDFILE_AVAILABLE", True)

    def test_init_loads_model(self):
        """Model and processor are loaded during init."""
        MusicGenerator(model_name="facebook/musicgen-small")
        self.mock_processor_class.from_pretrained.assert_called_once()
        self.mock_model_class.from_pretrained.assert_called_once()

    def test_init_custom_device(self):
        """Custom device is used when specified."""
        generator = MusicGenerator(device="cpu")
        assert generator.device == torch.device("cpu")

    def test_init_cuda_device(self):
        """CUDA device selected when available."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=1),
            patch("torch.cuda.set_device"),
            patch("torch.cuda.mem_get_info", return_value=(8e9, 8e9)),
        ):
            generator = MusicGenerator()
            assert generator.device.type == "cuda"

    def test_model_and_processor_set(self):
        """Model and processor are not None after init."""
        generator = MusicGenerator()
        assert generator.model is not None
        assert generator.processor is not None

    def test_generate_basic(self):
        """Basic generation returns audio array."""
        generator = MusicGenerator()
        audio, sr = generator.generate("test music", duration=1.0)
        assert isinstance(audio, np.ndarray)
        assert sr == 32000
        assert len(audio) > 0

    def test_generate_with_parameters(self):
        """Custom parameters are passed to model.generate()."""
        generator = MusicGenerator()
        generator.generate("test", duration=10.0, temperature=0.8, guidance_scale=5.0)
        kwargs = self.mock_model_instance.generate.call_args[1]
        assert kwargs["do_sample"] is True
        assert kwargs["temperature"] == 0.8
        assert kwargs["guidance_scale"] == 5.0

    def test_generate_with_progress_callback(self):
        """Progress callback is invoked during generation."""
        generator = MusicGenerator()
        callback_calls = []

        def on_progress(percent, message):
            callback_calls.append((percent, message))

        generator.generate("test music", duration=1.0, progress_callback=on_progress)
        assert len(callback_calls) > 0
        percents = [c[0] for c in callback_calls]
        assert 0 in percents
        assert 100 in percents

    def test_generate_error_propagates(self):
        """Model errors propagate to caller."""
        self.mock_model_instance.generate.side_effect = RuntimeError("CUDA out of memory")
        generator = MusicGenerator()
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            generator.generate("test music")

    def test_generate_empty_prompt_raises(self):
        """Empty prompt raises ValueError."""
        generator = MusicGenerator()
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            generator.generate("")

    def test_generate_extended_triggered(self):
        """Duration > 30s triggers extended generation."""
        generator = MusicGenerator()
        self.mock_model_instance.generate.return_value = torch.randn(1, 1, 32000 * 25)
        audio, sr = generator.generate("test", duration=60.0)
        assert isinstance(audio, np.ndarray)
        assert self.mock_model_instance.generate.call_count >= 2
