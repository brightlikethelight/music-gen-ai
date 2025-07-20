"""
Unit tests for musicgen.core.generator module.
Mocks heavy ML dependencies for CI compatibility.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, PropertyMock

import numpy as np
import pytest
import torch

from musicgen.core.generator import MusicGenerator


class TestMusicGenerator:
    """Test MusicGenerator class with mocked ML dependencies."""

    @pytest.fixture
    def mock_transformers(self):
        """Mock transformers imports."""
        with patch('musicgen.core.generator.AutoProcessor') as mock_processor, \
             patch('musicgen.core.generator.MusicgenForConditionalGeneration') as mock_model:
            
            # Mock processor
            mock_processor_instance = MagicMock()
            mock_processor.from_pretrained.return_value = mock_processor_instance
            
            # Mock model
            mock_model_instance = MagicMock()
            mock_model.from_pretrained.return_value = mock_model_instance
            
            # Mock model properties
            mock_config = MagicMock()
            mock_config.audio_encoder.sampling_rate = 32000
            mock_model_instance.config = mock_config
            
            yield mock_processor, mock_model, mock_processor_instance, mock_model_instance

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_init_default_device(self, mock_transformers):
        """Test initialization with default device selection."""
        mock_processor, mock_model, _, _ = mock_transformers
        
        with patch('torch.cuda.is_available', return_value=True):
            generator = MusicGenerator()
            assert generator.device == torch.device("cuda")
        
        with patch('torch.cuda.is_available', return_value=False):
            generator = MusicGenerator()
            assert generator.device == torch.device("cpu")

    def test_init_custom_device(self, mock_transformers):
        """Test initialization with custom device."""
        generator = MusicGenerator(device="cpu")
        assert generator.device == torch.device("cpu")

    def test_init_model_loading(self, mock_transformers):
        """Test model loading during initialization."""
        mock_processor, mock_model, _, _ = mock_transformers
        
        generator = MusicGenerator(model_name="facebook/musicgen-small")
        
        mock_processor.from_pretrained.assert_called_once()
        mock_model.from_pretrained.assert_called_once()
        
        # Check model name was passed
        call_args = mock_model.from_pretrained.call_args
        assert "facebook/musicgen-small" in str(call_args)

    def test_init_optimization(self, mock_transformers):
        """Test model optimization during initialization."""
        _, _, _, mock_model_instance = mock_transformers
        
        # Test with optimization enabled
        generator = MusicGenerator(optimize=True)
        mock_model_instance.to.assert_called()
        
        # Test with optimization disabled
        mock_model_instance.reset_mock()
        generator = MusicGenerator(optimize=False)
        mock_model_instance.to.assert_called()

    def test_cleanup(self, mock_transformers):
        """Test cleanup method."""
        generator = MusicGenerator()
        generator.cleanup()
        
        assert generator.model is None
        assert generator.processor is None

    @patch('torch.cuda.empty_cache')
    def test_cleanup_with_cuda(self, mock_empty_cache, mock_transformers):
        """Test cleanup with CUDA cache clearing."""
        with patch('torch.cuda.is_available', return_value=True):
            generator = MusicGenerator()
            generator.cleanup()
            mock_empty_cache.assert_called_once()

    def test_context_manager(self, mock_transformers):
        """Test context manager usage."""
        with MusicGenerator() as generator:
            assert generator.model is not None
            assert generator.processor is not None
        
        # After context, should be cleaned up
        assert generator.model is None
        assert generator.processor is None

    def test_generate_basic(self, mock_transformers, temp_cache_dir):
        """Test basic music generation."""
        _, _, mock_processor_instance, mock_model_instance = mock_transformers
        
        # Mock processor return
        mock_inputs = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        mock_processor_instance.return_value = mock_inputs
        
        # Mock model generation
        mock_audio_values = torch.randn(1, 1, 32000)  # 1 second of audio
        mock_model_instance.generate.return_value = mock_audio_values
        
        generator = MusicGenerator()
        
        with patch('musicgen.core.generator.sf.write') as mock_write:
            output_path = generator.generate(
                prompt="test music",
                duration=1.0,
                output_path=str(temp_cache_dir / "test.wav")
            )
            
            assert output_path == str(temp_cache_dir / "test.wav")
            mock_write.assert_called_once()

    def test_generate_with_parameters(self, mock_transformers):
        """Test generation with custom parameters."""
        _, _, mock_processor_instance, mock_model_instance = mock_transformers
        
        # Mock returns
        mock_processor_instance.return_value = {'input_ids': torch.tensor([[1]])}
        mock_model_instance.generate.return_value = torch.randn(1, 1, 32000)
        
        generator = MusicGenerator()
        
        with patch('musicgen.core.generator.sf.write'):
            generator.generate(
                prompt="test",
                duration=10.0,
                temperature=0.8,
                top_k=100,
                top_p=0.95,
                guidance_scale=5.0
            )
            
            # Check model.generate was called with correct parameters
            generate_call = mock_model_instance.generate.call_args
            assert generate_call is not None
            kwargs = generate_call[1]
            assert kwargs.get('do_sample') is True
            assert kwargs.get('temperature') == 0.8
            assert kwargs.get('top_k') == 100
            assert kwargs.get('top_p') == 0.95
            assert kwargs.get('guidance_scale') == 5.0

    def test_generate_duration_limits(self, mock_transformers):
        """Test duration validation."""
        generator = MusicGenerator()
        
        # Test minimum duration
        with pytest.raises(ValueError, match="Duration must be between"):
            generator.generate("test", duration=0.0)
        
        # Test maximum duration
        with pytest.raises(ValueError, match="Duration must be between"):
            generator.generate("test", duration=301.0)

    def test_generate_empty_prompt(self, mock_transformers):
        """Test generation with empty prompt."""
        generator = MusicGenerator()
        
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            generator.generate("")
        
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            generator.generate("   ")

    def test_generate_callback(self, mock_transformers):
        """Test generation with progress callback."""
        _, _, mock_processor_instance, mock_model_instance = mock_transformers
        
        mock_processor_instance.return_value = {'input_ids': torch.tensor([[1]])}
        mock_model_instance.generate.return_value = torch.randn(1, 1, 32000)
        
        generator = MusicGenerator()
        
        callback_called = False
        def progress_callback(step, total):
            nonlocal callback_called
            callback_called = True
            assert step >= 0
            assert total > 0
        
        with patch('musicgen.core.generator.sf.write'):
            generator.generate(
                "test music",
                duration=1.0,
                callback=progress_callback
            )
        
        assert callback_called

    def test_generate_error_handling(self, mock_transformers):
        """Test error handling during generation."""
        _, _, mock_processor_instance, mock_model_instance = mock_transformers
        
        # Mock generation failure
        mock_processor_instance.return_value = {'input_ids': torch.tensor([[1]])}
        mock_model_instance.generate.side_effect = RuntimeError("CUDA out of memory")
        
        generator = MusicGenerator()
        
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            generator.generate("test music")

    def test_save_audio_formats(self, mock_transformers, temp_cache_dir):
        """Test saving audio in different formats."""
        generator = MusicGenerator()
        
        # Mock audio data
        audio = np.random.randn(32000).astype(np.float32)
        sample_rate = 32000
        
        # Test WAV format
        with patch('musicgen.core.generator.sf.write') as mock_write:
            wav_path = generator._save_audio(
                audio, sample_rate, 
                str(temp_cache_dir / "test.wav")
            )
            mock_write.assert_called_once()
            assert wav_path.endswith(".wav")
        
        # Test MP3 format (with pydub)
        with patch('musicgen.core.generator.PYDUB_AVAILABLE', True), \
             patch('musicgen.core.generator.AudioSegment') as mock_segment:
            
            mock_audio_segment = MagicMock()
            mock_segment.from_wav.return_value = mock_audio_segment
            
            mp3_path = generator._save_audio(
                audio, sample_rate,
                str(temp_cache_dir / "test.mp3")
            )
            
            mock_audio_segment.export.assert_called_once()
            assert mp3_path.endswith(".mp3")

    def test_save_audio_fallback(self, mock_transformers, temp_cache_dir):
        """Test audio saving fallback when soundfile not available."""
        generator = MusicGenerator()
        
        audio = np.random.randn(32000).astype(np.float32)
        sample_rate = 32000
        
        with patch('musicgen.core.generator.SOUNDFILE_AVAILABLE', False), \
             patch('musicgen.core.generator.wavfile.write') as mock_wavwrite:
            
            wav_path = generator._save_audio(
                audio, sample_rate,
                str(temp_cache_dir / "test.wav")
            )
            
            mock_wavwrite.assert_called_once()

    def test_load_model_caching(self, mock_transformers, temp_cache_dir):
        """Test model caching behavior."""
        mock_processor, mock_model, _, _ = mock_transformers
        
        with patch.dict(os.environ, {'MUSICGEN_CACHE_DIR': str(temp_cache_dir)}):
            generator = MusicGenerator()
            
            # Check cache dir was used
            processor_call = mock_processor.from_pretrained.call_args
            model_call = mock_model.from_pretrained.call_args
            
            assert 'cache_dir' in processor_call[1]
            assert 'cache_dir' in model_call[1]

    def test_device_map_for_large_models(self, mock_transformers):
        """Test device mapping for large models."""
        mock_processor, mock_model, _, _ = mock_transformers
        
        # Test large model
        generator = MusicGenerator(model_name="facebook/musicgen-large")
        
        model_call = mock_model.from_pretrained.call_args
        assert 'device_map' in model_call[1]
        assert model_call[1]['device_map'] == "auto"

    def test_prompt_enhancement(self, mock_transformers):
        """Test prompt enhancement functionality."""
        _, _, mock_processor_instance, mock_model_instance = mock_transformers
        
        mock_processor_instance.return_value = {'input_ids': torch.tensor([[1]])}
        mock_model_instance.generate.return_value = torch.randn(1, 1, 32000)
        
        generator = MusicGenerator()
        
        # Mock prompt engineer
        with patch.object(generator, 'prompt_engineer') as mock_engineer:
            mock_engineer.enhance_prompt.return_value = "enhanced prompt"
            
            with patch('musicgen.core.generator.sf.write'):
                generator.generate("simple prompt", enhance_prompt=True)
            
            mock_engineer.enhance_prompt.assert_called_once_with("simple prompt")

    def test_multi_generation(self, mock_transformers, temp_cache_dir):
        """Test generating multiple outputs."""
        _, _, mock_processor_instance, mock_model_instance = mock_transformers
        
        mock_processor_instance.return_value = {'input_ids': torch.tensor([[1]])}
        # Return different audio for each generation
        mock_model_instance.generate.side_effect = [
            torch.randn(1, 1, 32000),
            torch.randn(1, 1, 32000),
            torch.randn(1, 1, 32000)
        ]
        
        generator = MusicGenerator()
        
        with patch('musicgen.core.generator.sf.write'):
            outputs = generator.generate_multiple(
                "test music",
                num_outputs=3,
                output_dir=str(temp_cache_dir)
            )
        
        assert len(outputs) == 3
        assert all(Path(p).name.startswith("test_music_") for p in outputs)

    def test_streaming_generation(self, mock_transformers):
        """Test streaming generation capability."""
        _, _, mock_processor_instance, mock_model_instance = mock_transformers
        
        mock_processor_instance.return_value = {'input_ids': torch.tensor([[1]])}
        
        # Mock streaming output
        chunks = [torch.randn(1, 1, 16000) for _ in range(4)]  # 4 chunks
        mock_model_instance.generate.return_value = chunks
        
        generator = MusicGenerator()
        
        if hasattr(generator, 'generate_stream'):
            chunk_count = 0
            for chunk in generator.generate_stream("test music", duration=2.0):
                chunk_count += 1
                assert isinstance(chunk, (np.ndarray, torch.Tensor))
            
            assert chunk_count == 4

    def test_batch_generation(self, mock_transformers):
        """Test batch generation efficiency."""
        _, _, mock_processor_instance, mock_model_instance = mock_transformers
        
        # Mock batch processing
        batch_size = 4
        mock_processor_instance.return_value = {
            'input_ids': torch.randn(batch_size, 10),
            'attention_mask': torch.ones(batch_size, 10)
        }
        mock_model_instance.generate.return_value = torch.randn(batch_size, 1, 32000)
        
        generator = MusicGenerator()
        
        prompts = ["prompt 1", "prompt 2", "prompt 3", "prompt 4"]
        
        if hasattr(generator, 'generate_batch'):
            with patch('musicgen.core.generator.sf.write'):
                outputs = generator.generate_batch(prompts, duration=1.0)
            
            assert len(outputs) == batch_size