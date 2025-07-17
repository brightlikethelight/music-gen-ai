"""
Core music generation with GPU optimization.
Clean implementation focused on what works.
"""

import logging
import os
import time
from typing import Tuple, Optional, Callable, Union

import numpy as np
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    import scipy.io.wavfile as wavfile
    SOUNDFILE_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

logger = logging.getLogger(__name__)


class MusicGenerator:
    """
    GPU-optimized music generation.
    
    Simple, clean, focused on instrumental music only.
    No vocals, no complexity, just music that works.
    """
    
    def __init__(
        self, 
        model_name: str = "facebook/musicgen-small",
        device: Optional[str] = None,
        optimize: bool = True
    ):
        """
        Initialize the generator.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use (auto-detect if None)
            optimize: Enable GPU optimizations
        """
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.optimize = optimize and torch.cuda.is_available()
        
        logger.info(f"Loading {model_name} on {self.device}")
        self._load_model()
        
        if self.optimize:
            self._apply_optimizations()
        
        logger.info("✓ Model ready for generation")
    
    def _setup_device(self, device: Optional[str]) -> str:
        """Setup and validate device."""
        if device:
            return device
            
        if torch.cuda.is_available():
            # Get GPU with most free memory
            gpu_id = 0
            if torch.cuda.device_count() > 1:
                free_memory = []
                for i in range(torch.cuda.device_count()):
                    torch.cuda.set_device(i)
                    free_memory.append(torch.cuda.mem_get_info()[0])
                gpu_id = np.argmax(free_memory)
            return f"cuda:{gpu_id}"
        
        return "cpu"
    
    def _load_model(self):
        """Load model with error handling."""
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = MusicgenForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.optimize else torch.float32
            )
            self.model.to(self.device)
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def _apply_optimizations(self):
        """Apply GPU optimizations for faster generation."""
        logger.info("Applying GPU optimizations...")
        
        # Enable cuDNN autotuner
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        # Compile model if PyTorch 2.0+
        if hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(
                    self.model,
                    mode='reduce-overhead',
                    fullgraph=True
                )
                logger.info("✓ Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Compilation failed, continuing without: {e}")
        
        # Enable flash attention if available
        if hasattr(self.model.config, 'use_flash_attention_2'):
            self.model.config.use_flash_attention_2 = True
            logger.info("✓ Flash Attention enabled")
    
    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        duration: float = 10.0,
        temperature: float = 1.0,
        guidance_scale: float = 3.0,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Generate music from text prompt.
        
        Args:
            prompt: Text description of the music
            duration: Duration in seconds (max 30 for single generation)
            temperature: Sampling temperature (0.1-2.0)
            guidance_scale: How closely to follow prompt (1.0-10.0)
            progress_callback: Optional callback(percent, message)
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        # Validate inputs
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        if duration > 30:
            logger.info(f"Duration {duration}s > 30s, using extended generation")
            return self.generate_extended(
                prompt, duration, temperature, guidance_scale, progress_callback
            )
        
        # Progress tracking
        if progress_callback:
            progress_callback(0, "Processing prompt...")
        
        start_time = time.time()
        
        # Process inputs
        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Calculate tokens needed
        max_new_tokens = int(256 * duration / 5)
        
        if progress_callback:
            progress_callback(10, f"Generating {duration}s of music...")
        
        # Generate with optimizations
        with torch.cuda.amp.autocast(enabled=self.optimize):
            audio_values = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                guidance_scale=guidance_scale
            )
        
        if progress_callback:
            progress_callback(90, "Processing audio...")
        
        # Extract audio
        audio = audio_values[0, 0].cpu().numpy()
        sample_rate = self.model.config.audio_encoder.sampling_rate
        
        # Log performance
        gen_time = time.time() - start_time
        speed = duration / gen_time
        logger.info(f"✓ Generated {duration:.1f}s in {gen_time:.1f}s ({speed:.1f}x realtime)")
        
        if progress_callback:
            progress_callback(100, "Complete!")
        
        return audio, sample_rate
    
    def generate_extended(
        self,
        prompt: str,
        duration: float,
        temperature: float = 1.0,
        guidance_scale: float = 3.0,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        segment_duration: float = 25.0,
        overlap: float = 2.0
    ) -> Tuple[np.ndarray, int]:
        """
        Generate music longer than 30 seconds using segmented approach.
        
        Args:
            prompt: Text description
            duration: Total duration in seconds
            temperature: Sampling temperature
            guidance_scale: Guidance strength
            progress_callback: Progress callback
            segment_duration: Duration of each segment
            overlap: Overlap between segments for crossfading
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        segments = []
        num_segments = int(np.ceil(duration / (segment_duration - overlap)))
        
        logger.info(f"Extended generation: {num_segments} segments for {duration}s")
        
        for i in range(num_segments):
            if progress_callback:
                percent = (i / num_segments) * 100
                progress_callback(percent, f"Segment {i+1}/{num_segments}")
            
            # Generate segment
            seg_audio, sr = self.generate(
                prompt,
                min(segment_duration, duration - i * (segment_duration - overlap)),
                temperature,
                guidance_scale
            )
            segments.append(seg_audio)
        
        # Blend segments with crossfade
        if progress_callback:
            progress_callback(90, "Blending segments...")
        
        final_audio = self._blend_segments(segments, sr, overlap)
        
        # Trim to exact duration
        target_samples = int(duration * sr)
        if len(final_audio) > target_samples:
            final_audio = final_audio[:target_samples]
        
        if progress_callback:
            progress_callback(100, "Complete!")
        
        return final_audio, sr
    
    def _blend_segments(
        self,
        segments: list,
        sample_rate: int,
        overlap_seconds: float
    ) -> np.ndarray:
        """Blend audio segments with crossfade."""
        if len(segments) == 1:
            return segments[0]
        
        overlap_samples = int(overlap_seconds * sample_rate)
        
        # Start with first segment
        result = segments[0]
        
        for next_segment in segments[1:]:
            if overlap_samples > 0 and len(result) > overlap_samples:
                # Create crossfade
                fade_out = np.linspace(1, 0, overlap_samples)
                fade_in = np.linspace(0, 1, overlap_samples)
                
                # Apply fades
                result[-overlap_samples:] *= fade_out
                next_segment[:overlap_samples] *= fade_in
                
                # Blend overlap
                blend = result[-overlap_samples:] + next_segment[:overlap_samples]
                
                # Concatenate
                result = np.concatenate([
                    result[:-overlap_samples],
                    blend,
                    next_segment[overlap_samples:]
                ])
            else:
                # Simple concatenation
                result = np.concatenate([result, next_segment])
        
        return result
    
    def save_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        filename: str,
        format: str = "auto"
    ) -> str:
        """
        Save audio to file with format detection.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            filename: Output filename
            format: Format (auto, wav, mp3)
            
        Returns:
            Path to saved file
        """
        # Auto-detect format
        if format == "auto":
            format = "mp3" if filename.lower().endswith(".mp3") else "wav"
        
        # Ensure extension matches format
        base = os.path.splitext(filename)[0]
        output_path = f"{base}.{format}"
        
        # Normalize audio
        audio = np.clip(audio, -1, 1)
        
        # Save based on format
        if format == "mp3" and PYDUB_AVAILABLE:
            # Save as WAV first
            temp_wav = f"{base}_temp.wav"
            self._save_wav(audio, sample_rate, temp_wav)
            
            # Convert to MP3
            try:
                audio_segment = AudioSegment.from_wav(temp_wav)
                audio_segment.export(output_path, format="mp3", bitrate="192k")
                os.remove(temp_wav)
                logger.info(f"✓ Saved as MP3: {output_path}")
            except Exception as e:
                logger.warning(f"MP3 conversion failed: {e}")
                output_path = f"{base}.wav"
                os.rename(temp_wav, output_path)
                logger.info(f"✓ Saved as WAV instead: {output_path}")
        else:
            # Save as WAV
            output_path = f"{base}.wav"
            self._save_wav(audio, sample_rate, output_path)
            logger.info(f"✓ Saved as WAV: {output_path}")
        
        return output_path
    
    def _save_wav(self, audio: np.ndarray, sample_rate: int, filename: str):
        """Save audio as WAV file."""
        if SOUNDFILE_AVAILABLE:
            sf.write(filename, audio, sample_rate, subtype='PCM_16')
        else:
            # Convert to 16-bit PCM
            audio_16bit = (audio * 32767).astype(np.int16)
            wavfile.write(filename, sample_rate, audio_16bit)
    
    def get_info(self) -> dict:
        """Get model and system information."""
        info = {
            "model": self.model_name,
            "device": str(self.device),
            "optimized": self.optimize,
            "sample_rate": self.model.config.audio_encoder.sampling_rate,
        }
        
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name()
            info["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        
        return info


# Convenience function
def quick_generate(
    prompt: str,
    output_file: str = "output.mp3",
    duration: float = 30.0,
    model: str = "small"
) -> str:
    """
    Quick generation helper.
    
    Args:
        prompt: Music description
        output_file: Output filename
        duration: Duration in seconds
        model: Model size (small, medium, large)
        
    Returns:
        Path to generated file
    """
    model_map = {
        "small": "facebook/musicgen-small",
        "medium": "facebook/musicgen-medium", 
        "large": "facebook/musicgen-large"
    }
    
    generator = MusicGenerator(model_map.get(model, model))
    audio, sr = generator.generate(prompt, duration)
    return generator.save_audio(audio, sr, output_file)