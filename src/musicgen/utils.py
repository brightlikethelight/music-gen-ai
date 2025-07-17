"""
Essential utilities for MusicGen Unified.
Only the helpers that actually matter.
"""

import os
import time
import hashlib
import logging
from typing import Optional, Union, List, Tuple
from pathlib import Path

import torch
import numpy as np
import soundfile as sf
from pydub import AudioSegment

logger = logging.getLogger(__name__)


def load_audio(
    file_path: Union[str, Path],
    target_sr: int = 32000,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load audio file and optionally resample.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
        mono: Convert to mono
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    import librosa
    
    # Load audio
    audio, sr = librosa.load(file_path, sr=target_sr, mono=mono)
    
    return audio, sr


def save_audio(
    audio: Union[np.ndarray, torch.Tensor],
    sample_rate: int,
    file_path: Union[str, Path],
    format: str = "wav"
) -> str:
    """
    Save audio to file with format conversion.
    
    Args:
        audio: Audio data
        sample_rate: Sample rate
        file_path: Output path
        format: Output format (wav/mp3/flac)
        
    Returns:
        Path to saved file
    """
    # Convert torch to numpy if needed
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    
    # Ensure 1D for mono
    if audio.ndim > 1:
        audio = audio.squeeze()
    
    # Normalize if needed
    if audio.max() > 1.0 or audio.min() < -1.0:
        audio = np.clip(audio, -1.0, 1.0)
    
    # Ensure output directory exists
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Handle format
    if format == "mp3":
        # First save as WAV
        temp_wav = output_path.with_suffix('.wav')
        sf.write(temp_wav, audio, sample_rate)
        
        # Convert to MP3
        try:
            audio_segment = AudioSegment.from_wav(str(temp_wav))
            audio_segment.export(str(output_path), format="mp3", bitrate="192k")
            os.remove(temp_wav)
            return str(output_path)
        except Exception as e:
            logger.warning(f"MP3 conversion failed: {e}. Saved as WAV instead.")
            return str(temp_wav)
    else:
        # Direct save
        sf.write(str(output_path), audio, sample_rate)
        return str(output_path)


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get best available device.
    
    Args:
        device: Specific device name or None for auto
        
    Returns:
        torch.device
    """
    if device:
        return torch.device(device)
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def format_time(seconds: float) -> str:
    """Format seconds to human readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {int(secs)}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"


def get_cache_dir() -> Path:
    """Get cache directory for models."""
    cache_dir = Path.home() / ".cache" / "musicgen-unified"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def hash_text(text: str) -> str:
    """Create hash of text for caching."""
    return hashlib.md5(text.encode()).hexdigest()[:8]


def estimate_memory_usage(
    duration: float,
    model_size: str = "small"
) -> dict:
    """
    Estimate memory usage for generation.
    
    Args:
        duration: Duration in seconds
        model_size: Model size (small/medium/large)
        
    Returns:
        Dict with memory estimates
    """
    # Base model sizes (approximate)
    model_memory = {
        "small": 0.5,   # 500MB
        "medium": 1.5,  # 1.5GB
        "large": 3.5    # 3.5GB
    }
    
    # Estimate based on duration (rough)
    # ~100MB per 10s of audio generation
    generation_memory = (duration / 10) * 0.1
    
    base = model_memory.get(model_size, 0.5)
    total = base + generation_memory
    
    return {
        "model_memory_gb": base,
        "generation_memory_gb": generation_memory,
        "total_memory_gb": total,
        "recommended_gpu_memory_gb": total * 1.5  # Safety margin
    }


def crossfade_audio(
    audio1: np.ndarray,
    audio2: np.ndarray,
    overlap_seconds: float,
    sample_rate: int
) -> np.ndarray:
    """
    Crossfade between two audio segments.
    
    Args:
        audio1: First audio segment
        audio2: Second audio segment
        overlap_seconds: Overlap duration
        sample_rate: Sample rate
        
    Returns:
        Crossfaded audio
    """
    overlap_samples = int(overlap_seconds * sample_rate)
    
    if len(audio1) < overlap_samples or len(audio2) < overlap_samples:
        # Just concatenate if too short
        return np.concatenate([audio1, audio2])
    
    # Create fade curves
    fade_out = np.linspace(1, 0, overlap_samples)
    fade_in = np.linspace(0, 1, overlap_samples)
    
    # Apply fades
    audio1_fade = audio1.copy()
    audio1_fade[-overlap_samples:] *= fade_out
    
    audio2_fade = audio2.copy()
    audio2_fade[:overlap_samples] *= fade_in
    
    # Combine
    result = np.concatenate([
        audio1[:-overlap_samples],
        audio1_fade[-overlap_samples:] + audio2_fade[:overlap_samples],
        audio2[overlap_samples:]
    ])
    
    return result


def apply_fade(
    audio: np.ndarray,
    sample_rate: int,
    fade_in: float = 0.1,
    fade_out: float = 0.1
) -> np.ndarray:
    """
    Apply fade in/out to audio.
    
    Args:
        audio: Audio data
        sample_rate: Sample rate
        fade_in: Fade in duration (seconds)
        fade_out: Fade out duration (seconds)
        
    Returns:
        Audio with fades applied
    """
    audio = audio.copy()
    
    # Fade in
    if fade_in > 0:
        fade_in_samples = int(fade_in * sample_rate)
        fade_in_curve = np.linspace(0, 1, fade_in_samples)
        audio[:fade_in_samples] *= fade_in_curve
    
    # Fade out
    if fade_out > 0:
        fade_out_samples = int(fade_out * sample_rate)
        fade_out_curve = np.linspace(1, 0, fade_out_samples)
        audio[-fade_out_samples:] *= fade_out_curve
    
    return audio


def validate_prompt_length(prompt: str, max_length: int = 256) -> str:
    """
    Validate and truncate prompt if needed.
    
    Args:
        prompt: Input prompt
        max_length: Maximum character length
        
    Returns:
        Validated prompt
    """
    prompt = prompt.strip()
    
    if len(prompt) > max_length:
        # Truncate at word boundary
        truncated = prompt[:max_length]
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.8:  # Keep at least 80%
            truncated = truncated[:last_space]
        
        logger.warning(f"Prompt truncated from {len(prompt)} to {len(truncated)} chars")
        return truncated + "..."
    
    return prompt


def setup_logging(
    level: str = "INFO",
    format: Optional[str] = None
) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Log level
        format: Log format string
    """
    if format is None:
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format
    )


class ProgressTracker:
    """Simple progress tracking for long operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, n: int = 1) -> None:
        """Update progress."""
        self.current += n
    
    def get_progress(self) -> dict:
        """Get progress info."""
        elapsed = time.time() - self.start_time
        percent = (self.current / self.total) * 100 if self.total > 0 else 0
        
        # Estimate remaining time
        if self.current > 0:
            rate = self.current / elapsed
            remaining = (self.total - self.current) / rate if rate > 0 else 0
        else:
            remaining = 0
        
        return {
            "current": self.current,
            "total": self.total,
            "percent": percent,
            "elapsed": elapsed,
            "remaining": remaining,
            "description": self.description
        }