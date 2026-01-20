"""
Async music generation with non-blocking model loading.
Provides responsive UI during heavy operations.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional, Tuple

import numpy as np
import torch

from .generator import MusicGenerator

logger = logging.getLogger(__name__)


class AsyncMusicGenerator:
    """Asynchronous wrapper for MusicGenerator with non-blocking operations."""

    def __init__(
        self,
        model_name: str = "facebook/musicgen-small",
        device: Optional[str] = None,
        optimize: bool = True,
        max_workers: int = 2,
    ):
        """
        Initialize async generator.

        Args:
            model_name: HuggingFace model name
            device: Device to use (auto-detect if None)
            optimize: Enable GPU optimizations
            max_workers: Max threads for background operations
        """
        self.model_name = model_name
        self.device_str = device
        self.optimize = optimize
        self.generator: Optional[MusicGenerator] = None
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.loading = False
        self.load_progress = 0.0
        self.load_message = ""

    async def load_model_async(
        self, progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> None:
        """
        Load model asynchronously without blocking the event loop.

        Args:
            progress_callback: Optional callback(percent, message) for progress updates
        """
        if self.loading:
            logger.warning("Model is already loading")
            return

        if self.generator is not None:
            logger.info("Model already loaded")
            if progress_callback:
                progress_callback(100, "Model ready")
            return

        self.loading = True
        self.load_progress = 0.0

        try:
            # Update progress
            self.load_message = "Initializing..."
            if progress_callback:
                progress_callback(0, self.load_message)

            # Run model loading in background thread
            loop = asyncio.get_event_loop()

            # Stage 1: Initialize generator (fast)
            self.load_message = f"Setting up {self.model_name}..."
            if progress_callback:
                progress_callback(10, self.load_message)

            def create_generator():
                """Create generator in thread."""
                return MusicGenerator(
                    model_name=self.model_name, device=self.device_str, optimize=self.optimize
                )

            # Load model in executor
            start_time = time.time()
            self.generator = await loop.run_in_executor(self.executor, create_generator)

            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.1f}s")

            # Complete
            self.load_progress = 100
            self.load_message = "Model ready!"
            if progress_callback:
                progress_callback(100, self.load_message)

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.load_message = f"Error: {str(e)}"
            if progress_callback:
                progress_callback(-1, self.load_message)
            raise
        finally:
            self.loading = False

    async def generate_async(
        self,
        prompt: str,
        duration: float = 30.0,
        temperature: float = 1.0,
        guidance_scale: float = 3.0,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate music asynchronously.

        Args:
            prompt: Text description of the music
            duration: Duration in seconds
            temperature: Sampling temperature
            guidance_scale: How closely to follow prompt
            progress_callback: Optional callback(percent, message)

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        # Ensure model is loaded
        if self.generator is None:
            await self.load_model_async(progress_callback)

        # Run generation in background thread
        loop = asyncio.get_event_loop()

        def generate_with_progress():
            """Wrapper to handle progress in thread."""
            progress_data = {"percent": 0, "message": "Starting generation..."}

            def sync_progress(percent, message):
                progress_data["percent"] = percent
                progress_data["message"] = message
                # Could use a queue here for real-time updates

            audio, sample_rate = self.generator.generate(
                prompt=prompt,
                duration=duration,
                temperature=temperature,
                guidance_scale=guidance_scale,
                progress_callback=sync_progress if progress_callback else None,
            )

            return audio, sample_rate, progress_data

        # Generate in executor
        result = await loop.run_in_executor(self.executor, generate_with_progress)

        audio, sample_rate, final_progress = result

        if progress_callback:
            progress_callback(100, "Generation complete!")

        return audio, sample_rate

    async def save_audio_async(
        self, audio: np.ndarray, sample_rate: int, filename: str, format: str = "auto"
    ) -> str:
        """
        Save audio asynchronously.

        Args:
            audio: Audio array
            sample_rate: Sample rate
            filename: Output filename
            format: Format (auto, wav, mp3)

        Returns:
            Path to saved file
        """
        if self.generator is None:
            raise RuntimeError("Generator not initialized")

        # Save in background thread
        loop = asyncio.get_event_loop()
        output_path = await loop.run_in_executor(
            self.executor, self.generator.save_audio, audio, sample_rate, filename, format
        )

        return output_path

    async def generate_and_save_async(
        self,
        prompt: str,
        output_file: str,
        duration: float = 30.0,
        temperature: float = 1.0,
        guidance_scale: float = 3.0,
        format: str = "auto",
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> str:
        """
        Generate and save music in one async operation.

        Args:
            prompt: Text description
            output_file: Output filename
            duration: Duration in seconds
            temperature: Sampling temperature
            guidance_scale: Guidance strength
            format: Output format
            progress_callback: Progress callback

        Returns:
            Path to saved file
        """
        # Generate
        audio, sample_rate = await self.generate_async(
            prompt=prompt,
            duration=duration,
            temperature=temperature,
            guidance_scale=guidance_scale,
            progress_callback=progress_callback,
        )

        # Save
        if progress_callback:
            progress_callback(95, "Saving audio...")

        output_path = await self.save_audio_async(
            audio=audio, sample_rate=sample_rate, filename=output_file, format=format
        )

        if progress_callback:
            progress_callback(100, f"Saved to {output_path}")

        return output_path

    def cleanup(self):
        """Clean up resources."""
        if self.generator:
            # Clean up GPU memory if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.generator = None
        self.executor.shutdown(wait=True)

    async def __aenter__(self):
        """Async context manager entry."""
        await self.load_model_async()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.cleanup()


# Convenience function for quick async generation
async def generate_music_async(
    prompt: str,
    output_file: str = "output.mp3",
    duration: float = 30.0,
    model: str = "small",
    show_progress: bool = True,
) -> str:
    """
    Quick async music generation.

    Args:
        prompt: Music description
        output_file: Output filename
        duration: Duration in seconds
        model: Model size (small, medium, large)
        show_progress: Show progress updates

    Returns:
        Path to generated file
    """
    model_map = {
        "small": "facebook/musicgen-small",
        "medium": "facebook/musicgen-medium",
        "large": "facebook/musicgen-large",
    }

    def progress_callback(percent, message):
        if show_progress:
            print(f"[{percent:3.0f}%] {message}")

    async with AsyncMusicGenerator(model_map.get(model, model)) as generator:
        output_path = await generator.generate_and_save_async(
            prompt=prompt,
            output_file=output_file,
            duration=duration,
            progress_callback=progress_callback if show_progress else None,
        )

    return output_path


# Example usage
if __name__ == "__main__":
    import asyncio

    async def main():
        """Example async generation."""
        output = await generate_music_async(
            prompt="peaceful piano melody with strings", output_file="async_test.mp3", duration=10
        )
        print(f"Generated: {output}")

    asyncio.run(main())
