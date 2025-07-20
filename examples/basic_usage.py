#!/usr/bin/env python3
"""
Basic usage example for MusicGen Unified.

This example demonstrates the simplest way to generate music using the MusicGen library.
"""

from musicgen import MusicGenerator


def main():
    """Generate a simple music track."""
    # Initialize the generator with default settings
    generator = MusicGenerator()
    
    # Generate a 10-second music track
    prompt = "upbeat jazz piano solo"
    duration = 10.0
    
    print(f"Generating {duration}s of music from prompt: '{prompt}'")
    
    # Generate the audio
    audio_data, sample_rate = generator.generate(
        prompt=prompt,
        duration=duration
    )
    
    print(f"Generated audio shape: {audio_data.shape}")
    print(f"Sample rate: {sample_rate} Hz")
    
    # Save the audio
    output_path = "basic_example_output.wav"
    generator.save_audio(audio_data, output_path, sample_rate)
    
    print(f"Audio saved to: {output_path}")


if __name__ == "__main__":
    main()