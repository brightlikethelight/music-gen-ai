#!/usr/bin/env python3
"""
Advanced features example for MusicGen Unified.

This example demonstrates advanced features like prompt engineering,
custom model selection, and audio processing.
"""

from musicgen import MusicGenerator, PromptEngineer
from musicgen.infrastructure.config.settings import get_config


def main():
    """Demonstrate advanced features."""
    # Load configuration
    config = get_config()
    
    # Initialize components
    generator = MusicGenerator(
        model_name="facebook/musicgen-medium",
        device="auto",
        optimize=True
    )
    
    prompt_engineer = PromptEngineer()
    
    # Use prompt engineering for better results
    base_prompt = "relaxing ambient music"
    enhanced_prompt = prompt_engineer.enhance_prompt(
        base_prompt,
        style="ambient",
        instruments=["synthesizer", "pad"],
        mood="peaceful",
        duration_hint="medium"
    )
    
    print(f"Original prompt: {base_prompt}")
    print(f"Enhanced prompt: {enhanced_prompt}")
    
    # Generate with different settings
    results = []
    
    for i, duration in enumerate([15.0, 30.0]):
        print(f"\nGenerating track {i+1} - {duration}s")
        
        audio_data, sample_rate = generator.generate(
            prompt=enhanced_prompt,
            duration=duration,
            temperature=0.8,
            top_k=200
        )
        
        output_path = f"advanced_example_{i+1}_{duration}s.wav"
        generator.save_audio(audio_data, output_path, sample_rate)
        
        results.append({
            'path': output_path,
            'duration': duration,
            'shape': audio_data.shape
        })
        
        print(f"Saved: {output_path}")
    
    # Summary
    print("\nGeneration Summary:")
    for result in results:
        print(f"- {result['path']}: {result['duration']}s, shape {result['shape']}")


if __name__ == "__main__":
    main()