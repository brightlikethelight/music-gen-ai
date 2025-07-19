#!/usr/bin/env python3
"""
Minimal test script to verify MusicGen works with transformers library only.
No audiocraft dependency required.
"""

import sys
import time
import numpy as np
import torch

# Test 1: Basic imports
print("=" * 50)
print("Testing MusicGen with transformers only (no audiocraft)")
print("=" * 50)
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")

# Test 2: Check if TensorFlow is absent (good!)
try:
    import tensorflow
    print("⚠️  WARNING: TensorFlow is installed (not needed)")
except ImportError:
    print("✅ TensorFlow not installed (good!)")

# Test 3: Import transformers components
try:
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    print("✅ Transformers MusicGen imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 4: Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Test 5: Load smallest model
print("\n" + "=" * 50)
print("Loading MusicGen model...")
print("=" * 50)

model_name = "facebook/musicgen-small"
try:
    start_time = time.time()
    
    # Load processor and model
    processor = AutoProcessor.from_pretrained(model_name)
    model = MusicgenForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    model.to(device)
    
    load_time = time.time() - start_time
    print(f"✅ Model loaded successfully in {load_time:.1f}s")
    
    # Print model info
    print(f"\nModel info:")
    print(f"- Name: {model_name}")
    print(f"- Sample rate: {model.config.audio_encoder.sampling_rate} Hz")
    print(f"- Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    sys.exit(1)

# Test 6: Basic inference
print("\n" + "=" * 50)
print("Testing basic inference...")
print("=" * 50)

test_prompt = "A cheerful acoustic guitar melody with a steady rhythm"
duration_seconds = 5.0

try:
    # Process prompt
    inputs = processor(
        text=[test_prompt],
        padding=True,
        return_tensors="pt"
    ).to(device)
    
    # Calculate tokens for duration
    max_new_tokens = int(256 * duration_seconds / 5)
    
    print(f"Prompt: '{test_prompt}'")
    print(f"Duration: {duration_seconds}s")
    print(f"Generating {max_new_tokens} tokens...")
    
    # Generate audio
    start_time = time.time()
    with torch.no_grad():
        audio_values = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            guidance_scale=3.0
        )
    
    gen_time = time.time() - start_time
    
    # Extract audio
    audio = audio_values[0, 0].cpu().numpy()
    sample_rate = model.config.audio_encoder.sampling_rate
    
    print(f"\n✅ Generation successful!")
    print(f"- Generation time: {gen_time:.1f}s")
    print(f"- Speed: {duration_seconds/gen_time:.1f}x realtime")
    print(f"- Audio shape: {audio.shape}")
    print(f"- Audio duration: {len(audio)/sample_rate:.1f}s")
    print(f"- Audio range: [{audio.min():.3f}, {audio.max():.3f}]")
    
    # Optional: Save audio
    save_audio = input("\nSave audio file? (y/n): ").lower().strip() == 'y'
    if save_audio:
        import scipy.io.wavfile as wavfile
        
        # Normalize and convert to 16-bit
        audio = np.clip(audio, -1, 1)
        audio_16bit = (audio * 32767).astype(np.int16)
        
        filename = "minimal_test_output.wav"
        wavfile.write(filename, sample_rate, audio_16bit)
        print(f"✅ Saved to {filename}")
    
except Exception as e:
    print(f"❌ Inference failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Memory usage
if device == "cuda":
    print(f"\nGPU Memory used: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

print("\n" + "=" * 50)
print("✅ All tests passed! MusicGen works with transformers only.")
print("No audiocraft dependency needed for basic usage.")
print("=" * 50)