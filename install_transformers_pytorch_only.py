#!/usr/bin/env python3
"""
Script to install transformers with PyTorch-only dependencies for MusicGen.
This script removes TensorFlow dependencies and installs transformers correctly.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(f"Return code: {result.returncode}")
        
        if result.stdout:
            print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
            
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} is not compatible (need 3.9+)")
        return False

def main():
    """Main installation process."""
    print("🎵 MusicGen PyTorch-only Transformers Installation")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        print("Please upgrade Python to 3.9+ and try again.")
        sys.exit(1)
    
    # Show current environment info
    print(f"Python executable: {sys.executable}")
    print(f"Working directory: {os.getcwd()}")
    
    # Step 1: Remove TensorFlow packages (optional, but recommended)
    print("\n🗑️  STEP 1: Removing TensorFlow packages")
    tensorflow_packages = [
        "tensorflow",
        "tensorflow-datasets",
        "tensorflow_estimator", 
        "tensorflow-metadata",
        "tensorboard"
    ]
    
    for package in tensorflow_packages:
        cmd = f"pip uninstall {package} -y"
        success = run_command(cmd, f"Removing {package}")
        if not success:
            print(f"Note: {package} was not installed or failed to remove")
    
    # Step 2: Install PyTorch first (to ensure correct version)
    print("\n🔥 STEP 2: Installing PyTorch")
    pytorch_cmd = "pip install torch>=2.2.0 torchaudio>=2.2.0 --index-url https://download.pytorch.org/whl/cpu"
    
    success = run_command(pytorch_cmd, "Installing PyTorch CPU version")
    if not success:
        print("Failed to install PyTorch. Trying standard installation...")
        fallback_cmd = "pip install torch>=2.2.0 torchaudio>=2.2.0"
        success = run_command(fallback_cmd, "Installing PyTorch (fallback)")
        if not success:
            print("❌ Failed to install PyTorch")
            sys.exit(1)
    
    # Step 3: Install transformers with PyTorch extras
    print("\n🤗 STEP 3: Installing Transformers with PyTorch support")
    transformers_cmd = "pip install 'transformers[torch]>=4.31.0'"
    
    success = run_command(transformers_cmd, "Installing transformers with PyTorch extras")
    if not success:
        print("Failed to install transformers with extras. Trying basic installation...")
        fallback_cmd = "pip install transformers>=4.31.0"
        success = run_command(fallback_cmd, "Installing transformers (fallback)")
        if not success:
            print("❌ Failed to install transformers")
            sys.exit(1)
    
    # Step 4: Install additional dependencies for MusicGen
    print("\n🎼 STEP 4: Installing MusicGen dependencies")
    musicgen_deps = [
        "audiocraft",
        "scipy>=1.14.0",
        "soundfile>=0.12.0",
        "librosa>=0.10.0",
        "pydub>=0.25.0"
    ]
    
    for dep in musicgen_deps:
        cmd = f"pip install '{dep}'"
        success = run_command(cmd, f"Installing {dep}")
        if not success:
            print(f"⚠️  Warning: Failed to install {dep}")
    
    # Step 5: Verify installation
    print("\n✅ STEP 5: Verifying installation")
    
    # Test PyTorch
    test_pytorch = """
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
"""
    
    # Test transformers
    test_transformers = """
import transformers
print(f"Transformers version: {transformers.__version__}")
from transformers import pipeline
print("✓ Transformers pipeline import successful")
"""
    
    # Test basic functionality
    test_basic = """
try:
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    print("✓ AutoTokenizer works")
    print("✓ Installation successful!")
except Exception as e:
    print(f"✗ Test failed: {e}")
"""
    
    print("\nTesting PyTorch...")
    result = subprocess.run([sys.executable, "-c", test_pytorch], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"PyTorch warnings: {result.stderr}")
    
    print("\nTesting Transformers...")
    result = subprocess.run([sys.executable, "-c", test_transformers], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Transformers warnings: {result.stderr}")
    
    print("\nTesting basic functionality...")
    result = subprocess.run([sys.executable, "-c", test_basic], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Basic test warnings: {result.stderr}")
    
    # Final package list
    print("\n📦 Final package versions:")
    run_command("pip list | grep -E '(torch|transformers|audiocraft)'", "Checking installed packages")
    
    print("\n🎉 Installation complete!")
    print("\nNext steps:")
    print("1. Test MusicGen functionality")
    print("2. If you encounter issues, check the error messages above")
    print("3. For GPU support, install CUDA-enabled PyTorch separately")

if __name__ == "__main__":
    main()