# PyTorch-Only Transformers Installation for MusicGen

This document provides the complete solution for installing transformers with only PyTorch dependencies, avoiding TensorFlow entirely for MusicGen/AudioCraft usage.

## Problem Summary

The issue occurs when transformers is installed with TensorFlow dependencies, causing:
- Recursion errors during import
- Unnecessary TensorFlow dependencies
- Conflicts between PyTorch and TensorFlow backends
- Large installation footprint

## Solution Overview

Install transformers with PyTorch-only dependencies using the `[torch]` extras specification.

## Quick Solution

### 1. Remove TensorFlow (Recommended)

```bash
pip uninstall tensorflow tensorflow-datasets tensorflow_estimator tensorflow-metadata tensorboard -y
```

### 2. Install PyTorch First

```bash
# For CPU-only (recommended for most use cases)
pip install torch>=2.2.0 torchaudio>=2.2.0 --index-url https://download.pytorch.org/whl/cpu

# For GPU support (if you have CUDA)
pip install torch>=2.2.0 torchaudio>=2.2.0 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Transformers with PyTorch Extras

```bash
pip install 'transformers[torch]>=4.31.0'
```

### 4. Install MusicGen Dependencies

```bash
pip install audiocraft scipy>=1.14.0 soundfile>=0.12.0 librosa>=0.10.0 pydub>=0.25.0
```

## Automated Installation

### Option 1: Using the Shell Script

```bash
# Make executable and run
chmod +x pytorch_only_install.sh
./pytorch_only_install.sh
```

### Option 2: Using the Python Script

```bash
python install_transformers_pytorch_only.py
```

### Option 3: Using Requirements File

```bash
# First remove TensorFlow manually
pip uninstall tensorflow tensorflow-datasets tensorflow_estimator tensorflow-metadata tensorboard -y

# Then install from requirements file
pip install -r requirements-pytorch-only.txt
```

## Key Installation Commands

| Command | Purpose |
|---------|---------|
| `pip install 'transformers[torch]>=4.31.0'` | Install transformers with PyTorch extras only |
| `pip install torch>=2.2.0 torchaudio>=2.2.0` | Install PyTorch CPU version |
| `pip uninstall tensorflow*` | Remove TensorFlow packages |
| `pip install audiocraft` | Install Facebook's AudioCraft (includes MusicGen) |

## Understanding the `[torch]` Extra

The `transformers[torch]` syntax installs transformers with PyTorch-specific dependencies:

- ✅ Includes PyTorch support
- ✅ Excludes TensorFlow dependencies  
- ✅ Lighter installation footprint
- ✅ Avoids backend conflicts

## Verification

### Test Your Installation

```bash
python test_pytorch_only_setup.py
```

### Manual Verification

```python
# Test 1: PyTorch works
import torch
print(f"PyTorch version: {torch.__version__}")

# Test 2: Transformers works
from transformers import pipeline
print("Transformers pipeline import successful")

# Test 3: TensorFlow is NOT imported
try:
    import tensorflow
    print("WARNING: TensorFlow is still installed")
except ImportError:
    print("SUCCESS: TensorFlow is not installed")

# Test 4: Basic functionality
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
print("Basic transformers functionality works")
```

## Troubleshooting

### Common Issues

1. **"No module named 'tensorflow'"** - This is expected and good!
2. **"RecursionError"** - Usually caused by TensorFlow conflicts, remove TensorFlow completely
3. **"ImportError: torch"** - Install PyTorch first before transformers
4. **Large download sizes** - Models download on first use, this is normal

### Environment-Specific Solutions

#### For Python 3.12
```bash
pip install torch>=2.2.0 --index-url https://download.pytorch.org/whl/cpu
pip install 'transformers[torch]>=4.31.0'
```

#### For Virtual Environments
```bash
# Create clean environment
python -m venv musicgen_env
source musicgen_env/bin/activate  # On Windows: musicgen_env\Scripts\activate

# Install in clean environment
pip install -r requirements-pytorch-only.txt
```

#### For Conda Environments
```bash
# Create conda environment
conda create -n musicgen python=3.12
conda activate musicgen

# Install PyTorch via conda
conda install pytorch torchaudio pytorch-cuda -c pytorch -c nvidia

# Install transformers via pip
pip install 'transformers[torch]>=4.31.0'
```

## Integration with MusicGen

### Example Usage

```python
from transformers import pipeline
import torch

# Verify setup
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Use with MusicGen (requires audiocraft)
try:
    from audiocraft.models import MusicGen
    model = MusicGen.get_pretrained('facebook/musicgen-small')
    print("MusicGen loaded successfully with PyTorch-only transformers")
except ImportError:
    print("Install audiocraft: pip install audiocraft")
```

### Performance Optimization

For better performance with MusicGen:

```python
# Enable torch optimizations
torch.backends.cudnn.benchmark = True  # If using GPU
torch.set_num_threads(4)  # Adjust based on your CPU
```

## Files Created

This solution includes these files:

1. **`pytorch_only_install.sh`** - Automated installation script
2. **`install_transformers_pytorch_only.py`** - Python installation script  
3. **`requirements-pytorch-only.txt`** - PyTorch-only requirements file
4. **`test_pytorch_only_setup.py`** - Verification test script
5. **`PYTORCH_ONLY_TRANSFORMERS_SOLUTION.md`** - This documentation

## Benefits of PyTorch-Only Setup

- ✅ **Smaller footprint**: No TensorFlow dependencies (~2GB savings)
- ✅ **Faster imports**: No TensorFlow backend initialization
- ✅ **Better compatibility**: Avoids PyTorch/TensorFlow conflicts
- ✅ **Cleaner environment**: Single ML framework
- ✅ **MusicGen optimized**: PyTorch is the native backend for MusicGen

## Official Documentation References

- [Hugging Face Transformers Installation](https://huggingface.co/docs/transformers/installation)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [AudioCraft Documentation](https://github.com/facebookresearch/audiocraft)

## Next Steps

1. Run the installation script or commands
2. Test with `python test_pytorch_only_setup.py`
3. Try MusicGen with your PyTorch-only setup
4. Report any issues for further troubleshooting

This setup ensures transformers works efficiently with MusicGen using only PyTorch dependencies, avoiding TensorFlow entirely.