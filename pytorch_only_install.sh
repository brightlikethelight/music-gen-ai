#!/bin/bash
# Install transformers with PyTorch-only dependencies for MusicGen
# This script removes TensorFlow and installs the correct PyTorch-only setup

set -e  # Exit on any error

echo "🎵 MusicGen PyTorch-only Transformers Installation"
echo "=================================================="

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Step 1: Remove TensorFlow (optional but recommended)
echo ""
echo "🗑️  STEP 1: Removing TensorFlow packages"
echo "----------------------------------------"
pip uninstall tensorflow tensorflow-datasets tensorflow_estimator tensorflow-metadata tensorboard -y || true

# Step 2: Install PyTorch first (ensure correct version)
echo ""
echo "🔥 STEP 2: Installing PyTorch"
echo "-----------------------------"
# For CPU-only (recommended for most use cases)
pip install torch>=2.2.0 torchaudio>=2.2.0 --index-url https://download.pytorch.org/whl/cpu

# For GPU support (uncomment if needed)
# pip install torch>=2.2.0 torchaudio>=2.2.0 --index-url https://download.pytorch.org/whl/cu121

# Step 3: Install transformers with PyTorch extras
echo ""
echo "🤗 STEP 3: Installing Transformers with PyTorch support"
echo "------------------------------------------------------"
pip install 'transformers[torch]>=4.31.0'

# Step 4: Install additional MusicGen dependencies
echo ""
echo "🎼 STEP 4: Installing MusicGen dependencies"
echo "-------------------------------------------"
pip install audiocraft
pip install 'scipy>=1.14.0'
pip install 'soundfile>=0.12.0'
pip install 'librosa>=0.10.0'
pip install 'pydub>=0.25.0'

# Step 5: Verify installation
echo ""
echo "✅ STEP 5: Verifying installation"
echo "---------------------------------"

echo "Testing PyTorch..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo "Testing Transformers..."
python -c "import transformers; print(f'Transformers version: {transformers.__version__}'); from transformers import pipeline; print('✓ Transformers pipeline import successful')"

echo "Testing basic functionality..."
python -c "
try:
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    print('✓ AutoTokenizer works')
    print('✓ Installation successful!')
except Exception as e:
    print(f'✗ Test failed: {e}')
"

echo ""
echo "📦 Final package versions:"
pip list | grep -E '(torch|transformers|audiocraft)' || echo "No matching packages found"

echo ""
echo "🎉 Installation complete!"
echo ""
echo "Next steps:"
echo "1. Test MusicGen functionality"
echo "2. If you encounter issues, check the error messages above"
echo "3. For GPU support, modify the PyTorch installation command"
echo ""
echo "Key commands used:"
echo "- pip install 'transformers[torch]>=4.31.0'"
echo "- pip install torch>=2.2.0 torchaudio>=2.2.0"
echo "- pip uninstall tensorflow* (removes TensorFlow)"