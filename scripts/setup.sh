#!/bin/bash
# MusicGen Unified - Development Environment Setup Script

set -e

echo "🎵 MusicGen Unified - Development Setup"
echo "======================================"

# Check Python version
python_version=$(python3 --version | cut -d' ' -f2)
echo "Python version: $python_version"

if ! python3 -c "import sys; assert sys.version_info >= (3, 10)" 2>/dev/null; then
    echo "❌ Error: Python 3.10+ is required"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -e .[dev,deployment]

# Install pre-commit hooks
if command -v pre-commit &> /dev/null; then
    echo "Setting up pre-commit hooks..."
    pre-commit install
else
    echo "Installing pre-commit..."
    pip install pre-commit
    pre-commit install
fi

# Create necessary directories
echo "Creating project directories..."
mkdir -p outputs
mkdir -p logs
mkdir -p models

# Download default model (optional)
echo "Would you like to download the default model? (y/n)"
read -r download_model

if [ "$download_model" = "y" ] || [ "$download_model" = "Y" ]; then
    echo "Downloading default model..."
    python3 -c "
from musicgen import MusicGenerator
print('Initializing generator to download model...')
generator = MusicGenerator()
print('Model downloaded successfully!')
"
fi

# Run basic tests
echo "Running basic tests..."
python3 -m pytest tests/test_basic_import.py -v

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the CLI:"
echo "  musicgen --help"
echo ""
echo "To start the web interface:"
echo "  musicgen serve"
echo ""