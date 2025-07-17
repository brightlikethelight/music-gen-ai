# MusicGen Unified

[![PyPI version](https://badge.fury.io/py/musicgen-unified.svg)](https://badge.fury.io/py/musicgen-unified)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A unified interface for AI music generation using Facebook's MusicGen model. This project provides a clean, easy-to-use command-line interface and Python API for generating music from text descriptions.

**Academic Context**: This work was inspired by and developed as part of Harvard's CS 109B: Advanced Data Science course, demonstrating practical applications of machine learning in audio generation.

## Installation

```bash
pip install musicgen-unified
```

## Quick Start

### Command Line Interface

```bash
# Generate music from text description
musicgen generate "upbeat jazz piano with drums" --duration 30

# Start web interface
musicgen serve

# Process multiple generations from CSV
musicgen batch input.csv

# View system information
musicgen info
```

### Python API

```python
from musicgen import MusicGenerator

# Initialize generator
generator = MusicGenerator()

# Generate music
audio = generator.generate(
    prompt="peaceful acoustic guitar melody",
    duration=30.0
)

# Save to file
generator.save_audio(audio, "output.wav")
```

## Features

- **Text-to-Music Generation**: Create instrumental music from natural language descriptions
- **Multiple Interfaces**: Command-line tool, Python API, and web interface
- **Batch Processing**: Generate multiple tracks efficiently
- **Prompt Engineering**: Built-in prompt enhancement for better results
- **Multiple Output Formats**: WAV, MP3 support

## Technical Details

- **Model**: Facebook's MusicGen (Small: 300M, Medium: 1.5B, Large: 3.3B parameters)
- **Audio Quality**: 32kHz, 16-bit
- **Dependencies**: PyTorch 2.2+, Transformers 4.43+, NumPy 1.26.x
- **Python**: 3.10+

## Project Structure

```
musicgen-unified/
├── src/musicgen/           # Core package
├── docs/                   # Documentation
│   └── cs109b/            # Academic project materials
├── tests/                  # Test suite
└── examples/              # Usage examples
```

## CS 109B Academic Materials

This project includes materials from Harvard's CS 109B course:

- **[Final Presentation](docs/cs109b/CS_109B_Final_Presentation.pdf)**: Project overview and technical approach
- **[Implementation Notebook](docs/cs109b/CS_109B_Final_Notebook.ipynb)**: Complete analysis and implementation details

## Documentation

See the `docs/cs109b/` directory for academic project materials and implementation details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Facebook Research for the original MusicGen model
- Harvard CS 109B teaching staff
- The open-source community

## Contact

- **Author**: Bright Liu
- **Email**: brightliu@college.harvard.edu
- **GitHub**: [brightlikethelight/music-gen-ai](https://github.com/brightlikethelight/music-gen-ai)
- **PyPI**: [musicgen-unified](https://pypi.org/project/musicgen-unified/)