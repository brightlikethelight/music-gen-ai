# MusicGen Unified

[![CI Pipeline](https://github.com/brightlikethelight/music-gen-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/brightlikethelight/music-gen-ai/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/brightlikethelight/music-gen-ai/graph/badge.svg)](https://codecov.io/gh/brightlikethelight/music-gen-ai)
[![Python 3.10 | 3.11 | 3.12](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An AI music generation system built on Facebook's MusicGen model. Developed as part of Harvard's CS 109B: Advanced Data Science course.

**Note**: This is an educational project. See [Limitations](docs/LIMITATIONS.md) for details.

## Requirements

- Python 3.10, 3.11, or 3.12
- 8GB+ RAM (16GB+ recommended)
- GPU with 8GB+ VRAM (optional, but recommended)

## Installation

```bash
git clone https://github.com/brightlikethelight/music-gen-ai.git
cd music-gen-ai
pip install -e ".[dev]"
```

## Quick Start

### Command Line

```bash
# Generate music (first run downloads models, ~2GB)
python -m musicgen.cli.main generate "upbeat jazz piano" --duration 30

# View system info
python -m musicgen.cli.main info
```

### Python API

```python
from musicgen.core.generator import MusicGenerator

generator = MusicGenerator()
audio, sample_rate = generator.generate("peaceful acoustic guitar", duration=30.0)
generator.save_audio(audio, sample_rate, "output.wav")
```

### REST API

```bash
python -m musicgen.api.rest.app
# API docs at http://localhost:8000/docs
```

## Features

- **Text-to-Music Generation**: Convert text descriptions to audio
- **Multiple Interfaces**: CLI, REST API, Python library
- **Authentication**: JWT-based user authentication
- **Rate Limiting**: Request throttling for API protection
- **Batch Processing**: Process multiple generation requests

## Project Structure

```
music-gen-ai/
├── src/musicgen/
│   ├── core/              # Music generation engine
│   ├── api/               # REST API (FastAPI)
│   ├── cli/               # Command-line interface
│   ├── services/          # Background processing
│   └── infrastructure/    # Config, logging, security
├── tests/
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
├── docs/
│   ├── cs109b/            # Academic materials
│   └── technical/         # Technical documentation
├── deployment/            # Docker and deployment configs
└── scripts/               # Utility scripts
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=src/musicgen

# Code formatting
black src tests
isort src tests

# Type checking
mypy src
```

## Technical Details

| Component | Details |
|-----------|---------|
| Base Model | Facebook MusicGen (300M - 3.3B parameters) |
| Audio Output | 32kHz, 16-bit WAV |
| Framework | PyTorch 2.2+, Transformers 4.43+ |
| API Framework | FastAPI with Uvicorn |
| Authentication | JWT with bcrypt password hashing |

## Academic Context

This project was developed for Harvard CS 109B, demonstrating:

- Transformer models for audio generation
- RESTful API design patterns
- Authentication and security implementation
- Software engineering for ML applications

See [Final Presentation](docs/cs109b/CS_109B_Final_Presentation.pdf) and [Implementation Notebook](docs/cs109b/CS_109B_Final_Notebook.ipynb) for details.

## Documentation

- [Architecture](docs/technical/ARCHITECTURE.md)
- [Limitations](docs/LIMITATIONS.md)
- [Known Issues](docs/KNOWN_ISSUES.md)
- [Contributing](CONTRIBUTING.md)
- [Security Policy](SECURITY.md)
- [Changelog](CHANGELOG.md)

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Facebook Research](https://github.com/facebookresearch/audiocraft) for the MusicGen model
- Harvard CS 109B teaching staff

## Contact

- **Author**: Bright Liu
- **Email**: brightliu@college.harvard.edu
- **Course**: Harvard CS 109B
