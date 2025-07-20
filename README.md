# 🎵 MusicGen Unified - Production-Ready AI Music Generation

[![PyPI version](https://badge.fury.io/py/musicgen-unified.svg)](https://badge.fury.io/py/musicgen-unified)
[![Python](https://img.shields.io/badge/python-3.10_|_3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://hub.docker.com/)

> **⚠️ CRITICAL**: This project requires Python 3.10 or 3.11. Python 3.12 is incompatible with ML dependencies.

A production-ready, unified interface for AI music generation using Facebook's MusicGen model. Features async APIs, Docker deployment, monitoring, and enterprise-grade architecture.

**Academic Context**: This work was inspired by and developed as part of Harvard's CS 109B: Advanced Data Science course, demonstrating practical applications of machine learning in audio generation.

## Installation

### Option 1: PyPI Package
```bash
pip install musicgen-unified
```

### Option 2: Docker (Recommended)
```bash
# Pre-built image with all dependencies
docker pull ashleykza/tts-webui:latest
docker run -d --gpus all -p 3000:3001 ashleykza/tts-webui:latest
```

### Option 3: From Source
```bash
# Requires Python 3.10 or 3.11
git clone https://github.com/brightlikethelight/music-gen-ai.git
cd music-gen-ai
./deploy.sh  # Automated deployment script
```

## 🚀 Production Features

- **Async API**: FastAPI with background task processing
- **Job Tracking**: Real-time generation status and progress
- **Monitoring**: Prometheus metrics and Grafana dashboards
- **Docker Support**: Multi-service deployment with nginx
- **Auto-scaling**: Kubernetes-ready with health checks
- **Configuration**: Environment-based settings management

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
│   ├── core/              # Business logic (generator, prompt)
│   ├── api/               # REST API with FastAPI
│   ├── services/          # Batch processing, background tasks
│   ├── infrastructure/    # Config, logging, monitoring
│   ├── cli/               # Command line interface
│   └── utils/             # Shared utilities
├── deployment/            # Docker, Kubernetes configs
├── docs/                  # Documentation
│   └── cs109b/           # Academic project materials
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── e2e/              # End-to-end tests
└── examples/             # Usage examples
```

## 🛠️ Development Tools

- **`validate_deployment.py`**: Check environment compatibility
- **`deploy.sh`**: Automated deployment with 4 options
- **`docker-compose.yml`**: Multi-service production stack
- **Test API**: `src/musicgen/api/rest/test_app.py` for testing without ML deps

## CS 109B Academic Materials

This project includes materials from Harvard's CS 109B course:

- **[Final Presentation](docs/cs109b/CS_109B_Final_Presentation.pdf)**: Project overview and technical approach
- **[Implementation Notebook](docs/cs109b/CS_109B_Final_Notebook.ipynb)**: Complete analysis and implementation details

## Documentation

See the `docs/cs109b/` directory for academic project materials and implementation details.

## 🚨 Troubleshooting

### Python 3.12 Compatibility
```bash
# Python 3.12 does NOT work. Install Python 3.10:
curl https://pyenv.run | bash
pyenv install 3.10.14
pyenv local 3.10.14
```

### Environment Validation
```bash
# Run validation script to check your setup
python validate_deployment.py
```

### Common Issues
- **TensorFlow recursion error**: Remove TensorFlow, use PyTorch only
- **AudioCraft installation fails**: Use Docker or Python 3.10
- **Out of memory**: Use smaller model (musicgen-small)
- **No GPU**: Expect ~15x slower generation on CPU

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## 📚 Additional Documentation

- **[Production Deployment Guide](PRODUCTION_DEPLOYMENT_GUIDE.md)**: Comprehensive deployment instructions
- **[Architecture Overview](ARCHITECTURE.md)**: System design and patterns
- **[API Reference](http://localhost:8000/docs)**: Interactive API documentation

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