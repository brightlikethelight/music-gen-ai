# 🎓 MusicGen Unified - Academic Research & Educational Project

[![Python](https://img.shields.io/badge/python-3.10_|_3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Academic](https://img.shields.io/badge/project-academic-orange.svg)](https://github.com/brightlikethelight/music-gen-ai)

> **🎓 ACADEMIC PROJECT**: This is an educational research project developed for Harvard's CS 109B: Advanced Data Science course. It is **NOT production-ready** and is intended for learning and experimentation purposes only.

> **⚠️ CRITICAL**: This project requires Python 3.10 or 3.11. Python 3.12 is incompatible with ML dependencies.

An academic research project exploring AI music generation using Facebook's MusicGen model, developed as part of Harvard CS 109B. This project demonstrates practical applications of machine learning in audio generation with a focus on educational value and learning objectives.

**Inspired by**: [Meta's AudioCraft](https://github.com/facebookresearch/audiocraft) - The original research implementation of MusicGen by Facebook Research.

## 🚨 Important Disclaimers

- **Experimental Software**: This is a learning project with limited testing and known issues
- **Not Production-Ready**: Current test coverage is ~13%, many features are incomplete
- **Educational Purpose**: Created for academic exploration, not commercial use
- **No Warranty**: Provided "as-is" for educational purposes only

See [ACADEMIC_DISCLAIMER.md](ACADEMIC_DISCLAIMER.md) for complete terms and limitations.

## Installation

### Option 1: From Source (Recommended for Learning)
```bash
# Requires Python 3.10 or 3.11
git clone https://github.com/brightlikethelight/music-gen-ai.git
cd music-gen-ai
pip install -r requirements.txt
pip install -e .
```

### Option 2: Development Setup
```bash
# For development and experimentation
git clone https://github.com/brightlikethelight/music-gen-ai.git
cd music-gen-ai
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .
```

## 📚 Learning Objectives

This project was designed to demonstrate:

- **Text-to-Audio ML Models**: Understanding how transformers work with audio data
- **API Design Patterns**: RESTful APIs with FastAPI framework
- **Authentication Systems**: JWT-based authentication implementation
- **Testing Strategies**: Unit testing, integration testing (though coverage is limited)
- **Software Architecture**: Modular design patterns for ML applications
- **Docker Concepts**: Containerization for ML applications (examples provided)

## 🎯 What Actually Works

✅ **Basic Functionality:**
- Text-to-music generation using MusicGen models
- Command-line interface for simple generation
- Web API with authentication
- Integration tests pass (20/20)

✅ **Educational Components:**
- Code examples for ML model integration
- RESTful API design patterns
- Basic authentication implementation
- Docker configuration examples

❌ **Known Limitations (See [LIMITATIONS.md](LIMITATIONS.md)):**
- Test coverage only ~13% (not production standard)
- Many unit tests skip due to missing dependencies (65+ skipped, 1-2 failing)
- Monitoring features are placeholder examples
- No actual PyPI package published
- Docker setup requires manual configuration

## Quick Start

### Command Line Interface

```bash
# Generate music from text description (requires model download)
python -m musicgen.cli.main generate "upbeat jazz piano with drums" --duration 30

# View system information
python -m musicgen.cli.main info
```

### Python API (Educational Example)

```python
from musicgen.core.generator import MusicGenerator

# Initialize generator (downloads model on first use)
generator = MusicGenerator()

# Generate music
audio = generator.generate(
    prompt="peaceful acoustic guitar melody",
    duration=30.0
)

# Save to file
generator.save_audio(audio, "output.wav")
```

### Web API Server

```bash
# Start development server
python -m musicgen.api.rest.app

# API documentation available at: http://localhost:8000/docs
```

## Academic Context - Harvard CS 109B

This project was developed as part of Harvard's CS 109B: Advanced Data Science course, focusing on:

- **Machine Learning Applications**: Practical implementation of transformer models
- **Data Science Workflow**: From research to prototype implementation
- **Software Engineering**: Building maintainable ML applications
- **Responsible AI**: Understanding limitations and ethical considerations

### Course Materials

- **[Final Presentation](docs/cs109b/CS_109B_Final_Presentation.pdf)**: Project overview and technical approach
- **[Implementation Notebook](docs/cs109b/CS_109B_Final_Notebook.ipynb)**: Complete analysis and implementation details

## Technical Details

- **Base Model**: Facebook's MusicGen (Small: 300M, Medium: 1.5B, Large: 3.3B parameters)
- **Audio Quality**: 32kHz, 16-bit WAV output
- **Dependencies**: PyTorch 2.2+, Transformers 4.43+, NumPy 1.26.x
- **Python**: 3.10+ (3.12 NOT supported)
- **Hardware**: GPU recommended (16GB+ VRAM for larger models)

## Project Structure

```
music-gen-ai/
├── src/musicgen/           # Core package
│   ├── core/              # ML model integration
│   ├── api/               # REST API implementation
│   ├── cli/               # Command line interface
│   ├── services/          # Background processing
│   ├── infrastructure/    # Configuration & monitoring examples
│   └── utils/             # Shared utilities
├── tests/                 # Test suite (limited coverage)
│   ├── unit/             # Unit tests (many failing)
│   ├── integration/      # Integration tests (working)
│   └── test_complete_system.py  # Full system tests
├── docs/                  # Documentation
│   └── cs109b/           # Academic project materials
├── deployment/           # Docker examples (educational)
└── examples/             # Usage examples
```

## 🔧 Development Status

This project is in **early development** with the following status:

- **Integration Tests**: ✅ 20/20 passing (basic functionality works)
- **Unit Tests**: ❌ 50+ failing (needs significant work)
- **Code Coverage**: ❌ 6.2% (far below production standards)
- **CI/CD Pipeline**: ❌ Currently failing (code quality issues)
- **Documentation**: ✅ Comprehensive (academic focus)

## 🚨 Troubleshooting

### Python 3.12 Compatibility
```bash
# Python 3.12 does NOT work. Install Python 3.10:
pyenv install 3.10.14
pyenv local 3.10.14
```

### Model Download Issues
```bash
# Models are downloaded automatically on first use
# Ensure you have sufficient disk space (several GB)
# Internet connection required for initial download
```

### Common Issues
- **TensorFlow conflicts**: Remove TensorFlow, use PyTorch only
- **Out of memory**: Use smaller model (`musicgen-small`)
- **No GPU**: Expect significantly slower generation on CPU
- **Import errors**: Ensure all dependencies installed correctly

## Contributing

This is an academic project, but contributions for educational purposes are welcome!

Please see [CONTRIBUTING_ACADEMIC.md](CONTRIBUTING_ACADEMIC.md) for guidelines on:
- Setting up development environment
- Understanding the codebase limitations
- Making educational improvements
- Submitting issues and pull requests

## 📚 Educational Resources

### For Students:
- Study the integration tests to understand expected behavior
- Examine API design patterns in `src/musicgen/api/`
- Learn about ML model integration in `src/musicgen/core/`
- Explore authentication implementation in `src/musicgen/api/middleware/`

### For Instructors:
- This project demonstrates common challenges in ML engineering
- Good example of the gap between academic prototypes and production systems
- Useful for discussing software engineering practices in data science

## Comparison with Meta's AudioCraft

| Feature | Meta AudioCraft | This Project |
|---------|----------------|--------------|
| Purpose | Research & Production | Academic Learning |
| Test Coverage | Comprehensive | Limited (6.2%) |
| Documentation | Production-grade | Educational |
| Model Training | Yes | No (uses pre-trained) |
| Community | Large | Educational |
| Status | Actively maintained | Course project |

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- **Facebook Research** for the original MusicGen model and AudioCraft library
- **Harvard CS 109B** teaching staff for guidance and inspiration
- **The open-source community** for tools and libraries used in this project

## Contact

- **Author**: Bright Liu (Harvard Student)
- **Email**: brightliu@college.harvard.edu
- **Course**: Harvard CS 109B: Advanced Data Science
- **GitHub**: [brightlikethelight/music-gen-ai](https://github.com/brightlikethelight/music-gen-ai)

---

*This project represents a learning journey in applying machine learning to audio generation. While not production-ready, it demonstrates key concepts in ML engineering and provides a foundation for further exploration and learning.*