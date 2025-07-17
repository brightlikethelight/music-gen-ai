# 🎵 MusicGen AI - Advanced Music Generation System

<div align="center">
  
[![PyPI version](https://badge.fury.io/py/musicgen-unified.svg)](https://badge.fury.io/py/musicgen-unified)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Harvard CS 109B](https://img.shields.io/badge/Harvard-CS%20109B-crimson.svg)](docs/cs109b/)

**State-of-the-art AI music generation system developed as part of Harvard CS 109B Advanced Data Science**

[📊 View CS 109B Presentation](docs/cs109b/CS_109B_Final_Presentation.pdf) | [📓 Explore Notebook](docs/cs109b/CS_109B_Final_Notebook.ipynb) | [🚀 Quick Start](#quick-start) | [📖 Documentation](docs/)

</div>

---

## 🎓 Academic Excellence

This project was developed as the final project for **Harvard's CS 109B: Advanced Data Science** course. It demonstrates cutting-edge machine learning techniques applied to music generation.

<div align="center">
  <table>
    <tr>
      <td align="center">
        <h3>📊 Final Presentation</h3>
        <a href="docs/cs109b/CS_109B_Final_Presentation.pdf">
          <img src="https://img.shields.io/badge/View-Presentation-red.svg?style=for-the-badge" alt="View Presentation">
        </a>
        <br><sub>Comprehensive overview of the project</sub>
      </td>
      <td align="center">
        <h3>📓 Implementation Notebook</h3>
        <a href="docs/cs109b/CS_109B_Final_Notebook.ipynb">
          <img src="https://img.shields.io/badge/Open-Notebook-orange.svg?style=for-the-badge" alt="Open Notebook">
        </a>
        <br><sub>Complete code and analysis</sub>
      </td>
    </tr>
  </table>
</div>

## ✨ Features

- 🎵 **High-Quality Music Generation** - Create 30+ second instrumental tracks from text descriptions
- 🚀 **Production-Ready** - Deployed on PyPI with comprehensive API and CLI interfaces
- 🔧 **Advanced Architecture** - Built on Facebook's MusicGen with custom enhancements
- 📊 **Batch Processing** - Generate multiple tracks simultaneously with parallel processing
- 🎯 **Prompt Engineering** - AI-powered prompt enhancement for better results
- 🌐 **Multiple Interfaces** - CLI, REST API, and Web UI

## 🚀 Quick Start

### Installation

```bash
pip install musicgen-unified
```

### Basic Usage

```bash
# Generate music from text
musicgen generate "upbeat jazz piano with drums"

# Start web interface
musicgen serve

# Process batch jobs
musicgen batch playlist.csv
```

## 🏗️ Architecture

```
music_gen/
├── src/musicgen/       # Core package implementation
├── docs/               # Comprehensive documentation
│   └── cs109b/        # Harvard CS 109B course materials ⭐
├── examples/          # Usage examples and demos
├── tests/            # Test suite
└── docker/           # Containerization configs
```

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [API Reference](docs/api/)
- [Configuration Options](docs/configuration.md)
- [Advanced Usage](docs/advanced.md)
- [CS 109B Project Details](docs/cs109b/)

## 🧪 Technical Specifications

- **Models**: Small (300M), Medium (1.5B), Large (3.3B) parameters
- **Audio Quality**: 32kHz, 16-bit PCM/MP3
- **Performance**: 0.1x-1.0x realtime (hardware dependent)
- **Dependencies**: PyTorch 2.2+, Transformers 4.43+, NumPy 1.26.x

## 🚀 Deployment

### Docker

```bash
docker run -p 8080:8080 musicgen-unified:latest
```

### Kubernetes

```bash
kubectl apply -f kubernetes/deployment.yaml
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Harvard CS 109B teaching staff for guidance and support
- Facebook Research for the original MusicGen model
- The open-source community for invaluable tools and libraries

## 📬 Contact

- **Author**: Bright Liu (brightliu@college.harvard.edu)
- **Course**: CS 109B - Advanced Data Science, Harvard University
- **GitHub**: [brightlikethelight/music-gen-ai](https://github.com/brightlikethelight/music-gen-ai)

---

<div align="center">
  <i>Developed with ❤️ at Harvard University</i>
</div>
