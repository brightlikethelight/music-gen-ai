# MusicGen Unified v2.0.1

A production-grade AI music generation library with clean architecture and comprehensive features.

## Quick Start

```bash
# Install the library
pip install musicgen-unified

# Generate music from command line
musicgen generate "upbeat jazz piano" --duration 15 --output jazz_piano.wav

# Start the web interface
musicgen serve

# Start the API server
musicgen api
```

## Features

- **Multiple Interfaces**: CLI, Web UI, REST API, and Python library
- **Batch Processing**: Process multiple requests efficiently
- **Prompt Engineering**: Enhanced prompts for better results  
- **Production Ready**: Docker, Kubernetes, monitoring, and logging
- **Extensible Architecture**: Clean separation of concerns

## Architecture

The library follows a modern layered architecture:

```
src/musicgen/
├── core/              # Core business logic
│   ├── generator.py   # Main generation engine
│   ├── prompt.py      # Prompt engineering
│   └── audio/         # Audio processing
├── api/               # API layer
│   ├── rest/          # REST API endpoints
│   └── streaming/     # WebSocket/SSE APIs
├── services/          # Business services
│   └── batch.py       # Batch processing
├── infrastructure/    # Cross-cutting concerns
│   ├── config/        # Configuration management
│   ├── monitoring/    # Metrics and logging
│   └── security/      # Security utilities
├── cli/               # Command line interface
├── web/               # Web interface
└── utils/             # Shared utilities
```

## Installation

### From PyPI

```bash
pip install musicgen-unified
```

### Development Installation

```bash
git clone https://github.com/brightlikethelight/music-gen-ai
cd music-gen-ai
./scripts/setup.sh
```

## Usage Examples

### Python Library

```python
from musicgen import MusicGenerator

# Initialize generator
generator = MusicGenerator()

# Generate music
audio, sample_rate = generator.generate(
    prompt="relaxing ambient music",
    duration=30.0
)

# Save audio
generator.save_audio(audio, "output.wav", sample_rate)
```

### Batch Processing

```python
from musicgen import BatchProcessor
import pandas as pd

# Create batch data
df = pd.DataFrame([
    {"prompt": "jazz piano", "duration": 15, "output_filename": "jazz.wav"},
    {"prompt": "rock guitar", "duration": 20, "output_filename": "rock.wav"}
])
df.to_csv("batch.csv", index=False)

# Process batch
processor = BatchProcessor()
results = processor.process_batch("batch.csv", "outputs/")
```

### REST API

```python
import requests

# Generate music via API
response = requests.post("http://localhost:8000/generate", json={
    "prompt": "electronic dance music",
    "duration": 15.0,
    "format": "wav"
})

with open("edm_track.wav", "wb") as f:
    f.write(response.content)
```

## Configuration

The library supports configuration via YAML files and environment variables:

```yaml
# configs/production.yaml
environment: production

models:
  default: facebook/musicgen-medium
  cache_dir: /app/models

generation:
  max_duration: 120.0
  batch_size: 4

api:
  host: 0.0.0.0
  port: 8000
  workers: 4
```

## Deployment

### Docker

```bash
# Build and run
docker build -t musicgen .
docker run -p 8000:8000 musicgen
```

### Kubernetes

```bash
# Apply manifests
kubectl apply -f deployment/kubernetes/
```

### Docker Compose

```bash
# Full stack with Redis and monitoring
docker-compose -f deployment/docker/docker-compose.yml up
```

## Monitoring

The library includes comprehensive monitoring:

- **Metrics**: Prometheus metrics for API usage, generation times, and errors
- **Logging**: Structured logging with configurable formats
- **Health Checks**: Built-in health check endpoints
- **Tracing**: Optional distributed tracing support

Access monitoring at:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

## API Documentation

Interactive API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Run with coverage
pytest --cov=musicgen --cov-report=html
```

## Development

### Project Structure

The project follows modern Python packaging standards:

- **src/**: Source code with namespace packages
- **tests/**: Comprehensive test suite (unit, integration, e2e)
- **docs/**: Documentation and guides
- **examples/**: Usage examples and tutorials
- **deployment/**: Docker, Kubernetes, and infrastructure configs
- **configs/**: Environment-specific configuration files

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run the test suite: `pytest`
5. Submit a pull request

### Code Style

The project uses:
- **Black** for code formatting
- **isort** for import sorting
- **mypy** for type checking
- **flake8** for linting

Run all checks:
```bash
black .
isort .
mypy .
flake8 .
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

- **GitHub Issues**: Bug reports and feature requests
- **Documentation**: Comprehensive guides and API docs
- **Examples**: Working code examples in the `examples/` directory

---

Built with ❤️ for the AI music generation community.