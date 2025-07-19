# Contributing to MusicGen AI

First off, thank you for considering contributing to MusicGen AI! It's people like you that make this project better for everyone.

## Code of Conduct

By participating in this project, you are expected to uphold our Code of Conduct:
- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Respect differing viewpoints and experiences

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When you create a bug report, include:

- **Clear title and description**
- **Steps to reproduce**
- **Expected behavior**
- **Actual behavior**
- **System information** (OS, Python version, GPU info)
- **Relevant logs or error messages**

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Use case**: Why is this enhancement needed?
- **Proposed solution**: How do you envision it working?
- **Alternatives considered**: What other solutions did you think about?
- **Additional context**: Mockups, examples, etc.

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Follow the setup instructions** in README.md
3. **Make your changes** following our coding standards
4. **Add tests** for any new functionality
5. **Ensure all tests pass** and coverage doesn't decrease
6. **Update documentation** as needed
7. **Submit a pull request** with a clear description

## Development Setup

### Prerequisites

- Python 3.9, 3.10, or 3.11 (NOT 3.12 due to ML ecosystem incompatibility)
- Docker Desktop (for containerized development)
- Git

### Local Development

```bash
# Clone your fork
git clone https://github.com/yourusername/music-gen-ai.git
cd music-gen-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Docker Development

```bash
# Use the unified deployment script
./deploy.sh

# Or build custom development image
docker build -t musicgen-dev -f Dockerfile.dev .
docker run -it --rm -v $(pwd):/app musicgen-dev
```

## Coding Standards

### Python Style Guide

We follow PEP 8 with these additions:
- Line length: 100 characters
- Use type hints for all functions
- Docstrings for all public functions (Google style)

### Code Quality Tools

```bash
# Format code
black src/ tests/ --line-length 100

# Sort imports
isort src/ tests/ --profile black

# Lint code
flake8 src/ tests/ --max-line-length=100

# Type checking
mypy src/ --ignore-missing-imports
```

### Git Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Test additions or modifications
- `chore:` Build process or auxiliary tool changes

Example: `feat: add streaming audio generation endpoint`

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/musicgen --cov-report=html

# Run specific test file
pytest tests/unit/test_api.py

# Run tests in parallel
pytest -n auto
```

### Writing Tests

- Write unit tests for all new functions
- Use pytest fixtures for common test data
- Mock external dependencies (APIs, models)
- Aim for >80% code coverage

### Test Structure

```python
def test_function_name_describes_what_it_tests():
    """Test that specific behavior works correctly."""
    # Arrange
    input_data = create_test_data()
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result == expected_output
```

## Documentation

### Code Documentation

- All public APIs must have docstrings
- Use Google style docstrings
- Include examples in docstrings when helpful

### Project Documentation

- Update README.md for user-facing changes
- Update technical docs in `/docs` for architectural changes
- Add entries to CHANGELOG.md following Keep a Changelog format

## CI/CD Pipeline

All pull requests must pass:

1. **Code quality checks**: black, isort, flake8, mypy
2. **Unit tests**: All platforms (Linux, macOS, Windows)
3. **Integration tests**: API and CLI functionality
4. **Security scans**: safety, bandit
5. **Coverage requirements**: No decrease from main branch

## Release Process

1. Update version in `pyproject.toml` and `src/musicgen/__init__.py`
2. Update CHANGELOG.md with release notes
3. Create a pull request titled `Release v{version}`
4. After merge, tag the release: `git tag v{version}`
5. Push tags: `git push origin --tags`

## Getting Help

- **Discord**: Join our community server (link in README)
- **GitHub Discussions**: For general questions
- **GitHub Issues**: For bugs and feature requests
- **Email**: brightliu@college.harvard.edu for direct contact

## Recognition

Contributors will be recognized in:
- The project README
- Release notes
- Our documentation

Thank you for contributing to MusicGen AI! 🎵