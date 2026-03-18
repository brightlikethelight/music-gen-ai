# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Request ID middleware for distributed tracing
- Security headers middleware (X-Content-Type-Options, X-Frame-Options, HSTS, CSP)
- pytest markers (unit, integration, slow, gpu, asyncio, auth, cli)
- `.env.example` for environment variable documentation
- Missing exception classes (AudioGenerationError, MusicGenException alias)
- Backward compatibility for exception handling

### Changed
- Dashboard and job status endpoints now require authentication
- DEBUG enforced off in production (fail-closed validation)
- Docker images pinned to specific versions
- CI pipeline: `ci-pass` gate job, Python 3.12 added to test matrix
- Updated GitHub Actions to v5 for all workflows

### Fixed
- User enumeration timing side-channel in registration
- Path traversal vulnerability in batch processing and audio serving
- CORS misconfiguration (fail-closed in production)
- Thread-safe application state
- Fail-closed token blacklist policy
- 41 lint violations (black, isort, flake8, mypy strict-optional)
- Fixed 2 failing tests by adding required exception classes

### Removed
- Dead `API_KEY` config field (JWT is the auth mechanism)
- Dead test files and legacy `requirements.txt`
- Redundant `bcrypt` dependency (covered by `passlib[bcrypt]`)
- Unused `types-PyYAML` lint dependency

### Security
- Timing side-channel mitigation in authentication
- Symlink protection on audio file serving
- Short JWT key rejection in production
- Prompt validation on generation endpoints
- HSTS and Content-Security-Policy headers

## [2.0.1] - 2025-08-27

### Added
- Comprehensive CI/CD monitoring and automation
- Dependabot configuration for automatic dependency updates
- Pre-commit hooks for code quality enforcement (Black, isort, flake8, bandit, mypy)
- Semantic versioning with commitizen
- Automated release pipeline with version bumping
- Status badges to README for CI visibility
- Release automation for GitHub releases and PyPI publishing
- Comprehensive test suite achieving 25.38% code coverage
- Test files for all major modules (api, config, helpers, prompt)
- Test coverage for:
  - API request/response models
  - Configuration management
  - Prompt engineering
  - Helper utilities
  - Exception handling
  - Logging infrastructure

### Changed
- Removed legacy requirements*.txt files in favor of pyproject.toml
- Updated generator to skip optimizations during testing

### Fixed
- Removed audiocraft dependency to resolve torch version conflict
- Fixed torch.compile timeout issue in test environments
- Resolved dependency conflict between audiocraft 1.3.0 (requires torch==2.1.0) and musicgen-unified (requires torch>=2.2.0)

### Security
- Added Bandit security scanning to CI pipeline
- Configured pre-commit hooks for security checks
- Added secret scanning and detection

## [2.0.0] - 2025-08-15

### Added
- Academic research platform for music generation
- Integration with Meta's MusicGen model
- GPU acceleration support for faster generation
- Web interface for interactive music creation
- RESTful API for programmatic access
- Batch processing capabilities
- Prompt engineering system
- Memory management optimization
- WebSocket support for real-time generation
- Docker support for containerized deployment
- Comprehensive CLI with multiple commands
- Support for multiple audio formats (MP3, WAV, FLAC)
- Configurable generation parameters
- Harvard CS109b course integration

### Changed
- Complete rewrite from prototype to production system
- Modular architecture with clean separation of concerns
- Async/await support throughout the codebase
- Improved error handling and recovery

### Fixed
- Memory leaks in long-running generation tasks
- Audio quality issues with certain prompts
- Rate limiting for API endpoints

## [1.0.0] - 2025-07-01

### Added
- Initial prototype release
- Basic music generation capabilities
- Simple CLI interface
- Support for text-to-music generation
- Basic prompt validation

### Notes
- Academic project for Harvard CS109b Advanced Topics in Data Science
- Experimental implementation for educational purposes
- Research prototype exploring AI music generation

[Unreleased]: https://github.com/brightlikethelight/music-gen-ai/compare/v2.0.1...HEAD
[2.0.1]: https://github.com/brightlikethelight/music-gen-ai/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/brightlikethelight/music-gen-ai/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/brightlikethelight/music-gen-ai/releases/tag/v1.0.0