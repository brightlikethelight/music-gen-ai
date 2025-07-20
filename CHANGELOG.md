# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Hybrid API with intelligent fallback chain (Docker → Replicate → Mock)
- Production-grade Kubernetes manifests (7 files) with auto-scaling
- Comprehensive monitoring stack (Prometheus + Grafana)
- Multiple deployment strategies (Docker, K8s, Cloud)
- Streaming audio generation capabilities
- Model hot-swapping API design

### Fixed
- Critical Python 3.12 incompatibility with ML ecosystem
- Replaced broken audiocraft dependency with transformers approach
- Fixed hardcoded Docker paths for local development
- Resolved numpy/scipy recursion errors

### Changed
- Migrated from audiocraft to transformers library for better compatibility
- Refactored API to use async FastAPI with background tasks
- Improved error handling with graceful fallbacks

## [2.0.1] - 2024-01-15
### Added
- Production Docker solution with Python 3.10
- Phase 2 production-grade architecture
- Comprehensive test workflows

### Fixed
- Foundation stabilization - Phase 1 complete
- CI/CD workflows updated to match repository structure

## [2.0.0] - 2024-01-10
### Changed
- Complete repository restructure for production readiness
- Migrated to src/ layout for better packaging
- Added comprehensive infrastructure layer

### Removed
- Legacy AI-generated content
- Outdated demo scripts

## [1.0.0] - 2023-12-01
### Added
- Initial MusicGen implementation
- Basic API endpoints
- Command-line interface
- CS 109B feature integration

[Unreleased]: https://github.com/yourusername/music-gen-ai/compare/v2.0.1...HEAD
[2.0.1]: https://github.com/yourusername/music-gen-ai/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/yourusername/music-gen-ai/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/yourusername/music-gen-ai/releases/tag/v1.0.0