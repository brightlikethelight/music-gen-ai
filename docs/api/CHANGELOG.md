# Music Gen AI API Changelog

All notable changes to the Music Gen AI API will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Real-time streaming generation with Server-Sent Events (SSE)
- Advanced melody conditioning support for musicgen-melody model
- Comprehensive monitoring with Prometheus metrics and Grafana dashboards
- SLI/SLO tracking with error budget monitoring
- Structured logging with correlation IDs
- Enhanced rate limiting with tier-based controls
- Detailed API documentation with OpenAPI 3.0 specification
- Webhook notifications for generation completion
- Model health checks and availability status
- Advanced audio format support (FLAC, high-resolution WAV)

### Changed
- Improved error handling with detailed error codes and descriptions
- Enhanced authentication with JWT refresh token support
- Updated model parameters with better default values
- Optimized generation pipeline for better performance
- Improved CORS configuration for enhanced security

### Security
- Added comprehensive input validation and sanitization
- Implemented advanced rate limiting with IP-based tracking
- Enhanced authentication middleware with session management
- Added CSRF protection for web-based interactions
- Implemented secure cookie handling with proper flags

## [1.0.0] - 2024-01-15

### Added
- Initial production release of Music Gen AI API
- Core music generation endpoints (`/generate`, `/generate/{task_id}`)
- JWT-based authentication system
- API key authentication support
- User profile and account management endpoints
- Model information and capabilities endpoints
- Health check and metrics endpoints
- Basic rate limiting by user tier
- Audio format conversion (WAV, MP3)
- Task status tracking and cancellation
- Basic error handling and validation

### Features
- **Text-to-Music Generation**: Create music from natural language descriptions
- **Multiple Model Support**: Small, medium, large, and melody models
- **Async Generation**: Non-blocking generation with task tracking
- **User Tiers**: Free, Pro, and Enterprise with different limits
- **Format Support**: WAV and MP3 output formats
- **Quality Controls**: Configurable sampling parameters

### API Endpoints
- `POST /auth/login` - User authentication
- `POST /auth/refresh` - Token refresh
- `POST /generate` - Music generation
- `GET /generate/{task_id}` - Generation status
- `POST /generate/{task_id}/cancel` - Cancel generation
- `GET /models` - List available models
- `GET /models/{model_id}` - Model details
- `GET /user/profile` - User profile
- `GET /user/api-keys` - API key management
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics

### Models Available
- **musicgen-small**: Fast generation, good quality (300M parameters)
- **musicgen-medium**: Balanced speed/quality (1.5B parameters)
- **musicgen-large**: Best quality, slower (3.3B parameters)
- **musicgen-melody**: Melody conditioning support (1.5B parameters)

### Rate Limits
- **Free Tier**: 10 generations/hour, 30s max duration
- **Pro Tier**: 100 generations/hour, 5min max duration
- **Enterprise Tier**: Unlimited, 10min max duration

### Authentication
- JWT tokens with 24-hour expiration
- API keys for programmatic access
- Refresh tokens for session management

## [0.9.0] - 2024-01-01 (Beta Release)

### Added
- Beta release for early access users
- Core generation functionality
- Basic authentication system
- Simple rate limiting
- Health monitoring

### Known Issues
- Limited error handling
- Basic rate limiting
- No real-time progress tracking
- Limited audio format support

## [0.8.0] - 2023-12-15 (Alpha Release)

### Added
- Alpha release for internal testing
- Basic music generation from text
- Simple API structure
- Development authentication

### Limitations
- Single model support only
- No rate limiting
- Minimal error handling
- Development environment only

---

## Version Support Policy

| Version | Support Status | Security Updates | Bug Fixes | New Features |
|---------|---------------|------------------|-----------|--------------|
| 1.x.x   | ✅ Supported  | ✅ Yes          | ✅ Yes    | ✅ Yes       |
| 0.9.x   | ⚠️ Limited    | ✅ Yes          | ⚠️ Critical Only | ❌ No |
| 0.8.x   | ❌ Unsupported | ❌ No          | ❌ No     | ❌ No        |

## Migration Guides

### Migrating from v0.9 to v1.0

#### Breaking Changes
1. **Authentication**: API keys now require `sk_` prefix
2. **Error Format**: Error responses now use standardized format
3. **Rate Limiting**: Headers changed to RFC-compliant format

#### Required Updates
```diff
# API Key Format
- X-API-Key: abc123def456
+ X-API-Key: sk_live_abc123def456

# Error Response Structure
{
-  "error": "Invalid prompt"
+  "error": {
+    "code": "VALIDATION_ERROR",
+    "message": "Invalid prompt",
+    "request_id": "req_123456789"
+  }
}

# Rate Limit Headers
- X-RateLimit-Limit: 100
- X-RateLimit-Remaining: 95
+ X-RateLimit-Limit: 100
+ X-RateLimit-Remaining: 95
+ X-RateLimit-Reset: 1642291200
```

#### New Features Available
- Real-time generation streaming
- Enhanced model selection
- Melody conditioning (musicgen-melody)
- Advanced sampling parameters
- Webhook notifications

### Migrating from v0.8 to v0.9

#### Breaking Changes
1. **Authentication**: API keys now required
2. **Endpoints**: Generation endpoint moved to `/generate`
3. **Response Format**: Task-based async responses

#### Required Updates
```diff
# Endpoint Changes
- POST /music/generate
+ POST /generate

# Response Format
{
-  "audio_url": "https://..."
+  "task_id": "task_abc123",
+  "status": "queued"
}
```

## Deprecation Policy

We follow a structured deprecation process:

1. **Announcement**: Deprecation announced with 6 months notice
2. **Warning Period**: API returns deprecation warnings in headers
3. **Support Period**: Deprecated features receive security updates only
4. **Removal**: Features removed in next major version

### Currently Deprecated Features

| Feature | Deprecated In | Warning Since | Removal In | Alternative |
|---------|--------------|---------------|------------|-------------|
| Legacy error format | v1.0.0 | v0.9.0 | v2.0.0 | Structured error responses |
| Non-prefixed API keys | v1.0.0 | v0.9.0 | v2.0.0 | `sk_` prefixed keys |

## API Versioning Strategy

- **URL Versioning**: `/api/v1/endpoint`
- **Header Versioning**: `Accept: application/vnd.musicgen.v1+json`
- **Backward Compatibility**: Maintained within major versions
- **Breaking Changes**: Only in major version updates

## Change Categories

### Added
- New features and capabilities
- New endpoints or parameters
- New response fields (non-breaking)

### Changed
- Modifications to existing functionality
- Updated default values
- Performance improvements

### Deprecated
- Features marked for future removal
- Still functional but not recommended

### Removed
- Features that no longer work
- Breaking changes from deprecated features

### Fixed
- Bug fixes and error corrections
- Security vulnerability patches

### Security
- Security-related improvements
- Vulnerability fixes
- Authentication enhancements

## Release Schedule

- **Major Releases**: Every 12-18 months (breaking changes)
- **Minor Releases**: Every 3-4 months (new features)
- **Patch Releases**: As needed (bug fixes, security)
- **Emergency Releases**: Critical security or stability issues

## Support Channels

- **Documentation**: [https://docs.musicgen.ai](https://docs.musicgen.ai)
- **API Support**: [support@musicgen.ai](mailto:support@musicgen.ai)
- **Status Updates**: [https://status.musicgen.ai](https://status.musicgen.ai)
- **Developer Community**: [https://discord.gg/musicgen](https://discord.gg/musicgen)

## Feedback and Contributions

We welcome feedback on API changes and improvements:

- **Feature Requests**: Create an issue in our [GitHub repository](https://github.com/musicgen-ai/api-feedback)
- **Bug Reports**: Report via [support portal](https://support.musicgen.ai)
- **Documentation Issues**: Submit PR to [docs repository](https://github.com/musicgen-ai/docs)