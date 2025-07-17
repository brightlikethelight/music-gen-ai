# Secret Dependencies Documentation

This document provides a comprehensive overview of all secrets required by the Music Gen AI system, their purposes, and management requirements.

## Core Application Secrets

### Authentication & Session Management
- **JWT_SECRET_KEY**: JSON Web Token signing key for API authentication
  - **Required**: Yes (Critical)
  - **Format**: Base64-encoded string, minimum 64 characters
  - **Rotation**: Every 90 days
  - **Used by**: `music_gen.api.middleware.auth`
  - **Environment**: All environments

- **SESSION_SECRET_KEY**: Session encryption key for web sessions
  - **Required**: Yes (Critical)
  - **Format**: Base64-encoded string, minimum 32 characters
  - **Rotation**: Every 90 days
  - **Used by**: `music_gen.api.utils.session`
  - **Environment**: All environments

- **CSRF_SECRET_KEY**: CSRF protection token generation key
  - **Required**: Yes (High)
  - **Format**: Base64-encoded string, minimum 32 characters
  - **Rotation**: Every 90 days
  - **Used by**: `music_gen.api.middleware.csrf`
  - **Environment**: All environments

### Database Secrets
- **DATABASE_PASSWORD**: PostgreSQL database password
  - **Required**: Yes (Critical)
  - **Format**: Strong password, minimum 16 characters
  - **Rotation**: Every 180 days
  - **Used by**: Database connection pools
  - **Environment**: Production, Staging

- **REDIS_PASSWORD**: Redis cache and task queue password
  - **Required**: Yes (High)
  - **Format**: Strong password, minimum 16 characters
  - **Rotation**: Every 180 days
  - **Used by**: Redis connections, Celery workers
  - **Environment**: Production, Staging

### External API Keys
- **OPENAI_API_KEY**: OpenAI API access for text processing features
  - **Required**: No (Optional)
  - **Format**: OpenAI API key format (sk-...)
  - **Rotation**: As needed based on OpenAI recommendations
  - **Used by**: Text analysis features
  - **Environment**: Production, Staging

- **WANDB_API_KEY**: Weights & Biases API key for experiment tracking
  - **Required**: No (Optional)
  - **Format**: WandB API key format
  - **Rotation**: As needed
  - **Used by**: Training pipeline monitoring
  - **Environment**: Training environments

### Monitoring & Operations
- **SENTRY_DSN**: Sentry error tracking DSN
  - **Required**: No (Optional)
  - **Format**: Sentry DSN URL
  - **Rotation**: As needed
  - **Used by**: Error reporting
  - **Environment**: Production, Staging

- **FLOWER_BASIC_AUTH**: Celery Flower monitoring authentication
  - **Required**: Yes (High)
  - **Format**: "username:password" format
  - **Rotation**: Every 90 days
  - **Used by**: Celery Flower monitoring interface
  - **Environment**: Production, Staging

### Encryption & Security
- **ENCRYPTION_KEY**: Application-level data encryption key
  - **Required**: Yes (Critical)
  - **Format**: Fernet key (44 characters, base64)
  - **Rotation**: Every 365 days with key versioning
  - **Used by**: Sensitive data encryption
  - **Environment**: All environments

- **MODEL_ENCRYPTION_KEY**: ML model file encryption key
  - **Required**: No (Optional)
  - **Format**: Fernet key (44 characters, base64)
  - **Rotation**: Every 365 days
  - **Used by**: Model file protection
  - **Environment**: Production

## Kubernetes Secrets

### Worker Deployment Secrets
- **redis-auth/password**: Redis authentication for workers
- **flower-auth/basic_auth**: Flower monitoring credentials
- **worker-secrets/encryption-key**: Worker-specific encryption keys

### Service Account Secrets
- **musicgen-service-account**: Kubernetes service account tokens
- **registry-credentials**: Container registry access tokens

## Development Environment

### Local Development
- **DEV_DATABASE_PASSWORD**: Local PostgreSQL password
- **DEV_REDIS_PASSWORD**: Local Redis password
- **DEV_JWT_SECRET**: Development JWT secret (can be static)

## Secret Management Requirements

### Rotation Schedule
- **Critical secrets** (JWT, Session, Database): 90 days
- **High-priority secrets** (Redis, CSRF, Flower): 90 days
- **Encryption keys**: 365 days with versioning
- **External API keys**: As recommended by provider

### Access Control
- **Production secrets**: Restricted to production service accounts only
- **Staging secrets**: Accessible to staging and development teams
- **Development secrets**: Local development only

### Backup Requirements
- All secrets must be backed up to secure vault
- Backup frequency: Daily for critical secrets, weekly for others
- Backup retention: 1 year minimum for critical secrets

### Validation Requirements
- All secrets must meet minimum complexity requirements
- Regular validation checks for expiration and strength
- Automated alerts for secrets nearing expiration

## Implementation Details

### Secret Sources
1. **Environment Variables**: Primary source for all secrets
2. **HashiCorp Vault**: Production secret storage and rotation
3. **Kubernetes Secrets**: Container-level secret injection
4. **Local Files**: Development environment only (encrypted)

### Secret Generation
- Use `music_gen.core.secrets_manager.SecretGenerator` for all secret generation
- Cryptographically secure random generation using `secrets` module
- Type-specific generation (passwords, API keys, JWT secrets, etc.)

### Monitoring
- Track secret usage and access patterns
- Monitor for secret expiration
- Alert on failed secret rotations
- Log secret access (without values) for audit trails

## Security Best Practices

### Never Commit Secrets
- All secret files are in `.gitignore`
- Use secret scanning tools in CI/CD
- Regular repository scanning for accidentally committed secrets

### Secret Handling
- Never log secret values
- Use masked values in debugging output
- Implement proper secret cleanup in memory
- Use secure transport (HTTPS/TLS) for secret transmission

### Environment Separation
- Separate secrets for each environment
- No production secrets in non-production environments
- Regular rotation testing in staging environments

## Troubleshooting

### Common Issues
1. **Missing secrets**: Use `python scripts/secrets_cli.py validate --production`
2. **Expired secrets**: Check expiration with `python scripts/secrets_cli.py health-advanced`
3. **Rotation failures**: Review rotation logs and retry with `python scripts/secrets_cli.py rotate SECRET_NAME`

### Emergency Procedures
1. **Compromised secret**: Immediately rotate using secrets manager
2. **Lost secrets**: Restore from backup vault
3. **System lockout**: Use emergency access procedures documented in operations runbook

## References
- [OWASP Secrets Management](https://owasp.org/www-project-secrets-management/)
- [NIST Cryptographic Standards](https://csrc.nist.gov/projects/cryptographic-standards-and-guidelines)
- [HashiCorp Vault Best Practices](https://learn.hashicorp.com/vault)