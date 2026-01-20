"""
Security utilities.

Provides input validation, authentication helpers, and security middleware.
"""

from musicgen.infrastructure.security.audit import (
    AuditEvent,
    get_client_info,
    log_access_denied,
    log_audit_event,
    log_login_attempt,
    log_registration,
)
from musicgen.infrastructure.security.password import (
    hash_password,
    needs_rehash,
    verify_password,
)
from musicgen.infrastructure.security.secrets import (
    SecretKeyError,
    generate_secret_key,
    get_api_key,
    get_jwt_secret,
    mask_secret,
)
from musicgen.infrastructure.security.validation import (
    ValidationError,
    sanitize_filename,
    validate_email,
    validate_prompt,
    validate_safe_path,
    validate_username,
)

__all__ = [
    # Password
    "hash_password",
    "verify_password",
    "needs_rehash",
    # Secrets
    "get_jwt_secret",
    "generate_secret_key",
    "get_api_key",
    "mask_secret",
    "SecretKeyError",
    # Validation
    "sanitize_filename",
    "validate_safe_path",
    "validate_prompt",
    "validate_email",
    "validate_username",
    "ValidationError",
    # Audit
    "log_audit_event",
    "log_login_attempt",
    "log_registration",
    "log_access_denied",
    "get_client_info",
    "AuditEvent",
]
