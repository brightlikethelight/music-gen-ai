"""
Secure secrets management for MusicGen API.

Educational demonstration of proper secret handling.
WHY: Hardcoded secrets can be extracted from source code. Environment
variables provide runtime configuration without code exposure.

References:
- OWASP Secrets Management Cheat Sheet
- CWE-798: Use of Hard-coded Credentials
"""

import logging
import os
import secrets
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)


class SecretKeyError(Exception):
    """Raised when a required secret key is missing or invalid."""

    pass


def _is_test_environment() -> bool:
    """
    Check if running in test environment safely.

    Only returns True when pytest is actively running a test.
    """
    return bool(os.getenv("PYTEST_CURRENT_TEST"))


def _is_development_environment() -> bool:
    """Check if running in development mode."""
    env = os.getenv("MUSICGEN_ENV", "development").lower()
    return env in ("development", "dev", "local")


@lru_cache(maxsize=1)
def get_jwt_secret() -> str:
    """
    Get JWT secret key with validation.

    Educational note: JWT secrets must be cryptographically random and
    at least 256 bits (32 bytes) for HS256. Never use predictable values.

    Returns:
        The JWT secret key

    Raises:
        SecretKeyError: If no valid secret is configured in production

    Example:
        # In production, set environment variable:
        # export JWT_SECRET_KEY=$(openssl rand -hex 32)

        >>> import os
        >>> os.environ["JWT_SECRET_KEY"] = "a" * 32
        >>> secret = get_jwt_secret()
        >>> len(secret) >= 32
        True
    """
    secret = os.getenv("JWT_SECRET_KEY")

    if secret:
        # Validate minimum length (32 chars = 256 bits)
        if len(secret) < 32:
            logger.warning(
                "JWT_SECRET_KEY is too short (%d chars). "
                "Use at least 32 characters. "
                "Generate with: openssl rand -hex 32",
                len(secret),
            )
        return secret

    # Only allow fallback in genuine test environments
    if _is_test_environment():
        # Generate a random key for each test run - NOT reusable
        test_key = secrets.token_hex(32)
        logger.debug("Generated ephemeral JWT key for testing")
        return test_key

    # Development mode: generate a key but warn
    if _is_development_environment():
        dev_key = secrets.token_hex(32)
        logger.warning(
            "JWT_SECRET_KEY not set. Generated ephemeral key for development. "
            "Set JWT_SECRET_KEY for production: export JWT_SECRET_KEY=$(openssl rand -hex 32)"
        )
        return dev_key

    # Production requires explicit configuration
    raise SecretKeyError(
        "JWT_SECRET_KEY environment variable is required in production. "
        "Set it with: export JWT_SECRET_KEY=$(openssl rand -hex 32)"
    )


def generate_secret_key(length: int = 32) -> str:
    """
    Generate a cryptographically secure random key.

    Educational note: This uses os.urandom() which draws from the OS
    cryptographic random source (/dev/urandom on Unix).

    Args:
        length: Number of bytes (output will be 2x in hex)

    Returns:
        Hex-encoded random string
    """
    return secrets.token_hex(length)


def get_api_key(key_name: str, required: bool = True) -> Optional[str]:
    """
    Get an API key from environment variables.

    Args:
        key_name: Name of the environment variable
        required: If True, raise error when not found

    Returns:
        The API key value or None

    Raises:
        SecretKeyError: If required key is not found
    """
    value = os.getenv(key_name)

    if value is None and required and not _is_test_environment():
        raise SecretKeyError(f"Required API key {key_name} is not set")

    return value


def mask_secret(secret: str, visible_chars: int = 4) -> str:
    """
    Mask a secret for safe logging.

    Args:
        secret: The secret to mask
        visible_chars: Number of characters to show at start/end

    Returns:
        Masked string like "abc...xyz"

    Example:
        >>> mask_secret("my-super-secret-key")
        'my-s...key'
    """
    if len(secret) <= visible_chars * 2:
        return "*" * len(secret)

    return f"{secret[:visible_chars]}...{secret[-visible_chars:]}"
