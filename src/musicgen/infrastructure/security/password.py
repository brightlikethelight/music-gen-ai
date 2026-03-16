"""
Password hashing utilities for MusicGen API.

Educational demonstration of secure password handling using bcrypt.
WHY: Plaintext passwords can be compromised if memory is dumped or logs
are exposed. Bcrypt provides one-way hashing with salt to prevent rainbow
table attacks.

References:
- OWASP Password Storage Cheat Sheet
- CWE-256: Plaintext Storage of a Password
"""

from typing import cast

from passlib.context import CryptContext

# Use bcrypt with automatic salt generation
# Rounds=12 provides ~250ms hashing time - good balance of security/performance
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.

    Educational note: Never store raw passwords. This function creates a
    one-way hash that cannot be reversed, protecting users even if the
    database is compromised.

    Args:
        password: The plaintext password to hash

    Returns:
        The hashed password string (includes algorithm, rounds, and salt)

    Example:
        >>> hashed = hash_password("secret123")
        >>> hashed.startswith("$2b$")  # bcrypt prefix
        True
    """
    return cast(str, pwd_context.hash(password))


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.

    Educational note: This uses constant-time comparison to prevent
    timing attacks where attackers measure response times to guess
    password lengths.

    Args:
        plain_password: The password to verify
        hashed_password: The stored hash to compare against

    Returns:
        True if password matches, False otherwise

    Example:
        >>> hashed = hash_password("secret123")
        >>> verify_password("secret123", hashed)
        True
        >>> verify_password("wrong", hashed)
        False
    """
    try:
        return cast(bool, pwd_context.verify(plain_password, hashed_password))
    except Exception:
        # Invalid hash format or other error - return False safely
        return False


def needs_rehash(hashed_password: str) -> bool:
    """
    Check if a password hash needs to be updated.

    Educational note: As computing power increases, hash algorithms may
    need more rounds. This function checks if a stored hash uses outdated
    parameters and should be rehashed on next login.

    Args:
        hashed_password: The stored hash to check

    Returns:
        True if the hash should be updated
    """
    return cast(bool, pwd_context.needs_update(hashed_password))
