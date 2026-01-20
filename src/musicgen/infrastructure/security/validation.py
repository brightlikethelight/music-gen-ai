"""
Input validation utilities for MusicGen API.

Educational demonstration of secure input handling.
WHY: Untrusted input is the #1 source of security vulnerabilities.
Always validate and sanitize user input.

References:
- OWASP Input Validation Cheat Sheet
- CWE-20: Improper Input Validation
- CWE-22: Path Traversal
"""

import re
import unicodedata
from pathlib import Path
from typing import Optional


class ValidationError(ValueError):
    """Raised when input validation fails."""

    pass


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize a filename to prevent path traversal and injection attacks.

    Educational note: Attackers may use "../" sequences, null bytes,
    or special characters to escape intended directories or execute
    commands.

    Args:
        filename: The user-provided filename
        max_length: Maximum allowed filename length

    Returns:
        Sanitized filename safe for filesystem operations

    Raises:
        ValidationError: If filename is invalid or malicious

    Example:
        >>> sanitize_filename("../../../etc/passwd")
        'etc_passwd'
        >>> sanitize_filename("normal_file.txt")
        'normal_file.txt'
    """
    if not filename:
        raise ValidationError("Filename cannot be empty")

    # Normalize unicode to prevent homograph attacks
    filename = unicodedata.normalize("NFKC", filename)

    # Extract just the filename, removing any path components
    filename = Path(filename).name

    # Remove null bytes and other control characters
    filename = re.sub(r"[\x00-\x1f\x7f]", "", filename)

    # Remove potentially dangerous characters
    # Allow: alphanumeric, dash, underscore, period
    filename = re.sub(r"[^\w\-.]", "_", filename)

    # Prevent hidden files and parent directory references
    filename = filename.lstrip(".")

    # Remove consecutive underscores/periods
    filename = re.sub(r"[_.]{2,}", "_", filename)

    # Enforce length limit
    if len(filename) > max_length:
        # Preserve extension
        stem = Path(filename).stem[: max_length - 10]
        suffix = Path(filename).suffix[:10]
        filename = stem + suffix

    if not filename:
        raise ValidationError("Filename contains no valid characters")

    return filename


def validate_safe_path(user_path: str, base_dir: Path, allow_creation: bool = False) -> Path:
    """
    Validate that a path is within an allowed base directory.

    Educational note: This prevents path traversal attacks where
    attackers use sequences like "../../../etc/passwd" to access
    files outside the intended directory.

    Args:
        user_path: User-provided path
        base_dir: The directory paths must be within
        allow_creation: Whether to allow non-existent paths

    Returns:
        Validated absolute Path object

    Raises:
        ValidationError: If path escapes base directory

    Example:
        >>> from pathlib import Path
        >>> base = Path("/app/uploads")
        >>> validate_safe_path("file.txt", base, allow_creation=True)
        PosixPath('/app/uploads/file.txt')
        >>> validate_safe_path("../secrets/key", base)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValidationError: Path traversal detected...
    """
    base_dir = base_dir.resolve()

    # Construct the full path
    full_path = (base_dir / user_path).resolve()

    # Check if path is within base directory
    try:
        full_path.relative_to(base_dir)
    except ValueError:
        raise ValidationError(f"Path traversal detected: path escapes {base_dir}")

    # Check existence if required
    if not allow_creation and not full_path.exists():
        raise ValidationError(f"Path does not exist: {full_path}")

    return full_path


def validate_prompt(prompt: str, max_length: int = 1000) -> str:
    """
    Validate and sanitize a music generation prompt.

    Educational note: Even text prompts can be attack vectors through
    injection attacks or resource exhaustion.

    Args:
        prompt: User-provided prompt text
        max_length: Maximum allowed length

    Returns:
        Sanitized prompt

    Raises:
        ValidationError: If prompt is invalid

    Example:
        >>> validate_prompt("  upbeat jazz piano  ")
        'upbeat jazz piano'
        >>> validate_prompt("")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValidationError: Prompt cannot be empty
    """
    if not prompt or not prompt.strip():
        raise ValidationError("Prompt cannot be empty")

    # Normalize unicode
    prompt = unicodedata.normalize("NFKC", prompt)

    # Remove control characters except newlines and tabs
    prompt = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", prompt)

    # Collapse multiple whitespace
    prompt = " ".join(prompt.split())

    # Truncate if too long
    if len(prompt) > max_length:
        prompt = prompt[:max_length]

    return prompt.strip()


def validate_email(email: str) -> str:
    """
    Validate an email address format.

    Args:
        email: User-provided email

    Returns:
        Normalized email address

    Raises:
        ValidationError: If email format is invalid
    """
    if not email:
        raise ValidationError("Email cannot be empty")

    # Basic email pattern
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

    email = email.strip().lower()

    if not re.match(pattern, email):
        raise ValidationError("Invalid email format")

    if len(email) > 254:
        raise ValidationError("Email address too long")

    return email


def validate_username(username: str, min_length: int = 3, max_length: int = 50) -> str:
    """
    Validate a username.

    Args:
        username: User-provided username
        min_length: Minimum allowed length
        max_length: Maximum allowed length

    Returns:
        Validated username

    Raises:
        ValidationError: If username is invalid
    """
    if not username:
        raise ValidationError("Username cannot be empty")

    username = username.strip()

    if len(username) < min_length:
        raise ValidationError(f"Username must be at least {min_length} characters")

    if len(username) > max_length:
        raise ValidationError(f"Username must be at most {max_length} characters")

    # Allow alphanumeric, underscore, dash
    if not re.match(r"^[\w-]+$", username):
        raise ValidationError("Username can only contain letters, numbers, underscores, and dashes")

    return username
