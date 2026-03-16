"""
Tests for security modules: password, secrets, validation.

Covers the primary attack surface with zero prior test coverage.
"""

import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest

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

# --- password.py tests ---


class TestPasswordHashing:
    """Test bcrypt password hashing."""

    def test_hash_password_returns_bcrypt(self):
        """Hashed output starts with bcrypt $2b$ prefix."""
        hashed = hash_password("testpassword")
        assert hashed.startswith("$2b$")

    def test_verify_password_correct(self):
        """Hash then verify with correct password returns True."""
        hashed = hash_password("correct-horse-battery-staple")
        assert verify_password("correct-horse-battery-staple", hashed) is True

    def test_verify_password_wrong(self):
        """Wrong password returns False."""
        hashed = hash_password("correct-password")
        assert verify_password("wrong-password", hashed) is False

    def test_verify_password_invalid_hash(self):
        """Garbage hash returns False instead of raising."""
        assert verify_password("anything", "not-a-valid-hash") is False

    def test_needs_rehash_current(self):
        """A freshly-hashed password does not need rehash."""
        hashed = hash_password("somepassword")
        assert needs_rehash(hashed) is False

    def test_hash_produces_unique_salts(self):
        """Two hashes of the same password should differ (unique salts)."""
        h1 = hash_password("samepassword")
        h2 = hash_password("samepassword")
        assert h1 != h2


# --- secrets.py tests ---


class TestSecretsManagement:
    """Test JWT secret and API key management."""

    def setup_method(self):
        """Clear lru_cache between tests."""
        get_jwt_secret.cache_clear()

    def teardown_method(self):
        """Clean up cache after each test."""
        get_jwt_secret.cache_clear()

    def test_get_jwt_secret_from_env(self):
        """When JWT_SECRET_KEY env var is set, it is returned."""
        secret = "a" * 64
        with patch.dict(os.environ, {"JWT_SECRET_KEY": secret}):
            result = get_jwt_secret()
            assert result == secret

    def test_get_jwt_secret_warns_short_key(self, caplog):
        """Short key (<32 chars) logs a warning."""
        short_key = "tooshort"
        with patch.dict(os.environ, {"JWT_SECRET_KEY": short_key}):
            with caplog.at_level(logging.WARNING):
                result = get_jwt_secret()
                assert result == short_key
                assert "too short" in caplog.text

    def test_get_jwt_secret_test_env_generates(self):
        """In test environment without env var, an ephemeral key is generated."""
        env = {
            "PYTEST_CURRENT_TEST": "1",
        }
        with patch.dict(os.environ, env, clear=False):
            # Remove JWT_SECRET_KEY if present
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("JWT_SECRET_KEY", None)
                result = get_jwt_secret()
                assert len(result) == 64  # token_hex(32) = 64 chars

    def test_generate_secret_key_length(self):
        """Output hex string length is 2x the input byte count."""
        key16 = generate_secret_key(16)
        assert len(key16) == 32
        key32 = generate_secret_key(32)
        assert len(key32) == 64

    def test_get_api_key_found(self):
        """Returns API key when env var is set."""
        with patch.dict(os.environ, {"MY_API_KEY": "secret123"}):
            assert get_api_key("MY_API_KEY") == "secret123"

    def test_get_api_key_missing_required(self):
        """Raises SecretKeyError for missing required key outside test env."""
        with patch.dict(os.environ, {"PYTEST_CURRENT_TEST": ""}, clear=False):
            os.environ.pop("NONEXISTENT_KEY", None)
            os.environ.pop("PYTEST_CURRENT_TEST", None)
            try:
                with pytest.raises(SecretKeyError):
                    get_api_key("NONEXISTENT_KEY", required=True)
            finally:
                os.environ["PYTEST_CURRENT_TEST"] = "1"

    def test_get_api_key_missing_optional(self):
        """Returns None for missing optional key."""
        os.environ.pop("NONEXISTENT_KEY", None)
        assert get_api_key("NONEXISTENT_KEY", required=False) is None

    def test_mask_secret_long(self):
        """Long secret shows first and last 4 chars."""
        assert mask_secret("my-super-secret-key") == "my-s...-key"

    def test_mask_secret_short(self):
        """Short secret is fully masked with asterisks."""
        assert mask_secret("ab") == "**"
        assert mask_secret("abcd1234") == "********"


# --- validation.py tests ---


class TestSanitizeFilename:
    """Test filename sanitization against path traversal."""

    def test_traversal_attack(self):
        """Path traversal sequences are stripped."""
        result = sanitize_filename("../../../etc/passwd")
        assert ".." not in result
        assert "/" not in result

    def test_empty_raises(self):
        """Empty filename raises ValidationError."""
        with pytest.raises(ValidationError):
            sanitize_filename("")

    def test_null_bytes_stripped(self):
        """Null bytes and control characters are removed."""
        result = sanitize_filename("file\x00name\x01.txt")
        assert "\x00" not in result
        assert "\x01" not in result

    def test_normal_filename_preserved(self):
        """Normal filenames pass through with minimal changes."""
        assert sanitize_filename("normal_file.txt") == "normal_file.txt"

    def test_hidden_file_prefix_stripped(self):
        """Leading dots are stripped to prevent hidden files."""
        result = sanitize_filename(".hidden")
        assert not result.startswith(".")


class TestValidateSafePath:
    """Test path traversal prevention."""

    def test_normal_path(self, tmp_path):
        """Normal path within base dir is accepted."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        result = validate_safe_path("test.txt", tmp_path)
        assert result == test_file.resolve()

    def test_traversal_rejected(self, tmp_path):
        """Path traversal escaping base dir raises ValidationError."""
        with pytest.raises(ValidationError, match="traversal"):
            validate_safe_path("../../etc/passwd", tmp_path)

    def test_allow_creation(self, tmp_path):
        """Non-existent path accepted when allow_creation=True."""
        result = validate_safe_path("new_file.txt", tmp_path, allow_creation=True)
        assert result == (tmp_path / "new_file.txt").resolve()


class TestValidatePrompt:
    """Test prompt validation and sanitization."""

    def test_valid_prompt(self):
        """Normal prompt is returned trimmed."""
        assert validate_prompt("  upbeat jazz piano  ") == "upbeat jazz piano"

    def test_empty_prompt(self):
        """Empty prompt raises ValidationError."""
        with pytest.raises(ValidationError, match="empty"):
            validate_prompt("")

    def test_whitespace_only_prompt(self):
        """Whitespace-only prompt raises ValidationError."""
        with pytest.raises(ValidationError, match="empty"):
            validate_prompt("   \t\n   ")

    def test_control_chars_stripped(self):
        """Control characters are removed from prompt."""
        result = validate_prompt("hello\x00world\x07test")
        assert "\x00" not in result
        assert "\x07" not in result
        assert "hello" in result

    def test_long_prompt_truncated(self):
        """Prompts exceeding max_length are truncated."""
        long_prompt = "a " * 600
        result = validate_prompt(long_prompt, max_length=100)
        assert len(result) <= 100


class TestValidateEmail:
    """Test email validation."""

    def test_valid_email(self):
        """Standard email format is accepted."""
        assert validate_email("user@example.com") == "user@example.com"

    def test_valid_email_normalized(self):
        """Email is lowercased and stripped."""
        assert validate_email("  User@Example.COM  ") == "user@example.com"

    def test_invalid_email(self):
        """Invalid format raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid email"):
            validate_email("not-an-email")

    def test_empty_email(self):
        """Empty email raises ValidationError."""
        with pytest.raises(ValidationError, match="empty"):
            validate_email("")


class TestValidateUsername:
    """Test username validation."""

    def test_valid_username(self):
        """Alphanumeric username is accepted."""
        assert validate_username("test_user-123") == "test_user-123"

    def test_too_short(self):
        """Username below min_length raises."""
        with pytest.raises(ValidationError, match="at least"):
            validate_username("ab")

    def test_special_chars_rejected(self):
        """Special characters raise ValidationError."""
        with pytest.raises(ValidationError, match="only contain"):
            validate_username("user@name!")

    def test_empty_username(self):
        """Empty username raises ValidationError."""
        with pytest.raises(ValidationError, match="empty"):
            validate_username("")
