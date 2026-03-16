"""
Tests for security audit logging module.

Covers audit event logging, client info extraction, and convenience loggers.
"""

import logging
from unittest.mock import MagicMock

import pytest

from musicgen.infrastructure.security.audit import (
    AuditEvent,
    AuditRecord,
    get_client_info,
    log_access_denied,
    log_audit_event,
    log_login_attempt,
    log_registration,
)

pytestmark = pytest.mark.unit


class TestAuditRecord:
    """Test AuditRecord dataclass."""

    def test_to_dict_success(self):
        """Successful record serializes with event value string."""
        record = AuditRecord(
            event=AuditEvent.USER_LOGIN,
            user_id="user123",
            ip_address="192.168.1.1",
            success=True,
        )
        d = record.to_dict()
        assert d["event"] == "user.login"
        assert d["user_id"] == "user123"
        assert d["success"] is True

    def test_to_dict_strips_none(self):
        """None values are excluded from the dict."""
        record = AuditRecord(event=AuditEvent.USER_LOGIN)
        d = record.to_dict()
        assert "user_id" not in d
        assert "ip_address" not in d
        assert "user_agent" not in d


class TestLogAuditEvent:
    """Test log_audit_event function."""

    def test_success_logs_at_info(self, caplog):
        """Successful events log at INFO level."""
        with caplog.at_level(logging.INFO, logger="musicgen.audit"):
            log_audit_event(
                AuditEvent.USER_LOGIN,
                user_id="user123",
                ip_address="10.0.0.1",
                success=True,
            )
            assert "user.login" in caplog.text

    def test_failure_logs_at_warning(self, caplog):
        """Failed events log at WARNING level."""
        with caplog.at_level(logging.WARNING, logger="musicgen.audit"):
            log_audit_event(
                AuditEvent.USER_LOGIN_FAILED,
                ip_address="10.0.0.1",
                success=False,
                reason="bad password",
            )
            assert "FAILED" in caplog.text

    def test_extra_details_passed(self, caplog):
        """Extra kwargs are stored in the audit record."""
        with caplog.at_level(logging.INFO, logger="musicgen.audit"):
            log_audit_event(
                AuditEvent.USER_REGISTER,
                user_id="u1",
                success=True,
                email="a@b.com",
                username="testuser",
            )
            assert "user.register" in caplog.text


class TestGetClientInfo:
    """Test client info extraction from request objects."""

    def test_with_request(self):
        """Extracts IP and user-agent from a mock request."""
        request = MagicMock()
        request.client.host = "192.168.1.100"
        request.headers.get.return_value = "Mozilla/5.0"

        ip, ua = get_client_info(request)
        assert ip == "192.168.1.100"
        assert ua == "Mozilla/5.0"

    def test_without_client(self):
        """Returns None when request.client is None."""
        request = MagicMock()
        request.client = None
        request.headers.get.return_value = "curl/7.0"

        ip, ua = get_client_info(request)
        assert ip is None
        assert ua == "curl/7.0"

    def test_without_headers(self):
        """Returns None for user_agent when headers missing."""
        request = MagicMock(spec=[])  # No attributes
        ip, ua = get_client_info(request)
        assert ip is None
        assert ua is None


class TestLogLoginAttempt:
    """Test login attempt logging."""

    def test_success(self, caplog):
        """Successful login logs USER_LOGIN at INFO."""
        request = MagicMock()
        request.client.host = "10.0.0.1"
        request.headers.get.return_value = "test-agent"

        with caplog.at_level(logging.INFO, logger="musicgen.audit"):
            log_login_attempt(
                request,
                email="user@test.com",
                success=True,
                user_id="u1",
            )
            assert "user.login" in caplog.text

    def test_failure(self, caplog):
        """Failed login logs USER_LOGIN_FAILED at WARNING."""
        request = MagicMock()
        request.client.host = "10.0.0.1"
        request.headers.get.return_value = "test-agent"

        with caplog.at_level(logging.WARNING, logger="musicgen.audit"):
            log_login_attempt(
                request,
                email="attacker@test.com",
                success=False,
                failure_reason="invalid password",
            )
            assert "FAILED" in caplog.text


class TestLogRegistration:
    """Test registration audit logging."""

    def test_logs_registration(self, caplog):
        """Registration logs USER_REGISTER with user details."""
        request = MagicMock()
        request.client.host = "10.0.0.1"
        request.headers.get.return_value = "test-agent"

        with caplog.at_level(logging.INFO, logger="musicgen.audit"):
            log_registration(
                request,
                user_id="new-user-id",
                email="new@test.com",
                username="newuser",
            )
            assert "user.register" in caplog.text


class TestLogAccessDenied:
    """Test access denied audit logging."""

    def test_logs_denial(self, caplog):
        """Access denial logs ACCESS_DENIED at WARNING."""
        request = MagicMock()
        request.client.host = "10.0.0.1"
        request.headers.get.return_value = "test-agent"

        with caplog.at_level(logging.WARNING, logger="musicgen.audit"):
            log_access_denied(
                request,
                resource="/admin/users",
                reason="insufficient permissions",
                user_id="u1",
            )
            assert "FAILED" in caplog.text
