"""
Security audit logging for MusicGen API.

Educational demonstration of security event logging.
WHY: Audit logs are essential for detecting attacks, investigating
incidents, and meeting compliance requirements.

References:
- OWASP Logging Cheat Sheet
- CWE-778: Insufficient Logging
"""

import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

# Use structured logging for machine-parseable audit trails
audit_logger = logging.getLogger("musicgen.audit")


class AuditEvent(str, Enum):
    """Security-relevant events to audit."""

    # Authentication events
    USER_REGISTER = "user.register"
    USER_LOGIN = "user.login"
    USER_LOGIN_FAILED = "user.login_failed"
    USER_LOGOUT = "user.logout"

    # Token events
    TOKEN_REFRESH = "token.refresh"
    TOKEN_REVOKED = "token.revoked"
    TOKEN_INVALID = "token.invalid"

    # Access control events
    ACCESS_DENIED = "access.denied"
    ACCESS_GRANTED = "access.granted"

    # Rate limiting events
    RATE_LIMIT_EXCEEDED = "rate_limit.exceeded"

    # Resource events
    FILE_ACCESS = "file.access"
    FILE_DOWNLOAD = "file.download"
    GENERATION_STARTED = "generation.started"
    GENERATION_COMPLETED = "generation.completed"

    # Admin events
    ADMIN_ACTION = "admin.action"
    CONFIG_CHANGE = "config.change"

    # Security events
    SUSPICIOUS_ACTIVITY = "suspicious.activity"
    VALIDATION_FAILED = "validation.failed"


@dataclass
class AuditRecord:
    """Structured audit log record."""

    event: AuditEvent
    timestamp: float = field(default_factory=time.time)
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    success: bool = True
    details: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON logging."""
        data = asdict(self)
        data["event"] = self.event.value
        # Remove None values for cleaner logs
        return {k: v for k, v in data.items() if v is not None}


def log_audit_event(
    event: AuditEvent,
    user_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    resource: Optional[str] = None,
    success: bool = True,
    **details: Any,
) -> None:
    """
    Log a security audit event.

    Educational note: Audit logs should be immutable, timestamped,
    and include enough context to reconstruct what happened.

    Args:
        event: The type of security event
        user_id: User identifier (if known)
        ip_address: Client IP address
        user_agent: Client user agent string
        resource: Resource being accessed (URL, file path, etc.)
        success: Whether the action succeeded
        **details: Additional event-specific details

    Example:
        >>> log_audit_event(
        ...     AuditEvent.USER_LOGIN,
        ...     user_id="user123",
        ...     ip_address="192.168.1.1",
        ...     success=True
        ... )
    """
    record = AuditRecord(
        event=event,
        user_id=user_id,
        ip_address=ip_address,
        user_agent=user_agent,
        resource=resource,
        success=success,
        details=details if details else None,
    )

    log_data = record.to_dict()

    if success:
        audit_logger.info("AUDIT: %s", event.value, extra={"audit": log_data})
    else:
        audit_logger.warning("AUDIT: %s [FAILED]", event.value, extra={"audit": log_data})


def get_client_info(request: Any) -> tuple[Optional[str], Optional[str]]:
    """
    Extract client info from a FastAPI/Starlette request.

    Educational note: We prefer the direct connection IP over forwarded
    headers to prevent IP spoofing. Only trust X-Forwarded-For from
    known reverse proxies.

    Args:
        request: FastAPI/Starlette Request object

    Returns:
        Tuple of (ip_address, user_agent)
    """
    ip_address = None
    user_agent = None

    if hasattr(request, "client") and request.client:
        ip_address = request.client.host

    if hasattr(request, "headers"):
        user_agent = request.headers.get("User-Agent")

    return ip_address, user_agent


def log_login_attempt(
    request: Any,
    email: str,
    success: bool,
    user_id: Optional[str] = None,
    failure_reason: Optional[str] = None,
) -> None:
    """
    Log a login attempt with full context.

    Args:
        request: FastAPI request object
        email: Email address used
        success: Whether login succeeded
        user_id: User ID if login succeeded
        failure_reason: Reason for failure if applicable
    """
    ip_address, user_agent = get_client_info(request)

    if success:
        log_audit_event(
            AuditEvent.USER_LOGIN,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            success=True,
            email=email,
        )
    else:
        log_audit_event(
            AuditEvent.USER_LOGIN_FAILED,
            ip_address=ip_address,
            user_agent=user_agent,
            success=False,
            attempted_email=email,
            reason=failure_reason,
        )


def log_registration(
    request: Any,
    user_id: str,
    email: str,
    username: str,
) -> None:
    """
    Log a successful user registration.

    Args:
        request: FastAPI request object
        user_id: New user's ID
        email: User's email
        username: User's username
    """
    ip_address, user_agent = get_client_info(request)

    log_audit_event(
        AuditEvent.USER_REGISTER,
        user_id=user_id,
        ip_address=ip_address,
        user_agent=user_agent,
        success=True,
        email=email,
        username=username,
    )


def log_access_denied(
    request: Any,
    resource: str,
    reason: str,
    user_id: Optional[str] = None,
) -> None:
    """
    Log an access denied event.

    Args:
        request: FastAPI request object
        resource: Resource that was denied
        reason: Why access was denied
        user_id: User ID if authenticated
    """
    ip_address, user_agent = get_client_info(request)

    log_audit_event(
        AuditEvent.ACCESS_DENIED,
        user_id=user_id,
        ip_address=ip_address,
        user_agent=user_agent,
        resource=resource,
        success=False,
        reason=reason,
    )
