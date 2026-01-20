"""
Request logging middleware for MusicGen API.

Provides comprehensive request/response logging with correlation IDs
for debugging and monitoring.
"""

import logging
import time
import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that logs all HTTP requests and responses.

    Features:
    - Assigns unique correlation ID to each request
    - Logs request method, path, and user info
    - Logs response status and duration
    - Adds X-Request-ID header to responses
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with logging."""
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        start_time = time.time()

        # Extract user info if available
        user_id = getattr(request.state, "user_id", None)
        client_ip = request.client.host if request.client else "unknown"

        # Log request start
        logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query": str(request.url.query) if request.url.query else None,
                "client_ip": client_ip,
                "user_id": user_id,
            },
        )

        try:
            response = await call_next(request)
        except Exception as e:
            # Log exception
            duration = time.time() - start_time
            logger.error(
                "Request failed with exception",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round(duration * 1000, 2),
                    "error": str(e),
                },
            )
            raise

        # Calculate duration
        duration = time.time() - start_time

        # Log response
        log_level = logging.INFO if response.status_code < 400 else logging.WARNING
        logger.log(
            log_level,
            "Request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration * 1000, 2),
                "user_id": user_id,
            },
        )

        # Add correlation ID to response headers
        response.headers["X-Request-ID"] = request_id

        return response


def get_request_id(request: Request) -> str:
    """
    Get the request ID from a request object.

    Args:
        request: The Starlette/FastAPI request

    Returns:
        The request ID or "unknown" if not set
    """
    return getattr(request.state, "request_id", "unknown")
