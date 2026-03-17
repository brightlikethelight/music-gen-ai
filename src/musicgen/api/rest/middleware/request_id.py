"""Request ID middleware for tracing requests through the system."""

import uuid
from typing import Any, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

_REQUEST_ID_HEADER = "X-Request-ID"


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique request ID to every request/response."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        request_id = request.headers.get(_REQUEST_ID_HEADER) or str(uuid.uuid4())
        request.state.request_id = request_id
        response: Response = await call_next(request)
        response.headers[_REQUEST_ID_HEADER] = request_id
        return response
