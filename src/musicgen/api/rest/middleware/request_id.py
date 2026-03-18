"""Request ID middleware for tracing requests through the system."""

import re
import uuid
from typing import Any, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

_REQUEST_ID_HEADER = "X-Request-ID"
_UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique request ID to every request/response."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        client_id = request.headers.get(_REQUEST_ID_HEADER)
        if client_id and _UUID_PATTERN.match(client_id):
            request_id = client_id
        else:
            request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response: Response = await call_next(request)
        response.headers[_REQUEST_ID_HEADER] = request_id
        return response
