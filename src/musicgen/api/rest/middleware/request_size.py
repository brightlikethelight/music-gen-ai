"""Request body size limit middleware."""

from typing import Any, Callable

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB


class ContentSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests with Content-Length exceeding the limit."""

    def __init__(self, app: Any, max_content_length: int = MAX_CONTENT_LENGTH) -> None:
        super().__init__(app)
        self.max_content_length = max_content_length

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_content_length:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={"error": "Request body too large", "max_bytes": self.max_content_length},
            )
        response: Response = await call_next(request)
        return response
