"""
Shared Microservices Components

Common utilities, models, and infrastructure components shared across
all MusicGen AI microservices.
"""

from .observability import get_tracer, create_span
from .exceptions import (
    ServiceError,
    ModelServiceError,
    ProcessingServiceError,
    StorageServiceError,
    AuthenticationError,
    RateLimitError
)
from .middleware import (
    MetricsMiddleware,
    TracingMiddleware,
    ErrorHandlingMiddleware,
    RequestLoggingMiddleware
)

__all__ = [
    "get_tracer",
    "create_span",
    "ServiceError",
    "ModelServiceError", 
    "ProcessingServiceError",
    "StorageServiceError",
    "AuthenticationError",
    "RateLimitError",
    "MetricsMiddleware",
    "TracingMiddleware",
    "ErrorHandlingMiddleware",
    "RequestLoggingMiddleware"
]