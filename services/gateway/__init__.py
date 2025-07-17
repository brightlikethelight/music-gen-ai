"""
API Gateway Service

Central entry point for all client requests with routing, authentication,
rate limiting, and cross-cutting concerns.
"""

from .app import create_gateway_app
from .middleware import (
    AuthenticationMiddleware,
    RateLimitingMiddleware,
    RequestLoggingMiddleware,
    CORSMiddleware
)
from .routing import GatewayRouter
from .config import GatewayConfig

__all__ = [
    "create_gateway_app",
    "AuthenticationMiddleware", 
    "RateLimitingMiddleware",
    "RequestLoggingMiddleware",
    "CORSMiddleware",
    "GatewayRouter",
    "GatewayConfig"
]