"""
REST API middleware components.

Provides authentication, CORS, rate limiting, and other middleware functionality.
"""

from .rate_limiting import RateLimiter, RateLimitMiddleware

__all__ = ["RateLimiter", "RateLimitMiddleware"]
