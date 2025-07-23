"""
Educational Rate Limiting Middleware Example for MusicGen Academic Project.
Simplified implementation for learning purposes - Harvard CS 109B.
⚠️ NOT TESTED FOR PRODUCTION USE - EDUCATIONAL EXAMPLE ONLY
"""

import ipaddress
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RateLimiter:
    """In-memory rate limiter with sliding window."""

    def __init__(self):
        self.requests = defaultdict(deque)  # IP -> deque of timestamps
        self.last_cleanup = time.time()

        # Rate limits (requests per time window)
        self.limits = {
            "per_minute": 60,  # 60 requests per minute
            "per_hour": 1000,  # 1000 requests per hour
            "per_day": 10000,  # 10000 requests per day
        }

        # Time windows in seconds
        self.windows = {"per_minute": 60, "per_hour": 3600, "per_day": 86400}

    def _cleanup_old_requests(self):
        """Remove expired request timestamps."""
        current_time = time.time()

        # Only cleanup every 60 seconds to avoid overhead
        if current_time - self.last_cleanup < 60:
            return

        self.last_cleanup = current_time

        # Remove timestamps older than 1 day
        cutoff = current_time - self.windows["per_day"]

        for ip in list(self.requests.keys()):
            # Remove old timestamps
            while self.requests[ip] and self.requests[ip][0] < cutoff:
                self.requests[ip].popleft()

            # Remove empty entries
            if not self.requests[ip]:
                del self.requests[ip]

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP with proxy support."""
        # Check for forwarded IP headers (educational example)
        forwarded_ips = [
            request.headers.get("X-Forwarded-For"),
            request.headers.get("X-Real-IP"),
            request.headers.get("CF-Connecting-IP"),  # Cloudflare
        ]

        for header_value in forwarded_ips:
            if header_value:
                # X-Forwarded-For can contain multiple IPs, take the first
                ip = header_value.split(",")[0].strip()
                if self._is_valid_ip(ip):
                    return ip

        # Fallback to direct connection IP
        client_host = request.client.host if request.client else "127.0.0.1"
        return client_host

    def _is_valid_ip(self, ip: str) -> bool:
        """Validate IP address format."""
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False

    def _is_exempt_ip(self, ip: str) -> bool:
        """Check if IP is exempt from rate limiting."""
        exempt_networks = [
            "127.0.0.0/8",  # Localhost
            "10.0.0.0/8",  # Private networks
            "172.16.0.0/12",  # Private networks
            "192.168.0.0/16",  # Private networks
        ]

        try:
            ip_addr = ipaddress.ip_address(ip)
            for network in exempt_networks:
                if ip_addr in ipaddress.ip_network(network):
                    return True
        except ValueError:
            pass

        return False

    def is_allowed(self, request: Request) -> Tuple[bool, Dict]:
        """Check if request is allowed under rate limits."""
        client_ip = self._get_client_ip(request)

        # Exempt internal IPs
        if self._is_exempt_ip(client_ip):
            return True, {}

        current_time = time.time()
        self._cleanup_old_requests()

        # Add current request timestamp
        self.requests[client_ip].append(current_time)

        # Check each time window
        limits_info = {}
        for window_name, window_seconds in self.windows.items():
            cutoff = current_time - window_seconds

            # Count requests in this window
            count = sum(1 for ts in self.requests[client_ip] if ts >= cutoff)
            limit = self.limits[window_name]

            limits_info[window_name] = {
                "count": count,
                "limit": limit,
                "remaining": max(0, limit - count),
                "reset_time": int(current_time + window_seconds),
            }

            # If any window is exceeded, deny request
            if count > limit:
                # Remove the request we just added since it's denied
                self.requests[client_ip].pop()

                return False, {
                    "error": "Rate limit exceeded",
                    "window": window_name,
                    "limit": limit,
                    "retry_after": int(cutoff + window_seconds - current_time + 1),
                }

        return True, limits_info


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""

    def __init__(self, app, rate_limiter: Optional[RateLimiter] = None):
        super().__init__(app)
        self.rate_limiter = rate_limiter or RateLimiter()

    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""

        # Skip rate limiting for health checks and static files
        if request.url.path in ["/health", "/docs", "/openapi.json"]:
            return await call_next(request)

        # Check rate limits
        allowed, info = self.rate_limiter.is_allowed(request)

        if not allowed:
            # Rate limit exceeded
            logger.warning(
                f"Rate limit exceeded for {self.rate_limiter._get_client_ip(request)}: "
                f"{info.get('error', 'Unknown error')}"
            )

            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": info.get("error", "Too many requests"),
                    "retry_after": info.get("retry_after", 60),
                },
                headers={
                    "Retry-After": str(info.get("retry_after", 60)),
                    "X-RateLimit-Limit": str(info.get("limit", "Unknown")),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(info.get("retry_after", 60)),
                },
            )

        # Process request normally
        response = await call_next(request)

        # Add rate limit headers to successful responses
        if "per_minute" in info:
            minute_info = info["per_minute"]
            response.headers["X-RateLimit-Limit"] = str(minute_info["limit"])
            response.headers["X-RateLimit-Remaining"] = str(minute_info["remaining"])
            response.headers["X-RateLimit-Reset"] = str(minute_info["reset_time"])

        return response


# Rate limiter instance for sharing across the application
rate_limiter = RateLimiter()
