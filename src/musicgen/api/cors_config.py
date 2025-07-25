"""
CORS configuration for MusicGen API.

Educational demonstration of Cross-Origin Resource Sharing configuration.
"""

import os
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse


class CORSConfig:
    """Configuration for CORS (Cross-Origin Resource Sharing)."""

    def __init__(self):
        """Initialize CORS configuration based on environment."""
        self.environment = os.environ.get("ENVIRONMENT", "development")
        self.allowed_origins: Set[str] = set()

        # Load environment-specific defaults
        self._load_environment_defaults()

        # Load custom allowed origins
        self._load_custom_origins()

        # Load allowed domains
        self._load_allowed_domains()

        # Load staging dev origins if applicable
        if self.environment == "staging":
            self._load_staging_dev_origins()

    def _load_environment_defaults(self):
        """Load default origins based on environment."""
        if self.environment == "development":
            # Allow localhost origins in development
            self.allowed_origins.update(
                [
                    "http://localhost:3000",
                    "http://localhost:8000",
                    "http://127.0.0.1:3000",
                    "http://127.0.0.1:8000",
                    "http://[::1]:3000",
                    "http://[::1]:8000",
                ]
            )
        elif self.environment == "staging":
            # Staging origins
            self.allowed_origins.update(
                [
                    "https://staging.example.edu",
                    "https://preview.example.edu",
                    "https://beta.example.edu",
                ]
            )
        elif self.environment == "production":
            # Production origins
            self.allowed_origins.update(
                [
                    "https://example.edu",
                    "https://www.example.edu",
                    "https://app.example.edu",
                ]
            )

    def _load_custom_origins(self):
        """Load custom allowed origins from environment variable."""
        custom_origins = os.environ.get("ALLOWED_ORIGINS", "")
        if custom_origins:
            for origin in custom_origins.split(","):
                origin = origin.strip()
                if origin and self._validate_origin(origin):
                    # In production, only allow HTTPS
                    if self.environment == "production" and origin.startswith("http://"):
                        continue
                    self.allowed_origins.add(origin)

    def _load_allowed_domains(self):
        """Load allowed domains and generate HTTPS variants."""
        allowed_domains = os.environ.get("ALLOWED_DOMAINS", "")
        if allowed_domains:
            for domain in allowed_domains.split(","):
                domain = domain.strip()
                if domain:
                    # Generate HTTPS variants
                    self.allowed_origins.add(f"https://{domain}")
                    self.allowed_origins.add(f"https://www.{domain}")

    def _load_staging_dev_origins(self):
        """Load development origins allowed in staging."""
        staging_dev_origins = os.environ.get("STAGING_DEV_ORIGINS", "")
        if staging_dev_origins:
            for origin in staging_dev_origins.split(","):
                origin = origin.strip()
                if origin and self._validate_origin(origin):
                    self.allowed_origins.add(origin)

    def _validate_origin(self, origin: str) -> bool:
        """Validate origin format."""
        if not origin:
            return False

        try:
            parsed = urlparse(origin)
            # Must have scheme and netloc
            if not parsed.scheme or not parsed.netloc:
                return False
            # Must be http or https
            if parsed.scheme not in ["http", "https"]:
                return False
            # Should not have path, query, or fragment
            if parsed.path and parsed.path != "/" or parsed.query or parsed.fragment:
                return False
            return True
        except Exception:
            return False

    def is_origin_allowed(self, origin: Optional[str]) -> bool:
        """Check if an origin is allowed."""
        if not origin:
            return False

        # Direct match
        if origin in self.allowed_origins:
            return True

        # Check wildcard subdomain matching if enabled
        if os.environ.get("ALLOW_SUBDOMAIN_WILDCARDS", "").lower() == "true":
            for allowed in self.allowed_origins:
                if allowed.startswith("https://*."):
                    # Extract domain from wildcard pattern
                    domain = allowed[10:]  # Remove "https://*."
                    # Check if origin matches the domain
                    parsed = urlparse(origin)
                    if parsed.scheme == "https" and (
                        parsed.netloc == domain or parsed.netloc.endswith(f".{domain}")
                    ):
                        return True

        return False

    def validate_origin_header(self, origin: str) -> bool:
        """Validate origin header (alias for is_origin_allowed)."""
        return self.is_origin_allowed(origin)

    def get_cors_options(self) -> Dict[str, any]:
        """Get CORS middleware options."""
        return {
            "allow_origins": list(self.allowed_origins),
            "allow_credentials": True,
            "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
            "allow_headers": [
                "Content-Type",
                "Authorization",
                "Accept",
                "Origin",
                "User-Agent",
                "X-Requested-With",
                "X-CSRF-Token",
            ],
            "expose_headers": [
                "Content-Length",
                "Content-Type",
                "X-Request-ID",
                "X-Process-Time",
            ],
            "max_age": 86400,  # 24 hours
        }

    def get_preflight_headers(
        self, origin: str, request_method: str, request_headers: Optional[str] = None
    ) -> Dict[str, str]:
        """Get headers for preflight response."""
        # Check if origin is allowed
        if not self.is_origin_allowed(origin):
            return {}

        # Check if method is allowed
        allowed_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
        if request_method not in allowed_methods:
            return {}

        # Check if headers are allowed
        if request_headers:
            allowed_headers = {
                "content-type",
                "authorization",
                "accept",
                "origin",
                "user-agent",
                "x-requested-with",
                "x-csrf-token",
            }
            for header in request_headers.split(","):
                header = header.strip().lower()
                if header and header not in allowed_headers:
                    return {}

        return {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": ", ".join(allowed_methods),
            "Access-Control-Allow-Headers": ", ".join(
                [
                    "Content-Type",
                    "Authorization",
                    "Accept",
                    "Origin",
                    "User-Agent",
                    "X-Requested-With",
                    "X-CSRF-Token",
                ]
            ),
            "Access-Control-Max-Age": "86400",
            "Vary": "Origin",
        }

    def get_response_headers(self, origin: str) -> Dict[str, str]:
        """Get headers for regular response."""
        if not self.is_origin_allowed(origin):
            return {}

        return {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Expose-Headers": ", ".join(
                ["Content-Length", "Content-Type", "X-Request-ID", "X-Process-Time"]
            ),
            "Vary": "Origin",
        }


# Global instance
cors_config = CORSConfig()


def get_cors_config() -> Dict[str, any]:
    """Get the CORS configuration options for middleware."""
    return cors_config.get_cors_options()
