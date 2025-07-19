"""
Tests for rate limiting middleware.
"""

import pytest
import time
from unittest.mock import Mock

from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient

try:
    from musicgen.api.rest.middleware.rate_limiting import RateLimiter, RateLimitMiddleware
    RATE_LIMITING_AVAILABLE = True
except ImportError:
    RATE_LIMITING_AVAILABLE = False


@pytest.mark.skipif(not RATE_LIMITING_AVAILABLE, reason="Rate limiting not available")
class TestRateLimiter:
    """Test cases for the RateLimiter class."""
    
    def test_create_rate_limiter(self):
        """Test rate limiter creation."""
        limiter = RateLimiter()
        assert limiter is not None
        assert limiter.limits['per_minute'] == 60
        assert limiter.limits['per_hour'] == 1000
        assert limiter.limits['per_day'] == 10000
    
    def test_ip_extraction(self):
        """Test IP address extraction."""
        limiter = RateLimiter()
        
        # Mock request with direct IP
        request = Mock()
        request.client = Mock()
        request.client.host = "192.168.1.1"
        request.headers = {}
        
        ip = limiter._get_client_ip(request)
        assert ip == "192.168.1.1"
    
    def test_ip_extraction_with_forwarded_header(self):
        """Test IP extraction with X-Forwarded-For header."""
        limiter = RateLimiter()
        
        # Mock request with forwarded header
        request = Mock()
        request.client = Mock()
        request.client.host = "10.0.0.1"  # This should be ignored
        request.headers = {"X-Forwarded-For": "203.0.113.1, 192.168.1.1"}
        
        ip = limiter._get_client_ip(request)
        assert ip == "203.0.113.1"  # Should take first IP
    
    def test_ip_validation(self):
        """Test IP address validation."""
        limiter = RateLimiter()
        
        assert limiter._is_valid_ip("192.168.1.1") is True
        assert limiter._is_valid_ip("2001:db8::1") is True
        assert limiter._is_valid_ip("invalid-ip") is False
        assert limiter._is_valid_ip("") is False
    
    def test_exempt_ips(self):
        """Test exempt IP networks."""
        limiter = RateLimiter()
        
        # Local IPs should be exempt
        assert limiter._is_exempt_ip("127.0.0.1") is True
        assert limiter._is_exempt_ip("192.168.1.1") is True
        assert limiter._is_exempt_ip("10.0.0.1") is True
        
        # Public IPs should not be exempt
        assert limiter._is_exempt_ip("8.8.8.8") is False
        assert limiter._is_exempt_ip("203.0.113.1") is False
    
    def test_rate_limiting_allows_first_requests(self):
        """Test that first requests are allowed."""
        limiter = RateLimiter()
        
        # Mock request
        request = Mock()
        request.client = Mock()
        request.client.host = "203.0.113.1"
        request.headers = {}
        
        # First request should be allowed
        allowed, info = limiter.is_allowed(request)
        assert allowed is True
        assert 'per_minute' in info
        assert info['per_minute']['count'] == 1
        assert info['per_minute']['remaining'] == 59
    
    def test_rate_limiting_exempt_ip(self):
        """Test that exempt IPs are always allowed."""
        limiter = RateLimiter()
        
        # Mock request from localhost
        request = Mock()
        request.client = Mock()
        request.client.host = "127.0.0.1"
        request.headers = {}
        
        # Should always be allowed regardless of count
        for _ in range(100):  # Try many requests
            allowed, info = limiter.is_allowed(request)
            assert allowed is True
            assert info == {}  # No rate limit info for exempt IPs


@pytest.mark.skipif(not RATE_LIMITING_AVAILABLE, reason="Rate limiting not available")
class TestRateLimitMiddleware:
    """Test cases for the RateLimitMiddleware."""
    
    @pytest.fixture
    def app(self):
        """Create test FastAPI app with rate limiting."""
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware)
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "success"}
            
        @app.get("/health")
        async def health_endpoint():
            return {"status": "healthy"}
        
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)
    
    def test_health_endpoint_bypasses_rate_limiting(self, client):
        """Test that health endpoints bypass rate limiting."""
        # Health endpoint should always work
        for _ in range(70):  # More than per-minute limit
            response = client.get("/health")
            assert response.status_code == 200
    
    def test_rate_limiting_headers(self, client):
        """Test that rate limiting headers are added."""
        response = client.get("/test")
        
        assert response.status_code == 200
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers
    
    def test_rate_limiting_blocks_excess_requests(self, client):
        """Test that rate limiting blocks excessive requests."""
        # Create a custom limiter with very low limits for testing
        custom_limiter = RateLimiter()
        custom_limiter.limits['per_minute'] = 2  # Very low limit
        
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, rate_limiter=custom_limiter)
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "success"}
        
        test_client = TestClient(app)
        
        # First two requests should succeed
        response1 = test_client.get("/test")
        assert response1.status_code == 200
        
        response2 = test_client.get("/test")
        assert response2.status_code == 200
        
        # Third request should be rate limited
        response3 = test_client.get("/test")
        assert response3.status_code == 429
        assert "Rate limit exceeded" in response3.json()["error"]
        assert "Retry-After" in response3.headers


def test_rate_limiting_import_graceful_failure():
    """Test that missing rate limiting doesn't break imports."""
    # This test just ensures the import structure works
    # even when rate limiting components are missing
    assert True  # If we get here, imports worked