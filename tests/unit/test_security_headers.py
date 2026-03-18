"""Tests for security headers middleware."""

import os

os.environ["JWT_SECRET_KEY"] = "test-key"
os.environ["PYTEST_CURRENT_TEST"] = "1"
os.environ["MUSICGEN_SKIP_MODEL_DOWNLOAD"] = "1"

import pytest
from httpx import ASGITransport, AsyncClient

from musicgen.api.rest.app import app

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_security_headers_present_on_health() -> None:
    """All security headers appear on /health response."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")

    assert resp.status_code == 200
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"
    assert resp.headers["X-XSS-Protection"] == "1; mode=block"
    assert resp.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert "camera=()" in resp.headers["Permissions-Policy"]
    assert "max-age=" in resp.headers["Strict-Transport-Security"]
    assert "default-src 'self'" in resp.headers["Content-Security-Policy"]


@pytest.mark.asyncio
async def test_csp_contains_expected_directives() -> None:
    """CSP header includes required directives for Swagger UI."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")

    csp = resp.headers["Content-Security-Policy"]
    assert "script-src" in csp
    assert "style-src" in csp
    assert "img-src" in csp
    assert "cdn.jsdelivr.net" in csp


@pytest.mark.asyncio
async def test_hsts_includes_subdomains() -> None:
    """HSTS header includes includeSubDomains directive."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")

    hsts = resp.headers["Strict-Transport-Security"]
    assert "includeSubDomains" in hsts
    assert "max-age=31536000" in hsts
