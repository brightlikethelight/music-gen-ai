"""Tests for CORS configuration."""

import os
from unittest.mock import patch

from musicgen.api.cors_config import CORSConfig


class TestCorsDefaults:
    """Test default origins per environment."""

    def test_dev_includes_localhost(self):
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=False):
            cfg = CORSConfig()
        assert "http://localhost:3000" in cfg.allowed_origins
        assert "http://127.0.0.1:8000" in cfg.allowed_origins

    def test_staging_includes_staging_domains(self):
        with patch.dict(os.environ, {"ENVIRONMENT": "staging"}, clear=False):
            cfg = CORSConfig()
        assert "https://staging.example.edu" in cfg.allowed_origins
        assert "https://preview.example.edu" in cfg.allowed_origins

    def test_production_includes_prod_domains(self):
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}, clear=False):
            cfg = CORSConfig()
        assert "https://example.edu" in cfg.allowed_origins
        assert "https://www.example.edu" in cfg.allowed_origins

    def test_unknown_env_has_no_defaults(self):
        with patch.dict(os.environ, {"ENVIRONMENT": "unknown"}, clear=False):
            cfg = CORSConfig()
        assert len(cfg.allowed_origins) == 0


class TestCustomOrigins:
    """Test custom origin loading and validation."""

    def test_custom_origins_added(self):
        env = {"ENVIRONMENT": "development", "ALLOWED_ORIGINS": "https://custom.example.com"}
        with patch.dict(os.environ, env, clear=False):
            cfg = CORSConfig()
        assert "https://custom.example.com" in cfg.allowed_origins

    def test_production_rejects_http_custom_origin(self):
        env = {"ENVIRONMENT": "production", "ALLOWED_ORIGINS": "http://insecure.example.com"}
        with patch.dict(os.environ, env, clear=False):
            cfg = CORSConfig()
        assert "http://insecure.example.com" not in cfg.allowed_origins

    def test_invalid_origin_rejected(self):
        env = {"ENVIRONMENT": "development", "ALLOWED_ORIGINS": "not-a-url"}
        with patch.dict(os.environ, env, clear=False):
            cfg = CORSConfig()
        assert "not-a-url" not in cfg.allowed_origins

    def test_allowed_domains_generates_https(self):
        env = {"ENVIRONMENT": "development", "ALLOWED_DOMAINS": "foo.edu"}
        with patch.dict(os.environ, env, clear=False):
            cfg = CORSConfig()
        assert "https://foo.edu" in cfg.allowed_origins
        assert "https://www.foo.edu" in cfg.allowed_origins


class TestOriginValidation:
    """Test is_origin_allowed and _validate_origin."""

    def test_allowed_origin_returns_true(self):
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=False):
            cfg = CORSConfig()
        assert cfg.is_origin_allowed("http://localhost:3000") is True

    def test_unknown_origin_returns_false(self):
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=False):
            cfg = CORSConfig()
        assert cfg.is_origin_allowed("https://evil.com") is False

    def test_none_origin_returns_false(self):
        cfg = CORSConfig()
        assert cfg.is_origin_allowed(None) is False

    def test_validate_origin_rejects_ftp(self):
        cfg = CORSConfig()
        assert cfg._validate_origin("ftp://files.example.com") is False

    def test_validate_origin_rejects_path(self):
        cfg = CORSConfig()
        assert cfg._validate_origin("https://example.com/path") is False


class TestCorsOptions:
    """Test get_cors_options return values."""

    def test_cors_options_keys(self):
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=False):
            cfg = CORSConfig()
        opts = cfg.get_cors_options()
        assert opts["allow_credentials"] is True
        assert "GET" in opts["allow_methods"]
        assert "Authorization" in opts["allow_headers"]
        assert opts["max_age"] == 86400

    def test_expose_headers_present(self):
        cfg = CORSConfig()
        opts = cfg.get_cors_options()
        assert "X-Request-ID" in opts["expose_headers"]


class TestPreflightHeaders:
    """Test preflight and response header generation."""

    def test_preflight_returns_headers_for_allowed_origin(self):
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=False):
            cfg = CORSConfig()
        headers = cfg.get_preflight_headers("http://localhost:3000", "GET")
        assert headers["Access-Control-Allow-Origin"] == "http://localhost:3000"

    def test_preflight_empty_for_disallowed_origin(self):
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=False):
            cfg = CORSConfig()
        assert cfg.get_preflight_headers("https://evil.com", "GET") == {}

    def test_preflight_empty_for_disallowed_method(self):
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=False):
            cfg = CORSConfig()
        assert cfg.get_preflight_headers("http://localhost:3000", "TRACE") == {}

    def test_preflight_rejects_unknown_header(self):
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=False):
            cfg = CORSConfig()
        assert cfg.get_preflight_headers("http://localhost:3000", "GET", "X-Evil-Header") == {}

    def test_response_headers_for_allowed_origin(self):
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=False):
            cfg = CORSConfig()
        headers = cfg.get_response_headers("http://localhost:3000")
        assert headers["Vary"] == "Origin"
        assert "Content-Length" in headers["Access-Control-Expose-Headers"]

    def test_response_headers_empty_for_disallowed(self):
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}, clear=False):
            cfg = CORSConfig()
        assert cfg.get_response_headers("http://evil.com") == {}
