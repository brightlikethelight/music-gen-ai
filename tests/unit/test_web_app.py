"""Tests for web application factory."""

import os

os.environ["JWT_SECRET_KEY"] = "test-key"
os.environ["PYTEST_CURRENT_TEST"] = "1"
os.environ["MUSICGEN_SKIP_MODEL_DOWNLOAD"] = "1"

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from musicgen.web.app import create_app

pytestmark = pytest.mark.unit


class TestWebApp:
    def test_create_app_returns_fastapi(self):
        app = create_app()
        assert isinstance(app, FastAPI)

    def test_create_app_metadata(self):
        app = create_app()
        assert app.title == "MusicGen Web UI"

    def test_root_returns_html(self):
        app = create_app()
        client = TestClient(app)
        resp = client.get("/")
        assert resp.status_code == 200
        assert "MusicGen" in resp.text

    def test_api_mounted_at_subpath(self):
        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"
