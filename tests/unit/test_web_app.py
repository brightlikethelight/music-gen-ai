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

    def test_root_with_static_index_html(self, tmp_path):
        """When index.html exists, root returns FileResponse."""
        from unittest.mock import patch as _patch

        index = tmp_path / "index.html"
        index.write_text("<html><body>MusicGen Static</body></html>")
        with _patch("musicgen.web.app.STATIC_DIR", tmp_path):
            app = create_app()
            client = TestClient(app)
            resp = client.get("/")
            assert resp.status_code == 200
            assert "MusicGen Static" in resp.text

    def test_root_fallback_html_no_index(self, tmp_path):
        """When STATIC_DIR exists but no index.html, root returns fallback HTML."""
        from unittest.mock import patch as _patch

        # tmp_path exists but has no index.html
        with _patch("musicgen.web.app.STATIC_DIR", tmp_path):
            app = create_app()
            client = TestClient(app)
            resp = client.get("/")
            assert resp.status_code == 200
            assert "Static files not found" in resp.text

    def test_static_dir_missing_no_mount(self, tmp_path):
        """When STATIC_DIR doesn't exist, StaticFiles is not mounted."""
        from unittest.mock import patch as _patch

        fake_dir = tmp_path / "nonexistent"
        with _patch("musicgen.web.app.STATIC_DIR", fake_dir):
            app = create_app()
            # StaticFiles should not be mounted, so /static should 404
            client = TestClient(app)
            resp = client.get("/static/foo.js")
            assert resp.status_code == 404

    def test_run_server_callable(self):
        """run_server and main are callable without actually starting."""
        from unittest.mock import patch as _patch

        from musicgen.web.app import main, run_server

        with _patch("uvicorn.run") as mock_run:
            run_server()
            assert mock_run.called

        with _patch("uvicorn.run") as mock_run:
            main()
            assert mock_run.called
