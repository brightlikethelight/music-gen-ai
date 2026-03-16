"""Tests for REST API endpoints.

Covers registration, login, generation, audio serving, search, models,
health services, and batch generation endpoints.
"""

import os
import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Import API modules - handle missing dependencies gracefully
try:
    from musicgen.api.middleware.auth import UserRole, get_auth_middleware
    from musicgen.api.rest.app import _jobs, _playlists, _users, app

    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    app = None

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not API_AVAILABLE, reason="API dependencies not available"),
]


def _reset_rate_limiters(application):
    """Clear rate limiter state to avoid cross-test 429s."""
    try:
        current = application.middleware_stack
        while current:
            if hasattr(current, "rate_limiter") and hasattr(current.rate_limiter, "requests"):
                current.rate_limiter.requests.clear()
            current = getattr(current, "app", None)
    except (AttributeError, TypeError):
        pass
    try:
        from musicgen.api.rest.middleware.rate_limiting import rate_limiter

        rate_limiter.requests.clear()
    except (ImportError, AttributeError):
        pass


@pytest.fixture
def client():
    """Create a fresh TestClient with cleared module-level state."""
    _jobs.clear()
    _users.clear()
    _playlists.clear()
    _reset_rate_limiters(app)
    with TestClient(app) as c:
        _reset_rate_limiters(app)
        yield c
    _jobs.clear()
    _users.clear()
    _playlists.clear()


@pytest.fixture
def auth_token():
    """Return a valid bearer token for authenticated endpoints."""
    auth = get_auth_middleware()
    token = auth.create_access_token(
        user_id="unit_test_user",
        email="unit@test.com",
        username="unittester",
        roles=[UserRole.USER.value],
    )
    return {"Authorization": f"Bearer {token}"}


def _register(client, username="alice", email="alice@example.com", password="securepass1"):
    """Helper: register a user and return the response."""
    return client.post(
        "/auth/register",
        json={
            "username": username,
            "email": email,
            "password": password,
        },
    )


# ── Registration ────────────────────────────────────────────────────────


class TestRegistration:
    """Test POST /auth/register."""

    def test_register_success(self, client):
        resp = _register(client)
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert data["user"]["username"] == "alice"
        assert data["user"]["email"] == "alice@example.com"
        assert data["user"]["roles"] == ["user"]

    def test_register_duplicate_email(self, client):
        _register(client, username="bob", email="dup@example.com")
        resp = _register(client, username="carol", email="dup@example.com")
        assert resp.status_code == 400
        assert "already be in use" in resp.json()["detail"]

    def test_register_duplicate_username(self, client):
        _register(client, username="dupname", email="a@example.com")
        resp = _register(client, username="dupname", email="b@example.com")
        assert resp.status_code == 400

    def test_register_validation_short_password(self, client):
        resp = _register(client, password="short")
        assert resp.status_code == 422

    def test_register_validation_bad_email(self, client):
        resp = _register(client, email="not-an-email")
        assert resp.status_code == 422

    def test_register_validation_short_username(self, client):
        resp = _register(client, username="ab")
        assert resp.status_code == 422


# ── Login ───────────────────────────────────────────────────────────────


class TestLogin:
    """Test POST /auth/login (OAuth2 form)."""

    def test_login_success(self, client):
        _register(client)
        resp = client.post(
            "/auth/login",
            data={"username": "alice@example.com", "password": "securepass1"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["user"]["email"] == "alice@example.com"

    def test_login_wrong_password(self, client):
        _register(client)
        resp = client.post(
            "/auth/login",
            data={"username": "alice@example.com", "password": "wrongpassword1"},
        )
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Invalid credentials"

    def test_login_nonexistent_user(self, client):
        resp = client.post(
            "/auth/login",
            data={"username": "nobody@example.com", "password": "whatever123"},
        )
        assert resp.status_code == 401


# ── Generate ────────────────────────────────────────────────────────────


class TestGeneration:
    """Test POST /generate and GET /status/{job_id}."""

    @patch("musicgen.api.rest.app.generate_music_task")
    def test_generate_creates_job(self, _mock_task, client, auth_token):
        resp = client.post(
            "/generate",
            json={"prompt": "upbeat jazz"},
            headers=auth_token,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued"
        assert "job_id" in data

    def test_generate_requires_auth(self, client):
        resp = client.post("/generate", json={"prompt": "test"})
        assert resp.status_code in (401, 403)

    def test_generate_validation_missing_prompt(self, client, auth_token):
        resp = client.post("/generate", json={}, headers=auth_token)
        assert resp.status_code == 422

    @patch("musicgen.api.rest.app.generate_music_task")
    def test_generate_validation_bad_duration(self, _mock_task, client, auth_token):
        resp = client.post(
            "/generate",
            json={"prompt": "test", "duration": 9999},
            headers=auth_token,
        )
        assert resp.status_code == 422

    @patch("musicgen.api.rest.app.generate_music_task")
    def test_generate_validation_bad_temperature(self, _mock_task, client, auth_token):
        resp = client.post(
            "/generate",
            json={"prompt": "test", "temperature": 0.0},
            headers=auth_token,
        )
        assert resp.status_code == 422

    @patch("musicgen.api.rest.app.generate_music_task")
    def test_job_status_found(self, _mock_task, client, auth_token):
        resp = client.post("/generate", json={"prompt": "test"}, headers=auth_token)
        job_id = resp.json()["job_id"]
        status_resp = client.get(f"/status/{job_id}")
        assert status_resp.status_code == 200
        assert status_resp.json()["job_id"] == job_id

    def test_job_status_not_found(self, client):
        resp = client.get(f"/status/{uuid.uuid4()}")
        assert resp.status_code == 404

    @patch("musicgen.api.rest.app.generate_music_task")
    def test_job_alias_endpoint(self, _mock_task, client, auth_token):
        resp = client.post("/generate", json={"prompt": "test"}, headers=auth_token)
        job_id = resp.json()["job_id"]
        alias_resp = client.get(f"/generate/job/{job_id}")
        assert alias_resp.status_code == 200
        assert alias_resp.json()["job_id"] == job_id

    def test_job_alias_not_found(self, client):
        resp = client.get(f"/generate/job/{uuid.uuid4()}")
        assert resp.status_code == 404


# ── Audio serving ───────────────────────────────────────────────────────


class TestAudioServing:
    """Test GET /audio/{filename}."""

    def test_audio_file_not_found(self, client):
        resp = client.get("/audio/nonexistent.wav")
        assert resp.status_code == 404

    def test_audio_path_traversal(self, client):
        resp = client.get("/audio/../../etc/passwd")
        # The endpoint strips directory components, so the resolved name
        # is just "passwd" — which won't exist. Either 404 or 403 is fine.
        assert resp.status_code in (403, 404)

    def test_audio_serves_existing_file(self, client):
        """Place a dummy wav in the outputs dir the endpoint resolves to."""
        from pathlib import Path

        outputs_dir = Path.cwd() / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        wav = outputs_dir / "unit_test_served.wav"
        try:
            wav.write_bytes(b"RIFF" + b"\x00" * 100)
            resp = client.get("/audio/unit_test_served.wav")
            assert resp.status_code == 200
        finally:
            wav.unlink(missing_ok=True)


# ── Batch generation ────────────────────────────────────────────────────


class TestBatchGeneration:
    """Test POST /generate/batch."""

    @patch("musicgen.api.rest.app.generate_music_task")
    def test_batch_creates_multiple_jobs(self, _mock_task, client, auth_token):
        payload = {
            "requests": [
                {"prompt": "track one"},
                {"prompt": "track two"},
            ],
        }
        resp = client.post("/generate/batch", json=payload, headers=auth_token)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_jobs"] == 2
        assert len(data["jobs"]) == 2

    @patch("musicgen.api.rest.app.generate_music_task")
    def test_batch_rejects_over_limit(self, _mock_task, client, auth_token):
        payload = {"requests": [{"prompt": f"t{i}"} for i in range(11)]}
        resp = client.post("/generate/batch", json=payload, headers=auth_token)
        assert resp.status_code == 400

    def test_batch_requires_auth(self, client):
        resp = client.post(
            "/generate/batch",
            json={"requests": [{"prompt": "t"}]},
        )
        assert resp.status_code in (401, 403)


# ── Models / metrics / health ───────────────────────────────────────────


class TestInfoEndpoints:
    """Test read-only informational endpoints."""

    def test_models_list(self, client):
        resp = client.get("/models")
        assert resp.status_code == 200
        models = resp.json()["models"]
        assert len(models) == 3
        names = [m["name"] for m in models]
        assert "facebook/musicgen-small" in names

    def test_health_services(self, client):
        resp = client.get("/health/services")
        assert resp.status_code == 200
        data = resp.json()
        assert "overall_status" in data
        assert "services" in data
        assert "generation" in data["services"]


# ── Search ──────────────────────────────────────────────────────────────


class TestSearch:
    """Test GET /search."""

    def test_search_all(self, client, auth_token):
        resp = client.get("/search?query=jazz", headers=auth_token)
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "jazz"
        assert "tracks" in data["results"]
        assert "playlists" in data["results"]
        assert "users" in data["results"]

    def test_search_tracks_only(self, client, auth_token):
        resp = client.get("/search?query=rock&type=tracks", headers=auth_token)
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "tracks"
        assert len(data["results"]["playlists"]) == 0
        assert len(data["results"]["users"]) == 0

    def test_search_requires_auth(self, client):
        resp = client.get("/search?query=test")
        assert resp.status_code in (401, 403)
