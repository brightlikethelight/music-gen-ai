"""
Thread-safe state management for MusicGen API.

All mutable global dicts are wrapped with asyncio.Lock to prevent
race conditions from concurrent async handlers.
"""

import asyncio
import logging
from typing import Any, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class JobStatus(BaseModel):
    """Job status response."""

    job_id: str
    status: str  # "queued", "processing", "completed", "failed"
    progress: float = 0.0
    message: str = ""
    audio_url: Optional[str] = None
    error: Optional[str] = None


class StateManager:
    """Thread-safe state manager for in-memory API state.

    Uses asyncio.Lock to prevent race conditions on concurrent
    mutation of jobs, users, playlists, and model cache.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, JobStatus] = {}
        self._users: dict[str, dict[str, Any]] = {}
        self._playlists: dict[str, dict[str, Any]] = {}
        self._model_cache: dict[str, Any] = {}
        self._jobs_lock = asyncio.Lock()
        self._users_lock = asyncio.Lock()
        self._playlists_lock = asyncio.Lock()
        self._model_lock = asyncio.Lock()

    # --- Jobs ---

    async def add_job(self, job_id: str, job: JobStatus) -> None:
        async with self._jobs_lock:
            self._jobs[job_id] = job

    async def get_job(self, job_id: str) -> Optional[JobStatus]:
        async with self._jobs_lock:
            return self._jobs.get(job_id)

    async def update_job(self, job_id: str, **kwargs: Any) -> None:
        async with self._jobs_lock:
            if job_id in self._jobs:
                for key, value in kwargs.items():
                    setattr(self._jobs[job_id], key, value)

    async def get_all_jobs(self) -> dict[str, JobStatus]:
        async with self._jobs_lock:
            return dict(self._jobs)

    # --- Users ---

    async def add_user(self, user_id: str, user: dict[str, Any]) -> None:
        async with self._users_lock:
            self._users[user_id] = user

    async def get_user(self, user_id: str) -> Optional[dict[str, Any]]:
        async with self._users_lock:
            return self._users.get(user_id)

    async def find_user_by_email(self, email: str) -> Optional[dict[str, Any]]:
        async with self._users_lock:
            for u in self._users.values():
                if u.get("email") == email:
                    return u
            return None

    async def find_user_by_username(self, username: str) -> Optional[dict[str, Any]]:
        async with self._users_lock:
            for u in self._users.values():
                if u.get("username") == username:
                    return u
            return None

    async def update_user(self, user_id: str, **kwargs: Any) -> None:
        async with self._users_lock:
            if user_id in self._users:
                self._users[user_id].update(kwargs)

    async def increment_user_field(self, user_id: str, field: str, amount: int = 1) -> None:
        """Atomically increment a numeric user field under lock."""
        async with self._users_lock:
            if user_id in self._users:
                current = self._users[user_id].get(field, 0)
                self._users[user_id][field] = current + amount

    async def get_all_users(self) -> dict[str, dict[str, Any]]:
        async with self._users_lock:
            return dict(self._users)

    # --- Playlists ---

    async def add_playlist(self, playlist_id: str, playlist: dict[str, Any]) -> None:
        async with self._playlists_lock:
            self._playlists[playlist_id] = playlist

    async def get_playlists_for_user(self, user_id: str) -> list[dict[str, Any]]:
        async with self._playlists_lock:
            return [p for p in self._playlists.values() if p["user_id"] == user_id]

    async def get_all_playlists(self) -> dict[str, dict[str, Any]]:
        async with self._playlists_lock:
            return dict(self._playlists)

    # --- Model Cache ---

    async def get_model(self, model_name: str) -> Optional[dict[str, Any]]:
        async with self._model_lock:
            return self._model_cache.get(model_name)

    async def set_model(self, model_name: str, model_data: dict[str, Any]) -> None:
        async with self._model_lock:
            self._model_cache[model_name] = model_data

    # --- Direct access for backward compatibility with tests ---
    # Tests directly access _jobs, _users, _playlists; these properties
    # expose the underlying dicts. New code should use the async methods.

    async def reset(self) -> None:
        """Clear all state. For test fixtures — prefer over direct dict access."""
        async with self._jobs_lock:
            self._jobs.clear()
        async with self._users_lock:
            self._users.clear()
        async with self._playlists_lock:
            self._playlists.clear()
        async with self._model_lock:
            self._model_cache.clear()

    @property
    def jobs(self) -> dict[str, JobStatus]:
        """Direct dict access — use async methods for new code."""
        return self._jobs

    @property
    def users(self) -> dict[str, dict[str, Any]]:
        """Direct dict access — use async methods for new code."""
        return self._users

    @property
    def playlists(self) -> dict[str, dict[str, Any]]:
        """Direct dict access — use async methods for new code."""
        return self._playlists

    @property
    def model_cache(self) -> dict[str, Any]:
        """Direct dict access — use async methods for new code."""
        return self._model_cache


# Singleton state instance
state = StateManager()
