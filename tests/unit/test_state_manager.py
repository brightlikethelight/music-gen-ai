"""Tests for StateManager gap coverage."""

import asyncio

import pytest

from musicgen.api.rest.state import StateManager

pytestmark = pytest.mark.unit


def _run(coro):
    """Run async coroutine in a new event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestStateManagerGaps:
    """Cover untested StateManager branches."""

    def test_update_user_nonexistent_is_noop(self):
        sm = StateManager()
        _run(sm.update_user("no-such-user", email="new@test.com"))
        assert _run(sm.get_user("no-such-user")) is None

    def test_get_all_playlists(self):
        sm = StateManager()
        _run(sm.add_playlist("p1", {"id": "p1", "user_id": "u1", "name": "PL1"}))
        _run(sm.add_playlist("p2", {"id": "p2", "user_id": "u2", "name": "PL2"}))
        all_pl = _run(sm.get_all_playlists())
        assert len(all_pl) == 2
        assert "p1" in all_pl
        assert "p2" in all_pl

    def test_increment_nonexistent_user_is_noop(self):
        sm = StateManager()
        _run(sm.increment_user_field("no-such-user", "tracks_generated"))
        assert _run(sm.get_user("no-such-user")) is None

    def test_reset_clears_all_state(self):
        sm = StateManager()
        _run(sm.add_user("u1", {"email": "a@b.com"}))
        _run(sm.add_playlist("p1", {"id": "p1", "user_id": "u1", "name": "PL"}))
        from musicgen.api.rest.state import JobStatus

        _run(sm.add_job("j1", JobStatus(job_id="j1", status="queued")))
        _run(sm.set_model("m1", {"model": "test"}))
        _run(sm.reset())
        assert _run(sm.get_all_users()) == {}
        assert _run(sm.get_all_playlists()) == {}
        assert _run(sm.get_all_jobs()) == {}
        assert _run(sm.get_model("m1")) is None
