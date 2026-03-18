"""Tests for authentication middleware."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from musicgen.api.middleware.auth import (
    AuthenticationMiddleware,
    RoleChecker,
    TierChecker,
    TokenType,
    UserClaims,
    UserRole,
    get_auth_middleware,
    get_bearer_scheme,
    get_oauth2_scheme,
    require_admin,
    require_developer,
    require_moderator,
    require_pro_tier,
    require_research_tier,
    require_researcher,
    require_user,
)
from musicgen.utils.exceptions import AuthenticationError, AuthorizationError

pytestmark = pytest.mark.unit


@pytest.fixture()
def auth():
    """Fresh AuthenticationMiddleware instance (Redis disabled via env)."""
    return AuthenticationMiddleware()


@pytest.fixture()
def sample_user_claims() -> UserClaims:
    """Minimal UserClaims for checker tests."""
    now = datetime.now(timezone.utc)
    return UserClaims(
        user_id="u1",
        email="u1@test.com",
        username="user1",
        roles=[UserRole.USER],
        tier="free",
        is_verified=True,
        token_type=TokenType.ACCESS,
        issued_at=now,
        expires_at=now + timedelta(minutes=30),
        jti="jti-abc",
    )


# ── RoleChecker ──────────────────────────────────────────────────────


class TestRoleChecker:
    """Test role-based access control."""

    def test_require_any_role_passes(self, sample_user_claims):
        checker = RoleChecker([UserRole.USER, UserRole.ADMIN], require_all=False)
        result = checker(sample_user_claims)
        assert result.user_id == "u1"

    def test_require_any_role_fails(self, sample_user_claims):
        checker = RoleChecker([UserRole.ADMIN], require_all=False)
        with pytest.raises(AuthorizationError, match="Insufficient permissions"):
            checker(sample_user_claims)

    def test_require_all_roles_passes(self):
        now = datetime.now(timezone.utc)
        user = UserClaims(
            user_id="u2",
            email="u2@test.com",
            username="user2",
            roles=[UserRole.ADMIN, UserRole.RESEARCHER],
            tier="pro",
            is_verified=True,
            token_type=TokenType.ACCESS,
            issued_at=now,
            expires_at=now + timedelta(minutes=30),
        )
        checker = RoleChecker([UserRole.ADMIN, UserRole.RESEARCHER], require_all=True)
        assert checker(user).user_id == "u2"

    def test_require_all_roles_fails_when_missing(self, sample_user_claims):
        checker = RoleChecker([UserRole.USER, UserRole.ADMIN], require_all=True)
        with pytest.raises(AuthorizationError, match="Insufficient permissions"):
            checker(sample_user_claims)

    def test_no_user_raises(self):
        checker = RoleChecker([UserRole.USER])
        with pytest.raises(AuthorizationError, match="Authentication required"):
            checker(None)


# ── TierChecker ──────────────────────────────────────────────────────


class TestTierChecker:
    """Test tier-based access control."""

    def test_matching_tier_passes(self, sample_user_claims):
        checker = TierChecker(["free", "pro"])
        assert checker(sample_user_claims).tier == "free"

    def test_non_matching_tier_raises(self, sample_user_claims):
        checker = TierChecker(["pro", "research"])
        with pytest.raises(AuthorizationError, match="Subscription upgrade required"):
            checker(sample_user_claims)

    def test_no_user_raises(self):
        checker = TierChecker(["free"])
        with pytest.raises(AuthorizationError, match="Authentication required"):
            checker(None)


# ── Token creation & verification ────────────────────────────────────


class TestTokenVerification:
    """Test JWT token verification."""

    def test_roundtrip_access_token(self, auth):
        token = auth.create_access_token(
            user_id="u1",
            email="u1@test.com",
            username="user1",
            roles=["user"],
        )
        claims = auth.verify_token(token)
        assert claims.user_id == "u1"
        assert claims.email == "u1@test.com"
        assert claims.token_type == TokenType.ACCESS
        assert UserRole.USER in claims.roles

    def test_expired_token_raises(self, auth):
        token = auth.create_access_token(
            user_id="u1",
            email="u1@test.com",
            username="user1",
            roles=["user"],
            expires_delta=timedelta(seconds=-1),
        )
        with pytest.raises(AuthenticationError, match="expired|verification failed"):
            auth.verify_token(token)

    def test_invalid_token_raises(self, auth):
        with pytest.raises(AuthenticationError):
            auth.verify_token("not-a-jwt")

    def test_unverified_user_rejected(self, auth):
        token = auth.create_access_token(
            user_id="u1",
            email="u1@test.com",
            username="user1",
            roles=["user"],
            is_verified=False,
        )
        with pytest.raises(AuthenticationError, match="not verified|verification failed"):
            auth.verify_token(token)

    def test_refresh_token_roundtrip(self, auth):
        token = auth.create_refresh_token(user_id="u1")
        claims = auth.verify_token(token)
        assert claims.user_id == "u1"
        assert claims.token_type == TokenType.REFRESH


# ── Blacklisting ─────────────────────────────────────────────────────


class TestBlacklist:
    """Test token blacklisting with mock Redis."""

    def test_blacklist_token_with_redis(self, auth):
        mock_redis = MagicMock()
        auth.redis_client = mock_redis

        future = datetime.now(timezone.utc) + timedelta(hours=1)
        result = auth.blacklist_token("jti-123", future)

        assert result is True
        mock_redis.setex.assert_called_once()
        args = mock_redis.setex.call_args[0]
        assert args[0] == "blacklist:jti-123"
        assert args[2] == "revoked"

    def test_blacklist_already_expired_returns_true(self, auth):
        mock_redis = MagicMock()
        auth.redis_client = mock_redis
        past = datetime.now(timezone.utc) - timedelta(hours=1)

        assert auth.blacklist_token("jti-old", past) is True
        mock_redis.setex.assert_not_called()

    def test_blacklist_no_redis_returns_false(self, auth):
        auth.redis_client = None
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        assert auth.blacklist_token("jti-x", future) is False

    def test_blacklist_fail_closed_rejects_when_no_redis(self, auth, monkeypatch):
        """Fail-closed policy: reject token when Redis is unavailable."""
        auth.redis_client = None
        monkeypatch.setenv("TOKEN_BLACKLIST_FAIL_POLICY", "closed")
        assert auth._is_token_blacklisted("some-jti") is True

    def test_blacklist_fail_open_accepts_when_no_redis(self, auth, monkeypatch):
        """Fail-open policy: accept token when Redis is unavailable."""
        auth.redis_client = None
        monkeypatch.setenv("TOKEN_BLACKLIST_FAIL_POLICY", "open")
        assert auth._is_token_blacklisted("some-jti") is False

    def test_blacklist_fail_closed_rejects_on_redis_error(self, auth, monkeypatch):
        """Fail-closed policy: reject token when Redis raises an error."""
        mock_redis = MagicMock()
        mock_redis.exists.side_effect = Exception("Connection refused")
        auth.redis_client = mock_redis
        monkeypatch.setenv("TOKEN_BLACKLIST_FAIL_POLICY", "closed")
        assert auth._is_token_blacklisted("jti-err") is True

    def test_blacklist_fail_open_accepts_on_redis_error(self, auth, monkeypatch):
        """Fail-open policy: accept token when Redis raises an error."""
        mock_redis = MagicMock()
        mock_redis.exists.side_effect = Exception("Connection refused")
        auth.redis_client = mock_redis
        monkeypatch.setenv("TOKEN_BLACKLIST_FAIL_POLICY", "open")
        assert auth._is_token_blacklisted("jti-err") is False

    def test_blacklist_empty_jti_returns_false(self, auth):
        """Empty JTI should never be blacklisted."""
        assert auth._is_token_blacklisted("") is False

    def test_blacklisted_token_rejected(self, auth):
        mock_redis = MagicMock()
        mock_redis.exists.return_value = 1
        auth.redis_client = mock_redis

        token = auth.create_access_token(
            user_id="u1",
            email="u1@test.com",
            username="user1",
            roles=["user"],
        )
        with pytest.raises(AuthenticationError, match="blacklisted|verification failed"):
            auth.verify_token(token)


# ── Refresh flow ─────────────────────────────────────────────────────


class TestRefreshFlow:
    """Test refresh_access_token flow."""

    def test_refresh_returns_new_token_pair(self, auth):
        refresh = auth.create_refresh_token(user_id="u1")
        new_access, new_refresh = auth.refresh_access_token(refresh)

        # Both should be valid JWT strings
        assert isinstance(new_access, str) and len(new_access) > 20
        assert isinstance(new_refresh, str) and len(new_refresh) > 20

        # New access token should verify
        claims = auth.verify_token(new_access)
        assert claims.user_id == "u1"
        assert claims.token_type == TokenType.ACCESS

    def test_refresh_with_access_token_fails(self, auth):
        access = auth.create_access_token(
            user_id="u1",
            email="u1@test.com",
            username="user1",
            roles=["user"],
        )
        with pytest.raises(AuthenticationError, match="refresh failed"):
            auth.refresh_access_token(access)

    def test_refresh_preserves_admin_roles(self, auth):
        """Verify that refreshing an admin token doesn't demote to USER."""
        refresh = auth.create_refresh_token(
            user_id="admin1",
            email="admin@test.com",
            username="admin_user",
            roles=[UserRole.ADMIN.value, UserRole.RESEARCHER.value],
            tier="research",
            is_verified=True,
        )
        new_access, _ = auth.refresh_access_token(refresh)
        claims = auth.verify_token(new_access)
        assert claims.user_id == "admin1"
        assert UserRole.ADMIN in claims.roles
        assert UserRole.RESEARCHER in claims.roles
        assert claims.tier == "research"
        assert claims.email == "admin@test.com"
        assert claims.username == "admin_user"

    def test_refresh_preserves_user_claims(self, auth):
        """Verify all user claims survive a refresh cycle."""
        refresh = auth.create_refresh_token(
            user_id="u5",
            email="pro@test.com",
            username="prouser",
            roles=[UserRole.USER.value],
            tier="pro",
            is_verified=True,
        )
        new_access, new_refresh = auth.refresh_access_token(refresh)
        claims = auth.verify_token(new_access)
        assert claims.tier == "pro"
        assert claims.email == "pro@test.com"
        assert claims.username == "prouser"
        # Verify chained refresh also preserves claims
        new_access2, _ = auth.refresh_access_token(new_refresh)
        claims2 = auth.verify_token(new_access2)
        assert claims2.tier == "pro"
        assert claims2.email == "pro@test.com"


# ── Factory functions ────────────────────────────────────────────────


class TestFactories:
    """Test lazy-init factory functions."""

    def test_get_auth_middleware_returns_instance(self):
        mw = get_auth_middleware()
        assert isinstance(mw, AuthenticationMiddleware)

    def test_get_auth_middleware_is_singleton(self):
        a = get_auth_middleware()
        b = get_auth_middleware()
        assert a is b

    def test_get_oauth2_scheme_returns_scheme(self):
        scheme = get_oauth2_scheme()
        assert scheme is not None

    def test_get_bearer_scheme_returns_instance(self):
        scheme = get_bearer_scheme()
        assert scheme is not None

    def test_create_access_token_with_int_expires(self, auth):
        token = auth.create_access_token(
            user_id="u1",
            email="u1@test.com",
            username="user1",
            roles=["user"],
            expires_delta=60,
        )
        claims = auth.verify_token(token)
        assert claims.user_id == "u1"

    def test_create_refresh_token_with_int_expires(self, auth):
        token = auth.create_refresh_token(
            user_id="u1",
            expires_delta=7,
        )
        claims = auth.verify_token(token)
        assert claims.user_id == "u1"
        assert claims.token_type == TokenType.REFRESH

    def test_parse_roles_from_single_string(self):
        now = datetime.now(timezone.utc)
        user = UserClaims(
            user_id="u1",
            email="u1@test.com",
            username="user1",
            roles="admin",
            tier="free",
            is_verified=True,
            token_type=TokenType.ACCESS,
            issued_at=now,
            expires_at=now + timedelta(minutes=30),
        )
        assert user.roles == [UserRole.ADMIN]

    def test_blacklist_token_redis_error(self, auth):
        mock_redis = MagicMock()
        mock_redis.setex.side_effect = Exception("Connection lost")
        auth.redis_client = mock_redis
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        result = auth.blacklist_token("jti-err", future)
        assert result is False

    def test_factory_functions_return_checkers(self):
        assert isinstance(require_admin(), RoleChecker)
        assert isinstance(require_user(), RoleChecker)
        assert isinstance(require_researcher(), RoleChecker)
        assert isinstance(require_moderator(), RoleChecker)
        assert isinstance(require_developer(), RoleChecker)
        from musicgen.api.middleware.auth import TierChecker

        assert isinstance(require_pro_tier(), TierChecker)
        assert isinstance(require_research_tier(), TierChecker)
