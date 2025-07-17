"""
User Management Service FastAPI Application

Microservice for user authentication, authorization, profile management,
and session handling with JWT tokens and role-based access control.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram
import bcrypt
import jwt
from sqlalchemy.ext.asyncio import AsyncSession

from .service import UserManagementService
from .config import UserManagementServiceConfig
from .models import (
    UserCreate,
    UserLogin,
    UserResponse,
    UserUpdate,
    TokenResponse,
    UserProfile,
    PasswordReset,
    PasswordChange,
    UserStats
)
from .dependencies import get_user_service, get_current_user, get_current_active_user
from .database import get_db_session
from ..shared.observability import get_tracer, create_span
from ..shared.middleware import MetricsMiddleware, TracingMiddleware
from ..shared.exceptions import (
    AuthenticationError,
    AuthorizationError,
    ValidationError,
    UserManagementServiceError
)


# Metrics
AUTH_REQUESTS = Counter(
    'auth_requests_total',
    'Total authentication requests',
    ['operation', 'status']
)
AUTH_DURATION = Histogram(
    'auth_request_duration_seconds',
    'Authentication request duration',
    ['operation']
)
ACTIVE_SESSIONS = Counter(
    'active_sessions_total',
    'Total active user sessions'
)

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)
security = HTTPBearer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager."""
    config = app.state.config
    
    # Initialize user service
    user_service = UserManagementService(config)
    await user_service.initialize()
    app.state.user_service = user_service
    
    logger.info("User management service started")
    
    yield
    
    # Cleanup
    await user_service.shutdown()
    
    logger.info("User management service stopped")


def create_user_management_app(config: Optional[UserManagementServiceConfig] = None) -> FastAPI:
    """
    Create and configure the User Management Service FastAPI application.
    
    Args:
        config: Service configuration
        
    Returns:
        Configured FastAPI application
    """
    if config is None:
        config = UserManagementServiceConfig()
    
    app = FastAPI(
        title="MusicGen AI User Management Service",
        description="User authentication, authorization, and profile management",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Store config in app state
    app.state.config = config
    
    # Add middleware
    app.add_middleware(TracingMiddleware)
    app.add_middleware(MetricsMiddleware, service_name="user_management")
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Service health check."""
        try:
            user_service = app.state.user_service
            health_status = await user_service.get_health()
            return health_status
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "error": str(e)}
            )
    
    # Authentication endpoints
    @app.post("/auth/register", response_model=UserResponse)
    async def register_user(
        user_data: UserCreate,
        user_service: UserManagementService = Depends(get_user_service)
    ):
        """Register a new user account."""
        with tracer.start_as_current_span("register_user") as span:
            span.set_attribute("email", user_data.email)
            
            try:
                with AUTH_DURATION.labels(operation="register").time():
                    user = await user_service.create_user(user_data)
                
                AUTH_REQUESTS.labels(operation="register", status="success").inc()
                
                return UserResponse(
                    id=user.id,
                    email=user.email,
                    full_name=user.full_name,
                    is_active=user.is_active,
                    subscription_tier=user.subscription_tier,
                    created_at=user.created_at
                )
                
            except ValueError as e:
                AUTH_REQUESTS.labels(operation="register", status="validation_error").inc()
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                raise HTTPException(
                    status_code=400,
                    detail=str(e)
                )
            except Exception as e:
                AUTH_REQUESTS.labels(operation="register", status="error").inc()
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                logger.error(f"User registration failed: {e}")
                raise HTTPException(
                    status_code=500,
                    detail="Registration failed"
                )
    
    @app.post("/auth/login", response_model=TokenResponse)
    async def login_user(
        login_data: UserLogin,
        user_service: UserManagementService = Depends(get_user_service)
    ):
        """Authenticate user and return access token."""
        with tracer.start_as_current_span("login_user") as span:
            span.set_attribute("email", login_data.email)
            
            try:
                with AUTH_DURATION.labels(operation="login").time():
                    token_data = await user_service.authenticate_user(
                        login_data.email,
                        login_data.password
                    )
                
                AUTH_REQUESTS.labels(operation="login", status="success").inc()
                ACTIVE_SESSIONS.inc()
                
                return TokenResponse(**token_data)
                
            except AuthenticationError as e:
                AUTH_REQUESTS.labels(operation="login", status="auth_failed").inc()
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                raise HTTPException(
                    status_code=401,
                    detail="Invalid credentials"
                )
            except Exception as e:
                AUTH_REQUESTS.labels(operation="login", status="error").inc()
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                logger.error(f"User login failed: {e}")
                raise HTTPException(
                    status_code=500,
                    detail="Login failed"
                )
    
    @app.post("/auth/logout")
    async def logout_user(
        current_user = Depends(get_current_active_user),
        user_service: UserManagementService = Depends(get_user_service)
    ):
        """Logout user and invalidate session."""
        try:
            await user_service.logout_user(current_user.id)
            ACTIVE_SESSIONS.dec()
            return {"message": "Logged out successfully"}
        except Exception as e:
            logger.error(f"Logout failed: {e}")
            raise HTTPException(
                status_code=500,
                detail="Logout failed"
            )
    
    @app.post("/auth/refresh", response_model=TokenResponse)
    async def refresh_token(
        refresh_token: str,
        user_service: UserManagementService = Depends(get_user_service)
    ):
        """Refresh access token using refresh token."""
        try:
            with AUTH_DURATION.labels(operation="refresh").time():
                token_data = await user_service.refresh_token(refresh_token)
            
            AUTH_REQUESTS.labels(operation="refresh", status="success").inc()
            
            return TokenResponse(**token_data)
            
        except AuthenticationError as e:
            AUTH_REQUESTS.labels(operation="refresh", status="auth_failed").inc()
            raise HTTPException(
                status_code=401,
                detail="Invalid refresh token"
            )
        except Exception as e:
            AUTH_REQUESTS.labels(operation="refresh", status="error").inc()
            logger.error(f"Token refresh failed: {e}")
            raise HTTPException(
                status_code=500,
                detail="Token refresh failed"
            )
    
    @app.post("/auth/forgot-password")
    async def forgot_password(
        email: str,
        user_service: UserManagementService = Depends(get_user_service)
    ):
        """Initiate password reset process."""
        try:
            await user_service.initiate_password_reset(email)
            return {"message": "Password reset email sent"}
        except Exception as e:
            # Don't reveal if email exists for security
            logger.error(f"Password reset initiation failed: {e}")
            return {"message": "Password reset email sent"}
    
    @app.post("/auth/reset-password")
    async def reset_password(
        reset_data: PasswordReset,
        user_service: UserManagementService = Depends(get_user_service)
    ):
        """Reset password using reset token."""
        try:
            await user_service.reset_password(
                reset_data.token,
                reset_data.new_password
            )
            return {"message": "Password reset successfully"}
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Password reset failed: {e}")
            raise HTTPException(
                status_code=500,
                detail="Password reset failed"
            )
    
    # User management endpoints
    @app.get("/users/me", response_model=UserProfile)
    async def get_current_user_profile(
        current_user = Depends(get_current_active_user)
    ):
        """Get current user's profile."""
        return UserProfile(
            id=current_user.id,
            email=current_user.email,
            full_name=current_user.full_name,
            subscription_tier=current_user.subscription_tier,
            usage_quota=current_user.usage_quota,
            preferences=current_user.preferences,
            is_active=current_user.is_active,
            created_at=current_user.created_at,
            last_login=current_user.last_login
        )
    
    @app.put("/users/me", response_model=UserResponse)
    async def update_current_user(
        user_update: UserUpdate,
        current_user = Depends(get_current_active_user),
        user_service: UserManagementService = Depends(get_user_service)
    ):
        """Update current user's profile."""
        try:
            updated_user = await user_service.update_user(
                current_user.id,
                user_update
            )
            
            return UserResponse(
                id=updated_user.id,
                email=updated_user.email,
                full_name=updated_user.full_name,
                is_active=updated_user.is_active,
                subscription_tier=updated_user.subscription_tier,
                created_at=updated_user.created_at
            )
            
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"User update failed: {e}")
            raise HTTPException(
                status_code=500,
                detail="User update failed"
            )
    
    @app.post("/users/me/change-password")
    async def change_password(
        password_change: PasswordChange,
        current_user = Depends(get_current_active_user),
        user_service: UserManagementService = Depends(get_user_service)
    ):
        """Change user's password."""
        try:
            await user_service.change_password(
                current_user.id,
                password_change.current_password,
                password_change.new_password
            )
            return {"message": "Password changed successfully"}
            
        except AuthenticationError as e:
            raise HTTPException(
                status_code=401,
                detail="Current password is incorrect"
            )
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Password change failed: {e}")
            raise HTTPException(
                status_code=500,
                detail="Password change failed"
            )
    
    @app.delete("/users/me")
    async def delete_current_user(
        current_user = Depends(get_current_active_user),
        user_service: UserManagementService = Depends(get_user_service)
    ):
        """Delete current user's account."""
        try:
            await user_service.delete_user(current_user.id)
            return {"message": "User account deleted successfully"}
        except Exception as e:
            logger.error(f"User deletion failed: {e}")
            raise HTTPException(
                status_code=500,
                detail="User deletion failed"
            )
    
    # Usage tracking endpoints
    @app.get("/users/me/usage")
    async def get_user_usage(
        current_user = Depends(get_current_active_user),
        user_service: UserManagementService = Depends(get_user_service)
    ):
        """Get current user's usage statistics."""
        try:
            usage_stats = await user_service.get_user_usage(current_user.id)
            return usage_stats
        except Exception as e:
            logger.error(f"Failed to get user usage: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve usage statistics"
            )
    
    @app.post("/users/me/usage/track")
    async def track_usage(
        operation: str,
        resource_type: str,
        amount: float,
        current_user = Depends(get_current_active_user),
        user_service: UserManagementService = Depends(get_user_service)
    ):
        """Track user resource usage."""
        try:
            await user_service.track_usage(
                current_user.id,
                operation,
                resource_type,
                amount
            )
            return {"message": "Usage tracked successfully"}
        except Exception as e:
            logger.error(f"Usage tracking failed: {e}")
            raise HTTPException(
                status_code=500,
                detail="Usage tracking failed"
            )
    
    # Admin endpoints (require admin role)
    @app.get("/admin/users", response_model=List[UserResponse])
    async def list_users(
        skip: int = 0,
        limit: int = 100,
        current_user = Depends(get_current_active_user),
        user_service: UserManagementService = Depends(get_user_service)
    ):
        """List all users (admin only)."""
        try:
            # Check admin permissions
            if not await user_service.is_admin(current_user.id):
                raise HTTPException(
                    status_code=403,
                    detail="Admin access required"
                )
            
            users = await user_service.list_users(skip=skip, limit=limit)
            
            return [
                UserResponse(
                    id=user.id,
                    email=user.email,
                    full_name=user.full_name,
                    is_active=user.is_active,
                    subscription_tier=user.subscription_tier,
                    created_at=user.created_at
                )
                for user in users
            ]
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to list users: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve users"
            )
    
    @app.get("/admin/users/{user_id}", response_model=UserProfile)
    async def get_user_by_id(
        user_id: str,
        current_user = Depends(get_current_active_user),
        user_service: UserManagementService = Depends(get_user_service)
    ):
        """Get user by ID (admin only)."""
        try:
            # Check admin permissions
            if not await user_service.is_admin(current_user.id):
                raise HTTPException(
                    status_code=403,
                    detail="Admin access required"
                )
            
            user = await user_service.get_user_by_id(user_id)
            if not user:
                raise HTTPException(
                    status_code=404,
                    detail="User not found"
                )
            
            return UserProfile(
                id=user.id,
                email=user.email,
                full_name=user.full_name,
                subscription_tier=user.subscription_tier,
                usage_quota=user.usage_quota,
                preferences=user.preferences,
                is_active=user.is_active,
                created_at=user.created_at,
                last_login=user.last_login
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get user: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve user"
            )
    
    @app.put("/admin/users/{user_id}/activate")
    async def activate_user(
        user_id: str,
        current_user = Depends(get_current_active_user),
        user_service: UserManagementService = Depends(get_user_service)
    ):
        """Activate user account (admin only)."""
        try:
            # Check admin permissions
            if not await user_service.is_admin(current_user.id):
                raise HTTPException(
                    status_code=403,
                    detail="Admin access required"
                )
            
            await user_service.activate_user(user_id)
            return {"message": "User activated successfully"}
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to activate user: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to activate user"
            )
    
    @app.put("/admin/users/{user_id}/deactivate")
    async def deactivate_user(
        user_id: str,
        current_user = Depends(get_current_active_user),
        user_service: UserManagementService = Depends(get_user_service)
    ):
        """Deactivate user account (admin only)."""
        try:
            # Check admin permissions
            if not await user_service.is_admin(current_user.id):
                raise HTTPException(
                    status_code=403,
                    detail="Admin access required"
                )
            
            await user_service.deactivate_user(user_id)
            return {"message": "User deactivated successfully"}
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to deactivate user: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to deactivate user"
            )
    
    # Service statistics
    @app.get("/admin/stats", response_model=UserStats)
    async def get_service_stats(
        current_user = Depends(get_current_active_user),
        user_service: UserManagementService = Depends(get_user_service)
    ):
        """Get user service statistics (admin only)."""
        try:
            # Check admin permissions
            if not await user_service.is_admin(current_user.id):
                raise HTTPException(
                    status_code=403,
                    detail="Admin access required"
                )
            
            stats = await user_service.get_service_stats()
            return stats
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get service stats: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve service statistics"
            )
    
    return app


# Factory for creating the app
def create_app() -> FastAPI:
    """Factory function for creating the app."""
    return create_user_management_app()


if __name__ == "__main__":
    import uvicorn
    
    app = create_user_management_app()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8005,
        log_level="info"
    )