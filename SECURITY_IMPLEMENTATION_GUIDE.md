# Security Implementation Guide - Phase 1

## Quick Start: Basic Rate Limiting

This guide shows how to implement the most critical security feature first: rate limiting.

### 1. Create Rate Limiting Middleware

```python
# src/musicgen/api/rest/middleware/rate_limiting.py

import time
from collections import defaultdict
from typing import Dict, Tuple, Optional
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

class InMemoryRateLimiter:
    """Simple in-memory rate limiter for getting started."""
    
    def __init__(self):
        # Storage: ip -> list of timestamps
        self.requests: Dict[str, list] = defaultdict(list)
        self.window_seconds = 60  # 1 minute window
        self.max_requests = 60    # 60 requests per minute
    
    def is_allowed(self, client_ip: str) -> Tuple[bool, Optional[int]]:
        """Check if request is allowed."""
        now = time.time()
        
        # Clean old requests
        self.requests[client_ip] = [
            ts for ts in self.requests[client_ip]
            if now - ts < self.window_seconds
        ]
        
        # Check limit
        if len(self.requests[client_ip]) >= self.max_requests:
            # Calculate retry after
            oldest = min(self.requests[client_ip])
            retry_after = int(self.window_seconds - (now - oldest))
            return False, retry_after
        
        # Add current request
        self.requests[client_ip].append(now)
        return True, None

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Basic rate limiting middleware."""
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.limiter = InMemoryRateLimiter()
        self.limiter.max_requests = requests_per_minute
    
    async def dispatch(self, request: Request, call_next):
        # Skip health checks
        if request.url.path in ["/health", "/docs", "/openapi.json"]:
            return await call_next(request)
        
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Check rate limit
        allowed, retry_after = self.limiter.is_allowed(client_ip)
        
        if not allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "rate_limit_exceeded",
                    "message": "Too many requests. Please try again later.",
                    "retry_after": retry_after
                },
                headers={"Retry-After": str(retry_after)}
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.limiter.max_requests)
        response.headers["X-RateLimit-Window"] = str(self.limiter.window_seconds)
        
        return response
```

### 2. Add to FastAPI App

```python
# src/musicgen/api/rest/app.py

from musicgen.api.rest.middleware.rate_limiting import RateLimitMiddleware

# After creating the app
app = FastAPI(...)

# Add rate limiting middleware (before CORS)
app.add_middleware(
    RateLimitMiddleware,
    requests_per_minute=30  # Adjust based on your needs
)

# Existing CORS middleware
app.add_middleware(CORSMiddleware, ...)
```

### 3. Environment Configuration

```bash
# .env
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_ENABLED=true
```

### 4. Enhanced Version with Proxy Support

```python
# For production environments behind proxies (nginx, cloudflare)

def get_client_ip(request: Request) -> str:
    """Extract real client IP from proxy headers."""
    # Check proxy headers in order of preference
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take first IP from comma-separated list
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fallback to direct connection
    return request.client.host if request.client else "unknown"
```

### 5. Testing Rate Limiting

```python
# tests/test_rate_limiting.py

import pytest
from httpx import AsyncClient
from fastapi import FastAPI
from musicgen.api.rest.middleware.rate_limiting import RateLimitMiddleware

@pytest.mark.asyncio
async def test_rate_limit_enforcement():
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, requests_per_minute=5)
    
    @app.get("/test")
    async def test_endpoint():
        return {"status": "ok"}
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Make 5 requests (should succeed)
        for i in range(5):
            response = await client.get("/test")
            assert response.status_code == 200
        
        # 6th request should fail
        response = await client.get("/test")
        assert response.status_code == 429
        assert "retry_after" in response.json()
```

## Phase 2 Preview: Adding JWT Authentication

### Basic JWT Setup (Next Sprint)

```python
# src/musicgen/api/rest/middleware/auth.py

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# Configuration
SECRET_KEY = "your-secret-key-from-env"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    username: str
    email: str
    disabled: Optional[bool] = None

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    # Get user from database
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user
```

### Protected Endpoints

```python
# Protect generation endpoint
@app.post("/generate", dependencies=[Depends(get_current_user)])
async def generate_music(request: GenerationRequest):
    # Existing generation logic
    ...

# Public health check (no auth)
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

## Migration Checklist

- [ ] Backup current deployment
- [ ] Test rate limiting in staging
- [ ] Configure rate limits based on usage patterns
- [ ] Monitor for false positives
- [ ] Adjust limits based on feedback
- [ ] Document rate limits in API docs
- [ ] Add rate limit info to error responses

## Monitoring and Alerts

```python
# Add metrics for rate limiting
from prometheus_client import Counter, Histogram

rate_limit_exceeded = Counter(
    'rate_limit_exceeded_total',
    'Total number of rate limit exceeded responses'
)

rate_limit_checks = Histogram(
    'rate_limit_check_duration_seconds',
    'Time spent checking rate limits'
)
```

## Common Issues and Solutions

### Issue: Rate limits too restrictive
**Solution:** Start with generous limits and tighten based on monitoring

### Issue: Memory usage growing
**Solution:** Add periodic cleanup of old request records

### Issue: Distributed deployment
**Solution:** Switch to Redis-based rate limiting (Phase 3)

## Security Headers (Bonus)

Add these security headers for additional protection:

```python
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.sessions import SessionMiddleware

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response
```

## Next Steps

1. Implement basic rate limiting (this guide)
2. Test thoroughly with your expected load
3. Monitor and adjust limits
4. Plan JWT auth implementation
5. Consider Redis for scaling

Remember: Start simple, iterate based on real usage!