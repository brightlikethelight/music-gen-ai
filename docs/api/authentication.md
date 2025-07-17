# Authentication Guide

The Music Gen AI API uses secure authentication to protect your account and ensure proper usage tracking. This guide covers all authentication methods and best practices.

## Overview

The API supports two primary authentication methods:

1. **JWT Tokens** - For web applications and user sessions
2. **API Keys** - For server-to-server communication and programmatic access

## Quick Start

### Using API Keys (Recommended for most use cases)

```bash
# Include your API key in the X-API-Key header
curl -H "X-API-Key: sk_live_your_api_key_here" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Upbeat jazz with saxophone", "duration": 30}' \
     https://api.musicgen.ai/v1/generate
```

### Using JWT Tokens

```bash
# First, obtain a JWT token
curl -X POST https://api.musicgen.ai/v1/auth/login \
     -H "Content-Type: application/json" \
     -d '{"email": "user@example.com", "password": "your_password"}'

# Use the token in subsequent requests
curl -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Classical piano piece", "duration": 60}' \
     https://api.musicgen.ai/v1/generate
```

## Authentication Methods

### 1. API Key Authentication

API keys are the simplest and most secure method for programmatic access.

#### Creating API Keys

1. **Log in to your dashboard**: [https://musicgen.ai/dashboard](https://musicgen.ai/dashboard)
2. **Navigate to API Keys**: Click on "API Keys" in the sidebar
3. **Create New Key**: Click "Create API Key"
4. **Configure permissions**: Select the permissions you need
5. **Set expiration**: Choose an expiration date (optional)
6. **Copy your key**: Save it securely - you won't see it again!

#### API Key Format

```
sk_live_1234567890abcdef1234567890abcdef    # Production key
sk_test_1234567890abcdef1234567890abcdef    # Test/development key
```

#### Using API Keys

Include your API key in the `X-API-Key` header:

```http
GET /v1/models HTTP/1.1
Host: api.musicgen.ai
X-API-Key: sk_live_your_api_key_here
Content-Type: application/json
```

#### API Key Permissions

| Permission | Description | Endpoints |
|------------|-------------|-----------|
| `generate` | Create music generations | `POST /generate`, `GET /generate/{id}` |
| `models:read` | View available models | `GET /models`, `GET /models/{id}` |
| `user:read` | View profile information | `GET /user/profile` |
| `user:write` | Modify profile settings | `PUT /user/profile` |
| `api-keys:manage` | Manage API keys | `GET/POST/DELETE /user/api-keys` |

#### API Key Best Practices

✅ **Do:**
- Store API keys securely (environment variables, secret managers)
- Use different keys for development and production
- Set appropriate permissions for each key
- Rotate keys regularly (every 90 days recommended)
- Set expiration dates for temporary access

❌ **Don't:**
- Commit API keys to version control
- Share API keys in plain text
- Use production keys in development
- Grant unnecessary permissions
- Use keys beyond their intended scope

#### Example: Secure API Key Storage

```python
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment
api_key = os.getenv('MUSICGEN_API_KEY')
if not api_key:
    raise ValueError("MUSICGEN_API_KEY environment variable not set")

# Use in requests
headers = {
    'X-API-Key': api_key,
    'Content-Type': 'application/json'
}
```

```javascript
// Node.js example
const apiKey = process.env.MUSICGEN_API_KEY;
if (!apiKey) {
    throw new Error('MUSICGEN_API_KEY environment variable not set');
}

const headers = {
    'X-API-Key': apiKey,
    'Content-Type': 'application/json'
};
```

### 2. JWT Token Authentication

JWT tokens are ideal for web applications where you need user-specific access.

#### Obtaining JWT Tokens

**Login with Email/Password:**
```http
POST /v1/auth/login HTTP/1.1
Host: api.musicgen.ai
Content-Type: application/json

{
    "email": "user@example.com",
    "password": "secure_password123"
}
```

**Login with API Key:**
```http
POST /v1/auth/login HTTP/1.1
Host: api.musicgen.ai
Content-Type: application/json

{
    "api_key": "sk_live_your_api_key_here"
}
```

**Response:**
```json
{
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "bearer",
    "expires_in": 3600,
    "refresh_token": "refresh_1234567890abcdef",
    "user": {
        "id": "user_1234567890",
        "email": "user@example.com",
        "tier": "pro",
        "usage": {
            "generations_remaining": 450,
            "reset_date": "2024-01-16T00:00:00Z"
        }
    }
}
```

#### Using JWT Tokens

Include the token in the `Authorization` header:

```http
GET /v1/user/profile HTTP/1.1
Host: api.musicgen.ai
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: application/json
```

#### Token Refresh

Access tokens expire after 1 hour. Use refresh tokens to obtain new access tokens:

```http
POST /v1/auth/refresh HTTP/1.1
Host: api.musicgen.ai
Content-Type: application/json

{
    "refresh_token": "refresh_1234567890abcdef"
}
```

#### JWT Implementation Example

```javascript
class MusicGenAuth {
    constructor(apiUrl = 'https://api.musicgen.ai/v1') {
        this.apiUrl = apiUrl;
        this.accessToken = localStorage.getItem('musicgen_access_token');
        this.refreshToken = localStorage.getItem('musicgen_refresh_token');
    }

    async login(email, password) {
        const response = await fetch(`${this.apiUrl}/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password })
        });

        if (response.ok) {
            const data = await response.json();
            this.accessToken = data.access_token;
            this.refreshToken = data.refresh_token;
            
            localStorage.setItem('musicgen_access_token', this.accessToken);
            localStorage.setItem('musicgen_refresh_token', this.refreshToken);
            
            return data;
        } else {
            throw new Error('Login failed');
        }
    }

    async makeAuthenticatedRequest(url, options = {}) {
        // Try request with current token
        let response = await this._makeRequest(url, options);
        
        // If unauthorized, try refreshing token
        if (response.status === 401) {
            await this.refreshAccessToken();
            response = await this._makeRequest(url, options);
        }
        
        return response;
    }

    async _makeRequest(url, options) {
        return fetch(url, {
            ...options,
            headers: {
                ...options.headers,
                'Authorization': `Bearer ${this.accessToken}`,
                'Content-Type': 'application/json'
            }
        });
    }

    async refreshAccessToken() {
        const response = await fetch(`${this.apiUrl}/auth/refresh`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ refresh_token: this.refreshToken })
        });

        if (response.ok) {
            const data = await response.json();
            this.accessToken = data.access_token;
            localStorage.setItem('musicgen_access_token', this.accessToken);
        } else {
            // Refresh failed, need to login again
            this.logout();
            throw new Error('Session expired');
        }
    }

    logout() {
        this.accessToken = null;
        this.refreshToken = null;
        localStorage.removeItem('musicgen_access_token');
        localStorage.removeItem('musicgen_refresh_token');
    }
}
```

## Security Best Practices

### API Key Security

1. **Environment Variables**: Store keys in environment variables
```bash
export MUSICGEN_API_KEY="sk_live_your_key_here"
```

2. **Secret Management**: Use dedicated secret management services
   - AWS Secrets Manager
   - Azure Key Vault
   - Google Secret Manager
   - HashiCorp Vault

3. **Key Rotation**: Implement automatic key rotation
```python
import os
from datetime import datetime, timedelta

def should_rotate_key(created_date, rotation_days=90):
    return datetime.now() - created_date > timedelta(days=rotation_days)

# Check and rotate keys periodically
if should_rotate_key(key_created_date):
    new_key = create_new_api_key()
    update_environment_variable('MUSICGEN_API_KEY', new_key)
    schedule_old_key_deletion(old_key, delay_days=7)
```

### JWT Token Security

1. **Secure Storage**: Store tokens securely on the client
```javascript
// Use secure, httpOnly cookies for web apps
document.cookie = `token=${accessToken}; Secure; HttpOnly; SameSite=Strict`;

// Or use secure storage in mobile apps
// iOS: Keychain Services
// Android: Android Keystore
```

2. **Token Validation**: Always validate tokens on the server
```python
import jwt
from datetime import datetime

def validate_token(token, secret_key):
    try:
        payload = jwt.decode(token, secret_key, algorithms=['HS256'])
        
        # Check expiration
        if datetime.fromtimestamp(payload['exp']) < datetime.now():
            raise jwt.ExpiredSignatureError()
            
        return payload
    except jwt.InvalidTokenError:
        return None
```

3. **HTTPS Only**: Always use HTTPS in production
```nginx
# Nginx configuration
server {
    listen 443 ssl;
    server_name api.musicgen.ai;
    
    # Redirect HTTP to HTTPS
    if ($scheme != "https") {
        return 301 https://$server_name$request_uri;
    }
}
```

## Error Handling

### Authentication Errors

| Status Code | Error Code | Description | Solution |
|-------------|------------|-------------|----------|
| 401 | `AUTHENTICATION_REQUIRED` | No authentication provided | Include API key or JWT token |
| 401 | `INVALID_TOKEN` | Token is invalid or malformed | Check token format and validity |
| 401 | `TOKEN_EXPIRED` | JWT token has expired | Refresh token or login again |
| 403 | `INSUFFICIENT_PERMISSIONS` | API key lacks required permissions | Update key permissions |
| 429 | `RATE_LIMIT_EXCEEDED` | Too many requests | Wait and retry with backoff |

### Error Response Format

```json
{
    "error": {
        "code": "INVALID_TOKEN",
        "message": "The provided authentication token is invalid or expired",
        "request_id": "req_1234567890abcdef",
        "details": "Token signature verification failed"
    }
}
```

### Handling Authentication Errors

```python
import requests
import time

def make_authenticated_request(url, headers, max_retries=3):
    for attempt in range(max_retries):
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            error = response.json().get('error', {})
            if error.get('code') == 'TOKEN_EXPIRED':
                # Refresh token and retry
                refresh_access_token()
                headers['Authorization'] = f'Bearer {new_access_token}'
                continue
            else:
                raise AuthenticationError(error.get('message'))
        elif response.status_code == 429:
            # Rate limited - exponential backoff
            wait_time = 2 ** attempt
            time.sleep(wait_time)
            continue
        else:
            response.raise_for_status()
    
    raise Exception(f"Max retries ({max_retries}) exceeded")
```

## Testing Authentication

### Test Environment

Use test API keys for development:

```bash
# Development environment
export MUSICGEN_API_KEY="sk_test_your_test_key_here"
export MUSICGEN_API_URL="https://api-staging.musicgen.ai/v1"
```

### Authentication Testing Script

```python
#!/usr/bin/env python3
"""
Authentication testing script for Music Gen AI API
"""
import os
import requests
import json

def test_api_key_auth():
    """Test API key authentication"""
    api_key = os.getenv('MUSICGEN_API_KEY')
    if not api_key:
        print("❌ MUSICGEN_API_KEY not set")
        return False
    
    headers = {
        'X-API-Key': api_key,
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.get(
            'https://api.musicgen.ai/v1/user/profile',
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ API key authentication successful")
            user = response.json()
            print(f"   User: {user.get('email')}")
            print(f"   Tier: {user.get('tier')}")
            return True
        else:
            print(f"❌ API key authentication failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

def test_jwt_auth():
    """Test JWT authentication"""
    email = os.getenv('MUSICGEN_EMAIL')
    password = os.getenv('MUSICGEN_PASSWORD')
    
    if not email or not password:
        print("❌ MUSICGEN_EMAIL or MUSICGEN_PASSWORD not set")
        return False
    
    # Login
    try:
        response = requests.post(
            'https://api.musicgen.ai/v1/auth/login',
            json={'email': email, 'password': password},
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ JWT login successful")
            auth_data = response.json()
            access_token = auth_data['access_token']
            
            # Test authenticated request
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }
            
            profile_response = requests.get(
                'https://api.musicgen.ai/v1/user/profile',
                headers=headers,
                timeout=10
            )
            
            if profile_response.status_code == 200:
                print("✅ JWT authenticated request successful")
                return True
            else:
                print(f"❌ JWT authenticated request failed: {profile_response.status_code}")
                return False
                
        else:
            print(f"❌ JWT login failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

if __name__ == '__main__':
    print("🔐 Testing Music Gen AI Authentication")
    print("=" * 40)
    
    api_key_success = test_api_key_auth()
    print()
    jwt_success = test_jwt_auth()
    
    print("\n" + "=" * 40)
    if api_key_success and jwt_success:
        print("✅ All authentication tests passed!")
    else:
        print("❌ Some authentication tests failed")
        exit(1)
```

## Rate Limiting & Authentication

Authentication method affects your rate limits:

| Method | Rate Limit | Burst Limit | Notes |
|--------|------------|-------------|-------|
| API Key | Based on user tier | 2x normal limit | Recommended for production |
| JWT Token | Based on user tier | 1.5x normal limit | Good for web apps |
| No Auth | 10 req/hour | 20 requests | Very limited |

## Troubleshooting

### Common Issues

**1. "Invalid API Key" Error**
```bash
# Check key format
echo $MUSICGEN_API_KEY | grep -E '^sk_(live|test)_[a-zA-Z0-9]{32}$'

# Verify key permissions in dashboard
curl -H "X-API-Key: $MUSICGEN_API_KEY" \
     https://api.musicgen.ai/v1/user/api-keys
```

**2. "Token Expired" Error**
```javascript
// Implement automatic token refresh
if (error.code === 'TOKEN_EXPIRED') {
    await auth.refreshAccessToken();
    // Retry original request
}
```

**3. "Rate Limit Exceeded" Error**
```python
# Implement exponential backoff
import time
import random

def exponential_backoff(attempt, base_delay=1, max_delay=60):
    delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
    time.sleep(delay)
```

### Debug Mode

Enable debug logging to troubleshoot authentication issues:

```python
import logging
import requests

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
requests_log = logging.getLogger("requests.packages.urllib3")
requests_log.setLevel(logging.DEBUG)
requests_log.propagate = True

# Your API requests will now show detailed debug info
```

## Support

If you encounter authentication issues:

1. **Check our status page**: [https://status.musicgen.ai](https://status.musicgen.ai)
2. **Review the troubleshooting section** above
3. **Contact support**: [support@musicgen.ai](mailto:support@musicgen.ai)
4. **Join our Discord**: [https://discord.gg/musicgen](https://discord.gg/musicgen)

Include the following information when contacting support:
- Your account email
- API key prefix (first 8 characters only)
- Error message and request ID
- Timestamp of the issue
- Steps to reproduce the problem