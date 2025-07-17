# Rate Limiting Guide

The Music Gen AI API implements sophisticated rate limiting to ensure fair usage and maintain service quality for all users. This guide explains how rate limiting works, tier-based limits, and best practices for handling rate limits in your applications.

## Overview

Rate limiting controls the number of API requests you can make within a specific time window. This ensures:

- **Fair Usage**: All users get equitable access to resources
- **Service Stability**: Prevents system overload and maintains performance
- **Quality Assurance**: Ensures consistent response times for all users
- **Resource Protection**: Protects against abuse and excessive usage

## Rate Limit Tiers

### Free Tier
- **API Requests**: 100 requests per hour
- **Generation Requests**: 10 generations per hour
- **Burst Limit**: 20 requests in 5 minutes
- **Maximum Duration**: 30 seconds per generation
- **Concurrent Generations**: 1
- **Model Access**: musicgen-small only

### Pro Tier
- **API Requests**: 1,000 requests per hour
- **Generation Requests**: 100 generations per hour
- **Burst Limit**: 200 requests in 5 minutes
- **Maximum Duration**: 5 minutes per generation
- **Concurrent Generations**: 3
- **Model Access**: All models except enterprise-exclusive

### Enterprise Tier
- **API Requests**: Unlimited*
- **Generation Requests**: Unlimited*
- **Burst Limit**: 1,000 requests in 5 minutes
- **Maximum Duration**: 10 minutes per generation
- **Concurrent Generations**: 10
- **Model Access**: All models including enterprise-exclusive

*Subject to fair usage policy and system capacity

## Rate Limit Headers

Every API response includes rate limit information in the headers:

```http
HTTP/1.1 200 OK
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 995
X-RateLimit-Reset: 1642291200
X-RateLimit-Used: 5
X-RateLimit-Window: 3600
X-RateLimit-Type: hourly
Retry-After: 3600
```

### Header Descriptions

| Header | Description | Example |
|--------|-------------|---------|
| `X-RateLimit-Limit` | Maximum requests allowed in the window | `1000` |
| `X-RateLimit-Remaining` | Requests remaining in current window | `995` |
| `X-RateLimit-Reset` | Unix timestamp when window resets | `1642291200` |
| `X-RateLimit-Used` | Requests used in current window | `5` |
| `X-RateLimit-Window` | Window duration in seconds | `3600` |
| `X-RateLimit-Type` | Type of rate limit applied | `hourly` |
| `Retry-After` | Seconds to wait before next request | `3600` |

## Rate Limit Types

### 1. Global API Rate Limits

Applied to all API endpoints except `/metrics` and `/health`:

```http
GET /v1/models HTTP/1.1
Host: api.musicgen.ai
X-API-Key: sk_live_your_key_here

HTTP/1.1 200 OK
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Type: api_global
```

### 2. Generation-Specific Rate Limits

Special limits for music generation endpoints:

```http
POST /v1/generate HTTP/1.1
Host: api.musicgen.ai
X-API-Key: sk_live_your_key_here

HTTP/1.1 202 Accepted
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 99
X-RateLimit-Type: generation
X-Generation-Quota-Remaining: 450
X-Generation-Quota-Reset: 1642291200
```

### 3. Burst Protection

Short-term limits to prevent sudden spikes:

```http
# After 20 requests in 1 minute
HTTP/1.1 429 Too Many Requests
X-RateLimit-Limit: 20
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1642287660
X-RateLimit-Type: burst
Retry-After: 60
```

### 4. Concurrent Request Limits

Limits on simultaneous active requests:

```http
# When exceeding concurrent generation limit
HTTP/1.1 429 Too Many Requests
X-RateLimit-Limit: 3
X-RateLimit-Remaining: 0
X-RateLimit-Type: concurrent_generations
Retry-After: 30
```

## Rate Limit Responses

### 429 Too Many Requests

When you exceed rate limits, the API returns a 429 status code:

```json
{
    "error": {
        "code": "RATE_LIMIT_EXCEEDED",
        "message": "Rate limit exceeded. Please wait before making another request.",
        "details": "You have exceeded the rate limit of 100 requests per hour",
        "request_id": "req_1234567890abcdef",
        "retry_after": 3600,
        "limit_type": "hourly",
        "quota_used": 100,
        "quota_limit": 100,
        "reset_time": "2024-01-15T11:00:00Z"
    }
}
```

### Rate Limit Error Types

| Error Code | Description | Typical Cause |
|------------|-------------|---------------|
| `RATE_LIMIT_EXCEEDED` | Standard rate limit hit | Too many requests in time window |
| `BURST_LIMIT_EXCEEDED` | Burst protection triggered | Too many requests too quickly |
| `CONCURRENT_LIMIT_EXCEEDED` | Too many simultaneous requests | Multiple parallel generations |
| `QUOTA_EXHAUSTED` | Monthly/daily quota used up | High usage over longer period |
| `IP_RATE_LIMIT_EXCEEDED` | IP-based limiting | Multiple accounts from same IP |

## Best Practices

### 1. Implement Exponential Backoff

```python
import time
import random
import requests

def exponential_backoff_request(url, headers, max_retries=5):
    for attempt in range(max_retries):
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            return response
        elif response.status_code == 429:
            # Get retry delay from header or calculate
            retry_after = int(response.headers.get('Retry-After', 0))
            if retry_after == 0:
                # Exponential backoff with jitter
                delay = min(300, (2 ** attempt) + random.uniform(0, 1))
            else:
                delay = retry_after
            
            print(f"Rate limited. Waiting {delay} seconds...")
            time.sleep(delay)
        else:
            response.raise_for_status()
    
    raise Exception(f"Max retries ({max_retries}) exceeded")

# Usage
response = exponential_backoff_request(
    'https://api.musicgen.ai/v1/models',
    headers={'X-API-Key': 'sk_live_your_key_here'}
)
```

### 2. Monitor Rate Limit Headers

```javascript
class RateLimitTracker {
    constructor() {
        this.limits = {};
    }

    updateFromResponse(response) {
        const headers = response.headers;
        const limitType = headers.get('x-ratelimit-type') || 'default';
        
        this.limits[limitType] = {
            limit: parseInt(headers.get('x-ratelimit-limit')),
            remaining: parseInt(headers.get('x-ratelimit-remaining')),
            reset: parseInt(headers.get('x-ratelimit-reset')),
            used: parseInt(headers.get('x-ratelimit-used')),
            window: parseInt(headers.get('x-ratelimit-window'))
        };
    }

    shouldMakeRequest(limitType = 'default') {
        const limit = this.limits[limitType];
        if (!limit) return true;

        // Check if we're close to the limit
        const usagePercent = limit.used / limit.limit;
        if (usagePercent > 0.9) {
            console.warn(`Approaching rate limit: ${limit.used}/${limit.limit}`);
        }

        // Check if we have remaining requests
        return limit.remaining > 0;
    }

    getResetTime(limitType = 'default') {
        const limit = this.limits[limitType];
        return limit ? new Date(limit.reset * 1000) : null;
    }

    getTimeUntilReset(limitType = 'default') {
        const resetTime = this.getResetTime(limitType);
        return resetTime ? Math.max(0, resetTime.getTime() - Date.now()) : 0;
    }
}

// Usage
const rateLimiter = new RateLimitTracker();

async function makeRequest(url, options) {
    if (!rateLimiter.shouldMakeRequest()) {
        const waitTime = rateLimiter.getTimeUntilReset();
        console.log(`Waiting ${waitTime}ms for rate limit reset...`);
        await new Promise(resolve => setTimeout(resolve, waitTime));
    }

    const response = await fetch(url, options);
    rateLimiter.updateFromResponse(response);
    return response;
}
```

### 3. Request Queuing

```python
import asyncio
import time
from collections import deque

class RateLimitedClient:
    def __init__(self, requests_per_hour=1000):
        self.requests_per_hour = requests_per_hour
        self.request_queue = deque()
        self.last_request_time = 0
        self.min_interval = 3600 / requests_per_hour  # seconds between requests
        
    async def make_request(self, url, headers):
        # Wait for rate limit window
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            await asyncio.sleep(wait_time)
        
        # Make the request
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                self.last_request_time = time.time()
                return response

# Usage
client = RateLimitedClient(requests_per_hour=1000)
response = await client.make_request(
    'https://api.musicgen.ai/v1/models',
    headers={'X-API-Key': 'sk_live_your_key_here'}
)
```

### 4. Batch Operations

```python
def batch_generate_music(prompts, batch_size=5):
    """Generate music for multiple prompts with rate limiting"""
    results = []
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        batch_results = []
        
        for prompt in batch:
            try:
                response = exponential_backoff_request(
                    'https://api.musicgen.ai/v1/generate',
                    headers={
                        'X-API-Key': os.getenv('MUSICGEN_API_KEY'),
                        'Content-Type': 'application/json'
                    },
                    json={'prompt': prompt, 'duration': 30}
                )
                batch_results.append(response.json())
                
                # Small delay between batch items
                time.sleep(1)
                
            except Exception as e:
                print(f"Failed to generate for prompt '{prompt}': {e}")
                batch_results.append(None)
        
        results.extend(batch_results)
        
        # Longer delay between batches
        if i + batch_size < len(prompts):
            time.sleep(10)
    
    return results
```

## Advanced Rate Limiting

### 1. IP-Based Rate Limiting

The API implements additional IP-based rate limiting to prevent abuse:

- **Free Tier IPs**: 500 requests per hour per IP
- **Authenticated IPs**: 10x user's tier limit per IP
- **Enterprise IPs**: Configurable limits

### 2. Geographic Rate Limiting

Different limits may apply based on geographic location:

```http
# Response may include geographic info
X-RateLimit-Region: us-east-1
X-RateLimit-Limit-Regional: 1000
X-RateLimit-Remaining-Regional: 995
```

### 3. Model-Specific Limits

Some models have additional restrictions:

```json
{
    "model": "musicgen-large",
    "rate_limits": {
        "free_tier": {
            "requests_per_hour": 5,
            "max_duration": 30
        },
        "pro_tier": {
            "requests_per_hour": 50,
            "max_duration": 300
        },
        "enterprise_tier": {
            "requests_per_hour": "unlimited",
            "max_duration": 600
        }
    }
}
```

## Monitoring and Optimization

### 1. Rate Limit Monitoring Dashboard

Track your usage with custom monitoring:

```python
import time
import json
from collections import defaultdict

class RateLimitMonitor:
    def __init__(self):
        self.usage_history = defaultdict(list)
        self.errors = []
        
    def log_request(self, endpoint, status_code, headers):
        timestamp = time.time()
        
        usage_data = {
            'timestamp': timestamp,
            'endpoint': endpoint,
            'status_code': status_code,
            'limit': int(headers.get('x-ratelimit-limit', 0)),
            'remaining': int(headers.get('x-ratelimit-remaining', 0)),
            'used': int(headers.get('x-ratelimit-used', 0))
        }
        
        self.usage_history[endpoint].append(usage_data)
        
        if status_code == 429:
            self.errors.append(usage_data)
    
    def get_usage_stats(self, hours=24):
        cutoff = time.time() - (hours * 3600)
        stats = {}
        
        for endpoint, history in self.usage_history.items():
            recent_requests = [r for r in history if r['timestamp'] > cutoff]
            
            if recent_requests:
                stats[endpoint] = {
                    'total_requests': len(recent_requests),
                    'rate_limit_errors': len([r for r in recent_requests if r['status_code'] == 429]),
                    'average_usage': sum(r['used'] for r in recent_requests) / len(recent_requests),
                    'peak_usage': max(r['used'] for r in recent_requests)
                }
        
        return stats
    
    def export_metrics(self, filename):
        with open(filename, 'w') as f:
            json.dump({
                'usage_history': self.usage_history,
                'errors': self.errors,
                'stats': self.get_usage_stats()
            }, f, indent=2)

# Usage
monitor = RateLimitMonitor()

# Log each request
response = requests.get(url, headers=headers)
monitor.log_request('/v1/models', response.status_code, response.headers)

# Get usage statistics
stats = monitor.get_usage_stats()
print(json.dumps(stats, indent=2))
```

### 2. Optimization Strategies

**Request Caching:**
```python
import requests_cache

# Cache responses to reduce API calls
session = requests_cache.CachedSession(
    cache_name='musicgen_cache',
    expire_after=300  # 5 minutes
)

# Cached requests don't count against rate limits
response = session.get(
    'https://api.musicgen.ai/v1/models',
    headers={'X-API-Key': 'sk_live_your_key_here'}
)
```

**Request Deduplication:**
```python
import hashlib
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_generate_music(prompt, duration, model):
    """Cache music generation requests by parameters"""
    response = requests.post(
        'https://api.musicgen.ai/v1/generate',
        headers={
            'X-API-Key': os.getenv('MUSICGEN_API_KEY'),
            'Content-Type': 'application/json'
        },
        json={
            'prompt': prompt,
            'duration': duration,
            'model': model
        }
    )
    return response.json()

# Identical requests will use cached results
result1 = cached_generate_music("Jazz piano", 30, "musicgen-small")
result2 = cached_generate_music("Jazz piano", 30, "musicgen-small")  # Cached
```

## Troubleshooting

### Common Rate Limit Issues

**1. Unexpected 429 Errors**
```bash
# Check current usage
curl -H "X-API-Key: sk_live_your_key_here" \
     https://api.musicgen.ai/v1/user/profile | jq '.usage'

# Check rate limit headers from last request
curl -I -H "X-API-Key: sk_live_your_key_here" \
     https://api.musicgen.ai/v1/models
```

**2. Inconsistent Rate Limits**
- Check if you're using multiple API keys
- Verify your account tier hasn't changed
- Ensure you're not hitting IP-based limits

**3. Slow Response Times**
- Implement request spacing
- Use appropriate retry delays
- Consider upgrading your tier

### Rate Limit Testing

```python
#!/usr/bin/env python3
"""
Rate limit testing script
"""
import time
import requests
import os

def test_rate_limits():
    api_key = os.getenv('MUSICGEN_API_KEY')
    headers = {'X-API-Key': api_key}
    
    print("Testing rate limits...")
    
    for i in range(20):
        try:
            response = requests.get(
                'https://api.musicgen.ai/v1/models',
                headers=headers
            )
            
            print(f"Request {i+1}: {response.status_code}")
            print(f"  Remaining: {response.headers.get('x-ratelimit-remaining')}")
            print(f"  Reset: {response.headers.get('x-ratelimit-reset')}")
            
            if response.status_code == 429:
                retry_after = int(response.headers.get('retry-after', 60))
                print(f"  Rate limited! Waiting {retry_after} seconds...")
                time.sleep(retry_after)
            else:
                time.sleep(1)  # Small delay between requests
                
        except Exception as e:
            print(f"Request {i+1} failed: {e}")

if __name__ == '__main__':
    test_rate_limits()
```

## Enterprise Rate Limiting

### Custom Limits

Enterprise customers can request custom rate limits:

- **Dedicated Capacity**: Reserved compute resources
- **Custom Windows**: Non-standard time windows (e.g., 15 minutes, 6 hours)
- **Burst Allowances**: Higher temporary limits for specific use cases
- **Geographic Distribution**: Different limits per region
- **Priority Queuing**: Faster processing during high load

### Webhook Rate Limits

Enterprise webhook endpoints have separate limits:

```json
{
    "webhook_limits": {
        "events_per_minute": 1000,
        "payload_size_mb": 10,
        "timeout_seconds": 30,
        "retry_attempts": 3,
        "concurrent_connections": 10
    }
}
```

## Rate Limit APIs

### Current Usage Endpoint

```http
GET /v1/user/rate-limits HTTP/1.1
Host: api.musicgen.ai
X-API-Key: sk_live_your_key_here

HTTP/1.1 200 OK
Content-Type: application/json

{
    "user_tier": "pro",
    "limits": {
        "api_requests": {
            "limit": 1000,
            "used": 245,
            "remaining": 755,
            "reset_time": "2024-01-15T12:00:00Z",
            "window": "hourly"
        },
        "generation_requests": {
            "limit": 100,
            "used": 23,
            "remaining": 77,
            "reset_time": "2024-01-15T12:00:00Z",
            "window": "hourly"
        },
        "concurrent_generations": {
            "limit": 3,
            "active": 1,
            "available": 2
        }
    },
    "quotas": {
        "monthly_generations": {
            "limit": 3000,
            "used": 1234,
            "remaining": 1766,
            "reset_date": "2024-02-01T00:00:00Z"
        }
    }
}
```

### Rate Limit History

```http
GET /v1/user/rate-limits/history?period=24h HTTP/1.1
Host: api.musicgen.ai
X-API-Key: sk_live_your_key_here

HTTP/1.1 200 OK
Content-Type: application/json

{
    "period": "24h",
    "data_points": [
        {
            "timestamp": "2024-01-15T10:00:00Z",
            "api_requests_used": 45,
            "generation_requests_used": 8,
            "rate_limit_errors": 0
        },
        {
            "timestamp": "2024-01-15T11:00:00Z",
            "api_requests_used": 67,
            "generation_requests_used": 12,
            "rate_limit_errors": 2
        }
    ]
}
```

## Support

For rate limiting issues:

1. **Check your current usage**: Use the `/user/rate-limits` endpoint
2. **Review rate limit headers**: Monitor X-RateLimit-* headers
3. **Implement proper error handling**: Use exponential backoff
4. **Consider upgrading**: Higher tiers have more generous limits
5. **Contact support**: [support@musicgen.ai](mailto:support@musicgen.ai) for custom limits

Include the following when contacting support:
- Your account tier and usage patterns
- Specific rate limit errors you're encountering
- Your use case and expected request volumes
- Any time-sensitive requirements