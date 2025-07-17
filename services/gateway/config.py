"""
API Gateway Configuration

Configuration management for the API Gateway service including routing rules,
rate limiting, authentication settings, and service discovery.
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""
    enabled: bool = True
    default_rpm: int = Field(default=100, ge=1, le=10000)  # requests per minute
    default_rph: int = Field(default=1000, ge=1, le=100000)  # requests per hour
    burst_size: int = Field(default=10, ge=1, le=1000)
    window_size: int = Field(default=60, ge=1, le=3600)  # seconds
    
    # Per-endpoint rate limits
    endpoint_limits: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    
    # Per-user rate limits
    user_limits: Dict[str, Dict[str, int]] = Field(default_factory=dict)


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration."""
    enabled: bool = True
    failure_threshold: int = Field(default=5, ge=1, le=100)
    timeout_duration: int = Field(default=60, ge=1, le=3600)  # seconds
    recovery_timeout: int = Field(default=30, ge=1, le=1800)  # seconds
    
    # Per-service circuit breaker settings
    service_configs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class AuthConfig(BaseModel):
    """Authentication configuration."""
    enabled: bool = True
    jwt_secret: str = Field(default="dev_secret_change_in_production")
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = Field(default=3600, ge=300, le=86400)  # seconds
    
    # Authentication exemptions
    public_endpoints: List[str] = Field(default_factory=lambda: [
        "/health",
        "/metrics", 
        "/api/v1/auth/login",
        "/api/v1/auth/register",
        "/docs",
        "/openapi.json"
    ])
    
    # Authentication providers
    providers: Dict[str, Dict[str, str]] = Field(default_factory=dict)


class ServiceConfig(BaseModel):
    """Individual service configuration."""
    name: str
    url: str
    health_endpoint: str = "/health"
    timeout: int = Field(default=30, ge=1, le=300)  # seconds
    retries: int = Field(default=3, ge=0, le=10)
    circuit_breaker: bool = True
    
    # Load balancing
    weight: int = Field(default=100, ge=1, le=1000)
    max_connections: int = Field(default=100, ge=1, le=10000)
    
    # Health checking
    health_check_interval: int = Field(default=30, ge=5, le=300)  # seconds
    health_check_timeout: int = Field(default=5, ge=1, le=30)  # seconds


class RoutingRule(BaseModel):
    """Request routing rule."""
    path_pattern: str
    service: str
    method: Optional[str] = None  # None means all methods
    strip_prefix: bool = True
    add_prefix: str = ""
    
    # Request transformation
    headers_to_add: Dict[str, str] = Field(default_factory=dict)
    headers_to_remove: List[str] = Field(default_factory=list)
    
    # Route-specific settings
    timeout: Optional[int] = None
    rate_limit: Optional[Dict[str, int]] = None
    auth_required: bool = True


class GatewayConfig(BaseModel):
    """Complete API Gateway configuration."""
    
    # Basic server settings
    host: str = "0.0.0.0"
    port: int = Field(default=8080, ge=1, le=65535)
    workers: int = Field(default=1, ge=1, le=32)
    
    # Security settings
    allowed_hosts: List[str] = Field(default_factory=lambda: ["*"])
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    cors_methods: List[str] = Field(default_factory=lambda: [
        "GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"
    ])
    cors_headers: List[str] = Field(default_factory=lambda: ["*"])
    
    # Request handling
    request_timeout: int = Field(default=60, ge=1, le=600)  # seconds
    max_request_size: int = Field(default=100 * 1024 * 1024, ge=1024)  # bytes
    max_connections: int = Field(default=1000, ge=1, le=100000)
    max_keepalive_connections: int = Field(default=100, ge=1, le=10000)
    
    # Service discovery
    service_discovery_type: str = Field(default="static", regex="^(static|consul|kubernetes)$")
    service_discovery_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Services configuration
    services: Dict[str, ServiceConfig] = Field(default_factory=dict)
    
    # Routing configuration
    routing_rules: List[RoutingRule] = Field(default_factory=list)
    
    # Feature configurations
    rate_limiting: RateLimitConfig = Field(default_factory=RateLimitConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    
    # Monitoring settings
    metrics_enabled: bool = True
    tracing_enabled: bool = True
    logging_level: str = Field(default="INFO", regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    
    # Redis configuration for rate limiting and caching
    redis_url: str = Field(default="redis://localhost:6379/0")
    redis_pool_size: int = Field(default=10, ge=1, le=100)
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        extra = "forbid"
        use_enum_values = True
    
    @validator('services')
    def validate_services(cls, v):
        """Validate services configuration."""
        if not v:
            # Set default services if none provided
            v = cls._get_default_services()
        return v
    
    @validator('routing_rules')
    def validate_routing_rules(cls, v, values):
        """Validate routing rules."""
        if not v:
            # Set default routing rules if none provided
            v = cls._get_default_routing_rules()
        
        # Validate service references
        services = values.get('services', {})
        for rule in v:
            if rule.service not in services:
                raise ValueError(f"Routing rule references unknown service: {rule.service}")
        
        return v
    
    @classmethod
    def _get_default_services(cls) -> Dict[str, ServiceConfig]:
        """Get default services configuration."""
        return {
            "generation": ServiceConfig(
                name="generation",
                url=os.getenv("GENERATION_SERVICE_URL", "http://generation-service:8001"),
                timeout=120,  # Generation can take longer
                max_connections=50
            ),
            "model": ServiceConfig(
                name="model", 
                url=os.getenv("MODEL_SERVICE_URL", "http://model-service:8002"),
                timeout=60,
                max_connections=100
            ),
            "processing": ServiceConfig(
                name="processing",
                url=os.getenv("PROCESSING_SERVICE_URL", "http://processing-service:8003"),
                timeout=90,
                max_connections=50
            ),
            "storage": ServiceConfig(
                name="storage",
                url=os.getenv("STORAGE_SERVICE_URL", "http://storage-service:8004"),
                timeout=30,
                max_connections=200
            ),
            "user_management": ServiceConfig(
                name="user_management",
                url=os.getenv("USER_MGMT_SERVICE_URL", "http://user-management-service:8005"),
                timeout=15,
                max_connections=100
            ),
            "analytics": ServiceConfig(
                name="analytics",
                url=os.getenv("ANALYTICS_SERVICE_URL", "http://analytics-service:8006"),
                timeout=30,
                max_connections=50
            )
        }
    
    @classmethod 
    def _get_default_routing_rules(cls) -> List[RoutingRule]:
        """Get default routing rules."""
        return [
            # Authentication endpoints
            RoutingRule(
                path_pattern="/api/v1/auth/*",
                service="user_management",
                auth_required=False
            ),
            
            # User management endpoints
            RoutingRule(
                path_pattern="/api/v1/users/*",
                service="user_management"
            ),
            
            # Generation endpoints
            RoutingRule(
                path_pattern="/api/v1/generate*",
                service="generation",
                timeout=180,  # Longer timeout for generation
                rate_limit={"rpm": 20, "rph": 100}  # Stricter limits
            ),
            
            RoutingRule(
                path_pattern="/api/v1/stream*",
                service="generation",
                timeout=300,  # Very long timeout for streaming
                rate_limit={"rpm": 10, "rph": 50}
            ),
            
            # Model management endpoints
            RoutingRule(
                path_pattern="/api/v1/models*",
                service="model"
            ),
            
            # Audio processing endpoints
            RoutingRule(
                path_pattern="/api/v1/process*",
                service="processing",
                timeout=120
            ),
            
            # Storage endpoints
            RoutingRule(
                path_pattern="/api/v1/files*",
                service="storage"
            ),
            
            RoutingRule(
                path_pattern="/download/*",
                service="storage",
                auth_required=False  # Downloads can be public
            ),
            
            # Analytics endpoints
            RoutingRule(
                path_pattern="/api/v1/analytics*",
                service="analytics"
            ),
            
            # Monitoring endpoints
            RoutingRule(
                path_pattern="/api/v1/monitoring*",
                service="generation",  # Route to generation for now
                auth_required=False
            )
        ]
    
    @classmethod
    def from_environment(cls) -> "GatewayConfig":
        """Create configuration from environment variables."""
        config_data = {}
        
        # Basic settings
        if host := os.getenv("GATEWAY_HOST"):
            config_data["host"] = host
        if port := os.getenv("GATEWAY_PORT"):
            config_data["port"] = int(port)
        if workers := os.getenv("GATEWAY_WORKERS"):
            config_data["workers"] = int(workers)
        
        # Security settings
        if allowed_hosts := os.getenv("GATEWAY_ALLOWED_HOSTS"):
            config_data["allowed_hosts"] = allowed_hosts.split(",")
        if cors_origins := os.getenv("GATEWAY_CORS_ORIGINS"):
            config_data["cors_origins"] = cors_origins.split(",")
        
        # Authentication
        auth_config = {}
        if jwt_secret := os.getenv("JWT_SECRET"):
            auth_config["jwt_secret"] = jwt_secret
        if jwt_algorithm := os.getenv("JWT_ALGORITHM"):
            auth_config["jwt_algorithm"] = jwt_algorithm
        if auth_config:
            config_data["auth"] = auth_config
        
        # Rate limiting
        rate_limit_config = {}
        if default_rpm := os.getenv("RATE_LIMIT_RPM"):
            rate_limit_config["default_rpm"] = int(default_rpm)
        if default_rph := os.getenv("RATE_LIMIT_RPH"):
            rate_limit_config["default_rph"] = int(default_rph)
        if rate_limit_config:
            config_data["rate_limiting"] = rate_limit_config
        
        # Redis
        if redis_url := os.getenv("REDIS_URL"):
            config_data["redis_url"] = redis_url
        
        # Service discovery
        if discovery_type := os.getenv("SERVICE_DISCOVERY_TYPE"):
            config_data["service_discovery_type"] = discovery_type
        
        return cls(**config_data)
    
    def get_service_url(self, service_name: str) -> Optional[str]:
        """Get URL for a service."""
        service_config = self.services.get(service_name)
        return service_config.url if service_config else None
    
    def get_routing_rule(self, path: str, method: str = "GET") -> Optional[RoutingRule]:
        """Get routing rule for a path and method."""
        import fnmatch
        
        for rule in self.routing_rules:
            # Check method match
            if rule.method and rule.method.upper() != method.upper():
                continue
            
            # Check path pattern match
            if fnmatch.fnmatch(path, rule.path_pattern):
                return rule
        
        return None
    
    def is_public_endpoint(self, path: str) -> bool:
        """Check if endpoint is public (no auth required)."""
        import fnmatch
        
        # Check auth config public endpoints
        for pattern in self.auth.public_endpoints:
            if fnmatch.fnmatch(path, pattern):
                return True
        
        # Check routing rule auth requirements
        rule = self.get_routing_rule(path)
        if rule and not rule.auth_required:
            return True
        
        return False