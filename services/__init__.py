"""
MusicGen AI Microservices Package

This package contains the microservices implementation for the MusicGen AI system.
Each service is designed to be independently deployable and scalable.
"""

__version__ = "1.0.0"
__author__ = "MusicGen AI Team"

# Service registry for service discovery
SERVICES = {
    "gateway": {
        "name": "API Gateway",
        "port": 8080,
        "health_endpoint": "/health",
        "description": "Central API gateway with routing and rate limiting"
    },
    "generation": {
        "name": "Generation Service", 
        "port": 8001,
        "health_endpoint": "/health",
        "description": "Core music generation orchestration"
    },
    "model": {
        "name": "Model Service",
        "port": 8002, 
        "health_endpoint": "/health",
        "description": "Model inference and lifecycle management"
    },
    "processing": {
        "name": "Audio Processing Service",
        "port": 8003,
        "health_endpoint": "/health", 
        "description": "Audio manipulation and effects processing"
    },
    "storage": {
        "name": "Storage Service",
        "port": 8004,
        "health_endpoint": "/health",
        "description": "Distributed file storage and CDN management"
    },
    "user_management": {
        "name": "User Management Service",
        "port": 8005,
        "health_endpoint": "/health",
        "description": "Authentication, authorization, and user profiles"
    },
    "analytics": {
        "name": "Analytics Service", 
        "port": 8006,
        "health_endpoint": "/health",
        "description": "Usage analytics and business intelligence"
    },
    "monitoring": {
        "name": "Monitoring Service",
        "port": 8007, 
        "health_endpoint": "/health",
        "description": "System health monitoring and observability"
    }
}

# Service communication patterns
COMMUNICATION_PATTERNS = {
    "synchronous": {
        "rest": ["gateway", "user_management", "storage"],
        "grpc": ["generation", "model", "processing"]
    },
    "asynchronous": {
        "queue": ["generation", "processing", "analytics"],
        "events": ["user_management", "analytics", "monitoring"]
    }
}

# Database assignments per service
DATABASE_STRATEGY = {
    "generation": "postgresql",
    "model": "redis", 
    "processing": "redis",
    "storage": "postgresql",
    "user_management": "postgresql", 
    "analytics": "timescaledb",
    "monitoring": "prometheus"
}