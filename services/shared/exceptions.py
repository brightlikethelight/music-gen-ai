"""
Shared Exception Classes

Common exception classes used across all microservices for consistent
error handling and service communication.
"""

from typing import Optional, Dict, Any


class ServiceError(Exception):
    """Base exception for all service errors."""
    
    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize service error.
        
        Args:
            message: Error message
            service_name: Name of the service that raised the error
            error_code: Specific error code
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.service_name = service_name
        self.error_code = error_code
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "service_name": self.service_name,
            "error_code": self.error_code,
            "details": self.details
        }


class ModelServiceError(ServiceError):
    """Error communicating with or from the model service."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, service_name="model", **kwargs)


class ProcessingServiceError(ServiceError):
    """Error communicating with or from the processing service."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, service_name="processing", **kwargs)


class StorageServiceError(ServiceError):
    """Error communicating with or from the storage service."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, service_name="storage", **kwargs)


class UserManagementServiceError(ServiceError):
    """Error communicating with or from the user management service."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, service_name="user_management", **kwargs)


class AnalyticsServiceError(ServiceError):
    """Error communicating with or from the analytics service."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, service_name="analytics", **kwargs)


class AuthenticationError(ServiceError):
    """Authentication-related errors."""
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(message, error_code="AUTH_FAILED", **kwargs)


class AuthorizationError(ServiceError):
    """Authorization-related errors."""
    
    def __init__(self, message: str = "Access denied", **kwargs):
        super().__init__(message, error_code="ACCESS_DENIED", **kwargs)


class RateLimitError(ServiceError):
    """Rate limiting errors."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, error_code="RATE_LIMIT_EXCEEDED", **kwargs)
        self.retry_after = retry_after
        if retry_after:
            self.details["retry_after"] = retry_after


class ValidationError(ServiceError):
    """Request validation errors."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        validation_errors: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(message, error_code="VALIDATION_FAILED", **kwargs)
        if field:
            self.details["field"] = field
        if validation_errors:
            self.details["validation_errors"] = validation_errors


class ResourceExhaustedError(ServiceError):
    """Resource exhaustion errors (memory, CPU, quota, etc.)."""
    
    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, error_code="RESOURCE_EXHAUSTED", **kwargs)
        if resource_type:
            self.details["resource_type"] = resource_type


class TimeoutError(ServiceError):
    """Request timeout errors."""
    
    def __init__(
        self,
        message: str = "Request timeout",
        timeout_seconds: Optional[float] = None,
        **kwargs
    ):
        super().__init__(message, error_code="TIMEOUT", **kwargs)
        if timeout_seconds:
            self.details["timeout_seconds"] = timeout_seconds


class CircuitBreakerError(ServiceError):
    """Circuit breaker errors when service is unavailable."""
    
    def __init__(
        self,
        message: str = "Service temporarily unavailable",
        **kwargs
    ):
        super().__init__(message, error_code="CIRCUIT_BREAKER_OPEN", **kwargs)


class ConfigurationError(ServiceError):
    """Configuration-related errors."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, error_code="CONFIGURATION_ERROR", **kwargs)
        if config_key:
            self.details["config_key"] = config_key


class DatabaseError(ServiceError):
    """Database operation errors."""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, error_code="DATABASE_ERROR", **kwargs)
        if operation:
            self.details["operation"] = operation


class QueueError(ServiceError):
    """Message queue operation errors."""
    
    def __init__(
        self,
        message: str,
        queue_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, error_code="QUEUE_ERROR", **kwargs)
        if queue_name:
            self.details["queue_name"] = queue_name


class ModelError(ServiceError):
    """ML model-related errors."""
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, error_code="MODEL_ERROR", **kwargs)
        if model_name:
            self.details["model_name"] = model_name


class AudioProcessingError(ServiceError):
    """Audio processing-related errors."""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, error_code="AUDIO_PROCESSING_ERROR", **kwargs)
        if operation:
            self.details["operation"] = operation


# Exception mapping for HTTP status codes
HTTP_STATUS_MAP = {
    AuthenticationError: 401,
    AuthorizationError: 403,
    RateLimitError: 429,
    ValidationError: 400,
    ResourceExhaustedError: 507,
    TimeoutError: 504,
    CircuitBreakerError: 503,
    ConfigurationError: 500,
    DatabaseError: 500,
    QueueError: 500,
    ModelError: 500,
    AudioProcessingError: 500,
    ModelServiceError: 502,
    ProcessingServiceError: 502,
    StorageServiceError: 502,
    UserManagementServiceError: 502,
    AnalyticsServiceError: 502,
    ServiceError: 500
}


def get_http_status_code(exception: Exception) -> int:
    """Get appropriate HTTP status code for an exception."""
    for exc_type, status_code in HTTP_STATUS_MAP.items():
        if isinstance(exception, exc_type):
            return status_code
    return 500  # Default to 500 for unknown exceptions