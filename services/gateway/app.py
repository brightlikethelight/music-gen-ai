"""
API Gateway FastAPI Application

Central API gateway providing request routing, load balancing, authentication,
rate limiting, and observability for the MusicGen AI microservices ecosystem.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import httpx
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from .config import GatewayConfig
from .middleware import (
    AuthenticationMiddleware,
    RateLimitingMiddleware, 
    RequestLoggingMiddleware,
    CircuitBreakerMiddleware
)
from .routing import GatewayRouter
from .health import HealthChecker
from .discovery import ServiceDiscovery
from ..shared.observability import get_tracer, create_span


# Metrics
REQUEST_COUNT = Counter(
    'gateway_requests_total',
    'Total gateway requests',
    ['method', 'endpoint', 'status']
)
REQUEST_DURATION = Histogram(
    'gateway_request_duration_seconds',
    'Gateway request duration',
    ['method', 'endpoint']
)
SERVICE_REQUEST_COUNT = Counter(
    'gateway_service_requests_total', 
    'Total requests to backend services',
    ['service', 'status']
)

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class APIGateway:
    """
    API Gateway for MusicGen AI microservices.
    
    Provides centralized routing, authentication, rate limiting,
    load balancing, and observability.
    """
    
    def __init__(self, config: GatewayConfig):
        """Initialize API Gateway."""
        self.config = config
        self.router = GatewayRouter(config)
        self.health_checker = HealthChecker(config)
        self.service_discovery = ServiceDiscovery(config)
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.request_timeout),
            limits=httpx.Limits(
                max_connections=config.max_connections,
                max_keepalive_connections=config.max_keepalive_connections
            )
        )
        
    async def startup(self):
        """Gateway startup tasks."""
        logger.info("Starting API Gateway...")
        
        # Initialize service discovery
        await self.service_discovery.start()
        
        # Start health checking
        await self.health_checker.start()
        
        # Warm up connections to services
        await self._warmup_connections()
        
        logger.info("API Gateway started successfully")
    
    async def shutdown(self):
        """Gateway shutdown tasks."""
        logger.info("Shutting down API Gateway...")
        
        # Stop health checking
        await self.health_checker.stop()
        
        # Stop service discovery
        await self.service_discovery.stop()
        
        # Close HTTP client
        await self.client.aclose()
        
        logger.info("API Gateway shutdown complete")
    
    async def _warmup_connections(self):
        """Warm up connections to backend services."""
        tasks = []
        for service_name in self.config.services:
            task = asyncio.create_task(self._warmup_service(service_name))
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _warmup_service(self, service_name: str):
        """Warm up connection to a specific service."""
        try:
            service_url = await self.service_discovery.get_service_url(service_name)
            if service_url:
                health_url = f"{service_url}/health"
                response = await self.client.get(health_url)
                if response.status_code == 200:
                    logger.debug(f"Warmed up connection to {service_name}")
        except Exception as e:
            logger.warning(f"Failed to warm up {service_name}: {e}")
    
    async def proxy_request(
        self, 
        request: Request, 
        service_name: str, 
        path: str
    ) -> Response:
        """
        Proxy request to backend service.
        
        Args:
            request: Incoming request
            service_name: Target service name
            path: Request path within service
            
        Returns:
            Response from backend service
        """
        with tracer.start_as_current_span("gateway_proxy_request") as span:
            span.set_attribute("service.name", service_name)
            span.set_attribute("request.path", path)
            
            # Get service URL from discovery
            service_url = await self.service_discovery.get_service_url(service_name)
            if not service_url:
                SERVICE_REQUEST_COUNT.labels(service=service_name, status="unavailable").inc()
                raise HTTPException(
                    status_code=503,
                    detail=f"Service {service_name} unavailable"
                )
            
            # Construct target URL
            target_url = f"{service_url}{path}"
            if request.query_params:
                target_url += f"?{request.query_params}"
            
            span.set_attribute("target.url", target_url)
            
            try:
                # Prepare request
                headers = dict(request.headers)
                
                # Add tracing headers
                span.inject_headers(headers)
                
                # Remove hop-by-hop headers
                headers.pop("host", None)
                headers.pop("connection", None)
                
                # Read request body
                body = await request.body()
                
                # Make request to backend service
                response = await self.client.request(
                    method=request.method,
                    url=target_url,
                    headers=headers,
                    content=body
                )
                
                # Update metrics
                SERVICE_REQUEST_COUNT.labels(
                    service=service_name,
                    status=response.status_code
                ).inc()
                
                span.set_attribute("response.status_code", response.status_code)
                
                # Prepare response headers
                response_headers = dict(response.headers)
                
                # Remove hop-by-hop headers
                response_headers.pop("connection", None)
                response_headers.pop("transfer-encoding", None)
                
                # Return response
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=response_headers,
                    media_type=response.headers.get("content-type")
                )
                
            except httpx.TimeoutException:
                SERVICE_REQUEST_COUNT.labels(service=service_name, status="timeout").inc()
                span.set_attribute("error", True)
                span.set_attribute("error.message", "Request timeout")
                raise HTTPException(status_code=504, detail="Gateway timeout")
                
            except httpx.ConnectError:
                SERVICE_REQUEST_COUNT.labels(service=service_name, status="connection_error").inc()
                span.set_attribute("error", True)
                span.set_attribute("error.message", "Connection error")
                raise HTTPException(status_code=502, detail="Bad gateway")
                
            except Exception as e:
                SERVICE_REQUEST_COUNT.labels(service=service_name, status="error").inc()
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                logger.error(f"Proxy error for {service_name}: {e}")
                raise HTTPException(status_code=500, detail="Internal gateway error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager."""
    gateway = app.state.gateway
    await gateway.startup()
    yield
    await gateway.shutdown()


def create_gateway_app(config: Optional[GatewayConfig] = None) -> FastAPI:
    """
    Create and configure the API Gateway FastAPI application.
    
    Args:
        config: Gateway configuration (optional, will use defaults)
        
    Returns:
        Configured FastAPI application
    """
    if config is None:
        config = GatewayConfig()
    
    # Create FastAPI app
    app = FastAPI(
        title="MusicGen AI Gateway",
        description="API Gateway for MusicGen AI microservices",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Create gateway instance
    gateway = APIGateway(config)
    app.state.gateway = gateway
    
    # Add middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=config.allowed_hosts
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=config.cors_methods,
        allow_headers=config.cors_headers
    )
    
    # Add custom middleware
    app.add_middleware(CircuitBreakerMiddleware, gateway=gateway)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RateLimitingMiddleware, config=config)
    app.add_middleware(AuthenticationMiddleware, config=config)
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Gateway health check."""
        health_status = await gateway.health_checker.get_health()
        status_code = 200 if health_status["status"] == "healthy" else 503
        return JSONResponse(
            content=health_status,
            status_code=status_code
        )
    
    # Metrics endpoint
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(
            generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )
    
    # Service routing
    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def route_request(request: Request, path: str):
        """Route requests to appropriate backend services."""
        
        # Update request metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=path,
            status="processing"
        ).inc()
        
        with REQUEST_DURATION.labels(
            method=request.method,
            endpoint=path
        ).time():
            
            # Determine target service
            service_name = gateway.router.get_service_for_path(path)
            if not service_name:
                REQUEST_COUNT.labels(
                    method=request.method,
                    endpoint=path, 
                    status="404"
                ).inc()
                raise HTTPException(status_code=404, detail="Service not found")
            
            # Proxy request
            try:
                response = await gateway.proxy_request(request, service_name, f"/{path}")
                
                REQUEST_COUNT.labels(
                    method=request.method,
                    endpoint=path,
                    status=response.status_code
                ).inc()
                
                return response
                
            except HTTPException as e:
                REQUEST_COUNT.labels(
                    method=request.method,
                    endpoint=path,
                    status=e.status_code
                ).inc()
                raise
            except Exception as e:
                REQUEST_COUNT.labels(
                    method=request.method,
                    endpoint=path,
                    status="500"
                ).inc()
                logger.error(f"Unexpected error routing request: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
    
    # Exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "Gateway Error",
                "message": exc.detail,
                "status_code": exc.status_code,
                "path": request.url.path
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error", 
                "message": "An unexpected error occurred",
                "status_code": 500,
                "path": request.url.path
            }
        )
    
    return app


# For running with uvicorn
def create_app() -> FastAPI:
    """Factory function for creating the app."""
    return create_gateway_app()


if __name__ == "__main__":
    import uvicorn
    
    app = create_gateway_app()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )