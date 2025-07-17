"""
Generation Service FastAPI Application

Music generation orchestration service that handles generation requests,
manages task queues, and coordinates with model and processing services.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.responses import JSONResponse, StreamingResponse
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge

from .service import GenerationService
from .config import GenerationServiceConfig
from .models import (
    GenerationRequest,
    GenerationResponse,
    GenerationTask,
    TaskStatus,
    StreamingRequest,
    BatchGenerationRequest,
    BatchGenerationResponse
)
from .dependencies import get_generation_service, get_redis_client
from ..shared.observability import get_tracer, create_span
from ..shared.middleware import MetricsMiddleware, TracingMiddleware


# Metrics
GENERATION_REQUESTS = Counter(
    'generation_requests_total',
    'Total generation requests',
    ['type', 'model', 'status']
)
GENERATION_DURATION = Histogram(
    'generation_duration_seconds',
    'Generation request duration',
    ['type', 'model']
)
ACTIVE_GENERATIONS = Gauge(
    'active_generations',
    'Number of active generation tasks'
)
QUEUE_SIZE = Gauge(
    'generation_queue_size',
    'Number of tasks in generation queue'
)

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager."""
    config = app.state.config
    
    # Initialize generation service
    generation_service = GenerationService(config)
    await generation_service.initialize()
    app.state.generation_service = generation_service
    
    # Initialize Redis connection
    redis_client = redis.from_url(config.redis_url)
    app.state.redis_client = redis_client
    
    logger.info("Generation service started")
    
    yield
    
    # Cleanup
    await generation_service.shutdown()
    await redis_client.close()
    
    logger.info("Generation service stopped")


def create_generation_app(config: Optional[GenerationServiceConfig] = None) -> FastAPI:
    """
    Create and configure the Generation Service FastAPI application.
    
    Args:
        config: Service configuration
        
    Returns:
        Configured FastAPI application
    """
    if config is None:
        config = GenerationServiceConfig()
    
    app = FastAPI(
        title="MusicGen AI Generation Service",
        description="Music generation orchestration and task management",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Store config in app state
    app.state.config = config
    
    # Add middleware
    app.add_middleware(TracingMiddleware)
    app.add_middleware(MetricsMiddleware, service_name="generation")
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Service health check."""
        try:
            generation_service = app.state.generation_service
            health_status = await generation_service.get_health()
            return health_status
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "error": str(e)}
            )
    
    # Generation endpoints
    @app.post("/generate", response_model=GenerationResponse)
    async def generate_music(
        request: GenerationRequest,
        background_tasks: BackgroundTasks,
        generation_service: GenerationService = Depends(get_generation_service)
    ):
        """
        Generate music from text prompt.
        
        Creates a generation task and returns task ID for status tracking.
        """
        with tracer.start_as_current_span("generate_music") as span:
            span.set_attribute("prompt", request.prompt)
            span.set_attribute("duration", request.duration)
            span.set_attribute("model", request.model or "default")
            
            try:
                # Create generation task
                task = await generation_service.create_generation_task(request)
                
                # Start generation in background
                background_tasks.add_task(
                    generation_service.process_generation_task,
                    task.id
                )
                
                # Update metrics
                GENERATION_REQUESTS.labels(
                    type="single",
                    model=request.model or "default",
                    status="started"
                ).inc()
                
                ACTIVE_GENERATIONS.inc()
                
                return GenerationResponse(
                    task_id=task.id,
                    status=task.status,
                    message="Generation task created successfully"
                )
                
            except Exception as e:
                GENERATION_REQUESTS.labels(
                    type="single",
                    model=request.model or "default", 
                    status="failed"
                ).inc()
                
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                
                logger.error(f"Failed to create generation task: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to create generation task: {str(e)}"
                )
    
    @app.get("/generate/{task_id}/status", response_model=GenerationTask)
    async def get_generation_status(
        task_id: str,
        generation_service: GenerationService = Depends(get_generation_service)
    ):
        """Get generation task status."""
        try:
            task = await generation_service.get_task(task_id)
            if not task:
                raise HTTPException(
                    status_code=404,
                    detail="Task not found"
                )
            return task
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get task status: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve task status"
            )
    
    @app.get("/generate/{task_id}/result")
    async def get_generation_result(
        task_id: str,
        generation_service: GenerationService = Depends(get_generation_service)
    ):
        """Get generation result (audio file or error)."""
        try:
            task = await generation_service.get_task(task_id)
            if not task:
                raise HTTPException(
                    status_code=404,
                    detail="Task not found"
                )
            
            if task.status == TaskStatus.PENDING:
                raise HTTPException(
                    status_code=202,
                    detail="Task still pending"
                )
            elif task.status == TaskStatus.PROCESSING:
                raise HTTPException(
                    status_code=202,
                    detail="Task still processing"
                )
            elif task.status == TaskStatus.FAILED:
                raise HTTPException(
                    status_code=500,
                    detail=f"Task failed: {task.error_message}"
                )
            elif task.status == TaskStatus.COMPLETED:
                if not task.result_url:
                    raise HTTPException(
                        status_code=500,
                        detail="Task completed but no result available"
                    )
                
                # Get audio file from storage service
                audio_data = await generation_service.get_result_data(task.result_url)
                
                return StreamingResponse(
                    iter([audio_data]),
                    media_type="audio/wav",
                    headers={
                        "Content-Disposition": f"attachment; filename=generated_{task_id}.wav"
                    }
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Unknown task status: {task.status}"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get generation result: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve generation result"
            )
    
    @app.delete("/generate/{task_id}")
    async def cancel_generation(
        task_id: str,
        generation_service: GenerationService = Depends(get_generation_service)
    ):
        """Cancel a generation task."""
        try:
            success = await generation_service.cancel_task(task_id)
            if not success:
                raise HTTPException(
                    status_code=404,
                    detail="Task not found or cannot be cancelled"
                )
            
            return {"message": "Task cancelled successfully"}
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to cancel task: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to cancel task"
            )
    
    @app.post("/stream")
    async def stream_generation(
        request: StreamingRequest,
        generation_service: GenerationService = Depends(get_generation_service)
    ):
        """
        Stream music generation in real-time.
        
        Returns a streaming response with audio chunks as they are generated.
        """
        with tracer.start_as_current_span("stream_generation") as span:
            span.set_attribute("prompt", request.prompt)
            span.set_attribute("chunk_size", request.chunk_size)
            
            try:
                # Create streaming generator
                audio_stream = generation_service.stream_generation(request)
                
                # Update metrics
                GENERATION_REQUESTS.labels(
                    type="streaming",
                    model=request.model or "default",
                    status="started"
                ).inc()
                
                return StreamingResponse(
                    audio_stream,
                    media_type="audio/wav",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive"
                    }
                )
                
            except Exception as e:
                GENERATION_REQUESTS.labels(
                    type="streaming",
                    model=request.model or "default",
                    status="failed"
                ).inc()
                
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                
                logger.error(f"Streaming generation failed: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Streaming generation failed: {str(e)}"
                )
    
    @app.post("/generate/batch", response_model=BatchGenerationResponse)
    async def batch_generate(
        request: BatchGenerationRequest,
        background_tasks: BackgroundTasks,
        generation_service: GenerationService = Depends(get_generation_service)
    ):
        """
        Generate multiple music pieces in batch.
        
        Creates multiple generation tasks and returns batch ID for tracking.
        """
        with tracer.start_as_current_span("batch_generation") as span:
            span.set_attribute("batch_size", len(request.requests))
            
            try:
                # Create batch generation tasks
                batch_id = str(uuid4())
                tasks = []
                
                for gen_request in request.requests:
                    task = await generation_service.create_generation_task(
                        gen_request,
                        batch_id=batch_id
                    )
                    tasks.append(task)
                
                # Start batch processing in background
                background_tasks.add_task(
                    generation_service.process_batch_generation,
                    batch_id,
                    [task.id for task in tasks]
                )
                
                # Update metrics
                GENERATION_REQUESTS.labels(
                    type="batch",
                    model="mixed",
                    status="started"
                ).inc()
                
                ACTIVE_GENERATIONS.inc(len(tasks))
                
                return BatchGenerationResponse(
                    batch_id=batch_id,
                    task_ids=[task.id for task in tasks],
                    total_tasks=len(tasks),
                    status="processing"
                )
                
            except Exception as e:
                GENERATION_REQUESTS.labels(
                    type="batch",
                    model="mixed",
                    status="failed"
                ).inc()
                
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                
                logger.error(f"Batch generation failed: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Batch generation failed: {str(e)}"
                )
    
    @app.get("/generate/batch/{batch_id}/status")
    async def get_batch_status(
        batch_id: str,
        generation_service: GenerationService = Depends(get_generation_service)
    ):
        """Get batch generation status."""
        try:
            batch_status = await generation_service.get_batch_status(batch_id)
            if not batch_status:
                raise HTTPException(
                    status_code=404,
                    detail="Batch not found"
                )
            return batch_status
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get batch status: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve batch status"
            )
    
    # Queue management endpoints
    @app.get("/queue/status")
    async def get_queue_status(
        generation_service: GenerationService = Depends(get_generation_service)
    ):
        """Get generation queue status."""
        try:
            queue_status = await generation_service.get_queue_status()
            
            # Update queue metrics
            QUEUE_SIZE.set(queue_status.get("pending_tasks", 0))
            
            return queue_status
            
        except Exception as e:
            logger.error(f"Failed to get queue status: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve queue status"
            )
    
    @app.get("/tasks")
    async def list_tasks(
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None,
        generation_service: GenerationService = Depends(get_generation_service)
    ):
        """List generation tasks with filtering."""
        try:
            tasks = await generation_service.list_tasks(
                limit=limit,
                offset=offset,
                status=status
            )
            return {"tasks": tasks, "limit": limit, "offset": offset}
            
        except Exception as e:
            logger.error(f"Failed to list tasks: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve tasks"
            )
    
    # Service statistics
    @app.get("/stats")
    async def get_service_stats(
        generation_service: GenerationService = Depends(get_generation_service)
    ):
        """Get service statistics."""
        try:
            stats = await generation_service.get_service_stats()
            return stats
            
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
    return create_generation_app()


if __name__ == "__main__":
    import uvicorn
    
    app = create_generation_app()
    uvicorn.run(
        app,
        host="0.0.0.0", 
        port=8001,
        log_level="info"
    )