"""
Audio Processing Service FastAPI Application

Microservice dedicated to audio manipulation, effects processing,
format conversion, and audio enhancement operations.
"""

import asyncio
import logging
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, status
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import Counter, Histogram, Gauge
import redis.asyncio as redis

from .service import AudioProcessingService
from .config import ProcessingServiceConfig
from .models import (
    ProcessingRequest,
    ProcessingResponse,
    AudioFormat,
    EffectType,
    ProcessingTask,
    TaskStatus,
    EnhancementRequest,
    ConversionRequest,
    MixingRequest,
    ProcessingStats
)
from .dependencies import get_processing_service, get_redis_client
from ..shared.observability import get_tracer, create_span
from ..shared.middleware import MetricsMiddleware, TracingMiddleware


# Metrics
PROCESSING_REQUESTS = Counter(
    'processing_requests_total',
    'Total processing requests',
    ['operation', 'format', 'status']
)
PROCESSING_DURATION = Histogram(
    'processing_duration_seconds',
    'Processing request duration',
    ['operation', 'format']
)
ACTIVE_PROCESSING_TASKS = Gauge(
    'active_processing_tasks',
    'Number of active processing tasks'
)
AUDIO_FILES_PROCESSED = Counter(
    'audio_files_processed_total',
    'Total audio files processed',
    ['operation']
)

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager."""
    config = app.state.config
    
    # Initialize processing service
    processing_service = AudioProcessingService(config)
    await processing_service.initialize()
    app.state.processing_service = processing_service
    
    # Initialize Redis connection
    redis_client = redis.from_url(config.redis_url)
    app.state.redis_client = redis_client
    
    logger.info("Audio processing service started")
    
    yield
    
    # Cleanup
    await processing_service.shutdown()
    await redis_client.close()
    
    logger.info("Audio processing service stopped")


def create_processing_app(config: Optional[ProcessingServiceConfig] = None) -> FastAPI:
    """
    Create and configure the Audio Processing Service FastAPI application.
    
    Args:
        config: Service configuration
        
    Returns:
        Configured FastAPI application
    """
    if config is None:
        config = ProcessingServiceConfig()
    
    app = FastAPI(
        title="MusicGen AI Audio Processing Service",
        description="Audio manipulation, effects, and format conversion",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Store config in app state
    app.state.config = config
    
    # Add middleware
    app.add_middleware(TracingMiddleware)
    app.add_middleware(MetricsMiddleware, service_name="processing")
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Service health check."""
        try:
            processing_service = app.state.processing_service
            health_status = await processing_service.get_health()
            return health_status
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "error": str(e)}
            )
    
    # Audio processing endpoints
    @app.post("/process", response_model=ProcessingResponse)
    async def process_audio(
        request: ProcessingRequest,
        processing_service: AudioProcessingService = Depends(get_processing_service)
    ):
        """
        Process audio with specified operations.
        
        Supports format conversion, effects application, and enhancement.
        """
        with tracer.start_as_current_span("process_audio") as span:
            span.set_attribute("operation", request.operation)
            span.set_attribute("input_format", request.input_format)
            span.set_attribute("output_format", request.output_format)
            
            try:
                # Create processing task
                task = await processing_service.create_processing_task(request)
                
                # Process audio
                result_url = await processing_service.process_audio(task)
                
                # Update metrics
                PROCESSING_REQUESTS.labels(
                    operation=request.operation,
                    format=request.output_format,
                    status="success"
                ).inc()
                
                AUDIO_FILES_PROCESSED.labels(operation=request.operation).inc()
                
                return ProcessingResponse(
                    task_id=task.id,
                    status=TaskStatus.COMPLETED,
                    result_url=result_url,
                    message="Audio processing completed successfully"
                )
                
            except Exception as e:
                PROCESSING_REQUESTS.labels(
                    operation=request.operation,
                    format=request.output_format,
                    status="failed"
                ).inc()
                
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                
                logger.error(f"Audio processing failed: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Audio processing failed: {str(e)}"
                )
    
    @app.post("/upload-process", response_model=ProcessingResponse)
    async def upload_and_process(
        file: UploadFile = File(...),
        operation: str = Form(...),
        output_format: str = Form("wav"),
        effects: Optional[str] = Form(None),
        processing_service: AudioProcessingService = Depends(get_processing_service)
    ):
        """
        Upload audio file and process it.
        
        Accepts audio file upload and applies specified processing operations.
        """
        with tracer.start_as_current_span("upload_and_process") as span:
            span.set_attribute("operation", operation)
            span.set_attribute("filename", file.filename)
            span.set_attribute("content_type", file.content_type)
            
            try:
                # Validate file type
                if not file.content_type.startswith("audio/"):
                    raise HTTPException(
                        status_code=400,
                        detail="File must be an audio file"
                    )
                
                # Read file data
                file_data = await file.read()
                
                # Create processing request
                request = ProcessingRequest(
                    audio_data=file_data,
                    operation=operation,
                    output_format=output_format,
                    effects=effects.split(",") if effects else None
                )
                
                # Create and process task
                task = await processing_service.create_processing_task(request)
                result_url = await processing_service.process_audio(task)
                
                # Update metrics
                PROCESSING_REQUESTS.labels(
                    operation=operation,
                    format=output_format,
                    status="success"
                ).inc()
                
                return ProcessingResponse(
                    task_id=task.id,
                    status=TaskStatus.COMPLETED,
                    result_url=result_url,
                    message="Audio upload and processing completed"
                )
                
            except HTTPException:
                raise
            except Exception as e:
                PROCESSING_REQUESTS.labels(
                    operation=operation,
                    format=output_format,
                    status="failed"
                ).inc()
                
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                
                logger.error(f"Upload and process failed: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Upload and processing failed: {str(e)}"
                )
    
    @app.post("/enhance", response_model=ProcessingResponse)
    async def enhance_audio(
        request: EnhancementRequest,
        processing_service: AudioProcessingService = Depends(get_processing_service)
    ):
        """
        Enhance audio quality using various algorithms.
        
        Supports noise reduction, normalization, and quality enhancement.
        """
        with tracer.start_as_current_span("enhance_audio") as span:
            span.set_attribute("enhancement_type", request.enhancement_type)
            span.set_attribute("quality_target", request.quality_target)
            
            try:
                task = await processing_service.enhance_audio(request)
                
                PROCESSING_REQUESTS.labels(
                    operation="enhancement",
                    format=request.output_format,
                    status="success"
                ).inc()
                
                return ProcessingResponse(
                    task_id=task.id,
                    status=task.status,
                    result_url=task.result_url,
                    message="Audio enhancement completed"
                )
                
            except Exception as e:
                PROCESSING_REQUESTS.labels(
                    operation="enhancement",
                    format=request.output_format,
                    status="failed"
                ).inc()
                
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                
                logger.error(f"Audio enhancement failed: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Audio enhancement failed: {str(e)}"
                )
    
    @app.post("/convert", response_model=ProcessingResponse)
    async def convert_format(
        request: ConversionRequest,
        processing_service: AudioProcessingService = Depends(get_processing_service)
    ):
        """
        Convert audio between different formats.
        
        Supports conversion between WAV, MP3, FLAC, and other formats.
        """
        with tracer.start_as_current_span("convert_format") as span:
            span.set_attribute("input_format", request.input_format)
            span.set_attribute("output_format", request.output_format)
            span.set_attribute("quality", request.quality)
            
            try:
                task = await processing_service.convert_format(request)
                
                PROCESSING_REQUESTS.labels(
                    operation="conversion",
                    format=request.output_format,
                    status="success"
                ).inc()
                
                return ProcessingResponse(
                    task_id=task.id,
                    status=task.status,
                    result_url=task.result_url,
                    message="Format conversion completed"
                )
                
            except Exception as e:
                PROCESSING_REQUESTS.labels(
                    operation="conversion",
                    format=request.output_format,
                    status="failed"
                ).inc()
                
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                
                logger.error(f"Format conversion failed: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Format conversion failed: {str(e)}"
                )
    
    @app.post("/mix", response_model=ProcessingResponse)
    async def mix_audio(
        request: MixingRequest,
        processing_service: AudioProcessingService = Depends(get_processing_service)
    ):
        """
        Mix multiple audio tracks together.
        
        Supports multi-track mixing with level control and effects.
        """
        with tracer.start_as_current_span("mix_audio") as span:
            span.set_attribute("track_count", len(request.tracks))
            span.set_attribute("mix_mode", request.mix_mode)
            
            try:
                task = await processing_service.mix_audio(request)
                
                PROCESSING_REQUESTS.labels(
                    operation="mixing",
                    format=request.output_format,
                    status="success"
                ).inc()
                
                return ProcessingResponse(
                    task_id=task.id,
                    status=task.status,
                    result_url=task.result_url,
                    message="Audio mixing completed"
                )
                
            except Exception as e:
                PROCESSING_REQUESTS.labels(
                    operation="mixing",
                    format=request.output_format,
                    status="failed"
                ).inc()
                
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                
                logger.error(f"Audio mixing failed: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Audio mixing failed: {str(e)}"
                )
    
    @app.get("/effects")
    async def list_available_effects():
        """List all available audio effects."""
        return {
            "effects": [
                {
                    "name": "reverb",
                    "description": "Add reverb effect",
                    "parameters": ["room_size", "damping", "wet_level"]
                },
                {
                    "name": "chorus",
                    "description": "Add chorus effect",
                    "parameters": ["rate", "depth", "feedback"]
                },
                {
                    "name": "eq",
                    "description": "Equalizer",
                    "parameters": ["low_gain", "mid_gain", "high_gain"]
                },
                {
                    "name": "compressor",
                    "description": "Dynamic range compression",
                    "parameters": ["threshold", "ratio", "attack", "release"]
                },
                {
                    "name": "distortion",
                    "description": "Add distortion effect",
                    "parameters": ["drive", "tone"]
                }
            ]
        }
    
    @app.get("/formats")
    async def list_supported_formats():
        """List all supported audio formats."""
        return {
            "input_formats": ["wav", "mp3", "flac", "ogg", "m4a", "aac"],
            "output_formats": ["wav", "mp3", "flac", "ogg"],
            "quality_levels": {
                "mp3": ["128", "192", "256", "320"],
                "ogg": ["128", "192", "256", "320"],
                "flac": ["16bit", "24bit"]
            }
        }
    
    # Task management endpoints
    @app.get("/tasks/{task_id}", response_model=ProcessingTask)
    async def get_task_status(
        task_id: str,
        processing_service: AudioProcessingService = Depends(get_processing_service)
    ):
        """Get processing task status."""
        try:
            task = await processing_service.get_task(task_id)
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
    
    @app.get("/tasks/{task_id}/result")
    async def get_task_result(
        task_id: str,
        processing_service: AudioProcessingService = Depends(get_processing_service)
    ):
        """Get processing task result."""
        try:
            task = await processing_service.get_task(task_id)
            if not task:
                raise HTTPException(
                    status_code=404,
                    detail="Task not found"
                )
            
            if task.status != TaskStatus.COMPLETED:
                raise HTTPException(
                    status_code=202,
                    detail=f"Task not completed, current status: {task.status}"
                )
            
            if not task.result_url:
                raise HTTPException(
                    status_code=500,
                    detail="Task completed but no result available"
                )
            
            # Get audio file from storage
            audio_data = await processing_service.get_result_data(task.result_url)
            
            return StreamingResponse(
                iter([audio_data]),
                media_type=f"audio/{task.output_format}",
                headers={
                    "Content-Disposition": f"attachment; filename=processed_{task_id}.{task.output_format}"
                }
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get task result: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve task result"
            )
    
    @app.delete("/tasks/{task_id}")
    async def cancel_task(
        task_id: str,
        processing_service: AudioProcessingService = Depends(get_processing_service)
    ):
        """Cancel a processing task."""
        try:
            success = await processing_service.cancel_task(task_id)
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
    
    @app.get("/tasks")
    async def list_tasks(
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None,
        processing_service: AudioProcessingService = Depends(get_processing_service)
    ):
        """List processing tasks with filtering."""
        try:
            tasks = await processing_service.list_tasks(
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
    @app.get("/stats", response_model=ProcessingStats)
    async def get_processing_stats(
        processing_service: AudioProcessingService = Depends(get_processing_service)
    ):
        """Get processing service statistics."""
        try:
            stats = await processing_service.get_service_stats()
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get service stats: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve service statistics"
            )
    
    # Batch processing endpoints
    @app.post("/batch/process")
    async def batch_process(
        requests: List[ProcessingRequest],
        processing_service: AudioProcessingService = Depends(get_processing_service)
    ):
        """Process multiple audio files in batch."""
        with tracer.start_as_current_span("batch_process") as span:
            span.set_attribute("batch_size", len(requests))
            
            try:
                if len(requests) > 100:
                    raise HTTPException(
                        status_code=400,
                        detail="Batch size cannot exceed 100 requests"
                    )
                
                batch_id = await processing_service.process_batch(requests)
                
                return {
                    "batch_id": batch_id,
                    "message": f"Batch processing started for {len(requests)} files",
                    "task_count": len(requests)
                }
                
            except HTTPException:
                raise
            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                
                logger.error(f"Batch processing failed: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Batch processing failed: {str(e)}"
                )
    
    @app.get("/batch/{batch_id}/status")
    async def get_batch_status(
        batch_id: str,
        processing_service: AudioProcessingService = Depends(get_processing_service)
    ):
        """Get batch processing status."""
        try:
            status = await processing_service.get_batch_status(batch_id)
            if not status:
                raise HTTPException(
                    status_code=404,
                    detail="Batch not found"
                )
            return status
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get batch status: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve batch status"
            )
    
    return app


# Factory for creating the app
def create_app() -> FastAPI:
    """Factory function for creating the app."""
    return create_processing_app()


if __name__ == "__main__":
    import uvicorn
    
    app = create_processing_app()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
        log_level="info"
    )