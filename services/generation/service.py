"""
Generation Service Core Logic

Main service class that orchestrates music generation, manages tasks,
and coordinates with other microservices.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, AsyncGenerator, Any
from uuid import uuid4
import json

import httpx
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge

from .config import GenerationServiceConfig
from .models import (
    GenerationRequest,
    GenerationTask,
    TaskStatus,
    StreamingRequest,
    BatchStatus,
    QueueStatus,
    ServiceStats
)
from ..shared.observability import get_tracer, create_span
from ..shared.exceptions import (
    ServiceError,
    ModelServiceError,
    ProcessingServiceError,
    StorageServiceError
)


# Metrics
TASK_PROCESSING_TIME = Histogram(
    'generation_task_processing_seconds',
    'Time spent processing generation tasks',
    ['model', 'quality']
)
ACTIVE_TASKS = Gauge(
    'generation_active_tasks',
    'Number of actively processing tasks'
)
TASK_QUEUE_SIZE = Gauge(
    'generation_task_queue_size',
    'Number of tasks in queue'
)
MODEL_REQUESTS = Counter(
    'generation_model_requests_total',
    'Total requests to model service',
    ['model', 'status']
)

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class GenerationService:
    """
    Core generation service that orchestrates music generation workflows.
    
    Manages task queues, coordinates with model and processing services,
    and handles both synchronous and streaming generation requests.
    """
    
    def __init__(self, config: GenerationServiceConfig):
        """Initialize the generation service."""
        self.config = config
        self.redis: Optional[redis.Redis] = None
        self.http_client: Optional[httpx.AsyncClient] = None
        
        # Service state
        self.is_initialized = False
        self.worker_tasks: Dict[str, asyncio.Task] = {}
        self.active_streams: Dict[str, Any] = {}
        
        # Statistics
        self.start_time = datetime.utcnow()
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        
        # Task storage (Redis keys)
        self.TASK_KEY_PREFIX = "musicgen:task:"
        self.QUEUE_KEY = "musicgen:queue"
        self.BATCH_KEY_PREFIX = "musicgen:batch:"
        self.PROCESSING_KEY = "musicgen:processing"
    
    async def initialize(self):
        """Initialize service dependencies."""
        if self.is_initialized:
            return
        
        logger.info("Initializing generation service...")
        
        # Initialize Redis connection
        self.redis = redis.from_url(
            self.config.redis_url,
            max_connections=self.config.redis_pool_size
        )
        
        # Test Redis connection
        await self.redis.ping()
        logger.info("Redis connection established")
        
        # Initialize HTTP client for service communication
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.service_timeout),
            limits=httpx.Limits(max_connections=100)
        )
        
        # Start background workers
        await self._start_workers()
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_expired_tasks())
        
        self.is_initialized = True
        logger.info("Generation service initialized successfully")
    
    async def shutdown(self):
        """Shutdown service and cleanup resources."""
        logger.info("Shutting down generation service...")
        
        # Stop workers
        for worker_id, task in self.worker_tasks.items():
            logger.info(f"Stopping worker {worker_id}")
            task.cancel()
        
        # Wait for workers to finish
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks.values(), return_exceptions=True)
        
        # Close connections
        if self.http_client:
            await self.http_client.aclose()
        
        if self.redis:
            await self.redis.close()
        
        self.is_initialized = False
        logger.info("Generation service shutdown complete")
    
    async def _start_workers(self):
        """Start background worker tasks."""
        for i in range(self.config.worker_count):
            worker_id = f"worker_{i}"
            task = asyncio.create_task(self._worker_loop(worker_id))
            self.worker_tasks[worker_id] = task
            logger.info(f"Started worker {worker_id}")
    
    async def _worker_loop(self, worker_id: str):
        """Main worker loop for processing tasks."""
        logger.info(f"Worker {worker_id} started")
        
        while True:
            try:
                # Get next task from queue
                task_data = await self.redis.blpop(self.QUEUE_KEY, timeout=5)
                if not task_data:
                    continue
                
                task_id = task_data[1].decode()
                logger.info(f"Worker {worker_id} processing task {task_id}")
                
                # Mark task as processing
                await self.redis.sadd(self.PROCESSING_KEY, task_id)
                ACTIVE_TASKS.inc()
                
                # Process the task
                await self._process_task(task_id, worker_id)
                
                # Remove from processing set
                await self.redis.srem(self.PROCESSING_KEY, task_id)
                ACTIVE_TASKS.dec()
                
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}", exc_info=True)
                await asyncio.sleep(1)  # Brief pause before retrying
    
    async def _process_task(self, task_id: str, worker_id: str):
        """Process a single generation task."""
        with tracer.start_as_current_span("process_generation_task") as span:
            span.set_attribute("task.id", task_id)
            span.set_attribute("worker.id", worker_id)
            
            try:
                # Get task data
                task = await self.get_task(task_id)
                if not task:
                    logger.error(f"Task {task_id} not found")
                    return
                
                span.set_attribute("task.prompt", task.request.prompt)
                span.set_attribute("task.model", task.request.model or "default")
                
                # Update task status
                task.update_status(TaskStatus.PROCESSING)
                task.worker_id = worker_id
                await self._save_task(task)
                
                with TASK_PROCESSING_TIME.labels(
                    model=task.request.model or "default",
                    quality=task.request.quality
                ).time():
                    
                    # Generate music via model service
                    result_url = await self._generate_music(task)
                    
                    # Process audio if needed
                    if self.config.auto_process_audio:
                        result_url = await self._process_audio(task, result_url)
                    
                    # Update task with results
                    task.result_url = result_url
                    task.update_status(TaskStatus.COMPLETED)
                    await self._save_task(task)
                    
                    self.success_count += 1
                    logger.info(f"Task {task_id} completed successfully")
                
            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}", exc_info=True)
                
                # Update task with error
                await self._handle_task_error(task_id, str(e))
                self.error_count += 1
                
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
    
    async def _generate_music(self, task: GenerationTask) -> str:
        """Generate music via model service."""
        with tracer.start_as_current_span("call_model_service") as span:
            
            model_service_url = self.config.model_service_url
            if not model_service_url:
                raise ModelServiceError("Model service URL not configured")
            
            # Prepare request payload
            payload = {
                "prompt": task.request.prompt,
                "duration": task.request.duration,
                "model": task.request.model,
                "config": task.request.config.dict()
            }
            
            span.set_attribute("model_service.url", model_service_url)
            span.set_attribute("model_service.model", task.request.model or "default")
            
            try:
                response = await self.http_client.post(
                    f"{model_service_url}/generate",
                    json=payload,
                    timeout=self.config.generation_timeout
                )
                
                MODEL_REQUESTS.labels(
                    model=task.request.model or "default",
                    status=response.status_code
                ).inc()
                
                if response.status_code != 200:
                    error_detail = response.text
                    raise ModelServiceError(
                        f"Model service returned {response.status_code}: {error_detail}"
                    )
                
                result = response.json()
                return result["result_url"]
                
            except httpx.TimeoutException:
                MODEL_REQUESTS.labels(
                    model=task.request.model or "default",
                    status="timeout"
                ).inc()
                raise ModelServiceError("Model service request timeout")
            
            except httpx.ConnectError:
                MODEL_REQUESTS.labels(
                    model=task.request.model or "default",
                    status="connection_error"
                ).inc()
                raise ModelServiceError("Cannot connect to model service")
    
    async def _process_audio(self, task: GenerationTask, audio_url: str) -> str:
        """Process audio via processing service."""
        with tracer.start_as_current_span("call_processing_service") as span:
            
            processing_service_url = self.config.processing_service_url
            if not processing_service_url:
                # No processing service configured, return original URL
                return audio_url
            
            payload = {
                "audio_url": audio_url,
                "format": task.request.output_format,
                "normalize": task.request.config.normalization
            }
            
            span.set_attribute("processing_service.url", processing_service_url)
            
            try:
                response = await self.http_client.post(
                    f"{processing_service_url}/process",
                    json=payload,
                    timeout=self.config.processing_timeout
                )
                
                if response.status_code != 200:
                    error_detail = response.text
                    raise ProcessingServiceError(
                        f"Processing service returned {response.status_code}: {error_detail}"
                    )
                
                result = response.json()
                return result["result_url"]
                
            except httpx.TimeoutException:
                raise ProcessingServiceError("Processing service request timeout")
            
            except httpx.ConnectError:
                raise ProcessingServiceError("Cannot connect to processing service")
    
    async def _handle_task_error(self, task_id: str, error_message: str):
        """Handle task processing error."""
        task = await self.get_task(task_id)
        if not task:
            return
        
        task.error_message = error_message
        task.retry_count += 1
        
        if task.can_retry:
            # Retry the task
            task.status = TaskStatus.PENDING
            await self._save_task(task)
            await self.redis.rpush(self.QUEUE_KEY, task_id)
            logger.info(f"Task {task_id} queued for retry {task.retry_count}")
        else:
            # Mark as failed
            task.update_status(TaskStatus.FAILED, error_message)
            await self._save_task(task)
            logger.error(f"Task {task_id} failed permanently: {error_message}")
    
    async def create_generation_task(
        self, 
        request: GenerationRequest,
        batch_id: Optional[str] = None
    ) -> GenerationTask:
        """Create a new generation task."""
        task = GenerationTask(
            request=request,
            batch_id=batch_id
        )
        
        # Save task to Redis
        await self._save_task(task)
        
        # Add to queue
        await self.redis.rpush(self.QUEUE_KEY, task.id)
        
        # Update queue size metric
        queue_size = await self.redis.llen(self.QUEUE_KEY)
        TASK_QUEUE_SIZE.set(queue_size)
        
        self.request_count += 1
        logger.info(f"Created generation task {task.id}")
        
        return task
    
    async def get_task(self, task_id: str) -> Optional[GenerationTask]:
        """Get task by ID."""
        task_key = f"{self.TASK_KEY_PREFIX}{task_id}"
        task_data = await self.redis.get(task_key)
        
        if not task_data:
            return None
        
        task_dict = json.loads(task_data)
        return GenerationTask.parse_obj(task_dict)
    
    async def _save_task(self, task: GenerationTask):
        """Save task to Redis."""
        task_key = f"{self.TASK_KEY_PREFIX}{task.id}"
        task_data = task.json()
        
        # Set expiration time
        expire_seconds = int(self.config.task_ttl.total_seconds())
        
        await self.redis.setex(task_key, expire_seconds, task_data)
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a generation task."""
        task = await self.get_task(task_id)
        if not task:
            return False
        
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            return False
        
        # Remove from queue if pending
        if task.status == TaskStatus.PENDING:
            await self.redis.lrem(self.QUEUE_KEY, 1, task_id)
        
        # Update task status
        task.update_status(TaskStatus.CANCELLED)
        await self._save_task(task)
        
        logger.info(f"Task {task_id} cancelled")
        return True
    
    async def stream_generation(self, request: StreamingRequest) -> AsyncGenerator[bytes, None]:
        """Stream music generation in real-time."""
        # This is a simplified implementation
        # In practice, you'd implement real streaming with the model service
        
        stream_id = str(uuid4())
        self.active_streams[stream_id] = {
            "request": request,
            "start_time": datetime.utcnow()
        }
        
        try:
            # Mock streaming implementation
            # In real implementation, this would stream from model service
            total_chunks = int(request.total_duration or 30.0 / request.chunk_size)
            
            for i in range(total_chunks):
                # Generate chunk via model service
                chunk_data = await self._generate_audio_chunk(request, i)
                yield chunk_data
                
                await asyncio.sleep(0.1)  # Simulate processing time
        
        finally:
            # Cleanup stream
            self.active_streams.pop(stream_id, None)
    
    async def _generate_audio_chunk(self, request: StreamingRequest, chunk_index: int) -> bytes:
        """Generate a single audio chunk."""
        # Placeholder implementation
        # In practice, this would call the model service for streaming generation
        return b"audio_chunk_placeholder"
    
    async def process_batch_generation(self, batch_id: str, task_ids: List[str]):
        """Process batch generation tasks."""
        logger.info(f"Processing batch {batch_id} with {len(task_ids)} tasks")
        
        # Create batch status
        batch_status = BatchStatus(
            batch_id=batch_id,
            total_tasks=len(task_ids),
            completed_tasks=0,
            failed_tasks=0,
            pending_tasks=len(task_ids),
            status="processing",
            created_at=datetime.utcnow(),
            started_at=datetime.utcnow()
        )
        
        await self._save_batch_status(batch_status)
        
        # Monitor batch progress
        asyncio.create_task(self._monitor_batch_progress(batch_id, task_ids))
    
    async def _monitor_batch_progress(self, batch_id: str, task_ids: List[str]):
        """Monitor batch generation progress."""
        while True:
            batch_status = await self.get_batch_status(batch_id)
            if not batch_status:
                break
            
            # Count task statuses
            completed = 0
            failed = 0
            
            for task_id in task_ids:
                task = await self.get_task(task_id)
                if task:
                    if task.status == TaskStatus.COMPLETED:
                        completed += 1
                    elif task.status == TaskStatus.FAILED:
                        failed += 1
            
            # Update batch status
            batch_status.completed_tasks = completed
            batch_status.failed_tasks = failed
            batch_status.pending_tasks = len(task_ids) - completed - failed
            
            if completed + failed == len(task_ids):
                # Batch complete
                batch_status.status = "completed" if failed == 0 else "partial"
                batch_status.completed_at = datetime.utcnow()
                batch_status.success_rate = completed / len(task_ids)
                
                await self._save_batch_status(batch_status)
                logger.info(f"Batch {batch_id} completed: {completed} success, {failed} failed")
                break
            
            await self._save_batch_status(batch_status)
            await asyncio.sleep(10)  # Check every 10 seconds
    
    async def get_batch_status(self, batch_id: str) -> Optional[BatchStatus]:
        """Get batch status."""
        batch_key = f"{self.BATCH_KEY_PREFIX}{batch_id}"
        batch_data = await self.redis.get(batch_key)
        
        if not batch_data:
            return None
        
        batch_dict = json.loads(batch_data)
        return BatchStatus.parse_obj(batch_dict)
    
    async def _save_batch_status(self, batch_status: BatchStatus):
        """Save batch status to Redis."""
        batch_key = f"{self.BATCH_KEY_PREFIX}{batch_status.batch_id}"
        batch_data = batch_status.json()
        
        # Set expiration time
        expire_seconds = int(self.config.batch_ttl.total_seconds())
        
        await self.redis.setex(batch_key, expire_seconds, batch_data)
    
    async def get_queue_status(self) -> QueueStatus:
        """Get current queue status."""
        queue_size = await self.redis.llen(self.QUEUE_KEY)
        processing_size = await self.redis.scard(self.PROCESSING_KEY)
        
        # Get task counts by status (simplified implementation)
        # In practice, you'd maintain counters in Redis
        
        return QueueStatus(
            total_tasks=self.request_count,
            pending_tasks=queue_size,
            processing_tasks=processing_size,
            completed_tasks=self.success_count,
            failed_tasks=self.error_count,
            active_workers=len(self.worker_tasks),
            healthy_workers=len([t for t in self.worker_tasks.values() if not t.done()]),
            queue_health="healthy" if queue_size < 100 else "degraded"
        )
    
    async def list_tasks(
        self,
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None
    ) -> List[GenerationTask]:
        """List tasks with filtering."""
        # Simplified implementation
        # In practice, you'd use Redis SCAN or maintain indexed task lists
        
        tasks = []
        # This is a placeholder implementation
        # Real implementation would scan Redis keys efficiently
        
        return tasks[:limit]
    
    async def get_service_stats(self) -> ServiceStats:
        """Get service statistics."""
        queue_status = await self.get_queue_status()
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        return ServiceStats(
            total_requests=self.request_count,
            successful_requests=self.success_count,
            failed_requests=self.error_count,
            queue_stats=queue_status,
            service_health="healthy" if self.is_initialized else "unhealthy",
            uptime_seconds=uptime
        )
    
    async def get_result_data(self, result_url: str) -> bytes:
        """Get result data from storage service."""
        try:
            response = await self.http_client.get(result_url)
            if response.status_code == 200:
                return response.content
            else:
                raise StorageServiceError(f"Failed to retrieve result: {response.status_code}")
        except Exception as e:
            raise StorageServiceError(f"Error retrieving result: {str(e)}")
    
    async def get_health(self) -> Dict[str, Any]:
        """Get service health status."""
        try:
            # Check Redis connection
            await self.redis.ping()
            redis_healthy = True
        except Exception:
            redis_healthy = False
        
        # Check worker health
        healthy_workers = len([t for t in self.worker_tasks.values() if not t.done()])
        
        # Overall health
        is_healthy = (
            self.is_initialized and
            redis_healthy and
            healthy_workers > 0
        )
        
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "redis_connected": redis_healthy,
            "active_workers": healthy_workers,
            "total_workers": len(self.worker_tasks),
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds()
        }
    
    async def _cleanup_expired_tasks(self):
        """Cleanup expired tasks and maintain Redis hygiene."""
        while True:
            try:
                # This runs every hour to cleanup expired data
                await asyncio.sleep(3600)
                
                # Remove completed tasks older than TTL
                # Implementation would scan and cleanup old task keys
                logger.debug("Performing periodic cleanup of expired tasks")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")