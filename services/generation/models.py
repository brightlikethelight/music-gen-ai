"""
Generation Service Data Models

Pydantic models for generation requests, responses, and internal data structures.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Union
from uuid import uuid4

from pydantic import BaseModel, Field, validator


class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class QualityMode(str, Enum):
    """Generation quality mode."""
    FAST = "fast"
    STANDARD = "standard"
    HIGH = "high"


class ModelSize(str, Enum):
    """Model size options."""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class GenerationConfig(BaseModel):
    """Generation configuration parameters."""
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    top_k: int = Field(default=250, ge=1, le=2048)
    top_p: float = Field(default=0.0, ge=0.0, le=1.0)
    cfg_coef: float = Field(default=7.5, ge=1.0, le=20.0)
    
    # Audio settings
    sample_rate: int = Field(default=24000, ge=8000, le=96000)
    format: str = Field(default="wav", regex="^(wav|mp3|flac)$")
    normalization: bool = True
    
    # Generation settings
    use_caching: bool = True
    seed: Optional[int] = Field(default=None, ge=0, le=2**32-1)
    guidance_scale: float = Field(default=1.0, ge=0.1, le=10.0)


class GenerationRequest(BaseModel):
    """Request for music generation."""
    prompt: str = Field(..., min_length=1, max_length=1000)
    duration: float = Field(default=10.0, ge=0.1, le=300.0)
    
    # Model selection
    model: Optional[str] = None
    model_size: Optional[ModelSize] = None
    quality: QualityMode = QualityMode.STANDARD
    
    # Generation parameters
    config: GenerationConfig = Field(default_factory=GenerationConfig)
    
    # User context
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Output preferences
    output_format: str = Field(default="wav", regex="^(wav|mp3|flac)$")
    include_metadata: bool = True
    
    @validator('prompt')
    def validate_prompt(cls, v):
        """Validate prompt content."""
        if not v.strip():
            raise ValueError("Prompt cannot be empty")
        
        # Basic content filtering
        forbidden_words = ["explicit", "violent", "hate"]  # Extend as needed
        lower_prompt = v.lower()
        for word in forbidden_words:
            if word in lower_prompt:
                raise ValueError(f"Prompt contains forbidden content: {word}")
        
        return v.strip()
    
    @validator('duration')
    def validate_duration(cls, v, values):
        """Validate duration based on quality mode."""
        quality = values.get('quality', QualityMode.STANDARD)
        
        # Adjust max duration based on quality
        max_durations = {
            QualityMode.FAST: 60.0,
            QualityMode.STANDARD: 120.0,
            QualityMode.HIGH: 300.0
        }
        
        max_duration = max_durations.get(quality, 120.0)
        if v > max_duration:
            raise ValueError(f"Duration {v}s exceeds maximum for {quality} quality: {max_duration}s")
        
        return v


class GenerationResponse(BaseModel):
    """Response for generation request."""
    task_id: str
    status: TaskStatus
    message: str
    estimated_duration: Optional[float] = None
    queue_position: Optional[int] = None


class GenerationTask(BaseModel):
    """Generation task data model."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    status: TaskStatus = TaskStatus.PENDING
    
    # Request data
    request: GenerationRequest
    
    # Processing information
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    result_url: Optional[str] = None
    result_metadata: Optional[Dict[str, Any]] = None
    
    # Error handling
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # Processing context
    worker_id: Optional[str] = None
    model_version: Optional[str] = None
    processing_duration: Optional[float] = None
    
    # Batch information
    batch_id: Optional[str] = None
    batch_position: Optional[int] = None
    
    # Priority and queue management
    priority: int = Field(default=0, ge=-10, le=10)
    queue_position: Optional[int] = None
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @property
    def is_completed(self) -> bool:
        """Check if task is in a completed state."""
        return self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
    
    @property
    def is_processing(self) -> bool:
        """Check if task is currently being processed."""
        return self.status == TaskStatus.PROCESSING
    
    @property
    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return (
            self.status == TaskStatus.FAILED and 
            self.retry_count < self.max_retries
        )
    
    def update_status(self, new_status: TaskStatus, message: Optional[str] = None):
        """Update task status with timestamp."""
        self.status = new_status
        
        if new_status == TaskStatus.PROCESSING and not self.started_at:
            self.started_at = datetime.utcnow()
        elif new_status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            if not self.completed_at:
                self.completed_at = datetime.utcnow()
            
            # Calculate processing duration
            if self.started_at:
                duration = (self.completed_at - self.started_at).total_seconds()
                self.processing_duration = duration
        
        if message:
            if new_status == TaskStatus.FAILED:
                self.error_message = message
            # Could extend to store status messages history


class StreamingRequest(BaseModel):
    """Request for streaming generation."""
    prompt: str = Field(..., min_length=1, max_length=1000)
    
    # Streaming parameters
    chunk_size: float = Field(default=2.0, ge=0.5, le=10.0)
    total_duration: Optional[float] = Field(default=None, ge=1.0, le=300.0)
    
    # Model selection
    model: Optional[str] = None
    quality: QualityMode = QualityMode.FAST  # Default to fast for streaming
    
    # Generation parameters
    config: GenerationConfig = Field(default_factory=GenerationConfig)
    
    # Streaming settings
    buffer_size: int = Field(default=4096, ge=1024, le=65536)
    crossfade_duration: float = Field(default=0.1, ge=0.0, le=1.0)
    
    # User context
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class BatchGenerationRequest(BaseModel):
    """Request for batch generation."""
    requests: List[GenerationRequest] = Field(..., min_items=1, max_items=100)
    
    # Batch settings
    batch_name: Optional[str] = None
    priority: int = Field(default=0, ge=-10, le=10)
    parallel_limit: int = Field(default=4, ge=1, le=20)
    
    # Output settings
    output_format: str = Field(default="wav")
    compress_results: bool = False
    
    # User context
    user_id: Optional[str] = None
    
    @validator('requests')
    def validate_batch_requests(cls, v):
        """Validate batch generation requests."""
        if len(v) > 100:
            raise ValueError("Batch size cannot exceed 100 requests")
        
        # Check for duplicate prompts in batch
        prompts = [req.prompt for req in v]
        if len(prompts) != len(set(prompts)):
            raise ValueError("Duplicate prompts found in batch")
        
        return v


class BatchGenerationResponse(BaseModel):
    """Response for batch generation request."""
    batch_id: str
    task_ids: List[str]
    total_tasks: int
    status: str
    estimated_completion: Optional[datetime] = None


class BatchStatus(BaseModel):
    """Batch generation status."""
    batch_id: str
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    pending_tasks: int
    
    status: str  # "processing", "completed", "failed", "partial"
    
    # Timing information
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results summary
    success_rate: float = 0.0
    average_duration: Optional[float] = None
    
    # Task details
    task_statuses: Dict[str, TaskStatus] = Field(default_factory=dict)
    
    @property
    def is_completed(self) -> bool:
        """Check if batch is completed."""
        return self.status in ["completed", "failed"]
    
    @property
    def progress_percentage(self) -> float:
        """Get batch completion percentage."""
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks + self.failed_tasks) / self.total_tasks * 100


class QueueStatus(BaseModel):
    """Generation queue status information."""
    total_tasks: int
    pending_tasks: int
    processing_tasks: int
    completed_tasks: int
    failed_tasks: int
    
    # Queue performance
    average_wait_time: Optional[float] = None
    average_processing_time: Optional[float] = None
    throughput_per_hour: Optional[float] = None
    
    # System health
    active_workers: int = 0
    healthy_workers: int = 0
    queue_health: str = "unknown"  # "healthy", "degraded", "critical"


class ServiceStats(BaseModel):
    """Generation service statistics."""
    # Request statistics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Performance metrics
    average_generation_time: Optional[float] = None
    p95_generation_time: Optional[float] = None
    
    # Model usage
    model_usage: Dict[str, int] = Field(default_factory=dict)
    
    # Queue statistics
    queue_stats: QueueStatus
    
    # Service health
    service_health: str = "unknown"
    uptime_seconds: float = 0.0
    
    # Resource usage
    cpu_usage_percent: Optional[float] = None
    memory_usage_percent: Optional[float] = None
    gpu_usage_percent: Optional[float] = None
    
    # Error analysis
    common_errors: Dict[str, int] = Field(default_factory=dict)
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }