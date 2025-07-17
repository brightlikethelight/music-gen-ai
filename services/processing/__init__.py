"""
Audio Processing Service

Dedicated microservice for audio manipulation, effects processing,
format conversion, and audio enhancement operations.
"""

from .app import create_processing_app
from .service import AudioProcessingService
from .models import (
    ProcessingRequest,
    ProcessingResponse,
    AudioFormat,
    EffectType,
    ProcessingTask,
    EnhancementRequest,
    ConversionRequest,
    MixingRequest
)
from .config import ProcessingServiceConfig

__all__ = [
    "create_processing_app",
    "AudioProcessingService", 
    "ProcessingRequest",
    "ProcessingResponse",
    "AudioFormat",
    "EffectType",
    "ProcessingTask",
    "EnhancementRequest",
    "ConversionRequest",
    "MixingRequest",
    "ProcessingServiceConfig"
]