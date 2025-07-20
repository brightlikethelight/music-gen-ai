"""
Metrics collection and monitoring utilities.

Provides Prometheus metrics for monitoring MusicGen performance and usage.
"""

import time
from typing import Dict, Any
from functools import wraps

try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class MetricsCollector:
    """Collects and exposes metrics for monitoring."""

    def __init__(self):
        if not PROMETHEUS_AVAILABLE:
            self.enabled = False
            return

        self.enabled = True
        self.registry = CollectorRegistry()

        # Generation metrics
        self.generation_requests = Counter(
            "musicgen_generation_requests_total",
            "Total number of generation requests",
            registry=self.registry,
        )

        self.generation_completed = Counter(
            "musicgen_generation_completed_total",
            "Total number of completed generations",
            registry=self.registry,
        )

        self.generation_failed = Counter(
            "musicgen_generation_failed_total",
            "Total number of failed generations",
            registry=self.registry,
        )

        self.generation_duration = Histogram(
            "musicgen_generation_duration_seconds",
            "Time spent generating music",
            ["model"],
            registry=self.registry,
        )

        self.audio_duration_generated = Histogram(
            "musicgen_audio_duration_generated_seconds",
            "Duration of audio generated",
            ["model"],
            registry=self.registry,
        )

        # System metrics
        self.active_generations = Gauge(
            "musicgen_active_generations",
            "Number of currently active generations",
            registry=self.registry,
        )

        self.model_load_time = Histogram(
            "musicgen_model_load_duration_seconds",
            "Time spent loading models",
            ["model"],
            registry=self.registry,
        )

    def record_generation_request(self, model: str, status: str):
        """Record a generation request."""
        if self.enabled:
            self.generation_requests.labels(model=model, status=status).inc()

    def record_generation_duration(self, model: str, duration: float):
        """Record generation timing."""
        if self.enabled:
            self.generation_duration.labels(model=model).observe(duration)

    def record_audio_duration(self, model: str, duration: float):
        """Record generated audio duration."""
        if self.enabled:
            self.audio_duration_generated.labels(model=model).observe(duration)

    def inc_active_generations(self):
        """Increment active generation counter."""
        if self.enabled:
            self.active_generations.inc()

    def dec_active_generations(self):
        """Decrement active generation counter."""
        if self.enabled:
            self.active_generations.dec()

    def record_model_load_time(self, model: str, duration: float):
        """Record model loading time."""
        if self.enabled:
            self.model_load_time.labels(model=model).observe(duration)

    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        if not self.enabled:
            return ""
        return generate_latest(self.registry).decode("utf-8")


# Global metrics instance
metrics = MetricsCollector()


def track_generation_time(model: str):
    """Decorator to track generation timing."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not metrics.enabled:
                return func(*args, **kwargs)

            start_time = time.time()
            metrics.inc_active_generations()

            try:
                result = func(*args, **kwargs)
                metrics.record_generation_request(model, "success")
                return result
            except Exception as e:
                metrics.record_generation_request(model, "error")
                raise
            finally:
                duration = time.time() - start_time
                metrics.record_generation_duration(model, duration)
                metrics.dec_active_generations()

        return wrapper

    return decorator


def track_model_loading(model: str):
    """Decorator to track model loading time."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not metrics.enabled:
                return func(*args, **kwargs)

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                metrics.record_model_load_time(model, duration)

        return wrapper

    return decorator
