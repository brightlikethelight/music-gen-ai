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
            self._setup_placeholder_metrics()
            return

        self.enabled = True
        self.registry = CollectorRegistry()
        
        # Track counts for summary
        self._prometheus_counts = {
            "generation_requests": 0,
            "generation_completed": 0,
            "generation_failed": 0,
            "active_generations": 0,
        }

        # Generation metrics
        self.generation_requests = Counter(
            "musicgen_generation_requests_total",
            "Total number of generation requests",
            ["model", "status"],  # Add label names
            registry=self.registry,
        )

        self.generation_completed = Counter(
            "musicgen_generation_completed_total",
            "Total number of completed generations",
            ["model"],  # Add label names
            registry=self.registry,
        )

        self.generation_failed = Counter(
            "musicgen_generation_failed_total",
            "Total number of failed generations",
            ["model"],  # Add label names
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

    def _setup_placeholder_metrics(self):
        """Setup placeholder metrics when prometheus is not available."""
        # Track counts manually when prometheus is not available
        self._mock_counts = {
            "generation_requests": 0,
            "generation_completed": 0,
            "generation_failed": 0,
            "active_generations": 0,
        }
        
        # Create mock metrics that support the same interface
        class MockMetric:
            def __init__(self, parent, name=None):
                self.parent = parent
                self.name = name
                # Create a mock _value object that mimics prometheus Counter structure
                self._value = MockValue()
            
            def inc(self):
                if self.name and self.name in self.parent._mock_counts:
                    self.parent._mock_counts[self.name] += 1
            
            def dec(self):
                if self.name and self.name in self.parent._mock_counts:
                    self.parent._mock_counts[self.name] = max(0, self.parent._mock_counts[self.name] - 1)
            
            def labels(self, **kwargs):
                return self
                
            def observe(self, value):
                pass
        
        class MockValue:
            def __init__(self):
                self._value = 0

        # Setup the same attributes that would be created with prometheus
        self.generation_requests = MockMetric(self, "generation_requests")
        self.generation_completed = MockMetric(self, "generation_completed")
        self.generation_failed = MockMetric(self, "generation_failed")
        self.generation_duration = MockMetric(self)
        self.audio_duration_generated = MockMetric(self)
        self.active_generations = MockMetric(self, "active_generations")
        self.model_load_time = MockMetric(self)

    def record_generation_request(self, model: str, status: str):
        """Record a generation request."""
        if self.enabled:
            self.generation_requests.labels(model=model, status=status).inc()
            # Update summary counts
            if status == "queued":
                self._prometheus_counts["generation_requests"] += 1
            elif status == "completed":
                self._prometheus_counts["generation_completed"] += 1
            elif status == "failed":
                self._prometheus_counts["generation_failed"] += 1
        else:
            # Update mock counts based on status
            if status == "queued":
                self._mock_counts["generation_requests"] += 1
            elif status == "completed":
                self._mock_counts["generation_completed"] += 1
            elif status == "failed":
                self._mock_counts["generation_failed"] += 1

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
            self._prometheus_counts["active_generations"] += 1
        else:
            self._mock_counts["active_generations"] += 1

    def dec_active_generations(self):
        """Decrement active generation counter."""
        if self.enabled:
            self.active_generations.dec()
            self._prometheus_counts["active_generations"] = max(0, self._prometheus_counts["active_generations"] - 1)
        else:
            self._mock_counts["active_generations"] = max(0, self._mock_counts["active_generations"] - 1)

    def record_model_load_time(self, model: str, duration: float):
        """Record model loading time."""
        if self.enabled:
            self.model_load_time.labels(model=model).observe(duration)

    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        if not self.enabled:
            return ""
        return generate_latest(self.registry).decode("utf-8")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of metrics as a dictionary."""
        if not self.enabled:
            # Return mock counts when prometheus is not available
            return self._mock_counts.copy()
        
        # For prometheus metrics, we need to track counts separately
        # since prometheus counters with labels don't provide easy totals
        if not hasattr(self, '_prometheus_counts'):
            self._prometheus_counts = {
                "generation_requests": 0,
                "generation_completed": 0,
                "generation_failed": 0,
                "active_generations": 0,
            }
            
        return self._prometheus_counts.copy()


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
