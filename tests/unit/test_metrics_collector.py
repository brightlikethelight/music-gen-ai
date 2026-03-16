"""Tests for metrics collection."""

from unittest.mock import patch

import pytest

from musicgen.infrastructure.monitoring.metrics import MetricsCollector

pytestmark = pytest.mark.unit


class TestMetricsCollector:
    """Test metrics collection with MockMetric fallback path."""

    def _make_collector(self):
        """Create a MetricsCollector with prometheus disabled."""
        with patch("musicgen.infrastructure.monitoring.metrics.PROMETHEUS_AVAILABLE", False):
            return MetricsCollector()

    def test_placeholder_metrics_setup(self):
        """Collector uses mock metrics when prometheus is unavailable."""
        mc = self._make_collector()
        assert mc.enabled is False
        assert hasattr(mc, "_mock_counts")
        assert mc._mock_counts["generation_requests"] == 0

    def test_mock_metric_labels_returns_self(self):
        """MockMetric.labels() returns self for chaining."""
        mc = self._make_collector()
        assert (
            mc.generation_requests.labels(model="small", status="queued") is mc.generation_requests
        )

    def test_record_generation_request_queued(self):
        mc = self._make_collector()
        mc.record_generation_request("small", "queued")
        assert mc._mock_counts["generation_requests"] == 1

    def test_record_generation_request_completed(self):
        mc = self._make_collector()
        mc.record_generation_request("small", "completed")
        assert mc._mock_counts["generation_completed"] == 1

    def test_record_generation_request_failed(self):
        mc = self._make_collector()
        mc.record_generation_request("small", "failed")
        assert mc._mock_counts["generation_failed"] == 1

    def test_record_generation_request_unknown_status_noop(self):
        mc = self._make_collector()
        mc.record_generation_request("small", "unknown")
        assert all(v == 0 for v in mc._mock_counts.values())

    def test_inc_active_generations(self):
        mc = self._make_collector()
        mc.inc_active_generations()
        mc.inc_active_generations()
        assert mc._mock_counts["active_generations"] == 2

    def test_dec_active_generations(self):
        mc = self._make_collector()
        mc.inc_active_generations()
        mc.dec_active_generations()
        assert mc._mock_counts["active_generations"] == 0

    def test_dec_active_generations_floor_at_zero(self):
        mc = self._make_collector()
        mc.dec_active_generations()
        assert mc._mock_counts["active_generations"] == 0

    def test_get_metrics_summary_returns_copy(self):
        mc = self._make_collector()
        mc.record_generation_request("small", "queued")
        mc.inc_active_generations()
        summary = mc.get_metrics_summary()
        assert summary == {
            "generation_requests": 1,
            "generation_completed": 0,
            "generation_failed": 0,
            "active_generations": 1,
        }
        # Must be a copy, not the internal dict
        summary["generation_requests"] = 999
        assert mc._mock_counts["generation_requests"] == 1

    def test_get_metrics_returns_empty_when_disabled(self):
        mc = self._make_collector()
        assert mc.get_metrics() == ""

    def test_mock_observe_is_noop(self):
        """MockMetric.observe() doesn't raise."""
        mc = self._make_collector()
        mc.generation_duration.labels(model="small").observe(1.5)
        mc.record_generation_duration("small", 2.0)
        mc.record_audio_duration("small", 5.0)
        mc.record_model_load_time("small", 0.3)
