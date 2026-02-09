"""Performance tests for latency requirements."""

import time

import numpy as np
import pytest

from horizon.event_bus import EventBus
from horizon.perception.smoothing import CursorSmoother, EMAFilter


class TestLatency:
    """Verify that processing meets SRS latency requirements."""

    def test_ema_filter_latency(self):
        """EMA filter should process in <1ms."""
        f = EMAFilter(alpha=0.3)
        f.update(0.0, 0.0)

        start = time.perf_counter()
        for _ in range(1000):
            f.update(np.random.rand(), np.random.rand())
        elapsed_ms = (time.perf_counter() - start) * 1000

        avg_ms = elapsed_ms / 1000
        assert avg_ms < 1.0, f"EMA filter avg latency {avg_ms:.3f}ms exceeds 1ms"

    def test_event_bus_publish_latency(self):
        """EventBus publish should complete in <1ms."""
        from horizon.types import Event, EventType

        bus = EventBus()
        received = []
        bus.subscribe(EventType.FRAME, lambda e: received.append(e))

        start = time.perf_counter()
        for _ in range(1000):
            bus.publish(Event(type=EventType.FRAME, data=None))
        elapsed_ms = (time.perf_counter() - start) * 1000

        avg_ms = elapsed_ms / 1000
        assert avg_ms < 1.0, f"EventBus publish avg latency {avg_ms:.3f}ms exceeds 1ms"
        assert len(received) == 1000

    def test_feature_extraction_latency(self):
        """Feature extraction from 21 landmarks should complete quickly."""
        from horizon.perception.feature_extractor import FeatureExtractor
        from horizon.types import Landmark, LandmarkSet

        bus = EventBus()
        fe = FeatureExtractor(event_bus=bus, temporal_window=15)

        landmarks = LandmarkSet(
            landmarks=[Landmark(x=np.random.rand(), y=np.random.rand(), z=0.0) for _ in range(21)]
        )

        # Fill buffer
        for _ in range(14):
            fe._buffer.append(landmarks)

        start = time.perf_counter()
        fe._buffer.append(landmarks)
        features = fe._extract_features()
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 10.0, f"Feature extraction took {elapsed_ms:.1f}ms, target <10ms"
        assert features is not None
