"""Performance tests for resource usage requirements."""

import sys

import numpy as np
import pytest


class TestResourceUsage:
    """Verify resource usage meets SRS requirements."""

    def test_landmark_set_memory(self):
        """A LandmarkSet should use minimal memory."""
        from horizon.types import Landmark, LandmarkSet

        landmarks = [Landmark(x=0.5, y=0.5, z=0.0) for _ in range(21)]
        ls = LandmarkSet(landmarks=landmarks)

        size = sys.getsizeof(ls) + sum(sys.getsizeof(lm) for lm in landmarks)
        assert size < 4096, f"LandmarkSet uses {size} bytes, expected <4KB"

    def test_event_bus_memory_with_many_subscribers(self):
        """EventBus should handle many subscribers without excessive memory."""
        from horizon.event_bus import EventBus
        from horizon.types import EventType

        bus = EventBus()
        for _ in range(100):
            bus.subscribe(EventType.FRAME, lambda e: None)

        # Should not raise or use excessive memory
        assert len(bus._subscribers[EventType.FRAME]) == 100

    def test_feature_buffer_bounded(self):
        """Feature extractor buffer should be bounded by temporal window."""
        from horizon.perception.feature_extractor import FeatureExtractor
        from horizon.event_bus import EventBus
        from horizon.types import Landmark, LandmarkSet

        bus = EventBus()
        fe = FeatureExtractor(event_bus=bus, temporal_window=15)

        landmarks = LandmarkSet(
            landmarks=[Landmark(x=0.5, y=0.5, z=0.0) for _ in range(21)]
        )

        # Add more frames than the window
        for _ in range(50):
            fe._buffer.append(landmarks)

        assert len(fe._buffer) == 15, "Buffer should be bounded by temporal window"
