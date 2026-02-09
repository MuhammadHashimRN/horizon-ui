"""Unit tests for feature extractor."""

import pytest

from horizon.event_bus import EventBus
from horizon.perception.feature_extractor import FeatureExtractor
from horizon.types import Event, EventType, Landmark, LandmarkSet


class TestFeatureExtractor:
    def test_init(self, event_bus):
        fe = FeatureExtractor(event_bus=event_bus, temporal_window=5)
        assert fe.temporal_window == 5

    def test_buffer_not_full_no_publish(self, event_bus):
        fe = FeatureExtractor(event_bus=event_bus, temporal_window=5)
        events = []
        event_bus.subscribe(EventType.GESTURE_RESULT, lambda e: events.append(e))

        landmarks = LandmarkSet(
            landmarks=[Landmark(x=0.5, y=0.5, z=0.0) for _ in range(21)]
        )
        for _ in range(3):
            event_bus.publish(Event(type=EventType.LANDMARKS, data=landmarks))

        assert len(events) == 0

    def test_reset(self, event_bus):
        fe = FeatureExtractor(event_bus=event_bus, temporal_window=5)
        fe.reset()
        assert len(fe._buffer) == 0
