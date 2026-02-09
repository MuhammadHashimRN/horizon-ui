"""Unit tests for landmark estimator."""

import pytest

from horizon.event_bus import EventBus
from horizon.perception.landmark_estimator import LandmarkEstimator
from horizon.types import Event, EventType


class TestLandmarkEstimator:
    def test_init(self, event_bus):
        le = LandmarkEstimator(event_bus=event_bus)
        assert le is not None

    def test_no_hands_no_publish(self, event_bus):
        le = LandmarkEstimator(event_bus=event_bus)
        events = []
        event_bus.subscribe(EventType.LANDMARKS, lambda e: events.append(e))

        event_bus.publish(Event(
            type=EventType.HAND_DETECTED,
            data={"has_hands": False, "results": None, "frame": None},
        ))
        assert len(events) == 0
