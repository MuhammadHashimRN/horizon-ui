"""Unit tests for fusion engine."""

import pytest

from horizon.event_bus import EventBus
from horizon.types import Event, EventType, GestureLabel, GestureResult


class TestFusionEngine:
    def test_init(self, event_bus):
        from horizon.decision.fusion_engine import FusionEngine
        fe = FusionEngine(event_bus=event_bus)
        assert fe is not None

    def test_none_gesture_ignored(self, event_bus):
        from horizon.decision.fusion_engine import FusionEngine
        fe = FusionEngine(event_bus=event_bus)
        events = []
        event_bus.subscribe(EventType.FUSED_ACTION, lambda e: events.append(e))

        result = GestureResult(label=GestureLabel.NONE, confidence=0.5)
        event_bus.publish(Event(type=EventType.GESTURE_RESULT, data=result))
        assert len(events) == 0
