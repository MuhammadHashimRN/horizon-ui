"""Unit tests for policy engine."""

import pytest

from horizon.decision.policy_engine import PolicyEngine
from horizon.event_bus import EventBus
from horizon.types import ActionType, Event, EventType, FusedAction, InputSource


class TestPolicyEngine:
    def test_low_confidence_rejected(self, event_bus):
        pe = PolicyEngine(event_bus=event_bus, confidence_threshold=0.8)
        events = []
        event_bus.subscribe(EventType.OS_EVENT, lambda e: events.append(e))

        action = FusedAction(action=ActionType.LEFT_CLICK, confidence=0.5, source=InputSource.GESTURE)
        event_bus.publish(Event(type=EventType.FUSED_ACTION, data=action, source="fusion_engine"))
        assert len(events) == 0

    def test_high_confidence_accepted(self, event_bus):
        pe = PolicyEngine(event_bus=event_bus, confidence_threshold=0.5)
        events = []
        event_bus.subscribe(EventType.OS_EVENT, lambda e: events.append(e))

        action = FusedAction(action=ActionType.LEFT_CLICK, confidence=0.9, source=InputSource.GESTURE)
        event_bus.publish(Event(type=EventType.FUSED_ACTION, data=action, source="fusion_engine"))
        assert len(events) == 1
