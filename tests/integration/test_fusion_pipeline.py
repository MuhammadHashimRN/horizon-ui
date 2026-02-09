"""Integration tests for the multimodal fusion pipeline."""

import pytest

from horizon.decision.fusion_engine import FusionEngine
from horizon.decision.policy_engine import PolicyEngine
from horizon.event_bus import EventBus
from horizon.types import (
    ActionType,
    Event,
    EventType,
    FusedAction,
    GestureLabel,
    GestureResult,
    InputSource,
    VoiceIntent,
)


class TestFusionPipeline:
    """Test gesture/voice → fusion → policy → OS event flow."""

    def test_gesture_to_os_event(self, event_bus):
        """Test that a valid gesture result flows through to OS event."""
        fusion = FusionEngine(event_bus=event_bus)
        policy = PolicyEngine(event_bus=event_bus, confidence_threshold=0.5, cooldown_ms=0)

        os_events = []
        event_bus.subscribe(EventType.OS_EVENT, lambda e: os_events.append(e))

        gesture = GestureResult(
            label=GestureLabel.PINCH,
            confidence=0.9,
            cursor_x=0.5,
            cursor_y=0.5,
        )
        event_bus.publish(Event(type=EventType.GESTURE_RESULT, data=gesture, source="gesture_classifier"))

        assert len(os_events) == 1
        assert os_events[0].data.action == ActionType.LEFT_CLICK

    def test_voice_to_os_event(self, event_bus):
        """Test that a voice intent flows through to OS event."""
        fusion = FusionEngine(event_bus=event_bus)
        policy = PolicyEngine(event_bus=event_bus, confidence_threshold=0.5, cooldown_ms=0)

        os_events = []
        event_bus.subscribe(EventType.OS_EVENT, lambda e: os_events.append(e))

        intent = VoiceIntent(
            action=ActionType.LEFT_CLICK,
            transcript="click",
            confidence=0.95,
        )
        event_bus.publish(Event(type=EventType.VOICE_INTENT, data=intent))

        assert len(os_events) == 1

    def test_low_confidence_blocked(self, event_bus):
        """Test that low-confidence actions are blocked by policy."""
        fusion = FusionEngine(event_bus=event_bus)
        policy = PolicyEngine(event_bus=event_bus, confidence_threshold=0.9, cooldown_ms=0)

        os_events = []
        event_bus.subscribe(EventType.OS_EVENT, lambda e: os_events.append(e))

        gesture = GestureResult(
            label=GestureLabel.PINCH,
            confidence=0.5,
            cursor_x=0.5,
            cursor_y=0.5,
        )
        event_bus.publish(Event(type=EventType.GESTURE_RESULT, data=gesture, source="gesture_classifier"))

        assert len(os_events) == 0
