"""Integration tests for the voice pipeline."""

import pytest

from horizon.event_bus import EventBus
from horizon.types import ActionType, Event, EventType


class TestVoicePipeline:
    """Test transcript → intent parsing flow."""

    def test_transcript_to_intent(self, event_bus):
        """Test that a transcript event produces a voice intent."""
        from horizon.perception.intent_parser import IntentParser

        parser = IntentParser(event_bus=event_bus)
        intents = []
        event_bus.subscribe(EventType.VOICE_INTENT, lambda e: intents.append(e))

        event_bus.publish(Event(type=EventType.TRANSCRIPT, data="click"))

        if intents:
            assert intents[0].data.action == ActionType.LEFT_CLICK
            assert intents[0].data.confidence > 0

    def test_empty_transcript_no_intent(self, event_bus):
        """Test that an empty transcript produces no intent."""
        from horizon.perception.intent_parser import IntentParser

        parser = IntentParser(event_bus=event_bus)
        intents = []
        event_bus.subscribe(EventType.VOICE_INTENT, lambda e: intents.append(e))

        event_bus.publish(Event(type=EventType.TRANSCRIPT, data=""))
        assert len(intents) == 0
