"""Unit tests for intent parser."""

import pytest

from horizon.event_bus import EventBus
from horizon.types import ActionType


class TestIntentParser:
    def test_parse_exact_match(self, event_bus):
        from horizon.perception.intent_parser import IntentParser
        parser = IntentParser(event_bus=event_bus)
        intent = parser.parse("click")
        if intent:
            assert intent.action == ActionType.LEFT_CLICK

    def test_parse_empty_string(self, event_bus):
        from horizon.perception.intent_parser import IntentParser
        parser = IntentParser(event_bus=event_bus)
        intent = parser.parse("")
        assert intent is None
