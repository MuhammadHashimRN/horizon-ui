"""Unit tests for action mapper."""

import pytest

from horizon.decision.action_mapper import ActionMapper
from horizon.types import ActionType, FusedAction, InputSource


class TestActionMapper:
    def test_map_basic_action(self):
        mapper = ActionMapper()
        action = FusedAction(
            action=ActionType.LEFT_CLICK,
            source=InputSource.GESTURE,
            confidence=0.9,
            cursor_x=100,
            cursor_y=200,
        )
        os_event = mapper.map(action)
        assert os_event.action == ActionType.LEFT_CLICK
        assert os_event.x == 100
        assert os_event.y == 200
