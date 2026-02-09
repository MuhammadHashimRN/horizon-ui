"""Unit tests for hand detector."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from horizon.event_bus import EventBus
from horizon.types import Event, EventType


class TestHandDetector:
    @patch("mediapipe.solutions.hands.Hands")
    def test_init(self, mock_hands, event_bus):
        from horizon.perception.hand_detector import HandDetector
        hd = HandDetector(event_bus=event_bus)
        assert mock_hands.called

    @patch("mediapipe.solutions.hands.Hands")
    def test_on_frame_publishes_event(self, mock_hands, event_bus):
        from horizon.perception.hand_detector import HandDetector

        mock_result = MagicMock()
        mock_result.multi_hand_landmarks = None
        mock_hands.return_value.process.return_value = mock_result

        hd = HandDetector(event_bus=event_bus)
        events = []
        event_bus.subscribe(EventType.HAND_DETECTED, lambda e: events.append(e))

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        event_bus.publish(Event(type=EventType.FRAME, data=frame))

        assert len(events) == 1
        assert not events[0].data["has_hands"]
