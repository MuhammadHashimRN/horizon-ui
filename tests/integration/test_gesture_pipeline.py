"""Integration tests for the full gesture pipeline."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from horizon.event_bus import EventBus
from horizon.types import Event, EventType


class TestGesturePipeline:
    """End-to-end test: frame → hand detection → landmarks → features → classification."""

    @patch("mediapipe.solutions.hands.Hands")
    def test_frame_to_hand_detected(self, mock_hands, event_bus):
        """Test that a frame event triggers hand detection."""
        from horizon.perception.hand_detector import HandDetector

        mock_result = MagicMock()
        mock_result.multi_hand_landmarks = None
        mock_hands.return_value.process.return_value = mock_result

        detector = HandDetector(event_bus=event_bus)
        events = []
        event_bus.subscribe(EventType.HAND_DETECTED, lambda e: events.append(e))

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        event_bus.publish(Event(type=EventType.FRAME, data=frame))

        assert len(events) == 1
        assert events[0].data["has_hands"] is False

    def test_landmarks_to_features(self, event_bus, sample_landmarks):
        """Test that landmarks flow into feature extraction."""
        from horizon.perception.feature_extractor import FeatureExtractor

        fe = FeatureExtractor(event_bus=event_bus, temporal_window=3)
        events = []
        event_bus.subscribe(EventType.GESTURE_RESULT, lambda e: events.append(e))

        # Send enough landmark frames to fill the temporal window
        for _ in range(3):
            event_bus.publish(Event(type=EventType.LANDMARKS, data=sample_landmarks))

        assert len(events) >= 1
        assert "features" in events[0].data
        assert events[0].data["needs_classification"] is True
