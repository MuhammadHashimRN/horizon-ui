"""Unit tests for frame capture."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from horizon.capture.frame_capture import FrameCapture
from horizon.event_bus import EventBus

class TestFrameCapture:
    def test_init(self, event_bus):
        fc = FrameCapture(event_bus=event_bus, device_index=0)
        assert fc.device_index == 0
        assert fc.resolution == (640, 480)
        assert fc.fps == 30
        assert not fc.is_running

    def test_get_frame_empty_queue(self, event_bus):
        fc = FrameCapture(event_bus=event_bus)
        assert fc.get_frame() is None

    @patch("cv2.VideoCapture")
    def test_stop(self, mock_cap, event_bus):
        fc = FrameCapture(event_bus=event_bus)
        fc.stop()
        assert not fc.is_running
