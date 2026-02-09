"""Unit tests for audio capture."""

from unittest.mock import MagicMock, patch

import pytest

from horizon.capture.audio_capture import AudioCapture
from horizon.event_bus import EventBus


class TestAudioCapture:
    def test_init(self, event_bus):
        ac = AudioCapture(event_bus=event_bus)
        assert ac.sample_rate == 16000
        assert ac.channels == 1
        assert not ac.is_running

    def test_stop_without_start(self, event_bus):
        ac = AudioCapture(event_bus=event_bus)
        ac.stop()
        assert not ac.is_running
