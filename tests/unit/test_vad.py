"""Unit tests for voice activity detection."""

import numpy as np
import pytest

from horizon.event_bus import EventBus
from horizon.types import Event, EventType


class TestVAD:
    def test_init(self, event_bus):
        from horizon.perception.vad import VAD
        vad = VAD(event_bus=event_bus)
        assert vad.sample_rate == 16000

    def test_reset(self, event_bus):
        from horizon.perception.vad import VAD
        vad = VAD(event_bus=event_bus)
        vad.reset()
        assert not vad._triggered
