"""Unit tests for ASR backends."""

import pytest


class TestASR:
    def test_unknown_backend(self, event_bus):
        from horizon.perception.asr import ASR
        asr = ASR(event_bus=event_bus, backend="unknown")
        assert asr._backend is None
