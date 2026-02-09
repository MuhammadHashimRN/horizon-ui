"""Shared test fixtures for Horizon UI tests."""

from __future__ import annotations

import numpy as np
import pytest

from horizon.event_bus import EventBus
from horizon.types import (
    ActionType,
    FusedAction,
    GestureLabel,
    GestureResult,
    InputSource,
    Landmark,
    LandmarkSet,
    VoiceIntent,
)


@pytest.fixture
def event_bus() -> EventBus:
    """Create a fresh EventBus for each test."""
    return EventBus()


@pytest.fixture
def sample_frame() -> np.ndarray:
    """Create a dummy 640x480 BGR frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_landmarks() -> LandmarkSet:
    """Create a sample LandmarkSet with 21 points."""
    landmarks = [
        Landmark(x=0.5 + i * 0.01, y=0.5 + i * 0.005, z=0.0)
        for i in range(21)
    ]
    return LandmarkSet(landmarks=landmarks, handedness="Right")


@pytest.fixture
def sample_gesture_result() -> GestureResult:
    """Create a sample GestureResult."""
    return GestureResult(
        label=GestureLabel.POINT,
        confidence=0.92,
        cursor_x=0.5,
        cursor_y=0.5,
    )


@pytest.fixture
def sample_voice_intent() -> VoiceIntent:
    """Create a sample VoiceIntent."""
    return VoiceIntent(
        action=ActionType.LEFT_CLICK,
        transcript="click",
        confidence=0.95,
    )


@pytest.fixture
def sample_fused_action() -> FusedAction:
    """Create a sample FusedAction."""
    return FusedAction(
        action=ActionType.LEFT_CLICK,
        source=InputSource.GESTURE,
        confidence=0.9,
        cursor_x=0.5,
        cursor_y=0.5,
    )


@pytest.fixture
def sample_audio_bytes() -> bytes:
    """Create sample 16kHz int16 PCM audio (0.5 seconds of silence)."""
    samples = np.zeros(8000, dtype=np.int16)
    return samples.tobytes()
