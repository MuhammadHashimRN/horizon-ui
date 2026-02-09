"""Unit tests for gesture classifier."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from horizon.event_bus import EventBus
from horizon.perception.gesture_classifier import GestureClassifier
from horizon.types import GestureLabel


class TestGestureClassifier:
    def test_init(self, event_bus):
        gc = GestureClassifier(event_bus=event_bus)
        assert gc.confidence_threshold == 0.75

    def test_classify_no_model(self, event_bus):
        gc = GestureClassifier(event_bus=event_bus, model_path="nonexistent.onnx")
        features = np.random.randn(100).astype(np.float32)
        result = gc.classify(features)
        assert result.label == GestureLabel.NONE
        assert result.confidence == 0.0
