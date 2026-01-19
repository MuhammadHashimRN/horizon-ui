"""Multimodal fusion engine combining gesture and voice signals."""

from __future__ import annotations

import logging
import time
from collections import deque

from horizon.constants import FUSION_ALIGNMENT_WINDOW_MS
from horizon.event_bus import EventBus
from horizon.types import (
    ActionType,
    Event,
    EventType,
    FusedAction,
    GestureLabel,
    GestureResult,
    InputSource,
    VoiceIntent,
)

logger = logging.getLogger(__name__)

# Default gesture label → action mapping
GESTURE_TO_ACTION: dict[GestureLabel, ActionType] = {
    GestureLabel.POINT: ActionType.MOUSE_MOVE,
    GestureLabel.PINCH: ActionType.LEFT_CLICK,
    GestureLabel.FIST: ActionType.RIGHT_CLICK,
    GestureLabel.OPEN_PALM: ActionType.PAUSE_TRACKING,
    GestureLabel.SWIPE_LEFT: ActionType.KEY_COMBO,
    GestureLabel.SWIPE_RIGHT: ActionType.KEY_COMBO,
    GestureLabel.PINCH_SPREAD: ActionType.ZOOM_IN,
    GestureLabel.PINCH_CLOSE: ActionType.ZOOM_OUT,
    GestureLabel.TWO_FINGER_SCROLL: ActionType.SCROLL,
    GestureLabel.THUMBS_UP: ActionType.CONFIRM,
}


class FusionEngine:
    """Fuses gesture and voice input signals using temporal alignment.

    Subscribes to GESTURE_RESULT and VOICE_INTENT events. Applies
    a temporal alignment window to detect simultaneous inputs and
    publishes FUSED_ACTION events.
    """

    def __init__(
        self,
        event_bus: EventBus,
        alignment_window_ms: float = FUSION_ALIGNMENT_WINDOW_MS,
    ) -> None:
        self.event_bus = event_bus
        self._alignment_window_s = alignment_window_ms / 1000.0

        self._recent_gestures: deque[GestureResult] = deque(maxlen=10)
        self._recent_voice: deque[VoiceIntent] = deque(maxlen=5)

        self.event_bus.subscribe(EventType.GESTURE_RESULT, self._on_gesture)
        self.event_bus.subscribe(EventType.VOICE_INTENT, self._on_voice)
        logger.info("FusionEngine initialized (window=%.0fms)", alignment_window_ms)

    def _on_gesture(self, event: Event) -> None:
        data = event.data
        # Only process classified GestureResult objects, not raw feature dicts
        if not isinstance(data, GestureResult):
            return
        if data.label == GestureLabel.NONE:
            return

        self._recent_gestures.append(data)

        # Check for concurrent voice intent
        voice_intent = self._find_aligned_voice(data.timestamp)

        if voice_intent:
            # Multimodal fusion — voice takes priority
            fused = FusedAction(
                action=voice_intent.action,
                source=InputSource.FUSED,
                confidence=(data.confidence + voice_intent.confidence) / 2,
                params=voice_intent.params,
                cursor_x=data.cursor_x,
                cursor_y=data.cursor_y,
            )
        else:
            # Gesture only
            action = GESTURE_TO_ACTION.get(data.label)
            if action is None:
                return

            params: dict = {}
            if data.label == GestureLabel.SWIPE_LEFT:
                params = {"keys": ["alt", "left"]}
            elif data.label == GestureLabel.SWIPE_RIGHT:
                params = {"keys": ["alt", "right"]}

            fused = FusedAction(
                action=action,
                source=InputSource.GESTURE,
                confidence=data.confidence,
                params=params,
                cursor_x=data.cursor_x,
                cursor_y=data.cursor_y,
            )

        self.event_bus.publish(Event(
            type=EventType.FUSED_ACTION,
            data=fused,
            source="fusion_engine",
        ))

    def _on_voice(self, event: Event) -> None:
        intent: VoiceIntent = event.data
        self._recent_voice.append(intent)

        # Check for concurrent gesture
        gesture = self._find_aligned_gesture(intent.timestamp)

        if gesture:
            # Already handled in _on_gesture
            return

        # Voice only
        fused = FusedAction(
            action=intent.action,
            source=InputSource.VOICE,
            confidence=intent.confidence,
            params=intent.params,
        )

        self.event_bus.publish(Event(
            type=EventType.FUSED_ACTION,
            data=fused,
            source="fusion_engine",
        ))

    def _find_aligned_voice(self, timestamp: float) -> VoiceIntent | None:
        for intent in reversed(self._recent_voice):
            if abs(intent.timestamp - timestamp) <= self._alignment_window_s:
                return intent
        return None

    def _find_aligned_gesture(self, timestamp: float) -> GestureResult | None:
        for gesture in reversed(self._recent_gestures):
            if abs(gesture.timestamp - timestamp) <= self._alignment_window_s:
                return gesture
        return None

    def close(self) -> None:
        self.event_bus.unsubscribe(EventType.GESTURE_RESULT, self._on_gesture)
        self.event_bus.unsubscribe(EventType.VOICE_INTENT, self._on_voice)
        logger.info("FusionEngine closed")
