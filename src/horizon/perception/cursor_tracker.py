"""Decoupled cursor tracker - updates cursor on every frame where a hand is visible."""

from __future__ import annotations

import logging

from horizon.event_bus import EventBus
from horizon.perception.coordinate_mapper import CoordinateMapper
from horizon.types import (
    ActionType,
    Event,
    EventType,
    FusedAction,
    InputSource,
)

logger = logging.getLogger(__name__)

INDEX_TIP = 8
INDEX_MCP = 5  # knuckle - more stable than fingertip
# Blend: 80% tip (responsive) + 20% knuckle (stable)
TIP_WEIGHT = 0.8
MCP_WEIGHT = 0.2

# Gestures that mean "I'm pointing / moving cursor"
_CURSOR_GESTURES = {"Pointing_Up", "None"}

# Minimum score to trust the gesture classification for freezing
_FREEZE_SCORE_THRESHOLD = 0.55


class CursorTracker:
    """Tracks index fingertip and moves cursor on every frame.

    Decoupled from gesture classification. Cursor moves when the hand
    is pointing or neutral. When a command gesture is detected (fist,
    thumbs up, etc.), the cursor freezes at its last position so the
    gesture action executes in the right place.
    """

    def __init__(
        self,
        event_bus: EventBus,
        coordinate_mapper: CoordinateMapper,
    ) -> None:
        self.event_bus = event_bus
        self._mapper = coordinate_mapper
        self._last_screen_x: int | None = None
        self._last_screen_y: int | None = None

        self.event_bus.subscribe(EventType.HAND_DETECTED, self._on_hand_detected)
        logger.info("CursorTracker initialized")

    def _on_hand_detected(self, event: Event) -> None:
        data = event.data
        if not data["has_hands"]:
            self._mapper.reset()
            self._last_screen_x = None
            self._last_screen_y = None
            return

        # Check if this is a command gesture (fist, thumbs up, etc.)
        # If so, freeze cursor at last position - don't track the curling finger
        gesture_labels = data.get("gesture_labels", [])
        if gesture_labels:
            top_label = gesture_labels[0].get("label", "None")
            top_score = gesture_labels[0].get("score", 0.0)
            if top_label not in _CURSOR_GESTURES and top_score > _FREEZE_SCORE_THRESHOLD:
                # Command gesture detected - cursor stays where it was
                return

        # Extract index fingertip position
        result_obj = data["result"]
        if not result_obj.hand_landmarks:
            return

        lms = result_obj.hand_landmarks[0]
        if len(lms) <= INDEX_TIP:
            return

        # Blend tip + knuckle for stability (tip jitters most)
        tip_x, tip_y = lms[INDEX_TIP].x, lms[INDEX_TIP].y
        mcp_x, mcp_y = lms[INDEX_MCP].x, lms[INDEX_MCP].y
        norm_x = TIP_WEIGHT * tip_x + MCP_WEIGHT * mcp_x
        norm_y = TIP_WEIGHT * tip_y + MCP_WEIGHT * mcp_y

        # Map through CoordinateMapper (smooths, converts to pixels)
        screen_x, screen_y = self._mapper.map(norm_x, norm_y)

        # Skip if position unchanged
        if screen_x == self._last_screen_x and screen_y == self._last_screen_y:
            return

        self._last_screen_x = screen_x
        self._last_screen_y = screen_y

        # Publish OS_EVENT directly for minimum latency
        # cursor_x/cursor_y > 1.0 tells OSEventInjector to use pixel values directly
        self.event_bus.publish(Event(
            type=EventType.OS_EVENT,
            data=FusedAction(
                action=ActionType.MOUSE_MOVE,
                source=InputSource.GESTURE,
                confidence=1.0,
                cursor_x=float(screen_x),
                cursor_y=float(screen_y),
            ),
            source="cursor_tracker",
        ))

    def close(self) -> None:
        self.event_bus.unsubscribe(EventType.HAND_DETECTED, self._on_hand_detected)
        logger.info("CursorTracker closed")
