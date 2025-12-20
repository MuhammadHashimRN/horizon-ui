"""Extracts and normalizes 21-point hand landmarks from MediaPipe results."""

from __future__ import annotations

import logging

from horizon.event_bus import EventBus
from horizon.types import Event, EventType, Landmark, LandmarkSet

logger = logging.getLogger(__name__)


class LandmarkEstimator:
    """Converts MediaPipe HandLandmarkerResult into normalized LandmarkSet.

    Subscribes to HAND_DETECTED events and publishes LANDMARKS events
    containing a LandmarkSet with 21 normalized (x, y, z) points.

    Compatible with the MediaPipe tasks API (HandLandmarkerResult).
    """

    def __init__(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self.event_bus.subscribe(EventType.HAND_DETECTED, self._on_hand_detected)
        logger.info("LandmarkEstimator initialized")

    def _on_hand_detected(self, event: Event) -> None:
        data = event.data
        if not data["has_hands"]:
            return

        result = data["result"]

        for idx, hand_landmarks in enumerate(result.hand_landmarks):
            # Determine handedness
            handedness = "Right"
            if result.handedness and idx < len(result.handedness):
                handedness = result.handedness[idx][0].category_name

            landmarks = [
                Landmark(x=lm.x, y=lm.y, z=lm.z)
                for lm in hand_landmarks
            ]

            landmark_set = LandmarkSet(
                landmarks=landmarks,
                handedness=handedness,
            )

            self.event_bus.publish(Event(
                type=EventType.LANDMARKS,
                data=landmark_set,
                source="landmark_estimator",
            ))

    def close(self) -> None:
        self.event_bus.unsubscribe(EventType.HAND_DETECTED, self._on_hand_detected)
        logger.info("LandmarkEstimator closed")
