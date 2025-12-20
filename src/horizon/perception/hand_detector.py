"""MediaPipe GestureRecognizer wrapper — detects hands, landmarks, AND gestures."""

from __future__ import annotations

import logging
import os

import mediapipe as mp
import numpy as np

from horizon.constants import (
    DEFAULT_DETECTION_CONFIDENCE,
    DEFAULT_MAX_HANDS,
    DEFAULT_TRACKING_CONFIDENCE,
    MODELS_DIR,
)
from horizon.event_bus import EventBus
from horizon.types import Event, EventType

logger = logging.getLogger(__name__)

_BaseOptions = mp.tasks.BaseOptions
_GestureRecognizer = mp.tasks.vision.GestureRecognizer
_GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
_RunningMode = mp.tasks.vision.RunningMode


def _find_model_path() -> str:
    """Locate the gesture_recognizer.task model file."""
    candidates = [
        os.path.join(MODELS_DIR, "gesture_recognizer.task"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", MODELS_DIR, "gesture_recognizer.task"),
    ]
    pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    candidates.append(os.path.join(pkg_root, "models", "gesture_recognizer.task"))

    for path in candidates:
        abspath = os.path.abspath(path)
        if os.path.isfile(abspath):
            return abspath

    raise FileNotFoundError(
        "gesture_recognizer.task not found. Download it from "
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
        "gesture_recognizer/float16/latest/gesture_recognizer.task "
        f"and place it in the '{MODELS_DIR}/' directory."
    )


class HandDetector:
    """Detects hands, landmarks, AND pre-trained gestures using MediaPipe GestureRecognizer.

    The GestureRecognizer provides:
      - hand_landmarks: 21 normalized (x,y,z) landmarks per hand
      - handedness: left/right classification
      - gestures: pre-trained gesture labels (Closed_Fist, Open_Palm, Pointing_Up,
                  Thumb_Down, Thumb_Up, Victory, ILoveYou, None)

    Subscribes to FRAME events and publishes HAND_DETECTED events with all data.
    """

    def __init__(
        self,
        event_bus: EventBus,
        max_num_hands: int = DEFAULT_MAX_HANDS,
        min_detection_confidence: float = DEFAULT_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = DEFAULT_TRACKING_CONFIDENCE,
    ) -> None:
        self.event_bus = event_bus
        self._frame_index = 0

        model_path = _find_model_path()
        logger.info("Loading gesture recognizer model from %s", model_path)

        options = _GestureRecognizerOptions(
            base_options=_BaseOptions(model_asset_path=model_path),
            running_mode=_RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._recognizer = _GestureRecognizer.create_from_options(options)

        self.event_bus.subscribe(EventType.FRAME, self._on_frame)
        logger.info("HandDetector initialized (GestureRecognizer, %d pre-trained gestures)", 7)

    def _on_frame(self, event: Event) -> None:
        frame: np.ndarray = event.data
        rgb_frame = frame[:, :, ::-1].copy()  # BGR to RGB, contiguous
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        self._frame_index += 1
        timestamp_ms = self._frame_index * 33  # ~30fps

        result = self._recognizer.recognize_for_video(mp_image, timestamp_ms)

        has_hands = len(result.hand_landmarks) > 0

        # Extract the top gesture label and score for each hand
        gesture_labels = []
        for hand_gestures in result.gestures:
            if hand_gestures:
                top = hand_gestures[0]
                gesture_labels.append({
                    "label": top.category_name,
                    "score": top.score,
                })
            else:
                gesture_labels.append({"label": "None", "score": 0.0})

        self.event_bus.publish(Event(
            type=EventType.HAND_DETECTED,
            data={
                "result": result,
                "frame": frame,
                "has_hands": has_hands,
                "gesture_labels": gesture_labels,
            },
            source="hand_detector",
        ))

    def close(self) -> None:
        self.event_bus.unsubscribe(EventType.FRAME, self._on_frame)
        self._recognizer.close()
        logger.info("HandDetector closed")
