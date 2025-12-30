"""Hybrid gesture classifier: MediaPipe pre-trained + rule-based dynamic gestures."""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from pathlib import Path

import numpy as np

from horizon.constants import DEFAULT_CONFIDENCE_THRESHOLD
from horizon.event_bus import EventBus
from horizon.types import Event, EventType, GestureLabel, GestureResult, LandmarkSet, Landmark

logger = logging.getLogger(__name__)

# MediaPipe pre-trained gesture name → our GestureLabel
MP_GESTURE_MAP: dict[str, GestureLabel] = {
    "Closed_Fist": GestureLabel.FIST,
    "Open_Palm": GestureLabel.OPEN_PALM,
    "Pointing_Up": GestureLabel.POINT,
    "Thumb_Up": GestureLabel.THUMBS_UP,
    "Thumb_Down": GestureLabel.FIST,       # remap thumb-down to fist (right-click)
    "Victory": GestureLabel.TWO_FINGER_SCROLL,
    "ILoveYou": GestureLabel.PINCH_SPREAD,  # remap ILY to zoom
    "None": GestureLabel.NONE,
}

# Landmark indices
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
INDEX_PIP = 6


class GestureClassifier:
    """Hybrid gesture classifier using MediaPipe pre-trained gestures + rule-based detection.

    Static gestures (fist, open palm, point, thumbs up, victory) come from
    MediaPipe's GestureRecognizer model. Dynamic gestures (swipe left/right,
    pinch, pinch-spread, scroll) are detected via rule-based analysis of
    landmark positions and velocities over a sliding window.

    Subscribes to HAND_DETECTED events (which now include gesture_labels from
    MediaPipe) and LANDMARKS events (for rule-based detection). Publishes
    GESTURE_RESULT events with the final classified gesture.
    """

    def __init__(
        self,
        event_bus: EventBus,
        model_path: str | Path | None = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.event_bus = event_bus
        self.confidence_threshold = confidence_threshold

        # Sliding window for rule-based dynamic gesture detection
        self._landmark_history: deque[LandmarkSet] = deque(maxlen=20)
        self._last_gesture_time: float = 0.0
        self._gesture_cooldown = 0.3  # 300ms cooldown between gesture triggers

        # Track the last MediaPipe gesture for state change detection
        self._last_mp_gesture = GestureLabel.NONE
        self._current_gesture_held = GestureLabel.NONE  # for discrete gesture dedup

        # POINT is handled by CursorTracker; TWO_FINGER_SCROLL is continuous
        self._continuous_gestures = {GestureLabel.TWO_FINGER_SCROLL}

        # Subscribe to hand detection results (includes pre-trained gestures)
        self.event_bus.subscribe(EventType.HAND_DETECTED, self._on_hand_detected)
        # Subscribe to landmarks for rule-based dynamic gestures
        self.event_bus.subscribe(EventType.LANDMARKS, self._on_landmarks)
        logger.info("GestureClassifier initialized (hybrid: MediaPipe + rule-based)")

    def _on_hand_detected(self, event: Event) -> None:
        """Process MediaPipe pre-trained gesture results."""
        data = event.data
        if not data["has_hands"]:
            self._last_mp_gesture = GestureLabel.NONE
            self._current_gesture_held = GestureLabel.NONE
            return

        gesture_labels = data.get("gesture_labels", [])
        if not gesture_labels:
            return

        top = gesture_labels[0]
        mp_label = top["label"]
        mp_score = top["score"]

        our_label = MP_GESTURE_MAP.get(mp_label, GestureLabel.NONE)

        if our_label == GestureLabel.NONE:
            self._last_mp_gesture = GestureLabel.NONE
            self._current_gesture_held = GestureLabel.NONE
            return

        if mp_score < self.confidence_threshold:
            self._last_mp_gesture = GestureLabel.NONE
            self._current_gesture_held = GestureLabel.NONE
            return

        self._last_mp_gesture = our_label

        # POINT gesture cursor movement is handled by CursorTracker
        if our_label == GestureLabel.POINT:
            return

        # For discrete gestures, only fire on state change (onset)
        if our_label not in self._continuous_gestures:
            if our_label == self._current_gesture_held:
                return  # same discrete gesture still held, skip
            self._current_gesture_held = our_label
        else:
            self._current_gesture_held = our_label

        # Build cursor position from index finger tip
        result_obj = data["result"]
        cursor_x, cursor_y = 0.0, 0.0
        if result_obj.hand_landmarks:
            lms = result_obj.hand_landmarks[0]
            if len(lms) > INDEX_TIP:
                cursor_x = lms[INDEX_TIP].x
                cursor_y = lms[INDEX_TIP].y

        gesture_result = GestureResult(
            label=our_label,
            confidence=mp_score,
            cursor_x=cursor_x,
            cursor_y=cursor_y,
        )

        self.event_bus.publish(Event(
            type=EventType.GESTURE_RESULT,
            data=gesture_result,
            source="gesture_classifier",
        ))

    def _on_landmarks(self, event: Event) -> None:
        """Accumulate landmarks for rule-based dynamic gesture detection."""
        landmark_set: LandmarkSet = event.data
        self._landmark_history.append(landmark_set)

        if len(self._landmark_history) < 5:
            return

        now = time.time()
        if now - self._last_gesture_time < self._gesture_cooldown:
            return

        # Don't override an active static gesture from MediaPipe
        if self._last_mp_gesture not in (GestureLabel.NONE, GestureLabel.POINT):
            return

        # Try to detect dynamic gestures
        dynamic = self._detect_dynamic_gesture()
        if dynamic and dynamic.label != GestureLabel.NONE:
            self._last_gesture_time = now
            self.event_bus.publish(Event(
                type=EventType.GESTURE_RESULT,
                data=dynamic,
                source="gesture_classifier_rule",
            ))

    def _detect_dynamic_gesture(self) -> GestureResult | None:
        """Rule-based detection for swipes, pinch, and scroll gestures."""
        history = list(self._landmark_history)
        if len(history) < 5:
            return None

        current = history[-1]
        cursor_x = current.landmarks[INDEX_TIP].x
        cursor_y = current.landmarks[INDEX_TIP].y

        # --- Swipe detection (large horizontal displacement of wrist) ---
        swipe = self._detect_swipe(history)
        if swipe:
            return GestureResult(
                label=swipe,
                confidence=0.85,
                landmarks=current,
                cursor_x=cursor_x,
                cursor_y=cursor_y,
            )

        # --- Pinch detection (thumb-index distance) ---
        pinch = self._detect_pinch(history)
        if pinch:
            return GestureResult(
                label=pinch,
                confidence=0.80,
                landmarks=current,
                cursor_x=cursor_x,
                cursor_y=cursor_y,
            )

        # --- Two-finger scroll detection ---
        scroll = self._detect_scroll(history)
        if scroll:
            return GestureResult(
                label=scroll,
                confidence=0.80,
                landmarks=current,
                cursor_x=cursor_x,
                cursor_y=cursor_y,
            )

        return None

    def _detect_swipe(self, history: list[LandmarkSet]) -> GestureLabel | None:
        """Detect horizontal swipe from wrist displacement over recent frames."""
        window = history[-8:]  # look at last 8 frames
        if len(window) < 6:
            return None

        wrist_xs = [f.landmarks[WRIST].x for f in window]
        dx = wrist_xs[-1] - wrist_xs[0]
        dt = max(window[-1].timestamp - window[0].timestamp, 0.001)
        velocity = dx / dt

        # Need significant displacement (>15% of frame) and speed
        if abs(dx) > 0.15 and abs(velocity) > 0.8:
            if dx > 0:
                return GestureLabel.SWIPE_RIGHT
            else:
                return GestureLabel.SWIPE_LEFT
        return None

    def _detect_pinch(self, history: list[LandmarkSet]) -> GestureLabel | None:
        """Detect pinch (close) or pinch-spread by tracking thumb-index distance."""
        if len(history) < 5:
            return None

        window = history[-5:]
        distances = []
        for lms in window:
            thumb = lms.landmarks[THUMB_TIP]
            index = lms.landmarks[INDEX_TIP]
            d = math.sqrt((thumb.x - index.x)**2 + (thumb.y - index.y)**2)
            distances.append(d)

        # Check if thumb-index are close (pinch) — static pinch
        if distances[-1] < 0.04:
            return GestureLabel.PINCH

        # Check for closing motion (pinch-close)
        if len(distances) >= 4:
            d_change = distances[-1] - distances[0]
            if d_change < -0.08:
                return GestureLabel.PINCH_CLOSE
            elif d_change > 0.08:
                return GestureLabel.PINCH_SPREAD

        return None

    def _detect_scroll(self, history: list[LandmarkSet]) -> GestureLabel | None:
        """Detect two-finger scroll (index + middle moving vertically together)."""
        if len(history) < 5:
            return None

        window = history[-5:]
        current = window[-1]

        # Check if index and middle fingers are extended and close together
        idx_tip = current.landmarks[INDEX_TIP]
        mid_tip = current.landmarks[MIDDLE_TIP]
        ring_tip = current.landmarks[RING_TIP]

        idx_mid_dist = math.sqrt((idx_tip.x - mid_tip.x)**2 + (idx_tip.y - mid_tip.y)**2)
        mid_ring_dist = math.sqrt((mid_tip.x - ring_tip.x)**2 + (mid_tip.y - ring_tip.y)**2)

        # Index and middle close, ring finger far (2-finger gesture)
        if idx_mid_dist < 0.06 and mid_ring_dist > 0.04:
            # Check vertical movement
            idx_ys = [f.landmarks[INDEX_TIP].y for f in window]
            dy = idx_ys[-1] - idx_ys[0]
            if abs(dy) > 0.05:
                return GestureLabel.TWO_FINGER_SCROLL

        return None

    def close(self) -> None:
        self.event_bus.unsubscribe(EventType.HAND_DETECTED, self._on_hand_detected)
        self.event_bus.unsubscribe(EventType.LANDMARKS, self._on_landmarks)
        logger.info("GestureClassifier closed")
