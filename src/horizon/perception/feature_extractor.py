"""Feature extraction from hand landmarks for gesture classification."""

from __future__ import annotations

import logging
import math
from collections import deque

import numpy as np

from horizon.constants import DEFAULT_TEMPORAL_WINDOW
from horizon.event_bus import EventBus
from horizon.types import Event, EventType, LandmarkSet

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extracts temporal features from a sliding window of hand landmarks.

    Maintains a buffer of recent LandmarkSets and computes features including:
    - Normalized landmark positions (63 values: 21 landmarks * 3 coords)
    - Inter-frame velocity of key landmarks (fingertips, wrist)
    - Finger extension ratios
    - Pairwise distances (thumb-index, thumb-middle, etc.)

    Subscribes to LANDMARKS events. When the temporal buffer is full,
    publishes feature vectors for gesture classification.
    """

    # Key landmark indices (MediaPipe hand model)
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20
    THUMB_MCP = 2
    INDEX_MCP = 5
    MIDDLE_MCP = 9
    RING_MCP = 13
    PINKY_MCP = 17
    INDEX_PIP = 6
    MIDDLE_PIP = 10
    RING_PIP = 14
    PINKY_PIP = 18

    def __init__(
        self,
        event_bus: EventBus,
        temporal_window: int = DEFAULT_TEMPORAL_WINDOW,
    ) -> None:
        self.event_bus = event_bus
        self.temporal_window = temporal_window
        self._buffer: deque[LandmarkSet] = deque(maxlen=temporal_window)
        self.event_bus.subscribe(EventType.LANDMARKS, self._on_landmarks)
        logger.info("FeatureExtractor initialized (window=%d)", temporal_window)

    def _on_landmarks(self, event: Event) -> None:
        landmark_set: LandmarkSet = event.data
        self._buffer.append(landmark_set)

        if len(self._buffer) < self.temporal_window:
            return

        features = self._extract_features()
        self.event_bus.publish(Event(
            type=EventType.GESTURE_RESULT,
            data={
                "features": features,
                "landmarks": landmark_set,
                "needs_classification": True,
            },
            source="feature_extractor",
        ))

    def _extract_features(self) -> np.ndarray:
        frames = list(self._buffer)
        all_features = []

        for frame_idx, lms in enumerate(frames):
            # Normalized positions relative to wrist
            wrist = lms.landmarks[self.WRIST]
            positions = []
            for lm in lms.landmarks:
                positions.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
            all_features.extend(positions)

            # Finger extension ratios
            extensions = self._finger_extensions(lms)
            all_features.extend(extensions)

            # Pairwise distances
            distances = self._pairwise_distances(lms)
            all_features.extend(distances)

            # Velocity (inter-frame) for key points
            if frame_idx > 0:
                prev = frames[frame_idx - 1]
                velocities = self._compute_velocities(prev, lms)
                all_features.extend(velocities)
            else:
                # Zero velocity for first frame
                all_features.extend([0.0] * 15)  # 5 key points * 3 coords

        return np.array(all_features, dtype=np.float32)

    def _finger_extensions(self, lms: LandmarkSet) -> list[float]:
        tips = [self.THUMB_TIP, self.INDEX_TIP, self.MIDDLE_TIP, self.RING_TIP, self.PINKY_TIP]
        mcps = [self.THUMB_MCP, self.INDEX_MCP, self.MIDDLE_MCP, self.RING_MCP, self.PINKY_MCP]
        wrist = lms.landmarks[self.WRIST]

        ratios = []
        for tip_idx, mcp_idx in zip(tips, mcps):
            tip = lms.landmarks[tip_idx]
            mcp = lms.landmarks[mcp_idx]
            tip_dist = math.sqrt(
                (tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2 + (tip.z - wrist.z) ** 2
            )
            mcp_dist = math.sqrt(
                (mcp.x - wrist.x) ** 2 + (mcp.y - wrist.y) ** 2 + (mcp.z - wrist.z) ** 2
            )
            ratios.append(tip_dist / max(mcp_dist, 1e-6))

        return ratios

    def _pairwise_distances(self, lms: LandmarkSet) -> list[float]:
        pairs = [
            (self.THUMB_TIP, self.INDEX_TIP),
            (self.THUMB_TIP, self.MIDDLE_TIP),
            (self.INDEX_TIP, self.MIDDLE_TIP),
            (self.INDEX_TIP, self.RING_TIP),
            (self.RING_TIP, self.PINKY_TIP),
        ]
        distances = []
        for a_idx, b_idx in pairs:
            a = lms.landmarks[a_idx]
            b = lms.landmarks[b_idx]
            dist = math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
            distances.append(dist)
        return distances

    def _compute_velocities(self, prev: LandmarkSet, curr: LandmarkSet) -> list[float]:
        key_indices = [self.WRIST, self.THUMB_TIP, self.INDEX_TIP, self.MIDDLE_TIP, self.PINKY_TIP]
        velocities = []
        dt = max(curr.timestamp - prev.timestamp, 1e-6)
        for idx in key_indices:
            p = prev.landmarks[idx]
            c = curr.landmarks[idx]
            velocities.extend([
                (c.x - p.x) / dt,
                (c.y - p.y) / dt,
                (c.z - p.z) / dt,
            ])
        return velocities

    def reset(self) -> None:
        self._buffer.clear()

    def close(self) -> None:
        self.event_bus.unsubscribe(EventType.LANDMARKS, self._on_landmarks)
        logger.info("FeatureExtractor closed")
