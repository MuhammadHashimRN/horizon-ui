"""OpenCV webcam frame capture running on a dedicated thread."""

from __future__ import annotations

import logging
import queue
import threading
import time

import cv2
import numpy as np

from horizon.constants import DEFAULT_CAMERA_INDEX, DEFAULT_FPS, DEFAULT_RESOLUTION, FRAME_QUEUE_MAX_SIZE
from horizon.event_bus import EventBus
from horizon.types import Event, EventType

logger = logging.getLogger(__name__)


class FrameCapture(threading.Thread):
    """Captures frames from the webcam on a background thread.

    Frames are placed into a bounded queue (dropping oldest on overflow)
    and published to the EventBus as FRAME events.
    """

    def __init__(
        self,
        event_bus: EventBus,
        device_index: int = DEFAULT_CAMERA_INDEX,
        resolution: tuple[int, int] = DEFAULT_RESOLUTION,
        fps: int = DEFAULT_FPS,
    ) -> None:
        super().__init__(daemon=True, name="FrameCapture")
        self.event_bus = event_bus
        self.device_index = device_index
        self.resolution = resolution
        self.fps = fps
        self._frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=FRAME_QUEUE_MAX_SIZE)
        self._running = threading.Event()
        self._cap: cv2.VideoCapture | None = None

    def run(self) -> None:
        self._running.set()
        self._cap = cv2.VideoCapture(self.device_index)
        if not self._cap.isOpened():
            logger.error("Failed to open camera at index %d", self.device_index)
            self.event_bus.publish(Event(
                type=EventType.ERROR,
                data={"source": "frame_capture", "message": f"Cannot open camera {self.device_index}"},
                source="frame_capture",
            ))
            return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        logger.info(
            "Camera opened: index=%d, resolution=%s, fps=%d",
            self.device_index, self.resolution, self.fps,
        )

        frame_interval = 1.0 / self.fps

        while self._running.is_set():
            start = time.perf_counter()
            ret, frame = self._cap.read()

            if not ret:
                logger.warning("Failed to read frame from camera")
                time.sleep(0.01)
                continue

            # Flip horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Drop oldest frame if queue is full
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass

            self._frame_queue.put(frame)

            self.event_bus.publish(Event(
                type=EventType.FRAME,
                data=frame,
                source="frame_capture",
            ))

            elapsed = time.perf_counter() - start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self._cleanup()

    def stop(self) -> None:
        self._running.clear()

    def get_frame(self) -> np.ndarray | None:
        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return None

    def _cleanup(self) -> None:
        if self._cap and self._cap.isOpened():
            self._cap.release()
            logger.info("Camera released")

    @property
    def is_running(self) -> bool:
        return self._running.is_set()
