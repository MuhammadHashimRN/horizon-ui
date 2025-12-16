"""Audio capture using sounddevice for voice command processing."""

from __future__ import annotations

import logging
import threading

import numpy as np
import sounddevice as sd

from horizon.constants import DEFAULT_AUDIO_CHANNELS, DEFAULT_CHUNK_DURATION_MS, DEFAULT_SAMPLE_RATE
from horizon.event_bus import EventBus
from horizon.types import Event, EventType

logger = logging.getLogger(__name__)


class AudioCapture:
    """Captures audio from the default microphone using sounddevice.

    Audio chunks are published to the EventBus as AUDIO_CHUNK events.
    Each chunk contains raw PCM int16 samples at 16kHz mono.
    """

    def __init__(
        self,
        event_bus: EventBus,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_AUDIO_CHANNELS,
        chunk_duration_ms: int = DEFAULT_CHUNK_DURATION_MS,
    ) -> None:
        self.event_bus = event_bus
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration_ms = chunk_duration_ms
        self._block_size = int(sample_rate * chunk_duration_ms / 1000)
        self._stream: sd.InputStream | None = None
        self._running = threading.Event()

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            logger.warning("Audio callback status: %s", status)

        if not self._running.is_set():
            return

        # Convert float32 to int16 PCM
        audio_int16 = (indata[:, 0] * 32767).astype(np.int16)

        self.event_bus.publish(Event(
            type=EventType.AUDIO_CHUNK,
            data=audio_int16.tobytes(),
            source="audio_capture",
        ))

    def start(self) -> None:
        self._running.set()
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                blocksize=self._block_size,
                callback=self._audio_callback,
            )
            self._stream.start()
            logger.info(
                "Audio capture started: rate=%d, channels=%d, chunk=%dms",
                self.sample_rate, self.channels, self.chunk_duration_ms,
            )
        except Exception:
            logger.exception("Failed to start audio capture")
            self._running.clear()
            self.event_bus.publish(Event(
                type=EventType.ERROR,
                data={"source": "audio_capture", "message": "Cannot open microphone"},
                source="audio_capture",
            ))

    def stop(self) -> None:
        self._running.clear()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("Audio capture stopped")

    @property
    def is_running(self) -> bool:
        return self._running.is_set()
