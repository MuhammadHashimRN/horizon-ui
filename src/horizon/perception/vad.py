"""Voice Activity Detection using WebRTC VAD."""

from __future__ import annotations

import logging
import struct
import time
from collections import deque

import webrtcvad

from horizon.constants import MIN_SPEECH_DURATION_MS, SPEECH_PAD_DURATION_MS, VAD_FRAME_DURATION_MS, VAD_MODE
from horizon.event_bus import EventBus
from horizon.types import Event, EventType

logger = logging.getLogger(__name__)


class VAD:
    """Voice Activity Detector using WebRTC VAD.

    Subscribes to AUDIO_CHUNK events (16kHz, int16 PCM). Detects speech
    onset and offset, buffering speech segments. On speech end, publishes
    a SPEECH_SEGMENT event containing the complete speech audio bytes.
    """

    def __init__(
        self,
        event_bus: EventBus,
        sample_rate: int = 16000,
        frame_duration_ms: int = VAD_FRAME_DURATION_MS,
        aggressiveness: int = VAD_MODE,
        pad_duration_ms: int = SPEECH_PAD_DURATION_MS,
        min_speech_ms: int = MIN_SPEECH_DURATION_MS,
    ) -> None:
        self.event_bus = event_bus
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.pad_duration_ms = pad_duration_ms
        self.min_speech_ms = min_speech_ms

        self._vad = webrtcvad.Vad(aggressiveness)
        self._frame_size = int(sample_rate * frame_duration_ms / 1000) * 2  # bytes (int16)

        # Ring buffer for VAD decisions
        num_pad_frames = pad_duration_ms // frame_duration_ms
        self._ring_buffer: deque[tuple[bytes, bool]] = deque(maxlen=num_pad_frames)

        self._triggered = False
        self._voiced_frames: list[bytes] = []
        self._audio_buffer = bytearray()

        self.event_bus.subscribe(EventType.AUDIO_CHUNK, self._on_audio_chunk)
        logger.info("VAD initialized (mode=%d)", aggressiveness)

    def _on_audio_chunk(self, event: Event) -> None:
        audio_bytes: bytes = event.data
        self._audio_buffer.extend(audio_bytes)

        # Process complete frames
        while len(self._audio_buffer) >= self._frame_size:
            frame = bytes(self._audio_buffer[: self._frame_size])
            self._audio_buffer = self._audio_buffer[self._frame_size :]
            self._process_frame(frame)

    def _process_frame(self, frame: bytes) -> None:
        is_speech = self._vad.is_speech(frame, self.sample_rate)

        if not self._triggered:
            self._ring_buffer.append((frame, is_speech))
            num_voiced = sum(1 for _, speech in self._ring_buffer if speech)

            if num_voiced > 0.9 * self._ring_buffer.maxlen:
                self._triggered = True
                self._voiced_frames = [f for f, _ in self._ring_buffer]
                self._ring_buffer.clear()
                logger.debug("Speech onset detected")
        else:
            self._voiced_frames.append(frame)
            self._ring_buffer.append((frame, is_speech))
            num_unvoiced = sum(1 for _, speech in self._ring_buffer if not speech)

            if num_unvoiced > 0.9 * self._ring_buffer.maxlen:
                self._triggered = False
                speech_audio = b"".join(self._voiced_frames)
                self._voiced_frames = []
                self._ring_buffer.clear()

                duration_ms = len(speech_audio) / (self.sample_rate * 2) * 1000
                if duration_ms >= self.min_speech_ms:
                    logger.debug("Speech segment: %.0fms", duration_ms)
                    self.event_bus.publish(Event(
                        type=EventType.SPEECH_SEGMENT,
                        data=speech_audio,
                        source="vad",
                    ))

    def reset(self) -> None:
        self._triggered = False
        self._voiced_frames = []
        self._ring_buffer.clear()
        self._audio_buffer = bytearray()

    def close(self) -> None:
        self.event_bus.unsubscribe(EventType.AUDIO_CHUNK, self._on_audio_chunk)
        logger.info("VAD closed")
