"""Automatic Speech Recognition with swappable backends (Whisper / Vosk)."""

from __future__ import annotations

import io
import logging
import struct
import wave
from abc import ABC, abstractmethod

import numpy as np

from horizon.event_bus import EventBus
from horizon.types import Event, EventType

logger = logging.getLogger(__name__)


class ASRBackend(ABC):
    """Abstract base class for ASR backends."""

    @abstractmethod
    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        """Transcribe raw PCM int16 audio bytes to text."""
        ...

    @abstractmethod
    def close(self) -> None: ...


class WhisperBackend(ASRBackend):
    """ASR backend using faster-whisper (CTranslate2)."""

    def __init__(self, model_size: str = "small", language: str = "en") -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self._language = language
        logger.info("Whisper backend loaded (model=%s)", model_size)

    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        # Convert int16 PCM to float32 numpy array
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        segments, _ = self._model.transcribe(
            audio_float32,
            language=self._language,
            beam_size=3,
            vad_filter=True,
        )

        text = " ".join(seg.text.strip() for seg in segments)
        return text.strip()

    def close(self) -> None:
        self._model = None
        logger.info("Whisper backend closed")


class VoskBackend(ASRBackend):
    """ASR backend using Vosk offline speech recognition."""

    def __init__(self, model_path: str | None = None) -> None:
        import vosk

        vosk.SetLogLevel(-1)
        if model_path:
            self._model = vosk.Model(model_path)
        else:
            self._model = vosk.Model(lang="en-us")
        self._sample_rate = 16000
        logger.info("Vosk backend loaded")

    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        import json

        from vosk import KaldiRecognizer

        rec = KaldiRecognizer(self._model, sample_rate)
        rec.AcceptWaveform(audio_bytes)
        result = json.loads(rec.FinalResult())
        return result.get("text", "").strip()

    def close(self) -> None:
        self._model = None
        logger.info("Vosk backend closed")


class ASR:
    """Speech recognition service with swappable backends.

    Subscribes to SPEECH_SEGMENT events and publishes TRANSCRIPT events.
    """

    def __init__(
        self,
        event_bus: EventBus,
        backend: str = "whisper",
        whisper_model: str = "small",
        language: str = "en",
        vosk_model_path: str | None = None,
    ) -> None:
        self.event_bus = event_bus
        self._backend: ASRBackend | None = None
        self._backend_name = backend

        if backend == "whisper":
            self._backend = WhisperBackend(model_size=whisper_model, language=language)
        elif backend == "vosk":
            self._backend = VoskBackend(model_path=vosk_model_path)
        else:
            logger.error("Unknown ASR backend: %s", backend)

        self.event_bus.subscribe(EventType.SPEECH_SEGMENT, self._on_speech_segment)
        logger.info("ASR initialized (backend=%s)", backend)

    def _on_speech_segment(self, event: Event) -> None:
        audio_bytes: bytes = event.data

        if self._backend is None:
            return

        try:
            transcript = self._backend.transcribe(audio_bytes)
            if transcript:
                logger.debug("Transcript: %s", transcript)
                self.event_bus.publish(Event(
                    type=EventType.TRANSCRIPT,
                    data=transcript,
                    source="asr",
                ))
        except Exception:
            logger.exception("ASR transcription error")

    def close(self) -> None:
        self.event_bus.unsubscribe(EventType.SPEECH_SEGMENT, self._on_speech_segment)
        if self._backend:
            self._backend.close()
        logger.info("ASR closed")
