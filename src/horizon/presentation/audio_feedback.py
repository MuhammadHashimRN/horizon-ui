"""Audio feedback for system events using sound cues."""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import QUrl
from PyQt6.QtMultimedia import QSoundEffect

logger = logging.getLogger(__name__)


class AudioFeedback:
    """Plays WAV sound cues for activation, deactivation, and error events."""

    def __init__(self, assets_dir: str | Path = "assets/sounds") -> None:
        self._assets_dir = Path(assets_dir)
        self._sounds: dict[str, QSoundEffect] = {}
        self._enabled = True
        self._load_sounds()

    def _load_sounds(self) -> None:
        sound_files = {
            "activate": "activate.wav",
            "deactivate": "deactivate.wav",
            "error": "error.wav",
        }

        for name, filename in sound_files.items():
            path = self._assets_dir / filename
            if path.exists():
                effect = QSoundEffect()
                effect.setSource(QUrl.fromLocalFile(str(path.resolve())))
                effect.setVolume(0.5)
                self._sounds[name] = effect
                logger.debug("Loaded sound: %s", name)
            else:
                logger.debug("Sound file not found: %s", path)

    def play(self, sound_name: str) -> None:
        if not self._enabled:
            return

        effect = self._sounds.get(sound_name)
        if effect:
            effect.play()
        else:
            logger.debug("Sound not found: %s", sound_name)

    def play_activate(self) -> None:
        self.play("activate")

    def play_deactivate(self) -> None:
        self.play("deactivate")

    def play_error(self) -> None:
        self.play("error")

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def set_volume(self, volume: float) -> None:
        for effect in self._sounds.values():
            effect.setVolume(max(0.0, min(1.0, volume)))
