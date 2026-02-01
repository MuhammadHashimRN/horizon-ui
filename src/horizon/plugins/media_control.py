"""Built-in plugin for media playback control."""

from __future__ import annotations

import ctypes
import logging
from typing import Any

from horizon.plugins.base_plugin import BasePlugin

logger = logging.getLogger(__name__)

# Virtual key codes for media keys
VK_MEDIA_PLAY_PAUSE = 0xB3
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_VOLUME_UP = 0xAF
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_MUTE = 0xAD

KEYEVENTF_KEYUP = 0x0002


class MediaControlPlugin(BasePlugin):
    """Controls media playback via system media keys.

    Responds to voice commands and gestures for play/pause,
    next/previous track, and volume control.
    """

    def get_manifest(self) -> dict[str, Any]:
        return {
            "name": "Media Control",
            "version": "1.0.0",
            "description": "Control media playback with gestures and voice",
            "author": "Horizon UI",
            "permissions": ["input_injection"],
        }

    def on_activate(self) -> None:
        logger.info("MediaControlPlugin activated")

    def on_deactivate(self) -> None:
        logger.info("MediaControlPlugin deactivated")

    def on_event(self, event_type: str, data: dict[str, Any]) -> dict[str, Any] | None:
        action = data.get("action", "")

        action_map = {
            "media_play_pause": VK_MEDIA_PLAY_PAUSE,
            "media_next": VK_MEDIA_NEXT_TRACK,
            "media_previous": VK_MEDIA_PREV_TRACK,
            "volume_up": VK_VOLUME_UP,
            "volume_down": VK_VOLUME_DOWN,
            "volume_mute": VK_VOLUME_MUTE,
        }

        vk = action_map.get(action)
        if vk is not None:
            self._press_key(vk)
            return {"status": "ok", "action": action}

        return None

    @staticmethod
    def _press_key(vk_code: int) -> None:
        user32 = ctypes.windll.user32
        user32.keybd_event(vk_code, 0, 0, 0)
        user32.keybd_event(vk_code, 0, KEYEVENTF_KEYUP, 0)
