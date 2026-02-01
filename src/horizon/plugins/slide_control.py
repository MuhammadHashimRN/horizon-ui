"""Built-in plugin for presentation slide control."""

from __future__ import annotations

import ctypes
import logging
from typing import Any

from horizon.plugins.base_plugin import BasePlugin

logger = logging.getLogger(__name__)

# Virtual key codes
VK_LEFT = 0x25
VK_RIGHT = 0x27
VK_F5 = 0x74
VK_ESCAPE = 0x1B
KEYEVENTF_KEYUP = 0x0002


class SlideControlPlugin(BasePlugin):
    """Controls presentation slides (PowerPoint, Google Slides, etc.).

    Responds to gesture swipes and voice commands for slide navigation.
    """

    def get_manifest(self) -> dict[str, Any]:
        return {
            "name": "Slide Control",
            "version": "1.0.0",
            "description": "Control presentation slides with gestures and voice",
            "author": "Horizon UI",
            "permissions": ["input_injection"],
        }

    def on_activate(self) -> None:
        logger.info("SlideControlPlugin activated")

    def on_deactivate(self) -> None:
        logger.info("SlideControlPlugin deactivated")

    def on_event(self, event_type: str, data: dict[str, Any]) -> dict[str, Any] | None:
        action = data.get("action", "")

        action_map = {
            "next_slide": VK_RIGHT,
            "previous_slide": VK_LEFT,
            "start_slideshow": VK_F5,
            "end_slideshow": VK_ESCAPE,
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
