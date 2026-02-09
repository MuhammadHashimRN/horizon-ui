"""Unit tests for OS event injector."""

import pytest

from horizon.control.os_event_injector import VK_MAP


class TestOSEventInjector:
    def test_vk_map_has_common_keys(self):
        assert "alt" in VK_MAP
        assert "ctrl" in VK_MAP
        assert "shift" in VK_MAP
        assert "enter" in VK_MAP
        assert "escape" in VK_MAP
        assert "space" in VK_MAP
        assert "left" in VK_MAP
        assert "right" in VK_MAP
