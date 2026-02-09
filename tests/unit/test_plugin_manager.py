"""Unit tests for plugin manager."""

import pytest

from horizon.control.plugin_manager import PluginManager
from horizon.event_bus import EventBus


class TestPluginManager:
    def test_discover(self, event_bus):
        pm = PluginManager(event_bus=event_bus)
        plugins = pm.discover()
        assert "media_control" in plugins
        assert "slide_control" in plugins

    def test_load_builtin(self, event_bus):
        pm = PluginManager(event_bus=event_bus)
        assert pm.load("media_control") is True

    def test_load_unknown(self, event_bus):
        pm = PluginManager(event_bus=event_bus)
        assert pm.load("nonexistent") is False

    def test_activate_not_loaded(self, event_bus):
        pm = PluginManager(event_bus=event_bus)
        assert pm.activate("media_control") is False
