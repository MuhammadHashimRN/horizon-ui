"""Integration tests for the plugin system."""

import pytest

from horizon.control.ipc_auth import IPCAuth
from horizon.control.plugin_manager import PluginManager
from horizon.event_bus import EventBus


class TestPluginSystem:
    """Test plugin discovery, loading, activation, and event forwarding."""

    def test_full_plugin_lifecycle(self, event_bus):
        """Test load → activate → deactivate → unload cycle."""
        pm = PluginManager(event_bus=event_bus)

        # Discover
        plugins = pm.discover()
        assert "media_control" in plugins

        # Load
        assert pm.load("media_control") is True

        # Activate
        assert pm.activate("media_control") is True

        # Check loaded
        loaded = pm.get_loaded_plugins()
        assert len(loaded) == 1
        assert loaded[0]["name"] == "Media Control"
        assert loaded[0]["active"] is True

        # Deactivate
        pm.deactivate("media_control")
        loaded = pm.get_loaded_plugins()
        assert loaded[0]["active"] is False

        # Unload
        pm.unload("media_control")
        assert len(pm.get_loaded_plugins()) == 0

    def test_ipc_auth_flow(self):
        """Test token generation and validation."""
        auth = IPCAuth()

        token = auth.generate_token("test_plugin")
        assert auth.validate_token(token) == "test_plugin"

        auth.revoke_token("test_plugin")
        assert auth.validate_token(token) is None
