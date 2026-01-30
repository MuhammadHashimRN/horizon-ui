"""Plugin lifecycle management — discovery, loading, activation."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

from horizon.event_bus import EventBus
from horizon.plugins.base_plugin import BasePlugin
from horizon.types import Event, EventType

logger = logging.getLogger(__name__)

# Built-in plugins
BUILTIN_PLUGINS = {
    "media_control": "horizon.plugins.media_control.MediaControlPlugin",
    "slide_control": "horizon.plugins.slide_control.SlideControlPlugin",
}


class PluginManager:
    """Manages plugin discovery, lifecycle, and event forwarding.

    Discovers built-in and external plugins, manages their lifecycle
    (load, activate, deactivate, unload), and forwards events.
    """

    def __init__(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self._plugins: dict[str, BasePlugin] = {}
        self._active: set[str] = set()
        self.event_bus.subscribe(EventType.PLUGIN_EVENT, self._on_plugin_event)
        logger.info("PluginManager initialized")

    def discover(self) -> list[str]:
        """Discover available plugins. Returns list of plugin names."""
        available = list(BUILTIN_PLUGINS.keys())
        logger.info("Discovered %d plugins: %s", len(available), available)
        return available

    def load(self, name: str) -> bool:
        """Load a plugin by name."""
        if name in self._plugins:
            logger.warning("Plugin already loaded: %s", name)
            return True

        class_path = BUILTIN_PLUGINS.get(name)
        if not class_path:
            logger.error("Unknown plugin: %s", name)
            return False

        try:
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            plugin_class = getattr(module, class_name)
            plugin: BasePlugin = plugin_class()

            manifest = plugin.get_manifest()
            logger.info("Loaded plugin: %s v%s", manifest["name"], manifest["version"])

            self._plugins[name] = plugin
            return True
        except Exception:
            logger.exception("Failed to load plugin: %s", name)
            return False

    def activate(self, name: str) -> bool:
        """Activate a loaded plugin."""
        plugin = self._plugins.get(name)
        if not plugin:
            logger.error("Plugin not loaded: %s", name)
            return False

        if name in self._active:
            return True

        try:
            plugin.on_activate()
            self._active.add(name)
            logger.info("Plugin activated: %s", name)
            return True
        except Exception:
            logger.exception("Failed to activate plugin: %s", name)
            return False

    def deactivate(self, name: str) -> None:
        """Deactivate a plugin."""
        plugin = self._plugins.get(name)
        if plugin and name in self._active:
            try:
                plugin.on_deactivate()
            except Exception:
                logger.exception("Error deactivating plugin: %s", name)
            self._active.discard(name)
            logger.info("Plugin deactivated: %s", name)

    def unload(self, name: str) -> None:
        """Unload a plugin."""
        self.deactivate(name)
        self._plugins.pop(name, None)
        logger.info("Plugin unloaded: %s", name)

    def _on_plugin_event(self, event: Event) -> None:
        data = event.data
        if not isinstance(data, dict):
            return

        event_type = data.get("type", "")
        for name in self._active:
            plugin = self._plugins.get(name)
            if plugin:
                try:
                    plugin.on_event(event_type, data)
                except Exception:
                    logger.exception("Plugin %s error handling event", name)

    def forward_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Forward an event to all active plugins."""
        for name in self._active:
            plugin = self._plugins.get(name)
            if plugin:
                try:
                    plugin.on_event(event_type, data)
                except Exception:
                    logger.exception("Plugin %s error", name)

    def get_loaded_plugins(self) -> list[dict[str, Any]]:
        result = []
        for name, plugin in self._plugins.items():
            manifest = plugin.get_manifest()
            manifest["active"] = name in self._active
            result.append(manifest)
        return result

    def close(self) -> None:
        for name in list(self._active):
            self.deactivate(name)
        self._plugins.clear()
        self.event_bus.unsubscribe(EventType.PLUGIN_EVENT, self._on_plugin_event)
        logger.info("PluginManager closed")
