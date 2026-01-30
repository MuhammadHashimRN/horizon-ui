"""Abstract base class for Horizon UI plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BasePlugin(ABC):
    """Abstract base class that all plugins must implement.

    Plugins receive events from the system and can register
    custom actions and gesture/voice mappings.
    """

    @abstractmethod
    def get_manifest(self) -> dict[str, Any]:
        """Return plugin metadata.

        Returns:
            Dict with keys: name, version, description, author, permissions
        """
        ...

    @abstractmethod
    def on_activate(self) -> None:
        """Called when the plugin is activated."""
        ...

    @abstractmethod
    def on_deactivate(self) -> None:
        """Called when the plugin is deactivated."""
        ...

    @abstractmethod
    def on_event(self, event_type: str, data: dict[str, Any]) -> dict[str, Any] | None:
        """Called when a relevant event occurs.

        Args:
            event_type: The type of event (gesture, voice, system)
            data: Event data dictionary

        Returns:
            Optional response dictionary, or None
        """
        ...
