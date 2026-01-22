"""Maps fused actions to concrete OS events."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from horizon.constants import CONFIG_DIR, GESTURE_MAPPINGS_FILE
from horizon.types import ActionType, FusedAction, OSEvent

logger = logging.getLogger(__name__)


class ActionMapper:
    """Converts FusedAction into concrete OSEvent with parameters.

    Loads gesture mappings and voice command configs to resolve
    action-specific parameters (keys, text, etc.) and app overrides.
    """

    def __init__(
        self,
        gesture_mappings_file: str | Path | None = None,
    ) -> None:
        self._gesture_mappings: dict[str, Any] = {}
        self._app_overrides: dict[str, dict[str, Any]] = {}

        path = Path(gesture_mappings_file) if gesture_mappings_file else Path(CONFIG_DIR) / GESTURE_MAPPINGS_FILE
        self._load_mappings(path)
        logger.info("ActionMapper initialized")

    def _load_mappings(self, path: Path) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            self._gesture_mappings = data.get("gestures", {})
            self._app_overrides = data.get("app_overrides", {})
        except FileNotFoundError:
            logger.warning("Gesture mappings file not found: %s", path)
        except Exception:
            logger.exception("Failed to load gesture mappings from %s", path)

    def map(self, action: FusedAction, active_app: str = "") -> OSEvent:
        """Convert a FusedAction to a concrete OSEvent."""
        os_event = OSEvent(action=action.action)

        # Copy position for mouse actions
        os_event.x = int(action.cursor_x)
        os_event.y = int(action.cursor_y)

        # Resolve parameters
        params = dict(action.params)

        # Check for app-specific overrides
        if active_app and active_app in self._app_overrides:
            app_config = self._app_overrides[active_app]
            # Look up gesture name and override action/params
            for gesture_name, override in app_config.items():
                if isinstance(override, dict) and override.get("action") == action.action.value:
                    if "key" in override:
                        params["key"] = override["key"]
                    if "keys" in override:
                        params["keys"] = override["keys"]

        # Apply params to OSEvent
        if "keys" in params:
            os_event.keys = params["keys"]
        if "key" in params:
            os_event.key = params["key"]
        if "text" in params:
            os_event.text = params["text"]
        if "delta" in params:
            os_event.delta = params["delta"]

        os_event.params = params
        return os_event

    def get_app_overrides(self, app_name: str) -> dict[str, Any]:
        return self._app_overrides.get(app_name, {})
