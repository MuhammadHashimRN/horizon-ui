"""Accessibility profile management — load, switch, and apply profiles."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from horizon.constants import CONFIG_DIR
from horizon.event_bus import EventBus
from horizon.profiles.settings_store import SettingsStore
from horizon.types import Event, EventType

logger = logging.getLogger(__name__)


class ProfileManager:
    """Manages accessibility profiles (sterile, low_vision, voice_only).

    Profiles are YAML files in config/profiles/ that define setting
    overrides. When a profile is activated, its overrides are applied
    on top of the current settings.
    """

    def __init__(
        self,
        event_bus: EventBus,
        settings_store: SettingsStore,
        profiles_dir: str | Path | None = None,
    ) -> None:
        self.event_bus = event_bus
        self._settings = settings_store
        self._profiles_dir = Path(profiles_dir) if profiles_dir else Path(CONFIG_DIR) / "profiles"
        self._profiles: dict[str, dict[str, Any]] = {}
        self._active_profile: str | None = None

        self._load_profiles()
        logger.info("ProfileManager initialized (%d profiles)", len(self._profiles))

    def _load_profiles(self) -> None:
        if not self._profiles_dir.exists():
            logger.warning("Profiles directory not found: %s", self._profiles_dir)
            return

        for yaml_file in self._profiles_dir.glob("*.yaml"):
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict):
                    name = yaml_file.stem
                    self._profiles[name] = data
                    logger.debug("Loaded profile: %s", name)
            except Exception:
                logger.exception("Failed to load profile: %s", yaml_file)

    def activate(self, profile_name: str) -> bool:
        if profile_name not in self._profiles:
            logger.warning("Profile not found: %s", profile_name)
            return False

        profile = self._profiles[profile_name]
        overrides = profile.get("overrides", {})

        self._settings.apply_profile(overrides)
        self._active_profile = profile_name

        self.event_bus.publish(Event(
            type=EventType.PROFILE_CHANGED,
            data={"profile": profile_name, "overrides": overrides},
            source="profile_manager",
        ))

        logger.info("Profile activated: %s", profile_name)
        return True

    def deactivate(self) -> None:
        if self._active_profile:
            logger.info("Profile deactivated: %s", self._active_profile)
            self._active_profile = None
            self._settings.clear_profile()

            self.event_bus.publish(Event(
                type=EventType.PROFILE_CHANGED,
                data={"profile": None},
                source="profile_manager",
            ))

    def get_active_profile(self) -> str | None:
        return self._active_profile

    def list_profiles(self) -> list[dict[str, str]]:
        result = []
        for name, data in self._profiles.items():
            result.append({
                "name": name,
                "display_name": data.get("name", name),
                "description": data.get("description", ""),
            })
        return result
