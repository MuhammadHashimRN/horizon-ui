"""YAML-backed settings persistence with layered configuration."""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any

import yaml

from horizon.constants import APP_NAME, CONFIG_DIR, DEFAULT_SETTINGS_FILE
from horizon.profiles.schema import get_defaults, validate_settings

logger = logging.getLogger(__name__)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class SettingsStore:
    """Loads and manages application settings with layered configuration.

    Configuration precedence (highest to lowest):
    1. CLI arguments (applied at runtime)
    2. Active profile overrides
    3. User settings (~/.HorizonUI/settings.yaml or %APPDATA%/HorizonUI/settings.yaml)
    4. Project default settings (config/default_settings.yaml)
    5. Schema defaults
    """

    def __init__(
        self,
        defaults_file: str | Path | None = None,
        user_dir: str | Path | None = None,
    ) -> None:
        self._defaults_file = Path(defaults_file) if defaults_file else Path(CONFIG_DIR) / DEFAULT_SETTINGS_FILE
        self._user_dir = Path(user_dir) if user_dir else self._get_user_config_dir()
        self._user_file = self._user_dir / "settings.yaml"

        self._schema_defaults = get_defaults()
        self._file_defaults: dict[str, Any] = {}
        self._user_settings: dict[str, Any] = {}
        self._profile_overrides: dict[str, Any] = {}
        self._merged: dict[str, Any] = {}

        self._load()

    def _get_user_config_dir(self) -> Path:
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / APP_NAME
        return Path.home() / f".{APP_NAME}"

    def _load(self) -> None:
        # Load file defaults
        self._file_defaults = self._load_yaml(self._defaults_file)

        # Load user settings
        self._user_settings = self._load_yaml(self._user_file)

        # Merge layers
        self._rebuild_merged()

        errors = validate_settings(self._merged)
        if errors:
            for section, msgs in errors.items():
                for msg in msgs:
                    logger.warning("Settings validation: [%s] %s", section, msg)

        logger.info("Settings loaded (user_dir=%s)", self._user_dir)

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        try:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            logger.exception("Failed to load settings from %s", path)
        return {}

    def _rebuild_merged(self) -> None:
        merged = copy.deepcopy(self._schema_defaults)
        merged = _deep_merge(merged, self._file_defaults)
        merged = _deep_merge(merged, self._user_settings)
        merged = _deep_merge(merged, self._profile_overrides)
        self._merged = merged

    def get(self, section: str, key: str | None = None, default: Any = None) -> Any:
        section_data = self._merged.get(section, {})
        if key is None:
            return section_data
        return section_data.get(key, default)

    def set(self, section: str, key: str, value: Any) -> None:
        if section not in self._user_settings:
            self._user_settings[section] = {}
        self._user_settings[section][key] = value
        self._rebuild_merged()

    def apply_profile(self, overrides: dict[str, Any]) -> None:
        self._profile_overrides = overrides
        self._rebuild_merged()

    def clear_profile(self) -> None:
        self._profile_overrides = {}
        self._rebuild_merged()

    def save(self) -> None:
        self._user_dir.mkdir(parents=True, exist_ok=True)
        with open(self._user_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(self._user_settings, f, default_flow_style=False)
        logger.info("User settings saved to %s", self._user_file)

    def update_from_dict(self, settings: dict[str, Any]) -> None:
        self._user_settings = _deep_merge(self._user_settings, settings)
        self._rebuild_merged()

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._merged)

    def reset(self) -> None:
        self._user_settings = {}
        self._profile_overrides = {}
        self._rebuild_merged()
