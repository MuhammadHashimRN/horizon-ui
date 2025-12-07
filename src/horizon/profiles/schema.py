"""Settings validation schema."""

from __future__ import annotations

from typing import Any

# Schema defines valid keys, types, and ranges for settings
SETTINGS_SCHEMA: dict[str, dict[str, Any]] = {
    "camera": {
        "device_index": {"type": int, "min": 0, "max": 10, "default": 0},
        "resolution": {"type": list, "default": [640, 480]},
        "fps": {"type": int, "min": 10, "max": 60, "default": 30},
    },
    "audio": {
        "sample_rate": {"type": int, "choices": [8000, 16000, 44100], "default": 16000},
        "channels": {"type": int, "min": 1, "max": 2, "default": 1},
        "chunk_duration_ms": {"type": int, "min": 10, "max": 100, "default": 30},
    },
    "mediapipe": {
        "max_num_hands": {"type": int, "min": 1, "max": 2, "default": 1},
        "min_detection_confidence": {"type": float, "min": 0.1, "max": 1.0, "default": 0.7},
        "min_tracking_confidence": {"type": float, "min": 0.1, "max": 1.0, "default": 0.6},
    },
    "asr": {
        "backend": {"type": str, "choices": ["whisper", "vosk"], "default": "whisper"},
        "whisper_model": {"type": str, "choices": ["tiny", "base", "small", "medium"], "default": "small"},
        "language": {"type": str, "default": "en"},
    },
    "gesture": {
        "confidence_threshold": {"type": float, "min": 0.1, "max": 1.0, "default": 0.75},
        "smoothing_alpha": {"type": float, "min": 0.05, "max": 1.0, "default": 0.3},
        "temporal_window_frames": {"type": int, "min": 5, "max": 30, "default": 15},
    },
    "overlay": {
        "opacity": {"type": float, "min": 0.1, "max": 1.0, "default": 0.85},
        "theme": {"type": str, "choices": ["dark", "light", "high_contrast"], "default": "dark"},
        "show_hand_skeleton": {"type": bool, "default": True},
        "cursor_size": {"type": int, "min": 8, "max": 64, "default": 24},
    },
    "system": {
        "startup_with_os": {"type": bool, "default": False},
        "global_hotkey": {"type": str, "default": "Ctrl+Shift+H"},
        "log_level": {"type": str, "choices": ["DEBUG", "INFO", "WARNING", "ERROR"], "default": "INFO"},
    },
}


def validate_settings(settings: dict[str, Any]) -> dict[str, list[str]]:
    """Validate settings against schema. Returns dict of section -> error messages."""
    errors: dict[str, list[str]] = {}

    for section, fields in SETTINGS_SCHEMA.items():
        section_data = settings.get(section, {})
        if not isinstance(section_data, dict):
            errors.setdefault(section, []).append(f"Expected dict, got {type(section_data).__name__}")
            continue

        for key, rules in fields.items():
            value = section_data.get(key)
            if value is None:
                continue  # Use default

            expected_type = rules["type"]
            if not isinstance(value, expected_type):
                errors.setdefault(section, []).append(
                    f"{key}: expected {expected_type.__name__}, got {type(value).__name__}"
                )
                continue

            if "min" in rules and value < rules["min"]:
                errors.setdefault(section, []).append(
                    f"{key}: {value} below minimum {rules['min']}"
                )
            if "max" in rules and value > rules["max"]:
                errors.setdefault(section, []).append(
                    f"{key}: {value} above maximum {rules['max']}"
                )
            if "choices" in rules and value not in rules["choices"]:
                errors.setdefault(section, []).append(
                    f"{key}: '{value}' not in {rules['choices']}"
                )

    return errors


def get_defaults() -> dict[str, dict[str, Any]]:
    """Return a dict of all default settings."""
    defaults: dict[str, dict[str, Any]] = {}
    for section, fields in SETTINGS_SCHEMA.items():
        defaults[section] = {}
        for key, rules in fields.items():
            defaults[section][key] = rules["default"]
    return defaults
