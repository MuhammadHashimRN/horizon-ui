"""Unit tests for settings store."""

import pytest

from horizon.profiles.settings_store import SettingsStore, _deep_merge


class TestDeepMerge:
    def test_flat_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"a": {"x": 1, "y": 2}}
        override = {"a": {"y": 3, "z": 4}}
        result = _deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 3, "z": 4}}


class TestSettingsStore:
    def test_get_defaults(self):
        store = SettingsStore()
        camera = store.get("camera")
        assert isinstance(camera, dict)

    def test_set_and_get(self):
        store = SettingsStore()
        store.set("camera", "fps", 60)
        assert store.get("camera", "fps") == 60

    def test_reset(self):
        store = SettingsStore()
        store.set("camera", "fps", 60)
        store.reset()
        fps = store.get("camera", "fps")
        assert fps is not None
