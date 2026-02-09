"""Unit tests for profile manager."""

import pytest

from horizon.event_bus import EventBus
from horizon.profiles.profile_manager import ProfileManager
from horizon.profiles.settings_store import SettingsStore


class TestProfileManager:
    def test_list_profiles(self, event_bus):
        store = SettingsStore()
        pm = ProfileManager(event_bus=event_bus, settings_store=store)
        profiles = pm.list_profiles()
        assert isinstance(profiles, list)

    def test_activate_unknown_profile(self, event_bus):
        store = SettingsStore()
        pm = ProfileManager(event_bus=event_bus, settings_store=store)
        assert pm.activate("nonexistent") is False
