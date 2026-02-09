"""Unit tests for context manager."""

import pytest

from horizon.decision.context_manager import ContextManager


class TestContextManager:
    def test_extract_app_from_title_powerpoint(self):
        name = ContextManager._extract_app_from_title("Presentation1 - PowerPoint")
        assert name == "PowerPoint"

    def test_extract_app_from_title_chrome(self):
        name = ContextManager._extract_app_from_title("Google - Chrome")
        assert name == "Chrome"

    def test_extract_app_from_unknown(self):
        name = ContextManager._extract_app_from_title("Some - Unknown App")
        assert name == "Unknown App"
