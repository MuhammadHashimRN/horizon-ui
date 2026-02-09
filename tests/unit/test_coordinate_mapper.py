"""Unit tests for coordinate mapper."""

import pytest

from horizon.perception.coordinate_mapper import CoordinateMapper


class TestCoordinateMapper:
    def test_center_maps_to_center(self):
        cm = CoordinateMapper(screen_width=1920, screen_height=1080, margin=0.0)
        x, y = cm.map(0.5, 0.5)
        assert 900 <= x <= 1000
        assert 500 <= y <= 560

    def test_clamping(self):
        cm = CoordinateMapper(screen_width=1920, screen_height=1080, margin=0.0)
        x, y = cm.map(-1.0, -1.0)
        assert x >= 0
        assert y >= 0

    def test_reset(self):
        cm = CoordinateMapper(screen_width=1920, screen_height=1080)
        cm.map(0.5, 0.5)
        cm.reset()
