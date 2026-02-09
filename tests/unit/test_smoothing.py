"""Unit tests for smoothing filters."""

import pytest

from horizon.perception.smoothing import CursorSmoother, EMAFilter, KalmanFilter2D


class TestEMAFilter:
    def test_first_value_passthrough(self):
        f = EMAFilter(alpha=0.3)
        x, y = f.update(100.0, 200.0)
        assert x == 100.0
        assert y == 200.0

    def test_smoothing_effect(self):
        f = EMAFilter(alpha=0.3)
        f.update(0.0, 0.0)
        x, y = f.update(100.0, 100.0)
        assert x == pytest.approx(30.0)
        assert y == pytest.approx(30.0)

    def test_reset(self):
        f = EMAFilter(alpha=0.3)
        f.update(50.0, 50.0)
        f.reset()
        x, y = f.update(100.0, 100.0)
        assert x == 100.0


class TestKalmanFilter2D:
    def test_first_value_passthrough(self):
        kf = KalmanFilter2D()
        x, y = kf.update(100.0, 200.0)
        assert x == 100.0
        assert y == 200.0

    def test_smoothing(self):
        kf = KalmanFilter2D()
        kf.update(0.0, 0.0)
        x, y = kf.update(100.0, 100.0)
        assert 0.0 < x < 100.0


class TestCursorSmoother:
    def test_ema_only(self):
        cs = CursorSmoother(alpha=0.5, use_kalman=False)
        cs.update(0.0, 0.0)
        x, y = cs.update(100.0, 100.0)
        assert x == pytest.approx(50.0)
