"""Smoothing filters for cursor position — EMA and Kalman."""

from __future__ import annotations

import logging

import numpy as np

from horizon.constants import (
    CURSOR_ALPHA_MAX,
    CURSOR_ALPHA_MIN,
    CURSOR_VELOCITY_HIGH,
    CURSOR_VELOCITY_LOW,
    DEFAULT_SMOOTHING_ALPHA,
)

logger = logging.getLogger(__name__)


class EMAFilter:
    """Exponential Moving Average filter for 2D coordinates.

    Smooths noisy cursor positions by blending the current reading
    with the previous smoothed position.
    """

    def __init__(self, alpha: float = DEFAULT_SMOOTHING_ALPHA) -> None:
        self.alpha = alpha
        self._prev_x: float | None = None
        self._prev_y: float | None = None

    def update(self, x: float, y: float) -> tuple[float, float]:
        if self._prev_x is None:
            self._prev_x = x
            self._prev_y = y
            return x, y

        smooth_x = self.alpha * x + (1 - self.alpha) * self._prev_x
        smooth_y = self.alpha * y + (1 - self.alpha) * self._prev_y
        self._prev_x = smooth_x
        self._prev_y = smooth_y
        return smooth_x, smooth_y

    def reset(self) -> None:
        self._prev_x = None
        self._prev_y = None


class AdaptiveEMAFilter:
    """EMA filter with velocity-adaptive alpha.

    Increases alpha (more responsive) during fast hand movements,
    decreases alpha (more smooth) when hand is nearly still.
    """

    def __init__(
        self,
        base_alpha: float = 0.5,
        alpha_min: float = CURSOR_ALPHA_MIN,
        alpha_max: float = CURSOR_ALPHA_MAX,
        velocity_low: float = CURSOR_VELOCITY_LOW,
        velocity_high: float = CURSOR_VELOCITY_HIGH,
    ) -> None:
        self.base_alpha = base_alpha
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.velocity_low = velocity_low
        self.velocity_high = velocity_high
        self._prev_x: float | None = None
        self._prev_y: float | None = None
        self._prev_raw_x: float | None = None
        self._prev_raw_y: float | None = None

    def update(self, x: float, y: float) -> tuple[float, float]:
        if self._prev_x is None:
            self._prev_x = x
            self._prev_y = y
            self._prev_raw_x = x
            self._prev_raw_y = y
            return x, y

        dx = x - self._prev_raw_x
        dy = y - self._prev_raw_y
        velocity = (dx * dx + dy * dy) ** 0.5
        self._prev_raw_x = x
        self._prev_raw_y = y

        if velocity <= self.velocity_low:
            alpha = self.alpha_min
        elif velocity >= self.velocity_high:
            alpha = self.alpha_max
        else:
            t = (velocity - self.velocity_low) / (self.velocity_high - self.velocity_low)
            alpha = self.alpha_min + t * (self.alpha_max - self.alpha_min)

        smooth_x = alpha * x + (1 - alpha) * self._prev_x
        smooth_y = alpha * y + (1 - alpha) * self._prev_y
        self._prev_x = smooth_x
        self._prev_y = smooth_y
        return smooth_x, smooth_y

    def reset(self) -> None:
        self._prev_x = None
        self._prev_y = None
        self._prev_raw_x = None
        self._prev_raw_y = None


class KalmanFilter2D:
    """Simple 2D Kalman filter for position + velocity estimation.

    State: [x, y, vx, vy]
    Measurement: [x, y]
    """

    def __init__(
        self,
        process_noise: float = 1e-3,
        measurement_noise: float = 1e-1,
    ) -> None:
        self.dt = 1.0 / 30.0  # Assume ~30 FPS

        # State: [x, y, vx, vy]
        self.x = np.zeros(4, dtype=np.float64)

        # State transition matrix
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # Covariance
        self.P = np.eye(4, dtype=np.float64) * 1.0

        # Process noise
        self.Q = np.eye(4, dtype=np.float64) * process_noise

        # Measurement noise
        self.R = np.eye(2, dtype=np.float64) * measurement_noise

        self._initialized = False

    def update(self, x: float, y: float) -> tuple[float, float]:
        measurement = np.array([x, y], dtype=np.float64)

        if not self._initialized:
            self.x[:2] = measurement
            self._initialized = True
            return x, y

        # Predict
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Update
        y_residual = measurement - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        self.x = x_pred + K @ y_residual
        self.P = (np.eye(4) - K @ self.H) @ P_pred

        return float(self.x[0]), float(self.x[1])

    def reset(self) -> None:
        self.x = np.zeros(4, dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 1.0
        self._initialized = False


class CursorSmoother:
    """Combined smoother using EMA with optional Kalman filter."""

    def __init__(
        self,
        alpha: float = DEFAULT_SMOOTHING_ALPHA,
        use_kalman: bool = False,
    ) -> None:
        self._ema = EMAFilter(alpha=alpha)
        self._kalman = KalmanFilter2D() if use_kalman else None

    def update(self, x: float, y: float) -> tuple[float, float]:
        sx, sy = self._ema.update(x, y)
        if self._kalman:
            sx, sy = self._kalman.update(sx, sy)
        return sx, sy

    def reset(self) -> None:
        self._ema.reset()
        if self._kalman:
            self._kalman.reset()
