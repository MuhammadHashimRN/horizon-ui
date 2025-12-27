"""Maps normalized camera coordinates to screen pixel coordinates."""

from __future__ import annotations

import ctypes
import logging

import numpy as np

from horizon.constants import (
    CURSOR_ALPHA_MAX,
    CURSOR_ALPHA_MIN,
    DEFAULT_CURSOR_SMOOTHING_ALPHA,
    DEFAULT_MIRROR_X,
)
from horizon.perception.smoothing import AdaptiveEMAFilter, CursorSmoother

logger = logging.getLogger(__name__)


def _get_screen_size() -> tuple[int, int]:
    """Get primary screen resolution using Win32 API."""
    try:
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        width = user32.GetSystemMetrics(0)
        height = user32.GetSystemMetrics(1)
        return width, height
    except Exception:
        logger.warning("Failed to get screen size, defaulting to 1920x1080")
        return 1920, 1080


class CoordinateMapper:
    """Maps normalized hand landmark coordinates (0-1) to screen pixels.

    Supports calibration via a 4-point homography matrix, adaptive
    smoothing, and webcam x-axis mirroring.
    """

    def __init__(
        self,
        screen_width: int | None = None,
        screen_height: int | None = None,
        smoothing_alpha: float = DEFAULT_CURSOR_SMOOTHING_ALPHA,
        use_kalman: bool = False,
        margin: float = 0.1,
        mirror_x: bool = DEFAULT_MIRROR_X,
        adaptive_smoothing: bool = True,
        alpha_min: float = CURSOR_ALPHA_MIN,
        alpha_max: float = CURSOR_ALPHA_MAX,
    ) -> None:
        if screen_width is None or screen_height is None:
            self.screen_width, self.screen_height = _get_screen_size()
        else:
            self.screen_width = screen_width
            self.screen_height = screen_height

        self.margin = margin
        self.mirror_x = mirror_x
        self._homography: np.ndarray | None = None

        if adaptive_smoothing:
            self._smoother = AdaptiveEMAFilter(
                base_alpha=smoothing_alpha,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
            )
        else:
            self._smoother = CursorSmoother(alpha=smoothing_alpha, use_kalman=use_kalman)

        logger.info(
            "CoordinateMapper: screen=%dx%d mirror_x=%s adaptive=%s margin=%.2f",
            self.screen_width, self.screen_height, mirror_x, adaptive_smoothing, margin,
        )
        # Debug: confirm mirror_x value at startup
        if mirror_x:
            logger.warning("mirror_x=True: coordinates will be flipped (frame_capture already flips!)")
        else:
            logger.info("mirror_x=False: no extra flip (frame_capture handles mirroring)")

    def map(self, norm_x: float, norm_y: float) -> tuple[int, int]:
        """Map normalized coordinates (0-1) to screen pixels."""
        # Mirror x-axis for webcam (so moving hand right moves cursor right)
        if self.mirror_x:
            norm_x = 1.0 - norm_x

        # Apply homography if calibrated
        if self._homography is not None:
            pt = np.array([norm_x, norm_y, 1.0])
            mapped = self._homography @ pt
            norm_x = mapped[0] / mapped[2]
            norm_y = mapped[1] / mapped[2]

        # Apply margin (expand usable area from center)
        norm_x = (norm_x - self.margin) / (1.0 - 2 * self.margin)
        norm_y = (norm_y - self.margin) / (1.0 - 2 * self.margin)

        # Clamp to [0, 1]
        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))

        # Smooth
        norm_x, norm_y = self._smoother.update(norm_x, norm_y)

        # Convert to screen pixels
        screen_x = int(norm_x * self.screen_width)
        screen_y = int(norm_y * self.screen_height)

        return screen_x, screen_y

    def set_calibration(self, src_points: np.ndarray, dst_points: np.ndarray) -> None:
        """Set a homography matrix from 4 calibration point pairs."""
        import cv2
        self._homography, _ = cv2.findHomography(src_points, dst_points)
        logger.info("Calibration homography set")

    def clear_calibration(self) -> None:
        self._homography = None

    def reset(self) -> None:
        self._smoother.reset()
