"""HUD rendering utilities for the overlay window."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from PyQt6.QtCore import QRectF
from PyQt6.QtGui import QColor, QFont, QPainter, QPen

from horizon.presentation.themes import Theme

logger = logging.getLogger(__name__)


@dataclass
class HUDState:
    """Current state data for HUD rendering."""
    cursor_x: int = 0
    cursor_y: int = 0
    gesture_label: str = ""
    confidence: float = 0.0
    transcript: str = ""
    is_active: bool = True
    fps: float = 0.0
    latency_ms: float = 0.0
    landmarks: list[tuple[float, float]] = field(default_factory=list)
    show_debug: bool = False


class HUDRenderer:
    """Renders HUD elements onto the overlay QPainter."""

    def __init__(self, theme: Theme, screen_width: int, screen_height: int) -> None:
        self._theme = theme
        self._screen_w = screen_width
        self._screen_h = screen_height

    def render(self, painter: QPainter, state: HUDState) -> None:
        if not state.is_active:
            return

        self._render_gesture_info(painter, state)
        self._render_transcript(painter, state)
        self._render_status_bar(painter, state)

        if state.show_debug:
            self._render_debug_info(painter, state)

    def _render_gesture_info(self, painter: QPainter, state: HUDState) -> None:
        if not state.gesture_label or state.gesture_label == "none":
            return

        # Panel background
        panel_rect = QRectF(10, 10, 280, 80)
        bg = QColor(self._theme.background)
        painter.fillRect(panel_rect, bg)

        # Gesture name
        font = QFont("Segoe UI", self._theme.font_size, QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(QPen(QColor(self._theme.foreground)))
        painter.drawText(20, 38, f"Gesture: {state.gesture_label}")

        # Confidence bar
        bar_rect = QRectF(20, 50, 250, 10)
        painter.fillRect(bar_rect, QColor(self._theme.confidence_bar_bg))
        fill_rect = QRectF(20, 50, 250 * state.confidence, 10)
        painter.fillRect(fill_rect, QColor(self._theme.confidence_bar_fg))

        # Confidence percentage
        small_font = QFont("Segoe UI", self._theme.font_size - 4)
        painter.setFont(small_font)
        painter.drawText(20, 78, f"{state.confidence:.0%}")

    def _render_transcript(self, painter: QPainter, state: HUDState) -> None:
        if not state.transcript:
            return

        y = self._screen_h - 80
        panel_rect = QRectF(10, y, 500, 40)
        bg = QColor(self._theme.background)
        painter.fillRect(panel_rect, bg)

        font = QFont("Segoe UI", self._theme.font_size - 1)
        painter.setFont(font)
        painter.setPen(QPen(QColor(self._theme.accent)))
        painter.drawText(20, y + 26, f'"{state.transcript}"')

    def _render_status_bar(self, painter: QPainter, state: HUDState) -> None:
        x = self._screen_w - 120
        y = 10
        panel_rect = QRectF(x, y, 110, 30)
        bg = QColor(self._theme.background)
        painter.fillRect(panel_rect, bg)

        # Status indicator
        indicator_color = self._theme.status_active if state.is_active else self._theme.status_inactive
        painter.setBrush(QColor(indicator_color))
        painter.setPen(QPen(Qt.PenStyle.NoPen) if hasattr(QPen, 'NoPen') else QPen())
        painter.drawEllipse(int(x + 10), int(y + 10), 10, 10)

        # Label
        font = QFont("Segoe UI", self._theme.font_size - 3)
        painter.setFont(font)
        painter.setPen(QPen(QColor(self._theme.foreground)))
        label = "ACTIVE" if state.is_active else "PAUSED"
        painter.drawText(int(x + 28), int(y + 20), label)

    def _render_debug_info(self, painter: QPainter, state: HUDState) -> None:
        x = self._screen_w - 200
        y = 50
        panel_rect = QRectF(x, y, 190, 60)
        bg = QColor(self._theme.background)
        painter.fillRect(panel_rect, bg)

        font = QFont("Consolas", 10)
        painter.setFont(font)
        painter.setPen(QPen(QColor(self._theme.foreground)))
        painter.drawText(int(x + 10), int(y + 18), f"FPS: {state.fps:.1f}")
        painter.drawText(int(x + 10), int(y + 36), f"Latency: {state.latency_ms:.0f}ms")
        painter.drawText(int(x + 10), int(y + 54), f"Landmarks: {len(state.landmarks)}")

    def set_theme(self, theme: Theme) -> None:
        self._theme = theme


# Import needed for _render_status_bar
from PyQt6.QtCore import Qt
