"""PyQt6 transparent overlay window for the HUD display."""

from __future__ import annotations

import ctypes
import logging
import sys

from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QFont
from PyQt6.QtWidgets import QApplication, QMainWindow

from horizon.event_bus import EventBus
from horizon.presentation.themes import Theme, get_theme
from horizon.types import ActionType, Event, EventType, FusedAction, GestureResult

logger = logging.getLogger(__name__)

# Win32 extended window styles for click-through
GWL_EXSTYLE = -20
WS_EX_TRANSPARENT = 0x00000020
WS_EX_LAYERED = 0x00080000


class OverlayWindow(QMainWindow):
    """Transparent always-on-top overlay for the Horizon UI HUD.

    Features:
    - Full-screen transparent window
    - Click-through (mouse events pass to underlying windows)
    - Renders cursor, hand skeleton, gesture labels, and status info
    """

    update_signal = pyqtSignal(dict)

    def __init__(
        self,
        event_bus: EventBus,
        theme_name: str = "dark",
        opacity: float = 0.85,
        show_skeleton: bool = True,
        cursor_size: int = 24,
    ) -> None:
        super().__init__()
        self.event_bus = event_bus
        self._theme: Theme = get_theme(theme_name)
        self._opacity = opacity
        self._show_skeleton = show_skeleton
        self._cursor_size = cursor_size

        # Current state for rendering
        self._cursor_x: int = 0
        self._cursor_y: int = 0
        self._gesture_label: str = ""
        self._confidence: float = 0.0
        self._transcript: str = ""
        self._is_active: bool = True
        self._landmarks: list[tuple[float, float]] = []

        self._setup_window()
        self._setup_signals()
        self._setup_timer()

        logger.info("OverlayWindow initialized (theme=%s)", theme_name)

    def _setup_window(self) -> None:
        self.setWindowTitle("Horizon UI Overlay")
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # Full screen
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.geometry()
            self.setGeometry(geo)

        # Make window click-through on Windows
        if sys.platform == "win32":
            try:
                hwnd = int(self.winId())
                user32 = ctypes.windll.user32
                style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
                user32.SetWindowLongW(
                    hwnd, GWL_EXSTYLE, style | WS_EX_TRANSPARENT | WS_EX_LAYERED
                )
            except Exception:
                logger.warning("Failed to set click-through style")

    def _setup_signals(self) -> None:
        self.update_signal.connect(self._handle_update)
        self.event_bus.subscribe(EventType.HAND_DETECTED, self._on_hand_detected)
        self.event_bus.subscribe(EventType.OS_EVENT, self._on_os_event)
        self.event_bus.subscribe(EventType.GESTURE_RESULT, self._on_gesture)
        self.event_bus.subscribe(EventType.TRANSCRIPT, self._on_transcript)
        self.event_bus.subscribe(EventType.SYSTEM_STATE, self._on_system_state)

    def _setup_timer(self) -> None:
        self._render_timer = QTimer()
        self._render_timer.timeout.connect(self.update)
        self._render_timer.start(33)  # ~30 FPS

    def _on_os_event(self, event: Event) -> None:
        """Track cursor position from CursorTracker's mouse_move events."""
        action: FusedAction = event.data
        if action.action == ActionType.MOUSE_MOVE:
            self.update_signal.emit({
                "cursor_px": int(action.cursor_x),
                "cursor_py": int(action.cursor_y),
            })

    def _on_hand_detected(self, event: Event) -> None:
        """Update landmarks every frame for smooth skeleton rendering."""
        data = event.data
        if not data.get("has_hands"):
            self.update_signal.emit({"landmarks": [], "gesture": "", "confidence": 0.0})
            return
        result_obj = data.get("result")
        if result_obj and result_obj.hand_landmarks:
            lms = result_obj.hand_landmarks[0]
            self.update_signal.emit({
                "landmarks": [(lm.x, lm.y) for lm in lms],
            })
        else:
            self.update_signal.emit({"landmarks": [], "gesture": "", "confidence": 0.0})

    def _on_gesture(self, event: Event) -> None:
        data = event.data
        if isinstance(data, GestureResult):
            self.update_signal.emit({
                "cursor_x": data.cursor_x,
                "cursor_y": data.cursor_y,
                "gesture": data.label.value,
                "confidence": data.confidence,
            })

    def _on_transcript(self, event: Event) -> None:
        self.update_signal.emit({"transcript": event.data})

    def _on_system_state(self, event: Event) -> None:
        data = event.data
        if isinstance(data, dict):
            self.update_signal.emit(data)

    def _handle_update(self, data: dict) -> None:
        if "cursor_px" in data:
            self._cursor_x = data["cursor_px"]
            self._cursor_y = data["cursor_py"]
        if "gesture" in data:
            self._gesture_label = data["gesture"]
        if "confidence" in data:
            self._confidence = data["confidence"]
        if "transcript" in data:
            self._transcript = data["transcript"]
        if "landmarks" in data:
            self._landmarks = data["landmarks"]
        if "active" in data:
            self._is_active = data["active"]

    def paintEvent(self, event) -> None:
        if not self._is_active:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        self._draw_cursor(painter)
        if self._show_skeleton and self._landmarks:
            self._draw_skeleton(painter)
        self._draw_gesture_label(painter)
        self._draw_confidence_bar(painter)
        self._draw_transcript(painter)
        self._draw_status(painter)

        painter.end()

    def _draw_cursor(self, painter: QPainter) -> None:
        color = QColor(self._theme.cursor_color)
        color.setAlpha(int(255 * self._opacity))
        pen = QPen(color, 3)
        painter.setPen(pen)
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 80))

        r = self._cursor_size // 2
        painter.drawEllipse(self._cursor_x - r, self._cursor_y - r, self._cursor_size, self._cursor_size)

        # Crosshair
        painter.drawLine(self._cursor_x - r - 4, self._cursor_y, self._cursor_x - r + 2, self._cursor_y)
        painter.drawLine(self._cursor_x + r - 2, self._cursor_y, self._cursor_x + r + 4, self._cursor_y)
        painter.drawLine(self._cursor_x, self._cursor_y - r - 4, self._cursor_x, self._cursor_y - r + 2)
        painter.drawLine(self._cursor_x, self._cursor_y + r - 2, self._cursor_x, self._cursor_y + r + 4)

    def _draw_skeleton(self, painter: QPainter) -> None:
        if not self._landmarks:
            return

        screen = QApplication.primaryScreen()
        if not screen:
            return
        geo = screen.geometry()

        color = QColor(self._theme.skeleton_color)
        color.setAlpha(150)
        pen = QPen(color, 2)
        painter.setPen(pen)

        points = [(int(x * geo.width()), int(y * geo.height())) for x, y in self._landmarks]

        # Draw connections (simplified hand topology)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),       # Index
            (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
            (0, 13), (13, 14), (14, 15), (15, 16), # Ring
            (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
            (5, 9), (9, 13), (13, 17),             # Palm
        ]
        for a, b in connections:
            if a < len(points) and b < len(points):
                painter.drawLine(points[a][0], points[a][1], points[b][0], points[b][1])

        # Draw joints
        painter.setBrush(color)
        for px, py in points:
            painter.drawEllipse(px - 3, py - 3, 6, 6)

    def _draw_gesture_label(self, painter: QPainter) -> None:
        if not self._gesture_label or self._gesture_label == "none":
            return

        font = QFont("Segoe UI", self._theme.font_size, QFont.Weight.Bold)
        painter.setFont(font)
        color = QColor(self._theme.foreground)
        color.setAlpha(220)
        painter.setPen(QPen(color))

        text = f"Gesture: {self._gesture_label}"
        painter.drawText(20, 40, text)

    def _draw_confidence_bar(self, painter: QPainter) -> None:
        bar_x, bar_y = 20, 55
        bar_w, bar_h = 200, 12

        # Background
        bg = QColor(self._theme.confidence_bar_bg)
        painter.fillRect(bar_x, bar_y, bar_w, bar_h, bg)

        # Foreground
        fg = QColor(self._theme.confidence_bar_fg)
        fill_w = int(bar_w * self._confidence)
        painter.fillRect(bar_x, bar_y, fill_w, bar_h, fg)

    def _draw_transcript(self, painter: QPainter) -> None:
        if not self._transcript:
            return

        font = QFont("Segoe UI", self._theme.font_size - 2)
        painter.setFont(font)
        color = QColor(self._theme.accent)
        color.setAlpha(200)
        painter.setPen(QPen(color))

        screen = QApplication.primaryScreen()
        if screen:
            y = screen.geometry().height() - 60
            painter.drawText(20, y, f'Voice: "{self._transcript}"')

    def _draw_status(self, painter: QPainter) -> None:
        screen = QApplication.primaryScreen()
        if not screen:
            return

        geo = screen.geometry()
        indicator_size = 10
        x = geo.width() - 30
        y = 20

        color_str = self._theme.status_active if self._is_active else self._theme.status_inactive
        color = QColor(color_str)
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(x, y, indicator_size, indicator_size)

    def set_theme(self, theme_name: str) -> None:
        self._theme = get_theme(theme_name)

    def set_active(self, active: bool) -> None:
        self._is_active = active

    def close_overlay(self) -> None:
        self.event_bus.unsubscribe(EventType.HAND_DETECTED, self._on_hand_detected)
        self.event_bus.unsubscribe(EventType.OS_EVENT, self._on_os_event)
        self.event_bus.unsubscribe(EventType.GESTURE_RESULT, self._on_gesture)
        self.event_bus.unsubscribe(EventType.TRANSCRIPT, self._on_transcript)
        self.event_bus.unsubscribe(EventType.SYSTEM_STATE, self._on_system_state)
        self._render_timer.stop()
        self.close()
        logger.info("OverlayWindow closed")
