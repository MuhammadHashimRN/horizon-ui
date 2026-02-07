"""First-run calibration wizard for mapping camera to screen coordinates."""

from __future__ import annotations

import logging

import numpy as np
from PyQt6.QtCore import QPoint, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QApplication, QDialog, QLabel, QPushButton, QVBoxLayout

logger = logging.getLogger(__name__)

# Screen corner targets (normalized coordinates)
CALIBRATION_TARGETS = [
    (0.15, 0.15),   # Top-left
    (0.85, 0.15),   # Top-right
    (0.85, 0.85),   # Bottom-right
    (0.15, 0.85),   # Bottom-left
]


class CalibrationWizard(QDialog):
    """Step-by-step calibration wizard.

    Instructs the user to point at each screen corner, records the
    corresponding camera landmark positions, and computes a homography
    matrix for accurate coordinate mapping.
    """

    calibration_complete = pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Horizon UI — Calibration")
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        screen = QApplication.primaryScreen()
        if screen:
            self.setGeometry(screen.geometry())

        self._current_step = 0
        self._camera_points: list[tuple[float, float]] = []
        self._current_camera_point: tuple[float, float] | None = None
        self._countdown = 3
        self._timer = QTimer()
        self._timer.timeout.connect(self._on_tick)

        self._build_ui()
        logger.info("CalibrationWizard initialized")

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._instruction_label = QLabel("Point at the target marker")
        self._instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._instruction_label.setStyleSheet(
            "color: white; font-size: 24px; background: rgba(0,0,0,150); padding: 20px; border-radius: 10px;"
        )
        layout.addWidget(self._instruction_label)

        self._countdown_label = QLabel("")
        self._countdown_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._countdown_label.setStyleSheet(
            "color: #00FF88; font-size: 48px; font-weight: bold;"
        )
        layout.addWidget(self._countdown_label)

        self._skip_btn = QPushButton("Skip Calibration")
        self._skip_btn.setStyleSheet(
            "color: white; background: rgba(100,100,100,150); padding: 10px; border-radius: 5px;"
        )
        self._skip_btn.clicked.connect(self.reject)
        layout.addWidget(self._skip_btn, alignment=Qt.AlignmentFlag.AlignCenter)

    def start(self) -> None:
        self._current_step = 0
        self._camera_points = []
        self._update_step()
        self.show()

    def set_camera_point(self, x: float, y: float) -> None:
        """Called by the perception layer with current camera coordinates."""
        self._current_camera_point = (x, y)

    def _update_step(self) -> None:
        if self._current_step >= len(CALIBRATION_TARGETS):
            self._finish_calibration()
            return

        target = CALIBRATION_TARGETS[self._current_step]
        self._instruction_label.setText(
            f"Step {self._current_step + 1}/{len(CALIBRATION_TARGETS)}: "
            f"Point at the green target"
        )
        self._countdown = 3
        self._countdown_label.setText(str(self._countdown))
        self._timer.start(1000)
        self.update()

    def _on_tick(self) -> None:
        self._countdown -= 1
        if self._countdown <= 0:
            self._timer.stop()
            self._capture_point()
        else:
            self._countdown_label.setText(str(self._countdown))

    def _capture_point(self) -> None:
        if self._current_camera_point:
            self._camera_points.append(self._current_camera_point)
            logger.info(
                "Calibration point %d captured: camera=(%.3f, %.3f)",
                self._current_step, *self._current_camera_point,
            )
        else:
            # Use a default if no camera point received
            self._camera_points.append(CALIBRATION_TARGETS[self._current_step])
            logger.warning("No camera point received for step %d, using default", self._current_step)

        self._current_step += 1
        self._update_step()

    def _finish_calibration(self) -> None:
        src_points = np.array(self._camera_points, dtype=np.float32)
        dst_points = np.array(CALIBRATION_TARGETS, dtype=np.float32)

        self.calibration_complete.emit(src_points, dst_points)
        logger.info("Calibration complete")
        self.accept()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Semi-transparent background
        painter.fillRect(self.rect(), QColor(0, 0, 0, 120))

        # Draw target marker
        if self._current_step < len(CALIBRATION_TARGETS):
            target = CALIBRATION_TARGETS[self._current_step]
            screen = QApplication.primaryScreen()
            if screen:
                geo = screen.geometry()
                tx = int(target[0] * geo.width())
                ty = int(target[1] * geo.height())

                # Pulsing target circle
                pen = QPen(QColor("#00FF88"), 4)
                painter.setPen(pen)
                painter.setBrush(QColor(0, 255, 136, 60))
                size = 40
                painter.drawEllipse(tx - size // 2, ty - size // 2, size, size)

                # Crosshair
                painter.drawLine(tx - 25, ty, tx + 25, ty)
                painter.drawLine(tx, ty - 25, tx, ty + 25)

        # Draw completed points
        for i, (cx, cy) in enumerate(self._camera_points):
            screen = QApplication.primaryScreen()
            if screen:
                geo = screen.geometry()
                target = CALIBRATION_TARGETS[i]
                tx = int(target[0] * geo.width())
                ty = int(target[1] * geo.height())
                painter.setPen(QPen(QColor("#AAAAAA"), 2))
                painter.setBrush(QColor(100, 100, 100, 100))
                painter.drawEllipse(tx - 15, ty - 15, 30, 30)

                # Checkmark
                font = QFont("Segoe UI", 14, QFont.Weight.Bold)
                painter.setFont(font)
                painter.setPen(QPen(QColor("#00FF88")))
                painter.drawText(tx - 6, ty + 6, "✓")

        painter.end()
