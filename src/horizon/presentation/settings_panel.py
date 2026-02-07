"""Settings dialog for configuring Horizon UI."""

from __future__ import annotations

import logging

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class SettingsPanel(QDialog):
    """Settings dialog with tabs for all configuration categories."""

    def __init__(self, settings: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = dict(settings)
        self.setWindowTitle("Horizon UI Settings")
        self.setMinimumSize(500, 400)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        tabs = QTabWidget()
        tabs.addTab(self._build_general_tab(), "General")
        tabs.addTab(self._build_gesture_tab(), "Gestures")
        tabs.addTab(self._build_voice_tab(), "Voice")
        tabs.addTab(self._build_overlay_tab(), "Overlay")
        tabs.addTab(self._build_accessibility_tab(), "Accessibility")
        layout.addWidget(tabs)

        # Buttons
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._on_save)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        reset_btn = QPushButton("Reset Defaults")
        reset_btn.clicked.connect(self._on_reset)
        btn_layout.addWidget(reset_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(save_btn)
        layout.addLayout(btn_layout)

    def _build_general_tab(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)

        cam = self._settings.get("camera", {})
        self._camera_index = QSpinBox()
        self._camera_index.setRange(0, 10)
        self._camera_index.setValue(cam.get("device_index", 0))
        form.addRow("Camera Index:", self._camera_index)

        self._fps = QSpinBox()
        self._fps.setRange(10, 60)
        self._fps.setValue(cam.get("fps", 30))
        form.addRow("FPS:", self._fps)

        sys_settings = self._settings.get("system", {})
        self._hotkey = QLabel(sys_settings.get("global_hotkey", "Ctrl+Shift+H"))
        form.addRow("Global Hotkey:", self._hotkey)

        self._startup = QCheckBox()
        self._startup.setChecked(sys_settings.get("startup_with_os", False))
        form.addRow("Start with OS:", self._startup)

        self._log_level = QComboBox()
        self._log_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        current_level = sys_settings.get("log_level", "INFO")
        self._log_level.setCurrentText(current_level)
        form.addRow("Log Level:", self._log_level)

        return widget

    def _build_gesture_tab(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)

        gesture = self._settings.get("gesture", {})
        mp = self._settings.get("mediapipe", {})

        self._confidence_threshold = QDoubleSpinBox()
        self._confidence_threshold.setRange(0.1, 1.0)
        self._confidence_threshold.setSingleStep(0.05)
        self._confidence_threshold.setValue(gesture.get("confidence_threshold", 0.75))
        form.addRow("Confidence Threshold:", self._confidence_threshold)

        self._smoothing_alpha = QDoubleSpinBox()
        self._smoothing_alpha.setRange(0.05, 1.0)
        self._smoothing_alpha.setSingleStep(0.05)
        self._smoothing_alpha.setValue(gesture.get("smoothing_alpha", 0.3))
        form.addRow("Smoothing Alpha:", self._smoothing_alpha)

        self._temporal_window = QSpinBox()
        self._temporal_window.setRange(5, 30)
        self._temporal_window.setValue(gesture.get("temporal_window_frames", 15))
        form.addRow("Temporal Window (frames):", self._temporal_window)

        self._detection_conf = QDoubleSpinBox()
        self._detection_conf.setRange(0.1, 1.0)
        self._detection_conf.setSingleStep(0.05)
        self._detection_conf.setValue(mp.get("min_detection_confidence", 0.7))
        form.addRow("Detection Confidence:", self._detection_conf)

        self._tracking_conf = QDoubleSpinBox()
        self._tracking_conf.setRange(0.1, 1.0)
        self._tracking_conf.setSingleStep(0.05)
        self._tracking_conf.setValue(mp.get("min_tracking_confidence", 0.6))
        form.addRow("Tracking Confidence:", self._tracking_conf)

        return widget

    def _build_voice_tab(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)

        asr = self._settings.get("asr", {})

        self._asr_backend = QComboBox()
        self._asr_backend.addItems(["whisper", "vosk"])
        self._asr_backend.setCurrentText(asr.get("backend", "whisper"))
        form.addRow("ASR Backend:", self._asr_backend)

        self._whisper_model = QComboBox()
        self._whisper_model.addItems(["tiny", "base", "small", "medium"])
        self._whisper_model.setCurrentText(asr.get("whisper_model", "small"))
        form.addRow("Whisper Model:", self._whisper_model)

        self._language = QComboBox()
        self._language.addItems(["en", "es", "fr", "de", "zh", "ja", "ar"])
        self._language.setCurrentText(asr.get("language", "en"))
        form.addRow("Language:", self._language)

        return widget

    def _build_overlay_tab(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)

        overlay = self._settings.get("overlay", {})

        self._overlay_opacity = QSlider(Qt.Orientation.Horizontal)
        self._overlay_opacity.setRange(10, 100)
        self._overlay_opacity.setValue(int(overlay.get("opacity", 0.85) * 100))
        form.addRow("Opacity:", self._overlay_opacity)

        self._theme = QComboBox()
        self._theme.addItems(["dark", "light", "high_contrast"])
        self._theme.setCurrentText(overlay.get("theme", "dark"))
        form.addRow("Theme:", self._theme)

        self._show_skeleton = QCheckBox()
        self._show_skeleton.setChecked(overlay.get("show_hand_skeleton", True))
        form.addRow("Show Hand Skeleton:", self._show_skeleton)

        self._cursor_size = QSpinBox()
        self._cursor_size.setRange(8, 64)
        self._cursor_size.setValue(overlay.get("cursor_size", 24))
        form.addRow("Cursor Size:", self._cursor_size)

        return widget

    def _build_accessibility_tab(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)

        self._profile_combo = QComboBox()
        self._profile_combo.addItems(["None", "sterile", "low_vision", "voice_only"])
        form.addRow("Active Profile:", self._profile_combo)

        info = QLabel(
            "Accessibility profiles override specific settings.\n"
            "Select a profile to apply its configuration."
        )
        info.setWordWrap(True)
        form.addRow(info)

        return widget

    def _on_save(self) -> None:
        self._settings["camera"]["device_index"] = self._camera_index.value()
        self._settings["camera"]["fps"] = self._fps.value()
        self._settings["system"]["startup_with_os"] = self._startup.isChecked()
        self._settings["system"]["log_level"] = self._log_level.currentText()
        self._settings["gesture"]["confidence_threshold"] = self._confidence_threshold.value()
        self._settings["gesture"]["smoothing_alpha"] = self._smoothing_alpha.value()
        self._settings["gesture"]["temporal_window_frames"] = self._temporal_window.value()
        self._settings["mediapipe"]["min_detection_confidence"] = self._detection_conf.value()
        self._settings["mediapipe"]["min_tracking_confidence"] = self._tracking_conf.value()
        self._settings["asr"]["backend"] = self._asr_backend.currentText()
        self._settings["asr"]["whisper_model"] = self._whisper_model.currentText()
        self._settings["asr"]["language"] = self._language.currentText()
        self._settings["overlay"]["opacity"] = self._overlay_opacity.value() / 100.0
        self._settings["overlay"]["theme"] = self._theme.currentText()
        self._settings["overlay"]["show_hand_skeleton"] = self._show_skeleton.isChecked()
        self._settings["overlay"]["cursor_size"] = self._cursor_size.value()

        self.accept()

    def _on_reset(self) -> None:
        logger.info("Settings reset to defaults requested")
        self.reject()

    def get_settings(self) -> dict:
        return self._settings
