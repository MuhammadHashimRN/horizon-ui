"""System tray icon and context menu."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import QMenu, QSystemTrayIcon, QWidget

logger = logging.getLogger(__name__)


class TrayIcon:
    """System tray icon with context menu for quick access controls."""

    def __init__(
        self,
        parent: QWidget,
        icon_path: str | Path = "assets/icons/tray_icon.png",
        on_toggle: Callable[[], None] | None = None,
        on_settings: Callable[[], None] | None = None,
        on_calibrate: Callable[[], None] | None = None,
        on_quit: Callable[[], None] | None = None,
    ) -> None:
        self._parent = parent
        self._on_toggle = on_toggle
        self._on_settings = on_settings
        self._on_calibrate = on_calibrate
        self._on_quit = on_quit

        self._tray = QSystemTrayIcon(parent)
        self._setup_icon(icon_path)
        self._setup_menu()
        self._tray.activated.connect(self._on_activated)

        logger.info("TrayIcon initialized")

    def _setup_icon(self, icon_path: str | Path) -> None:
        path = Path(icon_path)
        if path.exists():
            self._tray.setIcon(QIcon(str(path)))
        else:
            # Use a default icon
            self._tray.setIcon(QIcon())
        self._tray.setToolTip("Horizon UI — Touchless Control")

    def _setup_menu(self) -> None:
        menu = QMenu()

        self._toggle_action = QAction("Pause", self._parent)
        self._toggle_action.triggered.connect(self._handle_toggle)
        menu.addAction(self._toggle_action)

        menu.addSeparator()

        settings_action = QAction("Settings...", self._parent)
        settings_action.triggered.connect(self._handle_settings)
        menu.addAction(settings_action)

        calibrate_action = QAction("Calibrate...", self._parent)
        calibrate_action.triggered.connect(self._handle_calibrate)
        menu.addAction(calibrate_action)

        menu.addSeparator()

        about_action = QAction("About Horizon UI", self._parent)
        about_action.triggered.connect(self._show_about)
        menu.addAction(about_action)

        menu.addSeparator()

        quit_action = QAction("Quit", self._parent)
        quit_action.triggered.connect(self._handle_quit)
        menu.addAction(quit_action)

        self._tray.setContextMenu(menu)

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self._handle_toggle()

    def _handle_toggle(self) -> None:
        if self._on_toggle:
            self._on_toggle()

    def _handle_settings(self) -> None:
        if self._on_settings:
            self._on_settings()

    def _handle_calibrate(self) -> None:
        if self._on_calibrate:
            self._on_calibrate()

    def _handle_quit(self) -> None:
        if self._on_quit:
            self._on_quit()

    def _show_about(self) -> None:
        self._tray.showMessage(
            "Horizon UI",
            "Multimodal Desktop Overlay\nTouchless Control via Gesture & Voice\nv0.1.0",
            QSystemTrayIcon.MessageIcon.Information,
            3000,
        )

    def show(self) -> None:
        self._tray.show()

    def hide(self) -> None:
        self._tray.hide()

    def set_paused(self, paused: bool) -> None:
        self._toggle_action.setText("Resume" if paused else "Pause")
        tooltip = "Horizon UI — PAUSED" if paused else "Horizon UI — Active"
        self._tray.setToolTip(tooltip)

    def show_notification(self, title: str, message: str) -> None:
        self._tray.showMessage(title, message, QSystemTrayIcon.MessageIcon.Information, 3000)
