"""Active window and application context detection."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class ContextManager:
    """Detects the currently active application for app-specific overrides.

    Uses Win32 API to get the foreground window title and process name.
    """

    def __init__(self) -> None:
        self._cached_app: str = ""
        self._cached_title: str = ""
        logger.info("ContextManager initialized")

    def get_active_app(self) -> str:
        """Return the name of the currently active application."""
        try:
            import ctypes
            import ctypes.wintypes

            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()

            # Get window title
            length = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value

            # Get process name
            pid = ctypes.wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

            import psutil
            try:
                process = psutil.Process(pid.value)
                app_name = process.name().replace(".exe", "")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                app_name = self._extract_app_from_title(title)

            self._cached_app = app_name
            self._cached_title = title
            return app_name

        except ImportError:
            # psutil not available, fall back to title-based detection
            return self._get_app_from_title()
        except Exception:
            logger.debug("Failed to get active app", exc_info=True)
            return self._cached_app

    def _get_app_from_title(self) -> str:
        try:
            import ctypes

            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            length = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value
            self._cached_title = title
            app_name = self._extract_app_from_title(title)
            self._cached_app = app_name
            return app_name
        except Exception:
            return self._cached_app

    @staticmethod
    def _extract_app_from_title(title: str) -> str:
        """Best-effort extraction of app name from window title."""
        known_apps = {
            "PowerPoint": "PowerPoint",
            "Word": "Word",
            "Excel": "Excel",
            "Chrome": "Chrome",
            "Firefox": "Firefox",
            "Edge": "Edge",
            "VLC": "VLC",
            "Spotify": "Spotify",
            "Visual Studio Code": "VSCode",
            "Notepad": "Notepad",
        }
        for keyword, name in known_apps.items():
            if keyword.lower() in title.lower():
                return name
        # Return last segment after " - " (common pattern: "Document - App")
        parts = title.split(" - ")
        return parts[-1].strip() if parts else title

    def get_active_title(self) -> str:
        return self._cached_title
