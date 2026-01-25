"""OS abstraction layer for platform-specific operations."""

from __future__ import annotations

import logging
import platform
import sys
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class PlatformInterface(ABC):
    """Abstract interface for platform-specific operations."""

    @abstractmethod
    def get_screen_size(self) -> tuple[int, int]: ...

    @abstractmethod
    def get_foreground_window_title(self) -> str: ...

    @abstractmethod
    def set_dpi_aware(self) -> None: ...

    @abstractmethod
    def get_mouse_position(self) -> tuple[int, int]: ...


class WindowsPlatform(PlatformInterface):
    """Windows-specific implementation using ctypes and Win32 API."""

    def __init__(self) -> None:
        import ctypes
        self._user32 = ctypes.windll.user32

    def get_screen_size(self) -> tuple[int, int]:
        self.set_dpi_aware()
        return (
            self._user32.GetSystemMetrics(0),
            self._user32.GetSystemMetrics(1),
        )

    def get_foreground_window_title(self) -> str:
        import ctypes
        hwnd = self._user32.GetForegroundWindow()
        length = self._user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        self._user32.GetWindowTextW(hwnd, buf, length + 1)
        return buf.value

    def set_dpi_aware(self) -> None:
        try:
            self._user32.SetProcessDPIAware()
        except Exception:
            pass

    def get_mouse_position(self) -> tuple[int, int]:
        import ctypes
        import ctypes.wintypes
        point = ctypes.wintypes.POINT()
        self._user32.GetCursorPos(ctypes.byref(point))
        return point.x, point.y


class StubPlatform(PlatformInterface):
    """Stub platform for unsupported operating systems."""

    def get_screen_size(self) -> tuple[int, int]:
        return 1920, 1080

    def get_foreground_window_title(self) -> str:
        return ""

    def set_dpi_aware(self) -> None:
        pass

    def get_mouse_position(self) -> tuple[int, int]:
        return 0, 0


def get_platform() -> PlatformInterface:
    """Return the appropriate platform implementation."""
    if sys.platform == "win32":
        return WindowsPlatform()
    else:
        logger.warning("Unsupported platform: %s. Using stub.", platform.system())
        return StubPlatform()
