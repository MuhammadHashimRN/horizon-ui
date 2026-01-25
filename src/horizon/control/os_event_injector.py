"""Win32 OS event injection via ctypes SendInput."""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging
import subprocess
import time

from horizon.event_bus import EventBus
from horizon.types import ActionType, Event, EventType, FusedAction, OSEvent

logger = logging.getLogger(__name__)

# Win32 constants
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_ABSOLUTE = 0x8000
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
WHEEL_DELTA = 120

# Virtual key code mapping
VK_MAP: dict[str, int] = {
    "alt": 0x12,
    "ctrl": 0x11,
    "shift": 0x10,
    "enter": 0x0D,
    "tab": 0x09,
    "escape": 0x1B,
    "space": 0x20,
    "backspace": 0x08,
    "delete": 0x2E,
    "left": 0x25,
    "up": 0x26,
    "right": 0x27,
    "down": 0x28,
    "home": 0x24,
    "end": 0x23,
    "pageup": 0x21,
    "pagedown": 0x22,
    "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73,
    "f5": 0x74, "f6": 0x75, "f7": 0x76, "f8": 0x77,
    "f9": 0x78, "f10": 0x79, "f11": 0x7A, "f12": 0x7B,
}


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.wintypes.LONG),
        ("dy", ctypes.wintypes.LONG),
        ("mouseData", ctypes.wintypes.DWORD),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.wintypes.WORD),
        ("wScan", ctypes.wintypes.WORD),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
    ]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.wintypes.DWORD),
        ("union", INPUT_UNION),
    ]


class OSEventInjector:
    """Injects mouse and keyboard events into the OS via Win32 SendInput.

    Subscribes to OS_EVENT events from the PolicyEngine and executes
    the corresponding OS-level input actions.
    """

    def __init__(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self._screen_width = ctypes.windll.user32.GetSystemMetrics(0)
        self._screen_height = ctypes.windll.user32.GetSystemMetrics(1)
        self.event_bus.subscribe(EventType.OS_EVENT, self._on_os_event)
        logger.info("OSEventInjector initialized (screen=%dx%d)", self._screen_width, self._screen_height)

    def _get_cursor_pos(self) -> tuple[int, int]:
        """Get current OS cursor position."""
        point = ctypes.wintypes.POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
        return point.x, point.y

    def _on_os_event(self, event: Event) -> None:
        action: FusedAction = event.data

        # For MOUSE_MOVE, use provided coordinates
        # For click/key actions, use current cursor position (avoids jump)
        if action.action == ActionType.MOUSE_MOVE:
            x = int(action.cursor_x) if action.cursor_x > 1.0 else int(action.cursor_x * self._screen_width)
            y = int(action.cursor_y) if action.cursor_y > 1.0 else int(action.cursor_y * self._screen_height)
        else:
            x, y = self._get_cursor_pos()

        os_event = OSEvent(
            action=action.action,
            x=x,
            y=y,
            params=action.params,
            keys=action.params.get("keys", []),
            key=action.params.get("key", ""),
            text=action.params.get("text", ""),
        )

        try:
            self._execute(os_event)
        except Exception:
            logger.exception("Failed to inject event: %s", os_event.action.value)

    def _execute(self, event: OSEvent) -> None:
        action = event.action

        if action == ActionType.MOUSE_MOVE:
            self._mouse_move(event.x, event.y)
        elif action == ActionType.LEFT_CLICK:
            self._mouse_click(event.x, event.y, button="left")
        elif action == ActionType.RIGHT_CLICK:
            self._mouse_click(event.x, event.y, button="right")
        elif action == ActionType.DOUBLE_CLICK:
            self._mouse_click(event.x, event.y, button="left")
            time.sleep(0.05)
            self._mouse_click(event.x, event.y, button="left")
        elif action in (ActionType.SCROLL_UP, ActionType.SCROLL_DOWN, ActionType.SCROLL):
            delta = event.delta if event.delta else (WHEEL_DELTA if action == ActionType.SCROLL_UP else -WHEEL_DELTA)
            self._mouse_scroll(delta)
        elif action == ActionType.ZOOM_IN:
            self._key_combo(["ctrl"], scroll_delta=WHEEL_DELTA)
        elif action == ActionType.ZOOM_OUT:
            self._key_combo(["ctrl"], scroll_delta=-WHEEL_DELTA)
        elif action == ActionType.KEY_PRESS:
            self._key_press(event.key)
        elif action == ActionType.KEY_COMBO:
            self._key_combo(event.keys)
        elif action == ActionType.TYPE_TEXT:
            self._type_text(event.text)
        elif action == ActionType.OPEN_APPLICATION:
            self._open_application(event.params.get("app", ""))
        elif action == ActionType.MINIMIZE_WINDOW:
            self._key_combo(["alt", "space"])
            time.sleep(0.1)
            self._key_press("n")
        elif action == ActionType.MAXIMIZE_WINDOW:
            self._key_combo(["alt", "space"])
            time.sleep(0.1)
            self._key_press("x")
        elif action == ActionType.SCREENSHOT:
            self._key_combo(["alt", "print_screen"])
        elif action == ActionType.LOCK_SCREEN:
            ctypes.windll.user32.LockWorkStation()
        else:
            logger.debug("Unhandled action: %s", action.value)

    def _mouse_move(self, x: int, y: int) -> None:
        abs_x = int(x * 65535 / self._screen_width)
        abs_y = int(y * 65535 / self._screen_height)

        inp = INPUT()
        inp.type = INPUT_MOUSE
        inp.union.mi.dx = abs_x
        inp.union.mi.dy = abs_y
        inp.union.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))

    def _mouse_click(self, x: int, y: int, button: str = "left") -> None:
        self._mouse_move(x, y)
        time.sleep(0.01)

        down_flag = MOUSEEVENTF_LEFTDOWN if button == "left" else MOUSEEVENTF_RIGHTDOWN
        up_flag = MOUSEEVENTF_LEFTUP if button == "left" else MOUSEEVENTF_RIGHTUP

        inp_down = INPUT()
        inp_down.type = INPUT_MOUSE
        inp_down.union.mi.dwFlags = down_flag

        inp_up = INPUT()
        inp_up.type = INPUT_MOUSE
        inp_up.union.mi.dwFlags = up_flag

        inputs = (INPUT * 2)(inp_down, inp_up)
        ctypes.windll.user32.SendInput(2, ctypes.byref(inputs), ctypes.sizeof(INPUT))

    def _mouse_scroll(self, delta: int) -> None:
        inp = INPUT()
        inp.type = INPUT_MOUSE
        inp.union.mi.mouseData = delta
        inp.union.mi.dwFlags = MOUSEEVENTF_WHEEL
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))

    def _key_press(self, key: str) -> None:
        vk = self._resolve_vk(key)
        inp_down = INPUT()
        inp_down.type = INPUT_KEYBOARD
        inp_down.union.ki.wVk = vk

        inp_up = INPUT()
        inp_up.type = INPUT_KEYBOARD
        inp_up.union.ki.wVk = vk
        inp_up.union.ki.dwFlags = KEYEVENTF_KEYUP

        inputs = (INPUT * 2)(inp_down, inp_up)
        ctypes.windll.user32.SendInput(2, ctypes.byref(inputs), ctypes.sizeof(INPUT))

    def _key_combo(self, keys: list[str], scroll_delta: int | None = None) -> None:
        # Press all modifier keys
        for key in keys:
            vk = self._resolve_vk(key)
            inp = INPUT()
            inp.type = INPUT_KEYBOARD
            inp.union.ki.wVk = vk
            ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))

        if scroll_delta is not None:
            self._mouse_scroll(scroll_delta)

        # Release all modifier keys in reverse order
        for key in reversed(keys):
            vk = self._resolve_vk(key)
            inp = INPUT()
            inp.type = INPUT_KEYBOARD
            inp.union.ki.wVk = vk
            inp.union.ki.dwFlags = KEYEVENTF_KEYUP
            ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))

    def _type_text(self, text: str) -> None:
        for char in text:
            inp_down = INPUT()
            inp_down.type = INPUT_KEYBOARD
            inp_down.union.ki.wScan = ord(char)
            inp_down.union.ki.dwFlags = KEYEVENTF_UNICODE

            inp_up = INPUT()
            inp_up.type = INPUT_KEYBOARD
            inp_up.union.ki.wScan = ord(char)
            inp_up.union.ki.dwFlags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP

            inputs = (INPUT * 2)(inp_down, inp_up)
            ctypes.windll.user32.SendInput(2, ctypes.byref(inputs), ctypes.sizeof(INPUT))

    def _open_application(self, app_name: str) -> None:
        if app_name:
            try:
                subprocess.Popen(["start", app_name], shell=True)
            except Exception:
                logger.exception("Failed to open application: %s", app_name)

    @staticmethod
    def _resolve_vk(key: str) -> int:
        key_lower = key.lower()
        if key_lower in VK_MAP:
            return VK_MAP[key_lower]
        if len(key) == 1:
            return ord(key.upper())
        logger.warning("Unknown key: %s", key)
        return 0

    def close(self) -> None:
        self.event_bus.unsubscribe(EventType.OS_EVENT, self._on_os_event)
        logger.info("OSEventInjector closed")
