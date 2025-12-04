"""Shared data types used across all Horizon UI layers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class GestureLabel(Enum):
    POINT = "point"
    PINCH = "pinch"
    FIST = "fist"
    OPEN_PALM = "open_palm"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    PINCH_SPREAD = "pinch_spread"
    PINCH_CLOSE = "pinch_close"
    TWO_FINGER_SCROLL = "two_finger_scroll"
    THUMBS_UP = "thumbs_up"
    NONE = "none"


class ActionType(Enum):
    MOUSE_MOVE = "mouse_move"
    LEFT_CLICK = "left_click"
    RIGHT_CLICK = "right_click"
    DOUBLE_CLICK = "double_click"
    SCROLL_UP = "scroll_up"
    SCROLL_DOWN = "scroll_down"
    SCROLL = "scroll"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    KEY_PRESS = "key_press"
    KEY_COMBO = "key_combo"
    TYPE_TEXT = "type_text"
    DRAG_START = "drag_start"
    DRAG_END = "drag_end"
    PAUSE_TRACKING = "pause_tracking"
    CONFIRM = "confirm"
    OPEN_APPLICATION = "open_application"
    MINIMIZE_WINDOW = "minimize_window"
    MAXIMIZE_WINDOW = "maximize_window"
    TOGGLE_MIC_MUTE = "toggle_mic_mute"
    PAUSE_VOICE = "pause_voice"
    RESUME_VOICE = "resume_voice"
    BRIGHTNESS_UP = "brightness_up"
    BRIGHTNESS_DOWN = "brightness_down"
    VOLUME_UP = "volume_up"
    VOLUME_DOWN = "volume_down"
    SCREENSHOT = "screenshot"
    LOCK_SCREEN = "lock_screen"
    SHOW_HELP_OVERLAY = "show_help_overlay"
    START_CALIBRATION = "start_calibration"
    ACTIVATE_PROFILE = "activate_profile"


class EventType(Enum):
    FRAME = auto()
    AUDIO_CHUNK = auto()
    HAND_DETECTED = auto()
    LANDMARKS = auto()
    GESTURE_RESULT = auto()
    SPEECH_SEGMENT = auto()
    TRANSCRIPT = auto()
    VOICE_INTENT = auto()
    FUSED_ACTION = auto()
    OS_EVENT = auto()
    SYSTEM_STATE = auto()
    OVERLAY_UPDATE = auto()
    SETTINGS_CHANGED = auto()
    PROFILE_CHANGED = auto()
    PLUGIN_EVENT = auto()
    CALIBRATION = auto()
    ERROR = auto()


class InputSource(Enum):
    GESTURE = "gesture"
    VOICE = "voice"
    FUSED = "fused"


@dataclass
class Landmark:
    x: float
    y: float
    z: float


@dataclass
class LandmarkSet:
    landmarks: list[Landmark]
    handedness: str = "Right"
    timestamp: float = field(default_factory=time.time)

    @property
    def count(self) -> int:
        return len(self.landmarks)


@dataclass
class GestureResult:
    label: GestureLabel
    confidence: float
    landmarks: LandmarkSet | None = None
    cursor_x: float = 0.0
    cursor_y: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class VoiceIntent:
    action: ActionType
    transcript: str = ""
    confidence: float = 0.0
    params: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class FusedAction:
    action: ActionType
    source: InputSource = InputSource.FUSED
    confidence: float = 0.0
    params: dict[str, Any] = field(default_factory=dict)
    cursor_x: float = 0.0
    cursor_y: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class OSEvent:
    action: ActionType
    x: int = 0
    y: int = 0
    delta: int = 0
    key: str = ""
    keys: list[str] = field(default_factory=list)
    text: str = ""
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Event:
    type: EventType
    data: Any = None
    timestamp: float = field(default_factory=time.time)
    source: str = ""
