"""Device enumeration and health checking for cameras and microphones."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import sounddevice as sd

logger = logging.getLogger(__name__)


@dataclass
class CameraInfo:
    index: int
    name: str
    available: bool
    width: int = 0
    height: int = 0


@dataclass
class MicrophoneInfo:
    index: int
    name: str
    available: bool
    sample_rate: float = 0.0
    channels: int = 0


class DeviceManager:
    """Enumerates and checks availability of cameras and microphones."""

    @staticmethod
    def enumerate_cameras(max_index: int = 5) -> list[CameraInfo]:
        cameras: list[CameraInfo] = []
        for i in range(max_index):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cameras.append(CameraInfo(
                    index=i,
                    name=f"Camera {i}",
                    available=True,
                    width=w,
                    height=h,
                ))
                cap.release()
            else:
                cameras.append(CameraInfo(index=i, name=f"Camera {i}", available=False))
        return cameras

    @staticmethod
    def enumerate_microphones() -> list[MicrophoneInfo]:
        mics: list[MicrophoneInfo] = []
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                mics.append(MicrophoneInfo(
                    index=i,
                    name=dev["name"],
                    available=True,
                    sample_rate=dev["default_samplerate"],
                    channels=dev["max_input_channels"],
                ))
        return mics

    @staticmethod
    def check_camera(index: int) -> bool:
        cap = cv2.VideoCapture(index)
        available = cap.isOpened()
        if available:
            ret, _ = cap.read()
            available = ret
        cap.release()
        return available

    @staticmethod
    def check_microphone() -> bool:
        try:
            devices = sd.query_devices()
            default_input = sd.default.device[0]
            if default_input is not None and default_input >= 0:
                return devices[default_input]["max_input_channels"] > 0
            # Check if any input device exists
            return any(d["max_input_channels"] > 0 for d in devices)
        except Exception:
            logger.exception("Failed to check microphone")
            return False
