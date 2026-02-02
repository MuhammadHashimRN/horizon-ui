"""Theme definitions for the overlay UI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Theme:
    name: str
    background: str
    foreground: str
    accent: str
    cursor_color: str
    skeleton_color: str
    confidence_bar_bg: str
    confidence_bar_fg: str
    status_active: str
    status_inactive: str
    text_shadow: bool
    font_size: int


THEMES: dict[str, Theme] = {
    "dark": Theme(
        name="Dark",
        background="rgba(20, 20, 20, 200)",
        foreground="#FFFFFF",
        accent="#00AAFF",
        cursor_color="#00FF88",
        skeleton_color="#00AAFF",
        confidence_bar_bg="rgba(255, 255, 255, 40)",
        confidence_bar_fg="#00FF88",
        status_active="#00FF88",
        status_inactive="#FF4444",
        text_shadow=True,
        font_size=14,
    ),
    "light": Theme(
        name="Light",
        background="rgba(240, 240, 240, 200)",
        foreground="#222222",
        accent="#0066CC",
        cursor_color="#00AA44",
        skeleton_color="#0066CC",
        confidence_bar_bg="rgba(0, 0, 0, 40)",
        confidence_bar_fg="#00AA44",
        status_active="#00AA44",
        status_inactive="#CC0000",
        text_shadow=False,
        font_size=14,
    ),
    "high_contrast": Theme(
        name="High Contrast",
        background="rgba(0, 0, 0, 240)",
        foreground="#FFFF00",
        accent="#FFFF00",
        cursor_color="#FFFF00",
        skeleton_color="#FFFFFF",
        confidence_bar_bg="rgba(255, 255, 255, 80)",
        confidence_bar_fg="#FFFF00",
        status_active="#00FF00",
        status_inactive="#FF0000",
        text_shadow=True,
        font_size=18,
    ),
}


def get_theme(name: str) -> Theme:
    return THEMES.get(name, THEMES["dark"])
