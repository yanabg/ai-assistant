"""Visual design tokens: palette, spacing, and font resolution.

Kept deliberately small and centralized so the UI reads as considered rather
than a default ``ttk`` demo.
"""

from __future__ import annotations

import tkinter.font as tkfont
from dataclasses import dataclass
from tkinter import Misc


@dataclass(frozen=True)
class Palette:
    """Colour tokens for the whole application."""

    window: str = "#EEF0F3"
    surface: str = "#FFFFFF"
    text_primary: str = "#1F2933"
    text_muted: str = "#6B7280"

    user_bg: str = "#2F6FED"
    user_fg: str = "#FFFFFF"
    assistant_bg: str = "#EDEFF3"
    assistant_fg: str = "#1F2933"
    error_bg: str = "#FCEBE9"
    error_fg: str = "#B42318"
    tool_fg: str = "#6C4CE0"

    accent: str = "#2F6FED"
    accent_active: str = "#2559C9"
    ghost_bg: str = "#E6E9EE"
    ghost_active: str = "#D8DCE3"
    disabled_bg: str = "#DDE1E6"
    disabled_fg: str = "#9AA1AC"
    border: str = "#DCE0E6"


PALETTE = Palette()


@dataclass(frozen=True)
class Spacing:
    """Pixel spacing tokens used across the layout and text tags."""

    gutter: int = 16
    transcript_padx: int = 20
    transcript_pady: int = 16
    bubble_gap: int = 8
    bubble_inset: int = 132


SPACING = Spacing()


_PREFERRED_FAMILIES: tuple[str, ...] = (
    "SF Pro Text",
    "Segoe UI",
    "Helvetica Neue",
    "Helvetica",
    "Arial",
    "DejaVu Sans",
)

_PREFERRED_MONO: tuple[str, ...] = (
    "SF Mono",
    "Menlo",
    "Consolas",
    "DejaVu Sans Mono",
    "Courier New",
)


def _pick(available: frozenset[str], preferred: tuple[str, ...], fallback: str) -> str:
    for family in preferred:
        if family in available:
            return family
    return fallback


def resolve_fonts(root: Misc) -> tuple[str, str]:
    """Return ``(ui_family, mono_family)`` best available for ``root``."""
    available = frozenset(tkfont.families(root))
    default = str(tkfont.nametofont("TkDefaultFont").actual("family"))
    fixed = str(tkfont.nametofont("TkFixedFont").actual("family"))
    ui_family = _pick(available, _PREFERRED_FAMILIES, default)
    mono_family = _pick(available, _PREFERRED_MONO, fixed)
    return ui_family, mono_family
