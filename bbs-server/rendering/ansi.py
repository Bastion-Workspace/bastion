"""
ANSI escape sequences and color themes for BBS output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class Theme:
    reset: str = "\033[0m"
    bold: str = "\033[1m"
    dim: str = "\033[2m"
    italic: str = "\033[3m"
    fg_green: str = "\033[32m"
    fg_yellow: str = "\033[33m"
    fg_cyan: str = "\033[36m"
    fg_white: str = "\033[37m"
    fg_bright_green: str = "\033[92m"
    fg_bright_yellow: str = "\033[93m"
    fg_bright_cyan: str = "\033[96m"
    clear_screen: str = "\033[2J\033[H"
    slow_blink: str = "\033[5m"
    slow_blink_off: str = "\033[25m"

    def title_color(self) -> str:
        return self.fg_bright_yellow + self.bold

    def header2_color(self) -> str:
        return self.fg_bright_cyan + self.bold

    def code_color(self) -> str:
        return self.fg_green

    def muted(self) -> str:
        return self.dim


def theme_from_name(name: str) -> Theme:
    n = (name or "green").lower().strip()
    if n == "none":
        return Theme(
            reset="",
            bold="",
            dim="",
            italic="",
            fg_green="",
            fg_yellow="",
            fg_cyan="",
            fg_white="",
            fg_bright_green="",
            fg_bright_yellow="",
            fg_bright_cyan="",
            clear_screen="",
            slow_blink="",
            slow_blink_off="",
        )
    if n == "amber":
        t = Theme()
        t.fg_green = "\033[33m"
        t.fg_bright_green = "\033[93m"
        t.fg_bright_yellow = "\033[93m"
        return t
    if n == "blue":
        t = Theme()
        t.fg_green = "\033[34m"
        t.fg_bright_green = "\033[94m"
        t.fg_bright_cyan = "\033[96m"
        return t
    return Theme()


def strip_ansi(text: str) -> str:
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", text)
