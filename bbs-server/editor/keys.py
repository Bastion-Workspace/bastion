"""
Telnet / terminal key parsing for the BBS document editor (CSI, SS3, UTF-8).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List


class KeyKind(Enum):
    CHAR = auto()
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()
    HOME = auto()
    END = auto()
    PGUP = auto()
    PGDN = auto()
    CTRL_UP = auto()
    CTRL_DOWN = auto()
    BACKSPACE = auto()
    DELETE = auto()
    ENTER = auto()
    TAB = auto()
    SAVE = auto()
    QUIT = auto()
    HELP = auto()
    ESCAPE = auto()
    AI_CHAT = auto()


@dataclass(frozen=True)
class KeyEvent:
    kind: KeyKind
    char: str = ""


def _utf8_follow_len(lead: int) -> int:
    if lead < 0x80:
        return 0
    if (lead & 0xE0) == 0xC0:
        return 1
    if (lead & 0xF0) == 0xE0:
        return 2
    if (lead & 0xF8) == 0xF0:
        return 3
    return -1


class EditorKeyParser:
    """Incremental parser: feed one byte, receive zero or more KeyEvents."""

    def __init__(self) -> None:
        self._expect_lf_after_cr = False
        self._utf8_buf: bytearray = bytearray()
        self._utf8_need = 0
        self._state = "normal"
        self._csi: bytearray = bytearray()

    def reset(self) -> None:
        self.__init__()

    def feed(self, byte: int) -> List[KeyEvent]:
        b = byte & 0xFF

        if self._utf8_need > 0:
            if 0x80 <= b <= 0xBF:
                self._utf8_buf.append(b)
                self._utf8_need -= 1
                if self._utf8_need == 0:
                    return self._flush_utf8()
                return []
            self._utf8_buf.clear()
            self._utf8_need = 0

        if self._state == "esc":
            return self._feed_after_esc(b)
        if self._state == "csi":
            return self._feed_csi(b)
        if self._state == "ss3":
            self._state = "normal"
            return self._map_ss3(b)

        if b == 0x1B:
            self._state = "esc"
            return []

        if b in (0x0D, 0x0A):
            if b == 0x0A:
                if self._expect_lf_after_cr:
                    self._expect_lf_after_cr = False
                    return []
                return [KeyEvent(KeyKind.ENTER)]
            self._expect_lf_after_cr = True
            return [KeyEvent(KeyKind.ENTER)]

        self._expect_lf_after_cr = False

        if b in (0x7F, 0x08):
            return [KeyEvent(KeyKind.BACKSPACE)]
        if b == 0x09:
            return [KeyEvent(KeyKind.TAB)]
        # Emacs-style motion (works when VT100 arrow CSI is not sent, e.g. some DOS telnet)
        if b == 0x10:
            return [KeyEvent(KeyKind.UP)]
        if b == 0x0E:
            return [KeyEvent(KeyKind.DOWN)]
        if b == 0x02:
            return [KeyEvent(KeyKind.LEFT)]
        if b == 0x06:
            return [KeyEvent(KeyKind.RIGHT)]
        if b == 0x13:
            return [KeyEvent(KeyKind.SAVE)]
        if b == 0x11:
            return [KeyEvent(KeyKind.QUIT)]
        if b == 0x07:
            return [KeyEvent(KeyKind.HELP)]
        if b == 0x03:
            return [KeyEvent(KeyKind.QUIT)]

        if b == 0x01:
            return [KeyEvent(KeyKind.AI_CHAT)]

        if b < 0x20:
            return []

        need = _utf8_follow_len(b)
        if need < 0:
            return []
        if need == 0:
            return [KeyEvent(KeyKind.CHAR, chr(b))]
        self._utf8_buf = bytearray([b])
        self._utf8_need = need
        return []

    def _feed_after_esc(self, b: int) -> List[KeyEvent]:
        if b == 0x5B:
            self._state = "csi"
            self._csi.clear()
            return []
        if b == 0x4F:
            self._state = "ss3"
            return []
        if b == 0x1B:
            self._state = "normal"
            return [KeyEvent(KeyKind.ESCAPE)]
        self._state = "normal"
        return [KeyEvent(KeyKind.CHAR, chr(b))]

    def _feed_csi(self, b: int) -> List[KeyEvent]:
        self._csi.append(b)
        if 0x40 <= b <= 0x7E:
            raw = bytes(self._csi)
            self._csi.clear()
            self._state = "normal"
            return self._parse_csi(raw)
        if len(self._csi) > 64:
            self._csi.clear()
            self._state = "normal"
        return []

    def _flush_utf8(self) -> List[KeyEvent]:
        try:
            s = self._utf8_buf.decode("utf-8")
        except UnicodeDecodeError:
            s = "\uFFFD"
        self._utf8_buf.clear()
        if not s:
            return []
        return [KeyEvent(KeyKind.CHAR, s)]

    def _map_ss3(self, b: int) -> List[KeyEvent]:
        if b == 0x41:
            return [KeyEvent(KeyKind.UP)]
        if b == 0x42:
            return [KeyEvent(KeyKind.DOWN)]
        if b == 0x43:
            return [KeyEvent(KeyKind.RIGHT)]
        if b == 0x44:
            return [KeyEvent(KeyKind.LEFT)]
        if b == 0x48:
            return [KeyEvent(KeyKind.HOME)]
        if b == 0x46:
            return [KeyEvent(KeyKind.END)]
        return [KeyEvent(KeyKind.CHAR, chr(b))]

    def _parse_csi(self, raw: bytes) -> List[KeyEvent]:
        if not raw or len(raw) < 2:
            return []
        final = raw[-1]
        inner = raw[:-1].decode("ascii", errors="ignore")
        if final == ord("~"):
            parts = inner.split(";") if inner else []
            nums: List[int] = []
            for p in parts:
                p = p.strip()
                if p.isdigit():
                    nums.append(int(p))
            n0 = nums[0] if nums else 0
            if n0 == 1:
                return [KeyEvent(KeyKind.HOME)]
            if n0 == 4:
                return [KeyEvent(KeyKind.END)]
            if n0 == 5:
                return [KeyEvent(KeyKind.PGUP)]
            if n0 == 6:
                return [KeyEvent(KeyKind.PGDN)]
            if n0 == 3:
                return [KeyEvent(KeyKind.DELETE)]
            return []

        parts = inner.split(";") if inner else []
        modifier = 1
        if len(parts) >= 2:
            last = parts[-1]
            if last.isdigit():
                modifier = int(last)
        key_part = final
        if modifier == 5:
            if key_part == 0x41:
                return [KeyEvent(KeyKind.CTRL_UP)]
            if key_part == 0x42:
                return [KeyEvent(KeyKind.CTRL_DOWN)]
        if key_part == 0x41:
            return [KeyEvent(KeyKind.UP)]
        if key_part == 0x42:
            return [KeyEvent(KeyKind.DOWN)]
        if key_part == 0x43:
            return [KeyEvent(KeyKind.RIGHT)]
        if key_part == 0x44:
            return [KeyEvent(KeyKind.LEFT)]
        if key_part == 0x48:
            return [KeyEvent(KeyKind.HOME)]
        if key_part == 0x46:
            return [KeyEvent(KeyKind.END)]
        return []
