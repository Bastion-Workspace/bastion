"""
Telnet line editing: UTF-8 code points, ANSI keys, bracketed paste, history.
Used by BBSSession.read_line and read_password.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum, auto
from typing import Awaitable, Callable, List, Optional


class OpKind(Enum):
    INSERT = auto()
    INSERT_BULK = auto()
    BACKSPACE = auto()
    DELETE = auto()
    LEFT = auto()
    RIGHT = auto()
    HOME = auto()
    END = auto()
    KILL_EOL = auto()
    KILL_BOL = auto()
    KILL_WORD = auto()
    REDRAW = auto()
    CANCEL = auto()
    EOF_EMPTY = auto()
    HIST_PREV = auto()
    HIST_NEXT = auto()
    TOGGLE_INSERT = auto()


@dataclass(frozen=True)
class LineOp:
    kind: OpKind
    text: str = ""


def display_width(ch: str) -> int:
    if not ch:
        return 0
    if unicodedata.combining(ch):
        return 0
    o = ord(ch)
    if o < 32 and ch != "\t":
        return 0
    if ch == "\t":
        return 4
    ea = unicodedata.east_asian_width(ch)
    if ea in ("W", "F"):
        return 2
    return 1


def prefix_width(prefix: str) -> int:
    return sum(display_width(c) for c in prefix)


_ANSI_ESCAPE = re.compile(r"\x1b\[[\d?;]*[A-Za-z]")


def ansi_visible_width(prefix: str) -> int:
    return prefix_width(_ANSI_ESCAPE.sub("", prefix))


def cells_width(cells: List[str], start: int, end: int) -> int:
    return sum(display_width(cells[i]) for i in range(max(0, start), min(len(cells), end)))


class HistoryRing:
    """Per-tag session history: Up = older in time (previous submission)."""

    def __init__(self, max_items: int = 50) -> None:
        self._items: List[str] = []
        self._max = max_items
        self._nav: Optional[int] = None
        self._draft: str = ""

    def reset_nav(self) -> None:
        self._nav = None
        self._draft = ""

    def add(self, line: str) -> None:
        s = line.strip()
        if not s:
            return
        if self._items and self._items[-1] == s:
            return
        self._items.append(s)
        if len(self._items) > self._max:
            self._items.pop(0)

    def prev(self, current: str) -> Optional[str]:
        if not self._items:
            return None
        if self._nav is None:
            self._draft = current
            self._nav = len(self._items) - 1
            return self._items[self._nav]
        if self._nav > 0:
            self._nav -= 1
            return self._items[self._nav]
        return self._items[0]

    def next(self) -> Optional[str]:
        if self._nav is None:
            return None
        if self._nav < len(self._items) - 1:
            self._nav += 1
            return self._items[self._nav]
        self._nav = None
        return self._draft


class LineInputParser:
    """
    Incremental byte stream to LineOp. Caller strips IAC before feeding.
    CR/LF is handled by the caller when not in bracketed paste.
    """

    def __init__(self, accept_bracketed_paste: bool = True) -> None:
        self.accept_bracketed_paste = accept_bracketed_paste
        self._state = "normal"
        self._csi = bytearray()
        self._utf8_buf = bytearray()
        self._utf8_need = 0
        self._paste_buf = bytearray()

    @property
    def in_bracketed_paste(self) -> bool:
        return self._state == "paste"

    def reset(self) -> None:
        self._state = "normal"
        self._csi.clear()
        self._utf8_buf.clear()
        self._utf8_need = 0
        self._paste_buf.clear()

    def _utf8_follow_len(self, lead: int) -> int:
        if lead < 0x80:
            return 0
        if (lead & 0xE0) == 0xC0:
            return 1
        if (lead & 0xF0) == 0xE0:
            return 2
        if (lead & 0xF8) == 0xF0:
            return 3
        return -1

    def _flush_utf8(self) -> List[LineOp]:
        if not self._utf8_buf:
            return []
        try:
            s = self._utf8_buf.decode("utf-8")
        except UnicodeDecodeError:
            s = "\ufffd"
        self._utf8_buf.clear()
        self._utf8_need = 0
        if not s:
            return []
        return [LineOp(OpKind.INSERT, s)]

    def feed(self, b: int) -> List[LineOp]:
        if self._state == "paste":
            return self._feed_paste(b)
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

        if b in (0x7F, 0x08):
            return [LineOp(OpKind.BACKSPACE)]
        if b == 0x09:
            return [LineOp(OpKind.INSERT, "\t")]
        if b == 0x01:
            return [LineOp(OpKind.HOME)]
        if b == 0x05:
            return [LineOp(OpKind.END)]
        if b == 0x02:
            return [LineOp(OpKind.LEFT)]
        if b == 0x06:
            return [LineOp(OpKind.RIGHT)]
        if b == 0x0B:
            return [LineOp(OpKind.KILL_EOL)]
        if b == 0x15:
            return [LineOp(OpKind.KILL_BOL)]
        if b == 0x17:
            return [LineOp(OpKind.KILL_WORD)]
        if b == 0x0C:
            return [LineOp(OpKind.REDRAW)]
        if b == 0x03:
            return [LineOp(OpKind.CANCEL)]
        if b == 0x04:
            return [LineOp(OpKind.EOF_EMPTY)]

        need = self._utf8_follow_len(b)
        if need < 0:
            return []
        if need == 0:
            if b < 0x20:
                return []
            return [LineOp(OpKind.INSERT, chr(b))]
        self._utf8_buf = bytearray([b])
        self._utf8_need = need
        return []

    def _feed_after_esc(self, b: int) -> List[LineOp]:
        if b == 0x5B:
            self._state = "csi"
            self._csi.clear()
            return []
        if b == 0x4F:
            self._state = "ss3"
            return []
        if b == 0x1B:
            self._state = "normal"
            return []
        self._state = "normal"
        return []

    def _feed_csi(self, b: int) -> List[LineOp]:
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

    def _parse_csi(self, raw: bytes) -> List[LineOp]:
        if not raw or len(raw) < 1:
            return []
        final = raw[-1]
        body = raw[:-1]

        if final == ord("~"):
            inner = body.decode("ascii", errors="ignore")
            if inner.isdigit():
                n = int(inner)
                if n == 200 and self.accept_bracketed_paste:
                    self._state = "paste"
                    self._paste_buf.clear()
                    return []
                if n == 201:
                    return []
                if n == 1:
                    return [LineOp(OpKind.HOME)]
                if n in (2,):
                    return [LineOp(OpKind.TOGGLE_INSERT)]
                if n == 3:
                    return [LineOp(OpKind.DELETE)]
                if n == 4:
                    return [LineOp(OpKind.END)]
                if n in (5, 6):
                    return []
                if n in (7,):
                    return [LineOp(OpKind.HOME)]
                if n in (8,):
                    return [LineOp(OpKind.END)]
            return []

        if final == ord("A"):
            return [LineOp(OpKind.HIST_PREV)]
        if final == ord("B"):
            return [LineOp(OpKind.HIST_NEXT)]
        if final == ord("C"):
            return [LineOp(OpKind.RIGHT)]
        if final == ord("D"):
            return [LineOp(OpKind.LEFT)]
        if final == ord("H"):
            return [LineOp(OpKind.HOME)]
        if final == ord("F"):
            return [LineOp(OpKind.END)]
        return []

    def _map_ss3(self, b: int) -> List[LineOp]:
        if b == 0x41:
            return [LineOp(OpKind.HIST_PREV)]
        if b == 0x42:
            return [LineOp(OpKind.HIST_NEXT)]
        if b == 0x43:
            return [LineOp(OpKind.RIGHT)]
        if b == 0x44:
            return [LineOp(OpKind.LEFT)]
        if b == 0x48:
            return [LineOp(OpKind.HOME)]
        if b == 0x46:
            return [LineOp(OpKind.END)]
        return []

    def _feed_paste(self, b: int) -> List[LineOp]:
        self._paste_buf.append(b)
        end_marker = b"\x1b[201~"
        if len(self._paste_buf) >= len(end_marker) and bytes(self._paste_buf[-len(end_marker) :]) == end_marker:
            inner = bytes(self._paste_buf[: -len(end_marker)])
            self._paste_buf.clear()
            self._state = "normal"
            try:
                text = inner.decode("utf-8", errors="replace")
            except Exception:
                text = ""
            text = text.replace("\x00", "")
            return [LineOp(OpKind.INSERT_BULK, text)]
        if len(self._paste_buf) > 1_000_000:
            self._paste_buf.clear()
            self._state = "normal"
        return []


WriteFn = Callable[[bytes], Awaitable[None]]


class LineEditor:
    """Buffer of Unicode scalar values (str length 1 each), cursor, redraw."""

    def __init__(
        self,
        term_width: int,
        line_prefix: str,
        mask_char: Optional[str],
        allow_multiline_paste: bool,
    ) -> None:
        self.term_width = max(40, term_width)
        self.line_prefix = line_prefix
        self.mask_char = mask_char
        self.allow_multiline_paste = allow_multiline_paste
        self.cells: List[str] = []
        self.cursor = 0
        self.insert_mode = True

    def get_text(self) -> str:
        return "".join(self.cells)

    def _visible_char(self, ch: str) -> str:
        if self.mask_char:
            return self.mask_char
        return ch

    def _line_bytes(self) -> bytes:
        parts: List[str] = [self.line_prefix]
        for c in self.cells:
            parts.append(self._visible_char(c))
        return "".join(parts).encode("utf-8", errors="replace")

    def _cursor_column_1based(self) -> int:
        return 1 + ansi_visible_width(self.line_prefix) + cells_width(self.cells, 0, self.cursor)

    def _end_column_1based(self) -> int:
        return 1 + ansi_visible_width(self.line_prefix) + cells_width(self.cells, 0, len(self.cells))

    async def redraw_full(self, write: WriteFn) -> None:
        line_b = self._line_bytes()
        end_col = self._end_column_1based()
        cur_col = self._cursor_column_1based()
        out = b"\r" + line_b + b"\x1b[K"
        await write(out)
        steps_back = end_col - cur_col
        if steps_back > 0:
            await write(f"\x1b[{steps_back}D".encode("ascii"))

    async def apply(self, op: LineOp, write: WriteFn, history: Optional[HistoryRing]) -> Optional[str]:
        """
        Apply one op. Returns 'cancel' | 'eof_empty' | None.
        """
        if op.kind == OpKind.CANCEL:
            self.cells.clear()
            self.cursor = 0
            return "cancel"
        if op.kind == OpKind.EOF_EMPTY:
            if not self.cells:
                return "eof_empty"
            if self.cursor < len(self.cells):
                del self.cells[self.cursor]
                await self.redraw_full(write)
            return None

        if op.kind == OpKind.HIST_PREV:
            if history is None:
                return None
            cur = self.get_text()
            nxt = history.prev(cur)
            if nxt is None:
                return None
            self.cells = list(nxt)
            self.cursor = len(self.cells)
            await self.redraw_full(write)
            return None

        if op.kind == OpKind.HIST_NEXT:
            if history is None:
                return None
            nxt = history.next()
            if nxt is None:
                return None
            self.cells = list(nxt)
            self.cursor = len(self.cells)
            await self.redraw_full(write)
            return None

        if op.kind == OpKind.TOGGLE_INSERT:
            self.insert_mode = not self.insert_mode
            return None

        if op.kind == OpKind.REDRAW:
            await self.redraw_full(write)
            return None

        if op.kind == OpKind.HOME:
            self.cursor = 0
            await self.redraw_full(write)
            return None

        if op.kind == OpKind.END:
            self.cursor = len(self.cells)
            await self.redraw_full(write)
            return None

        if op.kind == OpKind.LEFT:
            if self.cursor > 0:
                if self.cursor == len(self.cells):
                    self.cursor -= 1
                    w = display_width(self.cells[self.cursor])
                    if w > 0:
                        await write(f"\x1b[{w}D".encode("ascii"))
                else:
                    self.cursor -= 1
                    await self.redraw_full(write)
            return None

        if op.kind == OpKind.RIGHT:
            if self.cursor < len(self.cells):
                w = display_width(self.cells[self.cursor])
                self.cursor += 1
                if w > 0:
                    await write(f"\x1b[{w}C".encode("ascii"))
            return None

        if op.kind == OpKind.BACKSPACE:
            if self.cursor > 0:
                self.cursor -= 1
                del self.cells[self.cursor]
                await self.redraw_full(write)
            return None

        if op.kind == OpKind.DELETE:
            if self.cursor < len(self.cells):
                del self.cells[self.cursor]
                await self.redraw_full(write)
            return None

        if op.kind == OpKind.KILL_EOL:
            if self.cursor < len(self.cells):
                del self.cells[self.cursor :]
                await self.redraw_full(write)
            return None

        if op.kind == OpKind.KILL_BOL:
            if self.cursor > 0:
                del self.cells[: self.cursor]
                self.cursor = 0
                await self.redraw_full(write)
            return None

        if op.kind == OpKind.KILL_WORD:
            if self.cursor == 0:
                return None
            i = self.cursor - 1
            while i >= 0 and self.cells[i].isspace():
                i -= 1
            while i >= 0 and not self.cells[i].isspace():
                i -= 1
            start = i + 1
            if start < self.cursor:
                del self.cells[start : self.cursor]
                self.cursor = start
                await self.redraw_full(write)
            return None

        if op.kind == OpKind.INSERT:
            return await self._insert_text(op.text, write)

        if op.kind == OpKind.INSERT_BULK:
            t = op.text
            if not self.allow_multiline_paste:
                t = t.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
            return await self._insert_text(t, write)

        return None

    async def _insert_text(self, text: str, write: WriteFn) -> None:
        if not text:
            return
        start_len = len(self.cells)
        start_cur = self.cursor
        for ch in text:
            if ch == "\n" and not self.allow_multiline_paste:
                ch = " "
            if self.cursor < len(self.cells) and not self.insert_mode:
                self.cells[self.cursor] = ch
                self.cursor += 1
            else:
                self.cells.insert(self.cursor, ch)
                self.cursor += 1
        appended_only = start_cur == start_len and self.cursor == len(self.cells)
        if appended_only and text:
            piece = self.cells[start_len:]
            vis = "".join(self._visible_char(c) for c in piece)
            await write(vis.encode("utf-8", errors="replace"))
        else:
            await self.redraw_full(write)
