"""
Per-connection BBS session: I/O, login, idle handling, main loop.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, TYPE_CHECKING

from backend_client import BackendClient
from line_input import HistoryRing, LineEditor, LineInputParser
from config.settings import settings
from rendering.ansi import theme_from_name
from rendering.ansi import Theme
from screen_blank_guard import gate_before_next_byte
from telnet_input import discard_telnet_command_from_reader
from telnet_input import strip_telnet_from_buffer

if TYPE_CHECKING:
    from editor.keys import EditorKeyParser, KeyEvent

logger = logging.getLogger(__name__)

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def _trim_welcome_ansi_margins(lines: List[str]) -> List[str]:
    """Strip shared left padding and trailing spaces (e.g. 80-column canvas)."""
    stripped = [ln.rstrip(" \t") for ln in lines]
    nonempty = [ln for ln in stripped if ln.strip()]
    if not nonempty:
        return stripped
    min_lead = min(len(ln) - len(ln.lstrip(" ")) for ln in nonempty)
    out: List[str] = []
    for ln in stripped:
        if not ln.strip():
            out.append("")
        else:
            cut = ln[min_lead:] if len(ln) >= min_lead else ln.lstrip(" ")
            out.append(cut.rstrip(" \t"))
    return out


def _center_last_nonempty_welcome_line(lines: List[str]) -> None:
    """Center the last non-empty line to the width of the widest non-empty line."""
    idxs = [i for i, ln in enumerate(lines) if ln.strip()]
    if len(idxs) < 2:
        return
    maxw = max(len(lines[i]) for i in idxs)
    last_i = idxs[-1]
    s = lines[last_i].strip()
    pad = max(0, (maxw - len(s)) // 2)
    lines[last_i] = " " * pad + s


class BBSSession:
    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        session_id: str,
        connected_count: Optional[callable] = None,
        telnet_mode: bool = True,
    ) -> None:
        self.reader = reader
        self.writer = writer
        self.session_id = session_id
        self.telnet_mode = telnet_mode
        self.client = BackendClient()
        self._connected_count = connected_count
        self.user_id: str = ""
        self.jwt_token: str = ""
        self.username: str = ""
        self.role: str = "user"
        self.display_name: str = ""
        self.platform_chat_id = f"session-{session_id}"
        self.conversation_id = ""
        self.term_width = 80
        self.term_height = 24
        self.theme: Theme = theme_from_name(settings.BBS_THEME)
        self.chat_listing_cache: List[dict] = []
        self.messaging_rooms_cache: List[dict] = []
        # RSS article list: False = All, True = Unread only (persists until disconnect).
        self.rss_list_unread_only: bool = False
        # In-session ebook reading: KoSync digest (partial MD5) -> last chapter index.
        self.ebook_positions: Dict[str, int] = {}
        # Main-menu messaging unread: terminal bell when total unread increases (user-to-user rooms).
        self.messaging_bell_enabled: bool = True
        self._messaging_unread_baseline: Optional[int] = None
        self._last_activity = time.monotonic()
        self._idle_warned = False
        # Bytes pushed back after optional CRLF handling (StreamReader has no unget).
        self._input_unread: deque[int] = deque()
        self._line_history: Dict[str, HistoryRing] = {}
        self._screen_blanked: bool = False
        self._screen_blank_exempt: bool = False
        self._blank_lock = asyncio.Lock()
        self._blank_watcher_task: Optional[asyncio.Task] = None
        self.screen_blank_after_seconds: int = int(settings.BBS_SCREEN_BLANK_AFTER_SECONDS or 0)

    def set_screen_blank_exempt(self, exempt: bool) -> None:
        """When True, idle blanking is disabled (e.g. wallpaper / screensaver pane)."""
        self._screen_blank_exempt = bool(exempt)

    def _apply_naws(self, width: int, height: int) -> None:
        self.term_width = width
        self.term_height = height

    def _touch(self) -> None:
        self._last_activity = time.monotonic()
        self._idle_warned = False

    async def _write(self, text: str) -> None:
        data = text.replace("\n", "\r\n").encode("utf-8", errors="replace")
        self.writer.write(data)
        await self.writer.drain()

    async def _write_bytes(self, data: bytes) -> None:
        self.writer.write(data)
        await self.writer.drain()

    async def clear_screen(self) -> None:
        if self.theme.clear_screen:
            await self._write_bytes(self.theme.clear_screen.encode("utf-8"))

    def _default_conversation_id(self) -> str:
        return f"bbs:{self.user_id}:{self.platform_chat_id}"

    async def _next_input_byte(self, deadline: float) -> int:
        await gate_before_next_byte(self)
        if self._input_unread:
            return self._input_unread.popleft()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise asyncio.TimeoutError
        data = await asyncio.wait_for(self.reader.read(1), timeout=remaining)
        if not data:
            raise ConnectionError("connection closed")
        return data[0]

    async def _consume_optional_lf_after_cr(self, deadline: float) -> None:
        """If the client sends CRLF, swallow the LF after CR (DOS clients often send CR only)."""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        try:
            data = await asyncio.wait_for(
                self.reader.read(1), timeout=min(0.05, remaining)
            )
        except asyncio.TimeoutError:
            return
        if not data:
            raise ConnectionError("connection closed")
        if data[0] != 0x0A:
            self._input_unread.appendleft(data[0])

    async def drain_stray_line_terminators(self, max_bytes: int = 32) -> None:
        """
        Drop CR/LF bytes left in the input queue after read_line() (e.g. delayed LF from CRLF).
        Prevents paginate_text/read_pager_key from treating them as an immediate 'next' page.
        """
        deadline = time.monotonic() + 0.08
        n = 0
        while n < max_bytes and time.monotonic() < deadline:
            if self._input_unread:
                b = self._input_unread.popleft()
                if b in (0x0D, 0x0A):
                    n += 1
                    continue
                self._input_unread.appendleft(b)
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            try:
                data = await asyncio.wait_for(
                    self.reader.read(1), timeout=min(0.05, remaining)
                )
            except asyncio.TimeoutError:
                return
            if not data:
                return
            b = data[0]
            if b in (0x0D, 0x0A):
                n += 1
                continue
            self._input_unread.appendleft(b)
            return

    async def _fetch_messaging_unread_total(self) -> Optional[int]:
        """Sum of unread_count across rooms; None if messaging unavailable."""
        try:
            res = await self.client.messaging_list_rooms(self.jwt_token, limit=100)
        except Exception:
            logger.debug("messaging_list_rooms for unread failed", exc_info=True)
            return None
        if res.get("error"):
            return None
        rooms: List[dict] = list(res.get("rooms") or [])
        n = 0
        for r in rooms:
            try:
                n += int(r.get("unread_count") or 0)
            except (TypeError, ValueError):
                pass
        return n

    async def sync_messaging_unread_baseline(self) -> None:
        """Call when drawing the main menu so we do not beep for already-unread messages."""
        self._messaging_unread_baseline = await self._fetch_messaging_unread_total()

    async def poll_messaging_notify_while_waiting(self) -> None:
        """If messaging unread increased since baseline, emit ASCII BEL (terminal bell)."""
        total = await self._fetch_messaging_unread_total()
        if total is None:
            return
        baseline = self._messaging_unread_baseline
        if baseline is not None and total > baseline and self.messaging_bell_enabled:
            await self._write_bytes(b"\x07")
        self._messaging_unread_baseline = total

    async def read_menu_choice(
        self,
        *,
        allow_digit_suffix: bool = False,
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
        on_poll: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> str:
        """
        Menu prompts with [X] hotkeys: read the first key without requiring Enter.
        Letters and symbols echo and return immediately (optional CRLF after the key is drained).
        When allow_digit_suffix is True and the first key is a digit, additional digits are read
        until Enter so multi-digit list indices still work.
        When poll_interval and on_poll are set, on_poll runs after each short wait with no input
        (without shortening the overall idle deadline).
        """
        to = timeout
        if to is None:
            idle = settings.BBS_IDLE_TIMEOUT
            elapsed = time.monotonic() - self._last_activity
            remaining = idle - elapsed
            if remaining <= 0:
                await self._write("\r\nDisconnected (idle timeout).\r\n")
                raise ConnectionError("idle timeout")
            warn_at = idle - settings.BBS_IDLE_WARN_SECONDS
            if elapsed > warn_at and not self._idle_warned and settings.BBS_IDLE_WARN_SECONDS > 0:
                await self._write(
                    self.theme.dim
                    + f"\r\nIdle timeout in {settings.BBS_IDLE_WARN_SECONDS}s; press Enter to stay connected.\r\n"
                    + self.theme.reset
                )
                self._idle_warned = True
            to = max(1.0, remaining)

        deadline = time.monotonic() + to
        use_poll = (
            poll_interval is not None
            and poll_interval > 0
            and on_poll is not None
        )
        while True:
            next_deadline = deadline
            if use_poll:
                slice_end = time.monotonic() + float(poll_interval)
                if slice_end < deadline:
                    next_deadline = slice_end
            try:
                b = await self._next_input_byte(next_deadline)
            except asyncio.TimeoutError:
                if time.monotonic() >= deadline:
                    await self._write("\r\nDisconnected (idle timeout).\r\n")
                    raise ConnectionError("idle timeout") from None
                if use_poll:
                    try:
                        await on_poll()
                    except Exception:
                        logger.debug("read_menu_choice on_poll failed", exc_info=True)
                    continue
                await self._write("\r\nDisconnected (idle timeout).\r\n")
                raise ConnectionError("idle timeout") from None

            if self.telnet_mode and b == 0xFF:
                await discard_telnet_command_from_reader(self.reader)
                continue
            if b in (0x0D, 0x0A):
                if b == 0x0D:
                    await self._consume_optional_lf_after_cr(deadline)
                self._touch()
                return ""
            if 32 <= b <= 126:
                ch = chr(b)
                await self._write_bytes(bytes([b]))
                self._touch()
                if allow_digit_suffix and ch.isdigit():
                    rest = await self._read_digit_suffix_until_enter(deadline)
                    return ch + rest
                await self._drain_crlf_after_pager_key(deadline)
                if ch.isalpha():
                    return ch.lower()
                return ch
            continue

    async def _read_digit_suffix_until_enter(self, deadline: float) -> str:
        """Continue reading digits until CR/LF; does not echo the terminator."""
        buf = ""
        while True:
            try:
                b = await self._next_input_byte(deadline)
            except asyncio.TimeoutError:
                await self._write("\r\nDisconnected (idle timeout).\r\n")
                raise ConnectionError("idle timeout") from None

            if self.telnet_mode and b == 0xFF:
                await discard_telnet_command_from_reader(self.reader)
                continue
            if b in (0x0D, 0x0A):
                if b == 0x0D:
                    await self._consume_optional_lf_after_cr(deadline)
                self._touch()
                return buf
            if ord("0") <= b <= ord("9"):
                buf += chr(b)
                await self._write_bytes(bytes([b]))
                self._touch()
                continue
            if b in (0x7F, 0x08) and buf:
                buf = buf[:-1]
                await self._write_bytes(b"\b \b")
                self._touch()
                continue
            continue

    async def read_line(
        self,
        timeout: Optional[float] = None,
        history_tag: Optional[str] = None,
        allow_multiline_paste: bool = False,
        line_prefix: str = "",
    ) -> str:
        to = timeout
        if to is None:
            idle = settings.BBS_IDLE_TIMEOUT
            elapsed = time.monotonic() - self._last_activity
            remaining = idle - elapsed
            if remaining <= 0:
                await self._write("\r\nDisconnected (idle timeout).\r\n")
                raise ConnectionError("idle timeout")
            warn_at = idle - settings.BBS_IDLE_WARN_SECONDS
            if elapsed > warn_at and not self._idle_warned and settings.BBS_IDLE_WARN_SECONDS > 0:
                await self._write(
                    self.theme.dim
                    + f"\r\nIdle timeout in {settings.BBS_IDLE_WARN_SECONDS}s; press Enter to stay connected.\r\n"
                    + self.theme.reset
                )
                self._idle_warned = True
            to = max(1.0, remaining)

        deadline = time.monotonic() + to
        history: Optional[HistoryRing] = None
        if history_tag:
            history = self._line_history.setdefault(history_tag, HistoryRing())
            history.reset_nav()

        parser = LineInputParser(accept_bracketed_paste=True)
        editor = LineEditor(
            term_width=self.term_width,
            line_prefix=line_prefix,
            mask_char=None,
            allow_multiline_paste=allow_multiline_paste,
        )

        await self._write_bytes(b"\x1b[?2004h")
        try:
            while True:
                try:
                    b = await self._next_input_byte(deadline)
                except asyncio.TimeoutError:
                    await self._write("\r\nDisconnected (idle timeout).\r\n")
                    raise ConnectionError("idle timeout") from None

                if self.telnet_mode and b == 0xFF:
                    await discard_telnet_command_from_reader(self.reader)
                    continue

                if not parser.in_bracketed_paste and b in (0x0D, 0x0A):
                    if b == 0x0D:
                        await self._consume_optional_lf_after_cr(deadline)
                    text_raw = editor.get_text()
                    if history is not None and text_raw.strip():
                        history.add(text_raw)
                    self._touch()
                    clean = strip_telnet_from_buffer(
                        text_raw.encode("utf-8", errors="replace"), on_naws=self._apply_naws
                    )
                    text = clean.decode("utf-8", errors="replace").replace("\x00", "")
                    return text.strip()

                for op in parser.feed(b):
                    sig = await editor.apply(op, self._write_bytes, history)
                    if sig == "cancel":
                        await self._write("^C\r\n")
                        self._touch()
                        return ""
                    if sig == "eof_empty":
                        self._touch()
                        return ""
        finally:
            await self._write_bytes(b"\x1b[?2004l")

    async def _drain_crlf_after_pager_key(self, deadline: float) -> None:
        """If the user typed Space then Enter, consume the trailing CR/LF without a second advance."""
        while True:
            rem = deadline - time.monotonic()
            if rem <= 0:
                return
            try:
                data = await asyncio.wait_for(self.reader.read(1), timeout=min(0.08, rem))
            except asyncio.TimeoutError:
                return
            if not data:
                return
            b = data[0]
            if b == 0x0D:
                await self._consume_optional_lf_after_cr(deadline)
                return
            if b == 0x0A:
                return
            self._input_unread.appendleft(b)
            return

    async def read_pager_key(self, timeout: Optional[float] = None) -> str:
        """
        Single-key input for --More-- pagination: Space, Enter, or n advances; b/p go back; q quits.
        Unlike read_line(), Space does not require Enter (required for mTCP/DOS telnet).
        Returns 'next', 'prev', or 'quit'.
        """
        to = timeout
        if to is None:
            idle = settings.BBS_IDLE_TIMEOUT
            elapsed = time.monotonic() - self._last_activity
            remaining = idle - elapsed
            if remaining <= 0:
                await self._write("\r\nDisconnected (idle timeout).\r\n")
                raise ConnectionError("idle timeout")
            warn_at = idle - settings.BBS_IDLE_WARN_SECONDS
            if elapsed > warn_at and not self._idle_warned and settings.BBS_IDLE_WARN_SECONDS > 0:
                await self._write(
                    self.theme.dim
                    + f"\r\nIdle timeout in {settings.BBS_IDLE_WARN_SECONDS}s; press Enter to stay connected.\r\n"
                    + self.theme.reset
                )
                self._idle_warned = True
            to = max(1.0, remaining)

        deadline = time.monotonic() + to
        while True:
            try:
                b = await self._next_input_byte(deadline)
            except asyncio.TimeoutError:
                await self._write("\r\nDisconnected (idle timeout).\r\n")
                raise ConnectionError("idle timeout") from None

            if self.telnet_mode and b == 0xFF:
                await discard_telnet_command_from_reader(self.reader)
                continue
            if b in (0x0D, 0x0A):
                if b == 0x0D:
                    await self._consume_optional_lf_after_cr(deadline)
                self._touch()
                return "next"
            if b == 0x20:
                self._touch()
                await self._write_bytes(b" ")
                await self._drain_crlf_after_pager_key(deadline)
                return "next"
            if b in (ord("n"), ord("N")):
                self._touch()
                await self._write_bytes(bytes([b]))
                return "next"
            if b in (ord("b"), ord("B"), ord("p"), ord("P")):
                self._touch()
                await self._write_bytes(bytes([b]))
                return "prev"
            if b in (ord("q"), ord("Q")):
                self._touch()
                await self._write_bytes(bytes([b]))
                return "quit"
            if b == 0x03:
                self._touch()
                return "quit"

    async def read_reader_key(self, timeout: Optional[float] = None) -> str:
        """
        Single-key input for full-screen file reader (less-like): navigation and search trigger.
        Returns: next, prev, top, bottom, search, next_match, prev_match, quit.
        """
        to = timeout
        if to is None:
            idle = settings.BBS_IDLE_TIMEOUT
            elapsed = time.monotonic() - self._last_activity
            remaining = idle - elapsed
            if remaining <= 0:
                await self._write("\r\nDisconnected (idle timeout).\r\n")
                raise ConnectionError("idle timeout")
            warn_at = idle - settings.BBS_IDLE_WARN_SECONDS
            if elapsed > warn_at and not self._idle_warned and settings.BBS_IDLE_WARN_SECONDS > 0:
                await self._write(
                    self.theme.dim
                    + f"\r\nIdle timeout in {settings.BBS_IDLE_WARN_SECONDS}s; press Enter to stay connected.\r\n"
                    + self.theme.reset
                )
                self._idle_warned = True
            to = max(1.0, remaining)

        deadline = time.monotonic() + to
        while True:
            try:
                b = await self._next_input_byte(deadline)
            except asyncio.TimeoutError:
                await self._write("\r\nDisconnected (idle timeout).\r\n")
                raise ConnectionError("idle timeout") from None

            if self.telnet_mode and b == 0xFF:
                await discard_telnet_command_from_reader(self.reader)
                continue
            if b in (0x0D, 0x0A):
                if b == 0x0D:
                    await self._consume_optional_lf_after_cr(deadline)
                self._touch()
                return "next"
            if b == 0x20:
                self._touch()
                await self._write_bytes(b" ")
                await self._drain_crlf_after_pager_key(deadline)
                return "next"
            if b in (ord("b"), ord("B"), ord("-")):
                self._touch()
                await self._write_bytes(bytes([b]))
                return "prev"
            if b == ord("g"):
                self._touch()
                await self._write_bytes(b"g")
                return "top"
            if b == ord("G"):
                self._touch()
                await self._write_bytes(b"G")
                return "bottom"
            if b == ord("/"):
                self._touch()
                return "search"
            if b == ord("n"):
                self._touch()
                await self._write_bytes(b"n")
                return "next_match"
            if b == ord("N"):
                self._touch()
                await self._write_bytes(b"N")
                return "prev_match"
            if b in (ord("q"), ord("Q")):
                self._touch()
                await self._write_bytes(bytes([b]))
                return "quit"
            if b == 0x03:
                self._touch()
                return "quit"
            continue

    async def read_wallpaper_poll_key(self, timeout: float = 5.0) -> str:
        """
        Wait for a key or until timeout. Does not disconnect on timeout (unlike read_pager_key).
        Returns: 'quit' (q/Q/Esc/Ctrl-C), 'refresh' (r/R), or 'idle' when the wait expires.
        Bare Esc quits; Esc followed by CSI/SS3 (arrow keys, etc.) is consumed and ignored.
        Other keys are ignored and the wait continues until timeout or a recognized key.
        """
        deadline = time.monotonic() + max(0.1, timeout)
        while True:
            try:
                b = await self._next_input_byte(deadline)
            except asyncio.TimeoutError:
                self._touch()
                return "idle"

            if self.telnet_mode and b == 0xFF:
                await discard_telnet_command_from_reader(self.reader)
                continue
            if b in (0x0D, 0x0A):
                if b == 0x0D:
                    await self._consume_optional_lf_after_cr(deadline)
                continue
            if b in (ord("q"), ord("Q")):
                self._touch()
                await self._write_bytes(bytes([b]))
                return "quit"
            if b == 0x03:
                self._touch()
                return "quit"
            if b == 0x1B:
                self._touch()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return "quit"
                esc_follow_deadline = time.monotonic() + min(0.06, remaining)
                try:
                    b2 = await self._next_input_byte(esc_follow_deadline)
                except asyncio.TimeoutError:
                    return "quit"
                if b2 == ord("["):
                    while True:
                        bn = await self._next_input_byte(deadline)
                        if 0x40 <= bn <= 0x7E:
                            break
                    continue
                if b2 == ord("O"):
                    await self._next_input_byte(deadline)
                    continue
                self._input_unread.appendleft(b2)
                return "quit"
            if b in (ord("r"), ord("R")):
                self._touch()
                await self._write_bytes(bytes([b]))
                return "refresh"
            continue

    async def read_password(self, line_prefix: str = "") -> str:
        parser = LineInputParser(accept_bracketed_paste=False)
        editor = LineEditor(
            term_width=self.term_width,
            line_prefix=line_prefix,
            mask_char="*",
            allow_multiline_paste=False,
        )
        while True:
            deadline = time.monotonic() + 120.0
            try:
                b = await self._next_input_byte(deadline)
            except asyncio.TimeoutError:
                break
            if self.telnet_mode and b == 0xFF:
                await discard_telnet_command_from_reader(self.reader)
                continue
            if not parser.in_bracketed_paste and b in (0x0D, 0x0A):
                if b == 0x0D:
                    await self._consume_optional_lf_after_cr(deadline)
                break
            for op in parser.feed(b):
                sig = await editor.apply(op, self._write_bytes, None)
                if sig == "cancel":
                    await self._write("^C\r\n")
                    self._touch()
                    return ""
                if sig == "eof_empty":
                    self._touch()
                    return ""
        await self._write("\r\n")
        self._touch()
        return editor.get_text().replace("\x00", "")


    async def read_editor_key(self, parser: "EditorKeyParser") -> "KeyEvent":
        """Read one key for full-screen editor (character-at-a-time, CSI/UTF-8)."""
        while True:
            idle = settings.BBS_IDLE_TIMEOUT
            elapsed = time.monotonic() - self._last_activity
            remaining = idle - elapsed
            if remaining <= 0:
                await self._write("\r\nDisconnected (idle timeout).\r\n")
                raise ConnectionError("idle timeout")
            to = max(0.5, remaining)
            try:
                b = await asyncio.wait_for(self.reader.read(1), timeout=to)
            except asyncio.TimeoutError:
                await self._write("\r\nDisconnected (idle timeout).\r\n")
                raise ConnectionError("idle timeout") from None
            if not b:
                raise ConnectionError("connection closed")
            if self.telnet_mode and b[0] == 0xFF:
                await discard_telnet_command_from_reader(self.reader)
                continue
            events = parser.feed(b[0])
            for ev in events:
                self._touch()
                return ev
            self._touch()

    async def _show_welcome(self) -> None:
        welcome_path = _ASSETS_DIR / "welcome.ans"
        if welcome_path.is_file():
            try:
                raw = welcome_path.read_bytes()
                # Normalize newlines so we never emit accidental blank rows (e.g. \n\r splits).
                raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
                text = raw.decode("utf-8", errors="replace")
                lines = text.split("\n")
                while lines and lines[-1] == "":
                    lines.pop()
                while lines and lines[0] == "":
                    lines.pop(0)
                if lines:
                    lines = _trim_welcome_ansi_margins(lines)
                    while lines and lines[0] == "":
                        lines.pop(0)
                    _center_last_nonempty_welcome_line(lines)
                    # Telnet expects CRLF; LF-only makes many clients draw “stairs” (no CR).
                    out = "\r\n".join(lines) + "\r\n"
                    await self._write_bytes(out.encode("utf-8"))
                return
            except Exception as e:
                logger.debug("welcome.ans read failed: %s", e)
        await self._write(
            self.theme.title_color()
            + settings.BBS_NAME
            + self.theme.reset
            + "\r\n"
            + self.theme.dim
            + "Bastion text interface.\r\n"
            + self.theme.reset
            + "\r\n"
        )

    async def show_welcome(self) -> None:
        """Show welcome art (telnet before login; SSH after protocol-level auth)."""
        await self._show_welcome()

    async def login_flow(self) -> bool:
        for attempt in range(3):
            await self._write("Username: ")
            user = await self.read_line(timeout=300.0, line_prefix="Username: ")
            user = user.replace("\x00", "").strip()
            if not user:
                await self._write("Username required.\r\n")
                continue
            # Ensure a fresh line: many clients do not advance until the server sends CRLF.
            # Do not send IAC WILL/WONT ECHO here: some DOS clients print raw bytes (e.g. 0xFB
            # as a checkmark/sqrt on CP437). Password masking is server-side * in read_password().
            await self._write("\r\nPassword: ")
            password = await self.read_password(line_prefix="Password: ")
            password = password.replace("\x00", "")
            result = await self.client.login(user, password)
            if result.get("error"):
                await self._write(f"Login failed: {result['error'][:200]}\r\n")
                continue
            try:
                self.apply_login_result(result, username_fallback=user)
            except ValueError:
                await self._write("Invalid login response.\r\n")
                continue
            return True
        await self._write("Too many failures. Goodbye.\r\n")
        return False

    def apply_login_result(self, result: Dict[str, Any], username_fallback: str = "") -> None:
        """Set session identity from POST /api/auth/login JSON (same shape as login_flow)."""
        token = result.get("access_token") or result.get("accessToken")
        u = result.get("user") or {}
        if not token or not u.get("user_id"):
            raise ValueError("login result missing access_token or user_id")
        self.jwt_token = token
        self.user_id = str(u["user_id"])
        self.username = str(u.get("username") or username_fallback or "")
        self.role = str(u.get("role", "user"))
        self.display_name = str(u.get("display_name") or self.username)
        self.conversation_id = self._default_conversation_id()

    async def run_after_authenticated(self) -> None:
        """Post-auth menu (used after telnet login_flow or SSH password auth)."""
        # SSH skips login_flow (no _touch there); reset idle baseline before slow summary HTTP.
        self._touch()
        if self.screen_blank_after_seconds > 0 and self._blank_watcher_task is None:
            from screen_blank_guard import idle_blank_watcher_loop

            self._blank_watcher_task = asyncio.create_task(idle_blank_watcher_loop(self))
        try:
            if settings.BBS_MOTD.strip():
                await self._write("\r\n" + settings.BBS_MOTD.strip() + "\r\n\r\n")
            await self._show_post_login_summary()
            from menu_system import main_menu

            await main_menu(self)
        finally:
            t = self._blank_watcher_task
            if t is not None:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
                self._blank_watcher_task = None

    async def _show_post_login_summary(self) -> None:
        t = self.theme
        online = self._connected_count() if self._connected_count else "?"
        parts = [f"Welcome back, {self.display_name}.  {online} user(s) online."]
        try:
            convos = await self.client.list_user_conversations(self.user_id, limit=1)
            conv_list = convos.get("conversations") or []
            conv_count = convos.get("total") or len(conv_list)
        except Exception:
            conv_count = "?"
        try:
            unread_map = await self.client.rss_unread_by_feed(self.jwt_token)
            rss_unread = sum(unread_map.values()) if unread_map else 0
        except Exception:
            rss_unread = "?"
        parts.append(f"Conversations: {conv_count}  |  Unread RSS: {rss_unread}")
        await self._write(t.dim + "\r\n".join(parts) + t.reset + "\r\n\r\n")

    async def run(self) -> None:
        self._blank_watcher_task = None
        try:
            await self.show_welcome()
            if not await self.login_flow():
                return
            await self.run_after_authenticated()
        except ConnectionError:
            pass
        except Exception as e:
            logger.exception("Session error: %s", e)
            try:
                await self._write(f"\r\nError: {str(e)[:200]}\r\n")
            except Exception:
                pass
        finally:
            if self._blank_watcher_task:
                self._blank_watcher_task.cancel()
                try:
                    await self._blank_watcher_task
                except (asyncio.CancelledError, Exception):
                    pass
