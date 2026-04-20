"""
OLED-friendly idle screen blank: full black after inactivity; only Esc resumes (except wallpaper pane).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from telnet_input import discard_telnet_command_from_reader

if TYPE_CHECKING:
    from session import BBSSession

logger = logging.getLogger(__name__)


async def _read_blank_byte(session: "BBSSession") -> int:
    if session._input_unread:
        return session._input_unread.popleft()
    data = await session.reader.read(1)
    if not data:
        raise ConnectionError("connection closed")
    return data[0]


async def _consume_csi_after_open_bracket(session: "BBSSession", deadline: float) -> None:
    while time.monotonic() < deadline:
        try:
            data = await asyncio.wait_for(session.reader.read(1), timeout=0.08)
        except asyncio.TimeoutError:
            return
        if not data:
            return
        c = data[0]
        if 0x40 <= c <= 0x7E:
            return


async def _wake_on_esc(session: "BBSSession") -> None:
    """After reading ESC (0x1B), optionally consume a CSI sequence, then clear blank state."""
    try:
        data = await asyncio.wait_for(session.reader.read(1), timeout=0.06)
    except asyncio.TimeoutError:
        data = b""
    if not data:
        await leave_blank_screen(session)
        session._touch()
        return
    b0 = data[0]
    if b0 == ord("["):
        await _consume_csi_after_open_bracket(session, time.monotonic() + 0.25)
        await leave_blank_screen(session)
        session._touch()
        return
    session._input_unread.appendleft(b0)
    await leave_blank_screen(session)
    session._touch()


async def gate_before_next_byte(session: "BBSSession") -> None:
    """While blanked, block reads until Esc (other keys are discarded)."""
    while session._screen_blanked:
        b = await _read_blank_byte(session)
        if getattr(session, "telnet_mode", False) and b == 0xFF:
            await discard_telnet_command_from_reader(session.reader)
            continue
        if b == 0x1B:
            await _wake_on_esc(session)
            return
        # Burn-in guard: swallow all non-Esc input until Esc.


async def paint_blank_screen(session: "BBSSession") -> None:
    """Fill the terminal with black background (ANSI); no visible hint text."""
    tw = max(10, session.term_width)
    th = max(5, session.term_height)
    if session.theme.clear_screen:
        await session._write_bytes(b"\x1b[?25l\x1b[40m\x1b[2J\x1b[H")
        row = (" " * tw + "\r\n").encode("utf-8", errors="replace")
        for _ in range(th):
            await session._write_bytes(row)
        await session._write_bytes(b"\x1b[H")
    else:
        await session.clear_screen()
        for _ in range(th):
            await session._write("\r\n")


async def enter_blank_screen(session: "BBSSession") -> None:
    async with session._blank_lock:
        if session._screen_blank_exempt or session._screen_blanked:
            return
        session._screen_blanked = True
    try:
        await paint_blank_screen(session)
    except Exception as e:
        logger.warning("screen blank paint failed: %s", e)
        async with session._blank_lock:
            session._screen_blanked = False


async def leave_blank_screen(session: "BBSSession") -> None:
    async with session._blank_lock:
        if not session._screen_blanked:
            return
        session._screen_blanked = False
    try:
        if session.theme.clear_screen:
            await session._write_bytes(b"\x1b[0m\x1b[?25h")
        await session.clear_screen()
    except Exception as e:
        logger.warning("screen blank restore failed: %s", e)


async def idle_blank_watcher_loop(session: "BBSSession") -> None:
    sec = int(getattr(session, "screen_blank_after_seconds", 0) or 0)
    if sec <= 0:
        return
    interval = min(30.0, max(5.0, float(sec) / 3.0))
    try:
        while True:
            await asyncio.sleep(interval)
            if int(getattr(session, "screen_blank_after_seconds", 0) or 0) <= 0:
                continue
            if session._screen_blank_exempt:
                continue
            if session._screen_blanked:
                continue
            idle = time.monotonic() - session._last_activity
            threshold = int(getattr(session, "screen_blank_after_seconds", 0) or 0)
            if threshold > 0 and idle >= float(threshold):
                await enter_blank_screen(session)
    except asyncio.CancelledError:
        return
