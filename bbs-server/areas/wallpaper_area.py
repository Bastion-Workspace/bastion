"""
Full-screen BBS wallpaper from user setting bbs_wallpaper; polls for live updates.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List

from rendering.text import normalize_for_telnet

if TYPE_CHECKING:
    from session import BBSSession

logger = logging.getLogger(__name__)

_POLL_SECONDS = 5.0


def _split_wallpaper_lines(raw: str) -> List[str]:
    text = normalize_for_telnet(raw or "")
    return text.splitlines()


def _fit_wallpaper_line(ln: str, tw: int, *, center: bool) -> str:
    """Fit a line to terminal width without splitting escape sequences."""
    if tw <= 0:
        return ""
    esc = "\x1b" in ln or "\033" in ln
    if esc:
        if len(ln) >= tw:
            return ln
        return ln + (" " * (tw - len(ln)))
    if len(ln) >= tw:
        return ln[:tw]
    pad = tw - len(ln)
    if center:
        left = pad // 2
        return " " * left + ln + " " * (pad - left)
    return ln + (" " * pad)


async def _draw_wallpaper(
    session: "BBSSession",
    raw: str,
    *,
    cycling: bool = False,
    animated: bool = False,
    fps: float = 0.0,
    full_clear: bool = True,
    draw_footer: bool = False,
) -> None:
    tw = max(10, session.term_width)
    th = max(5, session.term_height)
    content_h = th - 1
    # Full erase every frame causes visible flicker (including the footer). After an initial
    # clear, move cursor home and overwrite the same grid — same geometry, no 2J flash.
    if full_clear:
        await session.clear_screen()
    elif session.theme.clear_screen:
        await session._write("\x1b[H")
    else:
        await session.clear_screen()

    lines = _split_wallpaper_lines(raw)
    if not any(ln.strip() for ln in lines):
        msg = " (No wallpaper - set in web Settings > Wallpaper tab) "
        msg = msg.strip()
        if len(msg) > tw:
            msg = msg[: tw - 3] + "..."
        pad = max(0, (tw - len(msg)) // 2)
        pad_top = max(0, (content_h - 1) // 2)
        pad_bot = max(0, content_h - 1 - pad_top)
        for _ in range(pad_top):
            await session._write("\r\n")
        await session._write(" " * pad + msg + "\r\n")
        for _ in range(pad_bot):
            await session._write("\r\n")
    else:
        if len(lines) > content_h:
            start = (len(lines) - content_h) // 2
            lines = lines[start : start + content_h]
        center_short = not (animated and fps > 0)
        lines = [_fit_wallpaper_line(ln, tw, center=center_short) for ln in lines]
        pad_v = max(0, (content_h - len(lines)) // 2)
        for _ in range(pad_v):
            await session._write("\r\n")
        for ln in lines:
            await session._write(ln + "\r\n")
        used = pad_v + len(lines)
        for _ in range(used, content_h):
            await session._write("\r\n")

    if not draw_footer:
        return

    if animated and fps > 0:
        footer = f" Wallpaper  animated ~{fps:.0f}fps  q=exit  r=refresh  poll "
    elif cycling:
        footer = " Wallpaper  q=exit  r=refresh  poll 5s + cycle "
    else:
        footer = " Wallpaper  q=exit  r=refresh  poll 5s "
    footer = footer.strip()
    if len(footer) > tw:
        footer = footer[: max(1, tw - 3)] + "..."
    pad_f = max(0, (tw - len(footer)) // 2)
    await session._write(" " * pad_f + footer + "\r\n")


async def _run_animation_loop(session: "BBSSession", anim: Dict[str, Any]) -> str:
    """Play frames until quit or refresh. Returns 'quit' or 'refresh'."""
    frames = anim.get("frames") or []
    if not isinstance(frames, list) or not frames:
        return "refresh"
    norm_frames: List[str] = []
    for fr in frames:
        norm_frames.append(fr if isinstance(fr, str) else str(fr))
    if not norm_frames:
        return "refresh"
    fps = float(anim.get("fps") or 12.0)
    fps = max(1.0, min(30.0, fps))
    loop_anim = bool(anim.get("loop", True))
    interval = 1.0 / fps
    incremental_ok = bool(session.theme.clear_screen)
    first_frame = True
    while True:
        for frame in norm_frames:
            paint_chrome = first_frame or not incremental_ok
            await _draw_wallpaper(
                session,
                frame,
                cycling=False,
                animated=True,
                fps=fps,
                full_clear=paint_chrome,
                draw_footer=False,
            )
            first_frame = False
            end = time.monotonic() + interval
            while time.monotonic() < end:
                slice_to = min(1.0, max(0.12, end - time.monotonic()))
                key = await session.read_wallpaper_poll_key(slice_to)
                if key == "quit":
                    return "quit"
                if key == "refresh":
                    return "refresh"
        if not loop_anim:
            while True:
                key = await session.read_wallpaper_poll_key(_POLL_SECONDS)
                if key == "quit":
                    return "quit"
                if key == "refresh":
                    return "refresh"


async def wallpaper_pane(session: "BBSSession") -> None:
    session.set_screen_blank_exempt(True)
    last_art: str | None = None
    last_cycling: bool | None = None
    last_tw = -1
    last_th = -1
    try:
        while True:
            tw, th = session.term_width, session.term_height
            content_h = max(5, th - 1)
            try:
                new_art, cycling, anim = await session.client.get_bbs_wallpaper(
                    session.jwt_token, tw, content_h
                )
            except Exception as e:
                logger.warning("wallpaper fetch failed: %s", e)
                new_art, cycling, anim = "", False, None

            if isinstance(anim, dict) and isinstance(anim.get("frames"), list) and anim["frames"]:
                key = await _run_animation_loop(session, anim)
                if key == "quit":
                    return
                continue

            size_changed = tw != last_tw or th != last_th
            cyc_changed = cycling != last_cycling
            if new_art != last_art or size_changed or cyc_changed:
                await _draw_wallpaper(session, new_art, cycling=cycling)
                last_art = new_art
                last_cycling = cycling
                last_tw, last_th = tw, th

            key = await session.read_wallpaper_poll_key(_POLL_SECONDS)
            if key == "quit":
                return
            if key == "refresh":
                last_art = None
    finally:
        session.set_screen_blank_exempt(False)
