"""
BBS games submenu (doors). Oregon Trail and future text-mode games.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rendering.text import format_header_context, section_header

if TYPE_CHECKING:
    from session import BBSSession


async def games_menu(session: "BBSSession") -> None:
    from areas import oregon_trail_area

    t = session.theme
    while True:
        await session.clear_screen()
        hdr = section_header(
            "Games",
            session.term_width - 2,
            t,
            context=format_header_context(session.display_name or session.username),
        )
        await session._write(
            f"\r\n{hdr}\r\n"
            f"{t.dim}Text-mode games.{t.reset}\r\n\r\n"
        )
        await session._write(
            f"  {t.fg_bright_green}[O]{t.reset} Oregon Trail\r\n"
            f"  {t.fg_bright_green}[B]{t.reset} Back to main menu\r\n\r\n"
            f"Choice: "
        )
        raw_line = await session.read_line()
        raw_s = raw_line.strip().lower()
        if not raw_s:
            continue
        if raw_s in ("b", "back", "q", "quit", "/quit", "/menu"):
            return
        choice = raw_s[0] if raw_s else ""
        if choice == "o" or raw_s.startswith("oregon") or raw_s.startswith("trail"):
            await oregon_trail_area.oregon_trail_menu(session, back_label="Back to Games")
        else:
            await session._write("Unknown option.\r\n\r\n")
