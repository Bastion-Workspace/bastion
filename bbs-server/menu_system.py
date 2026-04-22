"""
Main menu and navigation between BBS areas.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rendering.text import draw_box, format_header_datetime

if TYPE_CHECKING:
    from session import BBSSession

logger = logging.getLogger(__name__)


def _normalize_menu_choice(raw: str) -> str:
    """
    Accept C, c, [C], "Chat", "  m ", etc. Returns a single letter/symbol or lowered word.
    """
    s = (raw or "").strip().lower()
    if not s:
        return ""
    for ch in s:
        if ch.isalpha() or ch == "!":
            return ch
    return s[0]


async def main_menu(session: "BBSSession") -> None:
    from areas import admin_area
    from areas import chat_area
    from areas import data_area
    from areas import messaging_area
    from areas import documents_area
    from areas import games_area
    from areas import org_desk_area
    from areas import rss_area
    from areas import settings_area
    from areas import wallpaper_area

    t = session.theme
    while True:
        await session.clear_screen()
        lines = draw_box(
            session.display_name or session.username,
            session.term_width - 4,
            t,
            subtitle=format_header_datetime(),
        )
        for line in lines:
            await session._write(line + "\r\n")
        await session._write(
            f"{t.fg_bright_green}[C]{t.reset} System Chat    "
            f"{t.fg_bright_green}[M]{t.reset} Messaging\r\n"
        )
        await session._write(
            f"{t.fg_bright_green}[F]{t.reset} Files            "
            f"{t.fg_bright_green}[O]{t.reset} Org desk       "
            f"{t.fg_bright_green}[N]{t.reset} RSS News       "
            f"{t.fg_bright_green}[D]{t.reset} Data Workspace\r\n"
        )
        await session._write(
            f"{t.fg_bright_green}[S]{t.reset} Settings         "
            f"{t.fg_bright_green}[W]{t.reset} Wallpaper        "
            f"{t.fg_bright_green}[P]{t.reset} Games\r\n"
        )
        if session.role == "admin":
            await session._write(
                f"{t.fg_bright_yellow}[!]{t.reset} SysOp (admin)\r\n"
            )
        menu_prompt = (
            f"{t.fg_bright_green}[G]{t.reset} Goodbye (logout)\r\n\r\n"
            f"{t.dim}Welcome to your ultimate workspace.{t.reset}\r\n\r\nChoice: "
        )
        await session._write(menu_prompt)
        raw_line = await session.read_menu_choice()
        raw_s = raw_line.strip().lower()
        choice = _normalize_menu_choice(raw_line)
        if not choice:
            await session._write("Please enter a menu letter.\r\n\r\n")
            continue
        # Single-letter g/q only — not "games" (first letter g would match Goodbye otherwise).
        if raw_s in ("goodbye", "quit", "exit", "logout", "g", "q"):
            await session._write("Thanks for calling. Goodbye.\r\n")
            break
        if choice == "c" or raw_s.startswith("chat"):
            await chat_area.chat_loop(session)
        elif choice == "m" or raw_s.startswith("mess") or raw_s.startswith("dm"):
            await messaging_area.messaging_browser(session)
        elif choice == "t" or raw_s.startswith("conv") or raw_s.startswith("thread"):
            await chat_area.chat_loop(session)
        elif choice == "f" or raw_s.startswith("file") or raw_s.startswith("doc"):
            await documents_area.documents_browser(session)
        elif choice == "o" or raw_s.startswith("org"):
            await org_desk_area.org_desk_hub(session)
        elif choice == "n" or raw_s.startswith("news") or raw_s.startswith("rss"):
            await rss_area.rss_reader(session)
        elif choice == "d" or raw_s.startswith("data"):
            await data_area.data_explorer(session)
        elif choice == "s" or raw_s.startswith("sett"):
            await settings_area.settings_menu(session)
        elif choice == "w" or raw_s.startswith("wall"):
            await wallpaper_area.wallpaper_pane(session)
        elif (
            choice == "p"
            or raw_s.startswith("game")
            or raw_s.startswith("oregon")
            or raw_s.startswith("trail")
        ):
            await games_area.games_menu(session)
        elif choice == "!" or raw_s.startswith("sysop") or raw_s.startswith("admin"):
            if session.role == "admin":
                await admin_area.admin_menu(session)
            else:
                await session._write("That option is for administrators only.\r\n\r\n")
        else:
            await session._write("Unknown option.\r\n\r\n")
