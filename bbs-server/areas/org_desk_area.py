"""
Org desk hub: TODOs and agenda.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rendering.text import format_header_context, section_header

from areas.org_agenda_area import agenda_view
from areas.org_todos_area import todos_browser

if TYPE_CHECKING:
    from session import BBSSession


async def org_desk_hub(session: "BBSSession") -> None:
    t = session.theme
    while True:
        await session.clear_screen()
        hdr = section_header(
            "Org desk",
            session.term_width - 2,
            t,
            context=format_header_context(session.display_name or session.username),
        )
        await session._write(f"\r\n{hdr}\r\n\r\n")
        await session._write(
            f"{t.fg_bright_green}[T]{t.reset} TODOs (all org tasks)\r\n"
            f"{t.fg_bright_green}[A]{t.reset} Agenda (scheduled / deadlines)\r\n"
            f"{t.fg_bright_green}[B]{t.reset} Back to main menu\r\n\r\nChoice: "
        )
        raw = (await session.read_menu_choice()).strip().lower()
        if raw in ("b", "back", "q", ""):
            return
        if raw == "t" or raw.startswith("todo"):
            await todos_browser(session)
        elif raw == "a" or raw.startswith("agenda"):
            await agenda_view(session)
        else:
            await session._write("Unknown option.\r\nPress Enter... ")
            await session.read_line()
