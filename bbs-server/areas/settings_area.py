"""
Model selection, theme switching, and profile summary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rendering.ansi import theme_from_name
from rendering.text import format_header_context, section_header

if TYPE_CHECKING:
    from session import BBSSession

_AVAILABLE_THEMES = ("green", "amber", "blue", "none")


async def settings_menu(session: "BBSSession") -> None:
    await session.clear_screen()
    client = session.client
    uid = session.user_id
    t = session.theme

    while True:
        me = await client.get_current_user(session.jwt_token)
        if me.get("error"):
            await session._write(f"Error: {me.get('error', '')[:200]}\r\n")
            return
        hdr = section_header(
            "Settings",
            session.term_width - 2,
            t,
            context=format_header_context(session.display_name or session.username),
        )
        await session._write(
            f"\r\n{hdr}\r\n"
            f"User: {me.get('username', '')}  Role: {me.get('role', '')}\r\n"
            f"Display: {me.get('display_name', '')}\r\n\r\n"
        )
        result = await client.list_models(uid, conversation_id=session.conversation_id or "")
        if not result.get("error"):
            models = result.get("models") or []
            cur = result.get("current_model_id")
            await session._write("Models (use /model N in chat):\r\n")
            for m in models[:25]:
                mark = "*" if cur and m.get("id") == cur else " "
                await session._write(
                    f"  {mark}{m.get('index', '')}) {m.get('name', m.get('id', ''))}\r\n"
                )
        await session._write(
            f"\r\n{t.fg_bright_green}[T]{t.reset}heme  "
            f"{t.fg_bright_green}[R]{t.reset}efresh  "
            f"{t.fg_bright_green}[B]{t.reset}ack: "
        )
        choice = (await session.read_menu_choice()).strip().lower()
        if choice in ("b", "back", "q"):
            return
        if choice in ("r", "refresh"):
            continue
        if choice in ("t", "theme"):
            await _theme_picker(session)
            t = session.theme
            continue


async def _theme_picker(session: "BBSSession") -> None:
    t = session.theme
    await session._write("\r\nSelect theme:\r\n")
    for i, name in enumerate(_AVAILABLE_THEMES, 1):
        cur = " *" if name == _current_theme_name(session.theme) else ""
        await session._write(f"  {i}) {name}{cur}\r\n")
    await session._write("\r\nChoice [1-4]: ")
    raw = (await session.read_line()).strip()
    if raw.isdigit():
        idx = int(raw)
        if 1 <= idx <= len(_AVAILABLE_THEMES):
            chosen = _AVAILABLE_THEMES[idx - 1]
            session.theme = theme_from_name(chosen)
            await session._write(f"Theme set to {chosen}.\r\n")
            return
    for name in _AVAILABLE_THEMES:
        if raw.lower() == name:
            session.theme = theme_from_name(name)
            await session._write(f"Theme set to {name}.\r\n")
            return
    await session._write("No change.\r\n")


def _current_theme_name(theme) -> str:
    if not theme.reset:
        return "none"
    if theme.fg_green == "\033[33m":
        return "amber"
    if theme.fg_green == "\033[34m":
        return "blue"
    return "green"
