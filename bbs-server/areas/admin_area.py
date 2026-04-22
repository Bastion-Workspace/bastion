"""
SysOp menu: list users, create user, toggle active (admin JWT only).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rendering.tables import render_table
from rendering.text import format_header_context, section_header

if TYPE_CHECKING:
    from session import BBSSession


async def admin_menu(session: "BBSSession") -> None:
    if session.role != "admin":
        await session._write("Access denied.\r\n")
        return
    await session.clear_screen()
    client = session.client
    jwt = session.jwt_token
    t = session.theme

    while True:
        hdr = section_header(
            "SysOp",
            session.term_width - 2,
            t,
            context=format_header_context(session.display_name or session.username),
        )
        await session._write(f"\r\n{hdr}\r\n")
        await session._write(
            f"{t.fg_bright_green}[L]{t.reset}ist users  "
            f"{t.fg_bright_green}[C]{t.reset}reate user  "
            f"{t.fg_bright_green}[T]{t.reset}oggle active  "
            f"{t.fg_bright_green}[B]{t.reset}ack\r\nChoice: "
        )
        choice = (await session.read_menu_choice()).strip().lower()
        if choice in ("b", "back", "q"):
            return
        if choice in ("l", "list"):
            data = await client.list_users(jwt, limit=50)
            if data.get("error"):
                await session._write(f"Error: {data['error'][:200]}\r\n")
                continue
            users = data.get("users") or []
            rows = []
            for u in users:
                uid = u.get("user_id", "")[:8]
                un = (u.get("username") or "")[:20]
                act = "yes" if u.get("is_active", True) else "no"
                role = (u.get("role") or "")[:10]
                rows.append((uid, un, role, act))
            tbl = render_table(
                ("ID", "Username", "Role", "Active"),
                rows,
                col_widths=(10, 22, 10, 8),
                max_width=session.term_width,
            )
            await session._write(tbl + "\r\n")
        elif choice in ("c", "create"):
            await session._write("New username: ")
            username = (await session.read_line()).strip()
            await session._write("Email: ")
            email = (await session.read_line()).strip()
            await session._write("Password: ")
            password = await session.read_password(line_prefix="Password: ")
            await session._write("Role (user/admin) [user]: ")
            role = (await session.read_line()).strip() or "user"
            if role not in ("user", "admin"):
                role = "user"
            cr = await client.create_user(jwt, username, email, password, username, role)
            if cr.get("error"):
                await session._write(f"Create failed: {cr['error'][:250]}\r\n")
            else:
                await session._write("User created.\r\n")
        elif choice in ("t", "toggle"):
            await session._write("User ID (full UUID): ")
            uid = (await session.read_line()).strip()
            if len(uid) < 32:
                await session._write("Invalid id.\r\n")
                continue
            await session._write("Set active (y/n): ")
            yn = (await session.read_line()).strip().lower()
            is_active = yn.startswith("y")
            up = await client.update_user(jwt, uid, is_active=is_active)
            if up.get("error"):
                await session._write(f"Update failed: {up['error'][:200]}\r\n")
            else:
                await session._write("Updated.\r\n")
