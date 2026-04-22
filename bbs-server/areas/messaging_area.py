"""
User-to-user messaging (chat_rooms): list rooms, open, send, poll for new messages.
Uses the same REST API as the web UI; federated rooms work when messaging sends via the backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Set

from areas.chat_area import _format_relative_time
from rendering.tables import render_table
from rendering.text import format_header_context, section_header, word_wrap

if TYPE_CHECKING:
    from session import BBSSession

def _room_row_label(room: Dict) -> str:
    name = (room.get("display_name") or room.get("room_name") or "Room")[:36]
    rt = room.get("room_type") or ""
    if rt == "federated":
        peer = room.get("federation_peer_display_name") or ""
        if peer:
            name = f"{name[:28]} @{peer[:6]}"
        name = f"[Fed] {name}"
    elif rt == "group":
        name = f"[Grp] {name}"
    return name


def _sender_label(msg: Dict, current_user_id: str) -> str:
    if msg.get("sender_id") == current_user_id:
        return "You"
    if msg.get("is_federated"):
        return (msg.get("display_name") or msg.get("username") or "Remote")[:32]
    return (msg.get("display_name") or msg.get("username") or "?")[:32]


async def _print_new_messages(
    session: "BBSSession",
    messages: List[Dict],
    seen: Set[str],
    current_user_id: str,
) -> None:
    for msg in messages:
        mid = msg.get("message_id")
        if not mid or mid in seen:
            continue
        seen.add(mid)
        who = _sender_label(msg, current_user_id)
        raw = (msg.get("content") or "").replace("\r", "")
        for line in word_wrap(raw, max(20, session.term_width - 4)):
            await session._write(f"  {who}: {line}\r\n")


async def messaging_room_loop(session: "BBSSession", room_id: str, title: str) -> None:
    client = session.client
    jwt = session.jwt_token
    uid = session.user_id
    t = session.theme
    seen_ids: Set[str] = set()
    oldest_id: Optional[str] = None

    await session.clear_screen()
    hdr = section_header(
        f"Room: {title[:40]}",
        session.term_width - 2,
        t,
        context=format_header_context(session.display_name or session.username),
    )
    await session._write(f"\r\n{hdr}\r\n")
    await session._write(f"{t.dim}/back  /refresh  /older  - same APIs as web Messaging{t.reset}\r\n\r\n")

    res = await client.messaging_get_room_messages(jwt, room_id, limit=40)
    if res.get("error"):
        await session._write(f"Error: {str(res['error'])[:300]}\r\n")
        await session._write("\r\nPress Enter...")
        await session.read_line()
        return

    msgs: List[Dict] = list(res.get("messages") or [])
    if msgs:
        oldest_id = msgs[0].get("message_id")
    await _print_new_messages(session, msgs, seen_ids, uid)
    await session._write("\r\n")

    while True:
        poll = await client.messaging_get_room_messages(jwt, room_id, limit=50)
        if not poll.get("error"):
            incoming = list(poll.get("messages") or [])
            await _print_new_messages(session, incoming, seen_ids, uid)

        pfx = f"{t.fg_bright_green}>{t.reset} "
        await session._write(pfx)
        line = (await session.read_line(history_tag="messaging", line_prefix=pfx)).rstrip("\r\n")
        if not line:
            continue
        low = line.strip().lower()
        if low in ("/back", "/b", "/menu", "/quit", "/exit"):
            await session._write("\r\n")
            return
        if low in ("/refresh", "/r"):
            res2 = await client.messaging_get_room_messages(jwt, room_id, limit=40)
            if res2.get("error"):
                await session._write(f"Error: {str(res2['error'])[:200]}\r\n")
            else:
                msgs2 = list(res2.get("messages") or [])
                await _print_new_messages(session, msgs2, seen_ids, uid)
            await session._write("\r\n")
            continue
        if low in ("/older", "/more") and oldest_id:
            older = await client.messaging_get_room_messages(
                jwt, room_id, limit=25, before_message_id=oldest_id
            )
            if older.get("error"):
                await session._write(f"Error: {str(older['error'])[:200]}\r\n")
            else:
                batch = list(older.get("messages") or [])
                if not batch:
                    await session._write("(No older messages)\r\n")
                else:
                    await session._write(f"{t.dim}--- older ---{t.reset}\r\n")
                    for msg in batch:
                        mid = msg.get("message_id")
                        if mid and mid not in seen_ids:
                            who = _sender_label(msg, uid)
                            raw = (msg.get("content") or "").replace("\r", "")
                            for wl in word_wrap(raw, max(20, session.term_width - 4)):
                                await session._write(f"  {who}: {wl}\r\n")
                            seen_ids.add(mid)
                    oldest_id = batch[0].get("message_id") or oldest_id
            await session._write("\r\n")
            continue

        send_res = await client.messaging_send_message(jwt, room_id, line)
        if send_res.get("error"):
            await session._write(f"Send failed: {str(send_res['error'])[:300]}\r\n")
            continue
        mid = send_res.get("message_id")
        if mid:
            seen_ids.add(mid)


async def _pick_user_for_dm(session: "BBSSession") -> Optional[str]:
    client = session.client
    jwt = session.jwt_token
    uid = session.user_id
    await session._write("Filter username or display name (empty = show many): ")
    filt = (await session.read_line()).strip().lower()
    ures = await client.messaging_list_users(jwt)
    if ures.get("error"):
        await session._write(f"Error: {str(ures['error'])[:200]}\r\n")
        return None
    users: List[Dict] = list(ures.get("users") or [])
    if filt:
        users = [
            u
            for u in users
            if filt in (u.get("username") or "").lower()
            or filt in (u.get("display_name") or "").lower()
        ]
    if not users:
        await session._write("No matching users.\r\n")
        return None
    users = users[:40]
    rows = []
    for i, u in enumerate(users, 1):
        un = (u.get("username") or "")[:20]
        dn = (u.get("display_name") or "")[:24]
        rows.append((str(i), un, dn))
    tbl = render_table(("#", "Username", "Display"), rows, col_widths=(4, 22, 26), max_width=session.term_width)
    await session._write(f"\r\n{tbl}\r\nPick # for new DM (or 0 to cancel): ")
    pick = (await session.read_line()).strip()
    if not pick.isdigit() or int(pick) == 0:
        return None
    n = int(pick)
    if n < 1 or n > len(users):
        await session._write("Invalid.\r\n")
        return None
    other_id = users[n - 1].get("user_id")
    if not other_id or other_id == uid:
        return None
    return str(other_id)


async def messaging_browser(session: "BBSSession") -> None:
    await session.clear_screen()
    client = session.client
    jwt = session.jwt_token
    uid = session.user_id
    t = session.theme
    hdr = section_header(
        "Messaging",
        session.term_width - 2,
        t,
        context=format_header_context(session.display_name or session.username),
    )
    await session._write(f"\r\n{hdr}\r\n")
    await session._write(
        f"{t.dim}DMs, groups, and federated rooms (same backend as the web app).{t.reset}\r\n"
    )

    while True:
        result = await client.messaging_list_rooms(jwt, limit=25)
        if result.get("error"):
            err = str(result["error"])
            await session._write(f"\r\nError: {err[:400]}\r\n")
            if "503" in err or "not enabled" in err.lower():
                await session._write("Messaging may be disabled (MESSAGING_ENABLED).\r\n")
            await session._write("[B]ack: ")
            if (await session.read_menu_choice()).strip().lower() in ("b", "back", "q"):
                return
            continue

        rooms: List[Dict] = list(result.get("rooms") or [])
        session.messaging_rooms_cache = rooms
        if not rooms:
            await session._write("\r\nNo rooms yet.\r\n")
        else:
            rows = []
            for i, r in enumerate(rooms, 1):
                label = _room_row_label(r)[:44]
                when = _format_relative_time(
                    r.get("last_message_at") or r.get("created_at")
                )
                un = str(r.get("unread_count") or 0)
                rows.append((str(i), label, un, when))
            tbl = render_table(
                ("#", "Room", "Unr", "When"),
                rows,
                col_widths=(4, 46, 5, 12),
                max_width=session.term_width,
            )
            await session._write(f"\r\n{tbl}\r\n")

        await session._write(
            f"\r\n{t.fg_bright_green}[R]{t.reset}ead #  "
            f"{t.fg_bright_green}[N]{t.reset}ew DM  "
            f"{t.fg_bright_green}[B]{t.reset}ack: "
        )
        choice = (await session.read_line()).strip().lower()
        if choice in ("b", "back", "q"):
            return
        if choice in ("n", "new"):
            other = await _pick_user_for_dm(session)
            if not other:
                continue
            created = await client.messaging_create_room(jwt, [other])
            if created.get("error"):
                await session._write(f"Could not create room: {str(created['error'])[:300]}\r\n")
                continue
            rid = created.get("room_id")
            if not rid:
                await session._write("No room_id in response.\r\n")
                continue
            ttl = (created.get("display_name") or created.get("room_name") or "DM")[:50]
            await messaging_room_loop(session, str(rid), ttl)
            continue

        room_index: Optional[int] = None
        if choice in ("r", "read"):
            await session._write(
                "Enter a room number after r or read (examples: r 1, r2, read 3).\r\n"
            )
            continue
        if choice.startswith("read "):
            part = choice[5:].strip()
            if part.isdigit():
                room_index = int(part)
        elif choice.startswith("r ") or (
            choice.startswith("r") and len(choice) > 1 and choice[1:].strip().isdigit()
        ):
            part = choice.replace("r", "", 1).strip()
            if part.isdigit():
                room_index = int(part)

        if room_index is not None:
            if not rooms:
                await session._write("No rooms in list.\r\n")
                continue
            if room_index < 1 or room_index > len(rooms):
                await session._write("Invalid number.\r\n")
                continue
            r = rooms[room_index - 1]
            rid = r.get("room_id")
            if not rid:
                continue
            ttl = _room_row_label(r)[:50]
            await messaging_room_loop(session, str(rid), ttl)
            continue
