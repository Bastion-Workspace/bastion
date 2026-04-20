"""
Browse and switch AI assistant threads (orchestrator `conversations` table via internal API).
For user-to-user and federated rooms, use main menu Messaging [M].
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from areas.chat_area import _format_relative_time
from rendering.tables import render_table
from rendering.text import format_header_context, section_header

if TYPE_CHECKING:
    from session import BBSSession


async def conversations_browser(session: "BBSSession") -> None:
    await session.clear_screen()
    client = session.client
    uid = session.user_id
    t = session.theme
    hdr = section_header(
        "AI threads",
        session.term_width - 2,
        t,
        context=format_header_context(session.display_name or session.username),
    )
    await session._write(
        f"\r\n{hdr}\r\n"
        f"{t.dim}Assistant threads only (not [M] Messaging). [B]ack returns to System Chat.{t.reset}\r\n"
    )

    while True:
        result = await client.list_user_conversations(uid, limit=20)
        if result.get("error"):
            await session._write(f"Error: {result['error'][:200]}\r\n")
            return
        convos = result.get("conversations") or []
        session.chat_listing_cache = convos
        if not convos:
            await session._write("\r\nNo conversations.\r\n[N]ew / [B]ack: ")
        else:
            rows = []
            for i, c in enumerate(convos, 1):
                title = (c.get("title") or "Untitled")[:40]
                when = _format_relative_time(c.get("updated_at"))
                cnt = str(c.get("message_count", 0))
                cur = "*" if c.get("conversation_id") == session.conversation_id else ""
                rows.append((str(i), title + cur, cnt, when))
            tbl = render_table(
                ("#", "Title", "Msgs", "When"),
                rows,
                col_widths=(4, 36, 6, 12),
                max_width=session.term_width,
            )
            await session._write(f"\r\n{tbl}\r\n\r\n")
            await session._write(
                f"{t.fg_bright_green}[R]{t.reset}ead #  "
                f"{t.fg_bright_green}[N]{t.reset}ew chat  "
                f"{t.fg_bright_green}[B]{t.reset}ack: "
            )
        choice = (await session.read_line()).strip().lower()
        if choice in ("b", "back", "q"):
            return
        if choice in ("n", "new"):
            import uuid

            new_id = f"bbs:{uid}:{session.platform_chat_id}:{uuid.uuid4()}"
            session.conversation_id = new_id
            r2 = await client.start_new_conversation(
                user_id=uid,
                conversation_id=new_id,
                platform="bbs",
                platform_chat_id=session.platform_chat_id,
            )
            msg = r2.get("response", "New chat.") if not r2.get("error") else r2.get("error", "")[:200]
            await session._write(msg + "\r\n")
            continue
        if choice.startswith("r ") or (choice.startswith("r") and len(choice) > 1 and choice[1:].strip().isdigit()):
            part = choice.replace("r", "", 1).strip()
            if not part.isdigit():
                continue
            n = int(part)
            if n < 1 or n > len(convos):
                await session._write("Invalid number.\r\n")
                continue
            cid = convos[n - 1].get("conversation_id")
            if not cid:
                continue
            val = await client.validate_conversation(uid, cid)
            if not val.get("valid"):
                await session._write("Conversation not available.\r\n")
                continue
            session.conversation_id = cid
            await session._write(f"Switched to: {(val.get('title') or '')[:50]}\r\n")
