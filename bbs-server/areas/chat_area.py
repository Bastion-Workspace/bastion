"""
System Chat loop with slash commands (mirrors connections-service channel_listener_manager).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, List, Optional

from rendering.paginator import paginate_text
from rendering.text import (
    format_header_context,
    markdown_to_ansi,
    normalize_for_telnet,
    section_header,
    word_wrap,
)

if TYPE_CHECKING:
    from session import BBSSession

logger = logging.getLogger(__name__)


def _format_relative_time(updated_at: Optional[str]) -> str:
    if not updated_at:
        return "unknown"
    try:
        if isinstance(updated_at, str) and "T" in updated_at:
            dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        else:
            return str(updated_at)[:10]
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delta = now - dt
        if delta < timedelta(minutes=1):
            return "just now"
        if delta < timedelta(hours=1):
            return f"{int(delta.total_seconds() / 60)}m ago"
        if delta < timedelta(hours=24):
            return f"{int(delta.total_seconds() / 3600)}h ago"
        if delta < timedelta(days=2):
            return "yesterday"
        if delta < timedelta(days=7):
            return f"{delta.days}d ago"
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return "unknown"


async def _write_wrapped(session: "BBSSession", text: str) -> None:
    t = markdown_to_ansi(text, session.theme)
    lines = word_wrap(t, session.term_width - 2)
    page_h = max(5, session.term_height - 3)

    async def wl(s: str) -> None:
        await session._write(s)

    await paginate_text(
        lines,
        page_h,
        session.theme,
        wl,
        session.read_pager_key,
        after_line_input_drain=session.drain_stray_line_terminators,
    )


_HELP_ROWS = (
    ("/help", "Show this command list"),
    ("/quit", "Return to main menu (also /exit, /menu)"),
    ("", ""),
    ("-- Conversations --", ""),
    ("/chats", "List your recent conversations"),
    ("/loadchat N", "Switch to conversation N from /chats"),
    ("/newchat", "Start a fresh conversation"),
    ("/chat", "Show the current thread title (also /current)"),
    ("/threads", "Open the full thread browser"),
    ("/chatid", "Show this session's platform chat id"),
    ("", ""),
    ("-- History --", ""),
    ("/history", "Quick peek: last 20 messages, one line each"),
    ("/scrollback", "Full-screen reader over this conversation (also /back, /scroll)"),
    ("", ""),
    ("-- Model --", ""),
    ("/model", "List enabled models; * marks current"),
    ("/model N", "Select model by number"),
    ("", ""),
    ("-- Line editing --", ""),
    ("Up / Down", "Previous / next message you sent"),
    ("Ctrl-A / E", "Start / end of line"),
    ("Ctrl-U / K", "Delete to start / end of line"),
    ("Ctrl-W", "Delete previous word"),
    ("Ctrl-L", "Redraw current prompt"),
)


async def _show_help(session: "BBSSession") -> None:
    t = session.theme
    width = session.term_width - 2
    left_w = max(12, min(24, width // 3))
    hdr = section_header(
        "System Chat Help",
        width,
        t,
        context=format_header_context(session.display_name or session.username),
    )
    lines: List[str] = [hdr, ""]
    for cmd, desc in _HELP_ROWS:
        if not cmd and not desc:
            lines.append("")
            continue
        if cmd.startswith("--"):
            lines.append(f"{t.dim}{cmd}{t.reset}")
            continue
        cmd_col = cmd.ljust(left_w)
        for i, seg in enumerate(word_wrap(desc, max(10, width - left_w - 2))):
            prefix = cmd_col if i == 0 else " " * left_w
            lines.append(f"{prefix}  {seg}")
    lines.append("")
    lines.append(f"{t.dim}Any other line is sent to the assistant.{t.reset}")

    page_h = max(5, session.term_height - 3)

    async def wl(s: str) -> None:
        await session._write(s)

    await session._write("\r\n")
    await paginate_text(
        lines,
        page_h,
        t,
        wl,
        session.read_pager_key,
        after_line_input_drain=session.drain_stray_line_terminators,
    )
    await session._write("\r\n")


def _format_scrollback_lines(
    session: "BBSSession",
    messages: List[dict],
    conv_title: str,
) -> List[str]:
    """Render messages oldest-first with role badges, wrapped to terminal width."""
    t = session.theme
    width = session.term_width - 2
    body_indent = "  "
    body_width = max(20, width - len(body_indent))

    out: List[str] = []
    total = len(messages)
    out.append(f"{t.dim}{total} message(s) in: {conv_title[:60]}{t.reset}")
    out.append("")
    for idx, m in enumerate(messages, 1):
        role = (m.get("role") or "").lower()
        ts = _format_relative_time(m.get("created_at") or m.get("timestamp") or m.get("updated_at"))
        content = (m.get("content") or "").strip()
        if role == "user":
            label = f"{t.fg_bright_green}You{t.reset}"
        elif role == "assistant":
            label = f"{t.fg_bright_cyan}Assistant{t.reset}"
        elif role == "system":
            label = f"{t.dim}System{t.reset}"
        else:
            label = role or "?"
        header_line = f"[{idx:>3}] {label}  {t.dim}{ts}{t.reset}"
        out.append(header_line)
        if not content:
            out.append(body_indent + f"{t.dim}(empty){t.reset}")
        else:
            rendered = markdown_to_ansi(content, t)
            for seg in word_wrap(rendered, body_width):
                out.append(body_indent + seg)
        out.append("")
    return out


async def _open_scrollback(session: "BBSSession", conv_title: str) -> None:
    client = session.client
    t = session.theme
    res = await client.get_conversation_messages(
        session.jwt_token, session.conversation_id, limit=200,
    )
    if res.get("error"):
        await session._write(f"Sorry: {str(res['error'])[:200]}\r\n")
        return
    msgs = res.get("messages") or res.get("data") or []
    if isinstance(msgs, dict):
        msgs = msgs.get("messages") or []
    if not msgs:
        await session._write("No messages in this conversation yet.\r\n")
        return

    # Heuristic: many /messages endpoints return newest-first when most_recent=true.
    # Present in chronological order (oldest at top) so Space feels like normal reading.
    if len(msgs) >= 2:
        first_t = (msgs[0].get("created_at") or msgs[0].get("timestamp") or "")
        last_t = (msgs[-1].get("created_at") or msgs[-1].get("timestamp") or "")
        if first_t and last_t and first_t > last_t:
            msgs = list(reversed(msgs))

    await session.clear_screen()
    hdr = section_header(
        "Scrollback",
        session.term_width - 2,
        t,
        context=format_header_context(session.display_name or session.username),
    )
    await session._write(f"\r\n{hdr}\r\n")
    await session._write(
        f"{t.dim}Space/Enter = next  b/p = prev page  Q = back to chat{t.reset}\r\n\r\n"
    )

    lines = _format_scrollback_lines(session, msgs, conv_title)
    page_h = max(5, session.term_height - 5)

    async def wl(s: str) -> None:
        await session._write(s)

    await paginate_text(
        lines,
        page_h,
        t,
        wl,
        session.read_pager_key,
        after_line_input_drain=session.drain_stray_line_terminators,
    )
    await session._write(f"\r\n{t.dim}End of scrollback.{t.reset}\r\n")
    await session._write("Press Enter to return to chat... ")
    await session.read_line()


async def _resolve_conv_title(client, uid: str, conversation_id: str) -> str:
    try:
        val = await client.validate_conversation(uid, conversation_id)
        if val.get("valid") and val.get("title"):
            return (val["title"])[:50]
    except Exception:
        pass
    return "(new)"


async def chat_loop(session: "BBSSession") -> None:
    await session.clear_screen()
    client = session.client
    uid = session.user_id
    plat = "bbs"
    chat_key = session.platform_chat_id

    conv_title = await _resolve_conv_title(client, uid, session.conversation_id)

    hdr = section_header(
        "System Chat",
        session.term_width - 2,
        session.theme,
        context=format_header_context(session.display_name or session.username),
    )
    await session._write(
        f"\r\n{hdr}\r\n"
        f"  Orchestrator assistant (not Messaging rooms). /help for commands, /quit to menu.\r\n"
        f"  Conversation: {conv_title}\r\n\r\n"
    )

    while True:
        prompt_label = conv_title[:20] if conv_title and conv_title != "(new)" else "You"
        pfx = f"{prompt_label}> "
        await session._write(pfx)
        text = await session.read_line(history_tag="chat", line_prefix=pfx)
        if not text:
            continue
        tl = text.lower().strip()
        if tl in ("/quit", "/exit", "/menu"):
            await session._write("\r\n")
            return

        if tl in ("/help", "/?", "/h"):
            await _show_help(session)
            continue

        if tl in ("/scrollback", "/back", "/scroll"):
            await _open_scrollback(session, conv_title)
            continue

        if tl in ("/threads", "/thread"):
            from areas import conversations_area

            await conversations_area.conversations_browser(session)
            conv_title = await _resolve_conv_title(client, uid, session.conversation_id)
            await session._write(f"\r\n[System Chat] Current thread: {conv_title[:50]}\r\n")
            continue

        if tl in ("/chatid", "/chat_id"):
            await session._write(f"This session id: `{session.platform_chat_id}`\r\n")
            continue

        if tl == "/newchat":
            new_id = f"{plat}:{uid}:{chat_key}:{uuid.uuid4()}"
            session.conversation_id = new_id
            result = await client.start_new_conversation(
                user_id=uid,
                conversation_id=new_id,
                platform=plat,
                platform_chat_id=chat_key,
            )
            reply = (
                result.get("response", "Started a new chat.")
                if not result.get("error")
                else f"Sorry: {result.get('error', '')[:200]}"
            )
            conv_title = "(new)"
            await session._write(reply + "\r\n")
            continue

        if tl == "/model":
            result = await client.list_models(uid, conversation_id=session.conversation_id or "")
            if result.get("error"):
                await session._write(f"Sorry: {result.get('error', '')[:200]}\r\n")
            else:
                models = result.get("models", [])
                if not models:
                    await session._write("No enabled models configured.\r\n")
                else:
                    cur = result.get("current_model_id")
                    parts = []
                    for i, m in enumerate(models):
                        idx = m.get("index", i + 1)
                        name = m.get("name", m.get("id", "?"))
                        if cur and m.get("id") == cur:
                            parts.append(f"{idx}) **{name}** *")
                        else:
                            parts.append(f"{idx}) {name}")
                    await session._write(
                        "Reply with /model <number> to select.\r\n\r\n" + "\r\n".join(parts) + "\r\n"
                    )
            continue

        if tl.startswith("/model ") and tl[7:].strip().isdigit():
            idx = int(tl[7:].strip())
            result = await client.set_model(uid, session.conversation_id, idx)
            if result.get("error"):
                await session._write(f"Sorry: {result.get('error', '')[:200]}\r\n")
            else:
                await session._write(result.get("response", "Model set.") + "\r\n")
            continue

        if tl == "/chats":
            result = await client.list_user_conversations(uid, limit=15)
            if result.get("error"):
                await session._write(f"Sorry: {result.get('error', '')[:200]}\r\n")
            else:
                convos: List[dict] = result.get("conversations") or []
                session.chat_listing_cache = convos
                if not convos:
                    await session._write("No conversations yet. Use /newchat.\r\n")
                else:
                    lines = ["Your conversations:"]
                    for i, c in enumerate(convos, 1):
                        title = (c.get("title") or "Untitled")[:50]
                        when = _format_relative_time(c.get("updated_at"))
                        cnt = c.get("message_count", 0)
                        lines.append(f"{i}) {title} ({when}, {cnt} msgs)")
                    lines.append("")
                    lines.append("Use /loadchat N to switch.")
                    await session._write("\r\n".join(lines) + "\r\n")
            continue

        if tl.startswith("/loadchat "):
            rest = tl[len("/loadchat ") :].strip()
            if not rest.isdigit():
                await session._write("Use /loadchat N (e.g. /loadchat 2).\r\n")
                continue
            n = int(rest)
            cached = session.chat_listing_cache
            if not cached or n < 1 or n > len(cached):
                await session._write("Run /chats first, then /loadchat N.\r\n")
                continue
            chosen = cached[n - 1]
            conv_id = chosen.get("conversation_id")
            if not conv_id:
                await session._write("Invalid entry. Run /chats again.\r\n")
                continue
            val = await client.validate_conversation(uid, conv_id)
            if not val.get("valid"):
                await session._write("That conversation is no longer available.\r\n")
                continue
            session.conversation_id = conv_id
            conv_title = (val.get("title") or "Untitled")[:50]
            await session._write(f"Switched to: {conv_title}\r\n")
            continue

        if tl in ("/chat", "/current"):
            val = await client.validate_conversation(uid, session.conversation_id)
            title = (val.get("title") or "Untitled") if val.get("valid") else "(new)"
            await session._write(
                f"Current: {title[:50]}\r\nUse /chats for list, /newchat for fresh.\r\n"
            )
            continue

        if tl == "/history":
            res = await client.get_conversation_messages(
                session.jwt_token, session.conversation_id, limit=20,
            )
            if res.get("error"):
                await session._write(f"Sorry: {res['error'][:200]}\r\n")
            else:
                msgs = res.get("messages") or res.get("data") or []
                if isinstance(msgs, dict):
                    msgs = msgs.get("messages") or []
                if not msgs:
                    await session._write("No messages yet in this conversation.\r\n")
                else:
                    hist_lines: list[str] = []
                    for m in msgs:
                        role = m.get("role", "")
                        content = normalize_for_telnet((m.get("content") or "")[:200])
                        content = content.replace("\n", " ")
                        if role == "user":
                            hist_lines.append(f"  You> {content}")
                        else:
                            hist_lines.append(f"  Assistant: {content}")
                    hist_text = "\r\n".join(hist_lines)
                    await session._write(f"\r\n{hist_text}\r\n\r\n")
            continue

        t = session.theme
        await session._write("\r\n")

        task = asyncio.create_task(client.send_external_message(
            user_id=uid,
            conversation_id=session.conversation_id,
            query=text,
            platform=plat,
            platform_chat_id=chat_key,
            sender_name=session.display_name or session.username,
        ))
        frames = ("|", "/", "-", "\\")
        fi = 0
        while not task.done():
            status = f"  {t.dim}{frames[fi]} Assistant is working...{t.reset}"
            await session._write_bytes(b"\r" + status.encode("utf-8", errors="replace"))
            fi = (fi + 1) % len(frames)
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=0.25)
            except asyncio.TimeoutError:
                pass
            except Exception:
                break
        blank = " " * 40
        await session._write_bytes(b"\r" + blank.encode() + b"\r\n")
        result = task.result()
        if result.get("error"):
            err = normalize_for_telnet(result["error"][:400])
            if "service key" in err.lower():
                err += (
                    "\r\nSet INTERNAL_SERVICE_KEY on the BBS container to the same value "
                    "as the backend (see connections-service / docker-compose)."
                )
            await session._write(f"Error: {err}\r\n\r\n")
            continue
        response_text = result.get("response") or ""
        await session._write("Assistant:\r\n")
        await _write_wrapped(session, response_text)
        await session._write("\r\n")
