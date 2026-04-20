"""
Main loop for the BBS telnet text editor.
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any, Dict

import yaml

from config.settings import settings

from .buffer import TextBuffer
from .keys import EditorKeyParser, KeyEvent, KeyKind
from .screen import render_frame, scroll_ensure_visible
from .wrap_layout import WrapLayout

if TYPE_CHECKING:
    from session import BBSSession


_EDITABLE_EXT = (".md", ".txt", ".org")
_FRONTMATTER_RE = re.compile(r"^---\r?\n(.*?)\n---(?:\r?\n|$)", re.DOTALL)


def _parse_frontmatter(text: str) -> Dict[str, Any]:
    """Extract and parse YAML frontmatter from document text. Returns {} on failure."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}
    try:
        data = yaml.safe_load(m.group(1))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _language_for_filename(filename: str) -> str:
    fn = (filename or "").lower()
    if fn.endswith(".org"):
        return "org"
    return "markdown"


def is_editable_filename(name: str) -> bool:
    n = (name or "").lower()
    return any(n.endswith(ext) for ext in _EDITABLE_EXT)


async def _run_ai_chat(
    session: "BBSSession",
    buf: TextBuffer,
    doc_id: str,
    title: str,
) -> None:
    """Prompt for a query, send the open document as active_editor context, display the response."""
    from rendering.paginator import paginate_text
    from rendering.text import markdown_to_ansi, word_wrap

    # Prompt at the bottom of the screen
    await session._write("\r\n\r\nAsk AI (Enter to cancel): ")
    query = (
        await session.read_line(
            timeout=300.0,
            history_tag="doc_ai_chat",
            line_prefix="Ask AI (Enter to cancel): ",
        )
    ).strip()
    if not query:
        return

    # Build active_editor payload from the current buffer state
    content = buf.to_text()
    frontmatter = _parse_frontmatter(content)
    filename = title.replace(" - edit", "").strip() if title.endswith(" - edit") else title
    active_editor = {
        "is_editable": True,
        "filename": filename,
        "language": _language_for_filename(filename),
        "content": content,
        "content_length": len(content),
        "frontmatter": frontmatter,
        "cursor_offset": buf.cursor_offset(),
        "selection_start": -1,
        "selection_end": -1,
        "document_id": doc_id,
        "folder_id": None,
        "canonical_path": None,
    }

    # Show spinner while the assistant works
    t = session.theme
    task = asyncio.create_task(
        session.client.send_chat_with_editor_context(
            session.jwt_token,
            session.conversation_id,
            query,
            active_editor,
        )
    )
    frames = ("|", "/", "-", "\\")
    fi = 0
    await session._write("\r\n")
    while not task.done():
        status = f"\r  {t.dim}{frames[fi]} Assistant is working...{t.reset}"
        await session._write_bytes(status.encode("utf-8", errors="replace"))
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
        await session._write(f"Error: {result['error'][:400]}\r\n\r\nPress Enter to return to editor...")
        await session.read_line(timeout=60.0)
        return

    response_text = result.get("response") or ""
    await session.clear_screen()
    await session._write(f"{t.bold}AI Response{t.reset}\r\n\r\n")

    rendered = markdown_to_ansi(response_text, session.theme)
    lines = word_wrap(rendered, session.term_width - 2)
    page_h = max(5, session.term_height - 4)

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
    await session._write("\r\nPress Enter to return to editor...")
    await session.read_line(timeout=120.0)


async def run_document_editor(session: "BBSSession", doc_id: str, initial_text: str, title: str) -> None:
    max_b = settings.BBS_EDITOR_MAX_BYTES
    raw = initial_text.encode("utf-8")
    if len(raw) > max_b:
        await session._write(
            f"File too large for BBS editor ({len(raw)} bytes; max {max_b}). Use the web app.\r\n"
        )
        return

    buf = TextBuffer(initial_text)
    saved_snapshot = buf.to_text()
    dirty = False
    scroll_visual = 0
    parser = EditorKeyParser()
    show_help = False

    def _viewport_h() -> int:
        status_lines = 2 if show_help else 1
        return max(5, session.term_height - 1 - status_lines)

    def _text_width() -> int:
        return max(40, session.term_width) - 1

    def _visual_step(delta: int) -> None:
        layout = WrapLayout(buf.lines, _text_width())
        vi = layout.visual_from_logical(buf.row, buf.col)
        r, c = layout.step_visual(vi, delta, buf.col)
        buf.set_cursor(r, c)

    async def redraw() -> None:
        nonlocal scroll_visual
        vh = _viewport_h()
        layout = WrapLayout(buf.lines, _text_width())
        vi = layout.visual_from_logical(buf.row, buf.col)
        scroll_visual = scroll_ensure_visible(scroll_visual, vi, vh, layout.total_visual())
        await session.clear_screen()
        frame = render_frame(
            buf,
            layout,
            scroll_visual,
            session.term_height,
            session.term_width,
            session.theme,
            title,
            dirty,
            show_help,
        )
        await session._write(frame)

    await redraw()

    while True:
        ev = await session.read_editor_key(parser)
        if ev.kind == KeyKind.SAVE:
            result = await session.client.put_document_content(session.jwt_token, doc_id, buf.to_text())
            if result.get("error"):
                await session._write(f"\r\nSave failed: {result['error'][:300]}\r\nPress Enter...")
                await session.read_line(timeout=300.0)
            else:
                saved_snapshot = buf.to_text()
                dirty = False
            await redraw()
            continue

        if ev.kind in (KeyKind.QUIT, KeyKind.ESCAPE):
            if dirty:
                await session._write("\r\nQuit without saving? [y/N]: ")
                ans = (await session.read_line(timeout=120.0)).strip().lower()
                if ans not in ("y", "yes"):
                    await redraw()
                    continue
            parser.reset()
            return

        if ev.kind == KeyKind.HELP:
            show_help = not show_help
            await redraw()
            continue

        if ev.kind == KeyKind.AI_CHAT:
            await _run_ai_chat(session, buf, doc_id, title)
            await redraw()
            continue

        changed = False
        if ev.kind == KeyKind.CHAR and ev.char:
            buf.insert_char(ev.char)
            changed = True
        elif ev.kind == KeyKind.TAB:
            buf.insert_tab(4)
            changed = True
        elif ev.kind == KeyKind.ENTER:
            buf.newline()
            changed = True
        elif ev.kind == KeyKind.BACKSPACE:
            buf.backspace()
            changed = True
        elif ev.kind == KeyKind.DELETE:
            buf.delete()
            changed = True
        elif ev.kind == KeyKind.UP:
            _visual_step(-1)
        elif ev.kind == KeyKind.DOWN:
            _visual_step(1)
        elif ev.kind == KeyKind.LEFT:
            buf.move_left()
        elif ev.kind == KeyKind.RIGHT:
            buf.move_right()
        elif ev.kind == KeyKind.HOME:
            buf.move_home()
        elif ev.kind == KeyKind.END:
            buf.move_end()
        elif ev.kind == KeyKind.PGUP:
            _visual_step(-_viewport_h())
        elif ev.kind == KeyKind.PGDN:
            _visual_step(_viewport_h())
        elif ev.kind == KeyKind.CTRL_UP:
            _visual_step(-_viewport_h())
        elif ev.kind == KeyKind.CTRL_DOWN:
            _visual_step(_viewport_h())

        if changed:
            dirty = buf.to_text() != saved_snapshot

        await redraw()
