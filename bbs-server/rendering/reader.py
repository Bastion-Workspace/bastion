"""
Full-screen less-like file reader for telnet/ANSI clients (CP437-safe navigation).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List, Optional, Tuple

from rendering.text import format_header_context, markdown_to_ansi, section_header, word_wrap

if TYPE_CHECKING:
    from session import BBSSession

_REV_ON = "\x1b[7m"
_REV_OFF = "\x1b[27m"
_CSI = re.compile(r"\x1b\[[0-9;?]*[0-9A-Za-z]")


def _strip_ansi(s: str) -> str:
    return _CSI.sub("", s)


def _visible_char_bounds(line: str) -> List[int]:
    """Start index in line for each visible character; last entry is len(line)."""
    i = 0
    starts: List[int] = []
    while i < len(line):
        m = _CSI.match(line, i)
        if m:
            i = m.end()
            continue
        starts.append(i)
        i += 1
    starts.append(len(line))
    return starts


def _apply_rev_span(line: str, plain_start: int, plain_end: int) -> str:
    """Wrap visible characters [plain_start, plain_end) in reverse video."""
    bounds = _visible_char_bounds(line)
    vis = len(bounds) - 1
    if vis == 0:
        return line
    ps = max(0, min(plain_start, vis))
    pe = max(ps, min(plain_end, vis))
    if pe == ps:
        return line
    rs = bounds[ps]
    re_ = bounds[pe]
    return line[:rs] + _REV_ON + line[rs:re_] + _REV_OFF + line[re_:]


def _search_flags(pattern: str) -> int:
    if re.search(r"[A-Z]", pattern):
        return 0
    return re.IGNORECASE


def _collect_matches(pattern: str, display_lines: List[str]) -> List[Tuple[int, int, int]]:
    if not pattern:
        return []
    esc = re.escape(pattern)
    flags = _search_flags(pattern)
    out: List[Tuple[int, int, int]] = []
    for li, dl in enumerate(display_lines):
        pl = _strip_ansi(dl)
        try:
            for m in re.finditer(esc, pl, flags):
                out.append((li, m.start(), m.end()))
        except re.error:
            continue
    return out


def _clamp_offset(offset: int, total: int, page_h: int) -> int:
    if total <= page_h:
        return 0
    return max(0, min(offset, total - page_h))


def _scroll_to_line(offset: int, line_idx: int, total: int, page_h: int) -> int:
    if total <= page_h:
        return 0
    target = line_idx - page_h // 2
    return _clamp_offset(target, total, page_h)


async def view_text_document(
    session: "BBSSession",
    body: str,
    *,
    title: str,
    subtitle: str = "",
    truncated: bool = False,
) -> None:
    t = session.theme
    width = max(20, session.term_width - 2)
    rendered = markdown_to_ansi(body, t)
    display_lines = word_wrap(rendered, width)
    if not display_lines:
        display_lines = [""]

    total = len(display_lines)
    page_h = max(5, session.term_height - 5)
    offset = 0
    search_pattern: Optional[str] = None
    matches: List[Tuple[int, int, int]] = []
    match_cursor = 0
    transient = ""

    while True:
        offset = _clamp_offset(offset, total, page_h)
        at_end = total <= page_h or offset + page_h >= total
        first = offset + 1
        last = min(offset + page_h, total)
        pct = int(100 * last / total) if total else 100
        end_tag = "(END, truncated)" if truncated and at_end else "(END)" if at_end else ""

        status_parts = [
            f"Line {first}-{last} of {total} ({pct}%)",
            end_tag,
            "Space=next B=back g/G=top/bottom /=search n/N=match Q=quit",
        ]
        if transient:
            status_parts.insert(0, transient)
        transient = ""
        status = "  ".join(status_parts)
        if len(status) > session.term_width - 1:
            status = status[: max(20, session.term_width - 4)] + "..."

        await session.clear_screen()
        hdr = section_header(
            f"File: {(title or '?')[:50]}",
            session.term_width - 2,
            t,
            context=format_header_context(session.display_name or session.username),
        )
        await session._write(f"\r\n{hdr}\r\n")
        sub = (subtitle or "").strip()
        if sub:
            await session._write(f"{t.dim}{sub[: session.term_width - 4]}{t.reset}\r\n")
        if truncated:
            await session._write(f"{t.dim}(file truncated to 80000 chars){t.reset}\r\n")
        await session._write("\r\n")

        cur_match = matches[match_cursor] if matches and 0 <= match_cursor < len(matches) else None

        for row in range(page_h):
            idx = offset + row
            if idx >= total:
                await session._write("\r\n")
                continue
            line = display_lines[idx]
            if cur_match and cur_match[0] == idx:
                _, ms, me = cur_match
                line = _apply_rev_span(line, ms, me)
            await session._write(line + "\r\n")

        await session._write(f"{t.dim}{status}{t.reset}\r\n")

        action = await session.read_reader_key()

        if action == "quit":
            return
        if action == "top":
            offset = 0
            continue
        if action == "bottom":
            offset = _clamp_offset(max(0, total - page_h), total, page_h)
            continue
        if action == "prev":
            offset = max(0, offset - page_h)
            continue
        if action == "next":
            if at_end:
                continue
            offset = min(offset + page_h, max(0, total - page_h))
            continue
        if action == "search":
            await session._write(f"\r\n{t.dim}Search (empty to cancel):{t.reset} ")
            raw = await session.read_line(history_tag="file_viewer_search")
            pat = raw.strip()
            if not pat:
                transient = "(search cancelled)"
                continue
            search_pattern = pat
            matches = _collect_matches(pat, display_lines)
            match_cursor = 0
            if not matches:
                transient = "(pattern not found)"
            else:
                transient = f"({len(matches)} match(es))"
                offset = _scroll_to_line(offset, matches[0][0], total, page_h)
            continue
        if action == "next_match":
            if not search_pattern:
                transient = "(no search; press / first)"
                continue
            if not matches:
                transient = "(no matches)"
                continue
            match_cursor = (match_cursor + 1) % len(matches)
            offset = _scroll_to_line(offset, matches[match_cursor][0], total, page_h)
            continue
        if action == "prev_match":
            if not search_pattern:
                transient = "(no search; press / first)"
                continue
            if not matches:
                transient = "(no matches)"
                continue
            match_cursor = (match_cursor - 1) % len(matches)
            offset = _scroll_to_line(offset, matches[match_cursor][0], total, page_h)
            continue
