"""
RSS feed list and article reader (HTML stripped via html2text).
"""

from __future__ import annotations

import html2text
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from rendering.paginator import paginate_text
from rendering.text import format_header_context, section_header, word_wrap

if TYPE_CHECKING:
    from session import BBSSession

_h2t = html2text.HTML2Text()
_h2t.ignore_links = False
_h2t.body_width = 0

# Lines used after clear_screen before body paging: blank, header bar, blank, title, blank line.
_ARTICLE_VIEW_OVERHEAD_LINES = 7


def _article_body_page_height(term_height: int) -> int:
    """Body lines per pager step so header + chunk + --More-- fit in the terminal."""
    return max(5, term_height - _ARTICLE_VIEW_OVERHEAD_LINES)


def _rss_list_entry_lines(read: str, global_idx: int, title: str, term_width: int) -> List[str]:
    """One logical list item; long titles wrap with continuation indent aligned to the title column."""
    prefix = f" {read} {global_idx}) "
    cont = " " * len(prefix)
    raw = (title or "").strip() or "?"
    wrap_w = max(10, term_width - len(prefix))
    parts = word_wrap(raw, wrap_w)
    if not parts:
        return [prefix + "?"]
    out: List[str] = []
    for i, seg in enumerate(parts):
        if not seg.strip() and i > 0:
            continue
        out.append((prefix if i == 0 else cont) + seg)
    return out if out else [prefix + "?"]


def _rss_list_page_starts(
    articles: List[Dict[str, Any]], term_width: int, term_height: int
) -> List[int]:
    """Start index of each list screen so wrapped titles fit above the prompt."""
    overhead = 10
    budget = max(4, term_height - overhead)
    starts: List[int] = [0]
    n = len(articles)
    i = 0
    while i < n:
        used = 0
        j = i
        while j < n:
            read = "R" if articles[j].get("is_read") else " "
            need = len(_rss_list_entry_lines(read, j + 1, articles[j].get("title") or "?", term_width))
            if j > i and used + need > budget:
                break
            used += need
            j += 1
        if j == i:
            j = i + 1
        i = j
        if i < n:
            starts.append(i)
    return starts


def _parse_menu_number(raw: str) -> Optional[int]:
    """Parse a positive integer from line input; tolerate stray non-digit characters."""
    digits = "".join(ch for ch in (raw or "").strip() if ch.isdigit())
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


async def rss_reader(session: "BBSSession") -> None:
    await session.clear_screen()
    client = session.client
    jwt = session.jwt_token
    t = session.theme
    hdr = section_header(
        "RSS News",
        session.term_width - 2,
        t,
        context=format_header_context(session.display_name or session.username),
    )
    await session._write(f"\r\n{hdr}\r\n")

    while True:
        feeds = await client.list_rss_feeds(jwt)
        unread_map = await client.rss_unread_by_feed(jwt)
        if not feeds:
            await session._write("\r\nNo RSS feeds.\r\n[B]ack: ")
            if (await session.read_line()).strip().lower() in ("b", "back", "q"):
                return
            continue

        total_unread = sum(unread_map.values()) if unread_map else 0
        await session._write(f"\r\nRSS feeds (unread total: {total_unread})\r\n\r\n")
        for i, f in enumerate(feeds, 1):
            fid = f.get("feed_id") or f.get("id", "")
            name = (f.get("feed_name") or f.get("name") or "?")[:50]
            u = unread_map.get(str(fid), 0) if fid else 0
            await session._write(f"  {i}) {name}  (unread: {u})\r\n")
        await session._write(
            f"\r\n{t.fg_bright_green}[#]{t.reset} Open feed by number  "
            f"{t.fg_bright_green}[M]{t.reset}ark all read  "
            f"{t.fg_bright_green}[B]{t.reset}ack: "
        )
        choice_raw = (await session.read_line()).strip()
        choice_low = choice_raw.lower()
        if choice_low in ("b", "back", "q"):
            return
        if choice_low in ("m", "mark", "markall"):
            n = await client.rss_mark_all_read(jwt)
            await session._write(f"Marked {n} article(s) as read (all feeds).\r\n")
            continue
        n = _parse_menu_number(choice_raw)
        if n is None:
            await session._write("Enter a feed number, M, or B.\r\n")
            continue
        if n < 1 or n > len(feeds):
            await session._write(f"Invalid feed number (1-{len(feeds)}).\r\n")
            continue
        feed = feeds[n - 1]
        feed_id = feed.get("feed_id") or feed.get("id")
        if not feed_id:
            await session._write("Feed missing id.\r\n")
            continue
        await _feed_articles(session, feed_id)


async def _feed_articles(session: "BBSSession", feed_id: str) -> None:
    client = session.client
    jwt = session.jwt_token
    t = session.theme
    articles: List[Dict[str, Any]] = await client.get_rss_articles(jwt, feed_id, limit=120)
    page_index = 0

    while True:
        if not articles:
            await session.clear_screen()
            await session._write("\r\nNo articles.\r\n[B]ack: ")
            if (await session.read_line()).strip().lower() in ("b", "back", "q"):
                return
            return

        total = len(articles)
        page_starts = _rss_list_page_starts(articles, session.term_width, session.term_height)
        total_pages = len(page_starts)
        if page_index >= total_pages:
            page_index = max(0, total_pages - 1)
        start = page_starts[page_index]
        end = page_starts[page_index + 1] if page_index + 1 < total_pages else total
        page_slice = articles[start:end]
        page_num = page_index + 1 if total else 1

        await session.clear_screen()
        hdr_list = section_header(
            "Articles",
            session.term_width - 2,
            t,
            context=format_header_context(session.display_name or session.username),
        )
        await session._write(f"\r\n{hdr_list}\r\n")
        first = start + 1
        last = start + len(page_slice)
        await session._write(
            f"\r\nItems {first}-{last} of {total}  (page {page_num}/{total_pages})\r\n\r\n"
        )
        for j, a in enumerate(page_slice):
            global_idx = start + j + 1
            title = a.get("title") or "?"
            read = "R" if a.get("is_read") else " "
            for row in _rss_list_entry_lines(read, global_idx, title, session.term_width):
                await session._write(row + "\r\n")
        await session._write(
            f"\r\n{t.fg_bright_green}[#]{t.reset} Article number  "
            f"{t.fg_bright_green}[N]{t.reset}ext page  "
            f"{t.fg_bright_green}[P]{t.reset}rev page  "
            f"{t.fg_bright_green}[A]{t.reset}ll read  "
            f"{t.fg_bright_green}[B]{t.reset}ack: "
        )
        choice_raw = (await session.read_line()).strip()
        choice_low = choice_raw.lower()
        if choice_low in ("b", "back", "q"):
            return
        if choice_low in ("n", "next"):
            if page_index + 1 < total_pages:
                page_index += 1
            continue
        if choice_low in ("p", "prev", "previous"):
            page_index = max(0, page_index - 1)
            continue
        if choice_low in ("a", "all", "allread"):
            marked = await client.rss_mark_feed_all_read(jwt, feed_id)
            if marked:
                for a in articles:
                    a["is_read"] = True
            await session._write(f"Marked {marked} article(s) as read in this feed.\r\n")
            continue
        n = _parse_menu_number(choice_raw)
        if n is None:
            await session._write("Enter a number, N, P, A, or B.\r\n")
            continue
        if n < 1 or n > total:
            await session._write(f"Invalid article number (1-{total}).\r\n")
            continue
        art = articles[n - 1]
        aid = art.get("article_id")
        title = art.get("title") or ""
        body_html = art.get("full_content_html") or art.get("full_content") or art.get("description") or ""
        plain = _h2t.handle(str(body_html)) if body_html else ""
        if not plain.strip():
            plain = art.get("link") or "(no body)"
        await session.clear_screen()
        art_hdr = section_header(
            "Article",
            session.term_width - 2,
            t,
            context=format_header_context(session.display_name or session.username),
        )
        await session._write(f"\r\n{art_hdr}\r\n")
        safe_title = (title or "?")[: min(70, max(10, session.term_width - 4))]
        await session._write(f"{safe_title}\r\n\r\n")
        if aid:
            await client.mark_article_read(jwt, str(aid))
            art["is_read"] = True
        lines = word_wrap(plain[:50000], session.term_width - 2)
        page_h = _article_body_page_height(session.term_height)

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
        link = art.get("link") or ""
        if link:
            await session._write(f"\r\n  Link: {link}\r\n")
        await session._write(f"\r\n{t.dim}End of article.{t.reset}\r\n")
        await session._write("\r\nPress Enter to return to the list... ")
        await session.read_line()
