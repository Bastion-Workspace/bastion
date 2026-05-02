"""OPDS catalog browser and EPUB chapter reader for the BBS terminal."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

from epub_reader import extract_epub_chapters, opensearch_apply_template
from rendering.reader import view_text_document
from rendering.text import format_header_context, section_header, word_wrap

if TYPE_CHECKING:
    from session import BBSSession


def _parse_menu_number(raw: str) -> Optional[int]:
    digits = "".join(ch for ch in (raw or "").strip() if ch.isdigit())
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def _normalize_remote_percentage(val: Any) -> float:
    if isinstance(val, (int, float)):
        p = float(val)
        if p > 1.0:
            p = p / 100.0
        return max(0.0, min(1.0, p))
    return 0.0


def _chapter_from_percentage(chapters: List[Tuple[str, str]], pct: float) -> int:
    char_lens = [max(1, len(t)) for _, t in chapters]
    total = sum(char_lens)
    target = pct * total
    acc = 0
    for i, ln in enumerate(char_lens):
        if acc + ln >= target:
            return i
        acc += ln
    return max(0, len(chapters) - 1)


def _percentage_for_chapter(chapters: List[Tuple[str, str]], chapter_idx: int) -> float:
    n = len(chapters)
    if n <= 0:
        return 0.0
    char_lens = [max(1, len(t)) for _, t in chapters]
    total = sum(char_lens)
    if total <= 0:
        return min(1.0, (chapter_idx + 0.5) / n)
    acc = sum(char_lens[:chapter_idx])
    mid = acc + char_lens[chapter_idx] * 0.35
    return max(0.0, min(1.0, mid / total))


def _resolve_href(base: str, href: str) -> str:
    if not href:
        return base
    return urljoin(base if base.endswith("/") else base + "/", href)


def _feed_next_prev_urls(feed: Dict[str, Any], fetched_base: str) -> Tuple[Optional[str], Optional[str]]:
    next_u: Optional[str] = None
    prev_u: Optional[str] = None
    for lk in feed.get("feed_links") or []:
        rel = (lk.get("rel") or "").lower()
        href = lk.get("href")
        if not href:
            continue
        full = _resolve_href(fetched_base, href)
        if "next" in rel and "preview" not in rel:
            next_u = full
        if "previous" in rel or rel.endswith("/prev") or "/prev" in rel:
            prev_u = full
        elif "prev" in rel and "preview" not in rel:
            prev_u = full
    return next_u, prev_u


def _partition_entries(
    entries: List[Dict[str, Any]], fetched_base: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    nav_out: List[Dict[str, Any]] = []
    book_out: List[Dict[str, Any]] = []
    for e in entries:
        title = (e.get("title") or "?").strip()
        acq = e.get("acquisition_href")
        navs = e.get("navigation_links") or []
        if acq:
            book_out.append({**e, "_resolved_acq": _resolve_href(fetched_base, acq), "_display_title": title})
            continue
        if navs:
            href = navs[0].get("href")
            if href:
                nav_out.append(
                    {
                        **e,
                        "_resolved_nav": _resolve_href(fetched_base, href),
                        "_display_title": title,
                    }
                )
    return nav_out, book_out


async def ebooks_hub(session: "BBSSession") -> None:
    await session.clear_screen()
    client = session.client
    jwt = session.jwt_token
    t = session.theme
    hdr = section_header(
        "E-Books (OPDS)",
        session.term_width - 2,
        t,
        context=format_header_context(session.display_name or session.username),
    )
    await session._write(f"\r\n{hdr}\r\n")

    while True:
        settings = await client.get_ebooks_settings(jwt)
        if settings.get("error"):
            await session._write(f"\r\nCould not load ebook settings: {settings['error']}\r\n[B]ack: ")
            if (await session.read_menu_choice()).strip().lower() in ("b", "back", "q"):
                return
            continue

        catalogs = settings.get("catalogs") or []
        if not catalogs:
            await session._write(
                "\r\nNo OPDS catalogs configured.\r\n"
                "Add catalogs in the web app: Settings, Ebooks / OPDS.\r\n\r\n"
                "[B]ack: "
            )
            if (await session.read_menu_choice()).strip().lower() in ("b", "back", "q"):
                return
            continue

        kosync_ok = bool((settings.get("kosync") or {}).get("configured"))

        await session._write(f"\r\nOPDS catalogs (KoSync: {'on' if kosync_ok else 'off'})\r\n\r\n")
        for i, c in enumerate(catalogs, 1):
            name = (c.get("title") or c.get("id") or "?")[:52]
            await session._write(f"  {i}) {name}\r\n")
        await session._write(
            f"\r\n{t.fg_bright_green}[#]{t.reset} Open catalog  "
            f"{t.fg_bright_green}[B]{t.reset}ack: "
        )
        raw = (await session.read_menu_choice(allow_digit_suffix=True)).strip()
        low = raw.lower()
        if low in ("b", "back", "q"):
            return
        n = _parse_menu_number(raw)
        if n is None or n < 1 or n > len(catalogs):
            await session._write("Enter a catalog number or B.\r\n")
            continue
        cat = catalogs[n - 1]
        catalog_id = str(cat.get("id") or "")
        root_url = (cat.get("root_url") or "").strip()
        cat_title = (cat.get("title") or "Catalog")[:40]
        if not catalog_id or not root_url:
            await session._write("Invalid catalog entry.\r\n")
            continue
        stack: List[Dict[str, str]] = [{"name": cat_title, "url": root_url}]
        await _browse_opds_feed(session, catalog_id, stack)


async def _browse_opds_feed(session: "BBSSession", catalog_id: str, stack: List[Dict[str, str]]) -> None:
    client = session.client
    jwt = session.jwt_token
    t = session.theme
    while True:
        url = stack[-1]["url"]
        res = await client.opds_fetch_feed(jwt, catalog_id, url)
        if res.get("error"):
            await session._write(f"\r\nOPDS error: {res['error']}\r\n\r\n[B]ack: ")
            if (await session.read_menu_choice()).strip().lower() in ("b", "back", "q"):
                if len(stack) > 1:
                    stack.pop()
                    continue
                return
            if len(stack) > 1:
                stack.pop()
            continue
        feed = res.get("feed") or {}
        fetched = res.get("fetched_url") or url
        entries = feed.get("entries") or []
        nav_items, book_items = _partition_entries(entries, fetched)
        next_u, prev_u = _feed_next_prev_urls(feed, fetched)
        search_template = feed.get("search_template")
        crumb = " > ".join((s.get("name") or "?")[:36] for s in stack)
        feed_title = (feed.get("feed_title") or "").strip()
        await session.clear_screen()
        subhdr = section_header(
            "OPDS",
            session.term_width - 2,
            t,
            context=format_header_context(session.display_name or session.username),
        )
        await session._write(f"\r\n{subhdr}\r\n")
        if feed_title:
            ft = feed_title[: max(1, session.term_width - 2)]
            await session._write(f"{t.dim}{ft}{t.reset}\r\n")
        cb = crumb[: max(1, session.term_width - 2)]
        await session._write(f"{t.dim}{cb}{t.reset}\r\n\r\n")
        targets: List[Dict[str, Any]] = []
        lines: List[str] = []
        idx = 0
        for it in nav_items:
            idx += 1
            targets.append({"kind": "nav", "url": it["_resolved_nav"], "title": it["_display_title"]})
            lines.append(f"  {idx}) [nav]  {(it['_display_title'] or '?')[:60]}")
        for it in book_items:
            idx += 1
            targets.append({"kind": "book", "entry": it})
            lines.append(f"  {idx}) [book] {(it['_display_title'] or '?')[:58]}")
        if not lines:
            await session._write("This feed is empty.\r\n")
        cap = max(1, session.term_height - 14)
        for ln in lines[:cap]:
            for wln in word_wrap(ln, session.term_width - 2):
                await session._write(wln + "\r\n")
        parts = [f"\r\n{t.fg_bright_green}[#]{t.reset} Open  "]
        if search_template:
            parts.append(f"{t.fg_bright_green}[S]{t.reset}earch  ")
        if next_u:
            parts.append(f"{t.fg_bright_green}[N]{t.reset}ext page  ")
        if prev_u:
            parts.append(f"{t.fg_bright_green}[P]{t.reset}rev page  ")
        parts.append(f"{t.fg_bright_green}[B]{t.reset}ack  ")
        await session._write("".join(parts))
        choice_raw = (await session.read_menu_choice(allow_digit_suffix=True)).strip()
        low = choice_raw.lower()
        if low in ("b", "back", "q"):
            if len(stack) > 1:
                stack.pop()
                continue
            return
        if low in ("s", "search") and search_template:
            await session._write(f"\r\n{t.dim}Search (empty=cancel):{t.reset} ")
            q = (await session.read_line(history_tag="opds_search")).strip()
            if q:
                su = opensearch_apply_template(search_template, q)
                stack.append({"name": f"Search: {q[:30]}", "url": su})
            continue
        if low in ("n", "next") and next_u:
            stack.append({"name": "Next page", "url": next_u})
            continue
        if low in ("p", "prev", "previous") and prev_u:
            stack.append({"name": "Prev page", "url": prev_u})
            continue
        n = _parse_menu_number(choice_raw)
        if n is None or n < 1 or n > len(targets):
            await session._write("Enter a valid number or a menu key.\r\nPress Enter...")
            await session.read_line()
            continue
        sel = targets[n - 1]
        if sel["kind"] == "nav":
            nm = (sel.get("title") or "Feed")[:44]
            stack.append({"name": nm, "url": sel["url"]})
            continue
        await _show_book_detail(session, catalog_id, sel["entry"], fetched)


async def _show_book_detail(
    session: "BBSSession",
    catalog_id: str,
    entry: Dict[str, Any],
    fetched_base: str,
) -> None:
    t = session.theme
    title = entry.get("_display_title") or entry.get("title") or "?"
    acq = entry.get("_resolved_acq") or _resolve_href(fetched_base, entry.get("acquisition_href") or "")
    await session.clear_screen()
    hdr = section_header(
        "Book",
        session.term_width - 2,
        t,
        context=format_header_context(session.display_name or session.username),
    )
    await session._write(f"\r\n{hdr}\r\n\r\n")
    meta_lines = [
        f"Title:    {title}",
        f"Author:   {(entry.get('author') or '-')[:70]}",
        f"Language: {entry.get('language') or '-'}",
        f"Issued:   {entry.get('issued') or '-'}",
        f"Updated:  {entry.get('updated') or '-'}",
    ]
    for ln in meta_lines:
        for w in word_wrap(ln, session.term_width - 2):
            await session._write(w + "\r\n")
    summary = (entry.get("summary") or "").strip()
    if summary:
        await session._write("\r\nSummary:\r\n")
        sm = summary[:8000]
        for w in word_wrap(sm, session.term_width - 2):
            await session._write(w + "\r\n")
    await session._write(
        f"\r\n{t.fg_bright_green}[R]{t.reset} Read EPUB  "
        f"{t.fg_bright_green}[B]{t.reset}ack: "
    )
    while True:
        raw = (await session.read_menu_choice()).strip().lower()
        if raw in ("b", "back", "q"):
            return
        if raw in ("r", "read"):
            await _read_epub(session, catalog_id, acq, book_title=title)
            return
        await session._write("Press R to read or B to go back.\r\n")


async def _read_epub(session: "BBSSession", catalog_id: str, acquisition_url: str, book_title: str) -> None:
    client = session.client
    jwt = session.jwt_token
    t = session.theme
    await session.clear_screen()
    await session._write(f"{t.dim}Downloading EPUB...{t.reset}\r\n")
    blob = await client.opds_fetch_binary(jwt, catalog_id, acquisition_url)
    if not blob:
        await session._write("\r\nDownload failed (empty or denied).\r\nPress Enter...")
        await session.read_line()
        return
    digest, chapters = extract_epub_chapters(blob)
    if len(chapters) == 1 and chapters[0][0].lower() == "error":
        await session._write(f"\r\n{chapters[0][1]}\r\nPress Enter...")
        await session.read_line()
        return
    settings = await client.get_ebooks_settings(jwt)
    kosync_ok = bool((settings.get("kosync") or {}).get("configured")) and not settings.get("error")
    start_ch = 0
    if kosync_ok:
        remote = await client.kosync_get_progress(jwt, digest)
        pct = _normalize_remote_percentage(remote.get("percentage") if isinstance(remote, dict) else 0)
        if pct > 0.02:
            pct_i = int(pct * 100)
            await session._write(
                f"\r\nKoSync saved position (~{pct_i}%).\r\n"
                f"{t.fg_bright_green}[Y]{t.reset}es resume  "
                f"{t.fg_bright_green}[N]{t.reset}o start from beginning: "
            )
            ans = (await session.read_menu_choice()).strip().lower()
            if ans in ("y", "yes"):
                start_ch = _chapter_from_percentage(chapters, pct)
    if digest in session.ebook_positions:
        saved = session.ebook_positions[digest]
        if 0 <= saved < len(chapters):
            await session._write(
                f"\r\nResume in this session at chapter {saved + 1}? "
                f"{t.fg_bright_green}[Y]{t.reset}/{t.fg_bright_green}[N]{t.reset}: "
            )
            ans2 = (await session.read_menu_choice()).strip().lower()
            if ans2 in ("y", "yes"):
                start_ch = saved
    chapter_idx = max(0, min(len(chapters) - 1, start_ch))
    while 0 <= chapter_idx < len(chapters):
        ch_title, body = chapters[chapter_idx]
        sub = f"Chapter {chapter_idx + 1} of {len(chapters)} - {ch_title[:50]}"
        truncated = len(body) > 80000
        body_use = body[:80000] if truncated else body
        await view_text_document(
            session,
            body_use,
            title=book_title[:50],
            subtitle=sub,
            truncated=truncated,
        )
        await session.clear_screen()
        sid = session.session_id[:16]
        await session._write(
            f"\r\n{book_title[:60]} - end of section {chapter_idx + 1}/{len(chapters)}\r\n"
            f"{t.fg_bright_green}[N]{t.reset}ext chapter  "
            f"{t.fg_bright_green}[P]{t.reset}rev chapter  "
            f"{t.fg_bright_green}[J]{t.reset}ump  "
            f"{t.fg_bright_green}[X]{t.reset}it book\r\n\r\nChoice: "
        )
        c2 = (await session.read_menu_choice()).strip().lower()
        if c2 in ("x", "exit", "q", "quit", "b", "back"):
            session.ebook_positions[digest] = chapter_idx
            if kosync_ok:
                pct_done = _percentage_for_chapter(chapters, chapter_idx)
                await client.kosync_put_progress(
                    jwt,
                    {
                        "document": digest,
                        "progress": f"bbs:ch={chapter_idx}",
                        "percentage": pct_done,
                        "device": "BastionBBS",
                        "device_id": f"bbs-{sid}",
                    },
                )
            return
        if c2 in ("n", "next") and chapter_idx + 1 < len(chapters):
            chapter_idx += 1
            if kosync_ok:
                await client.kosync_put_progress(
                    jwt,
                    {
                        "document": digest,
                        "progress": f"bbs:ch={chapter_idx}",
                        "percentage": _percentage_for_chapter(chapters, chapter_idx),
                        "device": "BastionBBS",
                        "device_id": f"bbs-{sid}",
                    },
                )
            continue
        if c2 in ("p", "prev", "previous") and chapter_idx > 0:
            chapter_idx -= 1
            continue
        if c2 in ("j", "jump"):
            await session._write(f"\r\nChapter 1-{len(chapters)} (0=cancel): ")
            raw_j = (await session.read_line()).strip()
            nj = _parse_menu_number(raw_j)
            if nj is not None and 1 <= nj <= len(chapters):
                chapter_idx = nj - 1
            continue
        await session._write("Unknown key.\r\n")
