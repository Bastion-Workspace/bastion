"""
Interactive refile: pick target from discover-targets, call POST /api/org/refile.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from rendering.text import format_header_context, section_header

from areas.org_paths import relative_user_library_path

if TYPE_CHECKING:
    from session import BBSSession


async def run_refile_wizard(session: "BBSSession", todo_item: Dict[str, Any]) -> None:
    """
    todo_item: needs file_path, line_number (0-based), filename, heading (optional).
    """
    jwt = session.jwt_token
    client = session.client
    t = session.theme
    username = (session.username or "").strip() or "user"

    abs_path = str(todo_item.get("file_path") or "")
    filename = str(todo_item.get("filename") or "")
    line_0 = todo_item.get("line_number")
    if not isinstance(line_0, int):
        try:
            line_0 = int(line_0)
        except (TypeError, ValueError):
            line_0 = 0
    source_line = line_0 + 1
    source_file = relative_user_library_path(username, abs_path, filename)
    if not source_file:
        await session._write("Cannot determine source file path for refile.\r\n")
        await session._write("Press Enter... ")
        await session.read_line()
        return

    hdr = section_header(
        "Refile",
        session.term_width - 2,
        t,
        context=format_header_context(session.display_name or session.username),
    )
    await session._write(f"\r\n{hdr}\r\n")
    await session._write(f"Source: {source_file} line {source_line}\r\n")
    await session._write("Loading targets...\r\n")

    data = await client.discover_refile_targets(jwt)
    if not data.get("success"):
        err = str(data.get("error") or "Failed to load targets")[:220]
        await session._write(f"Error: {err}\r\nPress Enter... ")
        await session.read_line()
        return

    targets_full: List[Dict[str, Any]] = list(data.get("targets") or [])
    if not targets_full:
        await session._write("No refile targets found.\r\nPress Enter... ")
        await session.read_line()
        return

    await session._write("Filter substring (empty=show all, 'q' cancel): ")
    filt = (await session.read_line()).strip().lower()
    if filt == "q":
        return

    def _apply_filter(items: List[Dict[str, Any]], q: str) -> List[Dict[str, Any]]:
        if not q:
            return list(items)
        fl = q.lower()
        return [
            x
            for x in items
            if fl in str(x.get("display_name") or "").lower()
            or fl in str(x.get("filename") or "").lower()
            or fl in str(x.get("file") or "").lower()
        ]

    targets = _apply_filter(targets_full, filt)
    if not targets:
        await session._write("No targets match filter.\r\nPress Enter... ")
        await session.read_line()
        return

    page = 0
    page_size = max(8, min(18, session.term_height - 8))
    while True:
        await session.clear_screen()
        await session._write(f"{hdr}\r\n")
        await session._write(f"Targets {len(targets)}  Page {page + 1}\r\n\r\n")
        start = page * page_size
        chunk = targets[start : start + page_size]
        for i, tg in enumerate(chunk):
            global_idx = start + i + 1
            disp = str(tg.get("display_name") or "?")[: session.term_width - 8]
            await session._write(f"  {global_idx}) {disp}\r\n")
        await session._write(
            "\r\n# = pick target  N = next page  P = prev  F = filter again  B = cancel\r\nChoice: "
        )
        raw = (
            await session.read_menu_choice(allow_digit_suffix=True)
        ).strip().lower()
        if raw in ("b", "back", "q", ""):
            return
        if raw == "n":
            if start + page_size < len(targets):
                page += 1
            continue
        if raw == "p":
            if page > 0:
                page -= 1
            continue
        if raw == "f":
            await session._write("Filter: ")
            filt2 = (await session.read_line()).strip().lower()
            if filt2 == "q":
                return
            targets = _apply_filter(targets_full, filt2)
            if not targets:
                await session._write("No targets. Press Enter...")
                await session.read_line()
                return
            page = 0
            continue
        if raw.isdigit():
            n = int(raw)
            if 1 <= n <= len(targets):
                sel = targets[n - 1]
                target_file = str(sel.get("file") or "")
                th = sel.get("heading_line")
                target_heading: Optional[int] = None
                if isinstance(th, int) and th > 0:
                    target_heading = th
                elif th is not None:
                    try:
                        thi = int(th)
                        if thi > 0:
                            target_heading = thi
                    except (TypeError, ValueError):
                        target_heading = None

                await session._write(
                    f"Refile to {target_file}"
                    f"{(' @ line ' + str(target_heading)) if target_heading else ''}? [y/N]: "
                )
                ok = (await session.read_menu_choice()).strip().lower()
                if ok not in ("y", "yes"):
                    return

                res = await client.refile_entry(
                    jwt,
                    source_file=source_file,
                    source_line=source_line,
                    target_file=target_file,
                    target_heading_line=target_heading,
                )
                if res.get("success"):
                    await session._write("Refile OK.\r\n")
                else:
                    await session._write(f"Refile failed: {str(res.get('error'))[:200]}\r\n")
                await session._write("Press Enter... ")
                await session.read_line()
                return
        await session._write("Unknown choice.\r\n")
