"""
Documents: library root (My Documents, Global, Team) then folder drill-down via /api/folders/*.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from rendering.text import format_header_context, section_header
from rendering.tables import render_table

from areas.document_open import open_document_by_id

if TYPE_CHECKING:
    from session import BBSSession

# Virtual RSS folder — use main menu RSS, not file contents
_RSS_VIRTUAL = frozenset({"rss_feeds_virtual", "global_rss_feeds_virtual"})


def _breadcrumb(stack: List[Dict[str, str]]) -> str:
    if not stack:
        return "Library"
    return " > ".join((x.get("name") or "?")[:44] for x in stack)


def _doc_type_str(d: Dict[str, Any]) -> str:
    dt = d.get("doc_type")
    if dt is None:
        return ""
    if isinstance(dt, dict):
        return str(dt.get("value", dt.get("type", "")))[:10]
    return str(dt)[:10]


def _folder_row(f: Dict[str, Any]) -> Tuple[str, str, str]:
    fid = f.get("folder_id") or ""
    name = (f.get("name") or "?")[:48]
    cnt = f.get("document_count")
    sc = f.get("subfolder_count")
    extra = ""
    if isinstance(cnt, int) or isinstance(sc, int):
        extra = f" ({int(cnt or 0)} docs"
        if isinstance(sc, int):
            extra += f", {sc} subfolders"
        extra += ")"
    return fid, name, extra


async def documents_browser(session: "BBSSession") -> None:
    await session.clear_screen()
    hdr = section_header(
        "Files",
        session.term_width - 2,
        session.theme,
        context=format_header_context(session.display_name or session.username),
    )
    await session._write(f"\r\n{hdr}\r\n")
    client = session.client
    jwt = session.jwt_token
    t = session.theme
    stack: List[Dict[str, str]] = []

    async def write_menu_line() -> None:
        await session._write(
            f"{t.fg_bright_green}[#]{t.reset} Open folder/doc by number  "
            f"{t.fg_bright_green}[S]{t.reset}earch  "
            f"{t.fg_bright_green}[B]{t.reset}ack  "
            f"{t.fg_bright_green}[H]{t.reset}ome (library)\r\n"
        )

    while True:
        if not stack:
            tree = await client.get_folder_tree(jwt, collection_type="user", shallow=True)
            if tree.get("error"):
                await session._write(f"Error: {tree['error'][:200]}\r\n")
                return
            roots = tree.get("folders") or []
            if not roots:
                await session._write("\r\nNo folders (empty library).\r\n[B]ack: ")
                if (await session.read_line()).strip().lower() in ("b", "back", "q"):
                    return
                continue

            await session._write(f"\r\n=== Documents - {_breadcrumb(stack)} ===\r\n")
            await session._write("Open a collection:\r\n\r\n")
            for i, node in enumerate(roots, 1):
                if not isinstance(node, dict):
                    continue
                _, name, extra = _folder_row(node)
                await session._write(f"  {i}) {name}{extra}\r\n")
            await session._write("\r\n")
            await write_menu_line()
            await session._write("Choice: ")
            choice = (await session.read_line()).strip().lower()
            if choice in ("b", "back", "q"):
                return
            if choice in ("h", "home"):
                continue
            if choice.startswith("s") or choice == "search":
                await _search_flow(session, client, jwt)
                continue
            if not choice.isdigit():
                continue
            n = int(choice)
            if n < 1 or n > len(roots):
                await session._write("Invalid number.\r\n")
                continue
            node = roots[n - 1]
            if not isinstance(node, dict):
                continue
            fid = node.get("folder_id")
            name = node.get("name") or "?"
            if not fid:
                continue
            stack.append({"id": str(fid), "name": str(name)})
            continue

        cur = stack[-1]
        fid = cur["id"]

        if fid in _RSS_VIRTUAL:
            await session._write(
                "\r\nRSS feeds live under **RSS News** on the main BBS menu.\r\n"
                "Press Enter to go back.\r\n"
            )
            await session.read_line()
            stack.pop()
            continue

        data = await client.get_folder_contents(jwt, fid, limit=100, offset=0)
        if data.get("error"):
            await session._write(f"Error: {data['error'][:220]}\r\n")
            stack.pop()
            continue

        subfolders = data.get("subfolders") or []
        documents = data.get("documents") or []
        total_docs = int(data.get("total_documents") or len(documents))

        await session._write(f"\r\n=== {_breadcrumb(stack)} ===\r\n")
        if total_docs > len(documents):
            await session._write(
                f"(Showing {len(documents)} of {total_docs} documents here; narrow by opening subfolders.)\r\n"
            )

        entries: List[Dict[str, Any]] = []
        await session._write("\r\nFolders:\r\n")
        if not subfolders:
            await session._write("  (none)\r\n")
        else:
            for i, sf in enumerate(subfolders, 1):
                if not isinstance(sf, dict):
                    continue
                sfid, sname, extra = _folder_row(sf)
                entries.append({"kind": "folder", "id": sfid, "name": sname})
                await session._write(f"  {i}) [folder] {sname}{extra}\r\n")

        doc_offset = len(entries)
        await session._write("\r\nFiles:\r\n")
        if not documents:
            await session._write("  (none)\r\n")
        else:
            rows = []
            for j, d in enumerate(documents):
                if not isinstance(d, dict):
                    continue
                idx = doc_offset + j + 1
                fn = (d.get("filename") or d.get("title") or "?")[:34]
                dt = _doc_type_str(d)[:8]
                sz = d.get("file_size")
                sz_s = f"{int(sz) // 1024}KB" if isinstance(sz, int) else ""
                did = d.get("document_id", "")
                rows.append((str(idx), fn, dt, sz_s, did))
                entries.append({"kind": "doc", "id": did, "name": fn})
            if rows:
                tbl = render_table(
                    ("#", "File", "Type", "Size"),
                    [(r[0], r[1], r[2], r[3]) for r in rows],
                    col_widths=(4, 36, 8, 8),
                    max_width=session.term_width,
                )
                await session._write(f"{tbl}\r\n")

        await session._write("\r\n")
        await write_menu_line()
        await session._write("Choice: ")
        choice = (await session.read_line()).strip().lower()
        if choice in ("b", "back"):
            stack.pop()
            continue
        if choice in ("h", "home"):
            stack.clear()
            continue
        if choice.startswith("s") or choice == "search":
            await _search_flow(session, client, jwt)
            continue
        if not choice.isdigit():
            continue
        num = int(choice)
        if num < 1 or num > len(entries):
            await session._write("Invalid number.\r\n")
            continue
        item = entries[num - 1]
        if item["kind"] == "folder":
            stack.append({"id": str(item["id"]), "name": str(item.get("name") or "?")})
            continue
        await _open_document(session, str(item["id"]))


async def _search_flow(session: "BBSSession", client, jwt: str) -> None:
    await session._write("Search query: ")
    q = await session.read_line()
    if not q.strip():
        return
    sr = await client.search_documents(jwt, q.strip(), limit=15)
    if sr.get("error"):
        await session._write(f"Search error: {sr['error'][:200]}\r\n")
        return
    res = sr.get("results") or []
    if not res:
        await session._write("No results.\r\n")
        return
    for j, r in enumerate(res, 1):
        did = r.get("document_id", "")
        title = (r.get("title") or r.get("filename") or did)[:60]
        await session._write(f"{j}) {title}\r\n  id={did}\r\n")
    await session._write("Enter # to open (0 to cancel): ")
    pick = (await session.read_line()).strip()
    if not pick.isdigit() or int(pick) < 1 or int(pick) > len(res):
        return
    doc_id = res[int(pick) - 1].get("document_id")
    if doc_id:
        await _open_document(session, doc_id)


async def _open_document(session: "BBSSession", doc_id: str) -> None:
    await open_document_by_id(session, doc_id)
