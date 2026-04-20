"""
Org TODO list: filters, paging, toggle/update/archive/delete/refile, quick add.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from rendering.paginator import paginate_text
from rendering.text import format_header_context, normalize_for_telnet, section_header, word_wrap

from areas.document_open import open_document_by_id
from areas.org_refile_wizard import run_refile_wizard

if TYPE_CHECKING:
    from session import BBSSession

_DEFAULT_ACTIVE = ["TODO", "NEXT", "STARTED", "WAITING", "HOLD"]
_DEFAULT_DONE = ["DONE", "CANCELED", "CANCELLED"]


async def _fetch_state_vocab(session: "BBSSession") -> Tuple[List[str], List[str]]:
    jwt = session.jwt_token
    data = await session.client.get_org_todo_states(jwt)
    if not data.get("success"):
        return list(_DEFAULT_ACTIVE), list(_DEFAULT_DONE)
    st = data.get("states") or {}
    active = [str(x).upper() for x in (st.get("active") or []) if x]
    done = [str(x).upper() for x in (st.get("done") or []) if x]
    if not active:
        active = list(_DEFAULT_ACTIVE)
    if not done:
        done = list(_DEFAULT_DONE)
    return active, done


def _fmt_item_line(session: "BBSSession", idx: int, row: Dict[str, Any]) -> str:
    st = str(row.get("todo_state") or "?")[:10]
    pri = str(row.get("priority") or "").strip()
    pri_s = f"[{pri}]" if pri else ""
    title = normalize_for_telnet(str(row.get("heading") or "?"))[:120]
    fn = str(row.get("filename") or "")[:24]
    dl = row.get("deadline") or row.get("scheduled")
    dl_s = ""
    if dl:
        ds = str(dl).split()
        dl_s = ds[0] if ds else ""
    return f"{idx:3}) {st:10} {pri_s:4} {dl_s:12} {title}"


async def _resolve_doc_id(session: "BBSSession", row: Dict[str, Any]) -> Optional[str]:
    did = row.get("document_id")
    if did:
        return str(did)
    fn = str(row.get("filename") or "").strip()
    if not fn:
        return None
    lu = await session.client.lookup_org_document(session.jwt_token, fn)
    if lu.get("success") and lu.get("document"):
        return str(lu["document"].get("document_id") or "") or None
    return None


async def todos_browser(session: "BBSSession") -> None:
    jwt = session.jwt_token
    client = session.client
    t = session.theme

    active_states, done_states = await _fetch_state_vocab(session)
    filter_mode = "active"
    tag_sub = ""
    text_q = ""
    limit = 300

    async def load_rows() -> List[Dict[str, Any]]:
        states: Optional[List[str]] = None
        if filter_mode == "active":
            states = active_states
        elif filter_mode == "done":
            states = done_states
        else:
            states = list(dict.fromkeys(active_states + done_states))
        tags = [tag_sub.strip()] if tag_sub.strip() else None
        res = await client.list_todos(
            jwt,
            scope="all",
            states=states,
            tags=tags,
            query=text_q.strip(),
            limit=limit,
            include_archives=False,
        )
        if not res.get("success"):
            return []
        return list(res.get("results") or [])

    while True:
        await session.clear_screen()
        hdr = section_header(
            "Org TODOs",
            session.term_width - 2,
            t,
            context=format_header_context(session.display_name or session.username),
        )
        await session._write(f"\r\n{hdr}\r\n")
        await session._write(
            f"Filter: {filter_mode}  tag={tag_sub or '-'}  q={text_q[:40] or '-'}\r\n"
            f"[F]ilter mode  [T]ag  [/]query  [L]imit  [R]eload  [N]ew  [B]ack\r\n\r\n"
        )

        rows = await load_rows()
        if not rows:
            await session._write("(No matching TODOs)\r\n")
        else:
            total_note = f"{len(rows)} shown (limit {limit})\r\n"
            await session._write(total_note)
            lines: List[str] = []
            for i, row in enumerate(rows, start=1):
                raw = _fmt_item_line(session, i, row)
                for w in word_wrap(raw, max(20, session.term_width - 2)):
                    lines.append(w)
            body_h = max(5, session.term_height - 12)

            async def wl(s: str) -> None:
                await session._write(s)

            async def rm() -> str:
                return (await session.read_line()).strip().lower()

            await paginate_text(lines, body_h, t, wl, rm)

        await session._write(
            "\r\n# = act on item  F/T//L/R/N/B: "
        )
        choice = (await session.read_line()).strip().lower()
        if not choice:
            continue
        if choice in ("b", "back", "q"):
            return
        if choice == "f" or choice.startswith("filter"):
            await session._write("Mode [a]ctive [d]one [x]all : ")
            m = (await session.read_line()).strip().lower()
            if m.startswith("a"):
                filter_mode = "active"
            elif m.startswith("d"):
                filter_mode = "done"
            elif m.startswith("x") or m.startswith("all"):
                filter_mode = "all"
            continue
        if choice == "t" or choice.startswith("tag"):
            await session._write("Tag contains (empty=clear): ")
            tag_sub = (await session.read_line()).strip()
            continue
        if choice.startswith("/"):
            if len(choice) > 1:
                text_q = choice[1:].strip()
            else:
                await session._write("Search text (empty=clear): ")
                text_q = (await session.read_line()).strip()
            continue
        if choice == "l" or choice.startswith("limit"):
            await session._write("Limit (50-500, default 300): ")
            raw = (await session.read_line()).strip()
            if raw.isdigit():
                limit = max(50, min(500, int(raw)))
            continue
        if choice == "r" or choice.startswith("reload"):
            continue
        if choice == "n" or choice.startswith("new"):
            await session._write("New TODO title: ")
            title = (await session.read_line()).strip()
            if not title:
                continue
            await session._write("File path relative (empty=inbox): ")
            fp = (await session.read_line()).strip()
            body: Dict[str, Any] = {"text": title, "state": "TODO"}
            if fp:
                body["file_path"] = fp
            cr = await client.create_todo(jwt, body)
            if cr.get("success") is False:
                await session._write(f"Create failed: {str(cr.get('error'))[:200]}\r\n")
            else:
                await session._write("Created.\r\n")
            await session._write("Press Enter... ")
            await session.read_line()
            continue

        if choice.isdigit():
            n = int(choice)
            if 1 <= n <= len(rows):
                await _todo_item_actions(session, rows[n - 1])
            continue


async def _todo_item_actions(session: "BBSSession", row: Dict[str, Any]) -> None:
    jwt = session.jwt_token
    client = session.client
    t = session.theme
    fp = str(row.get("file_path") or "")
    ln = row.get("line_number")
    if not isinstance(ln, int):
        try:
            ln = int(ln)
        except (TypeError, ValueError):
            ln = 0
    heading = str(row.get("heading") or "")

    while True:
        await session.clear_screen()
        hdr = section_header(
            "TODO item",
            session.term_width - 2,
            t,
            context=format_header_context(session.display_name or session.username),
        )
        await session._write(f"\r\n{hdr}\r\n")
        await session._write(normalize_for_telnet(heading[:500]) + "\r\n\r\n")
        await session._write(
            "[X]oggle  [S]tate  [A]rchive  [D]elete  [M]ove(refile)  [E]dit file  [B]ack\r\nChoice: "
        )
        c = (await session.read_line()).strip().lower()
        if c in ("b", "back", ""):
            return
        if c.startswith("x") or c == "toggle":
            r = await client.toggle_todo(jwt, fp, ln, heading_text=heading or None)
            if r.get("success") is False:
                await session._write(f"Toggle failed: {str(r.get('error'))[:200]}\r\n")
            else:
                await session._write("OK.\r\n")
            await session._write("Press Enter... ")
            await session.read_line()
            return
        if c.startswith("s") or c == "state":
            await session._write("New state (e.g. TODO NEXT DONE): ")
            ns = (await session.read_line()).strip().upper()
            if ns:
                r = await client.update_todo(
                    jwt, fp, ln, {"new_state": ns}, heading_text=heading or None
                )
                if r.get("success") is False:
                    await session._write(f"Update failed: {str(r.get('error'))[:200]}\r\n")
                else:
                    await session._write("OK.\r\n")
            await session._write("Press Enter... ")
            await session.read_line()
            return
        if c.startswith("a") or c == "archive":
            r = await client.archive_todo(jwt, fp, ln)
            if r.get("success") is False:
                await session._write(f"Archive failed: {str(r.get('error'))[:200]}\r\n")
            else:
                await session._write("Archived.\r\n")
            await session._write("Press Enter... ")
            await session.read_line()
            return
        if c.startswith("d") or c == "delete":
            await session._write("Type 'yes' to delete this entry: ")
            conf = (await session.read_line()).strip().lower()
            if conf != "yes":
                continue
            r = await client.delete_todo(jwt, fp, ln, heading_text=heading or None)
            if r.get("success") is False:
                await session._write(f"Delete failed: {str(r.get('error'))[:200]}\r\n")
            else:
                await session._write("Deleted.\r\n")
            await session._write("Press Enter... ")
            await session.read_line()
            return
        if c.startswith("m") or c.startswith("refile"):
            await run_refile_wizard(session, row)
            return
        if c.startswith("e") or c == "edit":
            did = await _resolve_doc_id(session, row)
            if not did:
                await session._write("No document_id for this file; use Files.\r\n")
                await session._write("Press Enter... ")
                await session.read_line()
                return
            await open_document_by_id(session, did)
            return
