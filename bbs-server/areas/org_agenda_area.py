"""
Org agenda: scheduled, deadlines, appointments via GET /api/org/agenda.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from rendering.paginator import paginate_text
from rendering.text import format_header_context, normalize_for_telnet, section_header, word_wrap

from areas.document_open import open_document_by_id

if TYPE_CHECKING:
    from session import BBSSession


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


def _flatten_agenda(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    grouped = data.get("grouped_by_date")
    if isinstance(grouped, dict) and grouped:
        out: List[Dict[str, Any]] = []
        for dk in sorted(grouped.keys()):
            chunk = grouped.get(dk) or []
            if isinstance(chunk, list):
                out.extend(chunk)
        return out
    items = list(data.get("agenda_items") or [])
    items.sort(key=lambda x: str(x.get("agenda_date") or ""))
    return items


async def agenda_view(session: "BBSSession") -> None:
    jwt = session.jwt_token
    client = session.client
    t = session.theme
    days_ahead = 14

    while True:
        await session.clear_screen()
        hdr = section_header(
            "Org agenda",
            session.term_width - 2,
            t,
            context=format_header_context(session.display_name or session.username),
        )
        await session._write(f"\r\n{hdr}\r\n")
        await session._write(
            f"Days ahead ({days_ahead}). [C]hange  [R]eload  [B]ack\r\n\r\n"
        )

        data = await client.get_org_agenda(
            jwt,
            days_ahead=days_ahead,
            include_scheduled=True,
            include_deadlines=True,
            include_appointments=True,
        )
        if not data.get("success"):
            err = str(data.get("error") or "Agenda failed")[:220]
            await session._write(f"Error: {err}\r\nPress Enter... ")
            await session.read_line()
            return

        flat = _flatten_agenda(data)
        if not flat:
            await session._write("(No agenda items in range)\r\n")
        else:
            await session._write(f"Items: {len(flat)}\r\n\r\n")
            lines: List[str] = []
            for i, row in enumerate(flat, start=1):
                ad = str(row.get("agenda_date") or "")[:12]
                at = str(row.get("agenda_type") or "?")[:11]
                tm = str(row.get("time") or "").strip()[:12]
                title = normalize_for_telnet(str(row.get("heading") or "?"))[:100]
                fn = str(row.get("filename") or "")[:20]
                raw = f"{i:3}) {ad} {at:11} {tm:12} {fn:20} {title}"
                for w in word_wrap(raw, max(20, session.term_width - 2)):
                    lines.append(w)

            body_h = max(5, session.term_height - 10)

            async def wl(s: str) -> None:
                await session._write(s)

            await paginate_text(
                lines,
                body_h,
                t,
                wl,
                session.read_pager_key,
                after_line_input_drain=session.drain_stray_line_terminators,
            )

        await session._write("\r\n# = open org file  C/R/B: ")
        choice = (await session.read_line()).strip().lower()
        if choice in ("b", "back", "q"):
            return
        if choice == "c" or choice.startswith("change"):
            await session._write("Days ahead (1-90): ")
            raw = (await session.read_line()).strip()
            if raw.isdigit():
                days_ahead = max(1, min(90, int(raw)))
            continue
        if choice == "r" or choice.startswith("reload"):
            continue
        if choice.isdigit():
            n = int(choice)
            flat2 = _flatten_agenda(data)
            if 1 <= n <= len(flat2):
                row = flat2[n - 1]
                did = await _resolve_doc_id(session, row)
                if not did:
                    await session._write("Could not resolve document. Press Enter... ")
                    await session.read_line()
                    continue
                await open_document_by_id(session, did)
            continue
