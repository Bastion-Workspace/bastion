"""
Open a document by id: view or edit (shared by Files and Org desk).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from rendering.reader import view_text_document

if TYPE_CHECKING:
    from session import BBSSession

_BINARY_CONTENT_SOURCES = frozenset(
    {"pdf_binary", "image_binary", "docx_binary", "pptx_binary"}
)


async def open_document_by_id(session: "BBSSession", doc_id: str) -> None:
    """Load document content, then view or edit (.md/.txt/.org) like Files area."""
    content = await session.client.get_document_content_for_editor(session.jwt_token, doc_id)
    if content.get("error"):
        await session._write(f"Cannot load: {content['error'][:200]}\r\n")
        return
    meta = content.get("metadata") or {}
    src = content.get("content_source") or ""
    body = content.get("content") or ""
    if not isinstance(body, str):
        body = str(body)
    fn = str(meta.get("filename") or meta.get("title") or "")

    if src in _BINARY_CONTENT_SOURCES:
        await session._write(
            "(Binary file - not shown in terminal)\r\n"
            f"Title: {meta.get('title', '')}  File: {fn}\r\n"
        )
        return

    from editor.run import is_editable_filename, run_document_editor

    can_edit = is_editable_filename(fn)
    await session._write(f"\r\nFile: {fn[:70]}\r\n")
    if can_edit:
        await session._write("[V]iew formatted  [E]dit raw text  [B]ack: ")
    else:
        await session._write("[V]iew formatted  [B]ack (editing not available for this type): ")
    choice = (await session.read_line()).strip().lower()
    if choice in ("b", "back", "q", ""):
        return
    if choice.startswith("v") or choice == "view":
        await _view_text_document(session, body, meta)
        return
    if choice.startswith("e") or choice == "edit":
        if not can_edit:
            await session._write(
                "Only .md, .txt, and .org files can be edited on the BBS. Use the web app for other types.\r\n"
            )
            return
        title = (fn[:44] or doc_id[:12]) + " - edit"
        await run_document_editor(session, doc_id, body, title)
        return
    await session._write("Unknown choice.\r\n")


async def _view_text_document(session: "BBSSession", body: str, meta: Dict[str, Any]) -> None:
    if not body:
        await session._write("(Empty document)\r\n")
        await session._write("Press Enter to return... ")
        await session.read_line()
        return
    truncated = len(body) > 80000
    title = str(meta.get("filename") or meta.get("title") or "")
    subtitle_bits = []
    if meta.get("title") and str(meta.get("title")) != title:
        subtitle_bits.append(str(meta["title"])[:50])
    if meta.get("file_size") is not None:
        try:
            subtitle_bits.append(f"{int(meta['file_size']) // 1024} KB")
        except (TypeError, ValueError):
            pass
    subtitle = " | ".join(subtitle_bits)
    await view_text_document(
        session,
        body[:80000],
        title=title,
        subtitle=subtitle,
        truncated=truncated,
    )
