"""User document pins: PostgreSQL-backed list for dashboard and future document UI."""

from __future__ import annotations

import logging
from typing import Any, List

from models.document_pin_models import (
    DocumentPinCreateRequest,
    DocumentPinItem,
    DocumentPinsListResponse,
    DocumentPinReorderRequest,
)
from services.database_manager.database_helpers import execute_transaction, fetch_all, fetch_one

logger = logging.getLogger(__name__)

_PREVIEW_LEN = 400


async def _document_owned_by_user(document_id: str, user_id: str) -> bool:
    row = await fetch_one(
        """
        SELECT 1 FROM document_metadata
        WHERE document_id = $1 AND user_id = $2
        """,
        document_id,
        user_id,
    )
    return row is not None


def _row_to_item(row: dict, include_preview: bool) -> DocumentPinItem:
    preview = None
    if include_preview and row.get("content_preview"):
        raw = row["content_preview"]
        if isinstance(raw, str) and len(raw) > _PREVIEW_LEN:
            preview = raw[:_PREVIEW_LEN] + "…"
        elif isinstance(raw, str):
            preview = raw
    return DocumentPinItem(
        pin_id=str(row["pin_id"]),
        document_id=row["document_id"],
        label=row.get("label"),
        sort_order=int(row["sort_order"]),
        title=row.get("title"),
        filename=row.get("filename"),
        content_preview=preview,
    )


async def list_pins(user_id: str, include_preview: bool = False) -> DocumentPinsListResponse:
    if include_preview:
        rows = await fetch_all(
            """
            SELECT p.pin_id, p.document_id, p.label, p.sort_order,
                   d.title, d.filename,
                   LEFT(COALESCE(d.description, ''), $2) AS content_preview
            FROM user_document_pins p
            JOIN document_metadata d ON d.document_id = p.document_id
            WHERE p.user_id = $1
            ORDER BY p.sort_order ASC, p.pinned_at ASC
            """,
            user_id,
            _PREVIEW_LEN,
        )
    else:
        rows = await fetch_all(
            """
            SELECT p.pin_id, p.document_id, p.label, p.sort_order,
                   d.title, d.filename,
                   NULL::text AS content_preview
            FROM user_document_pins p
            JOIN document_metadata d ON d.document_id = p.document_id
            WHERE p.user_id = $1
            ORDER BY p.sort_order ASC, p.pinned_at ASC
            """,
            user_id,
        )
    return DocumentPinsListResponse(
        pins=[_row_to_item(dict(r), include_preview) for r in rows]
    )


async def add_pin(user_id: str, body: DocumentPinCreateRequest) -> DocumentPinItem:
    if not await _document_owned_by_user(body.document_id, user_id):
        raise LookupError("Document not found or not owned by user")

    max_ord = await fetch_one(
        """
        SELECT COALESCE(MAX(sort_order), -1)::int AS m
        FROM user_document_pins WHERE user_id = $1
        """,
        user_id,
    )
    next_ord = int(max_ord["m"]) + 1 if max_ord else 0

    row = await fetch_one(
        """
        INSERT INTO user_document_pins (user_id, document_id, label, sort_order)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (user_id, document_id)
        DO UPDATE SET
            label = COALESCE(EXCLUDED.label, user_document_pins.label)
        RETURNING pin_id, document_id, label, sort_order
        """,
        user_id,
        body.document_id,
        body.label,
        next_ord,
    )
    if not row:
        raise RuntimeError("Failed to insert pin")

    meta = await fetch_one(
        """
        SELECT title, filename FROM document_metadata
        WHERE document_id = $1 AND user_id = $2
        """,
        body.document_id,
        user_id,
    )
    d = dict(row)
    if meta:
        d["title"] = meta.get("title")
        d["filename"] = meta.get("filename")
    else:
        d["title"] = None
        d["filename"] = None
    d["content_preview"] = None
    return _row_to_item(d, False)


async def delete_pin(user_id: str, pin_id: str) -> None:
    row = await fetch_one(
        """
        DELETE FROM user_document_pins
        WHERE pin_id = $1::uuid AND user_id = $2
        RETURNING pin_id
        """,
        pin_id,
        user_id,
    )
    if not row:
        raise LookupError("Pin not found")


async def reorder_pins(user_id: str, body: DocumentPinReorderRequest) -> DocumentPinsListResponse:
    ids: List[str] = body.pin_ids
    rows = await fetch_all(
        """
        SELECT pin_id::text AS pin_id FROM user_document_pins
        WHERE user_id = $1 AND pin_id::text = ANY($2::text[])
        """,
        user_id,
        ids,
    )
    found = {r["pin_id"] for r in rows}
    if len(found) != len(ids) or len(found) != len(set(ids)):
        raise ValueError("One or more pin ids are invalid or not owned by user")

    async def _op(conn: Any) -> None:
        for i, pid in enumerate(ids):
            await conn.execute(
                """
                UPDATE user_document_pins SET sort_order = $1
                WHERE pin_id = $2::uuid AND user_id = $3
                """,
                i,
                pid,
                user_id,
            )

    await execute_transaction([_op])
    return await list_pins(user_id, include_preview=False)
