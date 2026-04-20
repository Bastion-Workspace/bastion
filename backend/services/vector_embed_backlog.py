"""
Persist vector sync operations when vector-service/Qdrant is unavailable; drain when healthy.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from config import settings

logger = logging.getLogger(__name__)


async def enqueue_sync_document_vectors(
    document_id: str,
    user_id: Optional[str],
    team_id: Optional[str],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Queue a full document vector upsert (chunks must already exist in document_chunks)."""
    try:
        from services.database_manager.database_helpers import execute

        payload: Dict[str, Any] = dict(extra or {})
        if team_id:
            payload["team_id"] = team_id
        await execute(
            """
            INSERT INTO vector_embed_backlog (op, document_id, user_id, payload, updated_at)
            VALUES ('sync_document_vectors', $1, $2, $3::jsonb, now())
            """,
            document_id,
            user_id or "",
            json.dumps(payload),
            rls_context=None,
        )
    except Exception as e:
        logger.warning("vector_embed_backlog enqueue sync_document_vectors failed: %s", e)


async def enqueue_delete_document_vectors(
    document_id: str,
    user_id: Optional[str],
    team_id: Optional[str] = None,
) -> None:
    """Queue Qdrant point deletion for a document."""
    try:
        from services.database_manager.database_helpers import execute

        payload: Dict[str, Any] = {}
        if team_id:
            payload["team_id"] = team_id
        await execute(
            """
            INSERT INTO vector_embed_backlog (op, document_id, user_id, payload, updated_at)
            VALUES ('delete_document_vectors', $1, $2, $3::jsonb, now())
            """,
            document_id,
            user_id or "",
            json.dumps(payload),
            rls_context=None,
        )
    except Exception as e:
        logger.warning("vector_embed_backlog enqueue delete_document_vectors failed: %s", e)


async def backlog_count() -> int:
    try:
        from services.database_manager.database_helpers import fetch_value

        n = await fetch_value(
            "SELECT count(*)::bigint FROM vector_embed_backlog WHERE attempts < $1",
            settings.VECTOR_EMBED_BACKLOG_MAX_ATTEMPTS,
            rls_context=None,
        )
        return int(n or 0)
    except Exception:
        return 0


async def purge_expired_rows() -> None:
    try:
        from services.database_manager.database_helpers import execute

        max_age = max(1, int(settings.VECTOR_EMBED_BACKLOG_PURGE_DAYS))
        await execute(
            """
            DELETE FROM vector_embed_backlog
            WHERE created_at < (NOW() AT TIME ZONE 'utc') - ($1::int * INTERVAL '1 day')
            """,
            max_age,
            rls_context=None,
        )
    except Exception as e:
        logger.debug("vector_embed_backlog purge skipped: %s", e)


async def fetch_backlog_batch(limit: int) -> List[Dict[str, Any]]:
    from services.database_manager.database_helpers import fetch_all

    return await fetch_all(
        """
        SELECT id, op, document_id, user_id, payload, attempts
        FROM vector_embed_backlog
        WHERE attempts < $1
        ORDER BY id ASC
        LIMIT $2
        """,
        settings.VECTOR_EMBED_BACKLOG_MAX_ATTEMPTS,
        limit,
        rls_context=None,
    )


async def delete_backlog_row(row_id: int) -> None:
    from services.database_manager.database_helpers import execute

    await execute("DELETE FROM vector_embed_backlog WHERE id = $1", row_id, rls_context=None)


async def mark_backlog_failure(row_id: int, attempts: int, err: str) -> None:
    from services.database_manager.database_helpers import execute

    await execute(
        """
        UPDATE vector_embed_backlog
        SET attempts = $2, last_error = $3, updated_at = now()
        WHERE id = $1
        """,
        row_id,
        attempts + 1,
        (err or "")[:2000],
        rls_context=None,
    )


async def drain_backlog_batch(embedding: Any, batch_size: int = 100) -> int:
    """Replay backlog rows. Returns number of rows successfully removed."""
    rows = await fetch_backlog_batch(batch_size)
    done = 0
    for row in rows:
        rid = row["id"]
        op = row["op"]
        doc_id = row["document_id"]
        uid = (row.get("user_id") or "") or None
        payload = row.get("payload") or {}
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                payload = {}
        attempts = int(row.get("attempts") or 0)
        try:
            if op == "sync_document_vectors":
                ok = await embedding.replay_sync_document_vectors(doc_id, uid, payload)
                if ok:
                    await delete_backlog_row(rid)
                    done += 1
                else:
                    await mark_backlog_failure(
                        rid, attempts, "replay_sync_document_vectors returned false"
                    )
            elif op == "delete_document_vectors":
                ok = await embedding.replay_delete_document_vectors(doc_id, uid, payload)
                if ok:
                    await delete_backlog_row(rid)
                    done += 1
                else:
                    await mark_backlog_failure(
                        rid, attempts, "replay_delete_document_vectors returned false"
                    )
            else:
                await mark_backlog_failure(rid, attempts, f"unknown op {op}")
        except Exception as e:
            await mark_backlog_failure(rid, attempts, str(e))
    return done
