"""
Persist Neo4j write operations when the graph is unavailable; drain when connectivity returns.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from config import settings

logger = logging.getLogger(__name__)


async def enqueue_store_entities(
    document_id: str,
    user_id: Optional[str],
    entities: List[Any],
    chunks: Optional[List[Any]],
) -> None:
    """Insert a backlog row for a failed or skipped store_entities."""
    try:
        from services.database_manager.database_helpers import execute

        payload: Dict[str, Any] = {
            "entities": [e.model_dump(mode="json") for e in entities],
            "chunks": [c.model_dump(mode="json") for c in chunks] if chunks else [],
        }
        await execute(
            """
            INSERT INTO kg_write_backlog (op, document_id, user_id, payload, updated_at)
            VALUES ('store_entities', $1, $2, $3::jsonb, now())
            """,
            document_id,
            user_id or "",
            json.dumps(payload),
            rls_context=None,
        )
    except Exception as e:
        logger.warning("kg_write_backlog enqueue store_entities failed: %s", e)


async def enqueue_delete_document(document_id: str, user_id: Optional[str]) -> None:
    try:
        from services.database_manager.database_helpers import execute

        await execute(
            """
            INSERT INTO kg_write_backlog (op, document_id, user_id, payload, updated_at)
            VALUES ('delete_document', $1, $2, '{}'::jsonb, now())
            """,
            document_id,
            user_id or "",
            rls_context=None,
        )
    except Exception as e:
        logger.warning("kg_write_backlog enqueue delete_document failed: %s", e)


async def backlog_count() -> int:
    try:
        from services.database_manager.database_helpers import fetch_value

        n = await fetch_value(
            "SELECT count(*)::bigint FROM kg_write_backlog WHERE attempts < $1",
            settings.KG_BACKLOG_MAX_ATTEMPTS,
            rls_context=None,
        )
        return int(n or 0)
    except Exception:
        return 0


async def purge_expired_rows() -> None:
    try:
        from services.database_manager.database_helpers import execute

        max_age = max(1, int(settings.KG_BACKLOG_PURGE_DAYS))
        await execute(
            """
            DELETE FROM kg_write_backlog
            WHERE created_at < (NOW() AT TIME ZONE 'utc') - ($1::int * INTERVAL '1 day')
            """,
            max_age,
            rls_context=None,
        )
    except Exception as e:
        logger.debug("kg_write_backlog purge skipped: %s", e)


async def fetch_backlog_batch(limit: int) -> List[Dict[str, Any]]:
    from services.database_manager.database_helpers import fetch_all

    return await fetch_all(
        """
        SELECT id, op, document_id, user_id, payload, attempts
        FROM kg_write_backlog
        WHERE attempts < $1
        ORDER BY id ASC
        LIMIT $2
        """,
        settings.KG_BACKLOG_MAX_ATTEMPTS,
        limit,
        rls_context=None,
    )


async def delete_backlog_row(row_id: int) -> None:
    from services.database_manager.database_helpers import execute

    await execute("DELETE FROM kg_write_backlog WHERE id = $1", row_id, rls_context=None)


async def mark_backlog_failure(row_id: int, attempts: int, err: str) -> None:
    from services.database_manager.database_helpers import execute

    await execute(
        """
        UPDATE kg_write_backlog
        SET attempts = $2, last_error = $3, updated_at = now()
        WHERE id = $1
        """,
        row_id,
        attempts + 1,
        (err or "")[:2000],
        rls_context=None,
    )


async def drain_backlog_batch(kg_service: Any, batch_size: int = 100) -> int:
    """
    Replay backlog rows against an already-connected kg_service.
    Returns number of rows successfully removed.
    """
    from models.api_models import Chunk, Entity

    rows = await fetch_backlog_batch(batch_size)
    done = 0
    for row in rows:
        rid = row["id"]
        op = row["op"]
        doc_id = row["document_id"]
        payload = row["payload"] or {}
        attempts = int(row.get("attempts") or 0)
        try:
            if op == "store_entities":
                raw_e = payload.get("entities") or []
                raw_c = payload.get("chunks") or []
                entities = [Entity(**e) for e in raw_e]
                chunks = [Chunk(**c) for c in raw_c] if raw_c else None
                ok = await kg_service.replay_store_entities(doc_id, entities, chunks)
                if ok:
                    await delete_backlog_row(rid)
                    done += 1
                else:
                    await mark_backlog_failure(rid, attempts, "replay_store_entities returned false")
            elif op == "delete_document":
                ok = await kg_service.replay_delete_document(doc_id)
                if ok:
                    await delete_backlog_row(rid)
                    done += 1
                else:
                    await mark_backlog_failure(rid, attempts, "replay_delete_document returned false")
            else:
                await mark_backlog_failure(rid, attempts, f"unknown op {op}")
        except Exception as e:
            await mark_backlog_failure(rid, attempts, str(e))
    return done
