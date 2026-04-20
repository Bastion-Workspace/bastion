"""
Document Version Repository - Database operations for document_versions table.
Used by DocumentVersionService for snapshot, rollback, and history.
"""

import json
import logging
from typing import List, Optional, Dict, Any
from uuid import UUID

from ds_db.database_manager.database_helpers import execute, fetch_one, fetch_all, fetch_value

logger = logging.getLogger(__name__)


async def create(
    document_id: str,
    version_number: int,
    content_hash: str,
    file_size: int,
    storage_path: str,
    change_source: str,
    created_by: Optional[str] = None,
    change_summary: Optional[str] = None,
    parent_version_id: Optional[UUID] = None,
    operations_json: Optional[List[Dict[str, Any]]] = None,
    is_current: bool = True,
    rls_context: Optional[Dict[str, str]] = None,
) -> Optional[UUID]:
    """Insert a new document version and return its version_id."""
    from uuid import uuid4
    version_id = uuid4()
    operations_raw = json.dumps(operations_json) if operations_json is not None else None
    await execute(
        """
        INSERT INTO document_versions (
            version_id, document_id, version_number, content_hash, file_size,
            created_by, change_source, change_summary, parent_version,
            operations_json, storage_path, is_current
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11, $12)
        """,
        version_id,
        document_id,
        version_number,
        content_hash,
        file_size,
        created_by,
        change_source,
        change_summary,
        parent_version_id,
        operations_raw,
        storage_path,
        is_current,
        rls_context=rls_context,
    )
    return version_id


async def list_for_document(
    document_id: str,
    skip: int = 0,
    limit: int = 100,
    rls_context: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """List versions for a document, newest first."""
    rows = await fetch_all(
        """
        SELECT version_id, document_id, version_number, content_hash, file_size,
               created_at, created_by, change_source, change_summary,
               parent_version, operations_json, storage_path, is_current
        FROM document_versions
        WHERE document_id = $1
        ORDER BY version_number DESC
        OFFSET $2 LIMIT $3
        """,
        document_id,
        skip,
        limit,
        rls_context=rls_context,
    )
    return list(rows) if rows else []


async def get_by_id(
    version_id: UUID,
    rls_context: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    """Get a single version by version_id."""
    return await fetch_one(
        """
        SELECT version_id, document_id, version_number, content_hash, file_size,
               created_at, created_by, change_source, change_summary,
               parent_version, operations_json, storage_path, is_current
        FROM document_versions
        WHERE version_id = $1
        """,
        version_id,
        rls_context=rls_context,
    )


async def get_latest(
    document_id: str,
    rls_context: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    """Get the latest version (highest version_number) for a document."""
    return await fetch_one(
        """
        SELECT version_id, document_id, version_number, content_hash, file_size,
               created_at, created_by, change_source, change_summary,
               parent_version, operations_json, storage_path, is_current
        FROM document_versions
        WHERE document_id = $1
        ORDER BY version_number DESC
        LIMIT 1
        """,
        document_id,
        rls_context=rls_context,
    )


async def get_current(
    document_id: str,
    rls_context: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    """Get the version marked is_current for a document."""
    return await fetch_one(
        """
        SELECT version_id, document_id, version_number, content_hash, file_size,
               created_at, created_by, change_source, change_summary,
               parent_version, operations_json, storage_path, is_current
        FROM document_versions
        WHERE document_id = $1 AND is_current = TRUE
        LIMIT 1
        """,
        document_id,
        rls_context=rls_context,
    )


async def clear_current_for_document(
    document_id: str,
    rls_context: Optional[Dict[str, str]] = None,
) -> None:
    """Set is_current = FALSE for all versions of this document."""
    await execute(
        "UPDATE document_versions SET is_current = FALSE WHERE document_id = $1",
        document_id,
        rls_context=rls_context,
    )


async def set_current(
    document_id: str,
    version_id: UUID,
    rls_context: Optional[Dict[str, str]] = None,
) -> None:
    """Set is_current = FALSE for all versions of document, then TRUE for the given version_id."""
    await clear_current_for_document(document_id, rls_context=rls_context)
    await execute(
        "UPDATE document_versions SET is_current = TRUE WHERE version_id = $1 AND document_id = $2",
        version_id,
        document_id,
        rls_context=rls_context,
    )


async def count_for_document(
    document_id: str,
    rls_context: Optional[Dict[str, str]] = None,
) -> int:
    """Return the number of versions for a document."""
    val = await fetch_value(
        "SELECT COUNT(*) FROM document_versions WHERE document_id = $1",
        document_id,
        rls_context=rls_context,
    )
    return int(val) if val is not None else 0


async def delete_many(
    version_ids: List[UUID],
    rls_context: Optional[Dict[str, str]] = None,
) -> int:
    """Delete versions by id list. Returns number of rows deleted."""
    if not version_ids:
        return 0
    await execute(
        "DELETE FROM document_versions WHERE version_id = ANY($1::uuid[])",
        version_ids,
        rls_context=rls_context,
    )
    return len(version_ids)


async def get_document_ids_with_versions(
    rls_context: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Return distinct document_ids that have at least one version. For use by pruning task."""
    rows = await fetch_all(
        "SELECT DISTINCT document_id FROM document_versions ORDER BY document_id",
        rls_context=rls_context,
    )
    return [r["document_id"] for r in (rows or [])]
