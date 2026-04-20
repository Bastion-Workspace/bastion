"""
Admin APIs: document vector audit / re-embed (Phase 3d) and collection recreate (Phase 3b).
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from models.api_models import AuthenticatedUserResponse
from services.celery_tasks.document_tasks import reprocess_document_after_save_task
from services.document_vector_audit_service import (
    recreate_document_collections_and_queue,
    run_document_vector_audit,
)
from utils.auth_middleware import require_admin

logger = logging.getLogger(__name__)

router = APIRouter(tags=["admin"])


class AuditAndReembedRequest(BaseModel):
    scope: str = Field(
        default="all",
        description="one of: all, global, user, team",
    )
    user_id: Optional[str] = None
    team_id: Optional[str] = None
    dry_run: bool = True
    max_concurrent: int = Field(default=5, ge=1, le=500)
    throttle_seconds: float = Field(default=0.2, ge=0.0, le=60.0)


class RecreateDocumentCollectionsRequest(BaseModel):
    scope: str = Field(
        default="global",
        description="one of: all, global, user, team",
    )
    user_id: Optional[str] = None
    team_id: Optional[str] = None
    dry_run: bool = True
    queue_reembed: bool = True
    max_concurrent: int = Field(default=5, ge=1, le=500)
    throttle_seconds: float = Field(default=0.2, ge=0.0, le=60.0)
    include_all_qdrant_embedding_collections: bool = Field(
        default=False,
        description=(
            "When scope is 'all': list all Qdrant collections and also wipe+recreate every one "
            "that uses the text embedding model (documents, skills, tools, help_docs, team_*, "
            "user_*_documents), not only collections implied by document_metadata. "
            "Ignored for other scopes."
        ),
    )


def _normalize_scope(scope: str) -> str:
    s = (scope or "all").strip().lower()
    if s not in ("all", "global", "user", "team"):
        raise HTTPException(status_code=400, detail=f"Invalid scope: {scope}")
    return s


@router.post("/api/admin/audit-and-reembed")
async def audit_and_reembed(
    body: AuditAndReembedRequest,
    current_user: AuthenticatedUserResponse = Depends(require_admin()),
) -> Dict[str, Any]:
    """
    Compare Postgres documents that should have vectors to Qdrant payloads; optionally queue re-embed.

    Image bodies are not embedded; only image rows that have `document_chunks` (e.g. metadata
    sidecar text) are treated as expected in Qdrant. Plain images with no chunk rows are excluded
    so global libraries do not flood the re-embed queue.
    """
    scope = _normalize_scope(body.scope)
    logger.info(
        "audit-and-reembed by %s scope=%s dry_run=%s",
        current_user.username,
        scope,
        body.dry_run,
    )

    audit = await run_document_vector_audit(scope, body.user_id, body.team_id)
    if not audit.get("success"):
        raise HTTPException(status_code=500, detail=audit.get("error", "audit failed"))

    missing_by_collection = audit.get("missing_by_collection") or {}
    counts = {k: len(v) for k, v in missing_by_collection.items()}
    reembed_queue: List[Dict[str, str]] = audit.get("reembed_queue") or []

    queued = 0
    if not body.dry_run and reembed_queue:
        batch = max(1, int(body.max_concurrent))
        seen_ids: set = set()
        for item in reembed_queue:
            did = item["document_id"]
            if did in seen_ids:
                continue
            seen_ids.add(did)
            reprocess_document_after_save_task.delay(
                did,
                item.get("user_id") or "",
            )
            queued += 1
            if body.throttle_seconds > 0 and queued % batch == 0:
                await asyncio.sleep(body.throttle_seconds)

    return {
        "total_expected": audit.get("total_expected", 0),
        "total_present": audit.get("total_present", 0),
        "total_missing": audit.get("total_missing", 0),
        "missing_by_collection": counts,
        "missing_document_ids_sample": {
            k: v[:50] for k, v in missing_by_collection.items()
        },
        "queued_for_reembed": queued,
        "dry_run": body.dry_run,
    }


@router.post("/api/admin/recreate-document-collections")
async def recreate_document_collections(
    body: RecreateDocumentCollectionsRequest,
    current_user: AuthenticatedUserResponse = Depends(require_admin()),
) -> Dict[str, Any]:
    """
    Delete and recreate document Qdrant collections (named_hybrid when hybrid is enabled), then queue re-embed.

    Full one-time reset: `scope=all`, `dry_run=false`, `queue_reembed=true`,
    `include_all_qdrant_embedding_collections=true`. Then restart the backend (or wait for Celery)
    so built-in skills, help_docs, and tools vectors are repopulated by startup tasks.
    """
    scope = _normalize_scope(body.scope)
    logger.warning(
        "recreate-document-collections by %s scope=%s dry_run=%s queue=%s qdrant_scan=%s",
        current_user.username,
        scope,
        body.dry_run,
        body.queue_reembed,
        body.include_all_qdrant_embedding_collections,
    )

    result = await recreate_document_collections_and_queue(
        scope,
        body.user_id,
        body.team_id,
        dry_run=body.dry_run,
        queue_reembed=body.queue_reembed,
        throttle_seconds=body.throttle_seconds,
        max_concurrent=body.max_concurrent,
        include_all_qdrant_embedding_collections=body.include_all_qdrant_embedding_collections,
    )
    if not result.get("success") and result.get("errors"):
        raise HTTPException(
            status_code=500,
            detail=f"recreate failed: {result.get('errors')}",
        )
    return result
