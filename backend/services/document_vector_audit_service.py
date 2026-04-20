"""
Postgres vs Qdrant document vector audit (Phase 3d) and admin recreate helpers (Phase 3b).
"""

import asyncio
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from bastion_indexing.policy import sql_eligible_doc_types_tuple

from config import settings
from services.database_manager.database_helpers import fetch_all
from services.vector_audit_scroll_helper import collect_document_ids_from_collection

logger = logging.getLogger(__name__)

_ADMIN_RLS = {"user_id": "", "user_role": "admin"}

# Align with vector-service/service/grpc_service.py _validate_embedding_dimensions:
# which collections use Settings.EMBEDDING_DIMENSIONS (not fixed encoders).
_DOCUMENT_LIKE_NAMES = frozenset(
    ("documents", "skills", "help_docs"),
)


def collection_uses_configurable_text_embedding(collection_name: str) -> bool:
    """True if this Qdrant collection stores dense vectors from the app embedding model."""
    name = (collection_name or "").strip()
    if not name:
        return False
    if "face_encodings" in name or "object_features" in name:
        return False
    if name in _DOCUMENT_LIKE_NAMES:
        return True
    if name.startswith("team_"):
        return True
    if name.endswith("_documents"):
        return True
    return False


async def discover_configurable_embedding_collection_names(client: Any) -> List[str]:
    """List existing Qdrant collections that use EMBEDDING_DIMENSIONS."""
    result = await client.list_collections()
    if not result.get("success"):
        logger.warning(
            "discover_configurable_embedding_collection_names: list_collections failed: %s",
            result.get("error"),
        )
        return []
    names: List[str] = []
    for col in result.get("collections") or []:
        n = col.get("name")
        if n and collection_uses_configurable_text_embedding(n):
            names.append(n)
    return sorted(set(names))

# Expected vectors: canonical chunk-eligible types (bastion_indexing.policy), including image_sidecar.
def _base_sql_args() -> Tuple[str, List[Any]]:
    """Return (sql, args) for documents that should have Qdrant points; $1 = eligible types."""
    eligible = list(sql_eligible_doc_types_tuple())
    sql = """
SELECT document_id, user_id, team_id, collection_type
FROM document_metadata
WHERE processing_status = 'completed'
  AND (exempt_from_vectorization IS NOT TRUE)
  AND chunk_count > 0
  AND doc_type = ANY($1::text[])
"""
    return sql, [eligible]


def qdrant_collection_for_document_row(row: Dict[str, Any]) -> Optional[str]:
    ct = (row.get("collection_type") or "").strip().lower()
    if ct == "global":
        return settings.VECTOR_COLLECTION_NAME
    if ct == "user":
        uid = row.get("user_id")
        if not uid:
            return None
        return f"user_{uid}_documents"
    if ct == "team":
        tid = row.get("team_id")
        if tid is None:
            return None
        return f"team_{tid}"
    return None


async def load_expected_documents(
    scope: str,
    user_id: Optional[str],
    team_id: Optional[str],
) -> List[Dict[str, Any]]:
    scope_l = (scope or "all").strip().lower()
    sql, args = _base_sql_args()
    sql = sql.rstrip()

    if scope_l == "global":
        sql += " AND collection_type = 'global'"
    elif scope_l == "user":
        sql += " AND collection_type = 'user'"
        if user_id:
            sql += " AND user_id = $2"
            args.append(user_id)
    elif scope_l == "team":
        sql += " AND collection_type = 'team'"
        if team_id:
            sql += " AND team_id = $2::uuid"
            args.append(team_id)
    elif scope_l == "all":
        pass
    else:
        raise ValueError(f"Unknown scope: {scope}")

    rows = await fetch_all(sql, *args, rls_context=_ADMIN_RLS)
    return rows


def _reembed_user_id_for_row(row: Dict[str, Any]) -> str:
    uid = row.get("user_id")
    return str(uid) if uid else ""


def collection_targets_for_recreate(
    scope: str,
    user_id: Optional[str],
    team_id: Optional[str],
    rows: List[Dict[str, Any]],
) -> List[str]:
    """Qdrant collection names to delete+recreate for the given scope."""
    scope_l = (scope or "all").strip().lower()
    targets: Set[str] = set()

    if scope_l in ("global", "all"):
        targets.add(settings.VECTOR_COLLECTION_NAME)
    if scope_l == "user" and user_id:
        targets.add(f"user_{user_id}_documents")
    if scope_l == "team" and team_id:
        targets.add(f"team_{team_id}")

    for row in rows:
        c = qdrant_collection_for_document_row(row)
        if c:
            targets.add(c)

    return sorted(targets)


async def run_document_vector_audit(
    scope: str,
    user_id: Optional[str] = None,
    team_id: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        rows = await load_expected_documents(scope, user_id, team_id)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    by_collection: Dict[str, Set[str]] = defaultdict(set)
    reprocess_user: Dict[str, str] = {}

    for row in rows:
        coll = qdrant_collection_for_document_row(row)
        did = row.get("document_id")
        if not coll or not did:
            continue
        did_s = str(did)
        by_collection[coll].add(did_s)
        reprocess_user[did_s] = _reembed_user_id_for_row(row)

    total_expected = sum(len(s) for s in by_collection.values())
    total_present = 0
    missing_by_collection: Dict[str, List[str]] = {}

    for coll, expected in sorted(by_collection.items()):
        present, err = await collect_document_ids_from_collection(coll)
        if err:
            return {
                "success": False,
                "error": f"Qdrant scroll failed for {coll}: {err}",
            }
        total_present += len(expected & present)
        missing = sorted(expected - present)
        if missing:
            missing_by_collection[coll] = missing

    total_missing = sum(len(v) for v in missing_by_collection.values())
    reembed_queue: List[Dict[str, str]] = []
    for coll, ids in missing_by_collection.items():
        for did in ids:
            reembed_queue.append(
                {"document_id": did, "user_id": reprocess_user.get(did, ""), "collection": coll}
            )

    return {
        "success": True,
        "total_expected": total_expected,
        "total_present": total_present,
        "total_missing": total_missing,
        "missing_by_collection": missing_by_collection,
        "reembed_queue": reembed_queue,
    }


async def recreate_document_collections_and_queue(
    scope: str,
    user_id: Optional[str],
    team_id: Optional[str],
    *,
    dry_run: bool,
    queue_reembed: bool,
    throttle_seconds: float,
    max_concurrent: int,
    include_all_qdrant_embedding_collections: bool = False,
) -> Dict[str, Any]:
    """Delete and recreate document Qdrant collections as named_hybrid when hybrid is enabled."""
    from clients.vector_service_client import get_vector_service_client

    try:
        rows = await load_expected_documents(scope, user_id, team_id)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    targets = collection_targets_for_recreate(scope, user_id, team_id, rows)
    hybrid = getattr(settings, "HYBRID_SEARCH_ENABLED", False)
    dim = int(settings.EMBEDDING_DIMENSIONS)

    scope_l = (scope or "all").strip().lower()
    merge_from_qdrant = (
        include_all_qdrant_embedding_collections and scope_l == "all"
    )
    discovered_from_qdrant: List[str] = []
    if merge_from_qdrant:
        list_client = await get_vector_service_client(required=False)
        if list_client:
            discovered_from_qdrant = await discover_configurable_embedding_collection_names(
                list_client
            )
            if discovered_from_qdrant:
                db_only = len(targets)
                targets = sorted(set(targets) | set(discovered_from_qdrant))
                logger.info(
                    "Recreate targets: merged %d Qdrant embedding collection(s) with %d "
                    "from document_metadata → %d total",
                    len(discovered_from_qdrant),
                    db_only,
                    len(targets),
                )
        else:
            logger.warning(
                "include_all_qdrant_embedding_collections: vector service unavailable; "
                "using document_metadata-derived targets only"
            )

    if dry_run:
        return {
            "success": True,
            "dry_run": True,
            "collections": targets,
            "hybrid_sparse": hybrid,
            "documents_in_scope": len(rows),
            "would_queue_reembed": len(rows) if queue_reembed else 0,
            "qdrant_embedding_collections_discovered": discovered_from_qdrant,
        }

    client = await get_vector_service_client(required=True)
    errors: List[str] = []

    for name in targets:
        dr = await client.delete_collection(name)
        if not dr.get("success"):
            err = (dr.get("error") or "").lower()
            if err and all(
                x not in err for x in ("404", "not found", "doesn't exist", "does not exist")
            ):
                errors.append(f"delete {name}: {dr.get('error')}")
        cr = await client.create_collection(
            collection_name=name,
            vector_size=dim,
            distance="COSINE",
            enable_sparse=hybrid,
        )
        if not cr.get("success"):
            errors.append(f"create {name}: {cr.get('error')}")

    queued = 0
    if queue_reembed and rows:
        from services.celery_tasks.document_tasks import bulk_reindex_batch_task

        # Batch documents into groups to reduce Redis round-trips and isolate bulk
        # reindexing from user-facing tasks via the dedicated 'reindex' queue.
        REINDEX_BATCH_SIZE = 50
        valid_rows = [r for r in rows if r.get("document_id")]
        batches = [valid_rows[i:i + REINDEX_BATCH_SIZE] for i in range(0, len(valid_rows), REINDEX_BATCH_SIZE)]
        for batch_idx, batch in enumerate(batches):
            payload = [
                {
                    "document_id": str(r["document_id"]),
                    "user_id": _reembed_user_id_for_row(r),
                    "team_id": str(r["team_id"]) if r.get("team_id") else None,
                    "collection_type": (r.get("collection_type") or "user"),
                }
                for r in batch
            ]
            bulk_reindex_batch_task.apply_async(args=[payload], queue="reindex")
            queued += len(batch)
            if throttle_seconds > 0 and (batch_idx + 1) % max(1, int(max_concurrent)) == 0:
                await asyncio.sleep(throttle_seconds)

    return {
        "success": len(errors) == 0,
        "dry_run": False,
        "collections_recreated": targets,
        "errors": errors,
        "queued_for_reembed": queued,
        "qdrant_embedding_collections_discovered": discovered_from_qdrant,
    }
