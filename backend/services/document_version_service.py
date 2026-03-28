"""
Document Version Service - Snapshots, rollback, diff, and history for editable documents.
Phase 1: full snapshots in .versions/{document_id}/, metadata in document_versions table.
"""

import difflib
import hashlib
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from uuid import UUID

from config import settings
from repositories import document_version_repository as version_repo

logger = logging.getLogger(__name__)

EDITABLE_EXTENSIONS = {".md", ".org", ".txt"}
EDITABLE_DOC_TYPES = {"md", "org", "txt"}


def _is_editable(doc_info: Any) -> bool:
    """Return True if document content should be versioned (editable text file)."""
    if not doc_info:
        return False
    ext = Path(doc_info.filename or "").suffix.lower()
    if ext in EDITABLE_EXTENSIONS:
        return True
    doc_type = getattr(doc_info, "doc_type", None)
    if doc_type is not None:
        dt = str(doc_type).lower() if hasattr(doc_type, "value") else str(doc_type).lower()
        if dt in EDITABLE_DOC_TYPES:
            return True
    return False


def _rls_context(user_id: Optional[str], collection_type: str = "user") -> Optional[Dict[str, str]]:
    """Build RLS context for repository calls when we have a user."""
    if not user_id:
        return None
    return {"user_id": user_id, "role": "user"}


async def snapshot_before_write(
    document_id: str,
    user_id: str,
    change_source: str,
    change_summary: Optional[str] = None,
    operations: Optional[List[Dict[str, Any]]] = None,
) -> Optional[UUID]:
    """
    Snapshot current document content before overwriting. Call from all write paths.
    Returns version_id if a snapshot was created, None if skipped (no file, not editable, or unchanged).
    """
    try:
        from services.service_container import get_service_container

        container = await get_service_container()
        document_service = container.document_service
        folder_service = container.folder_service

        doc_info = await document_service.get_document(document_id)
        if not doc_info:
            return None
        if not _is_editable(doc_info):
            return None

        doc_user_id = getattr(doc_info, "user_id", None)
        doc_collection_type = getattr(doc_info, "collection_type", "user")
        team_id = getattr(doc_info, "team_id", None)

        file_path = await folder_service.get_document_file_path(
            filename=doc_info.filename,
            folder_id=getattr(doc_info, "folder_id", None),
            user_id=doc_user_id,
            collection_type=doc_collection_type,
            team_id=team_id,
        )
        if not file_path or not file_path.exists():
            return None

        current_content = file_path.read_text(encoding="utf-8")
        content_hash = hashlib.sha256(current_content.encode("utf-8")).hexdigest()
        file_size = len(current_content.encode("utf-8"))

        rls = _rls_context(doc_user_id, doc_collection_type)
        latest = await version_repo.get_latest(document_id, rls_context=rls)
        if latest and latest.get("content_hash") == content_hash:
            return None

        version_number = (latest["version_number"] + 1) if latest else 1
        suffix = file_path.suffix or ".md"
        version_dir = file_path.parent / ".versions" / document_id
        version_dir.mkdir(parents=True, exist_ok=True)
        version_filename = f"v{version_number:03d}_{content_hash[:8]}{suffix}"
        version_path = version_dir / version_filename
        shutil.copy2(file_path, version_path)

        upload_root = Path(settings.UPLOAD_DIR).resolve()
        try:
            storage_path = str(version_path.relative_to(upload_root))
        except ValueError:
            storage_path = str(version_path)

        version_id = await version_repo.create(
            document_id=document_id,
            version_number=version_number,
            content_hash=content_hash,
            file_size=file_size,
            storage_path=storage_path,
            change_source=change_source,
            created_by=user_id,
            change_summary=change_summary,
            operations_json=operations,
            is_current=True,
            rls_context=rls,
        )
        if version_id:
            await version_repo.set_current(document_id, version_id, rls_context=rls)
        return version_id
    except Exception as e:
        logger.warning("Document version snapshot failed (non-fatal): %s", e)
        return None


async def list_versions(
    document_id: str,
    skip: int = 0,
    limit: int = 100,
    user_id: Optional[str] = None,
    collection_type: str = "user",
) -> List[Dict[str, Any]]:
    """List versions for a document, newest first."""
    rls = _rls_context(user_id, collection_type)
    rows = await version_repo.list_for_document(document_id, skip=skip, limit=limit, rls_context=rls)
    return rows or []


async def get_version_content(
    version_id: UUID,
    user_id: Optional[str] = None,
    collection_type: str = "user",
) -> Optional[str]:
    """Return full content of a version, or None if not found."""
    rls = _rls_context(user_id, collection_type)
    version = await version_repo.get_by_id(version_id, rls_context=rls)
    if not version:
        return None
    storage_path = version.get("storage_path")
    if not storage_path:
        return None
    full_path = Path(settings.UPLOAD_DIR).resolve() / storage_path
    if not full_path.exists():
        logger.warning("Version file missing: %s", full_path)
        return None
    return full_path.read_text(encoding="utf-8")


async def diff_versions(
    document_id: str,
    version_a_id: UUID,
    version_b_id: UUID,
    user_id: Optional[str] = None,
    collection_type: str = "user",
) -> Optional[Dict[str, Any]]:
    """Return unified diff and line counts between two versions."""
    rls = _rls_context(user_id, collection_type)
    a = await version_repo.get_by_id(version_a_id, rls_context=rls)
    b = await version_repo.get_by_id(version_b_id, rls_context=rls)
    if not a or not b or a.get("document_id") != document_id or b.get("document_id") != document_id:
        return None
    content_a = await get_version_content(version_a_id, user_id=user_id, collection_type=collection_type)
    content_b = await get_version_content(version_b_id, user_id=user_id, collection_type=collection_type)
    if content_a is None or content_b is None:
        return None
    lines_a = content_a.splitlines(keepends=True)
    lines_b = content_b.splitlines(keepends=True)
    diff_lines = list(
        difflib.unified_diff(
            lines_a,
            lines_b,
            fromfile=f"v{a['version_number']}",
            tofile=f"v{b['version_number']}",
        )
    )
    diff_text = "".join(diff_lines)
    additions = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
    deletions = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))
    return {
        "diff": diff_text,
        "additions": additions,
        "deletions": deletions,
        "from_version": version_a_id,
        "to_version": version_b_id,
    }


async def rollback_to_version(
    document_id: str,
    target_version_id: UUID,
    user_id: str,
) -> Dict[str, Any]:
    """
    Rollback document to a prior version. Current state is snapshotted first so rollback is reversible.
    Returns dict with success, message, and new_version_id (the snapshot of current state).
    """
    try:
        from services.service_container import get_service_container

        container = await get_service_container()
        document_service = container.document_service
        folder_service = container.folder_service

        doc_info = await document_service.get_document(document_id)
        if not doc_info:
            return {"success": False, "error": "Document not found", "message": f"Document {document_id} not found"}

        doc_user_id = getattr(doc_info, "user_id", None)
        doc_collection_type = getattr(doc_info, "collection_type", "user")
        team_id = getattr(doc_info, "team_id", None)
        rls = _rls_context(doc_user_id, doc_collection_type)

        target = await version_repo.get_by_id(target_version_id, rls_context=rls)
        if not target or target.get("document_id") != document_id:
            return {"success": False, "error": "Version not found", "message": "Version not found or access denied"}

        new_version_id = await snapshot_before_write(
            document_id=document_id,
            user_id=user_id,
            change_source="rollback",
            change_summary=f"Rolled back to v{target['version_number']}",
            operations=None,
        )

        target_content = await get_version_content(target_version_id, user_id=doc_user_id, collection_type=doc_collection_type)
        if target_content is None:
            return {"success": False, "error": "Version content missing", "message": "Version file not found on disk"}

        file_path = await folder_service.get_document_file_path(
            filename=doc_info.filename,
            folder_id=getattr(doc_info, "folder_id", None),
            user_id=doc_user_id,
            collection_type=doc_collection_type,
            team_id=team_id,
        )
        if not file_path:
            return {"success": False, "error": "Path resolution failed", "message": "Could not resolve document path"}

        file_path.write_text(target_content, encoding="utf-8")
        await document_service.document_repository.update_file_size(
            document_id, len(target_content.encode("utf-8")), user_id=doc_user_id
        )

        exempt = getattr(doc_info, "exempt_from_vectorization", False)
        if exempt:
            from models.api_models import ProcessingStatus
            await document_service.document_repository.update_status(
                document_id, ProcessingStatus.COMPLETED, user_id=doc_user_id
            )
        else:
            from services.celery_tasks.document_tasks import reprocess_document_after_save_task
            from models.api_models import ProcessingStatus
            await document_service.document_repository.update_status(
                document_id, ProcessingStatus.EMBEDDING, user_id=doc_user_id
            )
            reprocess_document_after_save_task.delay(document_id, doc_user_id or "system")

        return {
            "success": True,
            "message": f"Rolled back to version {target['version_number']}",
            "new_version_id": str(new_version_id) if new_version_id else None,
            "target_version_number": target["version_number"],
        }
    except Exception as e:
        logger.exception("Rollback failed: %s", e)
        return {"success": False, "error": str(e), "message": str(e)}


async def prune_old_versions(
    document_id: str,
    retention_days: int = 90,
    keep_every_nth: int = 10,
    max_versions: Optional[int] = 200,
    user_id: Optional[str] = None,
    collection_type: str = "user",
) -> Dict[str, Any]:
    """
    Remove old versions per retention policy. Keeps first version, current version,
    all versions younger than retention_days, and every keep_every_nth version older than that.
    """
    from datetime import datetime, timezone, timedelta

    rls = _rls_context(user_id, collection_type)
    all_versions = await version_repo.list_for_document(
        document_id, skip=0, limit=10000, rls_context=rls
    )
    if not all_versions:
        return {"pruned": 0, "kept": 0}

    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    current_row = await version_repo.get_current(document_id, rls_context=rls)
    current_id = current_row["version_id"] if current_row else None

    to_keep = set()
    for v in all_versions:
        vid = v["version_id"]
        if v["version_number"] == 1:
            to_keep.add(vid)
            continue
        if current_id and vid == current_id:
            to_keep.add(vid)
            continue
        created = v.get("created_at")
        if created and created > cutoff:
            to_keep.add(vid)
            continue

    by_number = sorted(all_versions, key=lambda x: x["version_number"])
    for i, v in enumerate(by_number):
        if v["version_number"] == 1 or (current_id and v["version_id"] == current_id):
            continue
        if v.get("created_at") and v["created_at"] > cutoff:
            continue
        if i % keep_every_nth == 0:
            to_keep.add(v["version_id"])

    if max_versions is not None and len(to_keep) > max_versions:
        protected = {v["version_id"] for v in all_versions if v["version_number"] == 1 or (current_id and v["version_id"] == current_id)}
        candidates = [v for v in all_versions if v["version_id"] in to_keep and v["version_id"] not in protected]
        candidates.sort(key=lambda x: x.get("created_at") or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        allowed = max_versions - len(protected)
        to_keep = protected | set(v["version_id"] for v in candidates[:allowed])

    to_delete = [v["version_id"] for v in all_versions if v["version_id"] not in to_keep]
    if not to_delete:
        return {"pruned": 0, "kept": len(all_versions)}

    await version_repo.delete_many(to_delete, rls_context=rls)
    return {"pruned": len(to_delete), "kept": len(to_keep)}
