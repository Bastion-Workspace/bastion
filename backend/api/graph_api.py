"""
Graph API - link graph for file relation cloud (Obsidian-style).

GET /api/graph/link-graph returns nodes and edges for the current user's My Documents only.
Documents and links are filtered by user_id so file relations never leak across users.
"""

import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, Query

from models.api_models import AuthenticatedUserResponse
from services.database_manager.database_helpers import fetch_all
from services.link_extraction_service import get_link_extraction_service
from utils.auth_middleware import get_current_user, require_admin

logger = logging.getLogger(__name__)

router = APIRouter(tags=["graph"])


@router.get("/api/graph/link-graph")
async def get_link_graph(
    scope: str = Query("all", description="Scope: all or folder"),
    folder_id: Optional[str] = Query(None, description="Optional folder_id to scope graph"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """
    Return nodes and edges for the file relation graph. Only text files (.org, .md, .txt)
    are included. Graph is restricted to the current user's own documents (My Documents)
    so file relations never leak across users.
    """
    user_id = current_user.user_id
    # Use role 'user' so RLS never applies admin bypass; graph shows only this user's scope
    rls_context = {"user_id": user_id, "user_role": "user"}
    logger.info("Link graph request: scope=%s folder_id=%s user_id=%s", scope, folder_id, user_id)

    # Only include text files that can contain links (.org, .md, .txt)
    text_filter = (
        "AND (LOWER(filename) LIKE '%.org' OR LOWER(filename) LIKE '%.md' OR LOWER(filename) LIKE '%.txt')"
    )
    try:
        if folder_id and scope == "folder":
            doc_rows = await fetch_all(
                """
                SELECT document_id, filename, folder_id, collection_type,
                       COALESCE(title, filename) AS title
                FROM document_metadata
                WHERE folder_id = $1 AND user_id = $2
                """ + text_filter + """
                ORDER BY filename
                """,
                folder_id,
                user_id,
                rls_context=rls_context,
            )
        else:
            doc_rows = await fetch_all(
                """
                SELECT document_id, filename, folder_id, collection_type,
                       COALESCE(title, filename) AS title
                FROM document_metadata
                WHERE user_id = $1
                """ + text_filter + """
                ORDER BY filename
                """,
                user_id,
                rls_context=rls_context,
            )

        visible_text_ids = {r["document_id"] for r in doc_rows}
        if not visible_text_ids:
            return {
                "nodes": [],
                "edges": [],
                "unresolved_targets": [],
            }

        placeholders = ",".join(f"${i+1}" for i in range(len(visible_text_ids)))
        link_user_param = len(visible_text_ids) + 1
        link_rows = await fetch_all(
            f"""
            SELECT source_document_id, target_document_id, target_raw_path, link_type, description
            FROM document_links
            WHERE source_document_id IN ({placeholders}) AND user_id = ${link_user_param}
            """,
            *visible_text_ids,
            user_id,
            rls_context=rls_context,
        )

        # Include non-text targets (PDFs, images, etc.) only when linked from a text file
        target_ids = {
            r["target_document_id"]
            for r in link_rows
            if r.get("target_document_id") and r["target_document_id"] not in visible_text_ids
        }
        target_doc_rows = []
        if target_ids:
            target_placeholders = ",".join(f"${i+1}" for i in range(len(target_ids)))
            target_user_param = len(target_ids) + 1
            target_doc_rows = await fetch_all(
                f"""
                SELECT document_id, filename, folder_id, collection_type,
                       COALESCE(title, filename) AS title
                FROM document_metadata
                WHERE document_id IN ({target_placeholders}) AND user_id = ${target_user_param}
                ORDER BY filename
                """,
                *target_ids,
                user_id,
                rls_context=rls_context,
            )
        visible_ids = visible_text_ids | {r["document_id"] for r in target_doc_rows}

        def _doc_type(filename: str) -> str:
            fn = (filename or "").lower()
            if fn.endswith(".org"):
                return "org"
            if fn.endswith(".md"):
                return "markdown"
            if fn.endswith(".txt"):
                return "text"
            return "other"

        doc_type_by_id = {}
        folder_name_by_id = {}
        all_doc_rows = list(doc_rows) + target_doc_rows
        for r in all_doc_rows:
            doc_type_by_id[r["document_id"]] = _doc_type(r.get("filename") or "")
            folder_name_by_id[r["document_id"]] = r.get("folder_id") or ""

        out_degree = {}
        in_degree = {}
        for r in link_rows:
            src = r["source_document_id"]
            tgt = r.get("target_document_id")
            out_degree[src] = out_degree.get(src, 0) + 1
            if tgt and tgt in visible_ids:
                in_degree[tgt] = in_degree.get(tgt, 0) + 1

        nodes = []
        for r in all_doc_rows:
            doc_id = r["document_id"]
            link_count = (out_degree.get(doc_id, 0) + in_degree.get(doc_id, 0))
            nodes.append({
                "id": doc_id,
                "label": r.get("filename") or doc_id,
                "title": r.get("title") or r.get("filename") or doc_id,
                "type": doc_type_by_id.get(doc_id, "text"),
                "folder_name": folder_name_by_id.get(doc_id, ""),
                "collection_type": r.get("collection_type") or "user",
                "link_count": link_count,
            })

        edges = []
        unresolved_targets = []
        for r in link_rows:
            src = r["source_document_id"]
            tgt = r.get("target_document_id")
            if tgt and tgt in visible_ids:
                edges.append({
                    "source": src,
                    "target": tgt,
                    "link_type": r.get("link_type") or "org_file",
                    "description": r.get("description"),
                })
            elif not tgt:
                unresolved_targets.append({
                    "raw_path": r.get("target_raw_path") or "",
                    "source_id": src,
                })

        return {
            "nodes": nodes,
            "edges": edges,
            "unresolved_targets": unresolved_targets,
        }
    except Exception as e:
        logger.error("Link graph failed: %s", e)
        raise


@router.post("/api/admin/rebuild-all-links")
async def rebuild_all_links(
    current_user: AuthenticatedUserResponse = Depends(require_admin()),
):
    """
    Backfill document_links for all existing text documents (.org, .md, .txt).
    Admin only. Uses admin RLS context so all documents are visible.
    """
    from services.folder_service import FolderService

    rls_context = {"user_id": "", "user_role": "admin"}
    try:
        rows = await fetch_all(
            """
            SELECT document_id, filename, folder_id, user_id, collection_type, team_id
            FROM document_metadata
            WHERE LOWER(filename) LIKE '%.org'
               OR LOWER(filename) LIKE '%.md'
               OR LOWER(filename) LIKE '%.txt'
            ORDER BY document_id
            """,
            rls_context=rls_context,
        )
        folder_service = FolderService()
        await folder_service.initialize()
        link_service = await get_link_extraction_service()
        processed = 0
        errors = 0
        for r in rows:
            doc_id = r["document_id"]
            filename = r.get("filename") or ""
            folder_id = r.get("folder_id")
            user_id = r.get("user_id")
            collection_type = r.get("collection_type") or "user"
            team_id = r.get("team_id")
            if team_id and not isinstance(team_id, str):
                team_id = str(team_id)
            try:
                path = await folder_service.get_document_file_path(
                    filename=filename,
                    folder_id=folder_id,
                    user_id=user_id,
                    collection_type=collection_type,
                    team_id=team_id,
                )
                if not path or not Path(path).exists():
                    errors += 1
                    continue
                content = Path(path).read_text(encoding="utf-8")
                await link_service.extract_and_store_links(doc_id, content, rls_context)
                processed += 1
            except Exception as e:
                logger.warning("Backfill link failed for %s: %s", doc_id, e)
                errors += 1
        return {
            "success": True,
            "processed": processed,
            "errors": errors,
            "total": len(rows),
        }
    except Exception as e:
        logger.error("Rebuild all links failed: %s", e)
        raise
