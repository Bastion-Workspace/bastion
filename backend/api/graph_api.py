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


async def _fetch_link_graph_data(
    user_id: str,
    rls_context: dict,
    scope: str = "all",
    folder_id: Optional[str] = None,
) -> dict:
    """Internal: fetch link graph nodes and edges. Caller adds degree to nodes if needed."""
    text_filter = (
        "AND (LOWER(filename) LIKE '%.org' OR LOWER(filename) LIKE '%.md' OR LOWER(filename) LIKE '%.txt')"
    )
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

    # Deduplicate edges: one edge per (source, target) with weight = number of references
    edge_agg = {}
    unresolved_targets = []
    for r in link_rows:
        src = r["source_document_id"]
        tgt = r.get("target_document_id")
        if not tgt:
            unresolved_targets.append({
                "raw_path": r.get("target_raw_path") or "",
                "source_id": src,
            })
            continue
        if tgt not in visible_ids:
            continue
        key = (src, tgt)
        if key not in edge_agg:
            edge_agg[key] = {
                "count": 0,
                "link_type": r.get("link_type") or "org_file",
                "description": r.get("description"),
            }
        edge_agg[key]["count"] += 1

    edges = []
    for (src, tgt), agg in edge_agg.items():
        edges.append({
            "source": src,
            "target": tgt,
            "link_type": agg["link_type"],
            "description": agg["description"],
            "weight": agg["count"],
        })

    degree_map = {}
    for edge in edges:
        degree_map[edge["source"]] = degree_map.get(edge["source"], 0) + 1
        degree_map[edge["target"]] = degree_map.get(edge["target"], 0) + 1

    nodes = []
    for r in all_doc_rows:
        doc_id = r["document_id"]
        nodes.append({
            "id": doc_id,
            "label": r.get("filename") or doc_id,
            "title": r.get("title") or r.get("filename") or doc_id,
            "type": doc_type_by_id.get(doc_id, "text"),
            "folder_name": folder_name_by_id.get(doc_id, ""),
            "collection_type": r.get("collection_type") or "user",
            "link_count": degree_map.get(doc_id, 0),
            "degree": degree_map.get(doc_id, 0),
        })

    return {
        "nodes": nodes,
        "edges": edges,
        "unresolved_targets": unresolved_targets,
    }


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
    rls_context = {"user_id": user_id, "user_role": "user"}
    logger.info("Link graph request: scope=%s folder_id=%s user_id=%s", scope, folder_id, user_id)
    try:
        return await _fetch_link_graph_data(user_id, rls_context, scope, folder_id)
    except Exception as e:
        logger.error("Link graph failed: %s", e)
        raise


@router.get("/api/graph/entity-graph")
async def get_entity_graph(
    entity_limit: int = Query(100, description="Max entity nodes to return"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """
    Return nodes and edges for the entity (knowledge) graph. Entities and
    documents are restricted to the current user's documents (RLS). Nodes
    are entities and documents; edges are mentions (entity->doc) and
    co-occurrence (entity->entity).
    """
    user_id = current_user.user_id
    rls_context = {"user_id": user_id, "user_role": "user"}
    text_filter = (
        "AND (LOWER(filename) LIKE '%.org' OR LOWER(filename) LIKE '%.md' OR LOWER(filename) LIKE '%.txt')"
    )
    try:
        doc_rows = await fetch_all(
            """
            SELECT document_id, filename, COALESCE(title, filename) AS title
            FROM document_metadata
            WHERE user_id = $1
            """ + text_filter + """
            ORDER BY filename
            """,
            user_id,
            rls_context=rls_context,
        )
        document_ids = [r["document_id"] for r in doc_rows]
        doc_meta_by_id = {r["document_id"]: r for r in doc_rows}
        if not document_ids:
            return {
                "nodes": [],
                "edges": [],
                "entity_count": 0,
                "document_count": 0,
            }
        from services.knowledge_graph_service import KnowledgeGraphService
        kg_service = KnowledgeGraphService()
        await kg_service.initialize()
        data = await kg_service.get_entity_graph_for_documents(
            document_ids=document_ids,
            entity_limit=entity_limit,
        )
        entity_nodes = data.get("entity_nodes") or []
        kg_doc_ids = set(data.get("document_ids") or [])
        co_occurrence_edges = data.get("co_occurrence_edges") or []
        doc_ids_to_resolve = list(kg_doc_ids)
        doc_meta_resolved = {}
        if doc_ids_to_resolve:
            placeholders = ",".join(f"${i+1}" for i in range(len(doc_ids_to_resolve)))
            user_param = len(doc_ids_to_resolve) + 1
            resolved = await fetch_all(
                f"""
                SELECT document_id, filename, COALESCE(title, filename) AS title
                FROM document_metadata
                WHERE document_id IN ({placeholders}) AND user_id = ${user_param}
                """,
                *doc_ids_to_resolve,
                user_id,
                rls_context=rls_context,
            )
            doc_meta_resolved = {r["document_id"]: r for r in resolved}
        nodes = []
        for en in entity_nodes:
            name = en.get("name") or ""
            nodes.append({
                "id": f"entity:{name}",
                "label": name,
                "type": "entity",
                "entity_type": (en.get("type") or "MISC").upper(),
                "doc_count": en.get("doc_count") or 0,
            })
        for doc_id in kg_doc_ids:
            meta = doc_meta_resolved.get(doc_id) or doc_meta_by_id.get(doc_id)
            label = (meta and (meta.get("filename") or meta.get("title"))) or doc_id
            title = (meta and meta.get("title")) or label
            nodes.append({
                "id": f"doc:{doc_id}",
                "label": label,
                "title": title,
                "type": "document",
            })
        edges = []
        for en in entity_nodes:
            entity_id = f"entity:{en.get('name') or ''}"
            for doc_id in en.get("doc_ids") or []:
                if doc_id in kg_doc_ids:
                    edges.append({
                        "source": entity_id,
                        "target": f"doc:{doc_id}",
                        "edge_type": "mentions",
                    })
        for e in co_occurrence_edges:
            edges.append({
                "source": f"entity:{e.get('source') or ''}",
                "target": f"entity:{e.get('target') or ''}",
                "edge_type": "co_occurs",
                "weight": e.get("weight") or 0,
            })
        return {
            "nodes": nodes,
            "edges": edges,
            "entity_count": len(entity_nodes),
            "document_count": len(kg_doc_ids),
        }
    except Exception as e:
        logger.error("Entity graph failed: %s", e)
        raise


@router.get("/api/graph/unified-graph")
async def get_unified_graph(
    layers: str = Query("files,entities", description="Comma-separated: files, entities"),
    entity_limit: int = Query(100, description="Max entity nodes when entities layer is included"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """
    Return merged nodes and edges for file link graph and entity graph in one payload.
    Nodes have node_type 'file' or 'entity'; edges have edge_type 'file_link', 'mentions', or 'co_occurs'.
    """
    user_id = current_user.user_id
    rls_context = {"user_id": user_id, "user_role": "user"}
    include_files = "files" in (layers or "").lower().replace(" ", "").split(",")
    include_entities = "entities" in (layers or "").lower().replace(" ", "").split(",")
    try:
        nodes = []
        edges = []

        if include_files:
            link_data = await _fetch_link_graph_data(user_id, rls_context, "all", None)
            for node in link_data.get("nodes") or []:
                doc_id = node["id"]
                nodes.append({
                    "id": doc_id,
                    "label": node.get("label") or doc_id,
                    "node_type": "file",
                    "degree": node.get("degree", 0),
                    "file_type": node.get("type", "other"),
                })
            for edge in link_data.get("edges") or []:
                edges.append({
                    "source": edge["source"],
                    "target": edge["target"],
                    "edge_type": "file_link",
                    "weight": edge.get("weight", 1),
                })

        if include_entities:
            text_filter = (
                "AND (LOWER(filename) LIKE '%.org' OR LOWER(filename) LIKE '%.md' OR LOWER(filename) LIKE '%.txt')"
            )
            doc_rows = await fetch_all(
                """
                SELECT document_id, filename, COALESCE(title, filename) AS title
                FROM document_metadata
                WHERE user_id = $1
                """ + text_filter + """
                ORDER BY filename
                """,
                user_id,
                rls_context=rls_context,
            )
            document_ids = [r["document_id"] for r in doc_rows]
            doc_meta_by_id_from_entity = {r["document_id"]: r for r in doc_rows}
            if document_ids:
                from services.knowledge_graph_service import KnowledgeGraphService
                kg_service = KnowledgeGraphService()
                await kg_service.initialize()
                data = await kg_service.get_entity_graph_for_documents(
                    document_ids=document_ids,
                    entity_limit=entity_limit,
                )
                entity_nodes = data.get("entity_nodes") or []
                kg_doc_ids = set(data.get("document_ids") or [])
                co_occurrence_edges = data.get("co_occurrence_edges") or []
                seen_file_ids = {n["id"] for n in nodes if n.get("node_type") == "file"}
                for en in entity_nodes:
                    name = en.get("name") or ""
                    nodes.append({
                        "id": f"entity:{name}",
                        "label": name,
                        "node_type": "entity",
                        "entity_type": (en.get("type") or "MISC").upper(),
                        "doc_count": en.get("doc_count") or 0,
                    })
                for doc_id in kg_doc_ids:
                    if doc_id not in seen_file_ids:
                        meta = doc_meta_by_id_from_entity.get(doc_id)
                        label = (meta and (meta.get("filename") or meta.get("title"))) or doc_id
                        nodes.append({
                            "id": doc_id,
                            "label": label,
                            "node_type": "file",
                            "degree": 0,
                            "file_type": "other",
                        })
                        seen_file_ids.add(doc_id)
                for en in entity_nodes:
                    entity_id = f"entity:{en.get('name') or ''}"
                    for doc_id in en.get("doc_ids") or []:
                        if doc_id in kg_doc_ids:
                            edges.append({
                                "source": entity_id,
                                "target": doc_id,
                                "edge_type": "mentions",
                                "weight": 1,
                            })
                for e in co_occurrence_edges:
                    edges.append({
                        "source": f"entity:{e.get('source') or ''}",
                        "target": f"entity:{e.get('target') or ''}",
                        "edge_type": "co_occurs",
                        "weight": e.get("weight") or 0,
                    })

        return {"nodes": nodes, "edges": edges}
    except Exception as e:
        logger.error("Unified graph failed: %s", e)
        raise


@router.get("/api/graph/entity/{entity_name:path}")
async def get_entity_detail(
    entity_name: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """
    Return entity detail: type, confidence, document mentions with context and titles,
    and co-occurring entities. Scoped to the current user's documents (RLS).
    """
    user_id = current_user.user_id
    rls_context = {"user_id": user_id, "user_role": "user"}
    text_filter = (
        "AND (LOWER(filename) LIKE '%.org' OR LOWER(filename) LIKE '%.md' OR LOWER(filename) LIKE '%.txt')"
    )
    try:
        doc_rows = await fetch_all(
            """
            SELECT document_id, filename, COALESCE(title, filename) AS title
            FROM document_metadata
            WHERE user_id = $1
            """ + text_filter + """
            ORDER BY filename
            """,
            user_id,
            rls_context=rls_context,
        )
        document_ids = [r["document_id"] for r in doc_rows]
        doc_meta_by_id = {r["document_id"]: r for r in doc_rows}
        if not document_ids:
            return {
                "name": entity_name,
                "entity_type": None,
                "confidence": None,
                "document_mentions": [],
                "co_occurring_entities": [],
            }
        from services.knowledge_graph_service import KnowledgeGraphService
        kg_service = KnowledgeGraphService()
        await kg_service.initialize()
        data = await kg_service.get_entity_detail(
            entity_name=entity_name,
            user_document_ids=document_ids,
        )
        mentions = data.get("document_mentions") or []
        doc_ids_mentioned = [m["document_id"] for m in mentions]
        for m in mentions:
            meta = doc_meta_by_id.get(m["document_id"])
            m["title"] = (meta and meta.get("title")) or m["document_id"]
            m["filename"] = (meta and meta.get("filename")) or ""
        data["document_mentions"] = mentions
        return data
    except Exception as e:
        logger.error("Entity detail failed for %s: %s", entity_name, e)
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
