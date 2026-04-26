"""
Code workspace chunk indexing: Qdrant collection + hybrid search (vector + Postgres FTS).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

CODE_WORKSPACE_CHUNKS_COLLECTION = "code_workspace_chunks"
RRF_K = 60


def _merge_rrf(
    vector_results: List[Dict[str, Any]],
    fts_results: List[Dict[str, Any]],
    k: int = RRF_K,
) -> List[Dict[str, Any]]:
    scores: Dict[str, float] = {}
    for rank, r in enumerate(vector_results):
        cid = r.get("chunk_id")
        if cid:
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    for rank, r in enumerate(fts_results):
        cid = r.get("chunk_id")
        if cid:
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    seen = set()
    merged: List[Dict[str, Any]] = []
    for r in vector_results:
        cid = r.get("chunk_id")
        if cid and cid not in seen:
            seen.add(cid)
            merged.append({**r, "_rrf": scores.get(cid, 0.0)})
    for r in fts_results:
        cid = r.get("chunk_id")
        if cid and cid not in seen:
            seen.add(cid)
            merged.append({**r, "_rrf": scores.get(cid, 0.0)})
    merged.sort(key=lambda x: -float(x.get("_rrf", 0.0)))
    return merged


async def ensure_code_chunks_collection() -> bool:
    """Create Qdrant collection for code chunks if missing."""
    from config import settings
    from clients.vector_service_client import get_vector_service_client

    client = await get_vector_service_client(required=False)
    if not client:
        return False
    try:
        info = await client.get_collection_info(CODE_WORKSPACE_CHUNKS_COLLECTION)
        if info.get("success") and info.get("collection"):
            return True
    except Exception as e:
        logger.debug("get_collection_info code chunks: %s", e)

    try:
        res = await client.create_collection(
            collection_name=CODE_WORKSPACE_CHUNKS_COLLECTION,
            vector_size=settings.EMBEDDING_DIMENSIONS,
            distance="COSINE",
            enable_sparse=False,
        )
        return bool(res.get("success"))
    except Exception as e:
        logger.warning("ensure_code_chunks_collection failed: %s", e)
        return False


async def delete_workspace_vectors(workspace_id: str) -> None:
    from clients.vector_service_client import get_vector_service_client

    client = await get_vector_service_client(required=False)
    if not client:
        return
    try:
        await client.delete_vectors(
            CODE_WORKSPACE_CHUNKS_COLLECTION,
            [{"field": "workspace_id", "value": str(workspace_id), "operator": "equals"}],
        )
    except Exception as e:
        logger.warning("delete_workspace_vectors: %s", e)


async def embed_chunk_row(
    *,
    chunk_id: str,
    user_id: str,
    workspace_id: str,
    file_path: str,
    start_line: int,
    end_line: int,
    content: str,
    rls_context: Dict[str, str],
) -> bool:
    """Generate embedding and upsert one point; update code_chunks row."""
    from clients.vector_service_client import get_vector_service_client
    from services.database_manager.database_helpers import execute

    client = await get_vector_service_client(required=False)
    if not client:
        return False
    ok = await ensure_code_chunks_collection()
    if not ok:
        return False
    text = (content or "")[:12000]
    if not text.strip():
        return False
    try:
        vector = await client.generate_embedding(text)
    except Exception as e:
        logger.warning("embed_chunk_row generate_embedding failed: %s", e)
        return False
    payload = {
        "chunk_id": str(chunk_id),
        "user_id": str(user_id),
        "workspace_id": str(workspace_id),
        "file_path": str(file_path),
        "start_line": str(int(start_line)),
        "end_line": str(int(end_line)),
    }
    point = {"id": str(chunk_id), "vector": vector, "payload": payload}
    try:
        res = await client.upsert_vectors(CODE_WORKSPACE_CHUNKS_COLLECTION, [point])
        if not res.get("success"):
            logger.warning("upsert_vectors chunk %s: %s", chunk_id, res.get("error"))
            return False
    except Exception as e:
        logger.warning("upsert_vectors failed: %s", e)
        return False

    await execute(
        """
        UPDATE code_chunks
        SET qdrant_point_id = $3, embedding_pending = false, updated_at = NOW()
        WHERE id = $1::uuid AND user_id = $2
        """,
        chunk_id,
        user_id,
        str(chunk_id),
        rls_context=rls_context,
    )
    return True


async def embed_chunks_batch(
    rows: List[Dict[str, Any]],
    rls_context: Dict[str, str],
) -> int:
    """Embed up to len(rows) chunks sequentially (keeps vector rate modest)."""
    n = 0
    for r in rows:
        if await embed_chunk_row(
            chunk_id=str(r["id"]),
            user_id=str(r["user_id"]),
            workspace_id=str(r["workspace_id"]),
            file_path=str(r.get("file_path") or ""),
            start_line=int(r.get("start_line") or 1),
            end_line=int(r.get("end_line") or 1),
            content=str(r.get("content") or ""),
            rls_context=rls_context,
        ):
            n += 1
    return n


async def _fts_code_search(
    *,
    user_id: str,
    workspace_id: str,
    query: str,
    limit: int,
    file_glob: str,
) -> List[Dict[str, Any]]:
    from services.database_manager.database_helpers import fetch_all

    q = (query or "").strip()
    if not q:
        return []
    rls_context = {"user_id": user_id, "user_role": "user"}
    glob = (file_glob or "").strip()
    if glob:
        if "*" in glob or "?" in glob:
            like_pat = glob.replace("*", "%").replace("?", "_")
            sql = """
                SELECT c.id::text AS chunk_id, c.file_path, c.start_line, c.end_line, c.content,
                       ts_rank(c.content_tsv, plainto_tsquery('english', $3)) AS rank
                FROM code_chunks c
                WHERE c.user_id = $1 AND c.workspace_id = $2::uuid
                  AND c.content_tsv @@ plainto_tsquery('english', $3)
                  AND c.file_path ILIKE $4
                ORDER BY rank DESC
                LIMIT $5
            """
            rows = await fetch_all(sql, user_id, workspace_id, q, like_pat, limit, rls_context=rls_context)
        else:
            sql = """
                SELECT c.id::text AS chunk_id, c.file_path, c.start_line, c.end_line, c.content,
                       ts_rank(c.content_tsv, plainto_tsquery('english', $3)) AS rank
                FROM code_chunks c
                WHERE c.user_id = $1 AND c.workspace_id = $2::uuid
                  AND c.content_tsv @@ plainto_tsquery('english', $3)
                  AND c.file_path ILIKE $4
                ORDER BY rank DESC
                LIMIT $5
            """
            rows = await fetch_all(
                sql, user_id, workspace_id, q, f"%{glob}%", limit, rls_context=rls_context
            )
    else:
        sql = """
            SELECT c.id::text AS chunk_id, c.file_path, c.start_line, c.end_line, c.content,
                   ts_rank(c.content_tsv, plainto_tsquery('english', $3)) AS rank
            FROM code_chunks c
            WHERE c.user_id = $1 AND c.workspace_id = $2::uuid
              AND c.content_tsv @@ plainto_tsquery('english', $3)
            ORDER BY rank DESC
            LIMIT $4
        """
        rows = await fetch_all(sql, user_id, workspace_id, q, limit, rls_context=rls_context)

    out = []
    for row in rows or []:
        out.append(
            {
                "chunk_id": row.get("chunk_id"),
                "file_path": row.get("file_path"),
                "start_line": int(row.get("start_line") or 1),
                "end_line": int(row.get("end_line") or 1),
                "content": row.get("content") or "",
                "score": float(row.get("rank") or 0.0),
            }
        )
    return out


async def hybrid_code_search(
    *,
    user_id: str,
    workspace_id: str,
    query: str,
    limit: int,
    file_glob: str,
    similarity_threshold: float = 0.25,
) -> List[Dict[str, Any]]:
    """Vector + FTS merged by RRF; falls back to FTS-only if vector unavailable."""
    fts = await _fts_code_search(
        user_id=user_id,
        workspace_id=workspace_id,
        query=query,
        limit=limit * 3,
        file_glob=file_glob,
    )
    vec_hits: List[Dict[str, Any]] = []
    from clients.vector_service_client import get_vector_service_client

    client = await get_vector_service_client(required=False)
    if client and await ensure_code_chunks_collection():
        try:
            qv = await client.generate_embedding((query or "").strip()[:8000])
            raw = await client.search_vectors(
                CODE_WORKSPACE_CHUNKS_COLLECTION,
                qv,
                limit=limit * 3,
                score_threshold=similarity_threshold,
                filters=[
                    {"field": "workspace_id", "value": str(workspace_id), "operator": "equals"},
                    {"field": "user_id", "value": str(user_id), "operator": "equals"},
                ],
            )
            for r in raw or []:
                pl = r.get("payload") or {}
                vec_hits.append(
                    {
                        "chunk_id": pl.get("chunk_id") or r.get("id"),
                        "file_path": pl.get("file_path", ""),
                        "start_line": int(pl.get("start_line") or 1),
                        "end_line": int(pl.get("end_line") or 1),
                        "content": "",
                        "score": float(r.get("score") or 0.0),
                    }
                )
        except Exception as e:
            logger.warning("hybrid_code_search vector leg failed: %s", e)

    if not vec_hits:
        fts.sort(key=lambda x: -x["score"])
        return fts[:limit]

    merged = _merge_rrf(vec_hits, fts)
    # Load content for merged ids from DB
    from services.database_manager.database_helpers import fetch_all

    rls_context = {"user_id": user_id, "user_role": "user"}
    ids = [m.get("chunk_id") for m in merged if m.get("chunk_id")]
    if not ids:
        return []
    placeholders = ", ".join(f"${i + 3}" for i in range(len(ids)))
    sql = f"""
        SELECT id::text, file_path, start_line, end_line, content
        FROM code_chunks
        WHERE user_id = $1 AND workspace_id = $2::uuid AND id::text IN ({placeholders})
    """
    rows = await fetch_all(sql, user_id, workspace_id, *ids, rls_context=rls_context)
    by_id = {r["id"]: r for r in (rows or [])}
    out: List[Dict[str, Any]] = []
    for m in merged[:limit]:
        cid = m.get("chunk_id")
        row = by_id.get(str(cid)) if cid else None
        content = (row or {}).get("content") or m.get("content") or ""
        snippet = content[:400] + ("..." if len(content) > 400 else "")
        out.append(
            {
                "chunk_id": str(cid),
                "file_path": (row or m).get("file_path", ""),
                "start_line": int((row or m).get("start_line") or 1),
                "end_line": int((row or m).get("end_line") or 1),
                "snippet": snippet,
                "score": float(m.get("_rrf", m.get("score", 0.0))),
            }
        )
    return out
