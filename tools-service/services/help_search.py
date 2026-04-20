"""
Help documentation vector search for ToolService.

Used by gRPC SearchHelpDocs (handler maps dicts to protobuf). Imports only leaf
backend services — not grpc_handlers (avoids circular imports).
"""

from typing import Any, Dict, List, Optional


async def search_help_docs(
    *,
    query: str,
    limit: int,
    embedding_manager: Optional[Any],
) -> List[Dict[str, Any]]:
    """
    Search the help_docs Qdrant collection by semantic similarity.

    Args:
        query: Non-empty search string (caller may strip).
        limit: Max results (positive).
        embedding_manager: ToolService embedding wrapper; if falsy, returns [].

    Returns:
        List of dicts with keys: topic_id, title, content, score.
    """
    q = (query or "").strip()
    if not q:
        return []
    if not embedding_manager:
        return []

    embeddings = await embedding_manager.generate_embeddings([q])
    if not embeddings or len(embeddings) == 0:
        return []

    from services.vector_store_service import get_vector_store

    vector_store = await get_vector_store()
    if not vector_store.is_vector_available():
        return []

    raw = await vector_store.search_similar(
        query_embedding=embeddings[0],
        collection_name="help_docs",
        limit=limit,
        score_threshold=0.6,
        query_text=q,
    )

    out: List[Dict[str, Any]] = []
    for r in raw or []:
        content = (r.get("content") or "").strip()
        if not content:
            continue
        topic_id = r.get("document_id") or ""
        title = (r.get("metadata") or {}).get("title") or topic_id.replace("-", " ").title()
        score = float(r.get("score") or 0.0)
        out.append(
            {
                "topic_id": topic_id,
                "title": title,
                "content": content[:8000],
                "score": score,
            }
        )
    return out
