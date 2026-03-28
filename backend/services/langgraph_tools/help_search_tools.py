"""
Help Search Tools - Search app help documentation (vectorized help_docs collection).
Used by CentralizedToolRegistry for built-in agents. Agent Factory uses the gRPC tool.
"""

import logging
from typing import Optional

from services.embedding_service_wrapper import get_embedding_service
from services.vector_store_service import VectorStoreService

logger = logging.getLogger(__name__)

HELP_DOCS_COLLECTION = "help_docs"


async def search_help_docs(
    query: str,
    user_id: Optional[str] = None,
    limit: int = 5,
) -> str:
    """
    Search app help documentation for how-to questions about Bastion features.
    Use when the user asks how to do something in the app, what a feature does, or where to find something.
    """
    try:
        query = (query or "").strip()
        if not query:
            return "No search query provided."
        embedding_service = await get_embedding_service()
        embeddings = await embedding_service.generate_embeddings([query])
        if not embeddings or len(embeddings) == 0:
            return "Could not generate query embedding."
        vector_store = VectorStoreService()
        await vector_store.initialize()
        results = await vector_store.search_similar(
            query_embedding=embeddings[0],
            collection_name=HELP_DOCS_COLLECTION,
            limit=limit,
            score_threshold=0.6,
        )
        if not results:
            return "No help topics matched your query. Try rephrasing or ask about a specific feature."
        parts = []
        for i, r in enumerate(results, 1):
            content = (r.get("content") or "").strip()
            if not content:
                continue
            topic_id = r.get("document_id") or ""
            title = (r.get("metadata") or {}).get("title") or topic_id.replace("-", " ").title()
            score = r.get("score")
            parts.append(f"**{i}. {title}** (topic: {topic_id})")
            if score is not None:
                parts.append(f"Relevance: {score:.2f}")
            parts.append(content[:2000])
            parts.append("")
        return "\n".join(parts).strip()
    except Exception as e:
        logger.warning("search_help_docs failed: %s", e)
        return "Help documentation search is temporarily unavailable."
