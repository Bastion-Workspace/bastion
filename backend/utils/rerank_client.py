"""
Rerank Client

Calls the OpenRouter /api/v1/rerank endpoint to reorder a list of document
chunks by cross-encoder relevance to a query. Cross-encoder models evaluate
query + document together, giving significantly more precise relevance scores
than the bi-encoder cosine similarity used for initial retrieval.

Supported model: cohere/rerank-4-pro (32K context, multilingual, $0/M tokens on OpenRouter)
"""

import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_RERANK_URL = "https://openrouter.ai/api/v1/rerank"
DEFAULT_RERANK_TIMEOUT = 15.0


async def call_rerank_api(
    query: str,
    documents: List[str],
    top_n: int = 10,
    model: str = "cohere/rerank-4-pro",
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Call the OpenRouter rerank API and return results ordered by relevance.

    Args:
        query: The search query or question used to rank documents.
        documents: List of text chunks to rerank.
        top_n: Maximum number of results to return (highest-relevance first).
        model: OpenRouter rerank model identifier.
        api_key: OpenRouter API key. Falls back to settings.OPENROUTER_API_KEY.

    Returns:
        List of result dicts: [{"index": int, "relevance_score": float, "document": str}, ...]
        Ordered by relevance_score descending.

    Raises:
        httpx.HTTPStatusError: on non-2xx response (callers should catch and degrade).
        RuntimeError: if the API key is not configured.
    """
    if api_key is None:
        from config import settings
        api_key = settings.OPENROUTER_API_KEY

    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured; cannot call rerank API")

    if not documents:
        return []

    # Clamp top_n to available documents
    top_n = min(top_n, len(documents))

    payload = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": top_n,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=DEFAULT_RERANK_TIMEOUT) as client:
        response = await client.post(OPENROUTER_RERANK_URL, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

    raw_results = data.get("results", [])

    # Normalize: OpenRouter wraps document text in {"document": {"text": "..."}}
    normalized = []
    for r in raw_results:
        doc_field = r.get("document", "")
        if isinstance(doc_field, dict):
            doc_text = doc_field.get("text", "")
        else:
            doc_text = str(doc_field)
        normalized.append({
            "index": r.get("index", 0),
            "relevance_score": float(r.get("relevance_score", 0.0)),
            "document": doc_text,
        })

    return normalized
