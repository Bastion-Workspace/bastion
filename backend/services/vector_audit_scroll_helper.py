"""
Paginated Qdrant scroll via vector-service (payload-only) for document vector audits.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

from clients.vector_service_client import get_vector_service_client

logger = logging.getLogger(__name__)


async def collect_document_ids_from_collection(
    collection_name: str,
    *,
    filters: Optional[List[Dict]] = None,
    page_size: int = 512,
) -> Tuple[Set[str], Optional[str]]:
    """Return distinct document_id values found in point payloads for one collection."""
    client = await get_vector_service_client(required=True)
    seen: Set[str] = set()
    offset: Optional[str] = None

    while True:
        res = await client.scroll_points(
            collection_name=collection_name,
            filters=filters or [],
            limit=page_size,
            offset=offset,
            with_vectors=False,
        )
        if not res.get("success"):
            return seen, res.get("error") or "scroll_points failed"

        for p in res.get("points", []):
            did = (p.get("payload") or {}).get("document_id")
            if did is not None and str(did).strip():
                seen.add(str(did))

        next_off = (res.get("next_offset") or "").strip()
        if not next_off:
            break
        offset = next_off

    return seen, None
