"""
Primary chunk + vector indexing eligibility.

Bump APP_CHUNK_INDEX_SCHEMA_VERSION when chunking, embedding, or storage contracts change
so existing rows are re-indexed on the next backfill or default reprocess.
"""

from __future__ import annotations

from typing import FrozenSet, Optional, Tuple

# Increment when chunk/Qdrant/Postgres chunk contract changes (forces re-index for stale rows).
APP_CHUNK_INDEX_SCHEMA_VERSION: int = 1

# DocumentType values that produce primary body chunks in the current pipeline.
CHUNK_INDEX_ELIGIBLE_DOC_TYPES: FrozenSet[str] = frozenset(
    {
        "pdf",
        "epub",
        "txt",
        "md",
        "markdown",
        "text",
        "docx",
        "pptx",
        "html",
        "eml",
        "srt",
        "vtt",
        "image_sidecar",
    }
)


def is_chunk_index_eligible(doc_type: Optional[str], is_zip_container: Optional[bool]) -> bool:
    """Return True if this row may receive primary document chunk + vector indexing."""
    if not doc_type:
        return False
    dt = str(doc_type).strip().lower()
    if dt == "zip" and bool(is_zip_container):
        return False
    return dt in CHUNK_INDEX_ELIGIBLE_DOC_TYPES


def sql_eligible_doc_types_tuple() -> Tuple[str, ...]:
    """Stable tuple for PostgreSQL = ANY($1::text[])."""
    return tuple(sorted(CHUNK_INDEX_ELIGIBLE_DOC_TYPES))
