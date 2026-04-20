"""Shared indexing policy for backend and document-service (single source of truth)."""

from bastion_indexing.policy import (
    APP_CHUNK_INDEX_SCHEMA_VERSION,
    CHUNK_INDEX_ELIGIBLE_DOC_TYPES,
    is_chunk_index_eligible,
    sql_eligible_doc_types_tuple,
)

__all__ = [
    "APP_CHUNK_INDEX_SCHEMA_VERSION",
    "CHUNK_INDEX_ELIGIBLE_DOC_TYPES",
    "is_chunk_index_eligible",
    "sql_eligible_doc_types_tuple",
]
