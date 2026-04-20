"""
Instantiate the configured vector store backend.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from service.backends.base import VectorBackend


def get_vector_backend(settings) -> "VectorBackend":
    backend = (getattr(settings, "VECTOR_DB_BACKEND", None) or "qdrant").strip().lower()
    if backend == "qdrant":
        from service.backends.qdrant_backend import QdrantBackend

        return QdrantBackend(settings)
    if backend == "milvus":
        from service.backends.milvus_backend import MilvusBackend

        return MilvusBackend(settings)
    if backend == "elasticsearch":
        from service.backends.es_backend import ElasticsearchBackend

        return ElasticsearchBackend(settings)
    raise ValueError(
        f"Unsupported VECTOR_DB_BACKEND={backend!r}; "
        "supported values: 'qdrant', 'milvus', 'elasticsearch'."
    )
