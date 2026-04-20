"""
Vector store backend protocol and domain types (protobuf-free).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple


@dataclass
class SparseVectorData:
    indices: List[int]
    values: List[float]


@dataclass
class VectorFilterInput:
    field: str
    value: str
    operator: str
    values: List[str] = field(default_factory=list)


@dataclass
class VectorPointInput:
    """One point to upsert (payload values are string map from proto, parsed upstream)."""

    id: str
    vector: List[float]
    payload: Dict[str, Any]
    sparse: Optional[SparseVectorData] = None


@dataclass
class SearchHit:
    id: str
    score: float
    payload: Dict[str, Any]


@dataclass
class ScrolledPointOut:
    id: str
    payload: Dict[str, Any]


@dataclass
class ScrollResult:
    points: List[ScrolledPointOut]
    next_offset: str


@dataclass
class CollectionInfoOut:
    name: str
    vector_size: int
    distance: str
    points_count: int
    status: str
    schema_type: str


@dataclass
class GetCollectionInfoResult:
    success: bool
    collection: Optional[CollectionInfoOut] = None
    error: Optional[str] = None


@dataclass
class CreateCollectionInput:
    collection_name: str
    vector_size: int
    distance: str
    enable_sparse: bool


class VectorBackend(Protocol):
    """Abstract vector store used by the gRPC servicer."""

    def initialize(self) -> None: ...

    def is_configured(self) -> bool: ...

    def is_available(self) -> bool:
        """True when the backend can execute vector operations (connected / ready)."""
        ...

    def upsert_vectors(self, collection_name: str, points: List[VectorPointInput]) -> int:
        """Upsert points; raises ValueError on schema mismatch."""

    def search_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int,
        score_threshold: float,
        filters: List[VectorFilterInput],
        sparse_query: Optional[SparseVectorData],
        fusion_mode: str,
    ) -> List[SearchHit]:
        """Return hits; empty list if collection missing or 404 during search."""

    def scroll_points(
        self,
        collection_name: str,
        filters: List[VectorFilterInput],
        limit: int,
        offset: str,
        with_vectors: bool,
    ) -> ScrollResult:
        """Paginated scroll; empty page if collection missing."""

    def count_vectors(
        self, collection_name: str, filters: List[VectorFilterInput]
    ) -> int:
        """Exact count; 0 if collection missing."""

    def delete_vectors_equality(
        self, collection_name: str, filters: List[VectorFilterInput]
    ) -> Tuple[int, Optional[str]]:
        """Equality-only filters. Returns (points_deleted, error_message)."""

    def update_metadata_equality(
        self,
        collection_name: str,
        filters: List[VectorFilterInput],
        metadata_updates: Dict[str, Any],
    ) -> Tuple[int, Optional[str]]:
        """Equality-only filters; returns (points_updated, error_message)."""

    def create_collection(self, spec: CreateCollectionInput) -> Optional[str]:
        """Returns error string on failure, None on success."""

    def delete_collection(self, collection_name: str) -> Optional[str]:
        """Returns error string on failure, None on success."""

    def list_collections(self) -> Tuple[List[CollectionInfoOut], Optional[str]]:
        """Returns (collections, error_message)."""

    def get_collection_info(self, collection_name: str) -> GetCollectionInfoResult: ...

