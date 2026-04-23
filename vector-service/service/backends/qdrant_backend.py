"""
Qdrant implementation of VectorBackend.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchAny,
    MatchValue,
    PointStruct,
    Prefetch,
    SparseVector as QdrantSparseVector,
    SparseVectorParams,
    VectorParams,
)

from service.backends.base import (
    CollectionInfoOut,
    CreateCollectionInput,
    GetCollectionInfoResult,
    ScrolledPointOut,
    ScrollResult,
    SearchHit,
    SparseVectorData,
    VectorFilterInput,
    VectorPointInput,
)

logger = logging.getLogger(__name__)


def _build_search_filter(request_filters: List[VectorFilterInput]) -> Optional[Filter]:
    """Map VectorFilterInput to Qdrant Filter (Search / Scroll / Count)."""
    if not request_filters:
        return None
    must_conditions = []
    must_not_conditions = []
    for vf in request_filters:
        if vf.operator == "equals":
            must_conditions.append(
                FieldCondition(key=vf.field, match=MatchValue(value=vf.value))
            )
        elif vf.operator == "not_equals":
            must_not_conditions.append(
                FieldCondition(key=vf.field, match=MatchValue(value=vf.value))
            )
        elif vf.operator == "any_of" and vf.values:
            vals = list(vf.values)
            logger.debug(
                "Vector filter any_of field=%r values=%s",
                vf.field,
                vals[:10] if len(vals) > 10 else vals,
            )
            must_conditions.append(
                FieldCondition(key=vf.field, match=MatchAny(any=vals))
            )
        elif vf.operator == "in":
            must_conditions.append(
                FieldCondition(key=vf.field, match=MatchValue(value=vf.value))
            )
    if not must_conditions and not must_not_conditions:
        return None
    return Filter(
        must=must_conditions if must_conditions else None,
        must_not=must_not_conditions if must_not_conditions else None,
    )


def _parse_scroll_offset(offset_str: str):
    if not offset_str:
        return None
    try:
        return int(offset_str)
    except ValueError:
        return offset_str


def _serialize_scroll_offset(off) -> str:
    if off is None:
        return ""
    return str(off)


class QdrantBackend:
    """Qdrant-backed vector store."""

    def __init__(self, settings: Any) -> None:
        self._settings = settings
        self._client: Optional[QdrantClient] = None
        self._collection_vector_schema: Dict[str, str] = {}

    def initialize(self) -> None:
        if not self._settings.QDRANT_URL:
            self._client = None
            logger.warning("QDRANT_URL not set, vector store features will be unavailable")
            return
        client_kwargs: Dict[str, Any] = {
            "url": self._settings.QDRANT_URL,
            "timeout": self._settings.QDRANT_TIMEOUT,
        }
        api_key = getattr(self._settings, "QDRANT_API_KEY", None)
        if api_key and str(api_key).strip():
            client_kwargs["api_key"] = str(api_key).strip()
        self._client = QdrantClient(**client_kwargs)
        logger.info(
            "Connected to Qdrant at %s (timeout=%ss)",
            self._settings.QDRANT_URL,
            self._settings.QDRANT_TIMEOUT,
        )
        self._validate_embedding_dimensions()

    def is_configured(self) -> bool:
        return bool(self._settings.QDRANT_URL and str(self._settings.QDRANT_URL).strip())

    def is_available(self) -> bool:
        return self._client is not None

    @property
    def client(self) -> Optional[QdrantClient]:
        return self._client

    def _invalidate_collection_schema_cache(self, collection_name: str) -> None:
        self._collection_vector_schema.pop(collection_name, None)

    def _get_vectors_config_object(self, col_info: Any) -> Any:
        params = getattr(col_info.config, "params", None)
        if params is None:
            return None
        return getattr(params, "vectors", None)

    def _get_sparse_vectors_config(self, col_info: Any) -> Any:
        params = getattr(col_info.config, "params", None)
        if params is None:
            return None
        return getattr(params, "sparse_vectors", None)

    def _classify_collection_vector_schema(self, collection_name: str) -> str:
        if not self._client:
            return "unnamed"
        cached = self._collection_vector_schema.get(collection_name)
        if cached:
            return cached
        try:
            col_info = self._client.get_collection(collection_name)
        except Exception:
            self._collection_vector_schema[collection_name] = "unnamed"
            return "unnamed"
        vecs = self._get_vectors_config_object(col_info)
        if isinstance(vecs, dict) and "dense" in vecs:
            sparse = self._get_sparse_vectors_config(col_info)
            has_sparse = isinstance(sparse, dict) and len(sparse) > 0
            if has_sparse:
                self._collection_vector_schema[collection_name] = "named_hybrid"
                return "named_hybrid"
            self._collection_vector_schema[collection_name] = "named_dense"
            return "named_dense"
        self._collection_vector_schema[collection_name] = "unnamed"
        return "unnamed"

    def _dense_vector_size_and_distance(self, col_info: Any) -> Tuple[int, str]:
        vecs = self._get_vectors_config_object(col_info)
        if isinstance(vecs, dict):
            dp = vecs.get("dense")
            if dp is None:
                return 0, "COSINE"
            size = int(getattr(dp, "size", 0) or 0)
            dist = str(getattr(dp, "distance", Distance.COSINE))
            return size, dist
        if vecs is not None and hasattr(vecs, "size"):
            size = int(getattr(vecs, "size", 0) or 0)
            dist = str(getattr(vecs, "distance", Distance.COSINE))
            return size, dist
        return 0, "COSINE"

    def _ensure_collection_exists(self, collection_name: str) -> None:
        """Ensure a Qdrant collection exists (embedding-dimensions path; currently unused by RPCs)."""
        if not self._client:
            return
        try:
            collections = self._client.get_collections()
            collection_names = [c.name for c in collections.collections]
            dimensions = self._settings.EMBEDDING_DIMENSIONS
            hybrid = getattr(self._settings, "HYBRID_SEARCH_ENABLED", False)

            def _create_named_hybrid() -> None:
                self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "dense": VectorParams(
                            size=dimensions,
                            distance=Distance.COSINE,
                        )
                    },
                    sparse_vectors_config={"sparse": SparseVectorParams()},
                )
                self._collection_vector_schema[collection_name] = "named_hybrid"
                logger.info(
                    "Collection %r created with named dense + sparse vectors (%s dimensions)",
                    collection_name,
                    dimensions,
                )

            def _create_unnamed_dense() -> None:
                self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=dimensions,
                        distance=Distance.COSINE,
                    ),
                )
                self._collection_vector_schema[collection_name] = "unnamed"
                logger.info(
                    "Collection %r created with %s dimensions (unnamed dense)",
                    collection_name,
                    dimensions,
                )

            if collection_name not in collection_names:
                logger.info("Creating collection %r in Qdrant", collection_name)
                if hybrid:
                    _create_named_hybrid()
                else:
                    _create_unnamed_dense()
                return

            needs_recreate = False

            if hybrid:
                schema = self._classify_collection_vector_schema(collection_name)
                if schema != "named_hybrid":
                    logger.warning(
                        "Recreating collection %r for BM25 hybrid schema (previous schema=%s)",
                        collection_name,
                        schema,
                    )
                    needs_recreate = True

            col_info = self._client.get_collection(collection_name)
            size, _ = self._dense_vector_size_and_distance(col_info)
            if size and size != dimensions:
                logger.warning(
                    "Dimension mismatch for %r: existing=%d, configured=%d. Recreating.",
                    collection_name,
                    size,
                    dimensions,
                )
                needs_recreate = True

            if needs_recreate:
                self._client.delete_collection(collection_name=collection_name)
                self._invalidate_collection_schema_cache(collection_name)
                if hybrid:
                    _create_named_hybrid()
                else:
                    _create_unnamed_dense()
            else:
                logger.debug("Collection %r already exists", collection_name)
        except Exception as e:
            logger.error("Failed to ensure collection %r exists: %s", collection_name, e)

    def _validate_embedding_dimensions(self) -> None:
        if not self._client:
            return
        document_collection_suffixes = ("_documents",)
        document_collection_names = ("documents", "skills", "help_docs")
        try:
            collections = self._client.get_collections()
            configured = self._settings.EMBEDDING_DIMENSIONS
            for col in collections.collections:
                name = col.name
                if "face_encodings" in name or "object_features" in name:
                    continue
                is_team = name.startswith("team_")
                is_doc_like = (
                    name.endswith(document_collection_suffixes)
                    or name in document_collection_names
                    or is_team
                )
                if not is_doc_like:
                    continue
                try:
                    col_info = self._client.get_collection(name)
                    size, _dist = self._dense_vector_size_and_distance(col_info)
                    if size and size != configured:
                        logger.error(
                            "Embedding dimension mismatch: collection %r has vector_size=%s "
                            "but EMBEDDING_DIMENSIONS=%s. Re-index documents with the current "
                            "embedding model or set EMBEDDING_DIMENSIONS=%s to match existing data.",
                            name,
                            size,
                            configured,
                            size,
                        )
                except Exception as e:
                    logger.debug("Could not check collection %s dimensions: %s", name, e)
        except Exception as e:
            logger.warning("Could not validate embedding dimensions against Qdrant: %s", e)

    def _ensure_collection_for_vectors(
        self,
        collection_name: str,
        vector_size: int,
        wants_named_hybrid: bool,
    ) -> None:
        if not self._client:
            raise RuntimeError("Qdrant client not initialized")
        try:
            collections = self._client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if collection_name not in collection_names:
                if wants_named_hybrid:
                    logger.info(
                        "Creating collection %r with named dense + sparse vectors (%s dimensions)",
                        collection_name,
                        vector_size,
                    )
                    self._client.create_collection(
                        collection_name=collection_name,
                        vectors_config={
                            "dense": VectorParams(
                                size=vector_size,
                                distance=Distance.COSINE,
                            )
                        },
                        sparse_vectors_config={"sparse": SparseVectorParams()},
                    )
                    self._collection_vector_schema[collection_name] = "named_hybrid"
                else:
                    logger.info(
                        "Creating collection %r with %s dimensions (unnamed vector)",
                        collection_name,
                        vector_size,
                    )
                    self._client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=vector_size,
                            distance=Distance.COSINE,
                        ),
                    )
                    self._collection_vector_schema[collection_name] = "unnamed"
                return

            schema = self._classify_collection_vector_schema(collection_name)
            if wants_named_hybrid and schema == "unnamed":
                raise ValueError(
                    f"Collection '{collection_name}' uses single-vector schema; cannot upsert sparse vectors. "
                    "Delete the collection and recreate with enable_sparse=True, or use a new collection name."
                )
        except Exception as e:
            logger.error("Failed to ensure collection %r exists: %s", collection_name, e)
            raise

    def upsert_vectors(self, collection_name: str, points: List[VectorPointInput]) -> int:
        if not self._client:
            raise RuntimeError("Qdrant not configured")
        any_sparse = any(p.sparse is not None for p in points)
        vector_size = len(points[0].vector) if points[0].vector else 128
        self._ensure_collection_for_vectors(
            collection_name=collection_name,
            vector_size=vector_size,
            wants_named_hybrid=any_sparse,
        )
        schema = self._classify_collection_vector_schema(collection_name)
        qdrant_points = []
        for p in points:
            point_id: Union[int, str] = p.id
            try:
                point_id = int(p.id)
            except (ValueError, TypeError):
                pass
            if schema in ("named_dense", "named_hybrid"):
                vec_map: Dict[str, Any] = {"dense": list(p.vector)}
                if p.sparse is not None:
                    vec_map["sparse"] = QdrantSparseVector(
                        indices=list(p.sparse.indices),
                        values=list(p.sparse.values),
                    )
                qdrant_points.append(
                    PointStruct(id=point_id, vector=vec_map, payload=p.payload)
                )
            else:
                qdrant_points.append(
                    PointStruct(
                        id=point_id,
                        vector=list(p.vector),
                        payload=p.payload,
                    )
                )
        batch_size = 100
        total_stored = 0
        max_retries = getattr(self._settings, "QDRANT_UPSERT_MAX_RETRIES", 3)
        for i in range(0, len(qdrant_points), batch_size):
            batch = qdrant_points[i : i + batch_size]
            for attempt in range(max_retries):
                try:
                    self._client.upsert(
                        collection_name=collection_name,
                        points=batch,
                    )
                    total_stored += len(batch)
                    break
                except Exception as batch_err:
                    err_str = str(batch_err).lower()
                    status_code = getattr(batch_err, "status_code", None)
                    is_timeout_or_connection = (
                        "timeout" in err_str
                        or "timed out" in err_str
                        or "connection" in err_str
                        or "read" in err_str
                    )
                    is_collection_not_ready = (
                        isinstance(batch_err, UnexpectedResponse) and status_code == 404
                    ) or (
                        "not found" in err_str and "collection" in err_str
                    ) or "doesn't exist" in err_str
                    is_server_error = (
                        isinstance(batch_err, UnexpectedResponse)
                        and status_code in (500, 503)
                    ) or "500" in err_str or "503" in err_str or "internal" in err_str
                    is_retryable = (
                        is_timeout_or_connection
                        or is_collection_not_ready
                        or is_server_error
                    )
                    if is_retryable and attempt < max_retries - 1:
                        wait_time = 2**attempt
                        if is_collection_not_ready:
                            logger.warning(
                                "Qdrant collection not ready on all shards (attempt %s/%s); "
                                "retrying in %ss (cluster propagation)",
                                attempt + 1,
                                max_retries,
                                wait_time,
                            )
                        elif is_server_error:
                            logger.warning(
                                "Qdrant server error (attempt %s/%s); "
                                "retrying in %ss (shard/replica may be catching up)",
                                attempt + 1,
                                max_retries,
                                wait_time,
                            )
                        else:
                            logger.warning(
                                "Qdrant upsert batch failed (attempt %s/%s): %s; will retry in %ss",
                                attempt + 1,
                                max_retries,
                                batch_err,
                                wait_time,
                            )
                        time.sleep(wait_time)
                    else:
                        raise
        logger.info("Upserted %s vectors to collection %r", total_stored, collection_name)
        return total_stored

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
        if not self._client:
            return []
        query_filter = _build_search_filter(filters)
        collections = self._client.get_collections()
        collection_names = [c.name for c in collections.collections]
        if collection_name not in collection_names:
            logger.info(
                "SearchVectors: Collection %r doesn't exist, returning empty results",
                collection_name,
            )
            return []
        schema = self._classify_collection_vector_schema(collection_name)
        fusion = (fusion_mode or "").strip().lower()
        use_rrf = (
            sparse_query is not None
            and fusion == "rrf"
            and schema == "named_hybrid"
        )
        if sparse_query is not None and fusion == "rrf" and schema != "named_hybrid":
            logger.warning(
                "SearchVectors: RRF requested but collection %r is not named_hybrid (schema=%s); "
                "using dense-only search",
                collection_name,
                schema,
            )
        effective_threshold = score_threshold if score_threshold > 0 else 0.0
        try:
            if use_rrf and sparse_query is not None:
                prefetch_limit = max((limit or 50) * 3, 100)
                qr = self._client.query_points(
                    collection_name=collection_name,
                    prefetch=[
                        Prefetch(
                            query=list(query_vector),
                            using="dense",
                            limit=prefetch_limit,
                        ),
                        Prefetch(
                            query=QdrantSparseVector(
                                indices=list(sparse_query.indices),
                                values=list(sparse_query.values),
                            ),
                            using="sparse",
                            limit=prefetch_limit,
                        ),
                    ],
                    query=FusionQuery(fusion=Fusion.RRF),
                    query_filter=query_filter,
                    limit=limit or 50,
                )
                search_results = qr.points
            else:
                # qdrant-client >=1.16: use query_points instead of removed client.search()
                query_kwargs: Dict[str, Any] = {
                    "collection_name": collection_name,
                    "query": list(query_vector),
                    "limit": limit or 50,
                    "query_filter": query_filter,
                    "score_threshold": effective_threshold,
                }
                if schema in ("named_dense", "named_hybrid"):
                    query_kwargs["using"] = "dense"
                qr = self._client.query_points(**query_kwargs)
                search_results = qr.points
        except UnexpectedResponse as e:
            if e.status_code == 404 or "doesn't exist" in str(e).lower() or "not found" in str(e).lower():
                logger.info(
                    "SearchVectors: Collection %r not found during search, returning empty results",
                    collection_name,
                )
                return []
            raise
        hits = []
        for hit in search_results:
            hits.append(SearchHit(id=str(hit.id), score=hit.score, payload=dict(hit.payload or {})))
        logger.info(
            "SearchVectors: Found %s results in %r",
            len(hits),
            collection_name,
        )
        if len(hits) == 0 and filters:
            try:
                col_info = self._client.get_collection(collection_name)
                pts = getattr(col_info, "points_count", None)
                logger.info(
                    "SearchVectors: 0 results for collection %s (points_count=%s); filter was applied",
                    collection_name,
                    pts,
                )
            except Exception as ex:
                logger.debug("SearchVectors: could not get collection info: %s", ex)
        return hits

    def scroll_points(
        self,
        collection_name: str,
        filters: List[VectorFilterInput],
        limit: int,
        offset: str,
        with_vectors: bool,
    ) -> ScrollResult:
        if not self._client:
            return ScrollResult(points=[], next_offset="")
        collections = self._client.get_collections()
        collection_names = [c.name for c in collections.collections]
        if collection_name not in collection_names:
            return ScrollResult(points=[], next_offset="")
        query_filter = _build_search_filter(filters)
        page_limit = limit or 256
        page_limit = max(1, min(int(page_limit), 8192))
        off = _parse_scroll_offset((offset or "").strip())
        try:
            records, next_page = self._client.scroll(
                collection_name=collection_name,
                scroll_filter=query_filter,
                limit=page_limit,
                offset=off,
                with_payload=True,
                with_vectors=with_vectors,
            )
        except UnexpectedResponse as e:
            if e.status_code == 404 or "doesn't exist" in str(e).lower() or "not found" in str(e).lower():
                return ScrollResult(points=[], next_offset="")
            raise
        out_points = []
        for pt in records or []:
            pl = getattr(pt, "payload", None) or {}
            out_points.append(ScrolledPointOut(id=str(pt.id), payload=dict(pl)))
        return ScrollResult(
            points=out_points,
            next_offset=_serialize_scroll_offset(next_page),
        )

    def count_vectors(self, collection_name: str, filters: List[VectorFilterInput]) -> int:
        if not self._client:
            return 0
        collections = self._client.get_collections()
        collection_names = [c.name for c in collections.collections]
        if collection_name not in collection_names:
            return 0
        query_filter = _build_search_filter(filters)
        try:
            cnt_result = self._client.count(
                collection_name=collection_name,
                count_filter=query_filter,
                exact=True,
            )
            return int(getattr(cnt_result, "count", 0) or 0)
        except UnexpectedResponse as e:
            if e.status_code == 404 or "doesn't exist" in str(e).lower() or "not found" in str(e).lower():
                return 0
            raise

    def delete_vectors_equality(
        self, collection_name: str, filters: List[VectorFilterInput]
    ) -> Tuple[int, Optional[str]]:
        if not self._client:
            return 0, "Service or Qdrant not initialized"
        if not filters:
            return 0, "No filters provided - use DeleteCollection for full deletion"
        filter_conditions = []
        for vf in filters:
            filter_conditions.append(
                FieldCondition(key=vf.field, match=MatchValue(value=vf.value))
            )
        query_filter = Filter(must=filter_conditions)
        delete_result = self._client.delete(
            collection_name=collection_name,
            points_selector=query_filter,
        )
        deleted_count = 0
        if hasattr(delete_result, "operation_id"):
            deleted_count = 1
        logger.info(
            "DeleteVectors: Deleted vectors from %r with filters",
            collection_name,
        )
        return deleted_count, None

    def update_metadata_equality(
        self,
        collection_name: str,
        filters: List[VectorFilterInput],
        metadata_updates: Dict[str, Any],
    ) -> Tuple[int, Optional[str]]:
        if not self._client:
            return 0, "Service or Qdrant not initialized"
        filter_conditions = []
        for vf in filters:
            filter_conditions.append(
                FieldCondition(key=vf.field, match=MatchValue(value=vf.value))
            )
        query_filter = Filter(must=filter_conditions)
        scroll_result = self._client.scroll(
            collection_name=collection_name,
            scroll_filter=query_filter,
            limit=10000,
            with_vectors=True,
        )
        updated_count = 0
        if scroll_result and scroll_result[0]:
            points_to_update = []
            for point in scroll_result[0]:
                new_payload = point.payload.copy()
                new_payload.update(metadata_updates)
                points_to_update.append(
                    PointStruct(
                        id=point.id,
                        vector=point.vector,
                        payload=new_payload,
                    )
                )
            if points_to_update:
                self._client.upsert(
                    collection_name=collection_name,
                    points=points_to_update,
                )
                updated_count = len(points_to_update)
        logger.info(
            "UpdateVectorMetadata: Updated %s vectors in %r",
            updated_count,
            collection_name,
        )
        return updated_count, None

    def create_collection(self, spec: CreateCollectionInput) -> Optional[str]:
        if not self._client:
            return "Service or Qdrant not initialized"
        try:
            collections = self._client.get_collections()
            collection_names = [c.name for c in collections.collections]
            if spec.collection_name in collection_names:
                self._invalidate_collection_schema_cache(spec.collection_name)
                logger.info(
                    "Collection '%s' already exists (idempotent success)",
                    spec.collection_name,
                )
                return None
            distance_map = {
                "COSINE": Distance.COSINE,
                "DOT": Distance.DOT,
                "EUCLIDEAN": Distance.EUCLID,
                "EUCLID": Distance.EUCLID,
            }
            distance_str = (spec.distance or "").strip().upper() or "COSINE"
            distance = distance_map.get(distance_str, Distance.COSINE)
            if spec.enable_sparse:
                self._client.create_collection(
                    collection_name=spec.collection_name,
                    vectors_config={
                        "dense": VectorParams(
                            size=spec.vector_size,
                            distance=distance,
                        )
                    },
                    sparse_vectors_config={"sparse": SparseVectorParams()},
                )
                self._collection_vector_schema[spec.collection_name] = "named_hybrid"
                logger.info(
                    "Created collection '%s' named_hybrid: dense=%s dims, distance=%s",
                    spec.collection_name,
                    spec.vector_size,
                    distance_str,
                )
            else:
                self._client.create_collection(
                    collection_name=spec.collection_name,
                    vectors_config=VectorParams(
                        size=spec.vector_size,
                        distance=distance,
                    ),
                )
                self._collection_vector_schema[spec.collection_name] = "unnamed"
                logger.info(
                    "Created collection '%s' with %s dimensions, distance=%s",
                    spec.collection_name,
                    spec.vector_size,
                    distance_str,
                )
            return None
        except Exception as e:
            logger.error("CreateCollection failed: %s: %s", type(e).__name__, e)
            return str(e)

    def delete_collection(self, collection_name: str) -> Optional[str]:
        if not self._client:
            return "Service or Qdrant not initialized"
        self._client.delete_collection(collection_name=collection_name)
        self._invalidate_collection_schema_cache(collection_name)
        logger.info("Deleted collection %r", collection_name)
        return None

    def list_collections(self) -> Tuple[List[CollectionInfoOut], Optional[str]]:
        if not self._client:
            return [], "Service or Qdrant not initialized"
        try:
            collections = self._client.get_collections()
            collection_infos: List[CollectionInfoOut] = []
            for col in collections.collections:
                try:
                    col_info = self._client.get_collection(col.name)
                    points_count = (
                        col_info.points_count if hasattr(col_info, "points_count") else 0
                    )
                    vector_size, distance = self._dense_vector_size_and_distance(col_info)
                    schema_type = self._classify_collection_vector_schema(col.name)
                except Exception:
                    points_count = 0
                    vector_size = 0
                    distance = "COSINE"
                    schema_type = "unnamed"
                collection_infos.append(
                    CollectionInfoOut(
                        name=col.name,
                        vector_size=vector_size,
                        distance=distance,
                        points_count=points_count,
                        status="green",
                        schema_type=schema_type,
                    )
                )
            return collection_infos, None
        except Exception as e:
            logger.error("ListCollections failed: %s", e)
            return [], str(e)

    def get_collection_info(self, collection_name: str) -> GetCollectionInfoResult:
        if not self._client:
            return GetCollectionInfoResult(
                success=False,
                error="Service or Qdrant not initialized",
            )
        collections = self._client.get_collections()
        collection_names = [c.name for c in collections.collections]
        if collection_name not in collection_names:
            logger.info(
                "GetCollectionInfo: Collection %r doesn't exist",
                collection_name,
            )
            return GetCollectionInfoResult(
                success=False,
                error=f"Collection '{collection_name}' doesn't exist",
            )
        try:
            col_info = self._client.get_collection(collection_name)
            vector_size, distance = self._dense_vector_size_and_distance(col_info)
            points_count = (
                col_info.points_count if hasattr(col_info, "points_count") else 0
            )
            schema_type = self._classify_collection_vector_schema(collection_name)
            info = CollectionInfoOut(
                name=collection_name,
                vector_size=vector_size,
                distance=distance,
                points_count=points_count,
                status="green",
                schema_type=schema_type,
            )
            return GetCollectionInfoResult(success=True, collection=info)
        except Exception as e:
            logger.error("GetCollectionInfo failed: %s", e)
            err = str(e)
            if (
                "404" in err
                or "doesn't exist" in err.lower()
                or "not found" in err.lower()
            ):
                return GetCollectionInfoResult(
                    success=False,
                    error=f"Collection '{collection_name}' doesn't exist",
                )
            return GetCollectionInfoResult(success=False, error=err)
