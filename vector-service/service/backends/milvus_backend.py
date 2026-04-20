"""
Milvus 2.4+ implementation of VectorBackend (dense + optional sparse, hybrid RRF).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

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

CONN_ALIAS = "bastion_vector"
ID_MAX_LEN = 64
DENSE_FIELD = "dense"
SPARSE_FIELD = "sparse"
PAYLOAD_FIELD = "payload"
PK_FIELD = "id"
# Milvus default max offset for query; beyond this use iterator skip
_MILVUS_MAX_OFFSET = 16384


def _sparse_to_dict(sp: SparseVectorData) -> Dict[int, float]:
    return {int(i): float(v) for i, v in zip(sp.indices, sp.values)}


def _escape_str(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _milvus_expr_from_filters(
    filters: List[VectorFilterInput],
    mode: str,
) -> Optional[str]:
    """
    Build Milvus boolean expr over JSON `payload` field.
    mode: "search" (equals, not_equals, any_of, in) or "equality" (equals only via value).
    """
    if not filters:
        return None
    parts: List[str] = []
    must_not: List[str] = []
    for vf in filters:
        key = _escape_str(vf.field)
        if mode == "equality":
            val = _escape_str(vf.value)
            parts.append(f'{PAYLOAD_FIELD}["{key}"] == "{val}"')
            continue
        if vf.operator == "equals":
            val = _escape_str(vf.value)
            parts.append(f'{PAYLOAD_FIELD}["{key}"] == "{val}"')
        elif vf.operator == "not_equals":
            val = _escape_str(vf.value)
            must_not.append(f'{PAYLOAD_FIELD}["{key}"] == "{val}"')
        elif vf.operator == "any_of" and vf.values:
            escaped = [f'"{_escape_str(v)}"' for v in vf.values]
            inner = ", ".join(escaped)
            parts.append(f'{PAYLOAD_FIELD}["{key}"] in [{inner}]')
        elif vf.operator == "in":
            val = _escape_str(vf.value)
            parts.append(f'{PAYLOAD_FIELD}["{key}"] == "{val}"')
    for n in must_not:
        parts.append(f"not ({n})")
    if not parts:
        return None
    return " and ".join(f"({p})" for p in parts)


def _distance_to_milvus_metric(distance_str: str) -> str:
    d = (distance_str or "").strip().upper()
    if d in ("DOT", "IP"):
        return "IP"
    if d in ("EUCLIDEAN", "EUCLID", "L2"):
        return "L2"
    return "COSINE"


def _metric_to_proto_distance(metric: str) -> str:
    m = (metric or "COSINE").upper()
    if m == "IP":
        return "DOT"
    if m == "L2":
        return "EUCLIDEAN"
    return "COSINE"


class MilvusBackend:
    """Milvus-backed vector store (pymilvus ORM)."""

    def __init__(self, settings: Any) -> None:
        self._settings = settings
        self._connected = False
        # collection_name -> "hybrid" | "dense_only"
        self._schema_kind: Dict[str, str] = {}
        # collection_name -> dense metric at index time
        self._dense_metric: Dict[str, str] = {}

    def initialize(self) -> None:
        if not self.is_configured():
            logger.warning("MILVUS_URI not set, vector store features will be unavailable")
            self._connected = False
            return
        from pymilvus import connections

        kwargs: Dict[str, Any] = {
            "alias": CONN_ALIAS,
            "uri": self._settings.MILVUS_URI,
        }
        tok = getattr(self._settings, "MILVUS_TOKEN", None)
        if tok and str(tok).strip():
            kwargs["token"] = str(tok).strip()
        try:
            connections.connect(**kwargs)
        except Exception as e:
            logger.error("Milvus connect failed: %s", e)
            self._connected = False
            return
        dbn = getattr(self._settings, "MILVUS_DB_NAME", "default") or "default"
        if dbn != "default":
            try:
                from pymilvus import db

                db.using_database(dbn)
            except Exception as ex:
                logger.warning("Milvus using_database(%r) failed: %s", dbn, ex)
        self._connected = True
        logger.info("Connected to Milvus at %s", self._settings.MILVUS_URI)

    def is_configured(self) -> bool:
        return bool(getattr(self._settings, "MILVUS_URI", "") or "").strip()

    def is_available(self) -> bool:
        return self._connected

    def _invalidate_cache(self, name: str) -> None:
        self._schema_kind.pop(name, None)
        self._dense_metric.pop(name, None)

    @staticmethod
    def _dense_field_dim(col: Any) -> int:
        for f in col.schema.fields:
            if f.name != DENSE_FIELD:
                continue
            d = getattr(f, "dim", None)
            if d is not None:
                return int(d)
            params = getattr(f, "params", None) or {}
            if isinstance(params, dict) and "dim" in params:
                return int(params["dim"])
        return 0

    def _has_collection(self, name: str) -> bool:
        from pymilvus import utility

        return utility.has_collection(name, using=CONN_ALIAS)

    def _classify_schema(self, name: str) -> str:
        cached = self._schema_kind.get(name)
        if cached:
            return cached
        if not self._has_collection(name):
            return "dense_only"
        from pymilvus import Collection, DataType

        col = Collection(name, using=CONN_ALIAS)
        for f in col.schema.fields:
            if f.name == SPARSE_FIELD and f.dtype == DataType.SPARSE_FLOAT_VECTOR:
                self._schema_kind[name] = "hybrid"
                return "hybrid"
        self._schema_kind[name] = "dense_only"
        return "dense_only"

    def _ensure_loaded(self, name: str) -> None:
        from pymilvus import Collection

        col = Collection(name, using=CONN_ALIAS)
        col.load()

    def _build_schema(
        self,
        dim: int,
        metric: str,
        hybrid: bool,
    ) -> Any:
        from pymilvus import CollectionSchema, FieldSchema, DataType

        fields = [
            FieldSchema(
                name=PK_FIELD,
                dtype=DataType.VARCHAR,
                is_primary=True,
                max_length=ID_MAX_LEN,
            ),
            FieldSchema(name=DENSE_FIELD, dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name=PAYLOAD_FIELD, dtype=DataType.JSON),
        ]
        if hybrid:
            fields.append(
                FieldSchema(name=SPARSE_FIELD, dtype=DataType.SPARSE_FLOAT_VECTOR)
            )
        return CollectionSchema(
            fields,
            description="bastion vector collection",
            enable_dynamic_field=False,
        )

    def _create_indexes_and_load(
        self,
        name: str,
        hybrid: bool,
        metric: str,
    ) -> None:
        from pymilvus import Collection

        col = Collection(name, using=CONN_ALIAS)
        idx_dense = {
            "index_type": "HNSW",
            "metric_type": metric,
            "params": {"M": 16, "efConstruction": 200},
        }
        col.create_index(DENSE_FIELD, idx_dense)
        if hybrid:
            idx_sparse = {
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "IP",
                "params": {},
            }
            col.create_index(SPARSE_FIELD, idx_sparse)
        col.load()

    def create_collection(self, spec: CreateCollectionInput) -> Optional[str]:
        if not self.is_available():
            return "Service or Qdrant not initialized"
        try:
            from pymilvus import Collection, utility

            if utility.has_collection(spec.collection_name, using=CONN_ALIAS):
                self._invalidate_cache(spec.collection_name)
                logger.info(
                    "Collection '%s' already exists (idempotent success)",
                    spec.collection_name,
                )
                self._classify_schema(spec.collection_name)
                return None

            metric = _distance_to_milvus_metric(spec.distance)
            hybrid = bool(spec.enable_sparse)
            schema = self._build_schema(spec.vector_size, metric, hybrid)
            col = Collection(spec.collection_name, schema, using=CONN_ALIAS)
            self._create_indexes_and_load(spec.collection_name, hybrid, metric)
            self._schema_kind[spec.collection_name] = "hybrid" if hybrid else "dense_only"
            self._dense_metric[spec.collection_name] = metric
            logger.info(
                "Created Milvus collection %r hybrid=%s dim=%s metric=%s",
                spec.collection_name,
                hybrid,
                spec.vector_size,
                metric,
            )
            return None
        except Exception as e:
            logger.error("CreateCollection failed: %s: %s", type(e).__name__, e)
            return str(e)

    def delete_collection(self, collection_name: str) -> Optional[str]:
        if not self.is_available():
            return "Service or Qdrant not initialized"
        try:
            from pymilvus import utility

            if utility.has_collection(collection_name, using=CONN_ALIAS):
                utility.drop_collection(collection_name, using=CONN_ALIAS)
            self._invalidate_cache(collection_name)
            logger.info("Deleted Milvus collection %r", collection_name)
            return None
        except Exception as e:
            logger.error("DeleteCollection failed: %s", e)
            return str(e)

    def list_collections(self) -> Tuple[List[CollectionInfoOut], Optional[str]]:
        if not self.is_available():
            return [], "Service or Qdrant not initialized"
        try:
            from pymilvus import Collection, utility

            names = utility.list_collections(using=CONN_ALIAS)
            out: List[CollectionInfoOut] = []
            for name in names:
                try:
                    self._ensure_loaded(name)
                    col = Collection(name, using=CONN_ALIAS)
                    n = col.num_entities
                    sk = self._classify_schema(name)
                    schema_type = "named_hybrid" if sk == "hybrid" else "named_dense"
                    dim = self._dense_field_dim(col)
                    metric = self._dense_metric.get(name) or "COSINE"
                    out.append(
                        CollectionInfoOut(
                            name=name,
                            vector_size=dim,
                            distance=_metric_to_proto_distance(metric),
                            points_count=n,
                            status="green",
                            schema_type=schema_type,
                        )
                    )
                except Exception:
                    out.append(
                        CollectionInfoOut(
                            name=name,
                            vector_size=0,
                            distance="COSINE",
                            points_count=0,
                            status="green",
                            schema_type="named_dense",
                        )
                    )
            return out, None
        except Exception as e:
            logger.error("ListCollections failed: %s", e)
            return [], str(e)

    def get_collection_info(self, collection_name: str) -> GetCollectionInfoResult:
        if not self.is_available():
            return GetCollectionInfoResult(
                success=False,
                error="Service or Qdrant not initialized",
            )
        try:
            from pymilvus import Collection, utility

            if not utility.has_collection(collection_name, using=CONN_ALIAS):
                return GetCollectionInfoResult(
                    success=False,
                    error=f"Collection '{collection_name}' doesn't exist",
                )
            self._ensure_loaded(collection_name)
            col = Collection(collection_name, using=CONN_ALIAS)
            dim = self._dense_field_dim(col)
            sk = self._classify_schema(collection_name)
            schema_type = "named_hybrid" if sk == "hybrid" else "named_dense"
            metric = self._dense_metric.get(collection_name) or "COSINE"
            info = CollectionInfoOut(
                name=collection_name,
                vector_size=dim,
                distance=_metric_to_proto_distance(metric),
                points_count=col.num_entities,
                status="green",
                schema_type=schema_type,
            )
            return GetCollectionInfoResult(success=True, collection=info)
        except Exception as e:
            logger.error("GetCollectionInfo failed: %s", e)
            err = str(e)
            if "not exist" in err.lower() or "not found" in err.lower():
                return GetCollectionInfoResult(
                    success=False,
                    error=f"Collection '{collection_name}' doesn't exist",
                )
            return GetCollectionInfoResult(success=False, error=err)

    def _ensure_collection_for_vectors(
        self,
        collection_name: str,
        vector_size: int,
        wants_hybrid: bool,
    ) -> None:
        from pymilvus import Collection, utility

        if not utility.has_collection(collection_name, using=CONN_ALIAS):
            metric = "COSINE"
            schema = self._build_schema(vector_size, metric, wants_hybrid)
            Collection(collection_name, schema, using=CONN_ALIAS)
            self._create_indexes_and_load(collection_name, wants_hybrid, metric)
            self._schema_kind[collection_name] = "hybrid" if wants_hybrid else "dense_only"
            self._dense_metric[collection_name] = metric
            logger.info(
                "Created Milvus collection %r (auto) hybrid=%s dim=%s",
                collection_name,
                wants_hybrid,
                vector_size,
            )
            return
        sk = self._classify_schema(collection_name)
        if wants_hybrid and sk != "hybrid":
            raise ValueError(
                f"Collection '{collection_name}' is dense-only in Milvus; cannot upsert sparse vectors. "
                "Delete the collection and recreate with enable_sparse=True, or use a new collection name."
            )

    def upsert_vectors(self, collection_name: str, points: List[VectorPointInput]) -> int:
        if not self.is_available():
            raise RuntimeError("Milvus not configured")
        any_sparse = any(p.sparse is not None for p in points)
        vector_size = len(points[0].vector) if points[0].vector else 128
        self._ensure_collection_for_vectors(
            collection_name, vector_size, any_sparse
        )
        sk = self._classify_schema(collection_name)
        from pymilvus import Collection

        col = Collection(collection_name, using=CONN_ALIAS)
        self._ensure_loaded(collection_name)

        ids: List[str] = []
        dense: List[List[float]] = []
        payloads: List[Dict[str, Any]] = []
        sparses: Optional[List[Optional[Dict[int, float]]]] = [] if sk == "hybrid" else None

        for p in points:
            sid = str(p.id)[:ID_MAX_LEN]
            ids.append(sid)
            dense.append(list(p.vector))
            payloads.append(dict(p.payload))
            if sparses is not None:
                if p.sparse is not None:
                    sparses.append(_sparse_to_dict(p.sparse))
                else:
                    sparses.append(None)

        batch_size = 100
        total = 0
        max_retries = getattr(self._settings, "MILVUS_UPSERT_MAX_RETRIES", 3)
        rows: List[Dict[str, Any]] = []
        for i, pid in enumerate(ids):
            row: Dict[str, Any] = {
                PK_FIELD: pid,
                DENSE_FIELD: dense[i],
                PAYLOAD_FIELD: payloads[i],
            }
            if sparses is not None and sparses[i] is not None:
                row[SPARSE_FIELD] = sparses[i]
            rows.append(row)

        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            for attempt in range(max_retries):
                try:
                    col.upsert(batch)
                    total += len(batch)
                    break
                except Exception as batch_err:
                    err_str = str(batch_err).lower()
                    is_retryable = any(
                        x in err_str
                        for x in (
                            "timeout",
                            "timed out",
                            "connection",
                            "unavailable",
                            "503",
                            "500",
                            "internal",
                        )
                    )
                    if is_retryable and attempt < max_retries - 1:
                        wait_time = 2**attempt
                        logger.warning(
                            "Milvus upsert batch failed (attempt %s/%s): %s; retry in %ss",
                            attempt + 1,
                            max_retries,
                            batch_err,
                            wait_time,
                        )
                        time.sleep(wait_time)
                    else:
                        raise
        try:
            col.flush()
        except Exception as ex:
            logger.debug("Milvus flush after upsert: %s", ex)
        logger.info("Upserted %s vectors to Milvus collection %r", total, collection_name)
        return total

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
        if not self.is_available():
            return []
        if not self._has_collection(collection_name):
            logger.info(
                "SearchVectors: Collection %r doesn't exist, returning empty results",
                collection_name,
            )
            return []
        from pymilvus import AnnSearchRequest, Collection, RRFRanker

        if score_threshold > 0:
            logger.debug(
                "Milvus search_vectors ignores score_threshold=%s (dense path)",
                score_threshold,
            )
        sk = self._classify_schema(collection_name)
        fusion = (fusion_mode or "").strip().lower()
        use_rrf = (
            sparse_query is not None
            and fusion == "rrf"
            and sk == "hybrid"
        )
        if sparse_query is not None and fusion == "rrf" and sk != "hybrid":
            logger.warning(
                "SearchVectors: RRF requested but collection %r is not hybrid (schema=%s); "
                "using dense-only search",
                collection_name,
                sk,
            )

        expr = _milvus_expr_from_filters(filters, "search")
        self._ensure_loaded(collection_name)
        col = Collection(collection_name, using=CONN_ALIAS)
        metric = self._dense_metric.get(collection_name, "COSINE")
        lim = limit or 50
        prefetch_limit = max(lim * 3, 100)
        search_param_dense = {"metric_type": metric, "params": {"ef": 128}}

        try:
            if use_rrf and sparse_query is not None:
                sparse_dict = _sparse_to_dict(sparse_query)
                dr = AnnSearchRequest(
                    data=[list(query_vector)],
                    anns_field=DENSE_FIELD,
                    param=search_param_dense,
                    limit=prefetch_limit,
                    expr=expr,
                )
                sr = AnnSearchRequest(
                    data=[sparse_dict],
                    anns_field=SPARSE_FIELD,
                    param={"metric_type": "IP"},
                    limit=prefetch_limit,
                    expr=expr,
                )
                hybrid_res = col.hybrid_search(
                    reqs=[dr, sr],
                    rerank=RRFRanker(k=60),
                    limit=lim,
                    output_fields=[PAYLOAD_FIELD],
                )
                hits_raw = hybrid_res[0] if hybrid_res else []
            else:
                kw: Dict[str, Any] = {
                    "data": [list(query_vector)],
                    "anns_field": DENSE_FIELD,
                    "param": search_param_dense,
                    "limit": lim,
                    "output_fields": [PAYLOAD_FIELD],
                }
                if expr:
                    kw["expr"] = expr
                # score_threshold: Milvus dense search differs from Qdrant; not applied here (see plan).
                res = col.search(**kw)
                hits_raw = res[0] if res else []
        except Exception as e:
            err = str(e).lower()
            if "not exist" in err or "not found" in err or "doesn't exist" in err:
                logger.info(
                    "SearchVectors: Collection %r error during search, returning empty: %s",
                    collection_name,
                    e,
                )
                return []
            raise

        hits: List[SearchHit] = []
        for hit in hits_raw:
            pl: Dict[str, Any] = {}
            hid = str(hit.id)
            score = float(
                getattr(hit, "distance", None)
                if getattr(hit, "distance", None) is not None
                else getattr(hit, "score", 0.0) or 0.0
            )
            entity = getattr(hit, "entity", None)
            if entity is not None:
                if hasattr(entity, "get"):
                    raw = entity.get(PAYLOAD_FIELD)
                elif isinstance(entity, dict):
                    raw = entity.get(PAYLOAD_FIELD)
                else:
                    raw = None
                if isinstance(raw, dict):
                    pl = raw
                elif isinstance(raw, str):
                    try:
                        pl = json.loads(raw)
                    except (json.JSONDecodeError, TypeError):
                        pl = {}
            hits.append(SearchHit(id=hid, score=score, payload=dict(pl)))
        logger.info("SearchVectors: Found %s results in %r", len(hits), collection_name)
        if len(hits) == 0 and filters:
            logger.info(
                "SearchVectors: 0 results for collection %s (entities=%s); filter was applied",
                collection_name,
                col.num_entities,
            )
        return hits

    def scroll_points(
        self,
        collection_name: str,
        filters: List[VectorFilterInput],
        limit: int,
        offset: str,
        with_vectors: bool,
    ) -> ScrollResult:
        del with_vectors  # proto scroll returns payload only, matching Qdrant gRPC mapping
        if not self.is_available():
            return ScrollResult(points=[], next_offset="")
        if not self._has_collection(collection_name):
            return ScrollResult(points=[], next_offset="")
        from pymilvus import Collection

        expr = _milvus_expr_from_filters(filters, "search")
        page_limit = limit or 256
        page_limit = max(1, min(int(page_limit), 8192))
        off_str = (offset or "").strip()
        off_int = 0
        if off_str:
            try:
                off_int = int(off_str)
            except ValueError:
                off_int = 0

        self._ensure_loaded(collection_name)
        col = Collection(collection_name, using=CONN_ALIAS)
        out_fields = [PK_FIELD, PAYLOAD_FIELD]

        try:
            if off_int + page_limit <= _MILVUS_MAX_OFFSET:
                qexpr = expr if expr else f'{PK_FIELD} != ""'
                rows = col.query(
                    expr=qexpr,
                    offset=off_int,
                    limit=page_limit,
                    output_fields=out_fields,
                )
                next_off = ""
                if len(rows) == page_limit:
                    next_off = str(off_int + page_limit)
            else:
                rows = self._scroll_via_iterator(col, expr, off_int, page_limit)
                next_off = (
                    str(off_int + len(rows)) if len(rows) == page_limit else ""
                )
        except Exception as e:
            err = str(e).lower()
            if "not exist" in err or "not found" in err:
                return ScrollResult(points=[], next_offset="")
            raise

        points: List[ScrolledPointOut] = []
        for r in rows or []:
            pid = r.get(PK_FIELD, "")
            pl = r.get(PAYLOAD_FIELD) or {}
            if isinstance(pl, str):
                try:
                    pl = json.loads(pl)
                except (json.JSONDecodeError, TypeError):
                    pl = {}
            points.append(ScrolledPointOut(id=str(pid), payload=dict(pl)))
        return ScrollResult(points=points, next_offset=next_off)

    def _scroll_via_iterator(
        self,
        col: Any,
        expr: Optional[str],
        skip: int,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Skip `skip` entities then return up to `limit` (large offset path)."""
        qexpr = expr if expr else f'{PK_FIELD} != ""'
        it = col.query_iterator(
            expr=qexpr,
            batch_size=min(512, max(limit, 1)),
            output_fields=[PK_FIELD, PAYLOAD_FIELD],
        )
        skipped = 0
        buf: List[Dict[str, Any]] = []
        while True:
            batch = it.next()
            if not batch:
                return buf
            for row in batch:
                if skipped < skip:
                    skipped += 1
                    continue
                buf.append(row)
                if len(buf) >= limit:
                    return buf

    def count_vectors(self, collection_name: str, filters: List[VectorFilterInput]) -> int:
        if not self.is_available():
            return 0
        if not self._has_collection(collection_name):
            return 0
        from pymilvus import Collection

        self._ensure_loaded(collection_name)
        col = Collection(collection_name, using=CONN_ALIAS)
        expr = _milvus_expr_from_filters(filters, "search")
        try:
            if not expr:
                return int(col.num_entities)
            rows = col.query(
                expr=expr,
                output_fields=["count(*)"],
                limit=1,
            )
            if rows and "count(*)" in rows[0]:
                return int(rows[0]["count(*)"])
        except Exception as e:
            logger.debug("Milvus count(*) query failed, falling back: %s", e)
        try:
            qex = expr if expr else f'{PK_FIELD} != ""'
            it = col.query_iterator(
                expr=qex,
                batch_size=2048,
                output_fields=[PK_FIELD],
            )
            n = 0
            while True:
                batch = it.next()
                if not batch:
                    break
                n += len(batch)
            return n
        except Exception:
            return 0

    def delete_vectors_equality(
        self, collection_name: str, filters: List[VectorFilterInput]
    ) -> Tuple[int, Optional[str]]:
        if not self.is_available():
            return 0, "Service or Qdrant not initialized"
        if not filters:
            return 0, "No filters provided - use DeleteCollection for full deletion"
        if not self._has_collection(collection_name):
            return 0, None
        from pymilvus import Collection

        expr = _milvus_expr_from_filters(filters, "equality")
        if not expr:
            return 0, "No filters provided - use DeleteCollection for full deletion"
        self._ensure_loaded(collection_name)
        col = Collection(collection_name, using=CONN_ALIAS)
        col.delete(expr)
        logger.info("DeleteVectors: Milvus delete on %r expr=%r", collection_name, expr)
        return 1, None

    def update_metadata_equality(
        self,
        collection_name: str,
        filters: List[VectorFilterInput],
        metadata_updates: Dict[str, Any],
    ) -> Tuple[int, Optional[str]]:
        if not self.is_available():
            return 0, "Service or Qdrant not initialized"
        if not self._has_collection(collection_name):
            return 0, None
        from pymilvus import Collection

        expr = _milvus_expr_from_filters(filters, "equality")
        if not expr:
            return 0, None
        self._ensure_loaded(collection_name)
        col = Collection(collection_name, using=CONN_ALIAS)
        sk = self._classify_schema(collection_name)
        out_f = [PK_FIELD, DENSE_FIELD, PAYLOAD_FIELD]
        if sk == "hybrid":
            out_f.append(SPARSE_FIELD)
        rows = col.query(expr=expr, limit=10000, output_fields=out_f)
        updated = 0
        upsert_rows: List[Dict[str, Any]] = []
        for r in rows or []:
            pid = r.get(PK_FIELD)
            pl = dict(r.get(PAYLOAD_FIELD) or {})
            if isinstance(pl, str):
                try:
                    pl = json.loads(pl)
                except (json.JSONDecodeError, TypeError):
                    pl = {}
            pl.update(metadata_updates)
            row: Dict[str, Any] = {
                PK_FIELD: pid,
                DENSE_FIELD: r[DENSE_FIELD],
                PAYLOAD_FIELD: pl,
            }
            if sk == "hybrid" and SPARSE_FIELD in r and r[SPARSE_FIELD] is not None:
                row[SPARSE_FIELD] = r[SPARSE_FIELD]
            upsert_rows.append(row)
        if upsert_rows:
            col.upsert(upsert_rows)
            updated = len(upsert_rows)
            try:
                col.flush()
            except Exception:
                pass
        logger.info(
            "UpdateVectorMetadata: Updated %s vectors in Milvus %r",
            updated,
            collection_name,
        )
        return updated, None
