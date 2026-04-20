"""
Elasticsearch 8.14+ / OpenSearch 2.x implementation of VectorBackend.
"""

from __future__ import annotations

import base64
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

DENSE_FIELD = "dense"
SPARSE_FIELD = "sparse"
PAYLOAD_FIELD = "payload"
OS_HYBRID_PIPELINE = "bastion_vector_hybrid"
# Elasticsearch retriever RRF API (8.14+)
_MIN_ES_FOR_RETRIEVER_RRF = (8, 14, 0)
# OpenSearch hybrid + search_pipeline (2.10+)
_MIN_OS_FOR_HYBRID_PIPELINE = (2, 10, 0)


def _sparse_str_dict(sp: SparseVectorData) -> Dict[str, float]:
    return {str(int(i)): float(v) for i, v in zip(sp.indices, sp.values)}


def _es_filter_from_filters(
    filters: List[VectorFilterInput],
    mode: str,
) -> Optional[Dict[str, Any]]:
    """
    Build ES bool query from VectorFilterInput.
    mode: "search" (equals, not_equals, any_of, in) or "equality" (term per filter).
    """
    if not filters:
        return None
    must: List[Dict[str, Any]] = []
    must_not: List[Dict[str, Any]] = []
    for vf in filters:
        field_path = f"{PAYLOAD_FIELD}.{vf.field}"
        if mode == "equality":
            must.append({"term": {field_path: vf.value}})
            continue
        if vf.operator == "equals":
            must.append({"term": {field_path: vf.value}})
        elif vf.operator == "not_equals":
            must_not.append({"term": {field_path: vf.value}})
        elif vf.operator == "any_of" and vf.values:
            must.append({"terms": {field_path: list(vf.values)}})
        elif vf.operator == "in":
            must.append({"term": {field_path: vf.value}})
    if not must and not must_not:
        return None
    out: Dict[str, Any] = {"bool": {}}
    if must:
        out["bool"]["must"] = must
    if must_not:
        out["bool"]["must_not"] = must_not
    return out


def _parse_version_tuple(version_str: str) -> Tuple[int, ...]:
    if not version_str:
        return (0, 0, 0)
    parts = []
    for p in version_str.split("."):
        num = ""
        for ch in p:
            if ch.isdigit():
                num += ch
            else:
                break
        parts.append(int(num) if num else 0)
    return tuple(parts) if parts else (0, 0, 0)


def _distance_to_es_similarity(distance_str: str) -> str:
    d = (distance_str or "").strip().upper()
    if d in ("DOT", "IP"):
        return "dot_product"
    if d in ("EUCLIDEAN", "EUCLID", "L2"):
        return "l2_norm"
    return "cosine"


def _similarity_to_proto_distance(sim: str) -> str:
    s = (sim or "cosine").lower()
    if s == "dot_product":
        return "DOT"
    if s in ("l2_norm", "l2"):
        return "EUCLIDEAN"
    return "COSINE"


class _ESCompat:
    """Elasticsearch 8.x: sparse_vector field, retriever RRF."""

    sparse_field_type_es = "sparse_vector"

    @staticmethod
    def build_hybrid_search_kwargs(
        index: str,
        query_vector: List[float],
        sparse_dict: Dict[str, float],
        limit: int,
        prefetch: int,
        post_filter: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        retriever = {
            "rrf": {
                "retrievers": [
                    {
                        "knn": {
                            "field": DENSE_FIELD,
                            "query_vector": list(query_vector),
                            "k": prefetch,
                            "num_candidates": prefetch,
                        }
                    },
                    {
                        "standard": {
                            "query": {
                                "sparse_vector": {
                                    "field": SPARSE_FIELD,
                                    "query_vector": sparse_dict,
                                }
                            }
                        }
                    },
                ],
                "rank_window_size": prefetch,
                "rank_constant": 60,
            }
        }
        kw: Dict[str, Any] = {
            "index": index,
            "retriever": retriever,
            "size": limit or 50,
        }
        if post_filter:
            kw["post_filter"] = post_filter
        return kw


class _OpenSearchCompat:
    """OpenSearch 2.x: dynamic object `sparse` for script_score; hybrid + pipeline."""

    @staticmethod
    def sparse_mapping_fragment() -> Dict[str, Any]:
        return {
            "type": "object",
            "dynamic": True,
        }

    @staticmethod
    def hybrid_script_source() -> str:
        # Dot-product over dynamically mapped sparse.* doc values
        return (
            "double s = 0.0; "
            "for (def e : params.q.entrySet()) { "
            "  String f = 'sparse.' + e.getKey(); "
            "  if (doc.containsKey(f) && doc[f].size() > 0) { "
            "    s += e.getValue() * doc[f].value; "
            "  } "
            "} "
            "return s;"
        )

    @staticmethod
    def build_hybrid_search_body(
        query_vector: List[float],
        sparse_dict: Dict[str, float],
        limit: int,
        prefetch: int,
        post_filter: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        knn_sub = {
            "knn": {
                DENSE_FIELD: {
                    "vector": list(query_vector),
                    "k": prefetch,
                }
            }
        }
        script_sub = {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": _OpenSearchCompat.hybrid_script_source(),
                    "params": {"q": sparse_dict},
                },
            }
        }
        hybrid_q: Dict[str, Any] = {
            "hybrid": {
                "queries": [knn_sub, script_sub],
            }
        }
        body: Dict[str, Any] = {
            "size": limit or 50,
            "query": hybrid_q,
        }
        if post_filter:
            body["post_filter"] = post_filter
        return body


def _is_404(exc: BaseException) -> bool:
    meta = getattr(exc, "meta", None)
    if meta is not None:
        st = getattr(meta, "status", None) or meta.get("status") if isinstance(meta, dict) else None
        if st == 404:
            return True
    if getattr(exc, "status_code", None) == 404:
        return True
    s = str(exc).lower()
    return "not_found" in s or "index_not_found" in s or "404" in s


class ElasticsearchBackend:
    """Elasticsearch or OpenSearch-backed vector store."""

    def __init__(self, settings: Any) -> None:
        self._settings = settings
        self._client: Any = None
        self._is_opensearch = False
        self._version_tuple: Tuple[int, ...] = (0, 0, 0)
        self._supports_es_retriever_rrf = False
        self._supports_os_hybrid = False
        self._schema_kind: Dict[str, str] = {}
        self._os_pipeline_ensured = False

    def _hosts(self) -> List[str]:
        raw = (getattr(self._settings, "ES_URL", None) or "").strip()
        return [h.strip() for h in raw.split(",") if h.strip()]

    def _index_prefix(self) -> str:
        p = (getattr(self._settings, "ES_INDEX_PREFIX", None) or "bastion_").strip()
        return p if p else "bastion_"

    def _physical_index_name(self, collection_name: str) -> str:
        return f"{self._index_prefix()}{collection_name}".lower()

    def _logical_name(self, physical_index: str) -> str:
        prefix = self._index_prefix().lower()
        if physical_index.startswith(prefix):
            return physical_index[len(prefix) :]
        return physical_index

    def initialize(self) -> None:
        hosts = self._hosts()
        if not hosts:
            logger.warning("ES_URL not set, vector store features will be unavailable")
            self._client = None
            return

        verify = bool(getattr(self._settings, "ES_VERIFY_CERTS", True))
        ca = (getattr(self._settings, "ES_CA_CERTS", None) or "").strip() or None
        user = (getattr(self._settings, "ES_USERNAME", None) or "").strip()
        pw = (getattr(self._settings, "ES_PASSWORD", None) or "").strip()
        api_key = (getattr(self._settings, "ES_API_KEY", None) or "").strip()

        common: Dict[str, Any] = {
            "hosts": hosts,
            "verify_certs": verify,
        }
        if ca:
            common["ca_certs"] = ca
        if user and pw:
            common["basic_auth"] = (user, pw)
        if api_key and not user:
            # ES API key id:key base64 or raw
            if ":" in api_key:
                raw = base64.b64encode(api_key.encode("utf-8")).decode("ascii")
                common["headers"] = {"Authorization": f"ApiKey {raw}"}
            else:
                common["headers"] = {"Authorization": f"ApiKey {api_key}"}

        try:
            from elasticsearch import Elasticsearch

            es_client = Elasticsearch(**common)
            info = es_client.info()
            ver = info.get("version") or {}
            dist = str(ver.get("distribution", "")).lower()
            if dist == "opensearch":
                from opensearchpy import OpenSearch

                self._client = OpenSearch(**common)
                self._is_opensearch = True
                self._version_tuple = _parse_version_tuple(str(ver.get("number", "0")))
                logger.info(
                    "Connected to OpenSearch %s at %s",
                    ver.get("number"),
                    hosts,
                )
            else:
                self._client = es_client
                self._is_opensearch = False
                self._version_tuple = _parse_version_tuple(str(ver.get("number", "0")))
                logger.info(
                    "Connected to Elasticsearch %s at %s",
                    ver.get("number"),
                    hosts,
                )
        except Exception as e:
            logger.error("Elasticsearch/OpenSearch connect failed: %s", e)
            self._client = None
            return

        if self._is_opensearch:
            self._supports_os_hybrid = self._version_tuple >= _MIN_OS_FOR_HYBRID_PIPELINE
        else:
            self._supports_es_retriever_rrf = self._version_tuple >= _MIN_ES_FOR_RETRIEVER_RRF

    def is_configured(self) -> bool:
        return bool(self._hosts())

    def is_available(self) -> bool:
        return self._client is not None

    def _invalidate_cache(self, logical_name: str) -> None:
        self._schema_kind.pop(logical_name.lower(), None)

    def _mapping_has_sparse_vector(self, props: Dict[str, Any]) -> bool:
        sp = props.get(SPARSE_FIELD) or {}
        if isinstance(sp, dict) and sp.get("type") == "sparse_vector":
            return True
        return False

    def _mapping_has_sparse_object(self, props: Dict[str, Any]) -> bool:
        sp = props.get(SPARSE_FIELD) or {}
        return isinstance(sp, dict) and sp.get("type") == "object"

    def _classify_schema(self, logical_name: str) -> str:
        key = logical_name.lower()
        cached = self._schema_kind.get(key)
        if cached:
            return cached
        idx = self._physical_index_name(logical_name)
        if not self._client or not self._index_exists(idx):
            return "dense_only"
        try:
            m = self._get_mapping_props(idx)
            if self._mapping_has_sparse_vector(m) or self._mapping_has_sparse_object(m):
                self._schema_kind[key] = "hybrid"
                return "hybrid"
        except Exception as ex:
            logger.debug("classify_schema: %s", ex)
        self._schema_kind[key] = "dense_only"
        return "dense_only"

    def _index_exists(self, index: str) -> bool:
        if not self._client:
            return False
        try:
            return bool(self._client.indices.exists(index=index))
        except Exception:
            return False

    def _get_mapping_props(self, index: str) -> Dict[str, Any]:
        resp = self._client.indices.get_mapping(index=index)
        entry = resp.get(index) or (next(iter(resp.values())) if resp else {})
        root = (entry.get("mappings") or {}) if isinstance(entry, dict) else {}
        pr = root.get("properties")
        return pr if isinstance(pr, dict) else {}

    def _dense_props(self, index: str) -> Tuple[int, str]:
        props = self._get_mapping_props(index)
        d = props.get(DENSE_FIELD) or {}
        if not isinstance(d, dict):
            return 0, "cosine"
        dims = int(d.get("dims", 0) or 0)
        sim = str(d.get("similarity", "cosine"))
        return dims, sim

    def _doc_count(self, index: str) -> int:
        try:
            if self._is_opensearch:
                r = self._client.count(index=index, body={"query": {"match_all": {}}})
            else:
                r = self._client.count(index=index, query={"match_all": {}})
            return int(r.get("count", 0))
        except Exception:
            return 0

    def _ensure_os_hybrid_pipeline(self) -> None:
        if self._os_pipeline_ensured or not self._client or not self._is_opensearch:
            return
        body = {
            "description": "Bastion vector-service hybrid (dense + sparse script)",
            "phase_results_processors": [
                {
                    "normalization-processor": {
                        "normalization": {"technique": "min_max"},
                        "combination": {
                            "technique": "arithmetic_mean",
                            "parameters": {"weights": [0.5, 0.5]},
                        },
                    }
                }
            ],
        }
        try:
            self._client.transport.perform_request(
                "PUT",
                f"/_search/pipeline/{OS_HYBRID_PIPELINE}",
                body=body,
            )
            self._os_pipeline_ensured = True
        except Exception as e:
            logger.warning("Could not register OpenSearch hybrid pipeline: %s", e)

    def _indices_create(self, index: str, mappings: Dict[str, Any], settings: Dict[str, Any]) -> None:
        if self._is_opensearch:
            self._client.indices.create(
                index=index,
                body={"settings": settings, "mappings": mappings},
            )
        else:
            self._client.indices.create(index=index, mappings=mappings, settings=settings)

    def create_collection(self, spec: CreateCollectionInput) -> Optional[str]:
        if not self._client:
            return "Vector store not initialized"
        idx = self._physical_index_name(spec.collection_name)
        try:
            if self._index_exists(idx):
                self._invalidate_cache(spec.collection_name)
                self._schema_kind[spec.collection_name.lower()] = (
                    "hybrid" if spec.enable_sparse else "dense_only"
                )
                logger.info("Index %r already exists (idempotent success)", idx)
                return None
            similarity = _distance_to_es_similarity(spec.distance)
            dense_prop: Dict[str, Any] = {
                "type": "dense_vector",
                "dims": int(spec.vector_size),
                "index": True,
                "similarity": similarity,
            }
            props: Dict[str, Any] = {
                DENSE_FIELD: dense_prop,
                PAYLOAD_FIELD: {"type": "object", "dynamic": True},
            }
            dynamic_templates = [
                {
                    "payload_strings": {
                        "path_match": f"{PAYLOAD_FIELD}.*",
                        "match_mapping_type": "string",
                        "mapping": {"type": "keyword"},
                    }
                }
            ]
            if spec.enable_sparse:
                if self._is_opensearch:
                    props[SPARSE_FIELD] = _OpenSearchCompat.sparse_mapping_fragment()
                else:
                    props[SPARSE_FIELD] = {"type": _ESCompat.sparse_field_type_es}
            index_settings: Dict[str, Any] = {
                "number_of_shards": 1,
                "number_of_replicas": 0,
            }
            if self._is_opensearch:
                index_settings["knn"] = True
            mappings: Dict[str, Any] = {
                "dynamic_templates": dynamic_templates,
                "properties": props,
            }
            self._indices_create(idx, mappings, {"index": index_settings})
            self._invalidate_cache(spec.collection_name)
            self._schema_kind[spec.collection_name.lower()] = (
                "hybrid" if spec.enable_sparse else "dense_only"
            )
            logger.info(
                "Created index %r dense_dim=%s similarity=%s hybrid=%s",
                idx,
                spec.vector_size,
                similarity,
                spec.enable_sparse,
            )
            return None
        except Exception as e:
            logger.error("CreateCollection failed: %s: %s", type(e).__name__, e)
            return str(e)

    def delete_collection(self, collection_name: str) -> Optional[str]:
        if not self._client:
            return "Vector store not initialized"
        idx = self._physical_index_name(collection_name)
        try:
            self._client.indices.delete(index=idx, ignore=[404])
            self._invalidate_cache(collection_name)
            logger.info("Deleted index %r", idx)
            return None
        except Exception as e:
            logger.error("DeleteCollection failed: %s", e)
            return str(e)

    def list_collections(self) -> Tuple[List[CollectionInfoOut], Optional[str]]:
        if not self._client:
            return [], "Vector store not initialized"
        pattern = f"{self._index_prefix()}*".lower()
        try:
            try:
                resp = self._client.indices.get(index=pattern)
            except Exception as ge:
                if _is_404(ge):
                    return [], None
                raise
            if not isinstance(resp, dict):
                return [], None
            out: List[CollectionInfoOut] = []
            for physical in sorted(resp.keys()):
                try:
                    dims, sim = self._dense_props(physical)
                    cnt = self._doc_count(physical)
                    props = self._get_mapping_props(physical)
                    hybrid = self._mapping_has_sparse_vector(props) or self._mapping_has_sparse_object(
                        props
                    )
                    schema_type = "named_hybrid" if hybrid else "named_dense"
                    out.append(
                        CollectionInfoOut(
                            name=self._logical_name(physical),
                            vector_size=dims,
                            distance=_similarity_to_proto_distance(sim),
                            points_count=cnt,
                            status="green",
                            schema_type=schema_type,
                        )
                    )
                except Exception as ex:
                    logger.debug("list skip %s: %s", physical, ex)
            return out, None
        except Exception as e:
            if _is_404(e):
                return [], None
            logger.error("ListCollections failed: %s", e)
            return [], str(e)

    def get_collection_info(self, collection_name: str) -> GetCollectionInfoResult:
        if not self._client:
            return GetCollectionInfoResult(
                success=False,
                error="Vector store not initialized",
            )
        idx = self._physical_index_name(collection_name)
        if not self._index_exists(idx):
            return GetCollectionInfoResult(
                success=False,
                error=f"Collection '{collection_name}' doesn't exist",
            )
        try:
            dims, sim = self._dense_props(idx)
            cnt = self._doc_count(idx)
            props = self._get_mapping_props(idx)
            hybrid = self._mapping_has_sparse_vector(props) or self._mapping_has_sparse_object(props)
            schema_type = "named_hybrid" if hybrid else "named_dense"
            info = CollectionInfoOut(
                name=self._logical_name(idx),
                vector_size=dims,
                distance=_similarity_to_proto_distance(sim),
                points_count=cnt,
                status="green",
                schema_type=schema_type,
            )
            return GetCollectionInfoResult(success=True, collection=info)
        except Exception as e:
            logger.error("GetCollectionInfo failed: %s", e)
            if _is_404(e):
                return GetCollectionInfoResult(
                    success=False,
                    error=f"Collection '{collection_name}' doesn't exist",
                )
            return GetCollectionInfoResult(success=False, error=str(e))

    def _ensure_index_for_vectors(
        self,
        collection_name: str,
        vector_size: int,
        wants_hybrid: bool,
    ) -> None:
        if not self._client:
            raise RuntimeError("Elasticsearch not configured")
        idx = self._physical_index_name(collection_name)
        if self._index_exists(idx):
            schema = self._classify_schema(collection_name)
            if wants_hybrid and schema != "hybrid":
                raise ValueError(
                    f"Index '{collection_name}' is dense-only; cannot upsert sparse vectors. "
                    "Delete the index and recreate with enable_sparse=True, or use a new collection name."
                )
            return
        err = self.create_collection(
            CreateCollectionInput(
                collection_name=collection_name,
                vector_size=vector_size,
                distance="COSINE",
                enable_sparse=wants_hybrid,
            )
        )
        if err:
            raise RuntimeError(err)

    def _bulk_actions(
        self,
        index: str,
        points: List[VectorPointInput],
    ) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []
        for p in points:
            doc: Dict[str, Any] = {
                DENSE_FIELD: list(p.vector),
                PAYLOAD_FIELD: dict(p.payload or {}),
            }
            if p.sparse is not None:
                doc[SPARSE_FIELD] = _sparse_str_dict(p.sparse)
            actions.append({"index": {"_index": index, "_id": p.id}})
            actions.append(doc)
        return actions

    def upsert_vectors(self, collection_name: str, points: List[VectorPointInput]) -> int:
        if not self._client:
            raise RuntimeError("Elasticsearch not configured")
        if not points:
            return 0
        any_sparse = any(p.sparse is not None for p in points)
        vector_size = len(points[0].vector) if points[0].vector else 128
        self._ensure_index_for_vectors(collection_name, vector_size, any_sparse)

        idx = self._physical_index_name(collection_name)
        batch_size = 100
        max_retries = int(getattr(self._settings, "ES_UPSERT_MAX_RETRIES", 3) or 3)
        total = 0

        if self._is_opensearch:
            from opensearchpy.helpers import bulk as os_bulk
        else:
            from elasticsearch.helpers import bulk as es_bulk

        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            actions = self._bulk_actions(idx, batch)
            for attempt in range(max_retries):
                try:
                    if self._is_opensearch:
                        ok, errors = os_bulk(
                            self._client,
                            actions,
                            refresh="wait_for",
                            raise_on_error=False,
                        )
                    else:
                        ok, errors = es_bulk(
                            self._client,
                            actions,
                            refresh="wait_for",
                            raise_on_error=False,
                        )
                    if errors:
                        err_str = str(errors).lower()
                        retryable = (
                            "timeout" in err_str
                            or "503" in err_str
                            or "429" in err_str
                            or "connection" in err_str
                        )
                        if retryable and attempt < max_retries - 1:
                            time.sleep(2**attempt)
                            continue
                        raise RuntimeError(f"bulk index errors: {errors[:3]!r}")
                    total += ok
                    break
                except Exception as batch_err:
                    err_str = str(batch_err).lower()
                    retryable = (
                        "timeout" in err_str
                        or "connection" in err_str
                        or "503" in err_str
                        or "500" in err_str
                    )
                    if retryable and attempt < max_retries - 1:
                        logger.warning(
                            "ES bulk retry %s/%s in %ss: %s",
                            attempt + 1,
                            max_retries,
                            2**attempt,
                            batch_err,
                        )
                        time.sleep(2**attempt)
                    else:
                        raise
        logger.info("Upserted %s vectors to index %r", total, idx)
        return total

    def _search_hits_to_list(self, resp: Dict[str, Any]) -> List[SearchHit]:
        hits = []
        for h in (resp.get("hits") or {}).get("hits") or []:
            src = h.get("_source") or {}
            hid = str(h.get("_id", ""))
            score = float(h.get("_score") or 0.0)
            pl = src.get(PAYLOAD_FIELD) or {}
            hits.append(SearchHit(id=hid, score=score, payload=dict(pl)))
        return hits

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
        idx = self._physical_index_name(collection_name)
        if not self._index_exists(idx):
            logger.info(
                "SearchVectors: index %r does not exist, returning empty results",
                idx,
            )
            return []

        lim = limit or 50
        prefetch = max(lim * 3, 100)
        fq = _es_filter_from_filters(filters, "search")
        post_filter = fq
        schema = self._classify_schema(collection_name)
        fusion = (fusion_mode or "").strip().lower()
        use_hybrid = (
            sparse_query is not None
            and fusion == "rrf"
            and schema == "hybrid"
        )
        if sparse_query is not None and fusion == "rrf" and schema != "hybrid":
            logger.warning(
                "SearchVectors: RRF requested but collection %r is not hybrid (schema=%s); "
                "using dense-only search",
                collection_name,
                schema,
            )

        sparse_dict = _sparse_str_dict(sparse_query) if sparse_query else {}

        try:
            if use_hybrid and sparse_query is not None:
                if self._is_opensearch:
                    if not self._supports_os_hybrid:
                        logger.warning(
                            "OpenSearch version %s < 2.10: hybrid RRF unavailable; dense-only",
                            self._version_tuple,
                        )
                        resp = self._search_dense(idx, query_vector, lim, prefetch, post_filter)
                    else:
                        self._ensure_os_hybrid_pipeline()
                        body = _OpenSearchCompat.build_hybrid_search_body(
                            query_vector, sparse_dict, lim, prefetch, post_filter
                        )
                        resp = self._client.search(
                            index=idx,
                            body=body,
                            params={"search_pipeline": OS_HYBRID_PIPELINE},
                        )
                else:
                    if not self._supports_es_retriever_rrf:
                        logger.warning(
                            "Elasticsearch version %s < 8.14: retriever RRF unavailable; dense-only",
                            self._version_tuple,
                        )
                        resp = self._search_dense(idx, query_vector, lim, prefetch, post_filter)
                    else:
                        kw = _ESCompat.build_hybrid_search_kwargs(
                            idx,
                            query_vector,
                            sparse_dict,
                            lim,
                            prefetch,
                            post_filter,
                        )
                        resp = self._client.search(**kw)
            else:
                resp = self._search_dense(idx, query_vector, lim, prefetch, post_filter)
        except Exception as e:
            if _is_404(e):
                return []
            logger.error("SearchVectors failed: %s", e)
            raise

        hits = self._search_hits_to_list(resp)
        if score_threshold and score_threshold > 0:
            hits = [h for h in hits if h.score >= score_threshold]
        return hits

    def _search_dense(
        self,
        index: str,
        query_vector: List[float],
        limit: int,
        prefetch: int,
        post_filter: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if self._is_opensearch:
            body: Dict[str, Any] = {
                "size": limit,
                "query": {
                    "knn": {
                        DENSE_FIELD: {
                            "vector": list(query_vector),
                            "k": limit,
                        }
                    }
                },
            }
            if post_filter:
                body["post_filter"] = post_filter
            return self._client.search(index=index, body=body)
        kw: Dict[str, Any] = {
            "index": index,
            "size": limit,
            "knn": {
                "field": DENSE_FIELD,
                "query_vector": list(query_vector),
                "k": limit,
                "num_candidates": prefetch,
            },
        }
        if post_filter:
            kw["post_filter"] = post_filter
        return self._client.search(**kw)

    def scroll_points(
        self,
        collection_name: str,
        filters: List[VectorFilterInput],
        limit: int,
        offset: str,
        with_vectors: bool,
    ) -> ScrollResult:
        # Note: with_vectors reserved for parity; payload always returned.
        _ = with_vectors
        if not self._client:
            return ScrollResult(points=[], next_offset="")
        idx = self._physical_index_name(collection_name)
        if not self._index_exists(idx):
            return ScrollResult(points=[], next_offset="")

        page_limit = limit or 256
        page_limit = max(1, min(int(page_limit), 10000))
        fq = _es_filter_from_filters(filters, "search")
        base_q = fq if fq else {"match_all": {}}

        search_after = None
        off = (offset or "").strip()
        if off:
            try:
                search_after = json.loads(off)
            except json.JSONDecodeError:
                search_after = None

        scroll_body: Dict[str, Any] = {
            "query": base_q,
            "size": page_limit,
            "sort": [{"_id": {"order": "asc"}}],
            "_source": [PAYLOAD_FIELD],
        }
        if search_after:
            scroll_body["search_after"] = search_after
        try:
            resp = self._client.search(index=idx, body=scroll_body)
        except TypeError:
            # elasticsearch-py 8.x prefers explicit kwargs over body=
            resp = self._client.search(
                index=idx,
                query=scroll_body["query"],
                size=scroll_body["size"],
                sort=scroll_body["sort"],
                source=scroll_body.get("_source"),
                search_after=scroll_body.get("search_after"),
            )
        except Exception as e:
            if _is_404(e):
                return ScrollResult(points=[], next_offset="")
            raise

        hits = (resp.get("hits") or {}).get("hits") or []
        points: List[ScrolledPointOut] = []
        for h in hits:
            src = h.get("_source") or {}
            pl = src.get(PAYLOAD_FIELD) or {}
            points.append(ScrolledPointOut(id=str(h.get("_id", "")), payload=dict(pl)))

        next_off = ""
        if hits:
            sa = hits[-1].get("sort")
            if sa is not None:
                next_off = json.dumps(sa)

        return ScrollResult(points=points, next_offset=next_off)

    def count_vectors(self, collection_name: str, filters: List[VectorFilterInput]) -> int:
        if not self._client:
            return 0
        idx = self._physical_index_name(collection_name)
        if not self._index_exists(idx):
            return 0
        fq = _es_filter_from_filters(filters, "search")
        q = fq if fq else {"match_all": {}}
        try:
            if self._is_opensearch:
                r = self._client.count(index=idx, body={"query": q})
            else:
                r = self._client.count(index=idx, query=q)
            return int(r.get("count", 0))
        except Exception as e:
            if _is_404(e):
                return 0
            raise

    def delete_vectors_equality(
        self, collection_name: str, filters: List[VectorFilterInput]
    ) -> Tuple[int, Optional[str]]:
        if not self._client:
            return 0, "Vector store not initialized"
        if not filters:
            return 0, "No filters provided - use DeleteCollection for full deletion"
        idx = self._physical_index_name(collection_name)
        fq = _es_filter_from_filters(filters, "equality")
        if not fq:
            return 0, "Invalid filters for delete"
        try:
            if self._is_opensearch:
                r = self._client.delete_by_query(
                    index=idx,
                    body={"query": fq},
                    refresh=True,
                )
            else:
                r = self._client.delete_by_query(index=idx, query=fq, refresh=True)
            deleted = int(r.get("deleted", 0))
            return deleted, None
        except Exception as e:
            if _is_404(e):
                return 0, None
            return 0, str(e)

    def update_metadata_equality(
        self,
        collection_name: str,
        filters: List[VectorFilterInput],
        metadata_updates: Dict[str, Any],
    ) -> Tuple[int, Optional[str]]:
        if not self._client:
            return 0, "Vector store not initialized"
        idx = self._physical_index_name(collection_name)
        fq = _es_filter_from_filters(filters, "equality")
        if not fq:
            return 0, "Invalid filters for update"
        try:
            if self._is_opensearch:
                resp = self._client.search(
                    index=idx,
                    body={"query": fq, "size": 10000, "_source": True},
                )
            else:
                resp = self._client.search(index=idx, query=fq, size=10000)
        except Exception as e:
            if _is_404(e):
                return 0, None
            return 0, str(e)

        hits = (resp.get("hits") or {}).get("hits") or []
        if not hits:
            return 0, None

        actions: List[Dict[str, Any]] = []
        for h in hits:
            src = dict(h.get("_source") or {})
            pl = dict(src.get(PAYLOAD_FIELD) or {})
            pl.update(metadata_updates)
            src[PAYLOAD_FIELD] = pl
            actions.append({"index": {"_index": idx, "_id": h.get("_id")}})
            actions.append(src)

        if self._is_opensearch:
            from opensearchpy.helpers import bulk as os_bulk

            ok, errors = os_bulk(
                self._client,
                actions,
                refresh="wait_for",
                raise_on_error=False,
            )
        else:
            from elasticsearch.helpers import bulk as es_bulk

            ok, errors = es_bulk(
                self._client,
                actions,
                refresh="wait_for",
                raise_on_error=False,
            )
        if errors:
            return 0, str(errors[:2])
        return ok, None
