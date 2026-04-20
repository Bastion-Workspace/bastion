# Alternative Vector Database Backend Plan

For **deployment and environment variables** aimed at operators, see [`docs/VECTOR_STORE_BACKENDS.md`](../VECTOR_STORE_BACKENDS.md). This document is the engineering plan and architecture reference.

## Overview

Introduce a `VectorBackend` protocol inside the vector-service that abstracts all Qdrant SDK calls behind a common interface, with three implementations: `QdrantBackend` (wrapping current behavior), `MilvusBackend` (pymilvus), and `ElasticsearchBackend` (compatible with both Elasticsearch 8.x and OpenSearch 2.x). This enables corporate deployments that already operate Milvus or Elasticsearch/OpenSearch clusters to use Bastion without adding Qdrant to their infrastructure.

## Motivation

Qdrant is the best-in-class choice for Bastion's specific usage pattern (lightweight single-binary, native dense + sparse named vectors, RRF fusion). But enterprise customers often mandate use of their existing vector-capable infrastructure:

- **Elasticsearch/OpenSearch**: Already deployed for log aggregation, full-text search, or compliance indexing. Both now support dense k-NN (HNSW) and BM25 natively, with RRF fusion (Elasticsearch 8.8+, OpenSearch 2.10+).
- **Milvus**: Deployed for ML/AI workloads. Added sparse vector support in 2.4 with `SPARSE_FLOAT_VECTOR` and native `RRFRanker` for hybrid queries.

Neither alternative is *better* than Qdrant for our use case, but supporting them removes a procurement/security-review blocker for enterprise adoption.

## Current State

### Qdrant Integration Surface

All Qdrant SDK calls are isolated to a single file: `vector-service/service/grpc_service.py`. The backend communicates via gRPC through `VectorServiceClient` and never touches Qdrant directly.

| Operation | Qdrant SDK method | Notes |
|---|---|---|
| Create collection | `create_collection` | Unnamed dense, named dense, or named hybrid (dense + sparse) |
| Delete collection | `delete_collection` | — |
| List collections | `get_collections` | Returns names + point counts |
| Get collection info | `get_collection` | Vector size, distance, schema type |
| Upsert | `upsert` | Batched (100 pts), retry with backoff on timeout/5xx |
| Search (dense) | `search` | With `using="dense"` for named-vector collections |
| Search (hybrid RRF) | `query_points` | `Prefetch` dense + sparse, `FusionQuery(Fusion.RRF)` |
| Scroll | `scroll` | Paginated, with filter, optional vectors |
| Count | `count` | With filter, `exact=True` |
| Delete by filter | `delete` | `points_selector=Filter` |
| Update metadata | scroll + merge payload + `upsert` | No native `set_payload` used |

### Feature Requirements Matrix

Any alternative backend must support these capabilities:

| Capability | Required | Notes |
|---|---|---|
| Dense vector k-NN (cosine, dot, euclidean) | Yes | 128–3072 dimensions across collections |
| Sparse vectors (BM25) | Yes | Arbitrary indices + values, not engine-managed BM25 |
| Hybrid search (dense + sparse, RRF fusion) | Yes | Server-side RRF preferred; app-side RRF acceptable |
| Named/multi-vector fields | Yes | Dense + sparse on same record |
| Payload/metadata storage (JSON-compatible) | Yes | Arbitrary key-value, lists, nested objects |
| Payload filtering (equals, not_equals, any_of) | Yes | Must filter during search, not post-filter |
| Paginated scan/scroll | Yes | With filter, offset-based continuation |
| Exact count with filter | Yes | For audit/sync operations |
| Delete by metadata filter | Yes | Delete all points matching a payload condition |
| Collection CRUD | Yes | Create, delete, list, get info |
| Batch upsert | Yes | 100+ points per call |
| Mixed point ID types | Yes | Integer (content hash) and string (UUID) IDs |

### What the gRPC Layer Already Abstracts

The proto contract (`vector_service.proto`) is backend-agnostic by design:

- `VectorPoint` carries dense vector + optional `SparseVector` + string-encoded payload
- `VectorFilter` uses generic operators (`equals`, `not_equals`, `any_of`, `in`)
- `SearchVectorsRequest` has `fusion_mode` string, not a Qdrant-specific enum
- `CollectionInfo` reports `schema_type` as a string, not a Qdrant model

**No proto changes are needed.** The abstraction boundary sits entirely within `grpc_service.py`.

---

## Proposed Architecture

### VectorBackend Protocol

New file: `vector-service/service/backends/base.py`

```python
from typing import Protocol, Optional, Dict, List, Any, Tuple

class VectorBackend(Protocol):
    async def initialize(self, config: dict) -> None: ...
    async def health_check(self) -> bool: ...

    # Collection lifecycle
    async def create_collection(
        self, name: str, vector_size: int,
        distance: str, enable_sparse: bool,
    ) -> None: ...
    async def delete_collection(self, name: str) -> None: ...
    async def list_collections(self) -> List[CollectionInfo]: ...
    async def get_collection_info(self, name: str) -> Optional[CollectionInfo]: ...
    async def collection_exists(self, name: str) -> bool: ...

    # Point operations
    async def upsert(
        self, collection: str,
        points: List[VectorPoint],
    ) -> int: ...
    async def delete_by_filter(
        self, collection: str,
        filters: List[VectorFilter],
    ) -> int: ...

    # Search
    async def search_dense(
        self, collection: str,
        query_vector: List[float],
        limit: int,
        filters: Optional[List[VectorFilter]] = None,
        score_threshold: float = 0.0,
    ) -> List[SearchResult]: ...
    async def search_hybrid_rrf(
        self, collection: str,
        dense_vector: List[float],
        sparse_vector: SparseVectorData,
        limit: int,
        filters: Optional[List[VectorFilter]] = None,
        prefetch_multiplier: int = 3,
    ) -> List[SearchResult]: ...

    # Scan / audit
    async def scroll(
        self, collection: str,
        filters: Optional[List[VectorFilter]] = None,
        limit: int = 256,
        offset: Optional[str] = None,
        with_vectors: bool = False,
    ) -> Tuple[List[ScrolledPoint], Optional[str]]: ...
    async def count(
        self, collection: str,
        filters: Optional[List[VectorFilter]] = None,
    ) -> int: ...
```

Internal data classes (`VectorPoint`, `VectorFilter`, `SearchResult`, `SparseVectorData`, `CollectionInfo`, `ScrolledPoint`) live alongside the protocol. These mirror the proto messages but are plain Python — the gRPC handler converts between proto and these types.

### File Layout

```
vector-service/service/
    backends/
        __init__.py
        base.py              # VectorBackend protocol + data classes
        qdrant_backend.py    # Extract from current grpc_service.py
        milvus_backend.py    # pymilvus implementation
        es_backend.py        # elasticsearch-py / opensearch-py implementation
        factory.py           # get_vector_backend() from VECTOR_DB_BACKEND setting
```

### grpc_service.py Refactor

`VectorServiceImplementation` drops all `qdrant_client` imports and instead holds a `VectorBackend` instance. Each gRPC handler becomes a thin adapter: parse proto → call backend method → build proto response. The ~1300-line file shrinks to ~400 lines of proto conversion + error handling.

---

## Backend Implementation Details

### 1. QdrantBackend

**Goal:** Extract existing logic verbatim into the protocol shape. Zero behavioral change.

This is a mechanical refactoring of the current `grpc_service.py`:
- `_build_qdrant_filter_from_vector_filters` → internal to `QdrantBackend`
- `_classify_collection_vector_schema` cache → internal to `QdrantBackend`
- `_ensure_collection_for_vectors` → called within `upsert`
- Retry logic on upsert (timeout/5xx backoff) → preserved within `upsert`

**Validation checklist:**
- [ ] All existing integration tests pass with `VECTOR_DB_BACKEND=qdrant`
- [ ] Schema cache (`_collection_vector_schema`) still works for unnamed/named_dense/named_hybrid detection
- [ ] Retry behavior on upsert is identical (exponential backoff, max retries)
- [ ] `UnexpectedResponse` 404 handling returns empty results (not errors) for search/scroll/count
- [ ] The `QDRANT_API_KEY` setting is wired into client construction (currently missing — fix as part of extraction)

### 2. MilvusBackend

**SDK:** `pymilvus` (latest stable, currently 2.4.x+)

#### Concept Mapping

| Qdrant concept | Milvus equivalent | Notes |
|---|---|---|
| Collection | Collection | Same name |
| Point | Entity | — |
| Point ID (int or string) | Primary key field | Milvus requires a declared PK schema field. Use `VARCHAR(64)` to support both integer strings and UUIDs |
| Dense vector (unnamed) | `FloatVector` field | Named `dense` |
| Dense vector (named) | `FloatVector` field | Same — Milvus always uses named fields |
| Sparse vector | `SPARSE_FLOAT_VECTOR` field | Named `sparse` |
| Payload (arbitrary JSON) | `JSON` field type | Single `payload` field of type `JSON` |
| `VectorParams(size, distance)` | `FloatVector(dim)` + index params | Distance set at index creation, not collection creation |
| `SparseVectorParams()` | `SPARSE_FLOAT_VECTOR` field | No params needed |
| `Filter(must=[FieldCondition(...)])` | Boolean expression string | `payload["field"] == "value"` |
| `search()` | `collection.search()` | With `anns_field="dense"` |
| `query_points` + RRF | `collection.hybrid_search()` + `RRFRanker()` | Native in pymilvus 2.4+ |
| `scroll` | `collection.query()` with `offset` + `limit` | Milvus `query` is the equivalent of scroll |
| `count` | `collection.query(output_fields=["count(*)"])` | Or `collection.num_entities` for unfiltered |
| `delete(points_selector=Filter)` | `collection.delete(expr=...)` | Boolean expression |

#### Collection Schema

Every Milvus collection requires an explicit schema:

```python
from pymilvus import CollectionSchema, FieldSchema, DataType

def _build_schema(vector_size: int, enable_sparse: bool) -> CollectionSchema:
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema(name="dense", dtype=DataType.FLOAT_VECTOR, dim=vector_size),
        FieldSchema(name="payload", dtype=DataType.JSON),
    ]
    if enable_sparse:
        fields.append(
            FieldSchema(name="sparse", dtype=DataType.SPARSE_FLOAT_VECTOR)
        )
    return CollectionSchema(fields=fields)
```

#### Index Creation

Milvus requires explicit index creation after collection creation (unlike Qdrant which auto-indexes):

```python
# Dense index
index_params = {
    "metric_type": distance_map[distance],  # "COSINE", "IP", "L2"
    "index_type": "HNSW",
    "params": {"M": 16, "efConstruction": 200},
}
collection.create_index("dense", index_params)

# Sparse index (if hybrid)
if enable_sparse:
    sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
    collection.create_index("sparse", sparse_index)
```

#### Hybrid Search (RRF)

```python
from pymilvus import AnnSearchRequest, RRFRanker

dense_req = AnnSearchRequest(
    data=[query_vector], anns_field="dense",
    param={"metric_type": "COSINE", "params": {"ef": 128}},
    limit=prefetch_limit,
)
sparse_req = AnnSearchRequest(
    data=[sparse_dict], anns_field="sparse",
    param={"metric_type": "IP"},
    limit=prefetch_limit,
)
results = collection.hybrid_search(
    reqs=[dense_req, sparse_req],
    ranker=RRFRanker(k=60),
    limit=limit,
    expr=filter_expr,
    output_fields=["payload"],
)
```

#### Filter Translation

Milvus uses string expressions over JSON fields:

| VectorFilter operator | Milvus expression |
|---|---|
| `equals` | `payload["field"] == "value"` |
| `not_equals` | `payload["field"] != "value"` |
| `any_of` | `payload["field"] in ["v1", "v2", "v3"]` |
| `in` | `payload["field"] == "value"` |

Combine with `and`/`or` — all `must` conditions are `and`-joined, `must_not` are `not(...)`.

#### Things to Validate During Implementation

- [ ] **Load state management**: Milvus collections must be `load()`ed into memory before search. The backend must call `collection.load()` after creation and on startup for existing collections. Verify this doesn't blow memory with many small collections (face_encodings, object_features, per-user, per-team).
- [ ] **Schema immutability**: Milvus collections cannot add fields after creation. The `enable_sparse` decision is permanent. Verify the schema detection logic correctly identifies hybrid vs. dense-only collections and handles the recreation path.
- [ ] **Sparse vector format**: pymilvus expects a `dict[int, float]` for sparse vectors, not separate `indices` + `values` lists. The backend must convert from our `SparseVectorData(indices, values)` format.
- [ ] **VARCHAR primary key**: Milvus VARCHAR PK has a `max_length`. Verify 64 chars is sufficient for all point ID patterns (content hashes as int strings, UUIDs).
- [ ] **JSON field filtering performance**: Milvus JSON field filtering uses a scan, not an inverted index (as of 2.4). For collections with >100K points, payload-filtered searches may be slower than Qdrant. Benchmark with realistic data sizes.
- [ ] **Scroll/offset pagination**: Milvus `query()` supports `offset` + `limit` but has a max offset cap (configurable, default 16384). For collections with more points, the audit scroll path needs iterator-based pagination (`collection.query_iterator()`).
- [ ] **Count with filter**: `collection.query(expr=filter, output_fields=["count(*)"])` returns the count. Verify this works with complex JSON field expressions.
- [ ] **Delete by filter**: `collection.delete(expr=...)` works on loaded collections. Verify JSON field expressions work in delete.
- [ ] **Distance mapping**: Qdrant uses `COSINE`/`DOT`/`EUCLID`. Milvus uses `COSINE`/`IP`/`L2`. Map correctly — `DOT` → `IP`, `EUCLIDEAN` → `L2`.
- [ ] **Connection management**: pymilvus uses a connection alias system. Ensure cleanup on shutdown, and handle reconnection on transient failures.
- [ ] **Consistency level**: Milvus defaults to `Bounded` consistency. For our use case (upsert then immediately search), verify whether `Strong` consistency is needed or if `Bounded` with a flush is sufficient.
- [ ] **Auto-flush behavior**: pymilvus 2.4+ auto-flushes on search. Verify upserted data is searchable without explicit `flush()` calls, or add flush after upsert batches.

#### Milvus Docker Compose

```yaml
milvus-standalone:
  image: milvusdb/milvus:v2.4-latest
  container_name: ${COMPOSE_PROJECT_NAME:-bastion}-milvus
  command: ["milvus", "run", "standalone"]
  environment:
    ETCD_ENDPOINTS: etcd:2379
    MINIO_ADDRESS: minio:9000
  volumes:
    - bastion_milvus_data:/var/lib/milvus
  ports:
    - "19530:19530"
    - "9091:9091"
  depends_on:
    - etcd
    - minio

etcd:
  image: quay.io/coreos/etcd:v3.5.5
  container_name: ${COMPOSE_PROJECT_NAME:-bastion}-milvus-etcd
  environment:
    ETCD_AUTO_COMPACTION_MODE: revision
    ETCD_AUTO_COMPACTION_RETENTION: "1000"
    ETCD_QUOTA_BACKEND_BYTES: "4294967296"
  volumes:
    - bastion_milvus_etcd:/etcd
  command: >
    etcd
    --advertise-client-urls=http://0.0.0.0:2379
    --listen-client-urls=http://0.0.0.0:2379
    --data-dir=/etcd
```

**Operational note:** Milvus standalone requires etcd + MinIO (or local storage). This is heavier than Qdrant's single binary. For deployments that don't already have these, Qdrant remains the recommended default. If the deployment already uses S3-compatible storage (from the S3 storage backend plan), Milvus can share it.

### 3. ElasticsearchBackend (Elasticsearch 8.x / OpenSearch 2.x)

**SDK:** `elasticsearch[async]` for Elasticsearch, `opensearch-py` for OpenSearch. Both are nearly identical in API. Use a thin compatibility shim or conditional import.

#### Concept Mapping

| Qdrant concept | Elasticsearch/OpenSearch equivalent | Notes |
|---|---|---|
| Collection | Index | Same name, lowercase (ES requires lowercase) |
| Point | Document | — |
| Point ID | `_id` | String; integer IDs stored as strings |
| Dense vector | `dense_vector` field type | `dims`, `similarity` set at mapping time |
| Sparse vector | `rank_features` or `sparse_vector` field | ES 8.11+ supports `sparse_vector`; OpenSearch uses `rank_features` |
| Payload | Flat or nested fields in the mapping | Not a single JSON blob — each filterable field needs its own mapping |
| Filter | `bool` query with `must`/`must_not` + `term`/`terms` | — |
| `search()` | `knn` query (ES 8.x) / `neural` query (OpenSearch) | — |
| Hybrid RRF | `retriever.rrf` (ES 8.8+) / `search_pipeline` (OpenSearch 2.10+) | Different API shapes, same concept |
| `scroll` | `search_after` + `point_in_time` (preferred) or `scroll` API | `search_after` is stateless, preferred |
| `count` | `_count` API | With query filter |
| `delete(filter)` | `delete_by_query` | With bool query filter |
| `create_collection` | `PUT /{index}` with mappings | — |

#### Index Mapping

```json
{
  "mappings": {
    "properties": {
      "dense": {
        "type": "dense_vector",
        "dims": 3072,
        "index": true,
        "similarity": "cosine"
      },
      "sparse": {
        "type": "sparse_vector"
      },
      "payload": {
        "type": "object",
        "enabled": true
      }
    }
  },
  "settings": {
    "index": {
      "number_of_shards": 1,
      "number_of_replicas": 0,
      "knn": true
    }
  }
}
```

**Payload field strategy:** Elasticsearch requires fields to be mapped before filtering. Two approaches:

1. **Dynamic mapping (recommended for us):** Set `"dynamic": "true"` so ES auto-creates field mappings when new payload fields appear. All string values become `keyword` (filterable) by default with dynamic templates.
2. **Explicit mapping:** Pre-declare known payload fields (`document_id`, `user_id`, `filename`, etc.). More predictable but requires schema evolution management.

Use dynamic mapping with a template that maps string fields as `keyword`:

```json
{
  "dynamic_templates": [
    {
      "payload_strings": {
        "path_match": "payload.*",
        "match_mapping_type": "string",
        "mapping": { "type": "keyword" }
      }
    }
  ]
}
```

#### Hybrid Search (RRF)

**Elasticsearch 8.8+:**

```python
body = {
    "retriever": {
        "rrf": {
            "retrievers": [
                {
                    "standard": {
                        "query": {
                            "knn": {
                                "field": "dense",
                                "query_vector": query_vector,
                                "num_candidates": prefetch_limit,
                            }
                        }
                    }
                },
                {
                    "standard": {
                        "query": {
                            "sparse_vector": {
                                "field": "sparse",
                                "query_vector": sparse_dict,
                            }
                        }
                    }
                },
            ],
            "rank_window_size": prefetch_limit,
        }
    },
    "size": limit,
    "post_filter": filter_query,
}
```

**OpenSearch 2.10+ (search pipeline approach):**

```python
# Create pipeline (one-time setup per index)
pipeline = {
    "description": "RRF hybrid search",
    "phase_results_processors": [
        {
            "normalization-processor": {
                "normalization": {"technique": "min_max"},
                "combination": {"technique": "arithmetic_mean", "parameters": {"weights": [0.5, 0.5]}},
            }
        }
    ],
}

# Search with pipeline
body = {
    "query": {
        "hybrid": {
            "queries": [
                {"knn": {"dense": {"vector": query_vector, "k": prefetch_limit}}},
                {"neural_sparse": {"sparse": {"query_text": ..., "model_id": ...}}},
            ]
        }
    },
    "size": limit,
}
```

**Important caveat:** OpenSearch's `neural_sparse` expects a model ID for server-side sparse encoding. Since we bring our own BM25 sparse vectors, we may need to use `script_score` with a dot-product over `rank_features` instead. This is a key area requiring validation.

#### Filter Translation

| VectorFilter operator | ES/OpenSearch query |
|---|---|
| `equals` | `{"term": {"payload.field": "value"}}` |
| `not_equals` | `{"bool": {"must_not": [{"term": {"payload.field": "value"}}]}}` |
| `any_of` | `{"terms": {"payload.field": ["v1", "v2", "v3"]}}` |
| `in` | `{"term": {"payload.field": "value"}}` |

Wrapped in a `bool` query with `must` / `must_not` clauses, matching the current Qdrant filter structure.

#### Things to Validate During Implementation

- [ ] **Index name constraints**: Elasticsearch requires lowercase index names. Collection names like `user_ABC_documents` must be lowercased. Verify no collision with this transformation (e.g., `user_abc_documents` vs. `user_Abc_documents`).
- [ ] **Sparse vector field support**: ES 8.11+ has `sparse_vector` type natively. For ES 8.8–8.10, use `rank_features`. OpenSearch 2.x uses `rank_features`. The backend must detect the engine version and choose the right field type.
- [ ] **Sparse vector ingestion format**: ES `sparse_vector` and `rank_features` both expect `{"token_index_as_string": score}` — a dict with string keys. Our BM25 encoder produces integer indices. The backend must convert `{42: 0.85}` to `{"42": 0.85}`. Verify this round-trips correctly in search.
- [ ] **RRF API differences**: Elasticsearch uses `retriever.rrf` (8.8+). OpenSearch uses `search_pipeline` with normalization processors (2.10+). These are completely different APIs. The backend needs an internal `_is_opensearch` flag and separate search paths.
- [ ] **OpenSearch sparse search without model_id**: OpenSearch's `neural_sparse` assumes server-side encoding. For pre-computed sparse vectors, use `script_score` with dot-product, or store as `rank_features` and use `rank_feature` query. Test both approaches for correctness and performance.
- [ ] **k-NN exact vs. approximate**: ES uses HNSW by default (approximate). For small collections (face_encodings: 128 dims, ~100 points), exact search may be more appropriate. Check if ES auto-selects exact for small indices or if we need `"exact": true`.
- [ ] **Scroll implementation**: Use `search_after` with a PIT (point in time) for stateless pagination, not the deprecated `scroll` API. The backend must open/close PIT contexts. Verify PIT works with filtered k-NN queries.
- [ ] **Dynamic mapping and payload lists**: When a payload value is a JSON-encoded list (e.g., `tags_json`), ES may auto-map it as an array of keywords. Verify `terms` filter works correctly against these array-mapped fields.
- [ ] **Score normalization**: Qdrant cosine search returns scores in [0, 1]. ES k-NN scores vary by engine version and similarity function. RRF scores are rank-based. Verify the `score` returned in `SearchResult` is comparable across backends, or document that it's not.
- [ ] **Max result window**: ES defaults to `max_result_window: 10000`. For scroll/audit operations on large collections, this may need to be raised or `search_after` must be used.
- [ ] **Bulk API for upsert**: Use ES `_bulk` API for batch upserts (not individual index calls). Map our 100-point batch size to bulk operations.
- [ ] **Delete by query**: ES `delete_by_query` is eventually consistent. After deleting, an immediate count may still show the old value. Verify this doesn't break audit workflows; add `refresh=true` parameter if needed.
- [ ] **Connection pooling**: `elasticsearch[async]` uses `aiohttp` under the hood. Configure connection pool size appropriate for our concurrency level (multiple gRPC handlers in parallel).
- [ ] **Authentication**: ES/OpenSearch clusters often require auth (API key, basic auth, or AWS IAM for managed OpenSearch). The backend must support all three. Configuration: `ES_API_KEY`, `ES_USERNAME` + `ES_PASSWORD`, or AWS credential chain.

#### ES/OpenSearch Compatibility Shim

The two engines diverge on hybrid search, sparse vectors, and some API paths. Handle this with a strategy object:

```python
class ESCompatibility:
    """Elasticsearch 8.x-specific query builders."""
    def build_hybrid_query(self, ...): ...
    def sparse_field_type(self) -> str: return "sparse_vector"

class OpenSearchCompatibility:
    """OpenSearch 2.x-specific query builders."""
    def build_hybrid_query(self, ...): ...
    def sparse_field_type(self) -> str: return "rank_features"
```

Selected at init time based on the server's `info()` response (ES returns `"version": {"distribution": ...}"`; OpenSearch identifies itself differently).

---

## Configuration

New environment variables in `vector-service/config/settings.py`:

| Variable | Default | Description |
|---|---|---|
| `VECTOR_DB_BACKEND` | `qdrant` | `qdrant`, `milvus`, or `elasticsearch` (implemented) |
| **Qdrant (existing)** | | |
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant HTTP endpoint |
| `QDRANT_API_KEY` | `""` | API key (currently defined but not used — fix) |
| `QDRANT_TIMEOUT` | `30` | Request timeout in seconds |
| **Milvus** | | |
| `MILVUS_URI` | `http://milvus:19530` | Milvus gRPC endpoint |
| `MILVUS_TOKEN` | `""` | Auth token (for Zilliz Cloud or token-protected standalone) |
| `MILVUS_DB_NAME` | `default` | Database name (Milvus 2.4+ supports multi-DB) |
| `MILVUS_CONSISTENCY_LEVEL` | `Bounded` | `Strong`, `Bounded`, `Session`, or `Eventually` |
| **Elasticsearch / OpenSearch** | | |
| `ES_URL` | `http://elasticsearch:9200` | Cluster URL (comma-separated for multi-node) |
| `ES_API_KEY` | `""` | API key auth |
| `ES_USERNAME` | `""` | Basic auth username |
| `ES_PASSWORD` | `""` | Basic auth password |
| `ES_INDEX_PREFIX` | `bastion_` | Prefix for all index names (avoid collision with other indices) |
| `ES_VERIFY_CERTS` | `true` | TLS certificate verification |
| `ES_CA_CERTS` | `""` | Path to CA bundle for self-signed certs |

---

## Implementation Phases

### Phase 1: VectorBackend Protocol + QdrantBackend Extraction

**Goal:** Refactor `grpc_service.py` to use the protocol. Zero behavioral change.

**Scope:**
- Define `VectorBackend` protocol and data classes in `backends/base.py`
- Extract all Qdrant SDK code from `grpc_service.py` into `backends/qdrant_backend.py`
- Add `backends/factory.py` with `get_vector_backend()` that reads `VECTOR_DB_BACKEND` config
- Refactor `grpc_service.py` to delegate through `self.backend: VectorBackend`
- Fix existing bug: wire `QDRANT_API_KEY` into `QdrantClient` construction

**Risk:** Low — mechanical refactoring. Test by running the full system with `VECTOR_DB_BACKEND=qdrant` and verifying identical behavior.

**Validation:**
- [ ] All existing tests pass
- [ ] Schema cache behavior is preserved
- [ ] Retry logic on upsert is preserved
- [ ] 404 → empty results behavior is preserved for search/scroll/count/get_collection_info
- [ ] Health check still reports Qdrant connectivity

**Implementation status (done):** `VectorBackend` protocol and dataclasses in [`vector-service/service/backends/base.py`](vector-service/service/backends/base.py); `QdrantBackend` in [`vector-service/service/backends/qdrant_backend.py`](vector-service/service/backends/qdrant_backend.py); [`vector-service/service/backends/factory.py`](vector-service/service/backends/factory.py) with lazy import of `QdrantBackend`; [`vector-service/service/grpc_service.py`](vector-service/service/grpc_service.py) thin proto adapters; `VECTOR_DB_BACKEND` + startup validation in [`vector-service/config/settings.py`](vector-service/config/settings.py); `QDRANT_API_KEY` passed to `QdrantClient` when set; optional factory tests in [`vector-service/tests/test_vector_backend_factory.py`](vector-service/tests/test_vector_backend_factory.py). HealthCheck `details` includes `vector_db_backend` and `vector_store_configured`.

### Phase 2: MilvusBackend

**Goal:** Implement Milvus backend with full feature parity.

**Scope:**
- `backends/milvus_backend.py` implementing the protocol
- Add `pymilvus` to `vector-service/requirements.txt`
- Docker Compose profile for Milvus + etcd dependencies
- Integration tests against a Milvus standalone container

**Implementation order within Phase 2:**
1. Collection CRUD (create with schema, delete, list, info, exists)
2. Upsert (dense-only first, then hybrid)
3. Search dense
4. Search hybrid RRF
5. Scroll + count
6. Delete by filter
7. Metadata update (scroll + merge + upsert, same pattern as Qdrant)

**Risk:** Medium — Milvus has different semantics around schema declaration, index creation, and collection loading. Allow time for these behavioral differences.

**Implementation status (done):** [`vector-service/service/backends/milvus_backend.py`](vector-service/service/backends/milvus_backend.py) implements `VectorBackend` via pymilvus ORM (`connections` alias `bastion_vector`, `Collection`, `utility`). Schema: `id` (VARCHAR 64), `dense` (FLOAT_VECTOR), `payload` (JSON), optional `sparse` (SPARSE_FLOAT_VECTOR); HNSW + sparse inverted indexes; `hybrid_search` + `RRFRanker(k=60)` for RRF; filter expr over `payload["key"]`; scroll uses `query` / `query_iterator` when `offset+limit` exceeds 16384; `count(*)` with iterator fallback; `upsert` batches with retry. Settings: `MILVUS_URI`, `MILVUS_TOKEN`, `MILVUS_DB_NAME`, `MILVUS_CONSISTENCY_LEVEL`, `MILVUS_UPSERT_MAX_RETRIES`; factory supports `VECTOR_DB_BACKEND=milvus`. Compose profile `milvus` in [`docker-compose.example.yml`](docker-compose.example.yml). Unit tests: [`vector-service/tests/test_milvus_expr.py`](vector-service/tests/test_milvus_expr.py). **Known differences:** `CollectionInfo.schema_type` is `named_dense` or `named_hybrid` only (not `unnamed`); Milvus dense search ignores `score_threshold` (logged at debug); integration tests against live Milvus are left to CI / manual `--profile milvus` runs.

### Phase 3: ElasticsearchBackend

**Goal:** Implement ES/OpenSearch backend with full feature parity.

**Scope:**
- `backends/es_backend.py` implementing the protocol
- ES/OpenSearch compatibility shim for hybrid search and sparse field types
- Add `elasticsearch` (sync, 8.14–8.x) and `opensearch-py` to `vector-service/requirements.txt`
- Docker Compose profile for Elasticsearch
- Integration tests against both ES 8.x and OpenSearch 2.x containers

**Implementation order within Phase 3:**
1. Connection + auto-detection of ES vs. OpenSearch
2. Index CRUD (create with mapping, delete, list, info)
3. Upsert via `_bulk` API
4. Search dense (k-NN)
5. Search hybrid RRF (ES path first, then OpenSearch path)
6. Scroll via `search_after` + PIT
7. Count with filter
8. Delete by query
9. Metadata update (scroll + merge + bulk upsert)

**Risk:** Medium-high — ES and OpenSearch have diverged on hybrid search APIs. Sparse vector handling without server-side model is non-trivial on OpenSearch. Budget extra time for the OpenSearch compatibility path.

**Implementation status (done):** [`vector-service/service/backends/es_backend.py`](vector-service/service/backends/es_backend.py) implements `VectorBackend` with sync `elasticsearch` (8.14–8.x) and `opensearch-py` clients. `initialize()` probes with the Elasticsearch client, then switches to `OpenSearch` when `version.distribution == "opensearch"`. Index name: `{ES_INDEX_PREFIX}{collection}`.lower(); mapping: `dense` (`dense_vector`), `payload` (object + dynamic_templates for string→keyword), optional `sparse` as `sparse_vector` on Elasticsearch and dynamic `object` on OpenSearch for script_score dot-product in hybrid. Hybrid: Elasticsearch 8.14+ uses top-level `retriever` + `rrf` with `knn` + `sparse_vector` query (precomputed `query_vector`); OpenSearch 2.10+ uses `hybrid` query (kNN + `script_score` over `sparse.*` doc values) plus idempotent `_search/pipeline/bastion_vector_hybrid` (min_max + arithmetic_mean). Elasticsearch before 8.14 or OpenSearch before 2.10: hybrid falls back to dense-only with a warning. Scroll: `search_after` + opaque JSON `next_offset` (no PIT). Upsert: `_bulk` batches of 100, `refresh=wait_for`, retries. Settings: `ES_URL`, `ES_API_KEY`, `ES_USERNAME`/`ES_PASSWORD`, `ES_INDEX_PREFIX`, `ES_VERIFY_CERTS`, `ES_CA_CERTS`, `ES_UPSERT_MAX_RETRIES`; factory `VECTOR_DB_BACKEND=elasticsearch`. Compose profile `elasticsearch` in [`docker-compose.example.yml`](docker-compose.example.yml) (single-node ES 8.17). Unit tests: [`vector-service/tests/test_es_filter.py`](vector-service/tests/test_es_filter.py). **Known differences:** logical collection names are lowercased; `schema_type` is `named_dense` / `named_hybrid` only; dense `score_threshold` applied client-side after search; integration tests against live ES/OpenSearch are manual/CI.

### Phase 4: Testing, Documentation, Compose Profiles

**Goal:** Production-ready with tested migration paths.

**Scope:**
- Integration test suite that runs the same test cases against all three backends
- Docker Compose profiles: `--profile qdrant` (default), `--profile milvus`, `--profile elasticsearch`
- Configuration documentation in the admin guide
- Data migration utility: export from one backend, import to another (using scroll + upsert through the protocol, not vendor-specific dump/restore)

---

## Behavioral Differences to Document and Test

These are known semantic differences between backends that could cause subtle bugs if not handled:

| Behavior | Qdrant | Milvus | Elasticsearch |
|---|---|---|---|
| **Collection creation** | Immediate | Requires explicit `load()` before search | Immediate (yellow → green health) |
| **Upsert visibility** | Near-instant | Requires flush (auto-flush on search in 2.4+) | Near-real-time (~1s refresh) |
| **Exact count** | `count(exact=True)` | `query(count(*))` — exact | `_count` — exact |
| **Scroll offset type** | Integer or UUID point ID | Integer offset (with max cap) | `search_after` sort values |
| **Score range** | Cosine: [0, 1]; RRF: rank-based | Cosine: [0, 2] (IP-based); RRF: rank-based | Cosine: varies; RRF: rank-based |
| **Delete atomicity** | Synchronous | Synchronous | Eventually consistent |
| **Max batch size** | No hard limit | 16384 per insert | Bulk API size limit (~100MB) |
| **Field type evolution** | Payload is schemaless | Schema immutable after creation | Dynamic mapping + explicit mapping |
| **Index name constraints** | Any string | Any string | Lowercase only, no special chars |

### Score Normalization

RRF scores are inherently rank-based and backend-independent, so hybrid search scores should be comparable. Dense-only scores are not comparable across backends — document this clearly. Callers should not depend on absolute score values for cross-backend consistency.

---

## Estimated Effort

| Phase | Effort | Risk |
|---|---|---|
| Phase 1: Protocol + QdrantBackend extraction | 1 day | Low |
| Phase 2: MilvusBackend | 2–3 days | Medium |
| Phase 3: ElasticsearchBackend | 3–4 days | Medium-high |
| Phase 4: Testing + docs + compose profiles | 1–2 days | Low |
| **Total** | ~8–10 days | |

Phase 1 is prerequisite for Phases 2 and 3. Phases 2 and 3 are independent and can be developed in parallel by different people.

---

## Open Questions

1. **Sparse vector strategy for OpenSearch**: OpenSearch's `neural_sparse` assumes a model ID for server-side encoding. Our BM25 encoder produces client-side sparse vectors. The two options are: (a) store as `rank_features` and use `rank_feature` query with boosting per token, or (b) use a `script_score` query with dot-product. Option (a) is more idiomatic but may not support RRF directly. This needs a spike with a real OpenSearch instance before committing to an approach.

2. **Milvus dependency weight**: Milvus standalone requires etcd + object storage. For self-hosted deployments that don't already have these, the operational overhead may exceed the benefit of avoiding Qdrant. Consider documenting Milvus support as "bring your own Milvus" rather than shipping it in docker-compose by default.

3. **Multi-backend migration**: When switching backends (e.g., Qdrant → Elasticsearch), all vector data must be re-ingested. The protocol-level scroll + upsert migration tool handles this, but for large deployments (millions of points), this is a significant operation. Should we support a streaming migration that runs both backends in parallel during transition?

4. **Feature flag vs. exclusive backend**: The current plan assumes one backend at a time (`VECTOR_DB_BACKEND=elasticsearch`). An alternative is running two backends simultaneously with a routing layer (e.g., reads from ES, writes to both during migration). This adds complexity but enables zero-downtime backend switches. Defer unless customer demand materializes.

5. **pgvector**: If a future phase consolidates with PostgreSQL (pgvector for dense, `pg_search` / `tsvector` for sparse), the same `VectorBackend` protocol supports it. Not planned here because pgvector lacks native RRF fusion, but the architecture doesn't preclude it.

6. **Pinecone**: Cloud-only, no self-hosting. Some enterprise customers use managed Pinecone. Supporting it via the same protocol is feasible (Pinecone's API supports dense + sparse in a single upsert) but would require a separate plan for auth, namespaces, and serverless index management. Out of scope here.

## Related Architecture

- `vector-service/service/grpc_service.py` — gRPC adapters (vector ops delegate to `VectorBackend`)
- `vector-service/service/backends/qdrant_backend.py` — Qdrant SDK implementation
- `vector-service/service/backends/milvus_backend.py` — Milvus (pymilvus) implementation
- `protos/vector_service.proto` — gRPC contract (no changes needed)
- `vector-service/config/settings.py` — `VECTOR_DB_BACKEND`, Qdrant, Milvus, and Elasticsearch/OpenSearch connection settings
- `vector-service/service/backends/es_backend.py` — Elasticsearch / OpenSearch implementation
- `vector-service/service/bm25_encoder.py` — BM25 sparse vector generation (backend-independent)
- `CHANGELOG.md` (BM25 / hybrid search phases) and `vector-service/service/bm25_encoder.py` — sparse side of hybrid search alongside dense vectors
- `docs/dev-notes/S3_STORAGE_BACKEND_PLAN.md` — sibling plan using the same `Protocol` extraction pattern for storage backends
