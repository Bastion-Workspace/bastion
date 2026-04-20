# Vector store backends (administrator guide)

Bastion’s **vector-service** stores and searches embeddings through a single pluggable backend. You choose **one** backend at a time with `VECTOR_DB_BACKEND`. The backend app and other services talk to vector-service over gRPC; they do not connect to Qdrant, Milvus, or Elasticsearch directly.

| Backend | Value | Typical use |
|--------|--------|-------------|
| **Qdrant** | `qdrant` (default) | Single-binary or small cluster; native dense + sparse + RRF. |
| **Milvus** | `milvus` | Existing Milvus / Zilliz deployments. |
| **Elasticsearch / OpenSearch** | `elasticsearch` | Existing ES 8.14+ or OpenSearch 2.10+ clusters (auto-detected at runtime). |

## Requirements (all backends)

- **vector-service** must start with valid **embedding** settings (`EMBEDDING_PROVIDER`, API keys or URLs as required). See [`vector-service/config/settings.py`](../vector-service/config/settings.py).
- **`VECTOR_DB_BACKEND`** must be one of: `qdrant`, `milvus`, `elasticsearch` (case-insensitive). Startup validation fails if the backend-specific URL is missing.
- **Hybrid BM25 + dense search** (optional): set `HYBRID_SEARCH_ENABLED=true` on services that generate sparse vectors; collections must be created with sparse support where the backend supports it. See [`docs/dev-notes/VECTOR_DB_BACKEND_PLAN.md`](dev-notes/VECTOR_DB_BACKEND_PLAN.md) for behavior differences.

Health check details on vector-service include:

- `vector_db_backend` — active backend name.
- `vector_store_configured` — `true` when the backend client connected successfully (`is_available()`).

## Qdrant

**When to use:** Default Bastion layout; smallest operational footprint if you run Qdrant yourself or use a managed Qdrant.

| Variable | Description |
|----------|-------------|
| `VECTOR_DB_BACKEND` | `qdrant` (default). |
| `QDRANT_URL` | HTTP(S) base URL (e.g. `http://qdrant:6333` on Docker network, `http://localhost:6333` on host). |
| `QDRANT_API_KEY` | Optional API key for secured Qdrant. |
| `QDRANT_TIMEOUT` | Request timeout in seconds (default `30`). |
| `QDRANT_UPSERT_MAX_RETRIES` | Upsert batch retries (default `3`). |

**Docker:** The root [`docker-compose.example.yml`](../docker-compose.example.yml) does not start Qdrant by default. To run Qdrant in the same Compose project, merge the optional stack and override in [`docker/README.md`](../docker/README.md) (`compose.qdrant-stack.yml` + `compose.vector-qdrant.yml`).

**Validation:** With `VECTOR_DB_BACKEND=qdrant`, `QDRANT_URL` may be empty only for local experiments; the service will log that the vector store is unavailable until a URL is set.

## Milvus

**When to use:** You already run Milvus 2.4+ (standalone or cluster) with etcd and object storage as required by your deployment.

| Variable | Description |
|----------|-------------|
| `VECTOR_DB_BACKEND` | `milvus`. |
| `MILVUS_URI` | **Required.** gRPC/HTTP URI (e.g. `http://milvus:19530` inside Docker). |
| `MILVUS_TOKEN` | Optional (Zilliz Cloud / token auth). |
| `MILVUS_DB_NAME` | Database name (default `default`). |
| `MILVUS_CONSISTENCY_LEVEL` | e.g. `Bounded`, `Strong` (see Milvus docs). |
| `MILVUS_UPSERT_MAX_RETRIES` | Upsert retries (defaults follow `QDRANT_UPSERT_MAX_RETRIES` if unset). |

**Docker:** Start Milvus dependencies with Compose **profile** `milvus`, then apply the vector-service override from [`docker/compose.vector-milvus.yml`](../docker/compose.vector-milvus.yml). Exact commands are in [`docker/README.md`](../docker/README.md).

**Notes:**

- Collection names and schemas are managed by vector-service; hybrid collections need sparse-capable schemas at creation time.
- Known differences vs Qdrant are summarized in the [implementation plan](dev-notes/VECTOR_DB_BACKEND_PLAN.md#behavioral-differences-to-document-and-test).

## Elasticsearch and OpenSearch

**When to use:** Corporate standard is an existing Elasticsearch 8.x or OpenSearch 2.x cluster. The same code path is selected with `VECTOR_DB_BACKEND=elasticsearch`; the client inspects cluster `info()` and uses the appropriate wire behavior.

| Variable | Description |
|----------|-------------|
| `VECTOR_DB_BACKEND` | `elasticsearch`. |
| `ES_URL` | **Required.** Cluster URL(s); comma-separated for multiple nodes. |
| `ES_API_KEY` | Optional API key (Elasticsearch / compatible). |
| `ES_USERNAME` / `ES_PASSWORD` | Optional HTTP basic auth (if not using API key). |
| `ES_INDEX_PREFIX` | Prefix for index names (default `bastion_`). Logical collection names are lowercased for index names. |
| `ES_VERIFY_CERTS` | TLS verify (default `true`). |
| `ES_CA_CERTS` | Path to CA bundle for private TLS. |
| `ES_UPSERT_MAX_RETRIES` | Bulk upsert retries (default chains from `QDRANT_UPSERT_MAX_RETRIES`). |

**Version expectations:**

- **Hybrid RRF (dense + sparse):** Elasticsearch **8.14+** (retriever API). Older ES 8.x falls back to dense-only search with a warning.
- **OpenSearch:** Hybrid pipeline path targets **2.10+**. Older versions fall back to dense-only with a warning.

**Docker:** Use Compose profile `elasticsearch` from the example file plus [`docker/compose.vector-elasticsearch.yml`](../docker/compose.vector-elasticsearch.yml). See [`docker/README.md`](../docker/README.md).

## Switching backends

Vector data is **not** migrated automatically. Changing `VECTOR_DB_BACKEND` (or URLs pointing at an empty cluster) means **new empty** collections/indexes until you re-ingest from the application. Plan downtime or a reindex job; for a protocol-level migration approach, see Phase 4 in [`docs/dev-notes/VECTOR_DB_BACKEND_PLAN.md`](dev-notes/VECTOR_DB_BACKEND_PLAN.md).

## Compose examples and deeper architecture

| Resource | Purpose |
|----------|---------|
| [`docker/README.md`](../docker/README.md) | Commands to combine `docker-compose.example.yml` with backend-specific fragments. |
| [`docker-compose.example.yml`](../docker-compose.example.yml) | Full stack; Milvus and Elasticsearch are optional **profiles**. |
| [`docs/dev-notes/VECTOR_DB_BACKEND_PLAN.md`](dev-notes/VECTOR_DB_BACKEND_PLAN.md) | Architecture, phases, and engineer-oriented edge cases. |

## Related services

The **backend** uses `USE_VECTOR_SERVICE` and gRPC to call vector-service for embeddings and vector RPCs; it still needs a reachable Qdrant/Milvus/ES **only indirectly** via vector-service. Ensure `VECTOR_DB_BACKEND` and URLs are consistent across **vector-service** deployments (and any process that embeds vectors using the same dimension and hybrid flags).

For a service inventory, see [`docs/SERVICES.md`](SERVICES.md).
