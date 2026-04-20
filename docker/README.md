# Docker Compose fragments: vector store backends

These files are meant to be **merged** with the repository root [`docker-compose.example.yml`](../docker-compose.example.yml) using multiple `-f` flags. Always run commands from the **repository root** unless noted.

Compose **merges** `environment` maps: later files override keys from earlier files. List-style `environment:` blocks in the base file are replaced if an override uses a list—our fragments use **mapping** form so only listed keys are set/overridden.

## Qdrant (default backend)

**Option A — Qdrant already running elsewhere**

Set on `vector-service` (or in `.env`):

- `VECTOR_DB_BACKEND=qdrant`
- `QDRANT_URL=http://<host>:6333`

**Option B — Qdrant container in the same Compose project**

Starts a single-node Qdrant and points vector-service at it:

```bash
docker compose \
  -f docker-compose.example.yml \
  -f docker/compose.qdrant-stack.yml \
  -f docker/compose.vector-qdrant.yml \
  up -d
```

Files:

- [`compose.qdrant-stack.yml`](compose.qdrant-stack.yml) — `qdrant` service + volume.
- [`compose.vector-qdrant.yml`](compose.vector-qdrant.yml) — `vector-service` env: `VECTOR_DB_BACKEND=qdrant`, `QDRANT_URL=http://qdrant:6333`.

## Milvus

Milvus (etcd + MinIO + Milvus) is defined in `docker-compose.example.yml` under **profile** `milvus`. Apply the vector-service override so vector-service uses Milvus:

```bash
docker compose \
  -f docker-compose.example.yml \
  --profile milvus \
  -f docker/compose.vector-milvus.yml \
  up -d
```

File: [`compose.vector-milvus.yml`](compose.vector-milvus.yml) — sets `VECTOR_DB_BACKEND=milvus` and `MILVUS_URI=http://milvus:19530`.

Optional env for MinIO credentials (defaults exist): `MILVUS_MINIO_ACCESS_KEY`, `MILVUS_MINIO_SECRET_KEY` (see root compose).

## Elasticsearch

Elasticsearch is under **profile** `elasticsearch` in `docker-compose.example.yml`.

```bash
docker compose \
  -f docker-compose.example.yml \
  --profile elasticsearch \
  -f docker/compose.vector-elasticsearch.yml \
  up -d
```

File: [`compose.vector-elasticsearch.yml`](compose.vector-elasticsearch.yml) — sets `VECTOR_DB_BACKEND=elasticsearch` and `ES_URL=http://elasticsearch:9200`.

The bundled image disables security for **local development only**. For production, use your cluster’s TLS and `ES_API_KEY` or `ES_USERNAME` / `ES_PASSWORD`.

## Order of `-f` flags

Put **overrides last** so they win:

```text
-f docker-compose.example.yml  …  -f docker/compose.vector-<backend>.yml
```

## Documentation

Administrator-focused variable reference: [`docs/VECTOR_STORE_BACKENDS.md`](../docs/VECTOR_STORE_BACKENDS.md).
