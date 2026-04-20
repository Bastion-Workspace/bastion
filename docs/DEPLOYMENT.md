# Bastion deployment guide

This document is for **operators** who install and run Bastion: from a single-machine hobby setup through teams that use Compose in production, external databases, and registry-based deploys. It complements **`.env.example`** (field-by-field comments) and **`docker-compose.yml`** (wiring, ports, defaults).

---

## 1. What you are deploying

Bastion is a **multi-container** application: PostgreSQL (app metadata + LangGraph checkpoints), optional second Postgres for data workspaces, Redis, SearXNG, a FastAPI **backend**, **document-service**, **tools-service** (gRPC), **llm-orchestrator**, **vector-service**, **crawl4ai-service**, **data-service**, optional **connections-service**, **voice-service**, **image-vision-service**, **bbs-server**, Celery workers/beat/flower, and a **frontend** (nginx static build or Vite dev profile).

**External or bundled dependencies** (depending on compose file and env):

- **Qdrant** — vector search (URLs appear in compose; can be another host, another compose stack, or Kubernetes).
- **Neo4j** — knowledge graph (same).

The **default** [`docker-compose.yml`](../docker-compose.yml) in this repo assumes Qdrant and Neo4j **endpoints are reachable at the URLs set in that file** (adjust for your network or use a minimal compose overlay — see §4).

---

## 2. Requirements

### 2.1 Software

- **Docker Engine** and **Docker Compose v2** (Compose plugin).
- Enough **disk** for Postgres data, `uploads/` / `processed/`, optional PMTiles, Valhalla data, and embedding caches.
- **CPU / RAM**: usable from ~8 GB RAM for a light dev stack; production should be sized from concurrent users, Celery concurrency, `PARALLEL_WORKERS` on document-service, and Whisper/TTS if voice is enabled.

### 2.2 Network

- Outbound **HTTPS** for LLM providers, optional email/OAuth, and crawl features.
- Inbound ports depend on compose (typical: **3051** web UI, **8081** backend direct if exposed, **5555** Flower, **8002** Valhalla if enabled, **2222** BBS SSH if enabled). Restrict at the firewall in production.

### 2.3 Secrets and keys

- At minimum you need working **LLM / embedding** credentials where your stack is configured to use them (often `OPENAI_API_KEY`; see compose for `OPENROUTER_*`, etc.).
- **JWT** signing and app **SECRET_KEY** must be strong and unique in production.
- **`INTERNAL_SERVICE_KEY`** must match between **backend**, **connections-service**, and **bbs-server** when those services are used together.

---

## 3. Configuration sources (avoid drift)

| Source | Role |
|--------|------|
| [`.env.example`](../.env.example) | Annotated list of variables you commonly set in a project `.env`; **comments explain intent and pitfalls**. |
| [`docker-compose.yml`](../docker-compose.yml) | **Authoritative wiring**: which env vars are passed into which service, default literals, profiles, `depends_on`, ports. |
| [`docker-compose.example.yml`](../docker-compose.example.yml) | **Parameterized** compose: many passwords and URLs come from `.env` substitution — better template for **new** production-like stacks. |
| [README embedding section](../README.md) | **Embedding dimensions** (`EMBEDDING_DIMENSIONS`) and vector/tooling behavior — set in compose for several services even if not duplicated in `.env.example`. |

**Rule of thumb:** If a variable is not in `.env.example` but appears in `docker-compose.yml`, you can still add it to `.env` — Compose reads `.env` for interpolation.

---

## 4. Deployment scenarios

### 4.1 Default full stack (main compose)

**Command:** `docker compose up --build` (from repo root).

**Use when:** You already have Qdrant and Neo4j (or test endpoints) at the URLs expected by [`docker-compose.yml`](../docker-compose.yml), or you are developing against fixed lab IPs.

**Important — Postgres passwords in default compose:** The checked-in **`docker-compose.yml`** uses **fixed** literals such as `bastion_secure_password` for the app DB user, aligned with [`backend/sql/01_init.sql`](../backend/sql/01_init.sql). Those values are **not** read from `.env` in that file. Changing DB passwords for the default stack requires **consistent edits** across compose, pgbouncer healthchecks, and init SQL (or migrating to the example compose pattern).

**UI profile:** Set **`COMPOSE_PROFILES=prod`** in `.env` for the nginx frontend on port **3051**. For Vite HMR, use profile **`dev`** (`frontend-dev`) and **do not** activate `prod` at the same time (both bind **3051**). See comments at the top of `.env.example`.

### 4.2 Hardened / parameterized stack (`docker-compose.example.yml`)

**Use when:** You want **`POSTGRES_PASSWORD`**, **`POSTGRES_APP_PASSWORD`**, and related values driven from **`.env`** (recommended for anything internet-facing).

Follow the header comments in [`docker-compose.example.yml`](../docker-compose.example.yml). Keep **`POSTGRES_APP_PASSWORD`** consistent with the password your app user uses in `DATABASE_URL` / init expectations.

### 4.3 Minimal stack + external Qdrant and Neo4j

**Files:** [`docker-compose.minimal-external.yml`](../docker-compose.minimal-external.yml) + **`minimal-external.env.example`**.

**Command (from repo root):**

```bash
docker compose -f docker-compose.minimal-external.yml --env-file minimal-external.env.example up --build
```

**Use when:** You run Qdrant and Neo4j elsewhere (managed cloud, another VM, or `host.docker.internal`). Set **`QDRANT_URL`**, **`NEO4J_URI`**, and credentials in the env file.

### 4.4 Minimal stack + bundled Qdrant and Neo4j

**Merge overlay:** [`docker-compose.minimal-bundled-addons.yml`](../docker-compose.minimal-bundled-addons.yml) + **`minimal-bundled.env.example`**.

```bash
docker compose \
  -f docker-compose.minimal-external.yml \
  -f docker-compose.minimal-bundled-addons.yml \
  --env-file minimal-bundled.env.example \
  up --build
```

**Use when:** You want a **single** compose project that includes Qdrant and Neo4j containers (good for demos or small deployments without external clusters).

### 4.5 Pre-built images from GitHub Container Registry (GHCR)

On **`v*`** tags, CI builds and pushes first-party images (see [`.github/workflows/README.md`](../.github/workflows/README.md)).

**Image pattern:** `ghcr.io/<github_org_owner>/bastion-<service>:<tag>`  
Examples: `bastion-backend`, `bastion-document-service`, `bastion-tools-service`, …

**Typical operator pattern:**

1. Pull images by version tag (or `latest` / `latest-dev` per your policy).
2. Add a **compose override** (e.g. `docker-compose.override.yml`, not committed) that sets **`image:`** for each built service and **removes or replaces `build:`** so the host does not compile images.
3. Keep using the same **environment** and **volumes** as the main compose.

**UID/GID note:** CI builds with **`BASTION_RUNTIME_UID` / `BASTION_RUNTIME_GID` = 10001** in the image layers. Your compose **`user:`** line can still be overridden via `.env` for **runtime**, but bind mounts and named volumes must be **consistent** with the numeric owner you run as; mismatches cause permission or SQLite “readonly database” issues on caches.

---

## 5. `.env` topics (grouped)

The following groups match how **`.env.example`** is structured. Always verify each variable is **actually consumed** by the compose file you run (§3).

### 5.1 Compose and UI

| Area | Variables (examples) | Notes |
|------|----------------------|--------|
| Profiles | `COMPOSE_PROFILES` | `prod` = nginx UI; `dev` = Vite. Avoid both on same port. |

### 5.2 PostgreSQL (example / parameterized stacks)

| Area | Variables | Notes |
|------|-----------|--------|
| Superuser / app | `POSTGRES_PASSWORD`, `POSTGRES_APP_PASSWORD` | Must stay consistent with DB init and `DATABASE_URL` in compose. |

On the **default** `docker-compose.yml`, see §4.1 — literals may bypass these.

### 5.3 Auth and shared secrets

| Area | Variables | Notes |
|------|-----------|--------|
| JWT / session | `JWT_SECRET_KEY`, `SECRET_KEY` | Rotate for production; `SECRET_KEY` used for federation and other signing. |
| Service auth | `INTERNAL_SERVICE_KEY` | Same value on backend, connections-service, bbs-server. |
| GReader | `GREADER_API_ENABLED` | Optional mobile RSS API. |

### 5.4 Rootless runtime UID/GID

| Area | Variables | Notes |
|------|-----------|--------|
| Process owner | `BASTION_RUNTIME_UID`, `BASTION_RUNTIME_GID` (or `BASTION_RUNTIME_PUID` / `BASTION_RUNTIME_PGID`) | Compose passes these into **`user:`** and many **`build.args`**. Match host bind mounts (`./uploads`, `./logs`, …). After changes, **rebuild** app images and fix volume ownership. See comments in `.env.example` for vector embedding cache volume behavior. |

### 5.5 LLM and voice

| Area | Variables | Notes |
|------|-----------|--------|
| Chat / routing | `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `DEFAULT_MODEL` | Additional provider keys may be wired in compose. |
| Voice | `VOICE_TTS_PROVIDER`, `VOICE_STT_PROVIDER`, `ELEVENLABS_API_KEY` | Piper can reduce external dependencies. |

### 5.6 Neo4j and vector stack

| Area | Variables | Notes |
|------|-----------|--------|
| Optional behavior | `NEO4J_ENABLED`, `NEO4J_REQUIRED`, `NEO4J_RECONNECT_INTERVAL_SECONDS`, `NEO4J_USER`, `NEO4J_PASSWORD` | `*_REQUIRED=true` fails startup if dependency is down. |
| Vector | `VECTOR_EMBEDDING_ENABLED`, `VECTOR_EMBEDDING_REQUIRED`, backlog tuning | Lets stack start without vector process if configured. |
| Qdrant auth | `QDRANT_API_KEY` | When Qdrant is configured with API key. |

**URLs:** `QDRANT_URL`, `NEO4J_URI` are commonly set via compose or minimal env files — confirm in your chosen compose file.

### 5.7 Document pipeline and backend HTTP

| Area | Variables | Notes |
|------|-----------|--------|
| Document-service | `DOCUMENT_SERVICE_PARALLEL_WORKERS`, `DOCUMENT_SERVICE_MAX_INFLIGHT_POST_UPLOAD_TASKS` | Mapped in compose; tune with CPU and DB pool. |
| Advanced | `DS_MAX_CONCURRENT_DOCUMENTS`, `DS_THREAD_POOL_SIZE` | Often set via **override** (see `.env.example` comments). |
| Backend | `UVICORN_WORKERS` | >1 multiplies memory and DB connections. |

### 5.8 Image vision gRPC

| Area | Variables | Notes |
|------|-----------|--------|
| Flags | `IMAGE_VISION_ENABLED`, `IMAGE_VISION_REQUIRED` | Disable if service not deployed; required fails API startup when unhealthy. |

### 5.9 Data workspaces and SearXNG

| Area | Variables | Notes |
|------|-----------|--------|
| Data DB | `DATA_WORKSPACE_PASSWORD` | Used where compose wires postgres-data. |
| Metasearch | `SEARXNG_SECRET` | SearXNG container configuration. |

### 5.10 Microsoft / external connections

OAuth variables (`MICROSOFT_*`, GitHub OAuth in compose) — optional; used by external connections and Agent Factory integrations. See compose for exact names.

### 5.11 Maps and routing

| Area | Variables | Notes |
|------|-----------|--------|
| Basemap | `PMTILES_DATA_PATH`, `VITE_PMTILES_URL`, legacy `VITE_MAP_TILE_URL` | PMTiles volume mount for self-hosted tiles. |
| Routing | `VALHALLA_DATA_PATH`, `VALHALLA_BASE_URL`, `ROUTING_PROVIDER`, `OSRM_BASE_URL` | Valhalla service is optional in full compose. |

### 5.12 HTTP / CORS / debug

| Area | Variables | Notes |
|------|-----------|--------|
| Dev vs prod | `DEBUG`, `LOG_LEVEL`, `CORS_ORIGINS` | Tighten `CORS_ORIGINS` and turn `DEBUG` off in production. |

### 5.13 Federation and public URL

| Area | Variables | Notes |
|------|-----------|--------|
| URLs | `SITE_URL` | Public base URL; used for links and federation identity. |
| Federation | `FEDERATION_ENABLED`, `FEDERATION_DISPLAY_NAME`, tuning vars | Requires stable `SITE_URL` and `SECRET_KEY`. |

### 5.14 Orchestrator and Celery timeouts

Optional wall-clock limits: `PIPELINE_LLM_INVOKE_TIMEOUT_SEC`, `PLAYBOOK_GRAPH_INVOKE_TIMEOUT_SEC` (orchestrator env), `CELERY_ORCHESTRATOR_STREAM_TIMEOUT_SEC` (worker). See `.env.example` — unset or `0` usually means disabled.

### 5.15 BBS (telnet / SSH)

| Area | Variables | Notes |
|------|-----------|--------|
| Transports | `BBS_ENABLE_TELNET`, `BBS_ENABLE_SSH`, ports, `BBS_SSH_HOST_KEY` | At least one transport must be enabled in service env; host key persisted in named volume. |
| UX | `BBS_SCREEN_BLANK_AFTER_SECONDS` | Idle blanking; `0` disables. |

### 5.16 Embedding dimensions (critical for vector consistency)

`EMBEDDING_DIMENSIONS` is passed into **backend**, **Celery**, **vector-service**, and **tools-service** from compose. It must match your embedding model’s contract. See the **README** section *Embedding dimensions* for provider-specific rules and re-embedding implications.

---

## 6. Operations checklist (hobby → pro)

1. **Secrets:** Strong `JWT_SECRET_KEY`, `SECRET_KEY`, `INTERNAL_SERVICE_KEY`, DB passwords, API keys; store in a secret manager for production, not in shell history.
2. **TLS:** Terminate TLS at a reverse proxy (Traefik, nginx, cloud LB) in front of the web UI; do not expose Postgres/Redis/gRPC ports publicly unless strictly required and firewalled.
3. **Backups:** Postgres volumes (`bastion_*`), `uploads/`, `processed/`, and any PMTiles / Valhalla data you care about.
4. **Upgrades:** Pull new images or rebuild; run DB migrations if release notes require it; plan **downtime or rolling** strategy if you scale beyond one node.
5. **Health:** Backend `/health`, service healthchecks in compose, Flower for Celery queue inspection.
6. **Resource limits:** Use Compose `deploy.resources` or container runtime limits in production to cap memory/CPU per service.
7. **Observability:** Centralize logs from containers; set `LOG_LEVEL` appropriately.

---

## 7. Further reading

- [`.env.example`](../.env.example) — commented variable list  
- [`docker-compose.yml`](../docker-compose.yml) — default stack wiring  
- [`docker-compose.example.yml`](../docker-compose.example.yml) — parameterized template  
- [`docker-compose.minimal-external.yml`](../docker-compose.minimal-external.yml) — smaller footprint + external graph/search  
- [`.github/workflows/README.md`](../.github/workflows/README.md) — GHCR images and tag workflow  
- [`docs/SERVICES.md`](SERVICES.md) — service-oriented overview (ports, roles)

If something is missing here, **`docker-compose.yml` + `.env.example`** should be treated as the ground truth for your branch.
