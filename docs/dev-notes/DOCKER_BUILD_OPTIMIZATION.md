# Docker build optimization

Status tracker for container images: build context size, image layers, runtime security, and build-only dependency leakage.

## Principles

1. **`.dockerignore` at context root** — For `context: .`, exclude `.git/`, `node_modules/`, caches, logs, secrets, and large paths not needed by `COPY`. Docker only reads `.dockerignore` from the **context root**; nested `.dockerignore` files under subdirectories are **not** applied for a root context build.
2. **Multi-stage builds** — Compile / codegen / `pip install` with build toolchains in a builder stage; copy only wheels, site-packages, or generated artifacts into the final stage. Drop `grpcio-tools`, `build-essential`, `cmake`, etc. from the runtime image when possible.
3. **Non-root `USER`** — Run app processes as an unprivileged user unless the image must bind to privileged ports (not applicable for ports above 1024).
4. **`apt-get install -y --no-install-recommends`** — Reduces Debian image bloat from suggested packages.
5. **`HEALTHCHECK`** — Where the service exposes a TCP or HTTP port, add a lightweight check (aligned with [docker-compose.yml](../../docker-compose.yml) patterns used elsewhere).

## Per-image status

Legend: **Y** = done, **N** = not done, **—** = not applicable (e.g. no `apt-get`, or nginx official image).

| Image / Dockerfile | Multi-stage | Root `.dockerignore` (context `.`) | Non-root runtime | Build deps stripped from final | `apt` `--no-install-recommends` | `HEALTHCHECK` |
| --- | --- | --- | --- | --- | --- | --- |
| [backend/Dockerfile](../../backend/Dockerfile) | **Y** | **Y** | **Y** | **Y** (`grpcio-tools` build-only; dev tools in [requirements-dev.txt](../../backend/requirements-dev.txt)) | **Y** (runtime) | **Y** (HTTP `/health`) |
| [backend/Dockerfile.celery-worker](../../backend/Dockerfile.celery-worker) | **Y** | **Y** | **Y** | **Y** | **Y** (runtime) | N |
| [backend/Dockerfile.celery-beat](../../backend/Dockerfile.celery-beat) | **Y** | **Y** | **Y** | **Y** | **Y** (builder) | N |
| [backend/Dockerfile.celery-flower](../../backend/Dockerfile.celery-flower) | **Y** | **Y** | **Y** | **Y** | **Y** (builder) | N |
| [backend/Dockerfile.webdav](../../backend/Dockerfile.webdav) | N | **Y** | **Y** | **Y** (no apt / no compiler in image) | **—** | **Y** (TCP) |
| [tools-service/Dockerfile](../../tools-service/Dockerfile) | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** (TCP) |
| [llm-orchestrator/Dockerfile](../../llm-orchestrator/Dockerfile) | **Y** | **Y** | **Y** | **Y** (`grpcio-tools` in [requirements-build.txt](../../llm-orchestrator/requirements-build.txt)) | **Y** (builder) | **Y** (TCP) |
| [document-service/Dockerfile](../../document-service/Dockerfile) | **Y** | **Y** | **Y** | **Y** (`grpcio-tools` in [requirements-build.txt](../../document-service/requirements-build.txt)) | **Y** (runtime) | **Y** (TCP) |
| [vector-service/Dockerfile](../../vector-service/Dockerfile) | **Y** | **Y** | **Y** | **Y** | **Y** (builder) | **Y** |
| [data-service/Dockerfile](../../data-service/Dockerfile) | **Y** | **Y** | **Y** | **Y** | **Y** (builder) | **Y** |
| [connections-service/Dockerfile](../../connections-service/Dockerfile) | **Y** | **Y** | **Y** | **Y** (`grpcio-tools` in [requirements-build.txt](../../connections-service/requirements-build.txt)) | **Y** (builder) | **Y** (TCP) |
| [voice-service/Dockerfile](../../voice-service/Dockerfile) | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** |
| [image-vision-service/Dockerfile](../../image-vision-service/Dockerfile) | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** |
| [crawl4ai-service/Dockerfile](../../crawl4ai-service/Dockerfile) | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** |
| [cli-worker/Dockerfile](../../cli-worker/Dockerfile) | **Y** | **Y** | **Y** | **Y** (`grpcio-tools` build-only) | **Y** | N |
| [bbs-server/Dockerfile](../../bbs-server/Dockerfile) | — | **Y** | **Y** | — | — | **Y** |
| [frontend/Dockerfile](../../frontend/Dockerfile) | Y (node → nginx) | N/A (context `./frontend`) | N/A | Y | N/A | N/A |
| [frontend/Dockerfile.dev](../../frontend/Dockerfile.dev) | N/A | N/A | N/A | N/A | N/A | N/A |

## Completed in round 1 (cli-worker + bbs-server)

- **Root [`.dockerignore`](../../.dockerignore)** — Shrinks build context for all services that use `context: .`.
- **cli-worker** — Multi-stage build; [requirements-build.txt](../../cli-worker/requirements-build.txt) holds `grpcio-tools` only for codegen; runtime [requirements.txt](../../cli-worker/requirements.txt) keeps `grpcio` + `yt-dlp` only; runtime image copies generated protos from the builder stage.
- **bbs-server** — Non-root user `bbsuser`; TCP healthcheck on default port 2323.

## Completed in round 2 (voice-service + vector-service)

- **voice-service** — Multi-stage build; [requirements-build.txt](../../voice-service/requirements-build.txt) for `grpcio-tools` only; runtime installs under `--prefix=/install` and copies to `/usr/local`; strips `build-essential` and `grpcio-tools` from final image; runtime keeps `ffmpeg` with `--no-install-recommends`; non-root `voiceuser`; existing gRPC healthcheck retained.
- **vector-service** — Multi-stage build; [requirements-build.txt](../../vector-service/requirements-build.txt) pins `grpcio-tools==1.60.0` for codegen only; runtime copies `/install` and generated protos; no `apt` packages in final image; non-root `vectoruser`; TCP healthcheck on port 50053; removed unused nested `vector-service/.dockerignore` (root `.dockerignore` is canonical for `context: .`).

## Completed in round 10 (connections-service)

- **connections-service** — Multi-stage [Dockerfile](../../connections-service/Dockerfile); [requirements-build.txt](../../connections-service/requirements-build.txt) for `grpcio-tools==1.76.0` only; runtime [requirements.txt](../../connections-service/requirements.txt) pins `grpcio==1.76.0` and `protobuf` 6.x (no `grpcio-tools`); builder venv, `protoc` on `connections_service.proto` / `voice_service.proto`, `pip uninstall grpcio-tools`; runtime `python:3.11-slim` with **no** `apt`; non-root `appuser` (10001); TCP `HEALTHCHECK` on 50057; Compose `user: "${BASTION_RUNTIME_UID:-10001}:${BASTION_RUNTIME_GID:-10001}"`.

## Completed in round 9 (tools-service)

- **tools-service** — Multi-stage [Dockerfile](../../tools-service/Dockerfile): builder venv with [requirements-build.txt](../../backend/requirements-build.txt) then [requirements.txt](../../backend/requirements.txt), same `grpc_tools.protoc` set as before, `pip uninstall grpcio-tools`; runtime `python:3.11-slim` with Chromium stack, Node 22 (NodeSource), `uv`/`uvx` via `UV_INSTALL_DIR=/usr/local` (binaries under `/usr/local`; `PATH` includes `/usr/local`); non-root `appuser` (10001); `UV_CACHE_DIR` / `npm_config_cache` under `/app/.cache`; TCP `HEALTHCHECK` on 50052; Compose `user: "${BASTION_RUNTIME_UID:-10001}:${BASTION_RUNTIME_GID:-10001}"`, named volume `bastion_tools_service_logs`, MCP caches at `/app/.cache/uv` and `/app/.cache/npm`.

## Completed in round 8 (document-service)

- **document-service** — Multi-stage [Dockerfile](../../document-service/Dockerfile); [requirements-build.txt](../../document-service/requirements-build.txt) for `grpcio-tools` only; runtime [requirements.txt](../../document-service/requirements.txt) adds `protobuf` 6.x; builder venv, `spacy download`, `protoc` on `tool_service` / `document_service` / `vector_service`, `pip uninstall grpcio-tools`; runtime OCR/PDF stack with `libcairo2` / `libffi8` (no `build-essential`, `postgresql-client`, `curl`, `wget`, `gnupg`); non-root `appuser` (UID 10001); TCP `HEALTHCHECK`; Compose `user: "${BASTION_RUNTIME_UID:-10001}:${BASTION_RUNTIME_GID:-10001}"` (same shared runtime uid as webdav and other storage-mounting services; see [.env.example](../../.env.example)).

## Completed in round 7 (webdav)

- **webdav** — [Dockerfile.webdav](../../backend/Dockerfile.webdav): venv under `/opt/venv`, **no** `apt` / `libpq-dev` (wheels-only); `PYTHONUNBUFFERED` / `PYTHONDONTWRITEBYTECODE`; image `USER appuser` (10001); TCP `HEALTHCHECK` on port 8001; [docker-compose.yml](../../docker-compose.yml) `JWT_SECRET_KEY=${JWT_SECRET_KEY}` (no committed secret); Compose `user: "${BASTION_RUNTIME_UID:-10001}:${BASTION_RUNTIME_GID:-10001}"` (see [.env.example](../../.env.example)).

## Completed in round 6 (llm-orchestrator)

- **llm-orchestrator** — Multi-stage; [requirements-build.txt](../../llm-orchestrator/requirements-build.txt) for `grpcio-tools` only; runtime [requirements.txt](../../llm-orchestrator/requirements.txt) adds explicit `protobuf` 6.x and drops `grpcio-tools`; builder venv, same eight-proto `protoc` set as before, `pip uninstall grpcio-tools`; runtime `python:3.11-slim` with **no** `apt` packages (removed `build-essential`, `curl`, `libpq-dev`, `postgresql-client`); non-root `appuser` (UID 10001); TCP `HEALTHCHECK` on port 50051 (`127.0.0.1`).

## Completed in round 5 (celery-beat + celery-flower)

- **celery-beat** — Multi-stage; [requirements-celery-beat.txt](../../backend/requirements-celery-beat.txt) without `grpcio-tools`; `grpcio` + `protobuf` 6.x at runtime; builder installs beat deps then [requirements-build.txt](../../backend/requirements-build.txt), runs full protoc set (including `connections_service`, `image_vision`), `pip uninstall grpcio-tools`; runtime `python:3.11-slim` with **no** `apt` packages; non-root `appuser` (UID 10001); Compose `bastion_celery_beat_logs`.
- **celery-flower** — Same pattern with [requirements-celery-flower.txt](../../backend/requirements-celery-flower.txt); `EXPOSE 5555`; Compose `bastion_celery_flower_logs`.
- **celery_app** — `include` list extended with `skill_metrics_tasks` and `skill_promotion_tasks` so Beat-scheduled tasks match registered workers; [celery_database_helpers.py](../../backend/services/database_manager/celery_database_helpers.py) adds `run_async_db_task` used by those modules.

## Completed in round 4 (backend API + celery-worker)

- **backend** — Multi-stage build; [requirements-build.txt](../../backend/requirements-build.txt) for `grpcio-tools` only; runtime [requirements.txt](../../backend/requirements.txt) drops `grpcio-tools` and dev-only packages (`pytest`, `black`, `isort` → optional [requirements-dev.txt](../../backend/requirements-dev.txt)); explicit `protobuf` 6.x for generated `*_pb2.py`; builder venv + `pip uninstall grpcio-tools` before copying `/opt/venv`; runtime strips compilers and `postgresql-client`; `--no-install-recommends` on runtime `apt`; `libcairo2` + `libffi8` for WeasyPrint/FFI; non-root `appuser` (UID 10001); `HEALTHCHECK` via `curl` to `http://127.0.0.1:8000/health`; Compose uses named volume `bastion_backend_logs` and retargets `mcp-*` caches to `/app/.cache/*` with `UV_CACHE_DIR` / `npm_config_cache`.
- **celery-worker** — Same multi-stage pattern and full proto codegen list as the API image (includes `document_service`, `voice_service`, `image_vision`); non-root `appuser`; Compose named log volume `bastion_celery_worker_logs` / `bastion_celery_reindex_logs`.
- **tools-service (pip order / codegen)** — Builder installs [requirements-build.txt](../../backend/requirements-build.txt) before [requirements.txt](../../backend/requirements.txt), then `pip uninstall grpcio-tools` after protoc (see **round 9** for multi-stage runtime hardening).

## Completed in round 3 (crawl4ai-service + image-vision-service + data-service)

- **crawl4ai-service** — Multi-stage; [requirements-build.txt](../../crawl4ai-service/requirements-build.txt) for `grpcio-tools` only; builder runs Node.js, `playwright install chromium` with `PLAYWRIGHT_BROWSERS_PATH=/opt/ms-playwright`, `crawl4ai-setup`, and `protoc`; runtime copies `/install`, browser tree, and protos; Chromium system libs only with `--no-install-recommends`; non-root `crawluser`; consolidated pip installs (no duplicate `pip install playwright` / `crawl4ai` lines).
- **image-vision-service** — Multi-stage; [requirements-build.txt](../../image-vision-service/requirements-build.txt) for `grpcio-tools` only; builder uses `cmake` / `build-essential` / BLAS dev packages; runtime uses `libopenblas0`, `liblapack3`, `libgl1`, `libglib2.0-0`, `libgomp1` only; proto `RUN` cleaned to `test -f` checks only; non-root `visionuser`; TCP healthcheck on port 50056.
- **data-service** — Multi-stage; [requirements-build.txt](../../data-service/requirements-build.txt) for `grpcio-tools` only; builder uses `gcc` for wheel builds; runtime has no `apt` packages (dropped unused `postgresql-client`); non-root `datauser`; TCP healthcheck on port 50054.

## Remaining work (prioritized)

1. **Shared backend base image** — [backend/Dockerfile](../../backend/Dockerfile), [Dockerfile.celery-worker](../../backend/Dockerfile.celery-worker), [Dockerfile.celery-beat](../../backend/Dockerfile.celery-beat), and [Dockerfile.celery-flower](../../backend/Dockerfile.celery-flower) duplicate most layers; factor a common base to shrink storage and build time.
2. **Multi-stage Python images (general)** — Strip `build-essential` / `python3-dev` / build-only `grpcio-tools` from runtime for any remaining images still marked **N** for multi-stage in the table (e.g. **frontend** dev variants, **webdav** by design).
3. **Non-root for internet-facing services** — Any remaining single-stage or root-default images not listed as **Y** in the table above (backend, tools-service, connections-service, celery-worker, celery-beat, celery-flower, webdav, and llm-orchestrator are non-root).
4. **Consistent `apt-get`** — Add `--no-install-recommends` everywhere `apt-get install` is used for images not yet marked **Y** in the table.
5. **Optional: frontend context** — Add [frontend/.dockerignore](../../frontend/.dockerignore) for `context: ./frontend` builds (node_modules, dist) if not already covered.

## References

- Compose build definitions: [docker-compose.yml](../../docker-compose.yml)
