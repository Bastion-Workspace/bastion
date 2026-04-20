# S3-Compatible Storage Backend Plan

## Overview

Introduce a `StorageBackend` protocol that abstracts all blob I/O behind a common interface, with two initial implementations: `LocalFileSystemBackend` (wrapping current behavior) and `S3CompatibleBackend` (boto3/aiobotocore, works with MinIO self-hosted or AWS S3). This eliminates the shared-volume coupling between containers, unblocks horizontal scaling, and opens a path to Kubernetes deployment without a ReadWriteMany PVC.

## Current State

| Layer | Location | What it does |
|---|---|---|
| Config | `backend/config.py` | `UPLOAD_DIR = /app/uploads` — single env var, no backend selection |
| Path resolution | `backend/services/folder_service.py` | `FolderService.get_document_file_path()` builds `Path` objects from logical folder hierarchy |
| File writes | `backend/services/document_service_v2.py` | `aiofiles.open(file_path, 'wb')` hardcoded to disk |
| File reads / downloads | `backend/api/document_api.py` | `FileResponse(file_path)` for downloads; `aiofiles.open(...)` for content reads |
| Collab saves | `backend/services/collab_persist.py` | Resolves paths with `Path(settings.UPLOAD_DIR)` + fallback glob patterns |
| WebDAV | `backend/webdav/orgmode_provider.py` | Reads org files from the filesystem directly |
| Messaging attachments | `backend/api/document_api.py`, `backend/sql/01_init.sql` | `message_attachments.file_path VARCHAR(512)` stores a literal filesystem path |

**Key structural fact:** `document_metadata` does **not** store a `file_path` column. Physical paths are derived at runtime from the folder/user/team hierarchy by `FolderService`. This means migrating to S3 keys that mirror the same logical structure (e.g., `Users/{username}/{folder}/{filename}`) is straightforward — the derivation logic changes, not the database schema.

**The shared-volume problem:** Six containers share `./uploads:/app/uploads`. In Docker Compose this works; in Kubernetes it requires a ReadWriteMany PVC (NFS, EFS, Azure Files), which is operationally expensive and defeats horizontal pod autoscaling.

## The Hot/Cold Storage Question

The question of whether to support two simultaneous storage backends — one fast and local ("hot"), one cheap and slow ("cold") — is worth analyzing carefully before committing to the abstraction design.

### Option A: Application-managed tiering (two backend instances)

Configure a `HOT_STORAGE_BACKEND` (MinIO or local) and a `COLD_STORAGE_BACKEND` (S3 Glacier, Backblaze B2). A background job migrates documents not accessed in N days from hot to cold. Reads check hot first, then fall back to cold with a possible restore-on-demand delay for Glacier-class storage.

**Requires:**
- `last_accessed_at` tracking on blobs (currently absent)
- A background migration Celery job
- Restore-on-demand logic (Glacier Standard retrieval takes 3–5 hours)
- Two sets of S3 credentials and bucket config

**Verdict:** High complexity, low ROI for most deployments. Only justifiable at TB scale where the cost delta between S3 Standard and S3 Glacier is meaningful.

### Option B: Single S3 bucket with infrastructure-level tiering

Set up one S3 bucket with **S3 Intelligent Tiering** (AWS) or lifecycle rules to Glacier after N days. The application reads and writes the same bucket; AWS handles cold storage automatically. Objects in Frequent Access tier behave identically to S3 Standard.

**Requires:** Nothing from the application. AWS manages it.

**Verdict:** The right answer for AWS deployments. Zero application complexity.

### Option C: MinIO with ILM Tiering

MinIO supports a Tier API: define a remote tier (S3, GCS, Azure Blob) and set lifecycle rules. Objects older than N days are transparently migrated to the remote tier. Reads auto-restore on access. The application still talks to MinIO — no application-layer changes.

**Requires:** MinIO Tier configuration and a remote object store.

**Verdict:** The right answer for self-hosted deployments that want a local fast layer plus cheap cold storage. No application changes after the initial S3 abstraction is in place.

### Option D: RoutingStorageBackend (multi-backend routing in application)

A `RoutingStorageBackend` wraps two `StorageBackend` instances and routes by collection type, doc_type, or team/user. Example: team libraries go to S3, personal drafts stay local. An explicit "archive" action in the UI migrates a document from the primary to the archive backend.

**Requires:** A routing policy config, two backends configured simultaneously, explicit migration actions.

**Verdict:** Justified as an **explicit user-driven action** ("archive this folder to S3"), not as automatic tiering. The primary use case is organizations that want to keep their active library fast and local while offloading a large historical archive to cheap cloud storage — with the user in control of the boundary.

### Recommendation

- **Phase 1–3 of this plan:** Implement a single `StorageBackend` abstraction with `local` and `s3` implementations. No tiering.
- **For AWS deployments:** Use S3 Intelligent Tiering at the bucket level. The application does not need to know.
- **For self-hosted deployments:** Use MinIO as the primary backend; configure MinIO ILM Tiering to S3/B2 at the MinIO layer if cold storage is needed.
- **Phase 4 (optional):** Implement `RoutingStorageBackend` for explicit user-driven archival actions. Do **not** implement automatic application-level tiering; that belongs in the storage infrastructure, not the application.
- **Design constraint:** The `StorageBackend` protocol must be designed from the start to support a `RoutingStorageBackend` without breaking callers (i.e., no positional `Path` assumptions in the interface).

---

## Proposed Architecture

### StorageBackend Protocol

New file: `backend/storage/backends/base.py`

```python
from typing import Protocol, AsyncIterator

class StorageBackend(Protocol):
    async def write(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None: ...
    async def read(self, key: str) -> bytes: ...
    async def stream(self, key: str) -> AsyncIterator[bytes]: ...
    async def delete(self, key: str) -> None: ...
    async def exists(self, key: str) -> bool: ...
    async def get_download_url(self, key: str, expires_in: int = 3600) -> str: ...
    async def copy(self, src_key: str, dst_key: str) -> None: ...
    async def move(self, src_key: str, dst_key: str) -> None: ...
```

`get_download_url` is the key method for S3: returns a presigned URL for direct browser download, bypassing the backend entirely. For the local backend, it returns the existing API download endpoint URL.

### Key Derivation

S3 object keys mirror the current filesystem logical structure:

| Scope | Filesystem path | S3 key |
|---|---|---|
| Global | `/app/uploads/Global/{folder}/{filename}` | `Global/{folder}/{filename}` |
| User | `/app/uploads/Users/{username}/{folder}/{filename}` | `Users/{username}/{folder}/{filename}` |
| Team | `/app/uploads/Teams/{team_id}/documents/{folder}/{filename}` | `Teams/{team_id}/documents/{folder}/{filename}` |
| Legacy flat | `/app/uploads/{doc_id}_{filename}` | `_legacy/{doc_id}_{filename}` |
| Messaging | `/app/uploads/messaging/{filename}` | `messaging/{filename}` |

This means `FolderService.get_document_file_path()` returns a logical key (string) rather than a `Path`. Local backend prepends `settings.UPLOAD_DIR`; S3 backend uses the key directly.

**Refactor impact on `FolderService`:** Replace `Path`-returning methods with key-returning equivalents. The methods already contain all the derivation logic; only the return type and path-joining change.

### Implementation Files

```
backend/storage/
    __init__.py
    backends/
        __init__.py
        base.py          # StorageBackend Protocol + StorageError exceptions
        local.py         # LocalFileSystemBackend
        s3.py            # S3CompatibleBackend (aiobotocore)
    factory.py           # get_storage_backend() from settings
```

---

## Implementation Phases

### Phase 1: StorageBackend Protocol + LocalFileSystemBackend (no behavior change)

**Goal:** Introduce the abstraction without changing any observable behavior.

**Scope:**
- Define `StorageBackend` protocol in `backend/storage/backends/base.py`
- Implement `LocalFileSystemBackend` that wraps `aiofiles` reads/writes on `settings.UPLOAD_DIR`
- Add `STORAGE_BACKEND_TYPE: str = "local"` to `backend/config.py`
- Add `factory.py` with `get_storage_backend()` returning the configured backend
- **Do not yet change any callers** — this phase is purely additive

**Risk:** None. No callers are changed.

### Phase 2: Refactor FolderService to return storage keys

**Goal:** Replace filesystem `Path` objects with logical string keys throughout the path-resolution layer.

**Scope:**
- `FolderService.get_document_file_path()` → returns `str` (the logical key)
- `FolderService.get_user_base_path()` / `get_team_base_path()` → return key prefixes (`str`)
- `FolderService.initialize()` — remove `mkdir` calls (keys don't need directory creation)
- All callers of `FolderService` that currently do `Path(folder_path).exists()` etc. move to `storage.exists(key)`

**Primary call sites to update:**
- `backend/services/document_service_v2.py` — write, read, delete, process
- `backend/api/document_api.py` — download (`FileResponse` → presigned URL or streaming)
- `backend/services/collab_persist.py` — read/write for collaborative editing saves
- `backend/services/zip_processor_service.py` — bulk ZIP reads
- `backend/services/file_manager/file_manager_service.py` — move, rename, delete orchestration
- `backend/services/document_version_service.py` — versioned content reads
- `backend/services/file_recovery_service.py` — recovery scans (may need special handling)

**For downloads:** Replace `FileResponse(file_path)` with either:
1. `RedirectResponse(storage.get_download_url(key))` for S3 presigned URLs
2. A streaming response using `storage.stream(key)` for cases where redirect isn't appropriate (e.g., inline display)

The local backend's `get_download_url()` returns the existing `/api/documents/{id}/file` endpoint URL, so no behavioral change for local users.

**Risk:** Medium. Touches multiple service layers. Phase 2 should be tested end-to-end with the `local` backend before Phase 3 begins.

### Phase 3: S3CompatibleBackend

**Goal:** Implement S3 backend using `aiobotocore` (async boto3).

**Scope:**
- `backend/storage/backends/s3.py`
- `S3CompatibleBackend` implementing the protocol
- `write` → `put_object`; `read` → `get_object` + body read; `stream` → streaming `get_object`
- `delete` → `delete_object`; `exists` → `head_object` (catch 404)
- `get_download_url` → `generate_presigned_url("get_object", ...)` with configurable expiry
- `copy` / `move` → `copy_object` + `delete_object`
- New settings (see Configuration section below)
- Works with MinIO, AWS S3, Backblaze B2 (B2 S3-compatible API), Wasabi, Cloudflare R2 — any S3-compatible endpoint

**MinIO for local/self-hosted:** Add a `minio` service to `docker-compose.yml` (uses the `minio/minio` image). Map ports 9000 (API) and 9001 (console). Default bucket creation handled at startup via `ensure_bucket_exists()` in the backend factory.

**Risk:** Low (additive; existing local backend untouched). Integration testing against a MinIO container is straightforward.

### Phase 4 (Optional): RoutingStorageBackend

**Goal:** Route writes to different backends based on a configurable policy, enabling explicit user-driven archival.

**Scope:**
- `backend/storage/backends/routing.py`
- Config: `STORAGE_ARCHIVE_BACKEND_TYPE`, `STORAGE_ARCHIVE_BACKEND_*` credentials
- Routing policy: by `collection_type` (global/user/team), by `doc_type`, or explicit per-folder/document setting
- New document metadata field: `storage_location ENUM('primary', 'archive') DEFAULT 'primary'`
- Admin/user API endpoint: `POST /api/documents/{id}/archive` — moves blob from primary to archive backend, updates `storage_location`
- Read path: check `storage_location` in metadata, route read to correct backend

**Use case:** Large historical document libraries archived to S3 Glacier Instant Retrieval / Backblaze B2 while the active working set stays on MinIO or local disk.

**Note:** This phase is only warranted if users need explicit control over storage placement. For automatic tiering, use MinIO ILM or S3 Intelligent Tiering instead.

---

## Database Schema Changes

### document_metadata — no changes for Phase 1–3

The derivation-based key approach means no new column is needed. The logical key is always derivable from existing fields (`folder_id` → `FolderService` → key).

### Phase 4 addition (if implemented)

```sql
ALTER TABLE document_metadata
  ADD COLUMN storage_location VARCHAR(20) DEFAULT 'primary';
```

### message_attachments

`file_path VARCHAR(512)` currently stores a literal filesystem path. This needs to become a storage key in Phase 2.

```sql
ALTER TABLE message_attachments
  RENAME COLUMN file_path TO storage_key;
```

The messaging attachment write and read paths in `backend/api/document_api.py` and `backend/services/messaging/messaging_service.py` are updated to use `storage.write(key, ...)` and `storage.get_download_url(key)`.

---

## Configuration

New environment variables in `backend/config.py`:

| Variable | Default | Description |
|---|---|---|
| `STORAGE_BACKEND_TYPE` | `local` | `local` or `s3` |
| `STORAGE_S3_ENDPOINT_URL` | `""` | Override endpoint for MinIO or other S3-compatible (empty = AWS) |
| `STORAGE_S3_BUCKET` | `bastion-uploads` | Bucket name |
| `STORAGE_S3_ACCESS_KEY` | `""` | Access key ID |
| `STORAGE_S3_SECRET_KEY` | `""` | Secret access key |
| `STORAGE_S3_REGION` | `us-east-1` | Region (unused for MinIO but required by SDK) |
| `STORAGE_S3_PRESIGN_EXPIRY` | `3600` | Presigned URL TTL in seconds |
| `STORAGE_S3_PREFIX` | `""` | Optional key prefix (e.g. `instance-name/`) for multi-tenant buckets |
| `STORAGE_S3_PATH_STYLE` | `false` | Use path-style addressing (required for MinIO, Backblaze B2) |

Docker Compose additions for MinIO:

```yaml
minio:
  image: minio/minio:latest
  command: server /data --console-address ":9001"
  environment:
    MINIO_ROOT_USER: ${MINIO_ROOT_USER:-bastion}
    MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD:-bastion_password}
  volumes:
    - bastion_minio_data:/data
  ports:
    - "9000:9000"
    - "9001:9001"
```

When `STORAGE_BACKEND_TYPE=s3` with MinIO:
- `STORAGE_S3_ENDPOINT_URL=http://minio:9000`
- `STORAGE_S3_PATH_STYLE=true`
- Remove `./uploads:/app/uploads` mounts from all services

---

## Affected Components

| Component | Change required | Phase |
|---|---|---|
| `backend/config.py` | Add `STORAGE_BACKEND_*` settings | 1 |
| `backend/storage/` | New module (protocol, local, s3, factory) | 1–3 |
| `backend/services/folder_service.py` | Key-returning path resolution | 2 |
| `backend/services/document_service_v2.py` | `aiofiles` → `storage.write/read/delete` | 2 |
| `backend/api/document_api.py` | `FileResponse` → presigned URL / streaming | 2 |
| `backend/services/collab_persist.py` | Path I/O → storage I/O | 2 |
| `backend/services/zip_processor_service.py` | Path reads → `storage.read()` | 2 |
| `backend/services/file_manager/file_manager_service.py` | Move/rename → `storage.move/copy` | 2 |
| `backend/services/document_version_service.py` | Content reads → `storage.read()` | 2 |
| `backend/services/file_recovery_service.py` | Scan logic needs S3 list-objects equivalent | 2 |
| `backend/services/messaging/messaging_service.py` | Attachment writes → `storage.write()` | 2 |
| `backend/webdav/orgmode_provider.py` | Filesystem access — **special case** (see below) | 2 |
| `tools-service` | Document content reads — uses same paths | 2 |
| `image-vision-service` | Read-only mount — uses same paths | 2 |
| `docker-compose.yml` | Remove shared volume mounts; add MinIO | 3 |
| `backend/sql/01_init.sql` | Rename `message_attachments.file_path` | 2 |

### WebDAV Special Case

`webdav/orgmode_provider.py` uses WsgiDAV, which expects a real filesystem. Options:
1. **Keep the local filesystem for WebDAV only** and sync org files to S3 separately (background job or write-through). Simplest approach.
2. **Mount a minimal local scratch volume** just for WebDAV and org files, treating all other documents as S3-backed.
3. **Use a FUSE S3 mount** (e.g., `goofys`, `s3fs`) to expose an S3 bucket as a filesystem for WsgiDAV. Adds operational complexity.

Recommended: Option 1 (or 2 if WebDAV is not a priority use case). WebDAV is already a separate container. Keeping a small org-files volume for WebDAV while the rest of the document library moves to S3 is a reasonable partition.

---

## Migration Strategy for Existing Deployments

For users upgrading from a local-only deployment to S3:

1. **Run the migration script:** `python -m backend.scripts.migrate_uploads_to_s3`
   - Iterates all files under `UPLOAD_DIR`
   - Derives the S3 key from the local path (strip `UPLOAD_DIR` prefix)
   - Uploads each file to S3 via `storage.write(key, data)`
   - Verifies `storage.exists(key)` after upload
   - Logs failures for retry
2. **Verify:** Admin endpoint `GET /api/admin/storage/verify` compares `document_metadata` rows against `storage.exists()` for each derived key
3. **Switch backend:** Set `STORAGE_BACKEND_TYPE=s3` and restart
4. **Keep local copies** for N days as a fallback before deleting `./uploads`

---

## Estimated Effort

| Phase | Effort | Risk |
|---|---|---|
| Phase 1: Protocol + LocalFileSystemBackend | 0.5 days | None |
| Phase 2: FolderService + all caller refactor | 2–3 days | Medium (broad surface area) |
| Phase 3: S3CompatibleBackend + MinIO compose | 1 day | Low |
| Migration script + verify endpoint | 0.5 days | Low |
| Phase 4: RoutingStorageBackend | 1–2 days | Low (additive) |
| **Total (Phase 1–3)** | ~4 days | |

---

## Open Questions

1. **Key derivation vs. stored keys:** The plan derives S3 keys from folder structure at runtime. An alternative is to store `storage_key` in `document_metadata` at upload time. Stored keys are more robust to future folder renames/moves (the key doesn't change, only the display path). The derivation approach is simpler for Phase 1–3 but may need revisiting if document moves become frequent. Consider storing `storage_key` explicitly in Phase 2 as a one-time migration.

2. **tools-service and image-vision-service:** Both currently read from the shared `/app/uploads` mount. After migrating to S3, they need to either: (a) call `storage.read(key)` and write to a local temp file for processing, or (b) receive the blob data over gRPC rather than resolving it from disk. The gRPC boundary makes option (b) cleaner long-term.

3. **Large file handling:** Files up to 1.5 GB are supported (`UPLOAD_MAX_SIZE`). S3 multipart upload is required above 5 GB and recommended above 100 MB. The `S3CompatibleBackend.write()` should use multipart transparently above a configurable threshold.

4. **Encryption at rest:** The system has document-level encryption (`backend/services/file_encryption_service.py`). S3 server-side encryption (SSE-S3 or SSE-KMS) is an additional layer. These are independent and compatible, but the interaction should be documented.

5. **Celery workers and temp files:** Several Celery tasks write temp files during processing (e.g., PDF extraction). These use Python's `tempfile` module and are independent of `UPLOAD_DIR` — no change needed. But the final write of processed content does go through `document_service_v2.py` and will need to use `storage.write()` after Phase 2.

## Related Architecture

- `docs/dev-notes/FILE_SERVICE_SEPARATION.md` — notes that S3 migration would eliminate the shared-volume coupling between a future file-service and other services
- `backend/services/folder_service.py` — primary path resolution logic
- `backend/services/document_service_v2.py` — primary file write path
- `backend/sql/01_init.sql` — `document_metadata` schema (no `file_path` column; keys are derived)
- `docker-compose.yml` — current shared `./uploads` mounts
