# Document Service Consolidation Plan

Consolidate all document and file operations into the `document-service` container, transforming the backend into a thin HTTP/WebSocket gateway. Vectorization remains in vector-service; image processing remains in image-vision-service; the document-service owns blob I/O, metadata, folder hierarchy, document processing, and content access.

## Current State

### Where Document Operations Live Today

| Layer | Location | Responsibility |
|-------|----------|---------------|
| **API** | `backend/api/document_api.py` | Upload, download, content CRUD, metadata, reprocessing, messaging attachments |
| **API** | `backend/api/folder_api.py` | Folder tree, contents, CRUD, move, ZIP download |
| **API** | `backend/api/file_manager_api.py` | Place file, move, delete, rename, create folder structures |
| **Service** | `backend/services/document_service_v2.py` (~2,200 lines) | Upload processing, PDF analysis, chunking, text extraction, status management |
| **Service** | `backend/services/folder_service.py` (~1,600 lines) | Folder CRUD, physical path resolution, tree building, ZIP generation |
| **Service** | `backend/services/file_manager/file_manager_service.py` (~1,000 lines) | File placement, move, rename, delete orchestration, WebSocket notifications |
| **Service** | `backend/services/collab_persist.py` | Collaborative editing saves (Yjs room flush to disk) |
| **Service** | `backend/services/document_version_service.py` | Version snapshots, rollback, pruning |
| **Service** | `backend/services/file_encryption_service.py` | Document-level encryption/decryption |
| **Service** | `backend/services/file_recovery_service.py` | Scan filesystem for orphaned files, recover to DB |
| **Service** | `backend/services/zip_processor_service.py` | Bulk ZIP upload extraction |
| **Repository** | `backend/repositories/document_repository.py` | PostgreSQL CRUD, RLS, status tracking, pagination |
| **Background** | `celery_worker` / `celery_reindex_worker` containers | Async document processing pipeline (reprocess, reindex, backfill, audio export) |
| **gRPC reads** | `backend/services/grpc_handlers/document_handlers.py` | `GetDocumentContent`, `FindDocumentsByTags` (file reads served via tools-service process) |
| **gRPC writes** | `backend/services/grpc_handlers/file_creation_handlers.py` | `CreateUserFile`, `CreateUserFolder` (served via tools-service process) |
| **gRPC edits** | `backend/services/grpc_handlers/document_edit_handlers.py` | `UpdateDocumentContent`, edit proposals (served via tools-service process) |
| **WebDAV** | `backend/webdav/orgmode_provider.py` | Org file access (reads from `free_form_notes` DB table, not from uploads volume) |

### The Shared Uploads Volume Problem

Six containers currently mount `./uploads:/app/uploads`:

| Container | Access | What it does with uploads |
|-----------|--------|--------------------------|
| **backend** | Read/write | Upload writes, download reads, content reads, delete |
| **celery_worker** | Read/write | Document processing reads, audio export writes |
| **celery_reindex_worker** | Read/write | Reprocess reads |
| **tools-service** | Read/write | `GetDocumentContent`, `CreateUserFile`, edit proposals — all via backend code imported into the tools-service process |
| **webdav** | Read/write | Mounted but WebDAV actually reads from `free_form_notes` DB, not uploads |
| **image-vision-service** | Read-only | Face/object detection reads image files by path |

### The tools-service Architecture Detail

tools-service is not an independent codebase — it imports the entire `backend/` tree via `PYTHONPATH` manipulation. The gRPC handlers that serve document reads/writes (`document_handlers.py`, `file_creation_handlers.py`, `document_edit_handlers.py`) are `backend/` code running inside the tools-service container. This means:

- tools-service already shares `DocumentRepository`, `FolderService`, `DocumentProcessor`, and all filesystem I/O code with the backend
- Document read operations (search, get content, get chunks) go through tools-service's gRPC surface
- Document write operations (upload, folder CRUD, file management) go through backend's HTTP surface
- Both resolve file paths via `FolderService` and read/write to the same `./uploads` volume

This is an informal CQRS-like split that emerged organically, not by design.

### What document-service Is Today

`document-service` (port 50058) is a lightweight, standalone gRPC microservice that does one thing: entity extraction via spaCy. It has:

- Two RPCs: `ExtractEntities` and `HealthCheck`
- No database access
- No filesystem access
- No uploads volume mount
- Its own small codebase (`document-service/service/entity_extractor.py`)
- ~150 lines of application code total

The proto is `document_service.proto` with `ExtractEntitiesRequest`/`ExtractEntitiesResponse` messages.

## Target Architecture

### What Moves to document-service

The document-service absorbs all document/file/folder operations and becomes the single owner of blob storage and document metadata. It gains:

**From backend services:**
- `document_service_v2.py` — upload processing, PDF analysis, text extraction, content reads/writes, delete
- `folder_service.py` — folder CRUD, path resolution, tree building, ZIP generation
- `file_manager/file_manager_service.py` — file placement, move, rename, delete orchestration
- `collab_persist.py` — collaborative editing disk persistence
- `document_version_service.py` — version snapshots, rollback
- `file_encryption_service.py` — document-level encryption
- `file_recovery_service.py` — orphan file scanning and recovery
- `zip_processor_service.py` — ZIP upload extraction

**From backend repositories:**
- `document_repository.py` — all `document_metadata`, `document_folders`, `document_versions`, `document_edit_proposals` access
- `document_version_repository.py` — version CRUD

**From tools-service gRPC handlers:**
- `document_handlers.py` — `SearchDocuments`, `GetDocument`, `GetDocumentContent`, `GetDocumentChunks`, `FindDocumentsByTags`, `FindDocumentByPath`, `RerankDocuments`
- `file_creation_handlers.py` — `CreateUserFile`, `CreateUserFolder`, `GetFolderTree`, `ListFolderDocuments`, `PickRandomDocumentFromFolder`
- `document_edit_handlers.py` — `UpdateDocumentMetadata`, `UpdateDocumentContent`, `ProposeDocumentEdit`, `ApplyDocumentEditProposal`, `ListDocumentProposals`, `GetDocumentEditProposal`, `RejectDocumentEditProposal`, `ApplyOperationsDirectly`
- `search_utility_handlers.py` (document parts) — `FindDocumentsByEntities`, `FindRelatedDocumentsByEntities`

**From Celery workers:**
- `reprocess_document_after_save_task` — reads file, runs processing pipeline
- `bulk_reindex_batch_task` — batch reprocessing
- `backfill_document_chunks_task` — chunk backfill
- `prune_document_versions_task` — version cleanup
- `audio_export_tasks` — audio export writes under uploads

**Stays in document-service as-is:**
- Entity extraction via spaCy (`ExtractEntities`) — this is already here and stays

### What Stays in Backend

The backend becomes a thin gateway:

- **HTTP API gateway** — authentication, CORS, rate limiting, request routing
- **WebSocket management** — presence, notifications, collaborative editing coordination
- **Celery dispatch** — submits jobs to Redis queues (document-service workers consume them)
- **gRPC client calls** — proxies HTTP requests to document-service, llm-orchestrator, connections-service, etc.
- **Chat/messaging API** — non-document APIs
- **Auth/user management** — JWT, sessions, user CRUD
- **Admin endpoints** — system settings, user management
- **RSS management** — feed CRUD (feed content processing may dispatch to document-service)

### What Stays in Other Services (unchanged)

| Service | Role | Interaction with document-service |
|---------|------|----------------------------------|
| **vector-service** | Embeddings and Qdrant operations | document-service calls vector-service for embed/search/store |
| **image-vision-service** | Face/object detection | Receives image bytes or presigned URL from document-service (no direct volume mount) |
| **tools-service** | Non-document tools: RSS operations, weather, navigation, web/browser, data workspace proxy, agent factory | Calls document-service gRPC for any document reads needed by tools |
| **llm-orchestrator** | LangGraph agent orchestration | Calls document-service gRPC (via tools-service or directly) for document context |
| **cli-worker** | Sandboxed CLI tools (FFmpeg, Pandoc) | Receives bytes via gRPC, returns bytes — no change |
| **crawl4ai-service** | Web crawling | Returns crawled content; document-service stores it |
| **connections-service** | OAuth, email, chat channels | No document I/O |
| **data-service** | Data workspace, CSV/Excel | Separate domain; no change |

### New Proto: `document_service.proto` (expanded)

The existing `document_service.proto` (entity extraction) expands to become the canonical document API. All document RPCs currently split between `tool_service.proto` and backend HTTP endpoints consolidate here.

```
service DocumentService {
  // --- Existing ---
  rpc ExtractEntities(ExtractEntitiesRequest) returns (ExtractEntitiesResponse);

  // --- Upload & Processing ---
  rpc UploadDocument(stream UploadDocumentRequest) returns (UploadDocumentResponse);
  rpc UploadMultipleDocuments(stream UploadDocumentRequest) returns (UploadMultipleResponse);
  rpc ProcessUrl(ProcessUrlRequest) returns (UploadDocumentResponse);
  rpc ProcessZipUpload(stream UploadDocumentRequest) returns (UploadMultipleResponse);
  rpc ReprocessDocument(ReprocessRequest) returns (ReprocessResponse);

  // --- Content Access ---
  rpc GetDocument(GetDocumentRequest) returns (GetDocumentResponse);
  rpc GetDocumentContent(GetDocumentContentRequest) returns (GetDocumentContentResponse);
  rpc GetDocumentChunks(GetDocumentChunksRequest) returns (GetDocumentChunksResponse);
  rpc DownloadDocument(DownloadRequest) returns (stream DownloadChunk);
  rpc SearchDocuments(SearchRequest) returns (SearchResponse);
  rpc FindDocumentsByTags(FindByTagsRequest) returns (FindByTagsResponse);
  rpc FindDocumentByPath(FindByPathRequest) returns (FindByPathResponse);
  rpc RerankDocuments(RerankRequest) returns (RerankResponse);
  rpc FindDocumentsByEntities(FindByEntitiesRequest) returns (FindByEntitiesResponse);
  rpc FindRelatedDocumentsByEntities(FindByEntitiesRequest) returns (FindByEntitiesResponse);

  // --- Content Mutation ---
  rpc UpdateDocumentMetadata(UpdateMetadataRequest) returns (UpdateResponse);
  rpc UpdateDocumentContent(UpdateContentRequest) returns (UpdateResponse);
  rpc StoreTextDocument(StoreTextRequest) returns (StoreTextResponse);
  rpc FlushCollaborativeDocument(FlushCollabRequest) returns (FlushCollabResponse);

  // --- Edit Proposals ---
  rpc ProposeDocumentEdit(ProposeEditRequest) returns (ProposeEditResponse);
  rpc ApplyDocumentEditProposal(ApplyProposalRequest) returns (ApplyProposalResponse);
  rpc ApplyOperationsDirectly(ApplyDirectRequest) returns (ApplyDirectResponse);
  rpc ListDocumentProposals(ListProposalsRequest) returns (ListProposalsResponse);
  rpc GetDocumentEditProposal(GetProposalRequest) returns (GetProposalResponse);
  rpc RejectDocumentEditProposal(RejectProposalRequest) returns (RejectProposalResponse);

  // --- File Management ---
  rpc PlaceFile(PlaceFileRequest) returns (PlaceFileResponse);
  rpc MoveFile(MoveFileRequest) returns (MoveFileResponse);
  rpc RenameFile(RenameFileRequest) returns (RenameFileResponse);
  rpc DeleteDocument(DeleteRequest) returns (DeleteResponse);

  // --- Folders ---
  rpc CreateFolder(CreateFolderRequest) returns (CreateFolderResponse);
  rpc UpdateFolder(UpdateFolderRequest) returns (UpdateFolderResponse);
  rpc DeleteFolder(DeleteFolderRequest) returns (DeleteFolderResponse);
  rpc MoveFolder(MoveFolderRequest) returns (MoveFolderResponse);
  rpc GetFolderTree(GetFolderTreeRequest) returns (GetFolderTreeResponse);
  rpc GetFolderContents(GetFolderContentsRequest) returns (GetFolderContentsResponse);
  rpc ListFolderDocuments(ListFolderDocsRequest) returns (ListFolderDocsResponse);
  rpc PickRandomDocumentFromFolder(PickRandomRequest) returns (PickRandomResponse);
  rpc GenerateLibraryZip(GenerateZipRequest) returns (stream DownloadChunk);
  rpc CreateFolderStructure(CreateStructureRequest) returns (CreateStructureResponse);

  // --- Versioning ---
  rpc GetDocumentVersions(GetVersionsRequest) returns (GetVersionsResponse);
  rpc GetVersionContent(GetVersionContentRequest) returns (GetVersionContentResponse);
  rpc RollbackToVersion(RollbackRequest) returns (RollbackResponse);

  // --- Encryption ---
  rpc EncryptDocument(EncryptRequest) returns (EncryptResponse);
  rpc CreateDecryptSession(DecryptSessionRequest) returns (DecryptSessionResponse);
  rpc TryDecrypt(TryDecryptRequest) returns (TryDecryptResponse);
  rpc RemoveEncryption(RemoveEncryptionRequest) returns (RemoveEncryptionResponse);

  // --- Recovery & Admin ---
  rpc ScanAndRecoverFiles(RecoverRequest) returns (RecoverResponse);

  // --- Health ---
  rpc HealthCheck(HealthCheckRequest) returns (HealthCheckResponse);
}
```

Streaming RPCs (`UploadDocument`, `DownloadDocument`, `GenerateLibraryZip`) handle large files without buffering entire blobs in memory.

### What Gets Removed from tool_service.proto

All document-specific RPCs move out of `tool_service.proto`:

- `SearchDocuments`, `FindDocumentsByTags`, `GetDocument`, `GetDocumentContent`, `GetDocumentChunks`, `FindDocumentByPath`, `RerankDocuments`
- `FindDocumentsByEntities`, `FindRelatedDocumentsByEntities`
- `CreateUserFile`, `CreateUserFolder`, `GetFolderTree`, `ListFolderDocuments`, `PickRandomDocumentFromFolder`
- `UpdateDocumentMetadata`, `UpdateDocumentContent`
- All proposal RPCs

tools-service retains: RSS, weather, navigation, web/browser tools, data workspace proxy, agent factory, org-mode note operations, `AnalyzeTextContent`, and other non-document-storage RPCs. When tools-service needs document data (e.g., a tool that reads a file), it calls document-service over gRPC.

## Filesystem I/O Inventory

Every function that touches the filesystem needs to move into document-service. Complete inventory:

### Direct File Writes

| Location | Method | I/O Pattern |
|----------|--------|-------------|
| `document_service_v2.py` | `upload_and_process` | `aiofiles.open(path, 'wb')` — primary upload write |
| `document_service_v2.py` | `store_text_document` | `open(path, 'w')` — agent-created text files |
| `document_service_v2.py` | `_process_url_async` | `aiofiles.open` — temp file for URL content |
| `file_manager_service.py` | `place_file` | `Path.write_bytes()`, `open(path, 'w')` — file placement |
| `file_manager_service.py` | `_save_as_markdown_file` | `open(path, 'w')` — RSS/manual markdown |
| `collab_persist.py` | `flush_collaborative_document` | `open(path, 'w')` — Yjs room flush |
| `document_version_service.py` | `snapshot_before_write` | `shutil.copy2` — version snapshot |
| `document_version_service.py` | `rollback_to_version` | `Path.write_text` — restore from snapshot |
| `file_encryption_service.py` | `encrypt_document` / `write_encrypted_content_from_session` | `open(path, 'wb')` / `open(path, 'w')` |
| `zip_processor_service.py` | `_process_individual_file` | `aiofiles.open(path, 'wb')` — extracted files |
| Celery `audio_export_tasks` | Audio export | Writes under `UPLOAD_DIR/audio_exports` |

### Direct File Reads

| Location | Method | I/O Pattern |
|----------|--------|-------------|
| `document_service_v2.py` | `_analyze_pdf_type` | `open(path, 'rb')`, `pdfplumber.open` |
| `document_service_v2.py` | `_process_document_async` | `Path.read_text` for link extraction |
| `document_service_v2.py` | `_reextract_knowledge_graph` | `open(path, 'r')` |
| `document_api.py` | `/documents/{id}/file`, `/documents/{id}/pdf` | `FileResponse(path)` — download |
| `document_api.py` | `/documents/{id}/content` GET | `open(path)` — content read |
| `collab_persist.py` | `read_document_plaintext_for_collab` | `Path.read_text` |
| `collab_persist.py` | `flush_collaborative_document` | `open(path, 'r')` — compare before write |
| `document_version_service.py` | `snapshot_before_write` | `Path.read_text` |
| `document_version_service.py` | `get_version_content` | `Path.read_text` |
| `file_encryption_service.py` | Various decrypt/encrypt methods | `read_text` / `read_bytes` |
| `file_recovery_service.py` | `scan_and_recover_user_files` | `Path.rglob('*')`, `stat`, `is_file` |
| `grpc_handlers/document_handlers.py` | `GetDocumentContent`, `FindDocumentsByTags` | `open()` for content/preview |
| Celery `reprocess_document_after_save_task` | Reprocess | `folder_service.get_document_file_path` + read |

### Filesystem Structure Operations

| Location | Method | I/O Pattern |
|----------|--------|-------------|
| `folder_service.py` | `initialize` | `Path.mkdir` on Global/Users/Teams |
| `folder_service.py` | `_create_physical_directory` | `Path.mkdir(parents=True)` |
| `folder_service.py` | `update_folder` | `Path.rename`, `shutil.move` |
| `folder_service.py` | `delete_folder` | `shutil.rmtree` |
| `folder_service.py` | `build_library_zip` | `zipfile.ZipFile.write` reads files |
| `file_manager_service.py` | `move_file` | `shutil.move` + metadata sidecar |
| `file_manager_service.py` | `rename_file` | `Path.rename` |
| `document_service_v2.py` | `delete_document` | `Path.unlink` + sidecar + legacy glob cleanup |
| `collab_persist.py` | `_resolve_file_path` | `Path.exists`, `glob.glob` legacy fallback |

## Database Tables Owned by document-service

| Table | Current access | Notes |
|-------|---------------|-------|
| `document_metadata` | `DocumentRepository` | Primary document record; RLS via `set_config` |
| `document_folders` | `DocumentRepository`, `FolderService` | Folder hierarchy |
| `document_versions` | `DocumentVersionRepository` | Version snapshots |
| `document_edit_proposals` | Edit handlers | Proposal storage |
| `document_collab_state` | `collab_persist.py` | Yjs collaboration state |
| `message_attachments` | `document_api.py`, `messaging_service.py` | `file_path` column stores filesystem paths |

Backend retains ownership of: `users`, `teams`, `conversations`, `messages`, `rss_feeds`, `free_form_notes`, `system_settings`, `knowledge_graph_*`, and all other non-document tables.

Note: `SearchDocuments` uses the vector store (Qdrant via vector-service) and optionally the knowledge graph. The knowledge graph tables (`knowledge_graph_entities`, `knowledge_graph_relationships`) are a shared read concern — document-service reads them for entity-linked search, but the KG write pipeline (entity extraction → graph storage) may stay in the processing pipeline or become a separate concern. For Phase 1, document-service gets read access to KG tables.

## Implementation Phases

### Phase 1: Expand Proto and Migrate Processing Pipeline

**Goal:** Move the async document processing pipeline into the document-service container. The heaviest workload leaves the backend.

**Scope:**
- Expand `document_service.proto` with processing-related RPCs (`ReprocessDocument`, `GetDocumentContent`, `GetDocumentChunks`, `UploadDocument`)
- document-service gains its own Celery worker (or direct async processing) consuming from the existing Redis queues
- Move `document_service_v2.py` processing pipeline (`_process_document_async`, `_process_standard_document`, `_analyze_pdf_type`, `_process_native_pdf`) into document-service
- document-service gets: `./uploads:/app/uploads` volume mount, database access to `document_metadata`, gRPC client to vector-service
- Backend continues to accept HTTP uploads and forward to document-service via gRPC
- Celery tasks (`reprocess_document_after_save_task`, `bulk_reindex_batch_task`, `backfill_document_chunks_task`) move to document-service workers

**Backend changes:**
- Backend HTTP upload endpoints forward file bytes to document-service via streaming gRPC instead of writing to disk directly
- Backend's `celery_worker` and `celery_reindex_worker` containers stop consuming document processing queues — document-service workers take over
- Backend retains non-document Celery queues (RSS, agents, etc.)

**Dependencies added to document-service:**
- `aiofiles`, `pdfplumber`, `PyPDF2` (PDF processing)
- `asyncpg` or shared DB layer (document_metadata access)
- `grpcio` client for vector-service
- Celery (or lighter alternative) for async queue consumption

**Risk:** Medium. The processing pipeline is the largest single chunk of code to move. But it's already async (queue-based), so the backend/document-service boundary is at the queue — a natural seam.

**What the backend sheds:** PDF extraction, text chunking, pdfplumber/PyPDF2 dependencies, large file buffering during processing, CPU-intensive operations.

### Phase 2: Migrate File I/O and Folder Operations

**Goal:** All filesystem operations move to document-service. Backend no longer mounts `./uploads`.

**Scope:**
- Move `folder_service.py`, `file_manager_service.py`, `collab_persist.py`, `document_version_service.py`, `file_encryption_service.py`, `file_recovery_service.py`, `zip_processor_service.py` into document-service
- Move `document_repository.py` and `document_version_repository.py` into document-service
- Expand proto with all folder, file management, versioning, and encryption RPCs
- Backend HTTP endpoints become thin gRPC proxies (same pattern as llm-orchestrator proxy)
- Remove `./uploads:/app/uploads` mount from backend, celery_worker, celery_reindex_worker containers

**Backend API changes:**
- `document_api.py` upload endpoints → stream bytes to document-service `UploadDocument` RPC
- `document_api.py` download endpoints → proxy `DownloadDocument` streaming RPC (or redirect to presigned URL if S3 is in place)
- `document_api.py` content GET/PUT → `GetDocumentContent` / `UpdateDocumentContent` RPCs
- `folder_api.py` → all calls proxy to document-service folder RPCs
- `file_manager_api.py` → all calls proxy to document-service file management RPCs

**tools-service changes:**
- Document gRPC handlers (`document_handlers.py`, `file_creation_handlers.py`, `document_edit_handlers.py`) move from tools-service to document-service
- tools-service no longer imports document-related backend code
- tools-service no longer mounts `./uploads`
- When tools-service tools need document data, they call document-service gRPC

**image-vision-service changes:**
- Instead of reading images from a mounted volume by path, image-vision-service receives image bytes in the gRPC request (the `image_path` field becomes `image_data bytes`)
- Or: document-service provides a presigned URL / temp file mechanism
- Remove `./uploads:/app/uploads:ro` mount from image-vision-service

**Risk:** High. Broad surface area — every HTTP endpoint that currently calls a service function directly needs to become a gRPC proxy call. Thorough integration testing required.

### Phase 3: Storage Backend Abstraction (S3/MinIO)

**Goal:** Document-service abstracts its storage behind a `StorageBackend` protocol, enabling S3-compatible storage.

With Phases 1-2 complete, the storage abstraction is contained entirely within document-service. No other container touches the filesystem for document storage.

**Scope:**
- Implement `StorageBackend` protocol inside document-service: `write`, `read`, `stream`, `delete`, `exists`, `get_download_url`, `copy`, `move`
- `LocalFileSystemBackend` wraps current aiofiles behavior
- `S3CompatibleBackend` uses aiobotocore (works with MinIO self-hosted or AWS S3)
- `FolderService` path resolution returns logical keys (`str`) instead of `Path` objects
- Add MinIO container to docker-compose.yml for self-hosted deployments
- `DownloadDocument` RPC can return a presigned URL for direct browser download

**Impact on other services:** None. Document-service is the only service that owns blob storage. The storage backend choice is an internal implementation detail.

**Relationship to S3_STORAGE_BACKEND_PLAN.md:** That plan's Phase 1-3 (protocol + local backend + S3 backend) executes here, but scoped to document-service only instead of scattered across the backend monolith. The effort is smaller because there's one codebase to change, not eight files across four containers.

### Phase 4: Consolidate Entity Extraction

**Goal:** Clean up the entity extraction pipeline.

`ExtractEntities` already lives in document-service. After Phases 1-2, the document processing pipeline also lives there. Entity extraction can be called as an internal function during document processing instead of a separate gRPC round-trip, reducing latency for the extraction step.

This is a minor optimization, not a structural change.

## Docker Compose Changes

### Phase 1

```yaml
document-service:
  build:
    context: .
    dockerfile: ./document-service/Dockerfile
  environment:
    - DATABASE_URL=postgresql://bastion_user:...@pgbouncer:5432/bastion_knowledge_base
    - REDIS_URL=redis://redis:6379
    - VECTOR_SERVICE_URL=vector-service:50053
    # ... processing config
  volumes:
    - ./uploads:/app/uploads
  depends_on:
    postgres: { condition: service_healthy }
    pgbouncer: { condition: service_healthy }
    redis: { condition: service_started }
    vector-service: { condition: service_healthy }
```

### Phase 2 (backend loses uploads mount)

```yaml
backend:
  volumes:
    # - ./uploads:/app/uploads  # REMOVED
    - ./data_imports:/app/data_imports
    - ./logs:/app/logs

tools-service:
  volumes:
    # - ./uploads:/app/uploads  # REMOVED
    - ./logs:/app/logs

image-vision-service:
  volumes: []
    # - ./uploads:/app/uploads:ro  # REMOVED

celery_worker:
  volumes:
    # - ./uploads:/app/uploads  # REMOVED
    - ./logs:/app/logs

celery_reindex_worker:
  volumes:
    # - ./uploads:/app/uploads  # REMOVED
    - ./logs:/app/logs
```

Only document-service mounts `./uploads` (until Phase 3 replaces it with S3).

### Phase 3 (S3/MinIO)

```yaml
minio:
  image: minio/minio:latest
  command: server /data --console-address ":9001"
  environment:
    MINIO_ROOT_USER: ${MINIO_ROOT_USER:-bastion}
    MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD:-bastion_password}
  volumes:
    - bastion_minio_data:/data

document-service:
  environment:
    - STORAGE_BACKEND_TYPE=s3
    - STORAGE_S3_ENDPOINT_URL=http://minio:9000
    - STORAGE_S3_BUCKET=bastion-uploads
    - STORAGE_S3_PATH_STYLE=true
  volumes: []
    # - ./uploads:/app/uploads  # REMOVED — S3 replaces local storage
```

## WebDAV Special Case

WebDAV (`backend/webdav/orgmode_provider.py`) reads org files from the `free_form_notes` database table, not from the uploads volume. It currently mounts `./uploads` but doesn't use it for org file access. After Phase 2:

- WebDAV container keeps its own small volume only if it needs scratch space
- Org file reads continue from the database (no change)
- If WebDAV needs to serve non-org documents in the future, it calls document-service gRPC

## Resource Profile Impact

| Scenario | Backend (current) | Backend (after Phase 2) | document-service |
|----------|-------------------|------------------------|------------------|
| Memory baseline | ~400–600 MB | ~150–250 MB | ~300–400 MB |
| 10 concurrent uploads | +500 MB spike | No impact | +500 MB spike |
| PDF processing (large) | +200 MB, high CPU | No impact | +200 MB, high CPU |
| ZIP generation | +300 MB, high I/O | No impact | +300 MB, high I/O |
| Chat/WebSocket load | +50 MB | +50 MB | No impact |
| Entity extraction | N/A (separate container today) | N/A | Already included |

The backend sheds its heaviest dependencies: `pdfplumber`, `PyPDF2`, `aiofiles` (for document I/O), and the parallel document processor. Its Docker image shrinks significantly.

## Estimated Effort

| Phase | Effort | Risk |
|-------|--------|------|
| Phase 1: Processing pipeline migration | 3–4 days | Medium |
| Phase 2: Full file I/O and folder migration | 4–5 days | High (broad surface area) |
| Phase 3: Storage backend abstraction (S3) | 1–2 days | Low (contained in one service) |
| Phase 4: Entity extraction consolidation | 0.5 days | None |
| **Total** | **~9–12 days** | |

Phase 3 is notably cheaper than the original S3 plan's estimate (4 days) because by then, all filesystem I/O is already consolidated in one service. The storage abstraction only needs to be implemented once, in one place.

## Migration Strategy

### Parallel Deployment (Phase 1)

During Phase 1, both backend and document-service can process documents. A feature flag (`USE_DOCUMENT_SERVICE_PROCESSING=true`) routes new uploads through document-service while the backend retains its processing code as a fallback. Once document-service processing is validated, remove backend processing code.

### Backward Compatibility (Phase 2)

Backend HTTP endpoints maintain the same request/response contracts. From the frontend's perspective, nothing changes — the same URLs accept the same payloads and return the same responses. The only difference is that the backend proxies to document-service instead of handling I/O directly.

### Data Migration

No data migration is needed for Phases 1-2. The database schema is unchanged. The filesystem layout is unchanged. Document-service reads the same `./uploads` volume and the same database tables.

Phase 3 (S3) requires a one-time file migration (see `S3_STORAGE_BACKEND_PLAN.md` migration strategy).

## Open Questions

1. **Celery vs. direct async processing:** Document-service could use Celery workers (same infrastructure as today) or switch to a lighter model (direct async tasks, Redis Streams). Celery adds dependency weight but provides retry, monitoring (Flower), and queue routing out of the box.

2. **Shared code packaging:** Both tools-service and document-service need shared code (models, database helpers, proto stubs). Options: (a) continue `sys.path` imports from `backend/`, (b) extract a shared `bastion-common` package, (c) each service copies what it needs at build time. Option (b) is cleanest long-term but adds packaging overhead.

3. **Knowledge graph ownership:** The KG write pipeline (entity extraction → Neo4j storage) currently runs during document processing. If document-service owns processing, it also owns KG writes, which means it needs Neo4j credentials and the KG service code. Alternative: document-service emits events, a separate KG worker consumes them.

4. **gRPC streaming for uploads:** Large file uploads (up to 1.5 GB) over gRPC streaming require careful flow control. Alternative: document-service exposes an HTTP endpoint for uploads alongside gRPC, or backend streams chunks at a controlled rate.

5. **WebSocket notifications:** File operations currently emit WebSocket notifications (e.g., "document uploaded", "folder moved") via `WebSocketNotifier`. After migration, document-service needs a way to trigger WebSocket events in the backend — either via a callback RPC, Redis pub/sub, or an event bus.

## Related Architecture

- `docs/dev-notes/S3_STORAGE_BACKEND_PLAN.md` — S3 storage abstraction (executes as Phase 3 of this plan)
- `protos/document_service.proto` — current proto (to be expanded)
- `protos/tool_service.proto` — document RPCs to migrate out
- `docker-compose.yml` — container definitions
- `backend/config.py` — `UPLOAD_DIR`, `PROCESSED_DIR` settings that move to document-service config
