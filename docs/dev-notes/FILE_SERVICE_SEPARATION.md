# File/Folder Service Separation — Architecture Analysis

Analysis of whether to decouple file and folder operations from the backend into a dedicated microservice. Covers performance, reliability, security, trade-offs, and a phased implementation plan.

## Current State

### Where File Operations Live Today

| Layer | Location | Responsibility |
|-------|----------|---------------|
| **API** | `backend/api/document_api.py` | Upload, download, content CRUD, metadata, reprocessing |
| **API** | `backend/api/folder_api.py` | Folder tree, contents, CRUD, move, ZIP download |
| **API** | `backend/api/file_manager_api.py` | Place file, move, delete, rename, create folder structures |
| **Service** | `backend/services/document_service_v2.py` | Upload processing, PDF analysis, chunking, text extraction, status management |
| **Service** | `backend/services/folder_service.py` | Folder CRUD, physical path resolution, tree building, metadata inheritance, ZIP generation |
| **Service** | `backend/services/file_manager/file_manager_service.py` | Orchestration of file placement, folder structure creation, WebSocket notifications |
| **Repository** | `backend/repositories/document_repository.py` | PostgreSQL CRUD, RLS, status tracking, exemption management, pagination |
| **Background** | Celery workers (in backend container) | Async document processing pipeline |

### The Existing Partial Split

File operations are already partially split across services:

- **tools-service** (gRPC, port 50052): Handles document **reads** — `SearchDocuments`, `GetDocument`, `GetDocumentContent`, `UpdateDocumentContent`, `CreateUserFile`, `CreateUserFolder`
- **backend** (HTTP, port 8081): Handles document **writes** — upload, processing, folder management, Celery dispatch

This is an informal CQRS-like pattern that emerged organically.

### Current Container Landscape

| Container | Port | Protocol | Domain |
|-----------|------|----------|--------|
| backend | 8081 | HTTP | API gateway, file ops, auth, WebSocket, Celery |
| tools-service | 50052 | gRPC | Document search/retrieval, RSS, entities, org-mode |
| vector-service | 50053 | gRPC | Embeddings, Qdrant operations |
| image-vision-service | 50056 | gRPC | Face/object detection |
| data-service | 50054 | gRPC | Data workspace, CSV/Excel |
| crawl4ai-service | 50055 | gRPC | Web crawling |
| connections-service | 50057 | gRPC | OAuth, email, chat channels |
| llm-orchestrator | 50051 | gRPC | LangGraph agent orchestration |

## Arguments For Separation

### 1. Security (Strongest Argument)

File upload/processing is the largest attack surface in the system. Uploads accept arbitrary user content (PDFs, images, text, URLs). A vulnerability in file handling (path traversal, malicious PDF, zip bomb) currently has access to everything in the backend container:

- Database credentials for all tables
- API keys (LLM, search, etc.)
- Auth middleware and session secrets
- WebSocket connections
- Every gRPC client credential

A dedicated file service running with principle of least privilege:

- Only needs filesystem access and credentials for `document_metadata` / `folder` tables
- No access to auth secrets, LLM API keys, email OAuth tokens
- Can run with stricter container security (read-only root filesystem except `/uploads`, restricted network policy)
- File content scanning and validation happen in an isolated sandbox
- A compromised file service cannot escalate to auth or LLM operations

### 2. Reliability (Blast Radius Containment)

The backend currently handles too many responsibilities. A single container runs:

- HTTP API gateway (auth, routing, CORS)
- File upload buffering (memory-intensive for large files)
- Document processing pipeline (PDF extraction, text chunking — CPU-intensive)
- ZIP library generation (CPU + I/O intensive)
- Folder tree construction (recursive DB queries)
- WebSocket management
- Celery task dispatch
- gRPC client management for 6+ services

Failure scenarios that take down the entire API:

- Large PDF upload causes OOM during processing
- ZIP generation exhausts disk space or memory
- Recursive folder query causes connection pool exhaustion
- Malformed file triggers unhandled exception in processing pipeline

With separation, a file processing crash does not affect chat, search, agent operations, or WebSocket connections.

### 3. Performance (Independent Resource Profiles)

File operations have fundamentally different resource characteristics than API routing:

| Operation | CPU | Memory | I/O | Duration |
|-----------|-----|--------|-----|----------|
| File upload | Low | High (buffering) | High (disk write) | Seconds |
| PDF processing | High | High | Medium | Seconds–minutes |
| ZIP library download | High | High | High | Seconds–minutes |
| Folder tree query | Low | Medium | Low | Milliseconds |
| Chat API routing | Low | Low | Low | Milliseconds |
| WebSocket management | Low | Low | Low | Persistent |

Under load, 20 simultaneous document uploads degrade chat response times for all users because they compete for the same container's CPU and memory. Separation enables:

- Independent scaling (more upload throughput = more file-service replicas)
- Resource limits tuned per workload (file service gets higher memory, backend gets lower)
- Different restart policies and health checks per service

### 4. Architectural Consistency

Seven domain-specific services already have their own containers (vectorization, image processing, web crawling, data workspace, external connections, LLM orchestration, tools). File/folder operations are arguably a more fundamental domain than several of those. Leaving them in the backend is an inconsistency.

## Arguments Against Separation

### 1. Transactional Complexity

File creation currently does three things atomically (or near-atomically):

1. Write file to filesystem
2. Insert record in PostgreSQL
3. Kick off Celery processing task

Across a service boundary, you lose easy transaction guarantees. Solutions:

- **Saga pattern**: Create DB record in PENDING state → write file → update to ACTIVE
- **Shared volume**: Docker volumes give both services filesystem access (already the pattern with tools-service and `/uploads`)
- **Outbox pattern**: Write to DB, use CDC or polling to trigger file writes

Solvable, but adds complexity.

### 2. Shared Filesystem Coupling

Files live at `/uploads/Users/{username}/...`. Both backend and file service need access to this volume. Docker volumes handle this, and tools-service already accesses the same filesystem, but it means the services are not fully decoupled at the storage layer.

If we ever move to object storage (S3/MinIO), this coupling disappears entirely and the split becomes cleaner.

### 3. Additional Network Hop

Every write operation gains a gRPC call. On a Docker bridge network, this is sub-millisecond overhead — negligible compared to actual file I/O and database operations. Read operations already pay this cost via tools-service.

### 4. Operational Overhead

Another container to build, deploy, monitor, and debug. With 12+ containers already running, the marginal cost is low but nonzero. Adds to:

- Docker Compose complexity
- CI/CD pipeline
- Log aggregation scope
- Health check monitoring

### 5. Shared Codebase Entanglement

tools-service already imports from `backend/` (repositories, services) via Python path manipulation. A file service would follow the same pattern, meaning changes to repositories may need coordinated deployments. This is an existing trade-off, not a new one.

## Resource Profile Comparison

| Scenario | Backend (current) | Backend (after split) | File Service |
|----------|-------------------|----------------------|--------------|
| Memory baseline | ~400–600 MB | ~200–300 MB | ~200–300 MB |
| 10 concurrent uploads | +500 MB spike | No impact | +500 MB spike |
| PDF processing (large) | +200 MB, high CPU | No impact | +200 MB, high CPU |
| ZIP generation | +300 MB, high I/O | No impact | +300 MB, high I/O |
| Chat/WebSocket load | +50 MB | +50 MB | No impact |

The key insight: resource spikes from file operations currently affect all backend consumers. Separation isolates these spikes.

## Proposed Implementation Plan

### Phase 1: Extract Document Processing Pipeline (Highest Value, Lowest Risk)

**What:** Move the async processing pipeline (extract → chunk → embed → store) to a dedicated container.

**Why this first:**
- Already async via Celery — no API changes needed
- Most resource-intensive operations (PDF extraction, chunking)
- Purely a background worker — no new API surface
- Immediate wins: resource isolation, independent scaling, crash containment

**Scope:**
- New container: `document-processor` (or dedicated Celery worker container)
- Consumes jobs from existing Redis queue
- Handles: text extraction, PDF analysis, chunking, format conversion
- Calls vector-service for embeddings (already gRPC)
- Updates document status in DB

**Impact:** Zero API changes. Backend dispatches to queue as before; a different container consumes it.

### Phase 2: Formalize Read/Write Split with file-service.proto

**What:** Create a `file-service` gRPC service that owns all file write operations.

**New proto definition scope:**
- `UploadDocument` — file upload handling
- `CreateFolder` / `DeleteFolder` / `MoveFolder`
- `MoveFile` / `RenameFile` / `DeleteFile`
- `UpdateDocumentContent` / `UpdateDocumentMetadata`
- `GenerateLibraryZip`
- `GetFolderTree` / `GetFolderContents`

**Backend becomes:** A thin API gateway that authenticates HTTP requests and forwards to file-service via gRPC. Same pattern as how it currently proxies to llm-orchestrator.

**Port allocation:** 50058 (next available in current range)

### Phase 3: Consolidate File Operations (Optional)

**What:** Migrate document read operations from tools-service into file-service.

**Result:** Single `file-service` owns all document/folder operations. Tools-service focuses on search, RSS, entities, and non-file domains.

**Trade-off:** Cleaner domain boundaries vs. larger single service. Evaluate after Phase 2 stabilizes.

## Decision Matrix

| Factor | Keep in Backend | Phase 1 Only | Full Separation |
|--------|----------------|--------------|-----------------|
| Security | Low (shared attack surface) | Medium (processing isolated) | High (full isolation) |
| Reliability | Low (shared blast radius) | Medium (processing crashes isolated) | High (full isolation) |
| Performance | Low (resource contention) | Medium (processing offloaded) | High (independent scaling) |
| Complexity | Low (status quo) | Low (queue-based, no API changes) | Medium (new gRPC service) |
| Maintenance | Low (one codebase) | Low (shared code pattern) | Medium (additional service) |
| Migration effort | None | Small (container + worker config) | Medium (proto, clients, routing) |

## Recommendation

**Phase 1 is a clear win** — low risk, high reward, zero API changes. It should be implemented regardless of whether Phase 2/3 proceed.

**Phase 2 is justified** when any of these become true:
- File upload volume increases significantly
- Security audit flags the shared backend attack surface
- Backend container OOM or resource contention incidents occur
- Team size grows enough to benefit from service ownership boundaries

**Phase 3 is optional** and should be evaluated after Phase 2 has been running in production.

## Open Questions

1. **Object storage migration**: Should Phase 2 coincide with moving from filesystem to S3/MinIO? This would eliminate the shared volume coupling and make the split cleaner.
2. **Celery vs. dedicated queue**: Phase 1 could use existing Celery infrastructure or switch to a simpler Redis Streams / dedicated worker model. Celery adds overhead; a lighter worker may be preferable.
3. **File validation service**: Should file scanning/validation (antivirus, format validation, size limits) be a separate concern from file storage, or part of the file service?
4. **Shared code strategy**: Continue with Python path imports from `backend/`, or extract shared repositories into a proper shared package?

## Related Architecture

- Vector Service: `protos/vector_service.proto` — embeddings and Qdrant operations
- Tools Service: `protos/tool_service.proto` — document reads, search, RSS
- LLM Orchestrator: `protos/orchestrator.proto` — agent communication
- Image Vision: `protos/image_vision.proto` — face/object detection
- Connections Service: `protos/connections_service.proto` — OAuth, email, chat channels
