# Remote Data Catalog — Architecture & Implementation Plan

## Overview

A catalog service that connects to remote file datastores (NFS, SMB, S3, and future
source types) at runtime, crawls their file trees, analyzes content with AI, and stores
rich structured metadata in a queryable catalog. Agents can then search and browse
cataloged files across all connected sources.

**Core principles:**
- Connections are created at runtime through the UI — no Docker-level changes per source
- The backend never opens sockets to external file systems
- A dedicated `catalog-service` container owns all external storage protocol communication
- The catalog schema is universal and partitioned per connection for clean lifecycle management

---

## Architecture Overview

```
Frontend (Settings UI)
    │  REST API
    ▼
Backend (FastAPI)
  ├─ remote_data_sources CRUD
  ├─ Partition CREATE/DROP on connection lifecycle
  └─ Catalog query endpoints (search, browse, status)
    │  gRPC
    ▼
catalog-service (new container)
  ├─ NFSProvider    (libnfs-python — userspace NFS client)
  ├─ SMBProvider    (smbprotocol  — pure Python SMB2/3)
  ├─ S3Provider     (s3fs + aiobotocore)
  ├─ SFTPProvider   (asyncssh)
  └─ ... future providers
    │  asyncpg (shared PostgreSQL)
    ▼
PostgreSQL
  ├─ remote_data_sources      (connection registry)
  ├─ remote_scan_jobs         (partitioned parent)
  │   └─ remote_scan_jobs_p_<source_id>   (per-connection partition)
  └─ file_catalog             (partitioned parent)
      └─ file_catalog_p_<source_id>       (per-connection partition)
    │  gRPC
    ▼
vector-service  (Qdrant — for vectorizing catalog entries)
```

The backend's role is strictly: connection lifecycle management, triggering scans via
gRPC, and serving catalog data from PostgreSQL to agents and the UI. It is never a
network client to NFS, SMB, or S3.

---

## New Container: `catalog-service`

Modeled directly on `crawl4ai-service`. Same structure, same gRPC-server pattern.

```
catalog-service/
├── Dockerfile
├── requirements.txt
├── main.py                       ← gRPC server (mirrors crawl4ai-service/main.py)
├── config/
│   └── settings.py
└── service/
    ├── grpc_service.py           ← gRPC method dispatch
    ├── catalog_service.py        ← scan orchestration, analysis pipeline
    ├── analyzer.py               ← text extraction + LLM summarization
    ├── providers/
    │   ├── base_provider.py      ← BaseRemoteStorageProvider ABC
    │   ├── provider_registry.py  ← provider lookup by source_type string
    │   ├── nfs_provider.py
    │   ├── smb_provider.py
    │   ├── s3_provider.py
    │   ├── sftp_provider.py
    │   └── (gcs_provider.py, azure_provider.py — Tier 2)
    └── db/
        ├── catalog_repository.py ← asyncpg queries against file_catalog + scan_jobs
        └── source_repository.py  ← reads remote_data_sources + decrypts credentials
```

### `requirements.txt` (catalog-service)

```
# Core
asyncpg>=0.29.0
grpcio>=1.60.0
grpcio-tools>=1.60.0
protobuf>=4.25.0
pydantic>=2.5.0
cryptography>=42.0.0    # Fernet — same SECRET_KEY as backend

# MIME detection
python-magic>=0.4.27

# Storage providers (Tier 1)
smbprotocol>=1.13.0     # Pure Python SMB2/3, no system deps
s3fs>=2024.1.0
aiobotocore>=2.12.0
asyncssh>=2.14.0        # SFTP
libnfs-python>=0.1.0    # NFS userspace client (needs libnfs-dev in Dockerfile)

# Text extraction (mirrors backend DocumentProcessor capabilities)
PyPDF2>=3.0.0
pdfplumber>=0.10.0
python-docx>=1.1.0
beautifulsoup4>=4.12.0
```

### `Dockerfile` (catalog-service)

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libmagic1 \
    libnfs-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY ../protos /app/protos
RUN python -m grpc_tools.protoc \
    -I /app/protos \
    --python_out=/app/protos \
    --grpc_python_out=/app/protos \
    /app/protos/catalog_service.proto

CMD ["python", "main.py"]
```

---

## gRPC Contract

**`protos/catalog_service.proto`**

```protobuf
syntax = "proto3";
package catalog;

service CatalogService {
    rpc TestConnection   (TestConnectionRequest)   returns (TestConnectionResponse);
    rpc TriggerScan      (TriggerScanRequest)       returns (TriggerScanResponse);
    rpc CancelScan       (CancelScanRequest)        returns (CancelScanResponse);
    rpc GetScanStatus    (GetScanStatusRequest)     returns (ScanStatusResponse);
    rpc ListDirectory    (ListDirectoryRequest)     returns (ListDirectoryResponse);
}

message TestConnectionRequest {
    string source_id = 1;          // Backend passes source_id; service reads config from DB
}

message TestConnectionResponse {
    bool   success       = 1;
    string error         = 2;
    int32  sample_count  = 3;      // Number of files spotted in root (spot check)
    int64  latency_ms    = 4;
}

message TriggerScanRequest {
    string source_id  = 1;
    string scan_path  = 2;         // Optional: scan a sub-path only
    bool   full_rescan = 3;        // If true, ignore hashes — reprocess everything
}

message TriggerScanResponse {
    string job_id  = 1;
    string status  = 2;            // "started" or "already_running"
}

message CancelScanRequest  { string job_id = 1; }
message CancelScanResponse { bool success  = 1; }

message GetScanStatusRequest { string job_id = 1; }

message ScanStatusResponse {
    string job_id            = 1;
    string status            = 2;  // running | completed | failed | cancelled
    int32  files_discovered  = 3;
    int32  files_indexed     = 4;
    int32  files_skipped     = 5;
    int32  files_errored     = 6;
    int64  bytes_processed   = 7;
    string error_message     = 8;
}

message ListDirectoryRequest {
    string source_id = 1;
    string path      = 2;
}

message ListDirectoryResponse {
    bool          success = 1;
    string        error   = 2;
    repeated FileEntry entries = 3;
}

message FileEntry {
    string name          = 1;
    string path          = 2;
    bool   is_directory  = 3;
    int64  size_bytes    = 4;
    int64  modified_unix = 5;
    string mime_type     = 6;
}
```

---

## Database Schema

### `remote_data_sources` (connection registry)

```sql
CREATE TABLE remote_data_sources (
    source_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    name          VARCHAR(255) NOT NULL,
    description   TEXT,

    -- Protocol type: nfs | smb | s3 | gcs | azure_blob | sftp | ftp | webdav
    source_type   VARCHAR(50) NOT NULL,

    -- Non-secret connection parameters (host, bucket, share, region, etc.)
    -- Shape is provider-specific; see "connection_config shapes" below
    connection_config JSONB NOT NULL,

    -- Fernet-encrypted credentials (same encryption as external_connections_service.py)
    encrypted_credentials TEXT,

    -- What to scan and how
    scan_config JSONB NOT NULL DEFAULT '{
        "root_paths": ["/"],
        "include_extensions": null,
        "exclude_patterns": ["**/.git/**", "**/node_modules/**", "**/tmp/**"],
        "max_depth": -1,
        "max_file_size_mb": 500,
        "analyze_content": true,
        "vectorize": true
    }',

    -- Cron schedule string; null = manual only
    scan_schedule VARCHAR(100),

    -- Connection health
    status             VARCHAR(50) DEFAULT 'active',  -- active | paused | error
    last_tested_at     TIMESTAMPTZ,
    last_test_status   VARCHAR(50),
    last_test_error    TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(user_id, source_type, name)
);

CREATE INDEX idx_remote_sources_user   ON remote_data_sources(user_id);
CREATE INDEX idx_remote_sources_type   ON remote_data_sources(source_type);
CREATE INDEX idx_remote_sources_status ON remote_data_sources(status);
```

### `remote_scan_jobs` (partitioned by source_id)

```sql
CREATE TABLE remote_scan_jobs (
    job_id       UUID NOT NULL DEFAULT gen_random_uuid(),
    source_id    UUID NOT NULL REFERENCES remote_data_sources(source_id) ON DELETE CASCADE,
    triggered_by VARCHAR(50) DEFAULT 'scheduled',  -- scheduled | manual | api
    status       VARCHAR(50) DEFAULT 'running',    -- running | completed | failed | cancelled
    scan_path    TEXT,
    full_rescan  BOOLEAN DEFAULT false,

    files_discovered INTEGER DEFAULT 0,
    files_indexed    INTEGER DEFAULT 0,
    files_skipped    INTEGER DEFAULT 0,
    files_errored    INTEGER DEFAULT 0,
    files_deleted    INTEGER DEFAULT 0,
    bytes_processed  BIGINT  DEFAULT 0,

    started_at    TIMESTAMPTZ DEFAULT NOW(),
    completed_at  TIMESTAMPTZ,
    error_message TEXT,

    PRIMARY KEY (source_id, job_id)
) PARTITION BY LIST (source_id);

CREATE INDEX idx_scan_jobs_status ON remote_scan_jobs(status);
```

### `file_catalog` (partitioned by source_id)

```sql
CREATE TABLE file_catalog (
    catalog_id    UUID NOT NULL DEFAULT gen_random_uuid(),
    source_id     UUID NOT NULL REFERENCES remote_data_sources(source_id) ON DELETE CASCADE,

    -- File identity (path relative to source root)
    file_path     TEXT NOT NULL,
    file_name     TEXT NOT NULL,
    file_extension TEXT,
    mime_type     TEXT,
    file_size_bytes BIGINT,

    -- Change detection
    content_hash  TEXT,        -- SHA-256 (NFS/SMB/SFTP)
    etag          TEXT,        -- Object store ETag (S3/GCS/Azure)
    last_modified TIMESTAMPTZ,

    -- AI-generated metadata (populated after analysis)
    summary               TEXT,
    key_topics            TEXT[],
    entities              JSONB,   -- {persons[], orgs[], locations[], dates[]}
    language              TEXT,
    quality_score         FLOAT,
    document_type         TEXT,    -- report|code|data|correspondence|manual|other
    inferred_time_period  TEXT,    -- "2024" or "2022-2024" if identifiable

    -- Quick preview without re-fetching
    extracted_text_preview TEXT,   -- First 2000 characters

    -- Vector search
    vector_id TEXT,                -- Qdrant record ID if vectorized

    -- Lifecycle
    indexed_at          TIMESTAMPTZ,
    last_seen_at        TIMESTAMPTZ,
    status              VARCHAR(50) DEFAULT 'pending',
    -- pending | processing | indexed | error | deleted | skipped
    error_message       TEXT,
    last_scan_job_id    UUID,

    PRIMARY KEY (source_id, catalog_id),
    UNIQUE (source_id, file_path)
) PARTITION BY LIST (source_id);

CREATE INDEX idx_file_catalog_status   ON file_catalog(status);
CREATE INDEX idx_file_catalog_hash     ON file_catalog(content_hash);
CREATE INDEX idx_file_catalog_mime     ON file_catalog(mime_type);
CREATE INDEX idx_file_catalog_modified ON file_catalog(last_modified);
CREATE INDEX idx_file_catalog_topics   ON file_catalog USING GIN(key_topics);
CREATE INDEX idx_file_catalog_entities ON file_catalog USING GIN(entities);
```

### Partition lifecycle (called by backend repository, not DDL migrations)

When a connection is **created**:
```sql
CREATE TABLE file_catalog_p_<source_id>
    PARTITION OF file_catalog
    FOR VALUES IN ('<source_id>');

CREATE TABLE remote_scan_jobs_p_<source_id>
    PARTITION OF remote_scan_jobs
    FOR VALUES IN ('<source_id>');
```

When a connection is **deleted**:
```sql
DROP TABLE file_catalog_p_<source_id>;       -- instant, atomic cleanup
DROP TABLE remote_scan_jobs_p_<source_id>;   -- instant, atomic cleanup
-- then:
DELETE FROM remote_data_sources WHERE source_id = '<source_id>';
```

These DDL calls happen inside the `RemoteDataSourceRepository.create_source()` and
`delete_source()` methods, not in migrations.

---

## `connection_config` Shapes by Source Type

These are stored as JSONB in `remote_data_sources.connection_config`. Each provider's
`connection_config_schema()` class method returns the JSON Schema, which drives dynamic
form rendering in the UI.

```json
// NFS
{
  "host": "192.168.1.50",
  "export_path": "/volume1/archive",
  "nfs_version": "3",
  "options": "nolock,soft,timeo=30"
}

// SMB / CIFS
{
  "host": "192.168.1.50",
  "share": "projects",
  "domain": "WORKGROUP",
  "port": 445
}

// S3 / S3-compatible (MinIO, Backblaze B2, Wasabi, Cloudflare R2)
{
  "bucket": "my-archive",
  "prefix": "docs/",
  "region": "us-east-1",
  "endpoint_url": null
}
// endpoint_url = null → AWS S3; set to custom URL for self-hosted S3-compatible stores

// Azure Blob Storage
{
  "account_name": "mystorageaccount",
  "container": "mycontainer",
  "prefix": ""
}

// SFTP
{
  "host": "sftp.example.com",
  "port": 22,
  "root_path": "/data"
}
```

`encrypted_credentials` decrypted shape by type:

```json
// NFS (typically unauthenticated)
{ "type": "none" }

// SMB
{ "username": "svcaccount", "password": "..." }

// S3
{ "access_key_id": "AKIA...", "secret_access_key": "..." }

// Azure Blob
{ "account_key": "..." }
// OR: { "connection_string": "DefaultEndpointsProtocol=https;..." }

// SFTP
{ "username": "user", "password": "...", "private_key_pem": "-----BEGIN RSA..." }
```

---

## Provider Interface

**`catalog-service/service/providers/base_provider.py`**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional


@dataclass
class FileInfo:
    path: str
    name: str
    size_bytes: int
    last_modified: float    # Unix timestamp
    is_directory: bool
    content_hash: Optional[str] = None
    etag: Optional[str] = None
    mime_type: Optional[str] = None


@dataclass
class TestResult:
    success: bool
    error: Optional[str] = None
    sample_count: int = 0
    latency_ms: int = 0


class BaseRemoteStorageProvider(ABC):
    source_type: str  # class-level: "nfs", "smb", "s3", etc.

    @abstractmethod
    async def connect(self, config: dict, credentials: dict) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    @abstractmethod
    async def test_connection(self, config: dict, credentials: dict) -> TestResult: ...

    @abstractmethod
    async def walk(
        self, root_path: str, max_depth: int = -1
    ) -> AsyncIterator[FileInfo]: ...

    @abstractmethod
    async def read_file_chunk(self, path: str, max_bytes: int = 512_000) -> bytes: ...

    @abstractmethod
    async def get_file_stat(self, path: str) -> FileInfo: ...

    @classmethod
    @abstractmethod
    def connection_config_schema(cls) -> dict: ...  # JSON Schema — drives UI form

    @classmethod
    @abstractmethod
    def credentials_schema(cls) -> dict: ...        # JSON Schema — drives UI form
```

**`catalog-service/service/providers/provider_registry.py`**

```python
from typing import Dict, Type
from .base_provider import BaseRemoteStorageProvider


class RemoteStorageProviderRegistry:
    _providers: Dict[str, Type[BaseRemoteStorageProvider]] = {}

    @classmethod
    def register(cls, provider_class: Type[BaseRemoteStorageProvider]) -> None:
        cls._providers[provider_class.source_type] = provider_class

    @classmethod
    def get(cls, source_type: str) -> BaseRemoteStorageProvider:
        if source_type not in cls._providers:
            raise ValueError(f"Unknown source type: {source_type}")
        return cls._providers[source_type]()

    @classmethod
    def list_types(cls) -> list[str]:
        return list(cls._providers.keys())

    @classmethod
    def get_schemas(cls) -> dict:
        """Return all provider schemas for UI form generation."""
        return {
            st: {
                "connection_config": p.connection_config_schema(),
                "credentials": p.credentials_schema(),
            }
            for st, p in cls._providers.items()
        }
```

Registration at service startup (in `main.py` or `catalog_service.py`):

```python
from service.providers.provider_registry import RemoteStorageProviderRegistry
from service.providers.nfs_provider import NFSProvider
from service.providers.smb_provider import SMBProvider
from service.providers.s3_provider import S3Provider
from service.providers.sftp_provider import SFTPProvider

RemoteStorageProviderRegistry.register(NFSProvider)
RemoteStorageProviderRegistry.register(SMBProvider)
RemoteStorageProviderRegistry.register(S3Provider)
RemoteStorageProviderRegistry.register(SFTPProvider)
```

---

## Analysis Pipeline

Each file goes through this pipeline inside the catalog-service after its bytes are
fetched from the remote source.

```
read_file_chunk(path, max_bytes=512_000)
    │
    ▼
detect MIME type            (python-magic, libmagic)
    │
    ├─ text/plain, text/markdown, text/html  ──► decode UTF-8
    ├─ application/pdf                        ──► pdfplumber text extraction
    ├─ application/vnd.openxmlformats...docx  ──► python-docx extraction
    ├─ image/*                                ──► skip text; store EXIF metadata
    └─ other binary                           ──► store filename + MIME only
    │
    ▼
if text available and analyze_content:
    LLM summarization call → summary, key_topics, document_type, inferred_time_period
    (via backend gRPC / direct LLM endpoint — catalog-service is not an LLM orchestrator)
    │
    ▼
if text available and vectorize:
    vector-service gRPC → Qdrant upsert → vector_id
    │
    ▼
upsert file_catalog_p_<source_id>
```

**LLM summarization prompt** (structured output):

```json
{
  "summary": "2-3 sentence description of the file's content and purpose",
  "key_topics": ["topic1", "topic2", "topic3"],
  "document_type": "report | code | data | correspondence | manual | other",
  "inferred_time_period": "YYYY or YYYY-YYYY if identifiable, else null"
}
```

**Change detection strategy** (avoids reprocessing unchanged files):

1. Fetch `last_modified` (mtime) from the remote stat — O(1), no data transfer
2. If mtime matches catalog record: update `last_seen_at`, skip
3. If mtime differs: fetch file chunk, compute SHA-256
4. If hash matches: update `last_seen_at` and `last_modified`, skip analysis
5. If hash differs: full analysis pipeline

For S3/GCS/Azure: use the provider's ETag instead of SHA-256 — it is pre-computed
server-side at no cost.

---

## Scan Job Flow

```
User clicks "Scan Now" in UI
    │  REST POST /api/catalog/sources/{source_id}/scan
    ▼
Backend → gRPC TriggerScan(source_id)
    ▼
catalog-service:
  1. Read source config from remote_data_sources (asyncpg)
  2. Decrypt credentials with Fernet (SECRET_KEY from env)
  3. Instantiate provider: RemoteStorageProviderRegistry.get(source_type)
  4. provider.connect(config, credentials)
  5. INSERT INTO remote_scan_jobs_p_<source_id> → job_id

  async for file_info in provider.walk(root_path):
      apply include/exclude filters from scan_config
      apply max_file_size_mb filter
      check change detection (mtime → hash → skip or analyze)
      if new/changed:
          spawn analysis coroutine (semaphore-limited concurrency)
          upsert file_catalog_p_<source_id>
      else:
          UPDATE last_seen_at
      UPDATE remote_scan_jobs (progress counters) every N files

  Mark files status='deleted' where last_seen_at < job started_at
  UPDATE remote_scan_jobs status='completed', completed_at=NOW()
  6. provider.disconnect()
```

Concurrency is bounded by a semaphore (default: 8 parallel file analyses). LLM calls
are the bottleneck, so this prevents overwhelming the orchestrator.

---

## Backend API Changes

### New REST endpoints in a new `backend/api/catalog_api.py`

| Method | Path | Description |
|--------|------|-------------|
| `GET`    | `/api/catalog/sources`                          | List user's data sources |
| `POST`   | `/api/catalog/sources`                          | Create connection + partitions |
| `GET`    | `/api/catalog/sources/{source_id}`              | Get source details |
| `PUT`    | `/api/catalog/sources/{source_id}`              | Update config / schedule |
| `DELETE` | `/api/catalog/sources/{source_id}`              | Drop partitions + delete |
| `POST`   | `/api/catalog/sources/{source_id}/test`         | Test connection via gRPC |
| `POST`   | `/api/catalog/sources/{source_id}/scan`         | Trigger scan via gRPC |
| `GET`    | `/api/catalog/sources/{source_id}/scan/{job_id}`| Scan progress |
| `DELETE` | `/api/catalog/sources/{source_id}/scan/{job_id}`| Cancel scan |
| `GET`    | `/api/catalog/sources/{source_id}/browse`       | List directory via gRPC |
| `GET`    | `/api/catalog/search`                           | Search catalog (all sources) |
| `GET`    | `/api/catalog/sources/{source_id}/files`        | List catalog entries |
| `GET`    | `/api/catalog/provider-schemas`                 | Dynamic form schemas for UI |

### New gRPC client in backend

`backend/clients/catalog_service_client.py` — mirrors the pattern of
`backend/clients/connections_service_client.py`.

---

## Agent-Facing Tools (Zone 1 / Zone 2)

New gRPC handlers in `backend/services/grpc_tool_service.py` (or a new
`catalog_tool_handlers.py`):

- `SearchFileCatalog` — semantic + keyword search across catalog entries
- `BrowseCatalogDirectory` — list catalog entries under a path prefix
- `GetFileMetadata` — full catalog record for a specific file path
- `GetCatalogSummary` — stats for a source (file count, types, last scan)

Thin orchestrator wrappers in
`llm-orchestrator/orchestrator/tools/file_catalog_tools.py`, registered with the
Action I/O Registry per the tool-io-contracts rule. Each tool returns a dict with typed
fields and a `formatted` field.

---

## New Proto Entry in `docker-compose.yml`

```yaml
catalog-service:
  build:
    context: ./catalog-service
    dockerfile: Dockerfile
  environment:
    - GRPC_PORT=50057
    - POSTGRES_HOST=postgres
    - POSTGRES_PORT=5432
    - POSTGRES_USER=${POSTGRES_USER}
    - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    - POSTGRES_DB=${POSTGRES_DB}
    - SECRET_KEY=${SECRET_KEY}           # Same key as backend for Fernet decryption
    - VECTOR_SERVICE_HOST=vector-service
    - VECTOR_SERVICE_PORT=50052
  depends_on:
    - postgres
    - vector-service
  networks:
    - bastion-network
```

Backend gets a new env var: `CATALOG_SERVICE_HOST=catalog-service`,
`CATALOG_SERVICE_PORT=50057`.

---

## Security Notes

- `encrypted_credentials` uses the same `Fernet` + `PBKDF2HMAC(SECRET_KEY)` pattern as
  `external_connections_service.py`. The catalog-service receives the `SECRET_KEY` env
  var and decrypts credentials locally before use. Credentials never travel over gRPC.

- The catalog-service only requires `SELECT` on `remote_data_sources` and full
  `INSERT/UPDATE` on `file_catalog` and `remote_scan_jobs`. No write access to other
  tables.

- `scan_config.max_file_size_mb` prevents runaway memory usage on binary blobs. Default
  500 MB for fetching; analysis only runs on the first `512_000` bytes.

- NFS/SMB connections should be tested with `TestConnection` before a full scan is
  triggered. The test makes a single directory listing to verify auth/network before
  committing to a long job.

---

## Implementation Checklist

### Phase 1 — Infrastructure

- [ ] `protos/catalog_service.proto` — write and compile proto
- [ ] `catalog-service/` directory, Dockerfile, requirements.txt
- [ ] `catalog-service/main.py` — gRPC server boilerplate (copy crawl4ai-service pattern)
- [ ] `catalog-service/config/settings.py`
- [ ] `catalog-service/service/providers/base_provider.py` + `provider_registry.py`
- [ ] SQL migration: `remote_data_sources`, partitioned `file_catalog`, partitioned
  `remote_scan_jobs` (with RLS policies)
- [ ] `backend/clients/catalog_service_client.py`
- [ ] `backend/api/catalog_api.py` — REST endpoints (connection CRUD + scan trigger)
- [ ] Add `catalog-service` to `docker-compose.yml`

### Phase 2 — Providers (Tier 1)

- [ ] `SMBProvider` using `smbprotocol` — pure Python, no system deps
- [ ] `S3Provider` using `s3fs` / `aiobotocore` — supports AWS + S3-compatible stores
- [ ] `NFSProvider` using `libnfs-python` — userspace NFS client (NFSv3)
- [ ] `SFTPProvider` using `asyncssh`
- [ ] Each provider implements `connection_config_schema()` and `credentials_schema()`
  for dynamic UI form generation

### Phase 3 — Analysis Pipeline

- [ ] `catalog-service/service/analyzer.py` — MIME detection, text extraction
- [ ] LLM summarization call (via orchestrator gRPC or a simple HTTP call to the
  backend's internal LLM endpoint)
- [ ] Vector-service integration for catalog entry embedding
- [ ] `catalog-service/service/catalog_service.py` — full scan orchestration with
  semaphore-bounded concurrency and change detection

### Phase 4 — Agent Tools

- [ ] gRPC tool handlers: `SearchFileCatalog`, `BrowseCatalogDirectory`,
  `GetFileMetadata`, `GetCatalogSummary`
- [ ] `llm-orchestrator/orchestrator/tools/file_catalog_tools.py` — thin gRPC wrappers
- [ ] Action I/O Registry entries with typed I/O contracts (tool-io-contracts rule)
- [ ] Add tools to appropriate tool packs / skill definitions

### Phase 5 — UI

- [ ] "Data Sources" section in Settings (mirrors External Connections UI)
- [ ] Dynamic form rendering from `provider-schemas` endpoint
- [ ] Connection test feedback (latency, sample count, error)
- [ ] Scan status progress display (live polling of scan job status)
- [ ] Basic catalog browser (directory tree, file metadata cards)

---

## Future Source Types (Tier 2+)

The provider registry makes adding new types a single-file addition. Planned:

| Source Type | Library | Notes |
|-------------|---------|-------|
| GCS | `gcsfs` | Service account JSON credentials |
| Azure Blob | `azure-storage-blob` | Account key or connection string |
| FTP | stdlib `ftplib` | No new deps |
| WebDAV | `httpx` (already in backend) | Bastion already has a WebDAV server |
| Backblaze B2 | `s3fs` (B2 is S3-compatible) | Same S3Provider, different endpoint_url |
| MinIO | `s3fs` | Same S3Provider, custom endpoint_url |

These require no schema changes — only a new provider class and a one-line registration.
