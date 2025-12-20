# Bastion Services & Containers Documentation

Complete reference for all services and containers in the Bastion workspace platform.

**Last Updated:** December 2025  
**Version:** 0.11.0

---

## Table of Contents

1. [Infrastructure Services](#infrastructure-services)
2. [Core Application Services](#core-application-services)
3. [Background Workers](#background-workers)
4. [gRPC Microservices](#grpc-microservices)
5. [Supporting Services](#supporting-services)
6. [Service Dependencies](#service-dependencies)
7. [Port Reference](#port-reference)

---

## Infrastructure Services

### PostgreSQL (Main Database)

**Container:** `postgres`  
**Image:** `postgres:15-alpine`  
**Port:** `5433:5432` (external:internal)

**Purpose:**
Primary PostgreSQL database for all application data including:
- User accounts and authentication
- Document metadata and organization
- Conversation history and LangGraph state persistence
- Messaging system (rooms, messages, presence)
- Team management and collaboration
- RSS feed subscriptions
- Org-mode file metadata
- Settings and configuration

**Key Features:**
- Optimized for high concurrency (200 max connections)
- Tuned for performance (shared buffers, work memory, parallel workers)
- Automatic initialization from SQL scripts in `backend/sql/`
- Health checks for dependency management

**Volumes:**
- `bastion_postgres_data` - Persistent database storage
- `./backend/sql` - Initialization scripts

**Configuration:**
- Database: `postgres` (initial), `bastion_knowledge_base` (application)
- User: `postgres` / `bastion_user`
- Password: `bastion_secure_password` (change in production!)

**Related Documentation:**
- Database schema: `backend/sql/01_init.sql`
- Migrations: `backend/sql/migrations/`

---

### PostgreSQL (Data Workspace Database)

**Container:** `postgres-data`  
**Image:** `postgres:15-alpine`  
**Port:** `5434:5432` (external:internal)

**Purpose:**
Dedicated PostgreSQL instance for Data Workspace functionality:
- User-created custom databases
- Imported data tables (CSV, JSON, Excel)
- Data transformation results
- Workspace metadata and permissions

**Key Features:**
- Isolated from main application database
- Optimized for data workspace workloads
- Automatic initialization from `data-service/sql/`

**Volumes:**
- `bastion_data_workspace_db` - Persistent workspace data storage
- `./data-service/sql` - Initialization scripts

**Configuration:**
- Database: `data_workspace`
- User: `data_user`
- Password: `data_workspace_secure_password`

**Dependencies:**
- Used exclusively by `data-service` microservice

---

### Redis

**Container:** `redis`  
**Image:** `redis:alpine`  
**Port:** `6380:6379` (external:internal)

**Purpose:**
- **Task Queue:** Celery broker and result backend
- **Caching:** Session data, temporary results
- **Pub/Sub:** Real-time notifications (future)

**Key Features:**
- Persistent storage with AOF (Append Only File)
- Used by Celery workers for background task processing
- Lightweight and fast in-memory operations

**Volumes:**
- `bastion_redis_data` - Persistent Redis data

**Configuration:**
- AOF persistence enabled
- Default Redis configuration

**Dependencies:**
- Required by: `backend`, `celery_worker`, `celery_beat`, `celery_flower`, `webdav`

---

## Core Application Services

### Backend API

**Container:** `backend`  
**Build:** `./backend/Dockerfile`  
**Ports:**
- `8081:8000` - FastAPI HTTP/REST API
- `50052:50052` - gRPC Tool Service

**Purpose:**
Main application backend providing:
- REST API endpoints for frontend
- WebSocket connections for real-time updates
- Document processing and management
- User authentication and authorization
- File upload and processing
- Integration with external services (Qdrant, Neo4j)
- gRPC Tool Service for orchestrator communication

**Key Features:**
- FastAPI framework with async support
- LangGraph integration (local orchestrator fallback)
- Document processing pipeline
- WebSocket notifications
- File management and storage
- Team collaboration features
- RSS feed management
- Org-mode file handling

**Environment Variables:**
- Database: `DATABASE_URL` (PostgreSQL connection)
- External Services: `QDRANT_URL`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
- Cache/Queue: `REDIS_URL`
- Search: `SEARXNG_URL`
- LLM APIs: `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`
- Vector Service: `USE_VECTOR_SERVICE=true`
- Data Service: `DATA_SERVICE_HOST`, `DATA_SERVICE_PORT`
- Feature Flags: `USE_GRPC_ORCHESTRATOR`, `GRPC_ORCHESTRATOR_PERCENTAGE`

**Volumes:**
- `./uploads` - User uploaded files
- `./processed` - Processed document outputs
- `./logs` - Application logs

**Dependencies:**
- `postgres` (health check)
- `redis` (started)
- `searxng` (health check)
- `data-service` (started)

**Related Documentation:**
- API endpoints: `backend/main.py`
- gRPC Tool Service: `backend/services/grpc_tool_service.py`
- See also: `docs/GRPC_MICROSERVICES_ARCHITECTURE.md`

---

### Frontend Web UI

**Container:** `frontend`  
**Build:** `./frontend/Dockerfile`  
**Port:** `3051:80` (Nginx serving React app)

**Purpose:**
React-based web interface providing:
- Document management and viewing
- Chat interface with AI agents
- File tree navigation
- Team collaboration features
- Settings and configuration
- Real-time updates via WebSocket

**Key Features:**
- React 18 with Material-UI components
- Real-time WebSocket connections
- Tabbed interface for multiple documents
- Markdown and Org-mode editors
- Responsive design

**Environment Variables:**
- `REACT_APP_APP_NAME` - Application name
- `REACT_APP_VERSION` - Version display

**Dependencies:**
- `backend` (API endpoints)
- `webdav` (for WebDAV proxy routing)

**Related Documentation:**
- Frontend source: `frontend/src/`
- Nginx configuration: `frontend/nginx.conf`

---

## Background Workers

### Celery Worker

**Container:** `celery_worker`  
**Build:** `./backend/Dockerfile` (same image as backend)  
**Command:** `celery -A services.celery_app worker --loglevel=info --queues=orchestrator,agents,rss,default`

**Purpose:**
Background task processor for:
- Long-running agent operations
- RSS feed polling and updates
- Document processing tasks
- Scheduled jobs

**Key Features:**
- Multiple queue support (orchestrator, agents, rss, default)
- Same codebase as backend (shared services)
- Automatic task retry on failure
- Task result tracking

**Environment Variables:**
- Same as backend (database, Redis, API keys)
- `CELERY_WORKER_TYPE=orchestrator` - Worker type identifier

**Volumes:**
- `./uploads` - Access to uploaded files
- `./processed` - Write processed outputs
- `./logs` - Worker logs

**Dependencies:**
- `postgres` (health check)
- `redis` (started)
- `searxng` (health check)

**Related Documentation:**
- Celery configuration: `backend/services/celery_app.py`
- Task definitions: `backend/services/celery_tasks/`

---

### Celery Beat

**Container:** `celery_beat`  
**Build:** `./backend/Dockerfile`  
**Command:** `celery -A services.celery_app beat --loglevel=info`

**Purpose:**
Task scheduler for periodic jobs:
- RSS feed polling schedules
- Periodic maintenance tasks
- Scheduled data imports
- Cleanup operations

**Key Features:**
- Cron-like scheduling
- Persistent schedule storage
- Automatic task dispatch to workers

**Environment Variables:**
- Database and Redis connections
- API keys for scheduled tasks

**Dependencies:**
- `postgres` (health check)
- `redis` (started)

**Related Documentation:**
- Beat schedule: `backend/services/celery_app.py`

---

### Celery Flower

**Container:** `celery_flower`  
**Build:** `./backend/Dockerfile`  
**Command:** `celery -A services.celery_app flower --port=5555`  
**Port:** `5555:5555`

**Purpose:**
Web-based monitoring UI for Celery:
- Task status and history
- Worker status and statistics
- Queue monitoring
- Task execution details

**Key Features:**
- Real-time task monitoring
- Worker management
- Task inspection and retry
- Performance metrics

**Access:**
- Web UI: `http://localhost:5555`

**Dependencies:**
- `redis` (broker connection)
- `celery_worker` (monitoring target)

---

## gRPC Microservices

### LLM Orchestrator

**Container:** `llm-orchestrator`  
**Build:** `./llm-orchestrator/Dockerfile`  
**Port:** `50051:50051` (gRPC)

**Purpose:**
Dedicated microservice for AI agent orchestration:
- Multi-agent system with 14+ specialized agents
- Intent classification and routing
- LangGraph workflow execution
- State persistence in PostgreSQL
- Natural language query processing

**Key Features:**
- **Specialized Agents:** Research, chat, coding, weather, RSS, entertainment, fiction editing, proofreading, image generation, messaging, and more
- **Intent-Based Routing:** Automatic agent selection based on query analysis
- **Human-in-the-Loop (HITL):** Permission-based workflows
- **PostgreSQL State Persistence:** Full conversation history
- **Structured Outputs:** Type-safe Pydantic models

**Environment Variables:**
- `GRPC_PORT=50051`
- Database: `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`
- Backend Tool Service: `BACKEND_TOOL_SERVICE_HOST`, `BACKEND_TOOL_SERVICE_PORT`
- LLM Configuration: `OPENROUTER_API_KEY`, `DEFAULT_MODEL`, `FAST_MODEL`, `CLASSIFICATION_MODEL`
- Feature Flags: `ENABLE_STREAMING`, `ENABLE_TOOL_CALLBACKS`, `MAX_CONCURRENT_REQUESTS`

**Dependencies:**
- `postgres` (health check) - For LangGraph state persistence
- `backend` (started) - For tool service access

**Related Documentation:**
- Architecture: `docs/GRPC_MICROSERVICES_ARCHITECTURE.md`
- Quick Start: `docs/GRPC_QUICK_START.md`
- Service Index: `docs/MICROSERVICES_INDEX.md`
- Orchestrator README: `llm-orchestrator/README.md`

---

### Vector Service

**Container:** `vector-service`  
**Build:** `./vector-service/Dockerfile`  
**Port:** `50053:50053` (gRPC)

**Purpose:**
Dedicated microservice for embedding generation:
- Generate text embeddings via OpenAI API
- Intelligent caching (3-hour TTL, content-hash based)
- Batch processing with parallel workers
- Text truncation and token counting

**Key Features:**
- **Pure Embedding Service:** Only generates embeddings, does NOT store in Qdrant
- **Intelligent Caching:** SHA256 content hashing, 3-hour TTL
- **Batch Processing:** Parallel embedding generation for multiple texts
- **Performance Optimized:** Configurable workers and batch sizes

**What It Does NOT Do:**
- Does NOT store vectors in Qdrant (caller's responsibility)
- Does NOT search Qdrant (caller's responsibility)
- Does NOT handle metadata (caller's responsibility)

**Environment Variables:**
- `GRPC_PORT=50053`
- `OPENAI_API_KEY` - Required for embeddings
- `OPENAI_EMBEDDING_MODEL=text-embedding-3-large`
- `PARALLEL_WORKERS=4` - Concurrent processing
- `BATCH_SIZE=100` - Batch processing size
- `MAX_TEXT_LENGTH=8000` - Text truncation limit
- `EMBEDDING_CACHE_ENABLED=true`
- `EMBEDDING_CACHE_TTL=10800` (3 hours)

**gRPC Methods:**
- `GenerateEmbedding` - Single embedding generation
- `GenerateBatchEmbeddings` - Batch embedding generation
- `ClearEmbeddingCache` - Cache management
- `GetCacheStats` - Cache statistics
- `HealthCheck` - Service health

**Related Documentation:**
- Service README: `vector-service/README.md`
- Proto definition: `protos/vector_service.proto`
- Hotfix notes: `docs/VECTOR_SERVICE_HOTFIX.md`

---

### Data Service

**Container:** `data-service`  
**Build:** `./data-service/Dockerfile`  
**Port:** `50054:50054` (gRPC)

**Purpose:**
Dedicated microservice for Data Workspace operations:
- Workspace management (create, list, update, delete)
- Custom database creation and management
- Table operations (CRUD)
- Data import (CSV, JSON, Excel)
- SQL query execution
- Natural language queries
- Data transformations (filters, aggregations, joins)
- Visualization generation

**Key Features:**
- **Isolated Database:** Uses dedicated `postgres-data` instance
- **File Import:** Automatic schema inference from CSV/JSON/Excel
- **Query Support:** SQL and natural language queries
- **Transformations:** Built-in data transformation operations
- **Styling:** Color-coded workspaces and databases

**Environment Variables:**
- `GRPC_PORT=50054`
- `POSTGRES_HOST=postgres-data`
- `POSTGRES_PORT=5432`
- `POSTGRES_DB=data_workspace`
- `POSTGRES_USER=data_user`
- `POSTGRES_PASSWORD=data_workspace_secure_password`
- `MAX_IMPORT_FILE_SIZE=524288000` (500MB)
- `IMPORT_BATCH_SIZE=1000`

**gRPC Methods:**
- Workspace operations: `CreateWorkspace`, `ListWorkspaces`, `GetWorkspace`, `UpdateWorkspace`, `DeleteWorkspace`
- Database operations: `CreateDatabase`, `ListDatabases`, `GetDatabase`, `DeleteDatabase`
- Table operations: `CreateTable`, `ListTables`, `GetTable`, `DeleteTable`, `GetTableData`, `InsertRow`, `UpdateRow`, `UpdateCell`, `DeleteRow`
- Import operations: `PreviewImport`, `ExecuteImport`, `GetImportStatus`
- Query operations: `ExecuteSQLQuery`, `ExecuteNaturalLanguageQuery`
- Transformation operations: `ApplyFilter`, `ApplyAggregation`, `ApplyJoin`
- Visualization operations: `GenerateVisualization`, `ListVisualizations`

**Volumes:**
- `./uploads` - Access to uploaded data files

**Dependencies:**
- `postgres-data` (health check) - Dedicated workspace database

**Related Documentation:**
- Proto definition: `data-service/grpc/data_service.proto`
- Service implementation: `data-service/grpc_service.py`

---

## Supporting Services

### SearXNG

**Container:** `searxng`  
**Image:** `searxng/searxng:latest`  
**Port:** `8889:8080` (external:internal)

**Purpose:**
Self-hosted metasearch engine for:
- Web search queries from agents
- Privacy-respecting search aggregation
- Multiple search engine results

**Key Features:**
- Aggregates results from multiple search engines
- Privacy-focused (no tracking)
- Configurable search engines
- Used by research agents for web searches

**Environment Variables:**
- `SEARXNG_SECRET` - Secret key for session management
- `SEARXNG_URL` - Public URL for the service

**Volumes:**
- `./searxng` - Configuration files

**Dependencies:**
- Used by: `backend`, `celery_worker`

**Access:**
- Web UI: `http://localhost:8889`

---

### WebDAV Server

**Container:** `webdav`  
**Build:** `./backend/Dockerfile.webdav`  
**Port:** `8002:8001` (external:internal)

**Purpose:**
WebDAV server for Org-mode file synchronization:
- Mobile sync for org-mode files (beorg, Orgzly)
- File access via WebDAV protocol
- User authentication via Bastion user database
- Direct filesystem access to `uploads/` directory

**Key Features:**
- **Simple Filesystem Access:** Serves files directly from `uploads/` directory
- **Actual Folder Structure:** Preserves real file/folder hierarchy
- **User Authentication:** Integrates with Bastion's user authentication
- **All File Types:** Works with any document type, not just .org files
- **Mobile Compatible:** Works with mobile apps that support WebDAV

**Environment Variables:**
- `DATABASE_URL` - PostgreSQL connection for authentication
- `REDIS_URL` - Redis connection
- `WEBDAV_HOST=0.0.0.0`
- `WEBDAV_PORT=8001`
- `JWT_SECRET_KEY` - Authentication secret

**Volumes:**
- `./uploads` - Served filesystem directory

**Dependencies:**
- `postgres` (health check) - User authentication
- `redis` (started)

**Access:**
- WebDAV URL: `https://your-domain/dav/` (via nginx reverse proxy)
- Direct: `http://localhost:8002` (development)

**Related Documentation:**
- WebDAV README: `backend/webdav/README.md`
- Sync Strategy: `docs/WEBDAV_SYNC_STRATEGY.md`
- Nginx configuration: `frontend/nginx.conf` (proxy setup)

---

## Service Dependencies

### Dependency Graph

```
frontend
  ├── backend
  └── webdav

backend
  ├── postgres (health check)
  ├── redis (started)
  ├── searxng (health check)
  └── data-service (started)

celery_worker
  ├── postgres (health check)
  ├── redis (started)
  └── searxng (health check)

celery_beat
  ├── postgres (health check)
  └── redis (started)

celery_flower
  ├── redis (started)
  └── celery_worker (monitoring)

webdav
  ├── postgres (health check)
  └── redis (started)

llm-orchestrator
  ├── postgres (health check)
  └── backend (started)

data-service
  └── postgres-data (health check)

vector-service
  └── (standalone, no dependencies)
```

### External Services (Not in Docker Compose)

These services are expected to be running externally (Kubernetes or self-hosted):

- **Qdrant** (`http://192.168.80.128:6333`) - Vector database for embeddings
- **Neo4j** (`bolt://192.168.80.132:7687`) - Knowledge graph database

---

## Port Reference

### Application Ports

| Service | Internal Port | External Port | Protocol | Purpose |
|---------|--------------|---------------|----------|---------|
| `backend` | 8000 | 8081 | HTTP | FastAPI REST API |
| `backend` | 50052 | 50052 | gRPC | Tool Service |
| `frontend` | 80 | 3051 | HTTP | React Web UI |
| `webdav` | 8001 | 8002 | HTTP | WebDAV Server |
| `celery_flower` | 5555 | 5555 | HTTP | Celery Monitoring UI |
| `searxng` | 8080 | 8889 | HTTP | Search Engine UI |

### Database Ports

| Service | Internal Port | External Port | Protocol | Purpose |
|---------|--------------|---------------|----------|---------|
| `postgres` | 5432 | 5433 | PostgreSQL | Main database |
| `postgres-data` | 5432 | 5434 | PostgreSQL | Data workspace database |
| `redis` | 6379 | 6380 | Redis | Cache and queue |

### gRPC Microservice Ports

| Service | Port | Protocol | Purpose |
|---------|------|----------|---------|
| `llm-orchestrator` | 50051 | gRPC | AI orchestration |
| `vector-service` | 50053 | gRPC | Embedding generation |
| `data-service` | 50054 | gRPC | Data workspace operations |

---

## Service Communication Patterns

### Frontend → Backend
- **HTTP/REST:** `http://backend:8000/api/*`
- **WebSocket:** `ws://backend:8000/api/ws/*`

### Backend → Microservices
- **gRPC:** `llm-orchestrator:50051`, `vector-service:50053`, `data-service:50054`

### Backend → Infrastructure
- **PostgreSQL:** `postgres:5432`
- **Redis:** `redis:6379`
- **SearXNG:** `http://searxng:8080`

### Frontend → WebDAV (via Nginx)
- **HTTP/WebDAV:** `/dav/*` → proxied to `webdav:8001`

---

## Health Checks

All services with health checks:

- `postgres` - `pg_isready -U postgres`
- `postgres-data` - `pg_isready -U data_user -d data_workspace`
- `searxng` - HTTP GET to `/`
- `llm-orchestrator` - Socket connection test on port 50051
- `vector-service` - Socket connection test on port 50053
- `data-service` - Socket connection test on port 50054

---

## Volumes

### Persistent Volumes

- `bastion_postgres_data` - Main PostgreSQL database
- `bastion_data_workspace_db` - Data workspace PostgreSQL database
- `bastion_redis_data` - Redis persistent data

### Bind Mounts

- `./uploads` - User uploaded files (shared across: backend, celery_worker, webdav, data-service)
- `./processed` - Processed document outputs (backend, celery_worker)
- `./logs` - Application logs (backend, celery_worker, celery_beat)
- `./backend/sql` - Database initialization scripts (postgres)
- `./data-service/sql` - Data workspace initialization scripts (postgres-data)
- `./searxng` - SearXNG configuration (searxng)

---

## Custom vs Third-Party Services

### Custom Services (Built by Bastion)

- `backend` - Main FastAPI application
- `frontend` - React web interface
- `llm-orchestrator` - AI orchestration microservice
- `vector-service` - Embedding generation microservice
- `data-service` - Data workspace microservice
- `webdav` - WebDAV server for file sync
- `celery_worker` - Background task processor
- `celery_beat` - Task scheduler
- `celery_flower` - Task monitoring UI

### Third-Party Services (Repackaged)

- `postgres` - PostgreSQL database (official image)
- `postgres-data` - PostgreSQL database (official image)
- `redis` - Redis cache/queue (official image)
- `searxng` - SearXNG search engine (official image)

---

## Development & Debugging

### View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f backend
docker compose logs -f llm-orchestrator
docker compose logs -f vector-service
docker compose logs -f data-service
```

### Access Containers

```bash
# Backend
docker compose exec backend bash

# LLM Orchestrator
docker compose exec llm-orchestrator bash

# Vector Service
docker compose exec vector-service bash

# Data Service
docker compose exec data-service bash
```

### Health Checks

```bash
# Check all services
docker compose ps

# Test gRPC services
grpcurl -plaintext localhost:50051 orchestrator.OrchestratorService/HealthCheck
grpcurl -plaintext localhost:50053 vector_service.VectorService/HealthCheck
grpcurl -plaintext localhost:50054 dataservice.DataService/HealthCheck
```

### Rebuild Services

```bash
# Rebuild specific service
docker compose build backend
docker compose build llm-orchestrator
docker compose build vector-service
docker compose build data-service

# Rebuild all
docker compose build
```

---

## Related Documentation

- **gRPC Architecture:** `docs/GRPC_MICROSERVICES_ARCHITECTURE.md`
- **Microservices Index:** `docs/MICROSERVICES_INDEX.md`
- **gRPC Quick Start:** `docs/GRPC_QUICK_START.md`
- **Vector Service:** `vector-service/README.md`
- **LLM Orchestrator:** `llm-orchestrator/README.md`
- **WebDAV Sync:** `docs/WEBDAV_SYNC_STRATEGY.md`
- **Main README:** `README.md`

---

**Note:** This documentation reflects the current state of services as of version 0.11.0. For the most up-to-date information, refer to `docker-compose.yml` and individual service documentation.

