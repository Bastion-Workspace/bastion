# Bastion Workspace

**WORK IN PROGRESS!**

*This application is still being heavily developed and tested*

Data may be *lost*, databases may become *outdated* and things generally are not expected to work 100% just now. Nevertheless, here is Bastion.


## Overview

**Bastion** is a self-hosted **AI workspace**: documents, structured data, chat, teams, and automation in one stack—driven by **natural language** and a **LangGraph** orchestrator with **PostgreSQL**-backed state. You run it on **Docker Compose**, wire the model providers you trust (OpenAI, OpenRouter, Anthropic, and local-friendly options where supported), and keep the corpus on **your** disks and databases.

**Who it is for:** researchers, writers, builders, and small teams who want **RAG**, **graphs**, **Org-mode**, **RSS**, **messaging**, and **custom agents** without gluing half a dozen SaaS products together.

---

## Platform snapshot

| Area | What you get |
|------|----------------|
| **Knowledge library** | Folders, tags, versions, uploads, semantic + keyword search, citations, live processing status |
| **Agent Factory (visual)** | **Browser-first** authoring: build **custom agents** and **multi-step playbooks** from simple linear flows to complex graphs—wire **skills**, **tool packs**, and **connectors** with the **graphical** composer (start without hand-editing stack config). |
| **Chat & automation** | Natural-language **chat** and **research**; invoke custom agents by **`@handle`**; **agent lines** for scheduled / heartbeat runs, continuity context, and delivery hooks. |
| **Skills (smart retrieval)** | Every **skill** is indexed for search (name, description, tools, example queries). With **`HYBRID_SEARCH_ENABLED`**, lookup uses **dense vectors + BM25** (RRF-style fusion) so playbook steps can **auto-discover** the right skills—**lightweight agents** stay small and still **pull new capability at runtime** when a step’s prompt matches the catalog. |
| **Data** | **Data workspaces** with SQL, imports, external DB links, charts, NL→query |
| **People** | **User messaging** (direct & group rooms, presence, reactions), **Teams** (roles, invitations, feeds, team chat, shared document trees) |
| **Productivity** | **Org-mode** files, todos / agenda / refile / archive flows, **RSS**, optional **Google Reader–style** API |
| **Reading** | **OPDS** catalogs, **EPUB** reader with typography and progress sync |
| **Media** | **Images** (including searchable sidecar metadata), **image generation**, **audio** ingest & transcription, **video** library handling |
| **Integrations** | **External connections** (OAuth, messaging providers), **WebDAV** for mobile org sync, optional **Bastion-to-Bastion federation** |
| **Ops** | Compose-first deployment, optional **BBS** (SSH/telnet) text UI, Celery **Flower**, health endpoints |

---

## Supported file & media formats

The library accepts many **document**, **media**, and **archive** types. **Vector chunking** applies to text-heavy pipelines (PDF, EPUB, Office, HTML, subtitles, etc.); images and video follow the image / media pipelines. For the exact enum and indexing eligibility, see `backend/models/api_models.py` (`DocumentType`) and `bastion_indexing/policy.py`.

| Category | Formats |
|----------|---------|
| **Office & long-form** | PDF, DOCX, **PPTX**, EPUB |
| **Text & markup** | TXT, **Markdown** (`.md`), **HTML**, **Org** (`.org`) |
| **Mail & captures** | **EML**, URL-ingested artifacts (where configured) |
| **Archives** | **ZIP** (expanded into child documents where applicable) |
| **Subtitles** | **SRT**, **VTT** |
| **Images** | Raster formats as **image** documents; companion **`*.metadata.json`** sidecars for searchable image metadata |
| **Audio** | MP3, AAC, WAV, FLAC, OGG, M4A, WMA, Opus |
| **Video** | MP4, MKV, AVI, MOV, WebM |

---

## Agent Factory

**Agent Factory** is how you turn Bastion from “a smart chat UI” into **your** automation surface: a **visual** composer where domain experts—not only developers—wire **tools**, **LLM steps**, **data connectors**, **skills**, and **human approvals** into reusable **agents** and **playbooks**. Each published agent gets a unique **`@handle`** (for example `@legal-brief`) so you invoke it deliberately from chat—no mystery routing.

### Building blocks

| Block | Role |
|--------|------|
| **Agent profile** | Name, model preferences, default tool packs, journal / sharing settings |
| **Playbook** | The graph: ordered or branching **steps**—each step is a **tool call** (deterministic), an **LLM task** (your instructions + optional structured output), an **approval gate**, or control flow you compose in the UI |
| **Skills** | Reusable mini-playbooks or prompt+tool bundles; **vector-indexed** (and optionally **BM25 + dense hybrid**) so steps can **auto-discover** relevant skills from natural language |
| **Connectors** | REST/OAuth **data sources** you define once; steps call connector endpoints with mapped fields |
| **Tool packs** | Curated access to Bastion capabilities—**documents**, **search**, **web/crawl**, **Org/todos**, **RSS**, **messaging**, **data workspace**, **M365 / GitHub / DevOps** (where configured), and more |
| **Outputs** | Stream to **chat**, save **artifacts** (reports, tables, drafts), write to **Data Workspace** tables, enrich **Neo4j**, etc. |
| **Agent lines** | Operational wrapper: **templates**, **schedules** (interval or cron + timezone), **heartbeats**, delivery—same playbooks, different run context |

### Prompting: what “good” looks like

You are not sprinkling magic strings—you are **contracting** each LLM step: role, inputs from prior steps (`{variable}` wiring), output shape (markdown vs JSON vs tool-friendly), and **when** to escalate to a human. Tool steps stay predictable; LLM steps handle judgment. **Skills** encode repeatable prompt+tool combos so playbooks stay short and you iterate in one place.

### Examples that show what is possible

1. **One-shot research brief** — User: `@climate-brief carbon tariffs EU 2025`. Playbook: **web search** → **crawl selected results** → **LLM synthesize** with “executive summary + controversies + citations” → emit a **Markdown artifact** you pin to the home dashboard. No code—only composed steps and your prompt text.

2. **Zero-LLM ETL** — `@fec-pull` runs **connector** (public campaign finance API) → **transform** (sort / filter) → **save_to_workspace** table. Deterministic, auditable, runs the same every time—ideal when you do not want model drift on numeric data.

3. **Fiction “room”** — `@chapter-polish` gets **read active editor** + **load referenced character sheet** → LLM step with *your* house style rubric → **structured manuscript edit** operations (or plain markdown patch) you apply in the editor. The “voice” is your prompt; the **machinery** is shared.

4. **Inbox triage with a spine** — **list unread RSS** or **fetch connector inbox** → LLM classifies “urgent / FYI / ignore” → **approval** step before **post to team feed** or **send channel message**. Humans stay in the loop only when value is highest.

5. **Standing competitive monitor** — **Agent line** on a cron: **crawl** a pricing or policy URL → **LLM** compares the new text to the **last run** (excerpt from the agent journal or a stored artifact) and decides if the change matters against criteria *you* wrote → **notify** via messaging when it does. Same playbook, **scheduled** context—wakes you when the model believes something moved.

6. **Lightweight “skills find me” agent** — A minimal playbook with **skill auto-discovery** on: the step prompt says “user asked about deadlines in org files”; hybrid **skill search** pulls in the **Org todos** skill pack automatically so the agent gains **refile / agenda** tools without you enumerating ten tool names.

7. **Cross-stack analyst** — Connector hits your **internal REST catalog** → results land in **Data Workspace** → second LLM step writes SQL against that table → chart artifact for stand-up. One profile ties external API + internal SQL + narrative.

These are **patterns**, not shipped products—you assemble what your organization needs. Deeper YAML patterns and field-level docs live in **[docs/AGENT_FACTORY.md](docs/AGENT_FACTORY.md)**, **[docs/AGENT_FACTORY_EXAMPLES.md](docs/AGENT_FACTORY_EXAMPLES.md)**, and **[docs/AGENT_FACTORY_TOOLS.md](docs/AGENT_FACTORY_TOOLS.md)**.

---

## Capabilities in depth

### Intelligence & automation

- **LangGraph orchestration** — Checkpointer-backed workflows, resumable threads, structured tool I/O.
- **Agent Factory** — Described above: visual agents, playbooks, skills, connectors, artifacts, and lines. Tool packs cover documents, web, messaging, data workspace, and integrations such as **M365 / GitHub / DevOps** where you enable them. (**Code-workspace tooling is not production-ready yet**—do not plan rollouts around it.)
- **Skill catalog & auto-discovery** — Skills live in a dedicated **Qdrant** collection with rich embedding text (category, required/optional tools, example queries). Set **`HYBRID_SEARCH_ENABLED=true`** (see `docker-compose.yml` / `.env`) so skill search adds a **BM25 sparse signal** fused with dense vectors: better lexical matches (exact tool names, jargon) alongside semantics. The orchestrator can **auto-inject** discovered skills into playbook steps so agents **grow capability on demand** without listing every skill up front.
- **Agent lines** — **Templates**, **heartbeat schedules** (interval or cron with timezone), continuity and delivery hooks for recurring autonomous runs.
- **Tool service (gRPC)** — Document, search, org, RSS, messaging, crawl, data workspace, and domain tools for agents—centralized and typed.
- **Human-in-the-loop** — Permission gates for sensitive tools (e.g. web, outbound messaging).
- **Web & research** — SearXNG metasearch, Crawl4AI microservice, crawl and ingestion paths to grow your library from the web.

### Library, search, and graph

- **Hybrid retrieval** — Dense vectors (**Qdrant**), optional **BM25** / lexical paths where enabled (`HYBRID_SEARCH_ENABLED`), reranking where configured—the same hybrid switch also upgrades **skill** search used by Agent Factory auto-discovery (see above).
- **Knowledge graph** — **Neo4j** entities and relationships for exploration and entertainment catalogs.
- **Unified search** — One mental model for “find it” across documents, tools, and agents; answers with **citations** and previews.
- **Operator-grade UI** — Folder tree, tabs, batch folder hydration for large libraries, WebSockets for pipeline status.

### Long-form writing & editing (you define the agent)

There is no fixed catalog of “fiction agent vs proofreading agent” in code anymore—you **build** those behaviors in **Agent Factory**: grant **document** and **editor** tools, add **LLM steps** with *your* style guide or rubric, branch on **approval** if you want human sign-off, and publish as `@my-line-editor`. The same blocks power **newsletters**, **policy redlines**, **lore bibles**, or **grant narrative** checks—whatever prompts and tool packs you choose.

- **Collaborative editing** — Real-time shared editing on supported documents (Yjs-backed collab when a room is active; per-document encryption disables collaboration—see in-app help).
- **Document versioning** for editable text types (e.g. Markdown, Org, plain text).
- **Org-mode** — Structure preserved, **WebDAV** for mobile clients, capture and agenda in the app—**and** automatable via Org/todo tools inside your custom agents.

### Data workspaces & SQL

- Isolated **postgres-data** plus a **data-service** gRPC API.
- **CSV / JSON / Excel** import with schema inference, styled tables, query history, NL and SQL interfaces.
- **External connections** to PostgreSQL, MySQL, SQLite for federated analysis (see Agent Factory connectors).

### Collaboration & comms

- **User messaging** — In-app **chat rooms** (direct messages and ad hoc groups), **presence**, **reactions**, unread counts, optional **encryption at rest**, file attachments; agents can participate where configured.
- **Teams** — Long-lived **teams** with **admin / member / viewer** roles, invitations via messaging, **team feeds** (posts, attachments, comments), **team chat rooms**, and **shared team libraries** (folders and documents), all enforced with **PostgreSQL RLS**.
- **Collaborative documents** — Invite collaborators into a **collab session** on supported text documents so multiple signed-in users edit the same file with live cursors and synced text (started from the document viewer when collaboration is available).
- **Optional federation** — Pair Bastion instances for cross-server rooms when enabled (`SITE_URL`, `SECRET_KEY`).

### Feeds, news, and passive intake

- **RSS/Atom** subscriptions, Celery-driven polling, read/unread, import into the library, NL queries over feeds.
- **GReader-compatible** HTTP API for classic mobile readers (toggle via env).

### Media, voice, and reading room

- **Voice service** — STT (e.g. Whisper) and TTS providers (e.g. ElevenLabs, OpenAI, Piper) behind gRPC.
- **Image pipeline** — Upload, optional **vision** microservice for richer media understanding, **image generation** from chat and agents.
- **EPUB / OPDS** — Catalogs in settings, in-app reader, optional progress sync (KoReader-style workflows).

### Maps, routing, and location-aware workflows

- **PMTiles** basemap support and legacy tile-server URLs where configured.
- **Valhalla** (default) or **OSRM**-style routing via compose for maps and location-aware agents.

### Operator-facing extras

- **Celery** workers (orchestrator, agents, RSS, reindex queues), **Beat**, **Flower** UI.
- **Optional BBS** — SSH/telnet text UI for menus, chat, RSS, org surfaces, and ASCII / wallpaper modes.
- **Home dashboard** — Widgets, embeds, scratch-pad style blocks.

### Security posture (summary)

- **JWT** sessions, bcrypt password storage, **PostgreSQL RLS** on sensitive tables, path sanitization, parameterized SQL.
- **Secrets in environment**, not in the repo—see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for production hardening.

---

## Technical Architecture

### Backend Stack
- **Framework**: FastAPI (Python)
- **Orchestration**: LangGraph with PostgreSQL state persistence
- **Database**: PostgreSQL (metadata, state, messaging, org-mode, RSS)
- **Data Workspaces**: Dedicated PostgreSQL instance (postgres-data) with gRPC microservice
- **Vector Store**: Qdrant (external Kubernetes deployment)
- **Knowledge Graph**: Neo4j (external Kubernetes deployment)
- **Cache & Queue**: Redis (task queue, caching)
- **Search Engine**: SearXNG (self-hosted metasearch)
- **Task Queue**: Celery with Celery Beat for scheduling
- **File Storage**: Local filesystem
- **WebDAV Server**: For org-mode mobile sync

### Frontend Stack
- **Framework**: React 18
- **UI Library**: Material-UI (MUI)
- **State Management**: React Context + React Query
- **Real-time**: WebSocket connections for live updates
- **Styling**: Emotion (CSS-in-JS)

### AI & LLM Integration
- **LLM Providers**: OpenRouter (using OpenAI API)
- **Embeddings**: OpenAI text-embedding-3-large vectorization
- **Image Generation**: OpenRouter-supported image generation models
- **Speech**: OpenAI Whisper (transcription), TTS (future)
- **Intent Classification**: Select fast models for routing
- **Agent Execution**: Configurable model per agent

### Infrastructure
- **Deployment**: Docker Compose for application layer
- **Containerization**: Multi-stage builds for backend, frontend, and data-service
- **Microservices**: Dedicated data-service container with gRPC communication (port 50054)
- **External Services**: Qdrant and Neo4j
- **Networking**: Bridge network for inter-service communication
- **Volumes**: Persistent storage for uploads, processed files, operational database, and data workspaces
- **Operations**: Full install / env / GHCR / security guidance in **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**

## Quick Start

### Prerequisites
- Docker and Docker Compose
- OpenAI API key (required for embeddings and chat)
- **Optional**: OpenRouter, Anthropic, OpenWeatherMap API keys
- **External Infrastructure** (Kubernetes-hosted or self-hosted):
  - Qdrant vector database endpoint
  - Neo4j knowledge graph endpoint

### 1. Clone and Setup
```bash
git clone <repository-url>
cd bastion
cp .env.example .env
```

### 2. Configure Environment
Copy [`.env.example`](.env.example) to `.env` and set at least the variables you need for your stack. For **profiles**, **Postgres/password models**, **Qdrant/Neo4j topologies**, and **registry-based deploys**, use **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**.

Minimal example — keys and external services (your URLs will differ):

```env
# Required: OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Additional LLM Providers
OPENROUTER_API_KEY=your_openrouter_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Optional: Services
OPENWEATHERMAP_API_KEY=your_openweathermap_key_here

# External Infrastructure (Kubernetes-hosted or local)
QDRANT_URL=http://your-qdrant-endpoint:6333
NEO4J_URI=bolt://your-neo4j-endpoint:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password
```

### 3. Launch the Application
```bash
docker compose up --build
```

That's it! The application will automatically:
- Start PostgreSQL (operational database), postgres-data (data workspaces), Redis, and SearXNG services
- Start the data-service microservice (gRPC on port 50054)
- Start Celery workers and beat scheduler
- Initialize the database schemas
- Start the backend API (internal port 8000)
- Start the WebDAV server (internal port 8001)
- Start the frontend with NGINX (port 3051) - routes all traffic
- Start Celery Flower monitoring (port 5555)
- Connect to your external Qdrant and Neo4j instances

### 4. Access the Application
**Primary Access (via NGINX):**
- **Main Application**: http://localhost:3051
  - Frontend UI, API endpoints (`/api/*`), and WebDAV (`/dav/*`) all routed through NGINX
  - API Documentation: http://localhost:3051/api/docs
  - Health Check: http://localhost:3051/health

**Optional Direct Access (for debugging/development):**
- **Backend API (direct)**: http://localhost:8081
- **WebDAV (direct)**: http://localhost:8002
- **Celery Flower**: http://localhost:5555

### 5. Create Admin User
On first startup, an admin user is automatically created using environment variables:
- **Username**: Set via `ADMIN_USERNAME` env var (default: admin)
- **Password**: Set via `ADMIN_PASSWORD` env var

**IMPORTANT**: 
- Set strong credentials via environment variables before first startup
- Change the admin password immediately after first login!

## Usage Guide

### Natural Language Interactions
The power of Bastion lies in its natural language interface. Examples:

**Research Queries:**
- "What are the main themes in my uploaded documents about climate change?"
- "How are penguins related to sea lions?"

**Agent Commands:**
- "What's the weather in Tokyo?"
- "Generate an image of a sunset over mountains"
- "Analyze the character development in Chapter 5 of my novel"

**Org-Mode Management:**
- "Add to my inbox: Buy groceries tomorrow"
- "Create a project for Q1 planning"
- "What are my active TODO items?"

**Content Creation:**
- "Proofread this paragraph for clarity and grammar"
- "Generate a podcast script about AI safety"
- "Edit this fiction scene for better pacing"

### Document Management
1. **Upload Documents**: Drag files onto folders or use upload dialog
2. **Organize**: Create folders, move documents, apply tags and categories
3. **Search**: Use semantic search across all documents
4. **Query**: Ask natural language questions about your documents
5. **Collaborate** (supported docs): Start or join a **collaborative editing** session so teammates edit the same document in real time (not available for per-file encrypted documents)

### Teams
1. **Create or join a team** from the Teams area in the sidebar (roles: admin, member, viewer)
2. **Invite members** using the team invitation flow (often via messaging)
3. **Use the team feed** for announcements, files, and comments; open **team chat** for ongoing discussion
4. **Share libraries**: Add or move documents and folders into the team workspace so members see shared **document trees** with RLS-backed access

### Data Workspaces
1. **Create Workspace**: Click "Data Workspaces" in sidebar → Create New
2. **Import Data**: Upload CSV, JSON, or Excel files with automatic schema detection
3. **Create Tables**: Define custom tables with column types and styling
4. **Connect External DBs**: Link PostgreSQL, MySQL, or SQLite databases
5. **Query Data**: Use SQL or natural language queries to analyze data
6. **Visualize**: Create charts and visualizations from your data

### RSS Feeds
1. **Add Feed**: Right-click "RSS Feeds" → Add RSS Feed
2. **Browse Articles**: Click on a feed to see articles
3. **Import**: Click "Import" to add article to knowledge base
4. **Agent Queries**: "Summarize latest articles from TechCrunch feed"

### Messaging
1. **Open Drawer**: Click the floating mail icon (bottom-right)
2. **Create Room**: Click "+ New Conversation" for a **direct** or **group** chat room (separate from team feed posts—use Teams for long-lived orgs)
3. **Send Messages**: Real-time messaging with **presence** indicators
4. **Agent Integration**: AI can send messages to other users when tools and permissions allow

## Configuration

### Environment Variables
**Full deployment and env grouping:** [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md). The snippet below highlights common keys; **`docker-compose.yml`** defines every injected variable for the default stack.

Key configuration options (see `docker-compose.yml` for the authoritative list):

```yaml
# Core
DATABASE_URL=postgresql://bastion_user:bastion_secure_password@postgres:5432/bastion_knowledge_base
REDIS_URL=redis://redis:6379
SEARXNG_URL=http://searxng:8080

# External Services
QDRANT_URL=http://192.168.80.128:6333
NEO4J_URI=bolt://192.168.80.132:7687

# API Keys
OPENAI_API_KEY=${OPENAI_API_KEY}
OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
OPENWEATHERMAP_API_KEY=${OPENWEATHERMAP_API_KEY}

# Features
MESSAGING_ENABLED=true
MESSAGE_ENCRYPTION_AT_REST=false
UPLOAD_MAX_SIZE=1500MB
HYBRID_SEARCH_ENABLED=false

# Embeddings (see "Embedding dimensions" below)
EMBEDDING_DIMENSIONS=3072

# Authentication
JWT_SECRET_KEY="your-secret-key"
ADMIN_USERNAME=admin
ADMIN_PASSWORD="your-secure-password-here"
DEVELOPMENT_BYPASS_AUTH=false
```

### Embedding dimensions (`EMBEDDING_DIMENSIONS`)

Use **one** value in `.env` for dense vector size. Docker Compose passes `EMBEDDING_DIMENSIONS` into **backend**, **celery_worker**, **vector-service**, and **tools-service** (tools-service runs early startup skill sync via the same `skill_vector_service` code; without this variable it would keep the default 3072 and skip recreating the `skills` collection when you use a smaller embedding size).

- **`text-embedding-3-*` (OpenAI-compatible)**: The vector service sends the API `dimensions` parameter when the model id contains `text-embedding-3` (e.g. OpenAI or OpenRouter). Use a size the provider allows for that model (see provider docs). Default in compose is **3072**; lower values reduce storage and can speed search with small quality tradeoffs (Matryoshka-style truncation).

- **Ollama, VLLM, or other models**: Output length is fixed by the model. Set `EMBEDDING_DIMENSIONS` to **match that native length**. The service does not truncate arbitrary models to an arbitrary dimension; a mismatch will break Qdrant upserts or search.

- **Changing dimensions** after data exists: On startup the stack detects the change and migrates collections / re-embeds where configured (see `CHANGELOG` for BM25 / embedding migration behavior). Expect re-embedding cost and time.

### Model Selection
Configure which models agents use:
- **Intent Classification**: Uses `FAST_MODEL` (typically Claude Haiku)
- **Agent Execution**: Uses `DEFAULT_MODEL` (configurable per agent)
- **Embeddings**: Configured on the **vector service** (default `text-embedding-3-large` via `OPENAI_EMBEDDING_MODEL` / provider settings in `docker-compose.yml`), not the chat model dropdown

Models are configurable in the chat sidebar dropdown.

## Project Structure

```
/opt/bastion/
├── backend/
│   ├── api/                    # FastAPI route handlers
│   ├── models/                 # Pydantic models
│   ├── services/               # Business logic
│   │   ├── langgraph_agents/  # Agent implementations
│   │   ├── langgraph_tools/   # Tool implementations
│   │   └── messaging/         # Messaging system
│   ├── repositories/          # Data access layer
│   ├── sql/                   # Database migrations
│   ├── utils/                 # Utility functions
│   └── webdav/                # WebDAV server
├── data-service/              # Data workspace microservice
│   ├── db/                    # Database connection management
│   ├── services/              # Workspace, database, table services
│   ├── sql/                   # Data workspace schema
│   └── protos/                # gRPC protocol definitions
├── frontend/
│   ├── src/
│   │   ├── components/        # React components
│   │   │   └── data_workspace/ # Data workspace UI
│   │   ├── contexts/          # React contexts
│   │   └── services/          # API clients
│   └── public/                # Static assets
├── docs/                      # Documentation
│   ├── agent_plans/          # Future agent concepts
│   ├── fiction_agent/        # Fiction editing docs
│   ├── org_mode/             # Org-mode integration
│   ├── entertainment/        # Entertainment features
│   ├── future_plans/         # Roadmap documents
│   └── historical_summaries/ # Implementation history
├── uploads/                   # User-uploaded files
├── processed/                 # Processed documents
├── logs/                      # Application logs
└── docker-compose.yml        # Service orchestration
```
## Troubleshooting

### Common Issues

**Documents not processing:**
- Check API keys in `.env`
- Verify Qdrant and Neo4j connections
- Check logs: `docker compose logs backend`

**Agents not responding:**
- Verify LLM API keys are set
- Check model selection in chat sidebar
- Review Celery worker logs: `docker compose logs celery_worker`

**WebSocket connection issues:**
- Ensure backend is running on port 8081
- Check browser console for connection errors
- Verify Redis is running

**Database connection errors:**
- Wait for PostgreSQL to initialize (usually 10-15 seconds)
- Check database logs: `docker compose logs postgres`
- Verify DATABASE_URL is correct

### Performance Optimization

**Slow queries:**
- Increase Qdrant resources (external deployment)
- Optimize chunk size for embedding
- Use category/tag filtering for targeted searches

**High memory usage:**
- Limit concurrent document processing
- Adjust Celery worker concurrency
- Monitor with Flower: http://localhost:5555

**Long processing times:**
- Check internet connection for LLM API calls
- Verify Celery workers are running
- Review quality threshold settings

## Security Considerations

- **Authentication**: JWT-based authentication with secure secret key
- **API Keys**: Stored in environment variables, never in code
- **Message Encryption**: Optional at-rest encryption with master key
- **Input Validation**: All inputs validated with Pydantic models
- **SQL Injection**: Protected via parameterized queries
- **CORS**: Configured for secure cross-origin requests
- **WebDAV**: Basic auth for mobile sync

### **Database-Level Security**
- **Password Hashing**: All passwords hashed with bcrypt (cost factor 10+)
- **Row-Level Security (RLS)**: Comprehensive PostgreSQL RLS implementation enforcing data isolation at the database level
  - **Document Access**: Users see only their documents, team documents, and global documents
  - **Team Data**: Team members can only access their team's data
  - **Chat & Messaging**: Users can only access rooms they're participants in
  - **User Sessions**: Session isolation with authentication lookup policies
  - **Music Service**: User-specific music configuration and cache isolation
  - **Automatic Enforcement**: Database-level policies prevent unauthorized access even if application logic fails
  - **RLS Context**: All database operations set `app.current_user_id` and `app.current_user_role` session variables
  - **Admin Bypass**: System admins can access user documents for support, but not team documents (privacy-preserving)
- **Path Traversal Protection**: Filename sanitization prevents directory traversal attacks
- **Authorization Checks**: All document endpoints verify user access before operations
- **Password Requirements**: Minimum 8 characters enforced at API level (user creation and password changes)
- **Parameterized Queries**: All database queries use parameterized statements to prevent SQL injection

## Monitoring & Observability

- **Celery Flower**: Real-time task monitoring at http://localhost:5555
- **Structured Logging**: JSON logs with context and timestamps
- **WebSocket Events**: Real-time processing status updates
- **Health Checks**: `/health` endpoint for service status
- **Error Tracking**: Comprehensive error logging with stack traces

## Architecture Deep Dive

### Row-Level Security (RLS) Implementation

Bastion implements comprehensive PostgreSQL Row-Level Security (RLS) policies to enforce data isolation at the database level. This provides defense-in-depth security, ensuring users can only access their own data even if application-level checks fail.

#### RLS-Protected Tables

- **`document_metadata`**: Users see only their documents, team documents, and global documents
- **`document_folders`**: Folder access follows document access rules
- **`teams`** & **`team_members`**: Team data isolated to team members
- **`chat_rooms`** & **`room_participants`**: Users can only access rooms they're in
- **`user_sessions`**: Session isolation with special authentication lookup policy
- **`music_service_configs`** & **`music_cache`**: User-specific music data isolation
- **`checkpoints`** & **`checkpoint_blobs`**: LangGraph state isolation by user

#### RLS Context Setting

All database operations set PostgreSQL session variables:
```sql
SET app.current_user_id = 'user_id';
SET app.current_user_role = 'user' | 'admin';
```

This is handled automatically by:
- `DatabaseContextManager` for connection pool operations
- `DatabaseManager.execute_query()` with `rls_context` parameter
- `DocumentRepository.create_with_folder()` for document creation
- All agent tools that create/access user data

#### RLS Policy Pattern

Policies follow this pattern:
```sql
CREATE POLICY policy_name ON table_name
    FOR SELECT USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR collection_type = 'global'
        OR (team_id IN (SELECT team_id FROM team_members WHERE user_id = ...))
        OR current_setting('app.current_user_role', true) = 'admin'
    );
```

See `backend/postgres_init/01_init.sql` for complete RLS policy definitions.

### Orchestrator architecture

Chat and automation requests are handled by the **llm-orchestrator** service: LangGraph workflows compiled with a PostgreSQL checkpointer, unified dispatch, and playbook-driven steps. See [docs/AGENT_FACTORY.md](docs/AGENT_FACTORY.md) and [docs/dev-notes/AGENT_FACTORY_ARCHITECTURE.md](docs/dev-notes/AGENT_FACTORY_ARCHITECTURE.md) for current behavior.

## Future plans

Roadmap items are tracked in [CHANGELOG.md](CHANGELOG.md) and design notes under [docs/dev-notes/](docs/dev-notes/). Examples: voice conversation mode, richer PIM/calendar integration, and expanded tooling.

## Contributing

This is a personal knowledge base system. For questions or suggestions:
1. Review existing documentation in `/docs/`
2. See [docs/AGENT_FACTORY.md](docs/AGENT_FACTORY.md) for how agents and tools are wired today

## License

This project is for personal/internal use. Commercial use is permissible, but don't blame me if you lose all your data. See LICENSE file for details.

---

**Architecture Principles:**
- **Docker-first**: Everything runs via `docker compose up --build`
- **Modular Design**: Files limited to 500 lines, clear separation of concerns
- **Structured Outputs**: Pydantic models for type-safe communication
- **LangGraph Native**: Official patterns for agent orchestration
- **PostgreSQL Persistence**: Full conversation and state management
- **External Infrastructure**: Qdrant and Neo4j on Kubernetes for scalability

---

*A sophisticated multi-agent knowledge management system - making information access as natural as conversation!*