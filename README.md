# Bastion Workspace

**WORK IN PROGRESS!**

*This application is still being heavily developed and tested*

Data may be *lost*, databases may become *outdated* and things generally are not expected to work 100% just now. I'm not even fully confident in the intent classification system to always take you in the right direction, which bother me. Nevertheless, here is Bastion.


## Overview
A comprehensive personal workspace platform featuring specialized agents for knowledge management, creative writing, data engineering, collaboration, and productivity - all controlled through natural language.


Bastion is a multi-purpose workspace that combines:
- **Knowledge Management**: RAG-powered document search with vector similarity and knowledge graphs
- **Creative Tools**: Fiction editing, prose analysis, character development, and content generation
- **Data Workspaces**: Custom database management with CSV/Excel import and external database connections
- **Collaboration**: Real-time messaging, presence tracking, and team coordination
- **Productivity**: Org-mode integration, RSS feeds, task management, and WebDAV sync
- **Orchestration**: LangGraph-powered multi-agent system with intelligent routing and state persistence

Think of it as your personal command center - whether you're researching documents, writing fiction, managing custom databases, or coordinating with your team, Bastion's specialized agents handle it through simple conversational interfaces.

## Core Features

### **LangGraph Agent System**
- **Specialized Agents**: Research, chat, coding, weather, RSS, entertainment, fiction editing, proofreading, image generation, messaging, and more
- **Subgraph Architecture**: Modular, reusable subgraphs for complex workflows (context preparation, validation, generation, research, gap analysis, and more)
- **Intelligent Routing**: Automatic intent classification routes queries to the appropriate agent
- **Human-in-the-Loop (HITL)**: Permission-based workflows for sensitive operations
- **PostgreSQL State Persistence**: Full conversation history and state management with checkpointing
- **Structured Outputs**: Type-safe Pydantic models for all agent communications

### **Document Processing & Management**
- **Multi-Format Support**: PDF, DOCX, EPUB, TXT, HTML, MD, ORG, EML, SRT, ZIP archives
- **Hierarchical Organization**: Folder-based document management with drag & drop
- **Smart Processing**: Quality assessment, intelligent chunking, OCR support
- **Real-time Updates**: WebSocket notifications for processing status
- **Metadata Management**: Tags, categories, custom metadata per document
- **File Tree Sidebar**: Visual folder hierarchy with context menus

### **Advanced Search & Retrieval**
- **Vector Similarity Search**: Semantic search across all documents via Qdrant
- **Knowledge Graph**: Entity extraction and relationship mapping via Neo4j
- **Multi-Source Search**: Documents, videos, entities, web content
- **Unified Search Tools**: Single interface for all search types
- **Citation Support**: Answers include source references and relevance scores

### **Org-Mode Integration**
- **Native .org File Support**: Store org-mode files with structure preservation
- **WebDAV Sync**: Mobile sync via WebDAV server (compatible with beorg, Orgzly)
- **Org-Inbox Agent**: Natural language capture to org-mode
- **Non-Vectorized Storage**: Org files kept structured, not embedded

### **Messaging & Collaboration**
- **User-to-User Messaging**: Real-time chat rooms (direct & group)
- **Presence Indicators**: Online/offline/away status tracking
- **Message Reactions**: Emoji reactions on messages
- **Unread Tracking**: Per-room unread message counts

### **Team-Based Collaboration**
- **Team Management**: Create teams with roles (admin, member, viewer)
- **Team Invitations**: Invite users via messaging system with invitation messages
- **Team Feeds**: Social posting with text, images, and file attachments
- **Post Interactions**: Reactions and comments on team posts
- **Team Documents**: Share documents and folders with teams
- **Team Chat Rooms**: Dedicated chat rooms for team communication
- **Access Control**: Row-Level Security (RLS) policies ensure team data isolation

### **RSS Feed Management**
- **Feed Subscriptions**: Add and manage RSS/Atom feeds
- **Background Polling**: Automatic feed updates via Celery tasks
- **Article Import**: Import feed articles into knowledge base
- **Unread Tracking**: Track which articles have been read
- **RSS Agent**: Natural language feed queries and article summaries

### **Entertainment Knowledge Base**
- **Movies & TV Shows**: Track and query entertainment content
- **NEO4J Integration**: Rich relationship graphs for actors, directors, franchises
- **Entertainment Agent**: Natural language queries about movies and shows

### **Creative Writing & Editing**
- **Fiction Editing Agent**: Prose editing with style preservation
- **Character Development Agent**: Character arc analysis and suggestions
- **Outline Editing Agent**: Story structure and outline management
- **Story Analysis Agent**: Narrative structure analysis
- **Proofreading Agent**: Grammar, style, and clarity improvements

### **Data Workspaces**
- **Dedicated Data Platform**: Isolated PostgreSQL database for user data workspaces
- **Custom Databases**: Create and manage custom databases within workspaces
- **Data Import**: CSV, JSON, Excel file import with automatic schema inference
- **Visual Table Management**: Create, edit, and query custom tables
- **Styling Support**: Color-coded workspaces and databases for organization
- **External Connections**: Connect to external databases (PostgreSQL, MySQL, SQLite)
- **Data Transformations**: Built-in transformation operations on imported data
- **Query History**: Track and reuse LLM-powered data queries
- **Microservice Architecture**: Dedicated gRPC-based data service for performance

### **Weather & Location**
- **Weather Agent**: Current weather and forecasts via OpenWeatherMap
- **Location-Aware**: Query weather for any location
- **Natural Language**: "What's the weather in San Francisco?"

### **Image Generation**
- **Image Generation Agent**: Create images via OpenRouter-supported image generation models
- **Natural Language Prompts**: Describe what you want to create
- **Multiple Styles**: Support for different artistic styles and parameters

### **Specialized Agents**
- **Data Formatting Agent**: Format data into tables, charts, and structured outputs
- **Content Analysis Agent**: Deep content analysis and summarization
- **Fact Checking Agent**: Verify claims and check facts
- **Site Crawl Agent**: Extract content from websites for research
- **Website Crawler Agent**: Ingest entire websites into knowledge base
- **Podcast Script Agent**: Generate podcast scripts
- **Substack Agent**: Manage Substack content
- **Rules Editing Agent**: Game rules and documentation editing

### **Audio Processing**
- **Audio Transcription**: Upload audio files for transcription via OpenAI Whisper
- **Multiple Formats**: Support for MP3, WAV, M4A, and more

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
- **Embeddings**: OpenAI text-embedding-3-large vectorization)
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
Edit `.env` file with your API keys and service endpoints:

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
2. **Create Room**: Click "+ New Conversation"
3. **Send Messages**: Real-time messaging with presence indicators
4. **Agent Integration**: AI can send messages to other users

## Configuration

### Environment Variables
Key configuration options (see `docker-compose.yml` for full list):

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

# Authentication
JWT_SECRET_KEY="your-secret-key"
ADMIN_USERNAME=admin
ADMIN_PASSWORD="your-secure-password-here"
DEVELOPMENT_BYPASS_AUTH=false
```

### Model Selection
Configure which models agents use:
- **Intent Classification**: Uses `FAST_MODEL` (typically Claude Haiku)
- **Agent Execution**: Uses `DEFAULT_MODEL` (configurable per agent)
- **Embeddings**: OpenAI text-embedding-3-small (for vectorization)

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

See `backend/sql/01_init.sql` for complete RLS policy definitions.

### Subgraph-Capable Agent Architecture

Bastion's agents use LangGraph subgraphs for modular, reusable workflow components. This enables complex multi-step operations while maintaining clean separation of concerns.

#### Available Subgraphs

**Fiction Editing Subgraphs:**
- `build_context_preparation_subgraph`: Chapter detection, reference loading, scope analysis
- `build_validation_subgraph`: Continuity tracking, outline sync, consistency checks
- `build_generation_subgraph`: Context assembly, prompt building, LLM calls, output validation
- `build_resolution_subgraph`: Operation resolution with progressive search and validation
- `build_book_generation_subgraph`: Full automated book generation workflow

**Research & Knowledge Subgraphs:**
- `build_research_workflow_subgraph`: Complete research pipeline (local + web search, gap analysis, synthesis)
- `build_web_research_subgraph`: Web search + crawling workflow
- `build_gap_analysis_subgraph`: Universal gap analysis for any domain
- `build_fact_verification_subgraph`: Fact verification workflow
- `build_knowledge_document_subgraph`: Knowledge document creation workflow
- `build_intelligent_document_retrieval_subgraph`: Smart document retrieval with multiple strategies
- `build_full_document_analysis_subgraph`: Parallel document analysis with multiple queries

**Utility Subgraphs:**
- `build_assessment_subgraph`: Assessment and evaluation workflows

#### Subgraph Usage Pattern

Agents integrate subgraphs as nodes in their main workflow:

```python
def _build_workflow(self, checkpointer) -> StateGraph:
    workflow = StateGraph(AgentState)
    
    # Build subgraphs
    research_sg = build_research_workflow_subgraph(checkpointer)
    validation_sg = build_validation_subgraph(checkpointer, ...)
    
    # Add subgraphs as nodes
    workflow.add_node("research", research_sg)
    workflow.add_node("validate", validation_sg)
    
    # Connect in workflow
    workflow.add_edge("research", "validate")
    
    return workflow.compile(checkpointer=checkpointer)
```

#### Subgraph Benefits

- **Modularity**: Reusable components across multiple agents
- **State Preservation**: Subgraphs maintain critical state (user_id, metadata, shared_memory)
- **Checkpointing**: Full state persistence through LangGraph checkpoints
- **Testability**: Subgraphs can be tested independently
- **Complexity Management**: Break complex workflows into manageable pieces

#### State Management in Subgraphs

All subgraph nodes preserve critical state keys:
- `metadata`: User model preferences and configuration
- `user_id`: Database access context
- `shared_memory`: Cross-agent and cross-turn context
- `messages`: Conversation history
- `query`: Original user request

This ensures subgraphs work correctly with user preferences, RLS permissions, and conversation continuity.

See project documentation in `/docs/` for subgraph development guidelines.

## Future Plans

See `/docs/future_plans/` for detailed roadmaps:
- Voice conversation mode (speech-to-text, TTS, wake word)
- Personal Information Manager (calendar, contacts)
- Enhanced financial analysis agent
- Network security monitoring

## Contributing

This is a personal knowledge base system. For questions or suggestions:
1. Review existing documentation in `/docs/`
3. Consult the agent integration guide for adding new agents

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