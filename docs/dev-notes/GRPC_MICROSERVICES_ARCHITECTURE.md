# gRPC Microservices Architecture - Developer Guide

## Overview

This document describes our gRPC-based microservices architecture for LLM orchestration. This pattern enables independent scaling, upgrading, and maintenance of AI components while maintaining clean separation from the main application.

**Architecture Version:** Multi-Agent Phase (November 2025)
- **14 specialized agents** in production
- **Shared proto architecture** for zero-drift consistency
- **Intent-based routing** for intelligent agent selection
- **gRPC microservices** for backend/orchestrator communication

---

## Architecture Pattern

### Service Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND (React)                         │
│                      HTTP/REST/WebSocket                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BACKEND (FastAPI)                             │
│  ┌───────────────────────────────────────────────────────┐      │
│  │  Feature Flag Router (utils/feature_flags.py)         │      │
│  │  - Controls traffic routing                           │      │
│  │  - Gradual rollout support (0-100%)                   │      │
│  │  - Fallback to local on error                         │      │
│  └──────────────┬────────────────────────────────────────┘      │
│                 │                                                │
│       ┌─────────┴─────────┐                                     │
│       │                   │                                     │
│       ▼                   ▼                                     │
│  ┌─────────┐      ┌──────────────────┐                         │
│  │ Local   │      │ gRPC Proxy       │                         │
│  │ LangGraph│      │ (grpc_orchestrator│                        │
│  │ Orchestrator│   │  _proxy.py)      │                         │
│  └─────────┘      └────────┬─────────┘                         │
│                             │                                    │
│                             │ gRPC Client                        │
│                             │ (orchestrator.proto)               │
└─────────────────────────────┼────────────────────────────────────┘
                              │
                              │ gRPC (Port 50051)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              LLM ORCHESTRATOR MICROSERVICE                       │
│  ┌───────────────────────────────────────────────────────┐      │
│  │  gRPC Server (orchestrator/grpc_service.py)           │      │
│  │  - OrchestratorServiceServicer                        │      │
│  │  - Implements OrchestratorService RPCs                │      │
│  │  - Routes to appropriate agent based on intent        │      │
│  └──────────────┬────────────────────────────────────────┘      │
│                 │                                                │
│                 ▼                                                │
│  ┌───────────────────────────────────────────────────────┐      │
│  │  Multi-Agent System (14 Specialized Agents)           │      │
│  │  ┌─────────────────────────────────────────────┐      │      │
│  │  │ Research: FullResearchAgent, ResearchAgent  │      │      │
│  │  │ Content: PodcastScriptAgent, ArticleWritingAgent  │      │      │
│  │  │ Org-mode: OrgInboxAgent, OrgProjectAgent    │      │      │
│  │  │ Tools: WeatherAgent, ImageGenerationAgent   │      │      │
│  │  │ Data: RSSAgent, FactCheckingAgent           │      │      │
│  │  │ Support: ChatAgent, HelpAgent, Formatting   │      │      │
│  │  └─────────────────────────────────────────────┘      │      │
│  │  - Intent-based routing                               │      │
│  │  - Structured Pydantic responses                      │      │
│  │  - LangGraph workflows where appropriate              │      │
│  └──────────────┬────────────────────────────────────────┘      │
│                 │                                                │
│                 │ All agents use data access via gRPC            │
│                 ▼                                                │
│  ┌───────────────────────────────────────────────────────┐      │
│  │  Backend Tool Client (backend_tool_client.py)         │      │
│  │  - gRPC Client to Backend Tool Service                │      │
│  │  - Wraps gRPC calls in Python functions               │      │
│  │  - Used by all agents for data access                 │      │
│  └────────┬──────────────────────────────────────────────┘      │
│           │                                                      │
│           │ gRPC Client (tool_service.proto)                    │
└───────────┼──────────────────────────────────────────────────────┘
            │
            │ gRPC (Port 50052)
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BACKEND TOOL SERVICE                          │
│  ┌───────────────────────────────────────────────────────┐      │
│  │  gRPC Server (services/grpc_tool_service.py)          │      │
│  │  - ToolServiceServicer                                │      │
│  │  - Implements ToolService RPCs                        │      │
│  └──────────────┬────────────────────────────────────────┘      │
│                 │                                                │
│                 ▼                                                │
│  ┌───────────────────────────────────────────────────────┐      │
│  │  Data Access Layer                                    │      │
│  │  - UnifiedSearchService (documents)                   │      │
│  │  - WebContentTools (SearxNG, Crawl4AI)                │      │
│  │  - QueryExpansionService                              │      │
│  │  - ConversationCache                                  │      │
│  │  - EntityRepository                                   │      │
│  └───────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Multi-Agent Orchestrator Architecture

The LLM Orchestrator microservice implements a **multi-agent system** with specialized agents for different task types. This modular approach provides:

### Agent Roster

**Research Agents:**
- `FullResearchAgent` - Multi-round research with query expansion and gap analysis
- `ResearchAgent` - Single-round focused research

**Content Creation Agents:**
- `PodcastScriptAgent` - Generate podcast scripts from research
- `ArticleWritingAgent` - Generate long-form articles and short-form content (tweets)

**Org-mode Integration Agents:**
- `OrgInboxAgent` - Manage org-mode inbox items (add, toggle, archive, schedule)
- `OrgProjectAgent` - Capture and structure org-mode projects

**Utility Agents:**
- `WeatherAgent` - Weather data queries and forecasts
- `ImageGenerationAgent` - AI image generation via external APIs
- `RSSAgent` - RSS feed management and article retrieval

**Data Processing Agents:**
- `FactCheckingAgent` - Verify claims against knowledge base
- `DataFormattingAgent` - Format and structure data

**Support Agents:**
- `ChatAgent` - General conversational responses
- `HelpAgent` - System help and documentation

### Intent-Based Routing

The gRPC service (`orchestrator/grpc_service.py`) analyzes incoming requests and routes them to the appropriate agent based on:

1. **Query Intent Analysis** - Determines task type from user query
2. **Context Evaluation** - Considers conversation history and user preferences
3. **Agent Selection** - Routes to specialized agent best suited for the task
4. **Structured Response** - All agents return Pydantic-validated structured responses

### Agent Communication Pattern

```
User Request
    ↓
gRPC Server (orchestrator/grpc_service.py)
    ↓
Intent Analysis → Agent Selection
    ↓
Specialized Agent (e.g., WeatherAgent)
    ↓
Backend Tool Client (gRPC)
    ↓
Backend Tool Service (data access)
    ↓
Structured Response → User
```

All agents:
- Extend `BaseAgent` from `llm-orchestrator/orchestrator/agents/base_agent.py` for consistent interface
- Use `backend_tool_client` for data access via gRPC
- Return structured Pydantic models (not raw strings)
- Support streaming responses where appropriate

---

## Why gRPC for AI Microservices?

### Benefits

1. **Independent Scaling**
   - Scale LLM orchestrator separately from main backend
   - Different resource requirements (CPU vs. GPU)
   - Independent deployment cycles

2. **Language Agnostic**
   - gRPC supports multiple languages
   - Could rewrite orchestrator in Go/Rust for performance
   - Mix languages based on strengths

3. **Efficient Binary Protocol**
   - Smaller payload sizes than JSON
   - Faster serialization/deserialization
   - Built-in compression

4. **Strong Typing**
   - Protocol Buffers enforce contracts
   - Catch integration errors at compile time
   - Auto-generated client/server code

5. **Streaming Support**
   - Server-side streaming for LLM responses
   - Client-side streaming for batch requests
   - Bidirectional streaming for interactive flows

6. **Independent Maintenance**
   - Upgrade orchestrator without backend downtime
   - Test new LLM strategies in isolation
   - Roll back easily if issues occur

---

## Protocol Buffers (Proto Files)

### File Organization - Shared Proto Architecture

We use a **shared proto directory** at the repository root - a single source of truth for all service definitions:

```
/opt/bastion/protos/           # ← SINGLE SOURCE OF TRUTH
├── orchestrator.proto         # LLM Orchestrator service API
├── tool_service.proto         # Backend Tool service API  
├── vector_service.proto       # Vector service API
└── README.md                  # Proto documentation
```

**Benefits of Shared Protos:**

1. **Single Source of Truth** - Proto definitions exist in ONE place
2. **Zero Synchronization** - No copying files between services
3. **Atomic Changes** - Update API contract in one edit
4. **Version Control** - Track all API changes in one directory
5. **Build Simplicity** - Docker copies shared directory to both containers

**How It Works:**

Both `backend` and `llm-orchestrator` services:
1. Set Docker build context to repository root in `docker-compose.yml`
2. Copy the shared `protos/` directory during Docker build
3. Generate gRPC code (`*_pb2.py`, `*_pb2_grpc.py`) inside each container
4. Use identical protocol definitions - guaranteed consistency

### Why Shared Protos Matter

**Problem with Duplicated Protos:**

In many gRPC projects, each service has its own copy of proto files:
```
❌ backend/protos/tool_service.proto
❌ llm-orchestrator/protos/tool_service.proto
```

This creates maintenance nightmares:
- 🐛 **Drift Risk** - Files get out of sync, breaking service communication
- 📋 **Manual Syncing** - Must copy files between directories after every change
- 🔄 **Version Confusion** - Hard to tell which proto version a service is using
- 🚨 **Integration Bugs** - Services may use different field numbers/types
- ⏰ **Wasted Time** - Extra steps in every proto update workflow

**Our Solution: Repository-Root Shared Protos**

```
✅ /opt/bastion/protos/
   ├── orchestrator.proto
   ├── tool_service.proto
   └── vector_service.proto
```

**Guarantees:**
- 🎯 **Zero Drift** - Impossible for services to have different definitions
- ⚡ **One-Step Updates** - Edit once, rebuild services, done
- 🔒 **Type Safety** - Compile-time guarantee of contract compatibility
- 📝 **Clear Versioning** - Git history shows all API changes in one place
- 🚀 **Faster Development** - No manual copying, no sync checks

**Real-World Impact:**

```bash
# Old way (duplicated protos):
vim backend/protos/tool_service.proto      # Edit proto
vim llm-orchestrator/protos/tool_service.proto  # Copy changes
# Hope you didn't miss anything!
docker compose build backend llm-orchestrator

# New way (shared protos):
vim protos/tool_service.proto              # Edit once
docker compose build backend llm-orchestrator  # Both services use it
# Guaranteed consistency!
```

### Proto File Structure

```protobuf
syntax = "proto3";

package orchestrator;

// Service definition - the contract
service OrchestratorService {
  // RPC method: request type -> response type
  rpc StreamChat(ChatRequest) returns (stream ChatChunk);
  rpc HealthCheck(HealthCheckRequest) returns (HealthCheckResponse);
}

// Message definitions - the data structures
message ChatRequest {
  string query = 1;           // Field number (for serialization)
  string user_id = 2;
  string conversation_id = 3;
}

message ChatChunk {
  string type = 1;            // "status", "content", "complete", "error"
  string message = 2;
  string agent_name = 3;
  string timestamp = 4;
}
```

### Field Numbering Rules

- **Field numbers 1-15**: Use 1 byte (use for frequent fields)
- **Field numbers 16-2047**: Use 2 bytes
- **Reserved numbers**: Never reuse deleted field numbers
- **Required vs Optional**: All fields in proto3 are optional by default

### Adding New RPC Methods

1. **Define in shared proto file (`protos/tool_service.proto`):**

```protobuf
service ToolService {
  // Existing methods...
  
  // New method
  rpc SearchVideos(VideoSearchRequest) returns (VideoSearchResponse);
}

message VideoSearchRequest {
  string query = 1;
  int32 limit = 2;
  string user_id = 3;
}

message VideoSearchResponse {
  repeated VideoResult results = 1;
  int32 total_count = 2;
}

message VideoResult {
  string video_id = 1;
  string title = 2;
  string url = 3;
  float relevance_score = 4;
}
```

2. **Regenerate code** (happens automatically in Docker build):

```bash
python -m grpc_tools.protoc \
    -I. \
    --python_out=. \
    --grpc_python_out=. \
    ./protos/tool_service.proto
```

3. **Implement server side:**

```python
# backend/services/grpc_tool_service.py

async def SearchVideos(
    self,
    request: tool_service_pb2.VideoSearchRequest,
    context: grpc.aio.ServicerContext
) -> tool_service_pb2.VideoSearchResponse:
    """Search videos by query"""
    try:
        # Your implementation
        video_service = VideoSearchService()
        results = await video_service.search(
            query=request.query,
            limit=request.limit,
            user_id=request.user_id
        )
        
        # Convert to proto response
        response = tool_service_pb2.VideoSearchResponse(
            total_count=len(results)
        )
        
        for video in results:
            video_result = tool_service_pb2.VideoResult(
                video_id=video['id'],
                title=video['title'],
                url=video['url'],
                relevance_score=video['score']
            )
            response.results.append(video_result)
        
        return response
        
    except Exception as e:
        logger.error(f"SearchVideos error: {e}")
        await context.abort(
            grpc.StatusCode.INTERNAL, 
            f"Video search failed: {str(e)}"
        )
```

4. **Implement client side:**

```python
# llm-orchestrator/orchestrator/backend_tool_client.py

async def search_videos(
    self,
    query: str,
    limit: int = 10,
    user_id: str = "system"
) -> List[Dict[str, Any]]:
    """Search videos via backend tool service"""
    try:
        await self._ensure_connected()
        
        request = tool_service_pb2.VideoSearchRequest(
            query=query,
            limit=limit,
            user_id=user_id
        )
        
        response = await self._stub.SearchVideos(request)
        
        videos = []
        for video in response.results:
            videos.append({
                'video_id': video.video_id,
                'title': video.title,
                'url': video.url,
                'relevance_score': video.relevance_score
            })
        
        return videos
        
    except grpc.RpcError as e:
        logger.error(f"Video search failed: {e.code()} - {e.details()}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in video search: {e}")
        return []
```

5. **Create LangGraph tool wrapper:**

```python
# llm-orchestrator/orchestrator/tools/video_tools.py

from langchain_core.tools import tool

@tool
async def search_videos_tool(query: str, limit: int = 10) -> str:
    """
    Search for videos related to a query.
    
    Args:
        query: Search query
        limit: Maximum number of results
        
    Returns:
        Formatted string with video results
    """
    from orchestrator.backend_tool_client import backend_tool_client
    
    try:
        videos = await backend_tool_client.search_videos(
            query=query,
            limit=limit
        )
        
        if not videos:
            return f"No videos found for: {query}"
        
        result = f"Found {len(videos)} videos:\n\n"
        for i, video in enumerate(videos, 1):
            result += f"{i}. {video['title']}\n"
            result += f"   URL: {video['url']}\n"
            result += f"   Relevance: {video['relevance_score']:.2f}\n\n"
        
        return result
        
    except Exception as e:
        return f"Error searching videos: {str(e)}"
```

---

## Service Communication Patterns

### Pattern 1: Request-Response (Unary)

**Use for:** Simple queries with single response

```python
# Client
request = tool_service_pb2.SearchRequest(
    query="quantum computing",
    limit=10
)
response = await stub.SearchDocuments(request)
print(f"Found {response.total_count} documents")
```

### Pattern 2: Server-Side Streaming

**Use for:** LLM responses, progress updates

```python
# Server
async def StreamChat(self, request, context):
    yield ChatChunk(type="status", message="Starting research...")
    yield ChatChunk(type="content", message="Based on my analysis...")
    yield ChatChunk(type="complete", message="Research complete")

# Client
async for chunk in stub.StreamChat(request):
    print(f"{chunk.type}: {chunk.message}")
```

### Pattern 3: Client-Side Streaming

**Use for:** Batch uploads, continuous data

```python
# Client
async def upload_batch():
    async def request_generator():
        for item in batch:
            yield BatchItem(data=item)
    
    response = await stub.ProcessBatch(request_generator())
```

### Pattern 4: Bidirectional Streaming

**Use for:** Interactive conversations, real-time collaboration

```python
# Both sides stream
async def chat_session(stub):
    async def request_stream():
        while True:
            user_input = await get_user_input()
            yield ChatMessage(content=user_input)
    
    async for response in stub.InteractiveChat(request_stream()):
        display_response(response)
```

---

## Connection Management

### Client Connection Best Practices

```python
class BackendToolClient:
    def __init__(self, host='backend', port=50052):
        self._host = host
        self._port = port
        self._channel: Optional[grpc.aio.Channel] = None
        self._stub = None
        self._connection_lock = asyncio.Lock()
    
    async def _ensure_connected(self):
        """Lazy connection with connection pooling"""
        async with self._connection_lock:
            if self._channel is None:
                self._channel = grpc.aio.insecure_channel(
                    f'{self._host}:{self._port}',
                    options=[
                        ('grpc.max_send_message_length', 50 * 1024 * 1024),
                        ('grpc.max_receive_message_length', 50 * 1024 * 1024),
                        ('grpc.keepalive_time_ms', 30000),
                        ('grpc.keepalive_timeout_ms', 10000),
                    ]
                )
                self._stub = tool_service_pb2_grpc.ToolServiceStub(self._channel)
    
    async def close(self):
        """Graceful shutdown"""
        if self._channel:
            await self._channel.close()
            self._channel = None
            self._stub = None
```

### Connection Options Explained

| Option | Purpose | Value |
|--------|---------|-------|
| `grpc.max_send_message_length` | Max request size | 50MB for large documents |
| `grpc.max_receive_message_length` | Max response size | 50MB for large results |
| `grpc.keepalive_time_ms` | Ping interval | 30s to keep connection alive |
| `grpc.keepalive_timeout_ms` | Ping timeout | 10s before marking dead |
| `grpc.enable_retries` | Automatic retries | true for reliability |
| `grpc.initial_reconnect_backoff_ms` | Retry delay | 1000ms starting backoff |

---

## Error Handling

### Server-Side Error Handling

```python
async def SearchDocuments(self, request, context):
    try:
        # Validate request
        if not request.query:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Query cannot be empty"
            )
        
        # Perform operation
        results = await search_service.search(request.query)
        
        # Return response
        return SearchResponse(results=results)
        
    except ValueError as e:
        # Client error
        await context.abort(
            grpc.StatusCode.INVALID_ARGUMENT,
            f"Invalid input: {str(e)}"
        )
    
    except PermissionError as e:
        # Authorization error
        await context.abort(
            grpc.StatusCode.PERMISSION_DENIED,
            f"Access denied: {str(e)}"
        )
    
    except Exception as e:
        # Server error
        logger.error(f"SearchDocuments error: {e}", exc_info=True)
        await context.abort(
            grpc.StatusCode.INTERNAL,
            f"Internal server error: {str(e)}"
        )
```

### Client-Side Error Handling

```python
async def search_documents(self, query: str) -> List[Dict]:
    try:
        await self._ensure_connected()
        
        request = tool_service_pb2.SearchRequest(query=query)
        response = await self._stub.SearchDocuments(request)
        
        return [self._convert_result(r) for r in response.results]
        
    except grpc.RpcError as e:
        # Handle specific gRPC errors
        if e.code() == grpc.StatusCode.INVALID_ARGUMENT:
            logger.warning(f"Invalid query: {e.details()}")
            return []
        
        elif e.code() == grpc.StatusCode.UNAVAILABLE:
            logger.error(f"Service unavailable: {e.details()}")
            # Could trigger fallback to local service here
            return []
        
        elif e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
            logger.error(f"Request timeout: {e.details()}")
            return []
        
        else:
            logger.error(f"gRPC error: {e.code()} - {e.details()}")
            return []
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return []
```

### gRPC Status Codes Reference

| Code | When to Use | HTTP Equivalent |
|------|-------------|-----------------|
| `OK` | Success | 200 |
| `INVALID_ARGUMENT` | Bad input | 400 |
| `NOT_FOUND` | Resource missing | 404 |
| `PERMISSION_DENIED` | Auth failed | 403 |
| `UNAUTHENTICATED` | No credentials | 401 |
| `RESOURCE_EXHAUSTED` | Rate limit | 429 |
| `UNIMPLEMENTED` | Not implemented | 501 |
| `UNAVAILABLE` | Service down | 503 |
| `INTERNAL` | Server error | 500 |
| `DEADLINE_EXCEEDED` | Timeout | 504 |

---

## Docker Integration

### Service Discovery

Services discover each other by **service name** in Docker Compose:

```yaml
# docker-compose.yml
services:
  backend:
    # Accessible as 'backend' hostname
    ports:
      - "8081:8000"
      - "50052:50052"  # gRPC Tool Service
  
  llm-orchestrator:
    # Accessible as 'llm-orchestrator' hostname
    depends_on:
      - backend
    environment:
      - BACKEND_GRPC_HOST=backend
      - BACKEND_GRPC_PORT=50052
    ports:
      - "50051:50051"  # gRPC Orchestrator Service
```

Connection from orchestrator to backend:

```python
# Uses Docker service name as hostname
channel = grpc.aio.insecure_channel('backend:50052')
```

Connection from backend to orchestrator:

```python
# Uses Docker service name as hostname
channel = grpc.aio.insecure_channel('llm-orchestrator:50051')
```

### Dockerfile Proto Generation with Shared Protos

**Key Architecture Decision:** Docker build context is set to **repository root** in `docker-compose.yml`:

```yaml
# docker-compose.yml
services:
  backend:
    build:
      context: .              # ← Root context (not ./backend)
      dockerfile: ./backend/Dockerfile
  
  llm-orchestrator:
    build:
      context: .              # ← Root context (not ./llm-orchestrator)
      dockerfile: ./llm-orchestrator/Dockerfile
```

This allows both services to access the shared `protos/` directory:

```dockerfile
# Backend Dockerfile (or llm-orchestrator Dockerfile)
FROM python:3.11-slim

WORKDIR /app

# Install gRPC tools first
RUN pip install --no-cache-dir grpcio==1.76.0 grpcio-tools==1.76.0

# Copy shared protos from root (available because context is root)
COPY protos /app/protos

# Copy service-specific code
COPY backend /app
# OR for orchestrator: COPY llm-orchestrator /app

# Generate gRPC code from shared protos
RUN python -m grpc_tools.protoc \
    -I/app \
    --python_out=/app \
    --grpc_python_out=/app \
    /app/protos/orchestrator.proto \
    /app/protos/tool_service.proto \
    /app/protos/vector_service.proto

# Expose ports
EXPOSE 8000 50052

CMD ["./docker-entrypoint.sh"]
```

**Benefits:**
- ✅ **Single proto source** - both services use identical definitions
- ✅ **No synchronization** - no manual copying between services
- ✅ **Atomic updates** - edit proto once, rebuild both services
- ✅ **Generated files** - created during build, never committed to git

**Key points:**
- Build context is repository root (set in `docker-compose.yml`)
- Copy shared `protos/` directory before generating code
- Install `grpcio-tools` before generation
- Generate during build, not at runtime
- Generated files go into `/app/protos/` inside container

---

## Feature Flags for Gradual Rollout

### Implementation

```python
# backend/utils/feature_flags.py

class FeatureFlagService:
    def __init__(self):
        self._flags = {
            'USE_GRPC_ORCHESTRATOR': os.getenv('USE_GRPC_ORCHESTRATOR', 'false').lower() == 'true',
            'GRPC_ORCHESTRATOR_PERCENTAGE': int(os.getenv('GRPC_ORCHESTRATOR_PERCENTAGE', '0')),
        }
    
    def should_use_grpc_orchestrator(self, user_id: str = None) -> bool:
        """Determine if request should use gRPC orchestrator"""
        if not self._flags['USE_GRPC_ORCHESTRATOR']:
            return False
        
        percentage = self._flags['GRPC_ORCHESTRATOR_PERCENTAGE']
        
        # 0% = nobody, 100% = everyone
        if percentage == 0:
            return False
        if percentage == 100:
            return True
        
        # Gradual rollout: hash user_id for consistent routing
        if user_id:
            user_hash = hash(user_id) % 100
            return user_hash < percentage
        
        # No user_id: use percentage directly
        import random
        return random.randint(0, 99) < percentage
```

### Routing with Feature Flags

```python
# backend/api/async_orchestrator_api.py

@router.post("/stream")
async def stream_orchestrator_response(request, current_user):
    # Check feature flag
    from utils.feature_flags import use_grpc_orchestrator
    
    if use_grpc_orchestrator(current_user.user_id):
        logger.info("Routing to gRPC orchestrator")
        # Forward to gRPC microservice
        from api.grpc_orchestrator_proxy import stream_from_grpc_orchestrator
        return StreamingResponse(
            stream_from_grpc_orchestrator(
                query=request.query,
                conversation_id=request.conversation_id,
                user_id=current_user.user_id
            ),
            media_type="text/event-stream"
        )
    
    logger.info("Using local orchestrator")
    # Use local implementation
    # ... existing code ...
```

### Rollout Strategy

```yaml
# Phase 1: Testing (0%)
- USE_GRPC_ORCHESTRATOR=true
- GRPC_ORCHESTRATOR_PERCENTAGE=0

# Phase 2: Canary (5%)
- USE_GRPC_ORCHESTRATOR=true
- GRPC_ORCHESTRATOR_PERCENTAGE=5

# Phase 3: Gradual (50%)
- USE_GRPC_ORCHESTRATOR=true
- GRPC_ORCHESTRATOR_PERCENTAGE=50

# Phase 4: Full rollout (100%)
- USE_GRPC_ORCHESTRATOR=true
- GRPC_ORCHESTRATOR_PERCENTAGE=100
```

---

## Testing gRPC Services

### Unit Testing

```python
# test_grpc_service.py

import pytest
import grpc
from grpc.aio import insecure_channel
from protos import tool_service_pb2, tool_service_pb2_grpc

@pytest.mark.asyncio
async def test_search_documents():
    """Test document search RPC"""
    async with insecure_channel('localhost:50052') as channel:
        stub = tool_service_pb2_grpc.ToolServiceStub(channel)
        
        request = tool_service_pb2.SearchRequest(
            user_id="test_user",
            query="quantum computing",
            limit=5
        )
        
        response = await stub.SearchDocuments(request)
        
        assert response.total_count >= 0
        assert len(response.results) <= 5
        for result in response.results:
            assert result.document_id
            assert result.relevance_score >= 0.0

@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling for invalid input"""
    async with insecure_channel('localhost:50052') as channel:
        stub = tool_service_pb2_grpc.ToolServiceStub(channel)
        
        # Empty query should fail
        request = tool_service_pb2.SearchRequest(
            user_id="test_user",
            query="",  # Invalid!
            limit=5
        )
        
        with pytest.raises(grpc.RpcError) as exc_info:
            await stub.SearchDocuments(request)
        
        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
```

### Integration Testing

```python
# test_end_to_end.py

@pytest.mark.asyncio
async def test_full_research_workflow():
    """Test complete workflow: Backend → Orchestrator → Backend Tools"""
    
    # Connect to orchestrator
    async with insecure_channel('localhost:50051') as channel:
        stub = orchestrator_pb2_grpc.OrchestratorServiceStub(channel)
        
        request = orchestrator_pb2.ChatRequest(
            query="Research quantum computing applications",
            user_id="test_user",
            conversation_id="test_conv_123"
        )
        
        # Collect streaming response
        chunks = []
        async for chunk in stub.StreamChat(request):
            chunks.append(chunk)
            print(f"{chunk.type}: {chunk.message[:100]}")
        
        # Verify workflow
        assert any(c.type == "status" for c in chunks)
        assert any(c.type == "content" for c in chunks)
        assert any(c.type == "complete" for c in chunks)
        
        # Verify no errors
        assert not any(c.type == "error" for c in chunks)
```

### Load Testing

```python
# test_load.py

import asyncio
import time

async def load_test(num_concurrent=10, num_requests=100):
    """Test gRPC service under load"""
    
    async def single_request():
        async with insecure_channel('localhost:50052') as channel:
            stub = tool_service_pb2_grpc.ToolServiceStub(channel)
            request = tool_service_pb2.SearchRequest(
                query="test query",
                limit=10
            )
            start = time.time()
            response = await stub.SearchDocuments(request)
            duration = time.time() - start
            return duration
    
    # Run concurrent requests
    tasks = [single_request() for _ in range(num_requests)]
    
    start = time.time()
    durations = await asyncio.gather(*tasks)
    total_time = time.time() - start
    
    # Report metrics
    print(f"Total requests: {num_requests}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Requests/sec: {num_requests/total_time:.2f}")
    print(f"Avg latency: {sum(durations)/len(durations)*1000:.2f}ms")
    print(f"P95 latency: {sorted(durations)[int(len(durations)*0.95)]*1000:.2f}ms")
```

---

## Monitoring and Observability

### Structured Logging

```python
import logging
import json

class GRPCLogger:
    def __init__(self, service_name: str):
        self.logger = logging.getLogger(service_name)
        self.service_name = service_name
    
    def log_rpc_call(
        self,
        method: str,
        user_id: str,
        duration_ms: float,
        status: grpc.StatusCode,
        error: str = None
    ):
        """Structured logging for RPC calls"""
        log_data = {
            'service': self.service_name,
            'method': method,
            'user_id': user_id,
            'duration_ms': duration_ms,
            'status': status.name,
            'error': error,
            'timestamp': time.time()
        }
        
        if status == grpc.StatusCode.OK:
            self.logger.info(json.dumps(log_data))
        else:
            self.logger.error(json.dumps(log_data))

# Usage
grpc_logger = GRPCLogger('tool_service')

async def SearchDocuments(self, request, context):
    start = time.time()
    try:
        result = await self._search(request)
        
        grpc_logger.log_rpc_call(
            method='SearchDocuments',
            user_id=request.user_id,
            duration_ms=(time.time() - start) * 1000,
            status=grpc.StatusCode.OK
        )
        
        return result
        
    except Exception as e:
        grpc_logger.log_rpc_call(
            method='SearchDocuments',
            user_id=request.user_id,
            duration_ms=(time.time() - start) * 1000,
            status=grpc.StatusCode.INTERNAL,
            error=str(e)
        )
        raise
```

### Health Checks

```python
# Health check implementation
async def HealthCheck(
    self,
    request: orchestrator_pb2.HealthCheckRequest,
    context: grpc.aio.ServicerContext
) -> orchestrator_pb2.HealthCheckResponse:
    """Health check for orchestrator service"""
    
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': 'Multi-Agent Architecture',
        'agent_count': 14,
        'backend_connection': 'unknown'
    }
    
    # Check backend connectivity
    try:
        await backend_tool_client._ensure_connected()
        health_status['backend_connection'] = 'connected'
    except Exception as e:
        health_status['backend_connection'] = f'error: {str(e)}'
        health_status['status'] = 'degraded'
    
    return orchestrator_pb2.HealthCheckResponse(
        status=health_status['status'],
        details=health_status
    )
```

### Docker Health Checks

```dockerfile
# Dockerfile with health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import socket; s = socket.socket(); s.settimeout(5); s.connect(('localhost', 50051)); s.close()" || exit 1
```

---

## Performance Optimization

### Connection Pooling

gRPC channels are expensive to create - reuse them:

```python
# ❌ BAD: Creates new channel for each request
async def search(query):
    channel = grpc.aio.insecure_channel('backend:50052')
    stub = ToolServiceStub(channel)
    response = await stub.SearchDocuments(request)
    await channel.close()
    return response

# ✅ GOOD: Reuse channel across requests
class ToolClient:
    def __init__(self):
        self._channel = None
        self._stub = None
    
    async def _ensure_connected(self):
        if self._channel is None:
            self._channel = grpc.aio.insecure_channel('backend:50052')
            self._stub = ToolServiceStub(self._channel)
    
    async def search(self, query):
        await self._ensure_connected()
        return await self._stub.SearchDocuments(request)
```

### Message Size Limits

For large payloads (documents, embeddings), increase limits:

```python
options = [
    ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50MB
    ('grpc.max_receive_message_length', 50 * 1024 * 1024),
]
channel = grpc.aio.insecure_channel('backend:50052', options=options)
```

### Compression

Enable compression for large responses:

```python
# Server side
async def SearchDocuments(self, request, context):
    context.set_compression(grpc.Compression.Gzip)
    # ... return large response ...
```

### Timeouts

Set appropriate timeouts:

```python
# Per-call timeout
response = await stub.SearchDocuments(
    request,
    timeout=30.0  # 30 seconds
)

# Channel-level default timeout
options = [
    ('grpc.http2.max_pings_without_data', 0),
    ('grpc.keepalive_time_ms', 30000),
]
```

---

## Common Patterns and Recipes

### Pattern: Pagination

```protobuf
message SearchRequest {
  string query = 1;
  int32 page_size = 2;
  string page_token = 3;  // Opaque token for next page
}

message SearchResponse {
  repeated Result results = 1;
  string next_page_token = 2;
  int32 total_count = 3;
}
```

### Pattern: Batch Operations

```protobuf
message BatchSearchRequest {
  repeated SearchQuery queries = 1;
}

message BatchSearchResponse {
  repeated SearchResult results = 1;  // Ordered like requests
}
```

### Pattern: Progress Reporting

```protobuf
service OrchestratorService {
  rpc LongRunningTask(TaskRequest) returns (stream ProgressUpdate);
}

message ProgressUpdate {
  int32 percent_complete = 1;
  string current_step = 2;
  string estimated_time_remaining = 3;
  bool is_complete = 4;
}
```

### Pattern: Metadata/Headers

```python
# Client: Send metadata
metadata = (
    ('user-id', 'user123'),
    ('api-key', '<your-api-key>'),
)
response = await stub.SearchDocuments(request, metadata=metadata)

# Server: Read metadata
async def SearchDocuments(self, request, context):
    metadata = dict(context.invocation_metadata())
    user_id = metadata.get('user-id')
    api_key = metadata.get('api-key')
    # ... validate and process ...
```

---

## Migration Strategy

### Phase 1: Parallel Implementation

1. Build new gRPC service alongside existing code
2. Feature flag OFF (0% traffic)
3. Test in isolation

### Phase 2: Canary Deployment

1. Enable feature flag (5% traffic)
2. Monitor metrics, errors, latency
3. Compare with existing implementation

### Phase 3: Gradual Rollout

1. Increase percentage: 5% → 25% → 50% → 75%
2. Monitor at each step
3. Roll back if issues detected

### Phase 4: Full Migration

1. 100% traffic to gRPC service
2. Keep legacy code for emergency fallback
3. Monitor for 1-2 weeks

### Phase 5: Cleanup

1. Remove legacy implementation
2. Remove feature flags
3. Update documentation

---

## Troubleshooting

### Issue: "Connection refused"

**Cause:** Service not running or wrong port

**Fix:**
```bash
# Check service is running
docker ps | grep orchestrator

# Check logs
docker logs codex-dev-llm-orchestrator

# Verify port mapping
docker port codex-dev-llm-orchestrator
```

### Issue: "Module 'protos.xxx_pb2' has no attribute 'YYY'"

**Cause:** Proto files not regenerated after changes

**Fix:**
```bash
# Rebuild with fresh proto generation
docker compose build --no-cache llm-orchestrator
docker compose up -d
```

### Issue: "StatusCode.UNAVAILABLE"

**Cause:** Service down or network issue

**Fix:**
- Check service health
- Verify Docker network connectivity
- Check firewall rules
- Implement retry logic with exponential backoff

### Issue: "StatusCode.DEADLINE_EXCEEDED"

**Cause:** Request timeout

**Fix:**
- Increase timeout value
- Optimize slow operations
- Add caching
- Use streaming for long operations

---

## Best Practices Summary

### DO:

✅ Use shared proto directory at repository root (single source of truth)
✅ Use proto3 for new services
✅ Version your API (v1, v2) in package names
✅ Reuse gRPC channels (they're expensive)
✅ Set appropriate timeouts
✅ Use feature flags for gradual rollout
✅ Implement health checks
✅ Log structured data for monitoring
✅ Handle errors gracefully
✅ Test services independently
✅ Use streaming for large/progressive responses

### DON'T:

❌ Duplicate proto files across service directories
❌ Reuse field numbers after deletion
❌ Make breaking proto changes (rename/remove fields)
❌ Create new channel for each request
❌ Use gRPC for frontend communication (use REST/WebSocket)
❌ Return huge responses in unary calls (use streaming)
❌ Ignore errors or timeout values
❌ Deploy without health checks
❌ Skip integration testing
❌ Hardcode service URLs (use env vars)
❌ Log sensitive data

---

## References

- **gRPC Documentation:** https://grpc.io/docs/languages/python/
- **Protocol Buffers:** https://protobuf.dev/
- **Our Implementation:**
  - Backend gRPC Server: `/opt/bastion/backend/services/grpc_tool_service.py`
  - Orchestrator gRPC Server: `/opt/bastion/llm-orchestrator/orchestrator/grpc_service.py`
  - **Shared Proto Files:** `/opt/bastion/protos/` (single source of truth)
  - Proto Documentation: `/opt/bastion/protos/README.md`

---

**Architecture maintained by: Development Team**  
**Last updated: November 2025 (Multi-Agent Architecture)**  
**Agent Count:** 14 specialized agents in production  
**Questions?** See agent-specific documentation in `llm-orchestrator/orchestrator/agents/`



