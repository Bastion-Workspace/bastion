# LangGraph Context Gathering Infrastructure

## Overview

The **gRPC Context Gatherer** is the centralized infrastructure for assembling comprehensive user context and sending it to the `llm-orchestrator` microservice. This is the **SINGLE SOURCE OF TRUTH** for what context gets transmitted across the service boundary.

## Architecture

```
┌──────────────┐
│   Frontend   │ 
│  (React UI)  │
└──────┬───────┘
       │ HTTP POST with context fields
       │ (active_editor, pipeline, etc.)
       ▼
┌──────────────────────────────┐
│  Backend FastAPI Endpoint    │
│  grpc_orchestrator_proxy.py  │
└──────────┬───────────────────┘
           │ Extract request_context
           │ from frontend fields
           ▼
┌──────────────────────────────┐
│   GRPCContextGatherer        │
│  grpc_context_gatherer.py    │
│                              │
│  • Conversation history      │
│  • User persona              │
│  • Editor context (optional) │
│  • Pipeline context (opt)    │
│  • Permission grants (opt)   │
│  • Pending operations (opt)  │
│  • Routing locks (opt)       │
│  • Checkpointing (opt)       │
└──────────┬───────────────────┘
           │ Build ChatRequest proto
           ▼
┌──────────────────────────────┐
│   gRPC orchestrator.proto    │
│  Comprehensive Context Spec  │
└──────────┬───────────────────┘
           │ Send via gRPC
           ▼
┌──────────────────────────────┐
│   LLM Orchestrator Service   │
│   (llm-orchestrator:50051)   │
│                              │
│  • Intent classification     │
│  • Agent routing             │
│  • Agent execution           │
└──────────────────────────────┘
```

## Context Types

### 1. Core Identity (Always Required)

```python
{
    "user_id": "uuid",
    "conversation_id": "uuid",
    "query": "user question",
    "session_id": "default"
}
```

### 2. Conversation History (Optional but Recommended)

Extracted from LangGraph state if available, limited to last 20 messages:

```python
conversation_history = [
    {
        "role": "user",
        "content": "What is quantum computing?",
        "timestamp": "2024-01-15T10:30:00Z"
    },
    {
        "role": "assistant",
        "content": "Quantum computing is...",
        "timestamp": "2024-01-15T10:30:05Z"
    }
]
```

### 3. User Persona (Optional)

User preferences from settings:

```python
persona = {
    "ai_name": "Kodex",
    "persona_style": "professional",
    "political_bias": "neutral",
    "timezone": "America/New_York"
}
```

### 4. Active Editor Context (Conditional - Editor Page Only)

When user is on the editor page with an editable markdown file:

```python
active_editor = {
    "is_editable": true,
    "filename": "chapter1.md",
    "language": "markdown",
    "content": "# Chapter 1\n\nIt was a dark...",
    "content_length": 5000,
    "frontmatter": {
        "type": "fiction",
        "title": "The Great Adventure",
        "author": "John Doe",
        "tags": ["adventure", "fantasy"],
        "status": "draft"
    },
    "editor_preference": "prefer"  # or "ignore"
}
```

**Include When:**
- User is on editor page (`/editor/{file_id}`)
- `editor_preference != "ignore"`
- File is editable markdown (`.md`)

**Skip When:**
- User said to ignore editor context
- Not on editor page
- File is not editable

### 5. Pipeline Context (Conditional - Pipeline Page Only)

When user is on pipeline execution page:

```python
pipeline_context = {
    "pipeline_preference": "prefer",  # or "ignore"
    "active_pipeline_id": "pipeline-uuid",
    "pipeline_name": "My Research Pipeline"
}
```

**Include When:**
- User is on pipeline page (`/pipelines/{pipeline_id}`)
- `pipeline_preference != "ignore"`
- Pipeline is active/selected

**Skip When:**
- User said to ignore pipelines
- Not on pipeline page

### 6. Permission Grants (Conditional)

When user has granted HITL permissions in current conversation:

```python
permission_grants = {
    "web_search_permission": true,
    "web_crawl_permission": false,
    "file_write_permission": true,
    "external_api_permission": false
}
```

**Include When:**
- LangGraph state has `shared_memory["web_search_permission"] = True`
- Any permission has been granted in conversation

**Skip When:**
- No permissions granted yet
- Fresh conversation

### 7. Pending Operations (Conditional)

Operations awaiting user approval:

```python
pending_operations = [
    {
        "id": "op-uuid",
        "type": "web_search",
        "summary": "Search for quantum computing papers",
        "permission_required": true,
        "status": "pending",
        "created_at": "2024-01-15T10:30:00Z"
    }
]
```

**Include When:**
- LangGraph state has `pending_operations` array
- Operations are awaiting approval

**Skip When:**
- No pending operations

### 8. Routing Locks (Conditional)

When conversation is locked to specific agent:

```python
locked_agent = "fiction_editing_agent"
```

**Include When:**
- User has entered dedicated agent session (e.g., "Edit my fiction")
- `shared_memory["locked_agent"]` is set
- Request context includes `locked_agent`

**Skip When:**
- Normal conversation (auto-routing)

### 9. Checkpointing (Conditional)

For conversation branching:

```python
base_checkpoint_id = "checkpoint-uuid"
```

**Include When:**
- User wants to branch from specific conversation point
- Implementing conversation "what-if" scenarios

**Skip When:**
- Normal linear conversation

## Implementation

### Backend Integration

File: `/opt/bastion/backend/services/grpc_context_gatherer.py`

```python
from services.grpc_context_gatherer import get_context_gatherer

# Build comprehensive request
context_gatherer = get_context_gatherer()
grpc_request = await context_gatherer.build_chat_request(
    query=query,
    user_id=user_id,
    conversation_id=conversation_id,
    session_id=session_id,
    request_context=request_context,  # From frontend
    state=state,  # From LangGraph (if available)
    agent_type=agent_type,  # Optional routing
    routing_reason=routing_reason
)
```

### Frontend Integration

The frontend should send relevant context fields in the request:

```javascript
// POST /api/async/orchestrator/grpc/stream
{
  "query": "Proofread chapter 1",
  "conversation_id": "conv-uuid",
  "session_id": "default",
  
  // Include when on editor page
  "active_editor": {
    "is_editable": true,
    "filename": "chapter1.md",
    "content": "...",
    "frontmatter": {...}
  },
  "editor_preference": "prefer",
  
  // Include when on pipeline page
  "pipeline_preference": "prefer",
  "active_pipeline_id": "pipeline-uuid",
  
  // Include when conversation locked to agent
  "locked_agent": "fiction_editing_agent"
}
```

## Context Gathering Logic

### 1. Core Identity
**Always included** - extracted from request parameters.

### 2. Conversation History
```python
async def _add_conversation_history(
    self,
    grpc_request: orchestrator_pb2.ChatRequest,
    state: Optional[Dict[str, Any]],
    conversation_id: str,
    user_id: str
) -> None:
    """
    Extract last 20 messages from LangGraph state
    Convert from LangChain BaseMessage format to proto format
    """
```

### 3. User Persona
```python
async def _add_user_persona(
    self,
    grpc_request: orchestrator_pb2.ChatRequest,
    user_id: str
) -> None:
    """
    Fetch user settings from prompt_service
    Build UserPersona proto message
    """
```

### 4. Editor Context
```python
async def _add_editor_context(
    self,
    grpc_request: orchestrator_pb2.ChatRequest,
    request_context: Dict[str, Any]
) -> None:
    """
    Check if editor context exists
    Validate: is_editable, .md file, editor_preference != "ignore"
    Parse frontmatter and build ActiveEditor proto
    """
```

### 5. Pipeline Context
```python
async def _add_pipeline_context(
    self,
    grpc_request: orchestrator_pb2.ChatRequest,
    request_context: Dict[str, Any]
) -> None:
    """
    Check if pipeline context exists
    Validate: pipeline_preference != "ignore"
    Build PipelineContext proto
    """
```

### 6. Permission Grants
```python
async def _add_permission_grants(
    self,
    grpc_request: orchestrator_pb2.ChatRequest,
    state: Optional[Dict[str, Any]]
) -> None:
    """
    Extract permissions from shared_memory
    Build PermissionGrants proto if any granted
    """
```

### 7. Pending Operations
```python
async def _add_pending_operations(
    self,
    grpc_request: orchestrator_pb2.ChatRequest,
    state: Optional[Dict[str, Any]]
) -> None:
    """
    Extract pending_operations from state
    Build PendingOperationInfo proto messages
    """
```

### 8. Routing Locks
```python
async def _add_routing_locks(
    self,
    grpc_request: orchestrator_pb2.ChatRequest,
    request_context: Dict[str, Any],
    state: Optional[Dict[str, Any]]
) -> None:
    """
    Check request_context and shared_memory for locked_agent
    Add to proto if found
    """
```

### 9. Checkpointing
```python
async def _add_checkpoint_info(
    self,
    grpc_request: orchestrator_pb2.ChatRequest,
    request_context: Dict[str, Any]
) -> None:
    """
    Check for base_checkpoint_id in request_context
    Add to proto if present
    """
```

## Logging and Debugging

The context gatherer provides comprehensive logging:

```
🔧 CONTEXT GATHERER: Building gRPC request for user uuid-123
✅ CONTEXT: Added 15 messages to history
✅ CONTEXT: Added persona (ai_name=Kodex)
✅ CONTEXT: Added editor context (file=chapter1.md, type=fiction, 5000 chars)
✅ CONTEXT: Added permission grants (web_search, file_write)
🎯 CONTEXT GATHERER: Explicit routing to data_formatting
📦 CONTEXT SUMMARY: history(15), persona, editor(chapter1.md), permissions, route(data_formatting)
```

### Context Summary Format

The final log line summarizes what context was included:

- `history(N)` - N conversation messages
- `persona` - User persona included
- `editor(filename)` - Editor context for filename
- `pipeline` - Pipeline context included
- `permissions` - Permission grants included
- `pending_ops(N)` - N pending operations
- `locked(agent)` - Conversation locked to agent
- `route(agent_type)` - Explicit routing specified

## Testing Context Gathering

### Unit Testing

```python
from services.grpc_context_gatherer import get_context_gatherer

# Test with minimal context
request = await gatherer.build_chat_request(
    query="Hello",
    user_id="test-user",
    conversation_id="test-conv"
)
assert request.user_id == "test-user"
assert request.query == "Hello"

# Test with editor context
request_context = {
    "active_editor": {
        "is_editable": True,
        "filename": "test.md",
        "content": "# Test",
        "frontmatter": {"type": "fiction"}
    },
    "editor_preference": "prefer"
}

request = await gatherer.build_chat_request(
    query="Proofread this",
    user_id="test-user",
    conversation_id="test-conv",
    request_context=request_context
)
assert request.HasField("active_editor")
assert request.active_editor.filename == "test.md"
```

### Integration Testing

```bash
# Test via gRPC proxy endpoint
curl -X POST http://localhost:8000/api/async/orchestrator/grpc/stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Proofread chapter 1",
    "conversation_id": "test-conv",
    "active_editor": {
      "is_editable": true,
      "filename": "chapter1.md",
      "content": "# Chapter 1\nContent here...",
      "frontmatter": {"type": "fiction", "title": "Test"}
    },
    "editor_preference": "prefer"
  }'
```

Check logs for:
```
✅ CONTEXT: Added editor context (file=chapter1.md, type=fiction, 25 chars)
```

## Migration Roadmap

### Phase 1: Foundation (✅ COMPLETE)
- Create `GRPCContextGatherer` service
- Update `grpc_orchestrator_proxy` to use gatherer
- Extend `OrchesterRequest` with context fields

### Phase 2: State Integration (🔄 IN PROGRESS)
- Fetch conversation state from database
- Pass state to context gatherer
- Enable conversation history transmission

### Phase 3: Frontend Integration (📋 PLANNED)
- Update frontend to send editor context
- Update frontend to send pipeline context
- Add routing lock UI

### Phase 4: Advanced Context (📋 PLANNED)
- Implement conversation intelligence caching
- Add topic continuity tracking
- Optimize context window usage

## Best Practices

### 1. **Conditional Inclusion**
Only send context when relevant. Don't send editor context when user is on documents page.

### 2. **Size Limits**
- Conversation history: Last 20 messages
- Editor content: Full file (agents can truncate if needed)
- Pipeline variables: All (usually small)

### 3. **Fallback Handling**
If context gathering fails for any field, log warning and continue with partial context.

### 4. **Privacy**
Never log full editor content or conversation messages. Use truncated summaries in logs.

### 5. **Performance**
Context gathering should complete in <100ms. Use async operations and caching where appropriate.

## Troubleshooting

### Context Not Appearing in llm-orchestrator

**Check:**
1. Frontend sending correct fields in request
2. Backend extracting request_context properly
3. Context gatherer methods not throwing exceptions
4. Proto fields correctly populated
5. gRPC channel transmitting data

**Debug:**
```python
# In grpc_context_gatherer.py
logger.info(f"REQUEST CONTEXT: {request_context}")
logger.info(f"STATE KEYS: {list(state.keys()) if state else 'None'}")
```

### Editor Context Not Recognized

**Verify:**
- `is_editable = true`
- `filename` ends with `.md`
- `editor_preference != "ignore"`
- `frontmatter` is valid dict

### Permission Grants Not Passing

**Verify:**
- LangGraph state has `shared_memory` dict
- Permission keys use exact naming: `web_search_permission`, `web_crawl_permission`, etc.
- Values are `True` (boolean, not string)

## Future Enhancements

### Planned Features

1. **Conversation Intelligence Caching**
   - Cache research results between queries
   - Track topic transitions
   - Avoid redundant web searches

2. **Context Window Optimization**
   - Smart truncation of large editor files
   - Summarization of long conversation histories
   - Priority-based context inclusion

3. **Multi-Modal Context**
   - Image context from clipboard
   - Audio transcription context
   - Document extraction context

4. **Performance Monitoring**
   - Context gathering latency tracking
   - Context size metrics
   - Cache hit rates

## Related Documentation

- [LangGraph Agent Migration Guide](LANGGRAPH_AGENT_MIGRATION_GUIDE.md)
- [LangGraph Context Analysis](LANGGRAPH_CONTEXT_ANALYSIS.md)
- [LangGraph Best Practices](.cursor/rules/agent-architecture-patterns.mdc)
- [orchestrator.proto](/opt/bastion/protos/orchestrator.proto)

## Summary

The gRPC Context Gatherer provides a **robust, centralized infrastructure** for assembling comprehensive user context and transmitting it to the `llm-orchestrator`. This ensures:

- ✅ **Consistency** - All requests use same context gathering logic
- ✅ **Completeness** - No context lost in migration
- ✅ **Flexibility** - Easy to add new context types
- ✅ **Testability** - Clear interfaces for unit testing
- ✅ **Maintainability** - Single source of truth for context logic

When migrating additional agents, this infrastructure ensures they have full access to user context without backend LangGraph dependencies.

