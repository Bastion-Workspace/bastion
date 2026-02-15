# Proto Context Usage Guide

## Overview

The `orchestrator.proto` `ChatRequest` message supports comprehensive context via **conditional optional fields**. Backend sends only relevant context based on the user's current page/state.

---

## Core Fields (Always Send)

```python
request = orchestrator_pb2.ChatRequest(
    user_id=user_id,
    conversation_id=conversation_id,
    query=query,
    session_id=session_id or "default"
)
```

---

## Conversation History (Recommended)

**When**: Always (provides conversation context)

```python
# Convert LangGraph messages to proto messages
history = []
for msg in conversation_messages[-20:]:  # Last 20 messages
    history.append(orchestrator_pb2.ConversationMessage(
        role="user" if msg.type == "human" else "assistant",
        content=msg.content,
        timestamp=datetime.now().isoformat()
    ))

request.conversation_history.extend(history)
```

**Benefits**: Agents understand conversation context, follow-ups, clarifications

---

## User Persona (Recommended)

**When**: Always (personalization)

```python
# Get user settings
user_settings = await prompt_service.get_user_settings_for_service(user_id)

if user_settings:
    request.persona.CopyFrom(orchestrator_pb2.UserPersona(
        ai_name=user_settings.ai_name or "Kodex",
        persona_style=user_settings.persona_style.value,
        political_bias=user_settings.political_bias.value,
        timezone=user_settings.timezone or "UTC"
    ))
```

**Benefits**: Personalized responses matching user preferences

---

## Active Editor Context (Conditional)

**When**: User on editor page with markdown file open

**Check**: `request.active_editor is not None`

```python
# Frontend sends active_editor in request
if request.active_editor and request.editor_preference != "ignore":
    # Validate: must be .md file and editable
    if (request.active_editor.get('is_editable') and 
        request.active_editor.get('filename', '').endswith('.md')):
        
        # Parse frontmatter
        frontmatter_data = request.active_editor.get('frontmatter', {})
        frontmatter = orchestrator_pb2.EditorFrontmatter(
            type=frontmatter_data.get('type', ''),
            title=frontmatter_data.get('title', ''),
            author=frontmatter_data.get('author', ''),
            tags=frontmatter_data.get('tags', []),
            status=frontmatter_data.get('status', '')
        )
        
        # Add custom fields
        for key, value in frontmatter_data.items():
            if key not in ['type', 'title', 'author', 'tags', 'status']:
                frontmatter.custom_fields[key] = str(value)
        
        # Set active editor
        grpc_request.active_editor.CopyFrom(orchestrator_pb2.ActiveEditor(
            is_editable=True,
            filename=request.active_editor['filename'],
            language=request.active_editor.get('language', 'markdown'),
            content=request.active_editor.get('content', ''),
            content_length=len(request.active_editor.get('content', '')),
            frontmatter=frontmatter,
            editor_preference=request.editor_preference or "prefer"
        ))
```

**Used By**: 
- `fiction_editing_agent`
- `proofreading_agent`
- `story_analysis_agent`
- `character_development_agent`
- `rules_editing_agent`
- `outline_editing_agent`

**Benefits**: Agents can read/edit active document, understand context

---

## Pipeline Context (Conditional)

**When**: User on pipeline execution page (`/pipelines/:id`)

**Check**: `request.active_pipeline_id is not None`

```python
if request.active_pipeline_id and request.pipeline_preference != "ignore":
    grpc_request.pipeline_context.CopyFrom(orchestrator_pb2.PipelineContext(
        pipeline_preference=request.pipeline_preference or "prefer",
        active_pipeline_id=request.active_pipeline_id,
        pipeline_name=pipeline_name or "",  # Optional: fetch from DB
    ))
    
    # Optional: Add template variables
    if pipeline_variables:
        for key, value in pipeline_variables.items():
            grpc_request.pipeline_context.pipeline_variables[key] = str(value)
```

**Used By**:
- `pipeline_agent` (template execution)

**Benefits**: Pipeline agent knows which templates to use

---

## Permission Grants (Conditional)

**When**: User has granted permissions in conversation

**Check**: Permissions exist in shared_memory

```python
shared_memory = state.get("shared_memory", {})

if any([
    shared_memory.get("web_search_permission"),
    shared_memory.get("web_crawl_permission"),
    shared_memory.get("file_write_permission")
]):
    grpc_request.permission_grants.CopyFrom(orchestrator_pb2.PermissionGrants(
        web_search_permission=shared_memory.get("web_search_permission", False),
        web_crawl_permission=shared_memory.get("web_crawl_permission", False),
        file_write_permission=shared_memory.get("file_write_permission", False),
        external_api_permission=shared_memory.get("external_api_permission", False)
    ))
```

**Used By**:
- `research_agent` (web search)
- `site_crawl_agent` (web crawl)
- `website_crawler_agent` (web crawl)

**Benefits**: Agents know what operations are permitted without asking again

---

## Pending Operations (Conditional)

**When**: Operations awaiting user approval

**Check**: `len(pending_operations) > 0`

```python
pending_ops = state.get("pending_operations", [])

for op in pending_ops:
    grpc_request.pending_operations.append(orchestrator_pb2.PendingOperationInfo(
        id=op.get("id", ""),
        type=op.get("type", ""),
        summary=op.get("summary", ""),
        permission_required=op.get("permission_required", False),
        status=op.get("status", "pending"),
        created_at=op.get("created_at", datetime.now().isoformat())
    ))
```

**Used By**: All agents (for context awareness)

**Benefits**: Agents can mention pending operations, suggest approvals

---

## Conversation Intelligence (Conditional - Performance Optimization)

**When**: Relevant cached results exist

**Check**: Intelligence cache has entries relevant to query

```python
# Only send relevant cached results (not entire cache)
intelligence = state.get("conversation_intelligence", {})
cached_results = intelligence.get("cached_results", {})

# Filter relevant results (e.g., by topic similarity)
relevant_results = filter_relevant_cached_results(cached_results, query)

if relevant_results:
    intel_msg = orchestrator_pb2.ConversationIntelligence(
        last_updated=intelligence.get("last_updated", datetime.now().isoformat())
    )
    
    for result_hash, result_data in relevant_results.items():
        intel_msg.cached_results.append(orchestrator_pb2.CachedResult(
            content_hash=result_hash,
            content=result_data.get("content", ""),
            result_type=result_data.get("result_type", ""),
            source_agent=result_data.get("source_agent", ""),
            timestamp=result_data.get("timestamp", ""),
            confidence_score=result_data.get("confidence_score", 0.0),
            topics=result_data.get("topics", []),
            citations=result_data.get("citations", [])
        ))
    
    grpc_request.conversation_intelligence.CopyFrom(intel_msg)
```

**Used By**: All agents (for avoiding redundant work)

**Benefits**: 
- Avoid re-researching same topics
- Faster responses using cached results
- Better conversation continuity

**⚠️ Warning**: Can make requests large - filter to relevant results only!

---

## Routing Locks (Conditional)

**When**: Conversation locked to specific agent

**Check**: `locked_agent` in shared_memory or request

```python
locked_agent = shared_memory.get("locked_agent") or request.locked_agent

if locked_agent:
    grpc_request.locked_agent = locked_agent
```

**Used By**: Orchestrator (for routing decisions)

**Benefits**: Forces all queries in conversation to specific agent (e.g., wargaming session)

---

## Agent Type Routing (Optional)

**When**: Backend knows which agent should handle query

```python
# After intent classification in backend
if intent_result:
    grpc_request.agent_type = intent_result.get("agent_type", "auto")
    grpc_request.routing_reason = intent_result.get("reasoning", "")
```

**Benefits**: Explicit routing bypasses llm-orchestrator intent classification

---

## Usage Patterns by Page Type

### Chat Page (Default)
```python
✅ user_id, conversation_id, query, session_id
✅ conversation_history (last 20 messages)
✅ persona
❌ active_editor
❌ pipeline_context
✅ permission_grants (if any exist)
✅ pending_operations (if any exist)
❌ conversation_intelligence (usually not needed)
❌ locked_agent (unless set)
```

### Editor Page (`/editor/chapter1.md`)
```python
✅ user_id, conversation_id, query, session_id
✅ conversation_history
✅ persona
✅ active_editor (with full content and frontmatter!)
❌ pipeline_context
✅ permission_grants (if any exist)
✅ pending_operations (if any exist)
❌ conversation_intelligence
❌ locked_agent (unless set)
```

### Pipeline Page (`/pipelines/123`)
```python
✅ user_id, conversation_id, query, session_id
✅ conversation_history
✅ persona
❌ active_editor
✅ pipeline_context (with pipeline_id and variables)
✅ permission_grants (if any exist)
✅ pending_operations (if any exist)
❌ conversation_intelligence
❌ locked_agent (unless set)
```

### Wargaming Session (Locked Agent)
```python
✅ user_id, conversation_id, query, session_id
✅ conversation_history (critical for session continuity)
✅ persona
❌ active_editor
❌ pipeline_context
✅ permission_grants (if any exist)
✅ pending_operations (if any exist)
❌ conversation_intelligence
✅ locked_agent = "wargaming_agent"
```

---

## Backend Implementation Example

```python
async def build_grpc_request(
    query: str,
    user_id: str,
    conversation_id: str,
    request_context: RequestContext
) -> orchestrator_pb2.ChatRequest:
    """
    Build gRPC request with appropriate context based on page/state
    
    Args:
        request_context: Contains active_editor, pipeline_context, etc. from frontend
    """
    grpc_request = orchestrator_pb2.ChatRequest(
        user_id=user_id,
        conversation_id=conversation_id,
        query=query,
        session_id=request_context.session_id or "default"
    )
    
    # Always add conversation history
    history = await load_conversation_history(conversation_id, user_id)
    for msg in history[-20:]:
        grpc_request.conversation_history.append(
            orchestrator_pb2.ConversationMessage(
                role=msg.role,
                content=msg.content,
                timestamp=msg.timestamp
            )
        )
    
    # Always add persona
    persona = await load_user_persona(user_id)
    if persona:
        grpc_request.persona.CopyFrom(persona)
    
    # Conditionally add editor context
    if request_context.active_editor:
        grpc_request.active_editor.CopyFrom(
            build_editor_context(request_context.active_editor)
        )
    
    # Conditionally add pipeline context
    if request_context.active_pipeline_id:
        grpc_request.pipeline_context.CopyFrom(
            build_pipeline_context(request_context.active_pipeline_id)
        )
    
    # Conditionally add permissions
    permissions = await load_active_permissions(conversation_id)
    if permissions:
        grpc_request.permission_grants.CopyFrom(permissions)
    
    # Conditionally add pending operations
    pending_ops = await load_pending_operations(conversation_id)
    for op in pending_ops:
        grpc_request.pending_operations.append(op)
    
    # Conditionally add locked agent
    if request_context.locked_agent:
        grpc_request.locked_agent = request_context.locked_agent
    
    return grpc_request
```

---

## llm-orchestrator Usage

The llm-orchestrator receives the request and checks which fields are populated:

```python
async def StreamChat(self, request, context):
    """Handle incoming chat request with conditional context"""
    
    # Core fields (always present)
    user_id = request.user_id
    query = request.query
    
    # Check optional context
    has_editor = request.HasField("active_editor")
    has_pipeline = request.HasField("pipeline_context")
    has_permissions = request.HasField("permission_grants")
    
    # Route based on context
    if has_editor and request.active_editor.frontmatter.type == "fiction":
        # Route to fiction editing agent
        agent = self.fiction_editing_agent
    elif has_pipeline:
        # Route to pipeline agent
        agent = self.pipeline_agent
    elif request.agent_type:
        # Explicit routing
        agent = self._get_agent_by_type(request.agent_type)
    else:
        # Run intent classification with full context
        intent = await self.intent_classifier.classify(
            query=query,
            conversation_history=list(request.conversation_history),
            persona=request.persona,
            permissions=request.permission_grants
        )
        agent = self._route_by_intent(intent)
    
    # Pass relevant context to agent
    result = await agent.process(
        query=query,
        conversation_history=list(request.conversation_history),
        active_editor=request.active_editor if has_editor else None,
        permissions=request.permission_grants if has_permissions else None,
        # ... other context
    )
```

---

## Message Size Considerations

### Small Messages (<1KB)
- Chat page, simple queries
- No editor content, no intelligence cache

### Medium Messages (1-10KB)
- With conversation history (20 messages)
- With persona and permissions

### Large Messages (10-100KB)
- With active editor content (markdown files)
- With conversation intelligence cache

### Very Large Messages (100KB+)
- Long editor content (novels, large documents)
- Extensive conversation intelligence

**Optimization**: Filter conversation intelligence to relevant results only!

---

## Proto3 Optional Fields

Proto3 fields are optional by default. Check if field is set:

**Python**:
```python
if request.HasField("active_editor"):
    # Editor context provided
    filename = request.active_editor.filename
```

**Go**:
```go
if request.ActiveEditor != nil {
    // Editor context provided
}
```

**Note**: Repeated fields (`conversation_history`, `pending_operations`) are never nil, just empty lists.

---

## Future Extensions

The proto supports future extensions via `custom_fields` maps:

- `UserPersona.custom_preferences`
- `EditorFrontmatter.custom_fields`
- `PermissionGrants.custom_permissions`
- `PendingOperationInfo.metadata`

Add new context without changing proto structure!

