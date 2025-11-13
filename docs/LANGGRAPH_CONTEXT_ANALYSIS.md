# LangGraph Context Analysis - What Gets Passed to Agents

## Current Backend → LangGraph Flow

### Entry Point: `process_user_query()`

```python
async def process_user_query(
    user_message: str,
    user_id: str,
    conversation_id: str,
    persona: Optional[Dict[str, Any]] = None,
    extra_shared_memory: Optional[Dict[str, Any]] = None,
    base_checkpoint_id: Optional[str] = None
)
```

---

## ConversationState Structure

### Core Identity Fields
```python
user_id: str                    # User UUID
conversation_id: str            # Conversation UUID
messages: List[BaseMessage]     # Full conversation history (LangChain messages)
```

### Agent Coordination Fields
```python
current_query: Optional[str]             # Current user query being processed
active_agent: Optional[str]              # Which agent is currently active
shared_memory: Dict[str, Any]            # LangGraph shared memory for agent communication
```

### Agent Results Fields
```python
intent_classification: Optional[Dict[str, Any]]  # Intent classification results
agent_results: Optional[Dict[str, Any]]          # Agent processing results
latest_response: Optional[str]                   # Latest agent response
chat_agent_response: Optional[str]               # Chat agent specific response
```

### HITL/Editor Operations Fields
```python
editor_operations: Optional[List[Dict[str, Any]]]  # Structured editor operations for frontend
manuscript_edit: Optional[Dict[str, Any]]          # ManuscriptEdit metadata for editing operations
```

### Flow Control Fields
```python
requires_user_input: Optional[bool]      # True when waiting for HITL response
is_complete: Optional[bool]              # True when conversation turn complete
error_state: Optional[str]               # Error information if any
```

### Context Fields
```python
persona: Optional[Dict[str, Any]]        # User persona for personalization
conversation_topic: Optional[str]        # Current conversation topic
```

### Conversation Intelligence Fields
```python
conversation_intelligence: Optional[Dict[str, Any]]  # Built-in conversation context and cache
```

### Conversation Metadata (LangGraph-native)
```python
conversation_title: Optional[str]        # Generated conversation title
conversation_created_at: Optional[str]   # ISO timestamp when conversation started
conversation_updated_at: Optional[str]   # ISO timestamp of last activity
conversation_tags: Optional[List[str]]   # User-defined tags
conversation_description: Optional[str]  # Optional description
is_pinned: Optional[bool]                # Whether conversation is pinned
is_archived: Optional[bool]              # Whether conversation is archived
```

### Agent Autonomy Support Fields
```python
active_agent_checkpoint: Optional[Dict[str, Any]]      # Active agent HITL checkpoint info
orchestrator_routing_decision: Optional[Dict[str, Any]]  # Temporary routing decision
agent_insights: Dict[str, Any]           # Agent intelligence data
pending_operations: List[Dict[str, Any]]  # Pending operations list
pending_permission_requests: List[Dict[str, Any]]  # Pending permission requests
```

---

## shared_memory Structure

The `shared_memory` dict contains agent communication data:

### Always Present
```python
{
    "search_results": {},           # Results from search operations
    "conversation_context": {},     # Context for ongoing conversation
    "agent_handoffs": [],          # Record of agent handoffs
    "data_sufficiency": {}         # Assessment of available data
}
```

### Conditionally Added via `extra_shared_memory`

#### Active Editor (Fiction Editing, etc.)
```python
"active_editor": {
    "is_editable": bool,
    "filename": str,              # e.g. "chapter1.md"
    "language": str,              # e.g. "markdown"
    "content": str,               # Full file content
    "content_length": int,        # Length of content
    "frontmatter": {              # YAML frontmatter parsed
        "type": str,              # e.g. "fiction", "article"
        "title": str,
        "author": str,
        # ... any other frontmatter fields
    }
}
```

**When**: User has editor open with markdown file
**Used by**: `fiction_editing_agent`, `rules_editing_agent`, `character_development_agent`, `outline_editing_agent`, `proofreading_agent`, `story_analysis_agent`

#### Locked Agent (Conversation Routing Lock)
```python
"locked_agent": str  # e.g. "wargaming_agent", "fiction_editing_agent"
```

**When**: User explicitly locks conversation to specific agent
**Used by**: Orchestrator for routing decisions

#### Pipeline Context
```python
"pipeline_preference": str        # "prefer" | "ignore"
"active_pipeline_id": str         # Pipeline UUID when on /pipelines/:id page
```

**When**: User is on pipeline execution page
**Used by**: `pipeline_agent` for template execution

#### Permission Flags
```python
"web_search_permission": bool     # True when user granted web search
"web_crawl_permission": bool      # True when user granted web crawling
```

**When**: User grants permissions for HITL operations
**Used by**: `research_agent`, `site_crawl_agent`, `website_crawler_agent`

#### Research Context
```python
"research_findings": Dict[str, Any]  # Previous research results
"search_results": {
    "local": {...},
    "web": {...}
}
```

**When**: Research agent has completed searches
**Used by**: All agents for context awareness

---

## persona Structure

User personalization settings:

```python
{
    "ai_name": str,              # e.g. "Kodex", "Claude"
    "persona_style": str,        # "professional" | "casual" | "technical"
    "political_bias": str        # "neutral" | "liberal" | "conservative"
}
```

**Source**: `prompt_service.get_user_settings_for_service(user_id)`
**Used by**: All agents for response style personalization

---

## conversation_intelligence Structure

Roosevelt's Conversation Intelligence cache:

```python
{
    # CACHED RESULTS ARCHIVE
    "cached_results": {
        "<content_hash>": {
            "content": str,
            "result_type": "research_findings" | "chat_output" | "web_sources" | "local_documents",
            "source_agent": str,
            "timestamp": str (ISO),
            "confidence_score": float (0.0-1.0),
            "topics": List[str],
            "citations": List[str]
        }
    },
    
    # TOPIC CONTINUITY TRACKING
    "current_topic": Optional[str],
    "topic_history": List[str],
    "topic_transitions": List[Dict[str, Any]],
    
    # AGENT COLLABORATION HISTORY
    "agent_outputs": {
        "<agent_name>": List[str]  # Recent outputs by agent type
    },
    "collaboration_suggestions": List[Dict[str, Any]],
    
    # RESEARCH INTELLIGENCE CACHE
    "research_cache": {
        "<query_hash>": Dict[str, Any]  # Cached research results
    },
    "source_cache": {
        "<url>": Dict[str, Any]  # Cached web source content
    },
    
    # COVERAGE ANALYSIS
    "query_coverage_cache": {
        "<pattern>": float  # Pre-computed coverage scores
    },
    
    # METADATA
    "last_updated": str (ISO),
    "intelligence_version": str
}
```

**Used by**: All agents for conversation continuity and avoiding redundant work

---

## What Agents Actually Receive

When orchestrator calls `agent.process(agent_state)`, agents receive:

```python
{
    "messages": List[BaseMessage],      # Conversation history (last 20)
    "user_id": str,                     # For personalized tool access (Qdrant user collections)
    "current_query": str,               # Latest user message
    "persona": Optional[Dict],          # User persona settings
    "active_agent": str,                # Current agent name
    "shared_memory": Dict[str, Any],    # Full shared memory (including active_editor, permissions, etc.)
    "conversation_topic": Optional[str] # Current conversation topic
}
```

**NOTE**: Agents do NOT receive:
- `conversation_id` (they don't need it directly)
- `intent_classification` (already used for routing)
- `agent_results` (from previous agents, available in shared_memory)
- Flow control flags (`is_complete`, `requires_user_input`)
- Metadata fields (titles, tags, etc.)

---

## Additional Context Agents Extract

### From `shared_memory`
- Agents check `shared_memory["active_editor"]` for editor context
- Agents check `shared_memory["web_search_permission"]` before web tools
- Agents check `shared_memory["research_findings"]` for previous research
- Agents check `shared_memory["locked_agent"]` for routing lock status

### From `state.get("messages")`
- Last 10-20 messages for conversation context
- Conversation history patterns (follow-ups, clarifications)
- Previous agent responses (AIMessage type)
- User requests (HumanMessage type)

### From `conversation_intelligence`
- Cached research results to avoid duplication
- Topic continuity for contextual responses
- Previous agent collaboration patterns
- Coverage analysis for gap identification

---

## Conditional Context (Page-Specific)

### On Editor Page (e.g. `/editor/chapter1.md`)
```python
extra_shared_memory = {
    "active_editor": {
        "filename": "chapter1.md",
        "content": "...",
        "frontmatter": {"type": "fiction", ...}
    }
}
```

**Enables**: Fiction editing, proofreading, story analysis

### On Pipeline Page (e.g. `/pipelines/123`)
```python
extra_shared_memory = {
    "pipeline_preference": "prefer",
    "active_pipeline_id": "uuid-123"
}
```

**Enables**: Template execution, pipeline agent activation

### On Chat Page (default)
```python
extra_shared_memory = None  # No special context
```

**Enables**: All general agents (research, chat, weather, etc.)

---

## Proto Design Implications

### Required Proto Fields

**Always Send:**
- `user_id`
- `conversation_id`
- `query`
- `session_id`
- `conversation_history` (list of messages)
- `persona` (user preferences)

**Conditionally Send:**
- `active_editor` (when on editor page)
- `pipeline_context` (when on pipeline page)
- `permission_grants` (when user has granted permissions)
- `conversation_intelligence` (cached results)
- `locked_agent` (when conversation locked to agent)

### Proto Optimization

**Don't Send Every Time:**
- Full `conversation_intelligence` (too large, cache on llm-orchestrator side)
- Conversation metadata (titles, tags - not needed for agent processing)
- Internal flow control flags

**Send as Needed:**
- Editor content only when on editor page
- Pipeline ID only when on pipeline page
- Permission grants only when permissions exist
- Research findings only when relevant to query

---

## Next Steps

1. **Design Proto Messages** that capture all required context
2. **Implement Context Gathering** in backend API layer
3. **Update llm-orchestrator** to receive and use full context
4. **Test Conditional Context** for different page types
5. **Optimize Payload Size** by only sending relevant context

