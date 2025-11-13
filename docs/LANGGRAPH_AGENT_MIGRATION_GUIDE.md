# LangGraph Agent Migration Guide

## Overview

This guide documents the pattern for migrating LangGraph agents from the backend container to the llm-orchestrator microservice via gRPC.

**Benefits of Migration:**
- **Separation of Concerns**: Agents run in dedicated microservice
- **Scalability**: Orchestrator can scale independently
- **Resource Efficiency**: Backend focuses on data access, orchestrator on LLM tasks
- **Gradual Migration**: Feature-flag controlled rollout

---

## Migration Steps

### 1. Extend Proto Definition

✅ **COMPLETE**: Proto already supports comprehensive context!

See `/opt/bastion/protos/orchestrator.proto` for full structure including:
- Conversation history
- User persona  
- Active editor context (fiction editing)
- Pipeline context (template execution)
- Permission grants (HITL)
- Pending operations
- Conversation intelligence (performance cache)

⭐ **CONTEXT GATHERING**: The backend automatically gathers and sends all relevant context via `/opt/bastion/backend/services/grpc_context_gatherer.py`.

**Documentation**:
- [Context Gathering Infrastructure](LANGGRAPH_CONTEXT_GATHERING.md) - **Read this for full details!**
- [Context Analysis](LANGGRAPH_CONTEXT_ANALYSIS.md) - Original requirements

### 2. Create Agent in llm-orchestrator

**Location**: `llm-orchestrator/orchestrator/agents/<agent_name>.py`

**Base Pattern**:

```python
from .base_agent import BaseAgent, TaskStatus
from typing import Dict, Any, List, Optional

class MyNewAgent(BaseAgent):
    """Agent description"""
    
    def __init__(self):
        super().__init__("my_new_agent")
    
    def _build_prompt(self, context: str) -> str:
        """Build system prompt"""
        return f"""System prompt with structured output requirement...
        
STRUCTURED OUTPUT:
{{
    "response": "Your response",
    "task_status": "complete"
}}
"""
    
    async def process(self, query: str, metadata: Dict[str, Any] = None, messages: List[Any] = None) -> Dict[str, Any]:
        """Process agent request"""
        try:
            # Extract metadata
            metadata = metadata or {}
            
            # Build prompt
            system_prompt = self._build_prompt("")
            
            # Get conversation history
            conversation_history = []
            if messages:
                conversation_history = self._extract_conversation_history(messages, limit=10)
            
            # Build messages for LLM
            llm_messages = self._build_messages(system_prompt, query, conversation_history)
            
            # Call LLM
            llm = self._get_llm(temperature=0.7)
            response = await llm.ainvoke(llm_messages)
            
            # Parse structured response
            response_content = response.content if hasattr(response, 'content') else str(response)
            structured_response = self._parse_json_response(response_content)
            
            # Build result
            return {
                "response": structured_response.get("response", response_content),
                "task_status": structured_response.get("task_status", "complete"),
                "agent_type": "my_new_agent",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Agent error: {e}")
            return self._create_error_response(str(e))
```

**Key Points**:
- Extend `BaseAgent` from `llm-orchestrator/orchestrator/agents/base_agent.py`
- Use `self._get_llm()` for LLM access
- Parse structured responses with `self._parse_json_response()`
- Return standardized dictionary with `response`, `task_status`, `agent_type`

### 3. Register Agent in __init__.py

Update `llm-orchestrator/orchestrator/agents/__init__.py`:

```python
from orchestrator.agents.my_new_agent import MyNewAgent

__all__ = [
    # ... existing agents ...
    'MyNewAgent'
]
```

### 4. Update gRPC Service Routing

Update `llm-orchestrator/orchestrator/grpc_service.py`:

**Add to __init__:**
```python
def __init__(self):
    # ... existing agents ...
    self.my_new_agent = None
```

**Add to _ensure_agents_loaded:**
```python
if self.my_new_agent is None:
    self.my_new_agent = MyNewAgent()
    logger.info("✅ My new agent loaded")
```

**Add routing in StreamChat:**
```python
elif agent_type == "my_new_agent":
    yield orchestrator_pb2.ChatChunk(
        type="status",
        message="My new agent processing...",
        timestamp=datetime.now().isoformat(),
        agent_name="orchestrator"
    )
    
    result = await self.my_new_agent.process(
        query=request.query,
        metadata=metadata,
        messages=messages
    )
    
    yield orchestrator_pb2.ChatChunk(
        type="content",
        message=result.get("response", "No response generated"),
        timestamp=datetime.now().isoformat(),
        agent_name="my_new_agent"
    )
    
    yield orchestrator_pb2.ChatChunk(
        type="complete",
        message=f"Agent complete (status: {result.get('task_status', 'complete')})",
        timestamp=datetime.now().isoformat(),
        agent_name="system"
    )
```

**Update HealthCheck:**
```python
return orchestrator_pb2.HealthCheckResponse(
    status="healthy",
    details={
        "agents": "research,chat,data_formatting,my_new_agent",  # Add new agent
        # ... other details ...
    }
)
```

### 5. Backend Integration (Optional)

If you want the backend to route to the new agent automatically, update the backend orchestrator's intent classification to recognize the new agent type.

**Backend stays as-is if using explicit agent routing via API.**

---

## Testing Procedure

### 1. Rebuild Containers

```bash
docker compose build llm-orchestrator backend
docker compose up -d
```

### 2. Test via gRPC Proxy

```bash
curl -X POST http://localhost:8000/api/async/orchestrator/grpc/stream \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "query": "test query",
    "conversation_id": "test-123",
    "agent_type": "my_new_agent"
  }'
```

### 3. Check Logs

```bash
docker logs codex-dev-llm-orchestrator -f
docker logs codex-dev-backend -f
```

Expected output:
- llm-orchestrator: "My new agent loaded"
- llm-orchestrator: "Agent type requested: my_new_agent"
- llm-orchestrator: "My new agent processing..."

---

## Migration Checklist

- [ ] Proto definition updated with agent_type
- [ ] Agent created in llm-orchestrator/orchestrator/agents/
- [ ] Agent extends BaseAgent
- [ ] Agent registered in __init__.py
- [ ] gRPC service routing added
- [ ] gRPC service health check updated
- [ ] Containers rebuilt
- [ ] Agent tested via gRPC proxy
- [ ] Backend orchestrator updated (if automatic routing needed)
- [ ] Original backend agent deprecated/removed (optional, gradual)

---

## Completed Migrations

| Agent | Date | Status | Notes |
|-------|------|--------|-------|
| ResearchAgent | 2025-11-09 | ✅ Complete | Full multi-round research with web search |
| ChatAgent | 2025-11-10 | ✅ Complete | General conversation handler |
| DataFormattingAgent | 2025-11-10 | ✅ Complete | Table/timeline formatting specialist |

---

## Feature Flags

**Backend Config** (`backend/config.py`):

```python
# Feature flag for routing specific agents via gRPC
USE_GRPC_FOR_CHAT = False  # Set True to route chat via gRPC
USE_GRPC_FOR_FORMATTING = False  # Set True to route formatting via gRPC
```

**Usage in Backend Orchestrator:**

```python
if settings.USE_GRPC_FOR_CHAT and intent == "chat":
    # Route to gRPC orchestrator
    return await self._route_via_grpc(query, conversation_id, agent_type="chat")
else:
    # Use local backend agent
    return await self.chat_agent.process(state)
```

---

## Differences Between Backend and llm-orchestrator Agents

| Feature | Backend Agent | llm-orchestrator Agent |
|---------|--------------|------------------------|
| State Management | LangGraph StateGraph | Simple dictionary |
| Service Access | ServiceContainer | Direct LLM via ChatOpenAI |
| Tool Registry | Centralized registry | gRPC backend tools |
| Dependencies | Heavy (service container, repos) | Light (LLM, proto) |
| Complexity | High (workflows, nodes, edges) | Low (direct processing) |
| Best For | Complex multi-step workflows | Simple request-response patterns |

---

## Troubleshooting

### Proto Generation Errors

```
ModuleNotFoundError: No module named 'protos.orchestrator_pb2'
```

**Fix**: Rebuild containers to regenerate proto files:
```bash
docker compose build llm-orchestrator backend --no-cache
```

### Agent Not Loading

Check llm-orchestrator logs:
```bash
docker logs codex-dev-llm-orchestrator -f | grep "agent loaded"
```

Expected: "✅ My new agent loaded"

### gRPC Connection Refused

Check llm-orchestrator is running:
```bash
docker ps | grep llm-orchestrator
```

Test health check:
```bash
curl http://localhost:8000/api/async/orchestrator/grpc/health
```

---

## Successfully Migrated Agents ✅

### 1. ResearchAgent (FullResearchAgent)
- **Complexity**: Multi-round sophisticated research with gap analysis
- **Features**: Query expansion, web search, caching, multi-source synthesis
- **Migration Date**: Phase 5

### 2. ChatAgent
- **Complexity**: 524 lines - conversational interactions with context
- **Features**: Conversation history, persona support
- **Migration Date**: Phase 6

### 3. DataFormattingAgent
- **Complexity**: Data structuring and formatting
- **Features**: Multiple output formats (markdown, JSON, tables)
- **Migration Date**: Phase 6

### 4. HelpAgent
- **Complexity**: Application help and documentation
- **Features**: Context-aware help content
- **Migration Date**: Phase 6

### 5. WeatherAgent
- **Complexity**: 450 lines + 3 supporting modules (1,652 lines total)
- **Features**: 
  - Current weather conditions and forecasts
  - OpenWeatherMap API integration
  - LLM-powered location extraction
  - Multiple communication styles (Roosevelt, professional, casual)
  - Collaboration detection (suggests research for travel/activities)
  - Intelligent geocoding with state abbreviation expansion
- **Migration Date**: November 2025
- **Supporting Files**:
  - `weather_agent.py` - Main agent orchestration
  - `weather_tools.py` - OpenWeatherMap API integration (577 lines)
  - `weather_request_analyzer.py` - LLM location extraction (295 lines)
  - `weather_response_formatters.py` - Multi-style formatting (330 lines)

### 6. ImageGenerationAgent
- **Complexity**: 128 lines (simple orchestration)
- **Features**:
  - Image generation via OpenRouter (Gemini, DALL-E)
  - Multiple format support (PNG, JPG)
  - Size customization (512x512 to 2048x2048)
  - Seed control for reproducibility
  - Negative prompts support
  - Persona-aware response formatting
  - Backend file storage and URL serving
- **Migration Date**: November 2025
- **Architecture**:
  - **Agent**: `llm-orchestrator/orchestrator/agents/image_generation_agent.py`
  - **gRPC Tool**: Backend exposes `GenerateImage` via tool service
  - **Service**: Backend `ImageGenerationService` handles OpenRouter API + file I/O
  - **Proto**: Extended `tool_service.proto` with image generation messages
- **Key Pattern**: Hybrid architecture - agent in orchestrator, file operations in backend via gRPC

### 7. FactCheckingAgent
- **Complexity**: 473 lines (medium orchestration)
- **Features**:
  - Automated factual claim extraction from content
  - LLM-powered claim identification (statistics, dates, events, facts)
  - Web search verification for each claim (10 sources per claim)
  - Credibility analysis across multiple sources
  - Structured verification results (verified/disputed/unverified)
  - Confidence scoring (0.0-1.0) per claim
  - Correction suggestions with sources when claims are false
  - Overall verification rate calculation
  - Formatted summary with status indicators (✅❌⚠️)
- **Migration Date**: November 2025
- **Architecture**:
  - **Agent**: `llm-orchestrator/orchestrator/agents/fact_checking_agent.py`
  - **Backend Tool Client**: `orchestrator/clients/backend_tool_client.py` for web search
  - **gRPC Tools**: Uses backend `SearchWeb` RPC for web search
  - **Process**: Extract claims (LLM) → Search web → Analyze results (LLM) → Aggregate
- **Key Pattern**: Pure orchestration - no file I/O, uses gRPC for web search tools

### 8. RSSAgent
- **Complexity**: 632 lines (medium orchestration with NLP parsing)
- **Features**:
  - Natural language RSS feed management commands
  - Add feeds with automatic metadata extraction (title, category)
  - Category suggestion from URL analysis
  - User-specific and global feed support (admin-only for global)
  - List feeds with article counts and last poll times
  - Refresh feeds via Celery background tasks
  - Delete feeds with permission checking
  - Metadata request/response flow for incomplete commands
  - Regex-based command parsing (add, list, refresh patterns)
- **Migration Date**: November 2025
- **Architecture**:
  - **Agent**: `llm-orchestrator/orchestrator/agents/rss_agent.py`
  - **Backend Tool Client**: `orchestrator/clients/backend_tool_client.py` extended with RSS methods
  - **gRPC Tools**: Backend exposes RSS management via tool service
    - `AddRSSFeed` - Create new feeds with permission checking
    - `ListRSSFeeds` - Query feeds with scope filtering
    - `RefreshRSSFeed` - Trigger Celery background polling
    - `DeleteRSSFeed` - Remove feeds with permission
  - **Backend Services**: `RSSService`, `AuthService`, Celery tasks
  - **Process**: Parse command → Execute via gRPC → Format response
- **Key Pattern**: Hybrid architecture - agent orchestration in orchestrator, heavy backend operations (database, Celery) stay in backend
- **Background Tasks**: RSS polling and article processing remain in backend (not user-facing)

### 9. OrgInboxAgent ✨ **NEW**
- **Complexity**: 554 lines (high complexity with LLM interpretation)
- **Features**:
  - Natural language org-mode inbox.org management
  - LLM-powered request interpretation:
    - Pronoun resolution ("add it to my inbox")
    - Entry type classification (todo/event/contact/checkbox)
    - Schedule extraction from natural language
    - Repeater detection (+1w, .+1m)
    - Tag suggestion from conversation context
  - Contact entry support with PROPERTIES drawer
  - Full org-mode formatting compliance
  - 6 core operations: add, list, toggle, update, schedule, archive_done
  - Conversation context-aware processing
  - Persona-based response formatting
- **Migration Date**: November 2025
- **Architecture**:
  - **Agent**: `llm-orchestrator/orchestrator/agents/org_inbox_agent.py`
  - **Backend Tool Client**: `orchestrator/clients/backend_tool_client.py` extended with 9 org inbox methods
  - **gRPC Tools**: Backend exposes comprehensive org inbox management (9 RPCs)
    - `ListOrgInboxItems` - Query all inbox items with metadata
    - `AddOrgInboxItem` - Add todos/events/contacts/checkboxes
    - `ToggleOrgInboxItem` - Toggle DONE status
    - `UpdateOrgInboxItem` - Edit item text
    - `SetOrgInboxSchedule` - Set schedule + repeater
    - `ApplyOrgInboxTags` - Tag management
    - `ArchiveOrgInboxDone` - Archive completed items
    - `AppendOrgInboxText` - Raw org-mode text append
    - `GetOrgInboxPath` - Get user's inbox.org path
  - **Backend Tools**: `org_inbox_tools.py` (526 lines) - Direct file I/O for org-mode files
  - **Process**: LLM interprets → Build org-mode entry → Execute via gRPC → Format response
- **Key Pattern**: Full LLM Intelligence + File I/O via gRPC - Most sophisticated pattern yet
- **LLM Usage**: Two-stage LLM usage:
  1. Request interpretation (pronoun resolution, type classification, metadata extraction)
  2. Persona-based response confirmation formatting

### 10. SubstackAgent ✨ **NEW**
- **Complexity**: 448 lines (medium orchestration with content generation)
- **Features**:
  - Long-form article generation (2000-5000 words)
  - Tweet-sized content generation (280 char limit)
  - Multi-source synthesis (articles, tweets, background)
  - URL content fetching
  - Structured section extraction (Persona, Background, Articles, Tweets)
  - Editor integration with frontmatter (type='substack' or 'blog')
  - Markdown formatting (headers, bold, italics, blockquotes)
  - Persona-aware writing styles
- **Migration Date**: November 2025
- **Architecture**:
  - **Agent**: `llm-orchestrator/orchestrator/agents/substack_agent.py`
  - **Backend Tool Client**: Uses existing `search_web` for URL fetching
  - **No New gRPC RPCs Required**: Uses existing tools
  - **Process**: Extract sections → Fetch URLs → Generate article → Format response
- **Key Pattern**: Pure Orchestration - No backend file I/O, uses existing search tools
- **Editor Detection**: Requires active editor with `type: substack` or `type: blog` in frontmatter

### 11. PodcastScriptAgent
- **Complexity**: 394 lines (medium orchestration with TTS formatting)
- **Features**:
  - ElevenLabs TTS script generation with inline bracket cues
  - Emotional/pacing cue lexicon ([excited], [mocking], [shouting], etc.)
  - Natural stammering support ('F...folks', 'This is—ugh—OUTRAGEOUS')
  - Monologue and dialogue format support
  - Multi-speaker conversation tags ([interrupting], [overlapping])
  - Structured section extraction (Persona, Background, Articles, Tweets)
  - URL content fetching
  - 3,000 character limit enforcement
  - Editor integration with frontmatter (type='podcast')
  - Persona-aware voice and style
- **Migration Date**: November 2025
- **Architecture**:
  - **Agent**: `llm-orchestrator/orchestrator/agents/podcast_script_agent.py`
  - **Backend Tool Client**: Uses existing `search_web` for URL fetching
  - **No New gRPC RPCs Required**: Uses existing tools
  - **Process**: Extract sections → Fetch URLs → Generate script → Format response
- **Key Pattern**: Pure Orchestration - No backend file I/O, uses existing search tools
- **Editor Detection**: Requires active editor with `type: podcast` in frontmatter

### 12. OrgProjectAgent ✨ **NEW**
- **Complexity**: 492 lines (medium-high complexity with HITL workflow)
- **Features**:
  - Project capture into inbox.org with preview-confirm flow
  - LLM-powered smart enrichment (extracts description + tasks from initial message)
  - HITL (Human-in-the-Loop) preview and confirmation workflow
  - Pending state management across conversation turns
  - Multi-turn clarification for missing fields
  - Org-mode formatted project blocks with:
    - Title with tags (`:project:`)
    - Properties drawer (ID, CREATED timestamp)
    - Description paragraph
    - Optional SCHEDULED date
    - Up to 5 child TODO tasks
  - Labeled field parsing (Description:, Tasks:)
  - Bullet/comma-separated task parsing
  - Target date extraction from org timestamps (`<YYYY-MM-DD Dow>`)
  - Edit support during confirmation flow
  - Cancellation support
- **Migration Date**: November 2025
- **Architecture**:
  - **Agent**: `llm-orchestrator/orchestrator/agents/org_project_agent.py`
  - **Backend Tool Client**: Uses existing `AppendOrgInboxText` RPC (from OrgInboxAgent)
  - **No New gRPC RPCs Required**: Reuses existing org inbox infrastructure
  - **Process**: Extract intent → Smart LLM enrichment → Request missing fields → Build preview → Await confirmation → Write via gRPC
- **Key Pattern**: HITL Workflow with Pending State - Agent maintains state across turns, user confirms before file write
- **LLM Usage**: Two-stage LLM usage:
  1. Smart enrichment (extract description and tasks from initial message)
  2. Optional user clarification iterations
  3. Preview-confirm flow with edit support
- **State Management**: Uses `shared_memory["pending_project_capture"]` for multi-turn flow with fields:
  - `title`, `description`, `target_date`, `tags`, `initial_tasks`
  - `awaiting_confirmation` (boolean for HITL state)
  - `missing_fields` (list of fields needing clarification)
  - `preview_block` (org-mode formatted preview)

---

## Next Agents to Migrate

**Priority 1 (Simple/Medium)**:
- EntertainmentAgent (movie/TV recommendations)
- WebsiteCrawlerAgent (website content ingestion)

**Priority 2 (Complex Fiction/Content)**:
- FictionEditingAgent (fiction manuscript editing)
- StoryAnalysisAgent (narrative analysis)
- ContentAnalysisAgent (content quality analysis)
- ProofreadingAgent (grammar and style checking)

**Priority 3 (Specialized/Domain-Specific)**:
- WargamingAgent (military strategy gaming)
- SysMLAgent (systems modeling)
- EntertainmentAgent (entertainment recommendations)

**Goal**: Eventually remove all LangGraph dependencies from backend, keeping only data access and API services.

## Related Documentation

### Core Architecture
- [LangGraph Best Practices](.cursor/rules/agent-architecture-patterns.mdc)
- [orchestrator.proto](/opt/bastion/protos/orchestrator.proto) - gRPC service definition

### Context Infrastructure ⭐ **NEW**
- [LangGraph Context Gathering Infrastructure](LANGGRAPH_CONTEXT_GATHERING.md) - **Read this first!**
- [LangGraph Context Analysis](LANGGRAPH_CONTEXT_ANALYSIS.md) - Original context requirements
- [gRPC Context Gatherer](/opt/bastion/backend/services/grpc_context_gatherer.py) - Implementation

### Implementation Files
- [gRPC Service Implementation](/opt/bastion/llm-orchestrator/orchestrator/grpc_service.py)
- [gRPC Orchestrator Proxy](/opt/bastion/backend/api/grpc_orchestrator_proxy.py)
- [Base Agent (llm-orchestrator)](/opt/bastion/llm-orchestrator/orchestrator/agents/base_agent.py)

