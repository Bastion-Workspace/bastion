# Agent Integration Guide - llm-orchestrator Agent Development

## ⚠️ IMPORTANT: Agent Development Location

**All new agent development must use the llm-orchestrator service.**

The only agents in the backend are RSS agents (`RSSAgent` and `RSSBackgroundAgent`) used for scheduled feed polling. All other agent development happens in `llm-orchestrator/orchestrator/agents/`.

## Where to Develop New Agents

**Location:** `llm-orchestrator/orchestrator/agents/`

**Base Agent:** `llm-orchestrator/orchestrator/agents/base_agent.py`

**Orchestrator:** `llm-orchestrator/orchestrator/services/intent_classifier.py` (for routing)

**gRPC Service:** `llm-orchestrator/orchestrator/grpc_service.py` (for agent registration)

## Quick Start for llm-orchestrator Agents

### 1. **Agent Class Structure**

```python
from orchestrator.agents.base_agent import BaseAgent, TaskStatus
from typing import TypedDict, Dict, Any, List

class YourAgentState(TypedDict):
    """State for your agent LangGraph workflow"""
    query: str
    user_id: str
    metadata: Dict[str, Any]
    messages: List[Any]
    shared_memory: Dict[str, Any]
    response: Dict[str, Any]
    task_status: str
    error: str

class YourNewAgent(BaseAgent):
    """Your new agent following LangGraph best practices"""
    
    def __init__(self):
        super().__init__("your_agent")
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """Build LangGraph workflow for this agent"""
        workflow = StateGraph(YourAgentState)
        
        # Add nodes
        workflow.add_node("prepare_context", self._prepare_context_node)
        workflow.add_node("process_request", self._process_request_node)
        workflow.add_node("format_response", self._format_response_node)
        
        # Set entry point
        workflow.set_entry_point("prepare_context")
        
        # Define edges
        workflow.add_edge("prepare_context", "process_request")
        workflow.add_edge("process_request", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow.compile(checkpointer=checkpointer)
    
    async def _prepare_context_node(self, state: YourAgentState) -> Dict[str, Any]:
        """Prepare context for processing"""
        # Your logic here
        return {"task_status": "in_progress"}
    
    async def _process_request_node(self, state: YourAgentState) -> Dict[str, Any]:
        """Process the user request"""
        # Use self._get_llm(temperature=0.7, state=state) for LLM calls
        # Access shared_memory from state
        shared_memory = state.get("shared_memory", {})
        
        # Your processing logic
        return {"task_status": "complete"}
    
    async def _format_response_node(self, state: YourAgentState) -> Dict[str, Any]:
        """Format final response"""
        return {
            "response": {"content": "Your response"},
            "task_status": "complete"
        }
```

### 2. **Register in gRPC Service**

Add your agent to `llm-orchestrator/orchestrator/grpc_service.py`:

```python
# Import your agent
from orchestrator.agents.your_new_agent import YourNewAgent, get_your_agent

# In OrchestratorService.__init__, add:
self.your_agent = None

# In _get_your_agent method:
if self.your_agent is None:
    self.your_agent = get_your_agent()

# In ProcessQuery method, add routing case:
elif target_agent == "your_agent":
    result = await self.your_agent.process(
        query=query,
        metadata=metadata,
        messages=messages
    )
```

### 3. **Add Intent Classification**

Update `llm-orchestrator/orchestrator/services/intent_classifier.py`:

```python
# Add to agent routing logic
"your_agent": "generation",  # or appropriate intent type

# Add routing rules in system prompt
- **your_agent**
  - USE FOR: [describe when to use]
  - AVOID: [describe when not to use]
```

## Key Requirements

1. **Location**: `llm-orchestrator/orchestrator/agents/`
2. **Base Agent**: `orchestrator.agents.base_agent.BaseAgent`
3. **Workflow Pattern**: Must implement `_build_workflow()` with LangGraph StateGraph
4. **State Management**: Use TypedDict state classes, not Dict[str, Any]
5. **LLM Access**: Use `self._get_llm(temperature=X, state=state)` not direct ChatOpenAI
6. **Registration**: Register in gRPC service
7. **Routing**: Intent classification in llm-orchestrator

## Best Practices

### **State Preservation**
- Every node must return all critical state keys (metadata, user_id, shared_memory, messages, query)
- Preserve state even in error paths

### **LLM Access**
- Always use `self._get_llm(temperature=X, state=state)` to respect user model preferences
- Never create ChatOpenAI directly
- Never use `chat_service.openai_client`

### **Shared Memory**
- Access via `state.get("shared_memory", {})`
- Update by returning updated dict in node return
- Preserve existing shared_memory keys

### **⚠️ CRITICAL: Checkpoint Merge Order Pattern**

When merging `shared_memory` from checkpoint with fresh data from metadata, **merge order matters**:

```python
# ✅ CORRECT ORDER (new data overwrites old)
checkpoint_state = await workflow.aget_state(config)
existing_shared_memory = checkpoint_state.values.get("shared_memory", {}) if checkpoint_state and checkpoint_state.values else {}

# Start with checkpoint (old), then update with NEW data
shared_memory_merged = existing_shared_memory.copy()
shared_memory_merged.update(metadata.get("shared_memory", {}) or {})  # Fresh data takes precedence
```

```python
# ❌ WRONG ORDER (causes stale content bug!)
shared_memory = metadata.get("shared_memory", {}) or {}  # Start with new
shared_memory.update(existing_shared_memory)  # Old overwrites new - BUG!
```

**Why This Matters:**
- `active_editor` contains current file content (e.g., 2163 chars)
- Checkpoint may have stale `active_editor` from previous turn (e.g., 140 chars)
- Wrong order → stale checkpoint overwrites fresh content
- Result: Agent sees old file state, recreates content instead of editing

**Real Bug Example:**
1. Request 1: File has 140 chars (frontmatter only) → saved to checkpoint
2. Request 2: File now has 2163 chars → sent in `active_editor`
3. Wrong merge order → stale 140-char version overwrites fresh 2163-char version
4. Agent only sees 140 chars → treats as empty → recreates entire content

**Rule of Thumb:**
- **Checkpoint data = STALE** (from previous turn)
- **Metadata data = FRESH** (from current request)
- **ALWAYS let fresh overwrite stale**

This pattern is used correctly in:
- `character_development_agent.py`
- `writing_assistant_agent.py` (fixed 2026-01-28)

See commit history for detailed bug investigation.

### **Error Handling**
- Nodes should return error states, not raise exceptions
- Always preserve critical state keys even on error

### **File Size**
- Keep agent files under 500 lines
- Split complex agents into subgraphs

## Documentation References

- **LangGraph Best Practices**: `.cursor/rules/langgraph-best-practices.mdc`
- **Agent Architecture**: `.cursor/rules/agent-architecture-patterns.mdc`
- **gRPC Architecture**: `docs/GRPC_MICROSERVICES_ARCHITECTURE.md`

## RSS Agents (Backend Only)

The following RSS agents remain in the backend for scheduled tasks:

- `RSSAgent` - RSS feed management (interactive)
- `RSSBackgroundAgent` - RSS feed polling (scheduled Celery tasks)

These are the only agents that remain in the backend. All other agents are in llm-orchestrator.

For questions about agent development, refer to the llm-orchestrator documentation and LangGraph best practices.
