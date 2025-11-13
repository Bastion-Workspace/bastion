# Agent Integration Guide - Roosevelt's "Intelligence Sharing" Architecture

## Overview

This guide outlines how to integrate new agents into our LangGraph-based system with comprehensive shared state management. Our architecture ensures that **every agent has access to the full context** of what other agents have done and what the user has said.

## Core Architecture Principles

### 1. **Shared Memory First**
- Every agent receives the complete `shared_memory` containing all previous agent results
- No agent works in isolation - they build upon each other's work
- State persistence across conversation turns via LangGraph's native checkpointing

### 2. **LangGraph Native**
- Leverage LangGraph's `ConversationState` TypedDict with proper annotations
- Use built-in message handling with `add_messages` annotation
- Utilize native checkpointing for persistence (no custom mechanisms)

### 3. **Comprehensive Context**
- Research findings, code artifacts, user preferences, and conversation history
- Agent handoff tracking and tool result caching
- Permission grants and failed operation learning

## Shared Memory Structure

```python
shared_memory = {
    # Research & Analysis Results
    "research_findings": {},          # Key research results by topic
    "document_analyses": {},          # Document analysis results  
    "web_search_results": {},         # Web search findings
    "entity_insights": {},           # Entity extraction results
    
    # Code & Technical Artifacts
    "code_artifacts": {},            # Generated code by purpose
    "technical_solutions": {},       # Technical problem solutions
    "calculations": {},              # Mathematical results
    
    # User Context & Preferences
    "user_preferences": {
        "communication_style": None,
        "expertise_level": None,
        "preferred_detail_level": None,
        "domain_interests": []
    },
    
    # Conversation Context
    "conversation_context": {
        "main_topics": [],           # Key topics discussed
        "ongoing_projects": [],      # Multi-turn projects  
        "decisions_made": [],        # User decisions/choices
        "questions_asked": [],       # User's question history
        "goals_identified": []       # User objectives identified
    },
    
    # Agent Collaboration
    "agent_handoffs": [],            # Track agent transitions
    "tool_results": {},              # Cached tool outputs
    "permission_grants": [],         # User permissions granted
    "failed_operations": [],         # Operations that failed
    
    # Session Metadata
    "session_start": "2025-01-01T00:00:00",
    "last_updated": "2025-01-01T00:00:00", 
    "agents_involved": [],           # Which agents have been active
    "total_interactions": 0
}
```

## How to Create a New Agent

### 1. **Agent Class Structure**

```python
from services.langgraph_agents.base_agent import BaseAgent
from models.shared_memory_models import SharedMemory, validate_shared_memory

class YourNewAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.agent_name = "your_agent"
    
    async def process(self, state: Dict[str, Any]) -> str:
        # Extract shared memory
        shared_memory = state.get("shared_memory", {})
        
        # Access previous agent results
        research_findings = shared_memory.get("research_findings", {})
        code_artifacts = shared_memory.get("code_artifacts", {})
        user_preferences = shared_memory.get("user_preferences", {})
        
        # Your agent logic here...
        result = await self._your_agent_logic(state, shared_memory)
        
        # Update shared memory with your results - ROOSEVELT'S NEW APPROACH
        shared_memory.setdefault("your_category", {})[result_key] = {
            "data": result,
            "agent": self.agent_name,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
```

### 2. **Register in Orchestrator**

Add your agent to `langgraph_official_orchestrator.py`:

```python
# Import your agent
from .langgraph_agents.your_new_agent import YourNewAgent

# Initialize in __init__
self.your_agent = YourNewAgent()

# Add node to graph
self.graph.add_node("your_agent", self._your_agent_node)

# Create node method
async def _your_agent_node(self, state: ConversationState) -> ConversationState:
    logger.info("🎯 YOUR AGENT: Processing...")
    
    # Convert state for agent compatibility
    agent_state = self._convert_to_agent_state(state, "your_agent")
    
    # Process with your agent
    response = await self.your_agent.process(agent_state)
    
    # Update state
    state["messages"].append(AIMessage(content=response))
    state["active_agent"] = "your_agent"
    state["is_complete"] = True  # or logic to determine completion
    
    return state
```

### 3. **Add Routing Logic**

Update intent classification in `capability_based_intent_service.py`:

```python
class IntentType(str, Enum):
    # ... existing intents ...
    YOUR_INTENT = "your_intent"

# Add routing in orchestrator routing logic
```

## Best Practices

### **State Updates**
- Always update shared_memory directly with structured data and return it from agent nodes
- Include timestamp and agent attribution automatically
- Store results in appropriate categories for discoverability

### **Context Awareness**
- Check `shared_memory` for relevant previous work before starting
- Build upon previous agent findings rather than repeating work
- Respect user preferences learned by other agents

### **Error Handling**
- Log failed operations to `shared_memory["failed_operations"]` for learning
- Provide graceful fallbacks when shared context is incomplete
- Never break the state structure

### **Performance**
- Cache expensive tool results in `shared_memory["tool_results"]`
- Avoid re-running identical operations across agents
- Keep shared memory reasonably sized (consider cleanup strategies)

## Example Agent Chaining Scenarios

### **Research → Code → Chat Flow**
1. **Research Agent**: Finds information about a topic, stores in `research_findings`
2. **Coding Agent**: Uses research findings to generate code, stores in `code_artifacts`  
3. **Chat Agent**: Explains the code in context of the research, with user's preferred communication style

### **Multi-turn Project Flow**
1. **User**: "Help me analyze this dataset"
2. **Research Agent**: Analyzes data structure, stores findings
3. **User**: "Create a visualization"
4. **Coding Agent**: Uses data analysis to create appropriate visualization code
5. **User**: "Explain the insights"
6. **Chat Agent**: Uses both data analysis and visualization context to explain insights

## Migration Notes

- Existing agents automatically receive enhanced shared memory via `_convert_to_agent_state`
- Legacy `agent_handoff_context` still supported for backward compatibility
- No breaking changes to existing agent interfaces

## File Size Guidelines

- Keep individual agent files under 500 lines [[memory:4115124]]
- Use inheritance and composition to share common functionality
- Leverage existing `BaseAgent` infrastructure

**By George!** This architecture ensures every agent has the **full intelligence context** of the conversation!
