# 🎖️ Roosevelt's Agent Development Roadmap
## **"The Square Deal for LangGraph Agent Architecture"**

**BULLY!** This roadmap documents our **battle-tested patterns** for developing **top-tier LangGraph agents** with **bulletproof HITL capabilities**, **perfect citations**, and **Roosevelt-grade reliability**!

---

## 🚀 **AGENT DEVELOPMENT QUICK START**

### **The Roosevelt "Big Stick" Agent Pattern**

Every new agent follows this **proven cavalry charge formation**:

```
1. 🎯 INHERIT from BaseAgent
2. 🔧 IMPLEMENT core methods  
3. 🛡️ ADD structured outputs
4. 🌐 INTEGRATE with tools
5. 📚 EXTRACT citations
6. 🎭 SUPPORT personas
7. 🤝 SHARE via memory
```

---

## 📋 **AGENT DEVELOPMENT CHECKLIST**

### **Phase 1: Foundation Setup**
- [ ] Create agent file: `backend/services/langgraph_agents/{agent_name}_agent.py`
- [ ] Inherit from `BaseAgent` class
- [ ] Define agent constructor with unique name
- [ ] Add agent to tool registry imports

### **Phase 2: Core Implementation** 
- [ ] Implement `async def process(self, state: Dict[str, Any]) -> Dict[str, Any]`
- [ ] Build structured prompt with `_build_{agent}_prompt()`
- [ ] Add persona support via `_build_persona_prompt()`
- [ ] Configure tool access and permissions

### **Phase 3: HITL Integration**
- [ ] Define structured response format with Pydantic
- [ ] Implement permission requests for restricted operations
- [ ] Add task status handling (`complete`, `incomplete`, `permission_required`, `error`)
- [ ] Test HITL flow with permission scenarios

### **Phase 4: Citation & Sources**
- [ ] Implement `_extract_citations_from_tools()` method
- [ ] Store citations in `agent_results["citations"]`
- [ ] Test citation display in frontend accordion
- [ ] Verify source transparency and traceability

### **Phase 5: Orchestrator Integration**
- [ ] Add agent to orchestrator routing logic
- [ ] Update intent classification patterns
- [ ] Test agent delegation from orchestrator
- [ ] Verify state flow and response handling

### **Phase 6: Quality Assurance**
- [ ] Test with various persona configurations
- [ ] Verify shared memory integration
- [ ] Test error scenarios and fallbacks
- [ ] Validate output formatting and markdown

---

## 🎯 **AGENT ARCHITECTURE PATTERNS**

### **Pattern 1: Research Agent (Web + Local)**
```python
class ResearchAgent(BaseAgent):
    """
    CAPABILITIES: Local search, web search with HITL, citation extraction
    TOOLS: search_local, search_web, crawl_web_content, analyze_and_ingest
    HITL: Required for web operations
    OUTPUTS: Natural language + structured citations
    """
```

### **Pattern 2: Chat Agent (Conversational)**  
```python
class ChatAgent(BaseAgent):
    """
    CAPABILITIES: Natural conversation, context awareness, persona adaptation
    TOOLS: None (pure LLM reasoning)
    HITL: Not required
    OUTPUTS: Conversational responses with personality
    """
```

### **Pattern 3: Coding Agent (Development)**
```python
class CodingAgent(BaseAgent):
    """
    CAPABILITIES: Code generation, analysis, debugging, file operations
    TOOLS: read_file, write_file, analyze_code, run_tests
    HITL: Required for file modifications
    OUTPUTS: Code + explanations + citations to documentation
    """
```

---

## 🔧 **CORE IMPLEMENTATION PATTERNS**

### **1. Agent Constructor Pattern**
```python
class NewAgent(BaseAgent):
    def __init__(self):
        super().__init__("new_agent")  # Unique agent name
```

### **2. Process Method Signature**
```python
async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    ROOSEVELT'S AGENT PROCESSING PATTERN
    Input: LangGraph state with user context
    Output: Updated state with agent results
    """
```

### **3. Structured Response Pattern**
```python
# Always include in agent results:
state["agent_results"] = {
    "response": natural_language_response,
    "structured_response": structured_data.dict(),
    "citations": extracted_citations,
    "tools_used": tools_executed,
    "task_complete": boolean,
    "permission_needed": boolean
}
```

### **4. Citation Extraction Pattern**
```python
def _extract_citations_from_tools(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract citations from tool results for frontend display"""
    citations = []
    # Check shared_memory.search_results for citations
    # Check tool results for citation arrays
    # Return formatted citation list
    return citations
```

---

## 🌐 **TOOL INTEGRATION PATTERNS**

### **Permission-Based Tool Access**
```python
# NO PERMISSION REQUIRED
- search_local: Local knowledge base
- get_document: Document retrieval  
- calculate: Mathematical operations
- convert_units: Unit conversions

# PERMISSION REQUIRED (HITL)
- search_web: Web search operations
- crawl_web_content: Web scraping
- write_file: File modifications
- execute_code: Code execution
```

### **Tool Result Processing**
```python
# Always extract and preserve:
1. Citations from web tools
2. Source metadata from search results
3. Confidence scores from analysis
4. Error information for debugging
```

---

## 🎭 **PERSONA INTEGRATION PATTERN**

### **Persona Support Template**
```python
def _build_agent_prompt(self, persona: Optional[Dict[str, Any]] = None) -> str:
    base_prompt = """Your agent-specific instructions..."""
    
    # Add persona via inherited method
    persona_prompt = self._build_persona_prompt(persona)
    return base_prompt + persona_prompt
```

### **Roosevelt Persona Enforcement**
```python
# All agents MUST use Theodore Roosevelt speaking style:
- "BULLY!" for enthusiasm
- "By George!" for agreement  
- Military/cavalry references
- Decisive, energetic language
- Action-oriented personality
```

---

## 🤝 **SHARED MEMORY PATTERNS**

### **Research Findings Storage**
```python
state["shared_memory"]["research_findings"][query] = {
    "findings": agent_response,
    "sources_searched": ["local", "web"],
    "confidence_level": 0.9,
    "task_status": "complete",
    "tools_used": ["search_local", "search_web"],
    "citations": citation_list
}
```

### **Agent Handoff Context**
```python
state["shared_memory"]["agent_handoffs"].append({
    "from_agent": "research_agent",
    "to_agent": "coding_agent", 
    "context": "Research findings for implementation",
    "timestamp": datetime.now().isoformat()
})
```

---

## 🛡️ **HITL IMPLEMENTATION PATTERNS**

### **Permission Request Structure**
```python
from services.langgraph_official_orchestrator import PermissionRequest

permission_request = PermissionRequest(
    operation_type="web_search",  # web_search, web_crawl, data_modification
    query=user_query,
    reasoning="Local search insufficient, need comprehensive web results",
    safety_level="low"  # low, medium, high
)
```

### **Task Status Handling**
```python
from enum import Enum

class TaskStatus(Enum):
    COMPLETE = "complete"          # Task finished successfully
    INCOMPLETE = "incomplete"      # Task needs more work
    PERMISSION_REQUIRED = "permission_required"  # HITL needed
    ERROR = "error"               # Task failed
```

---

## 📊 **TESTING PATTERNS**

### **Agent Testing Checklist**
```python
# Test Scenarios:
1. ✅ Basic functionality with valid inputs
2. ✅ Error handling with invalid inputs  
3. ✅ HITL permission flow (request → approval → completion)
4. ✅ HITL permission denial handling
5. ✅ Citation extraction and display
6. ✅ Persona adaptation and voice consistency
7. ✅ Shared memory integration
8. ✅ Tool execution and result processing
9. ✅ Markdown formatting in responses
10. ✅ State management and cleanup
```

### **Manual Testing Commands**
```bash
# Test basic agent via orchestrator
curl -X POST http://localhost:8000/api/async/orchestrator/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Test query for new agent"}'

# Test HITL permission flow
# 1. Send query requiring permission
# 2. Verify permission request appears
# 3. Respond with "Yes" 
# 4. Verify agent continues and completes
```

---

## 🔮 **FUTURE AGENT ROADMAP**

### **Phase 1 Agents (Current)**
- ✅ **Research Agent**: Web + local search with HITL
- ✅ **Chat Agent**: Conversational intelligence
- 🚧 **Coding Agent**: Development assistance (in progress)

### **Phase 2 Agents (Planned)**
- 📅 **Calendar Agent**: Scheduling and time management
- 📁 **File Manager Agent**: Document organization and operations
- 📊 **Analytics Agent**: Data analysis and visualization
- 🔍 **Investigation Agent**: Deep research and fact-checking

### **Phase 3 Agents (Future Vision)**
- 🤖 **Automation Agent**: Workflow automation and scripting  
- 🎨 **Creative Agent**: Content creation and design assistance
- 🔒 **Security Agent**: Privacy and security analysis
- 🌐 **Integration Agent**: API and service connectivity

---

## 📚 **REFERENCE IMPLEMENTATION**

### **Perfect Agent Template**
```python
"""
{Agent Name} Implementation
{Brief description of agent capabilities}
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class {AgentName}Agent(BaseAgent):
    """
    ROOSEVELT'S {AGENT} AGENT
    
    CAPABILITIES:
    - {capability 1}
    - {capability 2}
    
    TOOLS: {list of tools}
    HITL: {Required/Not Required} for {operations}
    OUTPUTS: {output format description}
    """
    
    def __init__(self):
        super().__init__("{agent_name}")
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process {agent} request with Roosevelt reliability"""
        try:
            # Extract context
            current_query = state.get("current_query", "")
            shared_memory = state.get("shared_memory", {})
            persona = state.get("persona")
            
            # Build prompt
            system_prompt = self._build_{agent}_prompt(persona, shared_memory)
            
            # Execute agent logic
            response = await self._execute_agent_logic(state, system_prompt)
            
            # Extract citations
            citations = self._extract_citations_from_tools(state)
            
            # Update state
            state["agent_results"] = {
                "response": response,
                "citations": citations,
                "tools_used": [],
                "task_complete": True,
                "permission_needed": False
            }
            
            return state
            
        except Exception as e:
            logger.error(f"❌ {agent} agent error: {e}")
            state["agent_results"] = {
                "response": f"{Agent} processing error: {str(e)}",
                "error": str(e)
            }
            return state
    
    def _build_{agent}_prompt(self, persona: Optional[Dict[str, Any]] = None, shared_memory: Optional[Dict[str, Any]] = None) -> str:
        """Build {agent}-specific prompt with Roosevelt personality"""
        
        base_prompt = """You are a {AGENT_DESCRIPTION} with Roosevelt's vigorous personality.
        
INSTRUCTIONS:
- {instruction 1}
- {instruction 2}
- ALWAYS speak like Theodore Roosevelt (BULLY! By George!)
- {agent-specific guidelines}
        """
        
        # Add persona
        persona_prompt = self._build_persona_prompt(persona)
        return base_prompt + persona_prompt
    
    def _extract_citations_from_tools(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract citations for frontend display"""
        # Implementation specific to agent's tools
        return []
```

---

## 🎖️ **ROOSEVELT'S AGENT DEVELOPMENT COMMANDMENTS**

1. **"SPEAK SOFTLY AND CARRY A BIG PROMPT"** - Clear, decisive instructions
2. **"TRUST BUT VERIFY"** - Structured outputs with validation
3. **"WALK SOFTLY AND CARRY CITATIONS"** - Always provide sources
4. **"THE SQUARE DEAL"** - Fair access to tools and capabilities  
5. **"CONSERVATION OF STATE"** - Preserve shared memory for future agents
6. **"THE STRENUOUS LIFE"** - Robust error handling and fallbacks
7. **"SPEAK CLEARLY OR CARRY A BIG STICK"** - Permission-based HITL patterns

---

**BULLY!** **By George!** This roadmap ensures that **every future agent** follows our **battle-tested patterns** for **maximum reliability** and **Roosevelt-grade excellence**! 

**The agent development cavalry charge is now perfectly documented for future campaigns!** 🎖️⚡
