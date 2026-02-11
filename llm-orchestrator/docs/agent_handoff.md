# Agent Handoff Pattern

## Overview

This document defines the standard pattern for inter-agent communication and task delegation in the LLM orchestrator. When one agent needs to delegate work to another agent (e.g., reference agent delegating research to research agent), use the **handoff context pattern** via `shared_memory`.

## Core Principle

**Use `shared_memory` for structured data passing between agents, not query string concatenation.**

## The Handoff Context Pattern

### Structure

When delegating to another agent, pass context via `shared_memory.handoff_context`:

```python
handoff_shared_memory = {
    "user_chat_model": metadata.get("user_chat_model"),  # User preferences
    "handoff_context": {
        "source_agent": "source_agent_name",
        "handoff_type": "task_delegation_type",
        "context_data": {
            # Agent-specific context here
        }
    }
}

metadata = {
    "user_id": user_id,
    "conversation_id": conversation_id,
    "shared_memory": handoff_shared_memory
}

result = await target_agent.process(
    query=augmented_query,
    metadata=metadata,
    messages=messages
)
```

### Standard Fields

**Required:**
- `source_agent`: Name of agent initiating handoff
- `handoff_type`: Type of delegation (e.g., "research_delegation", "analysis_delegation")

**Optional but Recommended:**
- `context_data`: Agent-specific structured data
- `handoff_reason`: Why this handoff is occurring
- `expected_output`: What source agent expects back

## Example: Reference Agent → Research Agent

### Sending Agent (Reference Agent)

```python
# In reference_agent.py
async def _call_research_subgraph_node(self, state: ReferenceAgentState) -> Dict[str, Any]:
    reference_content = state.get("reference_content", "")
    
    # Build handoff context
    handoff_shared_memory = {
        "user_chat_model": metadata.get("user_chat_model"),
        "handoff_context": {
            "source_agent": "reference_agent",
            "handoff_type": "research_delegation",
            "reference_document": {
                "content": reference_content,
                "type": state.get("reference_type"),
                "filename": state.get("filename"),
                "has_content": bool(reference_content)
            },
            "referenced_files": state.get("referenced_files_content", {}),
            "analysis_context": {
                "needs_calculations": state.get("needs_calculations"),
                "calculation_results": state.get("calculation_results"),
                "complexity": state.get("complexity")
            }
        }
    }
    
    research_metadata = {
        "user_id": state.get("user_id"),
        "shared_memory": handoff_shared_memory
    }
    
    # Augment query with brief note (full data in shared_memory)
    augmented_query = f"""{research_query}

[Context: User has reference document available in shared_memory.]"""
    
    # Call research agent
    result = await research_agent.process(
        query=augmented_query,
        metadata=research_metadata,
        messages=state.get("messages", [])
    )
```

### Receiving Agent (Research Agent)

```python
# In full_research_agent.py quick answer check node
async def _quick_answer_check_node(self, state: ResearchState) -> Dict[str, Any]:
    # Check for handoff context
    shared_memory = state.get("shared_memory", {})
    handoff_context = shared_memory.get("handoff_context", {})
    
    if handoff_context:
        source_agent = handoff_context.get("source_agent")
        reference_doc = handoff_context.get("reference_document", {})
        
        if reference_doc.get("has_content"):
            # Access the document content
            ref_content = reference_doc.get("content", "")
            ref_filename = reference_doc.get("filename")
            
            # Include in LLM prompt
            handoff_note = f"""

**AGENT HANDOFF CONTEXT**:
- Delegated by: {source_agent}
- User has reference document: {ref_filename}
- Document content available

When answering, reference data from the user's document."""
            
            logger.info(f"🔗 Handoff context detected from {source_agent}")
    
    # Include handoff context in LLM call
    messages_for_llm.append(HumanMessage(content=evaluation_prompt + handoff_note))
```

## Benefits of This Pattern

### ✅ Architectural Advantages

1. **Structured Data:** No string concatenation, clean JSON structures
2. **Type Safety:** Can validate handoff_context structure
3. **Scalability:** Won't hit token limits with large documents
4. **Visibility:** Easy to log and debug handoff data
5. **Flexibility:** Easy to add new fields without changing signatures

### ✅ vs Query String Concatenation

**Bad Pattern (Don't Do This):**
```python
# ❌ Concatenating data into query string
augmented_query = f"""USER HAS DOCUMENT:
---START---
{huge_document_content}
---END---
RESEARCH REQUEST: {research_query}"""
```

**Problems:**
- Token limits with large documents
- Hard to parse and validate
- Pollutes the semantic query
- Difficult to debug

**Good Pattern (Use This):**
```python
# ✅ Structured data in shared_memory
handoff_shared_memory = {
    "handoff_context": {
        "source_agent": "reference_agent",
        "reference_document": {
            "content": document_content,
            "metadata": {...}
        }
    }
}
```

## Common Handoff Types

### Research Delegation
**Source:** Any agent needing external information  
**Target:** `research_agent`  
**Context Data:**
- `reference_document`: User's document for context
- `analysis_context`: What analysis has been done
- `constraints`: Time period, sources, etc.

### Analysis Delegation
**Source:** Agents with content needing evaluation  
**Target:** `content_analysis_agent`, `story_analysis_agent`  
**Context Data:**
- `content_to_analyze`: The content
- `analysis_type`: What kind of analysis
- `criteria`: Specific evaluation criteria

### Generation Delegation
**Source:** Agents needing content creation  
**Target:** `fiction_editing_agent`, `electronics_agent`, etc.  
**Context Data:**
- `generation_context`: What to create
- `style_guide`: Style preferences
- `constraints`: Length, tone, requirements

## Implementation Checklist

When implementing agent handoffs:

### Sending Agent
- [ ] Build `handoff_context` with `source_agent` and `handoff_type`
- [ ] Include all necessary context data in structured format
- [ ] Pass via `metadata.shared_memory.handoff_context`
- [ ] Add brief note in query (not full data)
- [ ] Log handoff for debugging

### Receiving Agent
- [ ] Check for `shared_memory.handoff_context` in relevant nodes
- [ ] Extract and validate handoff context
- [ ] Include handoff data in LLM prompts where needed
- [ ] Log handoff reception for debugging
- [ ] Handle missing handoff context gracefully

### Both Agents
- [ ] Document handoff contract in agent docstrings
- [ ] Add unit tests for handoff scenarios
- [ ] Log handoff metadata for observability

## Agent Handoff Matrix

Current implemented handoffs:

| Source Agent | Target Agent | Handoff Type | Context Passed |
|-------------|--------------|--------------|----------------|
| `reference_agent` | `research_agent` | research_delegation | reference_document, analysis_context |
| `chat_agent` | `research_agent` | quick_lookup | conversation_context |

**Chat → Research (quick_lookup):** LLM-based "should hand off" decision (not keyword heuristics). Option A: Chat returns Research's response (replaces Chat reply). Continuity: `last_agent` and `primary_agent_selected` set to `research_agent` so the next message can stay with Research. Does not hand off for comments, thanks, or non-research follow-ups.

Future planned handoffs:

| Source Agent | Target Agent | Handoff Type | Context Passed |
|-------------|--------------|--------------|----------------|
| `fiction_editing_agent` | `story_analysis_agent` | analysis_request | story_content, analysis_criteria |
| `general_project_agent` | `electronics_agent` | component_design | project_requirements, constraints |

## Debugging Handoffs

### Logging

Always log handoff events:

```python
# Sending agent
logger.info(f"🔗 Handing off to {target_agent} with {handoff_type}")
logger.debug(f"🔗 Handoff context: {handoff_context}")

# Receiving agent
logger.info(f"🔗 Handoff context detected from {source_agent}")
logger.debug(f"🔗 Handoff data keys: {list(handoff_context.keys())}")
```

### Common Issues

**Issue:** Receiving agent says "I don't have context"  
**Cause:** Handoff context not included in LLM prompt  
**Fix:** Add handoff context to messages_for_llm

**Issue:** Token limit exceeded  
**Cause:** Passing too much data in query instead of shared_memory  
**Fix:** Move large data to shared_memory, use preview in prompt

**Issue:** Data not accessible in receiving agent  
**Cause:** Not passing through state properly  
**Fix:** Ensure metadata.shared_memory flows to state.shared_memory

## Best Practices

1. **Keep Query Clean:** Query should be semantic, not data dump
2. **Use Previews:** For large documents, use first N chars in prompts
3. **Validate Early:** Check handoff_context structure on receive
4. **Log Liberally:** Log handoff events for debugging
5. **Document Contracts:** Document what each handoff type provides
6. **Graceful Degradation:** Handle missing handoff context gracefully
7. **Test Edge Cases:** Test with missing/invalid handoff data

## Anti-Patterns to Avoid

❌ **Don't concatenate large data into query strings**
❌ **Don't pass data in custom metadata fields** (use handoff_context)
❌ **Don't skip logging handoff events**
❌ **Don't assume handoff context always exists**
❌ **Don't put handoff logic in multiple places** (centralize in nodes)

## Future Enhancements

Potential improvements to handoff pattern:

1. **Handoff Context Validation:** Pydantic models for handoff_context
2. **Handoff Registry:** Central registry of valid handoff types
3. **Handoff Metrics:** Track handoff success/failure rates
4. **Handoff Timeouts:** Set timeouts for delegated work
5. **Handoff Chaining:** Support multi-level handoffs (A→B→C)

## Summary

The agent handoff pattern provides a **clean, scalable, and maintainable** way for agents to delegate work and share context. By using `shared_memory.handoff_context` for structured data and keeping queries semantic, we avoid common pitfalls like token limits and data pollution while maintaining clear architectural boundaries between agents.

**Remember:** When in doubt, use `shared_memory` for data, queries for semantics. 🏇

