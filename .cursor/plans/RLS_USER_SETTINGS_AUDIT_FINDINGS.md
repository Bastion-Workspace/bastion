# RLS User Settings Audit - Findings and Fix Plan

**Date:** December 19, 2025  
**Auditor:** Roosevelt AI Assistant  
**Scope:** User settings access (chat model preferences) with RLS enabled

---

## Executive Summary

**CRITICAL FINDING**: Multiple agents have nodes that **drop** `metadata` (containing `user_chat_model` preference) from state, causing LLM calls to fall back to DEFAULT_MODEL instead of user's selected model.

**Root Cause**: Nodes return dicts without the **Critical 5 Keys**:
1. `metadata` - Contains user model preference
2. `user_id` - Database RLS context
3. `shared_memory` - Cross-agent state
4. `messages` - Conversation history
5. `query` - Original user request

**Impact**: Users select a model in the sidebar (e.g., Claude Sonnet 4.5), but agents use the default model (e.g., Gemini 2.5 Pro) because metadata is dropped.

---

## Agent-by-Agent Findings

### ✅ **GOOD EXAMPLES** (Use as Templates)

#### Character Development Agent - `_load_references_node`
**File:** `llm-orchestrator/orchestrator/agents/character_development_agent.py`  
**Lines:** 322-332, 338-349

```python
# ✅ SUCCESS PATH preserves all 5 keys
return {
    "outline_body": outline_body,
    "rules_body": rules_body,
    "style_text": style_text,
    "character_bodies": character_bodies,
    "metadata": state.get("metadata", {}),        # ✅
    "user_id": state.get("user_id", "system"),   # ✅
    "shared_memory": state.get("shared_memory", {}),  # ✅
    "messages": state.get("messages", []),       # ✅
    "query": state.get("query", "")              # ✅
}

# ✅ ERROR PATH also preserves all 5 keys
return {
    "error": str(e),
    "metadata": state.get("metadata", {}),
    "user_id": state.get("user_id", "system"),
    "shared_memory": state.get("shared_memory", {}),
    "messages": state.get("messages", []),
    "query": state.get("query", "")
}
```

**Status:** ✅ **PERFECT** - This is the gold standard pattern!

---

#### Fiction Generation Subgraph - `build_generation_context_node`
**File:** `llm-orchestrator/orchestrator/subgraphs/fiction_generation_subgraph.py`  
**Lines:** 404-414, 435-445

```python
return {
    "generation_context_parts": context_parts,
    "generation_context_structure": context_structure,
    "is_empty_file": is_empty_file,
    "target_chapter_number": target_chapter_number,
    "current_chapter_label": current_chapter_label,
    # CRITICAL: Preserve state for subsequent nodes
    "system_prompt": state.get("system_prompt", ""),
    "datetime_context": state.get("datetime_context", ""),
    "metadata": state.get("metadata", {}),     # ✅
    "user_id": state.get("user_id", "system"),  # ✅
    # Note: Missing shared_memory, messages, query - should add
}
```

**Status:** ⚠️ **MOSTLY GOOD** - Preserves metadata/user_id, but missing shared_memory, messages, query

---

### ❌ **CRITICAL ISSUES** (Require Fixes)

### 1. Chat Agent ❌❌❌

**File:** `llm-orchestrator/orchestrator/agents/chat_agent.py`

#### Issue 1.1: `_prepare_context_node` (Lines 235-239)

```python
# ❌ CURRENT (BROKEN)
return {
    "persona": persona,
    "system_prompt": system_prompt,
    "llm_messages": llm_messages
}
# Missing: metadata, user_id, shared_memory, messages, query
```

**Fix Required:**
```python
# ✅ FIXED
return {
    "persona": persona,
    "system_prompt": system_prompt,
    "llm_messages": llm_messages,
    # ✅ CRITICAL: Preserve state
    "metadata": state.get("metadata", {}),
    "user_id": state.get("user_id", "system"),
    "shared_memory": state.get("shared_memory", {}),
    "messages": state.get("messages", []),
    "query": state.get("query", "")
}
```

**Error Path (Lines 243-246):**
```python
# ❌ CURRENT (BROKEN)
return {
    "error": str(e),
    "task_status": "error"
}
# Missing: metadata, user_id, shared_memory, messages, query
```

**Fix Required:**
```python
# ✅ FIXED
return {
    "error": str(e),
    "task_status": "error",
    # ✅ CRITICAL: Preserve state even on error
    "metadata": state.get("metadata", {}),
    "user_id": state.get("user_id", "system"),
    "shared_memory": state.get("shared_memory", {}),
    "messages": state.get("messages", []),
    "query": state.get("query", "")
}
```

---

#### Issue 1.2: `_check_local_data_node` (Lines 200-202, 204-206, 210-212)

**THREE return statements, all broken:**

```python
# ❌ CURRENT (BROKEN) - Success path
return {
    "local_data_results": result.get("formatted_context")
}

# ❌ CURRENT (BROKEN) - No results path
return {
    "local_data_results": None
}

# ❌ CURRENT (BROKEN) - Error path
return {
    "local_data_results": None
}
```

**Fix Required (ALL THREE paths):**
```python
# ✅ FIXED - Success path
return {
    "local_data_results": result.get("formatted_context"),
    "metadata": state.get("metadata", {}),
    "user_id": state.get("user_id", "system"),
    "shared_memory": state.get("shared_memory", {}),
    "messages": state.get("messages", []),
    "query": state.get("query", "")
}

# ✅ FIXED - No results path
return {
    "local_data_results": None,
    "metadata": state.get("metadata", {}),
    "user_id": state.get("user_id", "system"),
    "shared_memory": state.get("shared_memory", {}),
    "messages": state.get("messages", []),
    "query": state.get("query", "")
}

# ✅ FIXED - Error path
return {
    "local_data_results": None,
    "metadata": state.get("metadata", {}),
    "user_id": state.get("user_id", "system"),
    "shared_memory": state.get("shared_memory", {}),
    "messages": state.get("messages", []),
    "query": state.get("query", "")
}
```

---

#### Issue 1.3: `_detect_calculations_node` (Lines 273-275, 279-281)

```python
# ❌ CURRENT (BROKEN) - Success path
return {
    "needs_calculations": needs_calculations
}

# ❌ CURRENT (BROKEN) - Error path
return {
    "needs_calculations": False
}
```

**Fix Required:**
```python
# ✅ FIXED - Success path
return {
    "needs_calculations": needs_calculations,
    "metadata": state.get("metadata", {}),
    "user_id": state.get("user_id", "system"),
    "shared_memory": state.get("shared_memory", {}),
    "messages": state.get("messages", []),
    "query": state.get("query", "")
}

# ✅ FIXED - Error path
return {
    "needs_calculations": False,
    "metadata": state.get("metadata", {}),
    "user_id": state.get("user_id", "system"),
    "shared_memory": state.get("shared_memory", {}),
    "messages": state.get("messages", []),
    "query": state.get("query", "")
}
```

---

#### Issue 1.4: `_perform_calculations_node` (Lines 322-327, 332-335, 383-388, 393-396, 400-403)

**FIVE return statements, all broken:**

```python
# ❌ CURRENT (BROKEN) - All paths missing critical keys
return {
    "calculation_result": {...},
    "needs_calculations": False
}
```

**Fix Required (ALL FIVE paths):**
```python
# ✅ FIXED - Add to all return statements
return {
    "calculation_result": {...},  # or None
    "needs_calculations": False,
    "metadata": state.get("metadata", {}),
    "user_id": state.get("user_id", "system"),
    "shared_memory": state.get("shared_memory", {}),
    "messages": state.get("messages", []),
    "query": state.get("query", "")
}
```

---

#### Issue 1.5: `_generate_response_node` (Line 338 uses `_get_llm`)

**Current:**
```python
llm = self._get_llm(temperature=0.1, model=fast_model, state=state)
```

**Status:** ✅ **GOOD** - Already passes `state=state`, so it can access metadata

**BUT**: The node at line 413-417 returns error without preserving state:

```python
# ❌ CURRENT (BROKEN)
return {
    "error": "No LLM messages prepared",
    "task_status": "error",
    "response": {}
}
```

**Fix Required:**
```python
# ✅ FIXED
return {
    "error": "No LLM messages prepared",
    "task_status": "error",
    "response": {},
    "metadata": state.get("metadata", {}),
    "user_id": state.get("user_id", "system"),
    "shared_memory": state.get("shared_memory", {}),
    "messages": state.get("messages", []),
    "query": state.get("query", "")
}
```

---

### 2. Rules Editing Agent ❌❌

**File:** `llm-orchestrator/orchestrator/agents/rules_editing_agent.py`

#### Issue 2.1: `_prepare_context_node` (Lines 402-414, 418-421)

```python
# ❌ CURRENT (BROKEN) - Success path
return {
    "active_editor": active_editor,
    "rules": normalized_text,
    "filename": filename,
    "frontmatter": frontmatter,
    "cursor_offset": cursor_offset,
    "selection_start": selection_start,
    "selection_end": selection_end,
    "body_only": body_only,
    "para_start": para_start,
    "para_end": para_end,
    "current_request": current_request.strip()
}
# Missing: metadata, user_id, shared_memory, messages, query

# ❌ CURRENT (BROKEN) - Error path
return {
    "error": str(e),
    "task_status": "error"
}
```

**Fix Required:**
```python
# ✅ FIXED - Success path
return {
    "active_editor": active_editor,
    "rules": normalized_text,
    "filename": filename,
    "frontmatter": frontmatter,
    "cursor_offset": cursor_offset,
    "selection_start": selection_start,
    "selection_end": selection_end,
    "body_only": body_only,
    "para_start": para_start,
    "para_end": para_end,
    "current_request": current_request.strip(),
    # ✅ CRITICAL: Preserve state
    "metadata": state.get("metadata", {}),
    "user_id": state.get("user_id", "system"),
    "shared_memory": state.get("shared_memory", {}),
    "messages": state.get("messages", []),
    "query": state.get("query", "")
}

# ✅ FIXED - Error path
return {
    "error": str(e),
    "task_status": "error",
    "metadata": state.get("metadata", {}),
    "user_id": state.get("user_id", "system"),
    "shared_memory": state.get("shared_memory", {}),
    "messages": state.get("messages", []),
    "query": state.get("query", "")
}
```

---

#### Issue 2.2: `_load_references_node` (Lines 465-468, 474-478)

```python
# ❌ CURRENT (BROKEN) - Success path
return {
    "style_body": style_body,
    "characters_bodies": characters_bodies
}

# ❌ CURRENT (BROKEN) - Error path (incomplete in grep, but likely broken)
return {
    "rules_body": None,
    "style_body": None,
    "characters_bodies": [],
    "error": str(e)
}
```

**Fix Required:**
```python
# ✅ FIXED - Success path
return {
    "style_body": style_body,
    "characters_bodies": characters_bodies,
    # ✅ CRITICAL: Preserve state
    "metadata": state.get("metadata", {}),
    "user_id": state.get("user_id", "system"),
    "shared_memory": state.get("shared_memory", {}),
    "messages": state.get("messages", []),
    "query": state.get("query", "")
}

# ✅ FIXED - Error path
return {
    "rules_body": None,
    "style_body": None,
    "characters_bodies": [],
    "error": str(e),
    "metadata": state.get("metadata", {}),
    "user_id": state.get("user_id", "system"),
    "shared_memory": state.get("shared_memory", {}),
    "messages": state.get("messages", []),
    "query": state.get("query", "")
}
```

---

#### Issue 2.3: `_detect_request_type_node` Uses LLM (Line 532)

```python
# Current
llm = self._get_llm(temperature=0.1, state=state)  # ✅ GOOD - passes state
```

**Status:** ✅ **GOOD** - Already passes `state=state`

**BUT**: Need to verify return statements at lines 548-555 and 558-565 preserve state.

---

### 3. Fiction Context Subgraph ⚠️

**File:** `llm-orchestrator/orchestrator/subgraphs/fiction_context_subgraph.py`

#### Issue 3.1: `prepare_context_node` (Lines 83-94, 98-102)

```python
# ⚠️ CURRENT (PARTIAL) - Success path
return {
    "active_editor": active_editor,
    "manuscript": manuscript,
    "manuscript_content": manuscript,
    "filename": filename,
    "frontmatter": frontmatter,
    "cursor_offset": cursor_offset,
    "selection_start": selection_start,
    "selection_end": selection_end,
    "current_request": current_request,
    "user_id": state.get("user_id", "system")  # ✅ Has user_id
}
# Missing: metadata, shared_memory, messages, query

# ⚠️ CURRENT (PARTIAL) - Error path
return {
    "error": str(e),
    "task_status": "error",
    "user_id": state.get("user_id", "system")  # ✅ Has user_id
}
# Missing: metadata, shared_memory, messages, query
```

**Fix Required:**
```python
# ✅ FIXED - Success path
return {
    "active_editor": active_editor,
    "manuscript": manuscript,
    "manuscript_content": manuscript,
    "filename": filename,
    "frontmatter": frontmatter,
    "cursor_offset": cursor_offset,
    "selection_start": selection_start,
    "selection_end": selection_end,
    "current_request": current_request,
    "user_id": state.get("user_id", "system"),
    # ✅ ADD: Critical state keys
    "metadata": state.get("metadata", {}),
    "shared_memory": state.get("shared_memory", {}),
    "messages": state.get("messages", []),
    "query": state.get("query", "")
}

# ✅ FIXED - Error path
return {
    "error": str(e),
    "task_status": "error",
    "user_id": state.get("user_id", "system"),
    "metadata": state.get("metadata", {}),
    "shared_memory": state.get("shared_memory", {}),
    "messages": state.get("messages", []),
    "query": state.get("query", "")
}
```

---

### 4. Style Editing Agent ⚠️ (Needs Full Audit)

**File:** `llm-orchestrator/orchestrator/agents/style_editing_agent.py`

**Status:** Similar structure to Rules Agent - likely has same issues in:
- `_prepare_context_node`
- `_load_references_node`

**Action Required:** Full audit of all node return statements (similar to Rules Agent fixes).

---

### 5. Outline Editing Agent ⚠️ (Needs Full Audit)

**File:** `llm-orchestrator/orchestrator/agents/outline_editing_agent.py`

**Status:** Similar structure to Rules Agent - likely has same issues.

**Action Required:** Full audit of all node return statements.

---

### 6. Character Development Agent ✅

**File:** `llm-orchestrator/orchestrator/agents/character_development_agent.py`

**Status:** ✅ **EXCELLENT** - `_load_references_node` is the gold standard! Other nodes likely need verification, but this agent is ahead of the pack.

**Action Required:** Verify other nodes (`_prepare_context_node`, `_detect_request_type_node`, etc.) follow the same pattern.

---

### 7. Fiction Editing Agent (Main + Subgraphs) ⚠️

**Main File:** `llm-orchestrator/orchestrator/agents/fiction_editing_agent.py`

**Subgraphs:**
- `fiction_context_subgraph.py` - ⚠️ Needs fixes (see Issue 3.1)
- `fiction_generation_subgraph.py` - ⚠️ Mostly good, add shared_memory/messages/query
- `fiction_validation_subgraph.py` - ❓ Not audited yet
- `fiction_resolution_subgraph.py` - ❓ Not audited yet

**Action Required:** Full audit of main agent nodes and all subgraph nodes.

---

## Summary Statistics

| Agent | Status | Priority | Estimated Time |
|-------|--------|----------|----------------|
| Chat Agent | ❌ CRITICAL | P0 | 30 min |
| Rules Editing Agent | ❌ CRITICAL | P0 | 30 min |
| Fiction Context Subgraph | ⚠️ PARTIAL | P1 | 15 min |
| Fiction Generation Subgraph | ⚠️ PARTIAL | P1 | 15 min |
| Style Editing Agent | ⚠️ UNKNOWN | P1 | 30 min |
| Outline Editing Agent | ⚠️ UNKNOWN | P1 | 30 min |
| Character Development Agent | ✅ MOSTLY GOOD | P2 | 15 min (verification) |
| Fiction Validation Subgraph | ❓ NOT AUDITED | P2 | 20 min |
| Fiction Resolution Subgraph | ❓ NOT AUDITED | P2 | 20 min |

**Total Estimated Fix Time:** ~3.5 hours

---

## Fix Implementation Priority

### Phase 1: Critical Fixes (1 hour)
1. **Chat Agent** - All 4 broken nodes
2. **Rules Editing Agent** - `_prepare_context_node`, `_load_references_node`

### Phase 2: Subgraph Fixes (1 hour)
3. **Fiction Context Subgraph** - Add metadata/shared_memory/messages/query
4. **Fiction Generation Subgraph** - Add shared_memory/messages/query
5. **Complete audit of Fiction Validation/Resolution subgraphs**

### Phase 3: Remaining Agents (1.5 hours)
6. **Style Editing Agent** - Full node audit and fixes
7. **Outline Editing Agent** - Full node audit and fixes
8. **Character Development Agent** - Verify all nodes preserve state

---

## Testing Plan

After fixes, test each agent:

1. **Setup**: User selects `anthropic/claude-sonnet-4.5` in sidebar
2. **Trigger**: Send query to agent
3. **Verify**: Check logs for `🎯 SELECTED MODEL: anthropic/claude-sonnet-4.5`
4. **Fail Condition**: If logs show default model, state preservation failed

---

## Automated Fix Pattern (Code Template)

For any node that needs fixing, use this template:

```python
async def any_node(self, state: AgentState) -> Dict[str, Any]:
    try:
        # ... node logic ...
        
        return {
            # Node-specific outputs
            "node_output": result,
            
            # ✅ CRITICAL: ALWAYS preserve these 5 keys
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }
        
    except Exception as e:
        logger.error(f"Node failed: {e}")
        return {
            "error": str(e),
            "task_status": "error",
            
            # ✅ CRITICAL: Preserve even on error!
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }
```

---

## Success Criteria

✅ All nodes preserve `metadata`, `user_id`, `shared_memory`, `messages`, `query`  
✅ All LLM calls use `_get_llm(state=state)` to access user model preference  
✅ Test with user model selection shows correct model used (not default)  
✅ No agent falls back to DEFAULT_MODEL when user has preference set  
✅ Logs consistently show: `🎯 SELECTED MODEL: <user's choice>`

---

**BULLY!** This audit provides a complete battle plan for fixing RLS user settings access across all agents!



