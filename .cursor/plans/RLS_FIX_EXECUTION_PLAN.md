# RLS User Settings - Fix Execution Plan

**Date:** December 19, 2025  
**Status:** Ready for Implementation  
**Estimated Time:** 3.5 hours

---

## Execution Strategy

**Pattern:** Add Critical 5 Keys to ALL node return dicts (success AND error paths)

**Critical 5 Keys:**
```python
"metadata": state.get("metadata", {}),        # User model preference!
"user_id": state.get("user_id", "system"),   # Database RLS context
"shared_memory": state.get("shared_memory", {}),  # Cross-agent state
"messages": state.get("messages", []),       # Conversation history
"query": state.get("query", "")              # Original request
```

---

## Phase 1: Critical Fixes (Priority P0)

### Fix 1.1: Chat Agent - `_prepare_context_node`

**File:** `llm-orchestrator/orchestrator/agents/chat_agent.py`  
**Lines:** 235-239 (success), 243-246 (error)

**Changes:**
1. Line 235: Add 5 critical keys to success return
2. Line 243: Add 5 critical keys to error return

---

### Fix 1.2: Chat Agent - `_check_local_data_node`

**File:** `llm-orchestrator/orchestrator/agents/chat_agent.py`  
**Lines:** 200-202, 204-206, 210-212

**Changes:**
1. Line 200: Add 5 critical keys to success return
2. Line 204: Add 5 critical keys to no-results return
3. Line 210: Add 5 critical keys to error return

---

### Fix 1.3: Chat Agent - `_detect_calculations_node`

**File:** `llm-orchestrator/orchestrator/agents/chat_agent.py`  
**Lines:** 273-275 (success), 279-281 (error)

**Changes:**
1. Line 273: Add 5 critical keys to success return
2. Line 279: Add 5 critical keys to error return

---

### Fix 1.4: Chat Agent - `_perform_calculations_node`

**File:** `llm-orchestrator/orchestrator/agents/chat_agent.py`  
**Lines:** 322-327, 332-335, 383-388, 393-396, 400-403

**Changes:**
1. All FIVE return statements: Add 5 critical keys

---

### Fix 1.5: Chat Agent - `_generate_response_node` (error path)

**File:** `llm-orchestrator/orchestrator/agents/chat_agent.py`  
**Lines:** 413-417

**Changes:**
1. Line 413: Add 5 critical keys to error return

---

### Fix 2.1: Rules Editing Agent - `_prepare_context_node`

**File:** `llm-orchestrator/orchestrator/agents/rules_editing_agent.py`  
**Lines:** 402-414 (success), 418-421 (error)

**Changes:**
1. Line 402: Add 5 critical keys to success return
2. Line 418: Add 5 critical keys to error return

---

### Fix 2.2: Rules Editing Agent - `_load_references_node`

**File:** `llm-orchestrator/orchestrator/agents/rules_editing_agent.py`  
**Lines:** 465-468 (success), ~474-478 (error)

**Changes:**
1. Line 465: Add 5 critical keys to success return
2. Line ~474: Add 5 critical keys to error return

---

### Fix 2.3: Rules Editing Agent - `_detect_request_type_node`

**File:** `llm-orchestrator/orchestrator/agents/rules_editing_agent.py`  
**Lines:** Check all return statements in this method

**Changes:**
1. Verify all returns preserve 5 critical keys

---

## Phase 2: Subgraph Fixes (Priority P1)

### Fix 3.1: Fiction Context Subgraph - `prepare_context_node`

**File:** `llm-orchestrator/orchestrator/subgraphs/fiction_context_subgraph.py`  
**Lines:** 83-94 (success), 98-102 (error)

**Changes:**
1. Line 83: Add metadata, shared_memory, messages, query (already has user_id)
2. Line 98: Add metadata, shared_memory, messages, query (already has user_id)

---

### Fix 3.2: Fiction Context Subgraph - OTHER nodes

**File:** `llm-orchestrator/orchestrator/subgraphs/fiction_context_subgraph.py`

**Changes:**
1. Audit ALL nodes in subgraph
2. Add 5 critical keys to ALL return statements

---

### Fix 4.1: Fiction Generation Subgraph - `build_generation_context_node`

**File:** `llm-orchestrator/orchestrator/subgraphs/fiction_generation_subgraph.py`  
**Lines:** 404-414 (success), 435-445 (error)

**Changes:**
1. Line 404: Add shared_memory, messages, query (already has metadata, user_id)
2. Line 435: Add query (already has others)

---

### Fix 4.2: Fiction Generation Subgraph - ALL other nodes

**File:** `llm-orchestrator/orchestrator/subgraphs/fiction_generation_subgraph.py`

**Changes:**
1. `build_generation_prompt_node` - Add 5 keys to all returns
2. `call_generation_llm_node` - Add 5 keys to all returns
3. `validate_generated_output_node` - Add 5 keys to all returns

---

### Fix 5: Fiction Validation Subgraph - ALL nodes

**File:** `llm-orchestrator/orchestrator/subgraphs/fiction_validation_subgraph.py`

**Changes:**
1. Audit ALL nodes
2. Add 5 critical keys to ALL return statements (success AND error)

---

### Fix 6: Fiction Resolution Subgraph - ALL nodes

**File:** `llm-orchestrator/orchestrator/subgraphs/fiction_resolution_subgraph.py`

**Changes:**
1. Audit ALL nodes
2. Add 5 critical keys to ALL return statements (success AND error)

---

## Phase 3: Remaining Agents (Priority P1/P2)

### Fix 7: Style Editing Agent - ALL nodes

**File:** `llm-orchestrator/orchestrator/agents/style_editing_agent.py`

**Nodes to Fix:**
- `_prepare_context_node`
- `_load_references_node`
- `_detect_request_type_node`
- `_detect_mode_node`
- `_analyze_examples_node`
- `_generate_edit_plan_node` (verify only)
- `_resolve_operations_node`
- `_format_response_node`

**Changes:**
1. ALL nodes: Add 5 critical keys to ALL return statements

---

### Fix 8: Outline Editing Agent - ALL nodes

**File:** `llm-orchestrator/orchestrator/agents/outline_editing_agent.py`

**Nodes to Fix:**
- `_prepare_context_node`
- `_load_references_node`
- `_detect_request_type_node`
- `_detect_mode_and_intent_node`
- `_generate_edit_plan_node` (verify only)
- `_resolve_operations_node`
- `_format_response_node`

**Changes:**
1. ALL nodes: Add 5 critical keys to ALL return statements

---

### Fix 9: Character Development Agent - Verification

**File:** `llm-orchestrator/orchestrator/agents/character_development_agent.py`

**Status:** `_load_references_node` is PERFECT (gold standard!)

**Changes:**
1. Verify `_prepare_context_node` preserves 5 keys
2. Verify `_detect_request_type_node` preserves 5 keys
3. Verify `_generate_edit_plan_node` preserves 5 keys
4. Verify `_resolve_operations_node` preserves 5 keys
5. Verify `_format_response_node` preserves 5 keys

---

## Testing Checklist

After each fix, test:

### Test 1: Model Selection
1. User selects `anthropic/claude-sonnet-4.5` in sidebar
2. Send chat query: "Hello, what model are you using?"
3. Check logs for: `🎯 SELECTED MODEL: anthropic/claude-sonnet-4.5`
4. ❌ FAIL if logs show: `🎯 SELECTED MODEL: google/gemini-2.5-pro` (default)

### Test 2: Fiction Agent with Subgraphs
1. User selects `anthropic/claude-sonnet-4.5` in sidebar
2. Open fiction manuscript, send edit request
3. Check logs for generation subgraph LLM call
4. Verify: `🎯 SELECTED MODEL: anthropic/claude-sonnet-4.5`

### Test 3: Rules/Style/Outline Agents
1. User selects custom model
2. Open rules/style/outline document, send edit request
3. Verify correct model used in all LLM calls

### Test 4: Character Agent
1. User selects custom model
2. Open character document, send edit request
3. Verify correct model used (should already work!)

---

## Success Metrics

✅ **Zero fallbacks to default model** when user has preference set  
✅ **All logs show user's selected model** in LLM call traces  
✅ **All agents respect user_chat_model** from metadata  
✅ **Subgraphs preserve state** through all nodes  
✅ **Error paths also preserve state** (no state loss on failures)

---

## Implementation Notes

### Template for Quick Fixes

Use this template for EVERY node return statement:

```python
return {
    # ... existing node outputs ...
    
    # ✅ CRITICAL: Always add these 5 lines
    "metadata": state.get("metadata", {}),
    "user_id": state.get("user_id", "system"),
    "shared_memory": state.get("shared_memory", {}),
    "messages": state.get("messages", []),
    "query": state.get("query", "")
}
```

### Find & Replace Pattern

For each node, search for:
```python
return {
```

And ensure ALL return statements include the Critical 5 Keys.

---

## Rollout Strategy

**Approach:** Fix agents incrementally, test after each fix

**Order:**
1. ✅ Fix Chat Agent (highest impact - affects all chat)
2. ✅ Test Chat Agent
3. ✅ Fix Rules Agent
4. ✅ Test Rules Agent
5. ✅ Fix Fiction Subgraphs
6. ✅ Test Fiction Agent
7. ✅ Fix Style/Outline Agents
8. ✅ Test all agents
9. ✅ Final regression test

**Estimated Time:** ~3.5 hours total (including testing)

---

**BULLY!** This plan will ensure all agents respect user model preferences with RLS enabled!



