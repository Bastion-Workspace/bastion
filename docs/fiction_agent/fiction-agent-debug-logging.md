# Fiction Agent: Debug Logging for Anchor Resolution

**By George!** We've added detailed logging to see what's going wrong! **BULLY!**

---

## The Problem

Every fiction editing query results in:
```
❌ NO MATCH: revise operation failed to find target
🎯 RESOLVED: [99:99] confidence=0.00
```

**We're generating good prose but can't auto-insert it!**

---

## Debug Logging Added

### 1. Fiction Agent Logging
**File:** `backend/services/langgraph_agents/fiction_editing_agent.py`

**Added lines 676-686:**
```python
# **DEBUG: Log what the LLM actually provided**
logger.info(f"🔍 FICTION DEBUG: Structured response summary: {structured.summary}")
logger.info(f"🔍 FICTION DEBUG: Number of operations: {len(structured.operations)}")
for idx, op in enumerate(structured.operations):
    logger.info(f"🔍 FICTION DEBUG: Operation {idx}: op_type={op.op_type}")
    if hasattr(op, 'original_text') and op.original_text:
        logger.info(f"🔍 FICTION DEBUG: Operation {idx}: original_text (first 100 chars)='{op.original_text[:100]}'")
    if hasattr(op, 'anchor_text') and op.anchor_text:
        logger.info(f"🔍 FICTION DEBUG: Operation {idx}: anchor_text (first 100 chars)='{op.anchor_text[:100]}'")
    if hasattr(op, 'text') and op.text:
        logger.info(f"🔍 FICTION DEBUG: Operation {idx}: new text (first 100 chars)='{op.text[:100]}'")
```

**What this shows:**
- Summary the LLM provided
- Number of operations
- For each operation:
  - `op_type` (replace_range, insert_after_heading, delete_range)
  - `original_text` (first 100 chars) - for replace_range
  - `anchor_text` (first 100 chars) - for insert_after_heading
  - `text` (first 100 chars) - the new prose

---

### 2. Resolver Logging
**File:** `backend/utils/editor_operations_resolver.py`

**Added lines 354-359:**
```python
# **DEBUG: Log what the LLM provided**
logger.info(f"🔍 RESOLVER DEBUG: action={action}, require_anchors={require_anchors}")
logger.info(f"🔍 RESOLVER DEBUG: original_text={'[PRESENT: ' + original_text[:60] + '...]' if original_text else '[NONE]'}")
logger.info(f"🔍 RESOLVER DEBUG: anchor_text={'[PRESENT: ' + anchor_text[:60] + '...]' if anchor_text else '[NONE]'}")
logger.info(f"🔍 RESOLVER DEBUG: left_context={'[PRESENT]' if left_ctx else '[NONE]'}")
logger.info(f"🔍 RESOLVER DEBUG: right_context={'[PRESENT]' if right_ctx else '[NONE]'}")
```

**What this shows:**
- Normalized action (revise, insert, delete)
- Whether strict anchoring is required
- What `original_text` was provided (first 60 chars)
- What `anchor_text` was provided (first 60 chars)
- Whether left/right context was provided

---

## What to Look For

### Scenario 1: LLM Not Providing Anchor Text

**Expected logs:**
```
🔍 FICTION DEBUG: Operation 0: op_type=insert_after_heading
🔍 FICTION DEBUG: Operation 0: anchor_text (first 100 chars)='[NONE OR TOO SHORT]'
```

**Or:**
```
🔍 RESOLVER DEBUG: anchor_text=[NONE]
```

**Problem:** LLM isn't providing the required anchor_text field despite our prompting.

**Solution:** Need to strengthen the structured output schema enforcement or add validation.

---

### Scenario 2: LLM Providing Incomplete Anchor Text

**Expected logs:**
```
🔍 FICTION DEBUG: Operation 0: anchor_text (first 100 chars)='But a billion dollars is an extraordinary amount...'
```

**But manuscript has:**
```
"I suspect nothing yet," Fleet replied. "But a billion dollars is an extraordinary amount..."
```

**Problem:** LLM is providing partial paragraph (missing dialogue tag and first part).

**Solution:** Prompt is being ignored or LLM is hallucinating instead of copying exact text.

---

### Scenario 3: LLM Providing Paraphrased Text

**Expected logs:**
```
🔍 FICTION DEBUG: Operation 0: original_text (first 100 chars)='Fleet said he suspected nothing, but...'
```

**But manuscript has:**
```
"I suspect nothing yet," Fleet replied. "But a billion dollars..."
```

**Problem:** LLM is paraphrasing instead of copying EXACT text.

**Solution:** Need to explicitly instruct to copy verbatim, not summarize.

---

### Scenario 4: Wrong Operation Type

**Expected logs:**
```
🔍 FICTION DEBUG: Operation 0: op_type=replace_range
🔍 RESOLVER DEBUG: action=revise
```

**For a query like:** "End Chapter 1 with..."

**Problem:** LLM chose replace_range when it should use insert_after_heading.

**Solution:** Need to strengthen operation type guidance in prompt.

---

## Test Query to Run

**Query:** "Can we end Chapter 1 with a witty thought from Peterson?"

**Expected good behavior:**
```
🔍 FICTION DEBUG: Operation 0: op_type=insert_after_heading
🔍 FICTION DEBUG: Operation 0: anchor_text (first 100 chars)='"I suspect nothing yet," Fleet replied. "But a billion dollars is an extraordinary amount to spend on nostalgia alone."'
🔍 FICTION DEBUG: Operation 0: new text (first 100 chars)='

I nodded, already making mental notes about what to pack...'
🔍 RESOLVER DEBUG: action=insert
🔍 RESOLVER DEBUG: anchor_text=[PRESENT: "I suspect nothing yet," Fleet replied. "But a billion d...]
✅ EXACT MATCH: ...
🎯 RESOLVED: [1234:1350] confidence=1.00
```

**Expected bad behavior (current state):**
```
🔍 FICTION DEBUG: Operation 0: op_type=[CHECK THIS]
🔍 FICTION DEBUG: Operation 0: anchor_text=[CHECK IF PRESENT AND COMPLETE]
🔍 RESOLVER DEBUG: action=[CHECK THIS]
🔍 RESOLVER DEBUG: anchor_text=[CHECK THIS]
❌ NO MATCH: revise operation failed to find target
🎯 RESOLVED: [99:99] confidence=0.00
```

---

## Next Steps Based on Logs

### If anchor_text is [NONE]:
**Root cause:** LLM not generating the field at all  
**Fix:** Add field to structured output schema validation, make it required for insert operations

### If anchor_text is incomplete:
**Root cause:** LLM summarizing instead of copying exact text  
**Fix:** Add explicit "copy VERBATIM" instruction, provide good/bad examples in prompt

### If anchor_text is paraphrased:
**Root cause:** LLM interpreting/rephrasing instead of copying  
**Fix:** Add "DO NOT paraphrase, DO NOT summarize" warnings

### If wrong op_type:
**Root cause:** LLM misunderstanding when to use each operation type  
**Fix:** Strengthen operation type selection guidance, add decision tree

---

## Files Modified

**1. `backend/services/langgraph_agents/fiction_editing_agent.py`**
- Added lines 676-686: Debug logging for LLM structured output

**2. `backend/utils/editor_operations_resolver.py`**
- Added lines 354-359: Debug logging for resolver inputs

---

## How to Use

1. **Run the system** with `docker compose up --build`
2. **Make a fiction editing query** like "End Chapter 1 with Peterson's thought"
3. **Watch the logs** for the debug lines starting with `🔍`
4. **Identify the failure point:**
   - Is anchor_text present?
   - Is it complete?
   - Is it EXACT from the manuscript?
   - Is the operation type correct?
5. **Share the debug logs** so we can pinpoint the exact issue

---

## Roosevelt's Investigation Strategy

**BULLY!** We're conducting a proper reconnaissance mission!

**The cavalry must know:**
1. **What orders did we give?** (The prompt)
2. **What orders did the scout understand?** (The LLM output)
3. **What did the scout report back?** (The structured data)
4. **Why couldn't we find the target?** (The resolver search)

**By George!** Once we see the actual data flowing through the system, we'll know exactly where the breakdown occurs! 🏇

---

**Last Updated:** October 29, 2025  
**Status:** Active - Debug logging in place  
**Next Action:** Run a query and examine the debug logs







