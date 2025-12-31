# Fiction Agent: LLM-Driven Operation Selection

**BULLY!** Roosevelt's "Trust the LLM, Teach the Structure" Implementation - **By George!**

---

## What Changed

### Before: Pattern Matching Intent Detection
```python
# Hardcoded pattern matching
if "end with" in query:
    edit_intent = "targeted_edit"
    intent_guidance = "INSERT AT CHAPTER END MODE: ..."
elif "revise" in query:
    edit_intent = "revision"
    intent_guidance = "REVISION MODE: ..."
elif "write" in query:
    edit_intent = "generation"
    intent_guidance = "GENERATION MODE: ..."
```

**Problems:**
- ❌ Brittle keyword matching
- ❌ Misses phrasing variations
- ❌ Contradicts "trust the LLM" routing philosophy
- ❌ More code to maintain

---

### After: LLM Semantic Understanding
```python
# No pattern matching - LLM chooses operation type
# System prompt teaches three operations:
# 1. replace_range - change existing text
# 2. insert_after_heading - add new text after location
# 3. delete_range - remove text

# LLM understands user intent semantically and chooses appropriate operation
```

**Benefits:**
- ✅ Handles all phrasing naturally
- ✅ Consistent with routing philosophy
- ✅ Less code
- ✅ More flexible

---

## Enhanced System Prompt

### Teaches Three Operations Clearly

**1. replace_range**
```
USE WHEN: User wants to revise, improve, change, modify existing prose
ANCHORING: Provide 'original_text' with EXACT verbatim text (20-40 words)
EXAMPLES:
- "Make this dialogue wittier" → replace_range on dialogue paragraph
- "Improve the pacing" → replace_range on relevant paragraphs
```

**2. insert_after_heading**
```
USE WHEN: User wants to add, append, insert new content (not replace)
ANCHORING: Provide 'anchor_text' with exact line/paragraph to insert after
EXAMPLES:
- "End Chapter 1 with Peterson's thought" → insert_after last paragraph
- "Add a scene transition" → insert_after current paragraph
```

**3. delete_range**
```
USE WHEN: User wants to delete, remove, cut content
ANCHORING: Provide 'original_text' with exact text to delete
EXAMPLES:
- "Remove this description" → delete_range on that paragraph
```

---

### Chapter Boundary Teaching

**CRITICAL instruction in system prompt:**

```
=== CHAPTER BOUNDARIES ARE SACRED ===

Chapters are marked by "## Chapter N" headings.
⚠️ CRITICAL: NEVER include the next chapter's heading in your operation!

**To add content at END of a chapter:**
1. Find the LAST PARAGRAPH of the target chapter (before next "## Chapter")
2. OPTION A: Use insert_after_heading with anchor_text = last paragraph
3. OPTION B: Use replace_range with original_text = last paragraph, 
             text = original + new content

**Example - Ending Chapter 1:**
Last paragraph: "But a billion dollars is an extraordinary amount..."
Next line: "## Chapter 2"

✅ CORRECT (insert_after):
{
  "op_type": "insert_after_heading",
  "anchor_text": "But a billion dollars is an extraordinary amount...",
  "text": "\n\nI nodded, already making mental notes..."
}

✅ CORRECT (replace with append):
{
  "op_type": "replace_range",
  "original_text": "\"I suspect nothing yet,\" Fleet replied. \"But a billion dollars...\"",
  "text": "\"I suspect nothing yet,\" Fleet replied. \"But a billion dollars...\"\n\nI nodded..."
}

❌ WRONG (bleeding into Chapter 2):
{
  "op_type": "replace_range",
  "original_text": "But a billion dollars...\n\n## Chapter 2\n\nThe rest...",
  ...
}
```

---

## Removed Code

### 1. Intent Detection Pattern Matching
**Removed from lines 497-526** (old numbering):
- Hardcoded "end with", "revise", "write" keyword checks
- edit_intent variable assignment
- revision_mode flag

### 2. Mode-Specific Guidance
**Removed from lines 274-320** (old numbering):
- "INSERT AT CHAPTER END MODE" guidance
- "REVISION MODE" guidance  
- "TARGETED EDIT MODE" guidance
- "GENERATION MODE" guidance

### 3. Mode-Based Operation Processing
**Simplified lines 565-698**:
- No more edit_intent branching
- Uniform resolver usage for all operations
- Confidence-based warnings instead of mode-specific handling

---

## Simplified Operation Processing

### Before (Mode-Specific):
```python
if edit_intent == "revision":
    require_anchors = True
    if failed:
        logger.error("REVISION requires anchors! Skipping.")
        continue
elif edit_intent == "targeted_edit":
    require_anchors = False
    if failed:
        fallback to paragraph scope
else:  # generation
    require_anchors = False
    if failed:
        fallback to chapter bounds
```

### After (Uniform):
```python
# Use resolver for all operations
try:
    resolved_start, resolved_end, resolved_text, resolved_confidence = resolve_operation(
        manuscript,
        op_dict,
        require_anchors=False,  # Flexible - let confidence guide us
    )
    
    # Warn if low confidence
    if resolved_confidence < 0.7:
        logger.warning(f"⚠️ Low confidence ({resolved_confidence:.2f})")
    
except ValueError as e:
    # Fallback based on scope metadata (paragraph vs chapter)
    # Not based on hardcoded intent
```

---

## How LLM Chooses Operations

**Query:** "End Chapter 1 with a witty thought from Peterson"

**LLM's reasoning:**
1. User wants to **add** new content (not replace existing)
2. Target location: **end of Chapter 1**
3. Operation: `insert_after_heading`
4. Anchor: Last paragraph before "## Chapter 2"

**No pattern matching needed - semantic understanding!**

---

**Query:** "Make the dialogue in paragraph 3 wittier"

**LLM's reasoning:**
1. User wants to **change** existing content (dialogue)
2. Target location: **paragraph 3**
3. Operation: `replace_range`
4. Anchor: Exact text of that dialogue paragraph

---

**Query:** "Remove this description"

**LLM's reasoning:**
1. User wants to **delete** content
2. Target: **description** (from context)
3. Operation: `delete_range`
4. Anchor: Exact text of description paragraph

---

## Benefits Over Pattern Matching

### 1. Handles Natural Language Variations
**Pattern matching would need:**
- "end with", "finish with", "conclude with", "close with"
- "end chapter", "finish chapter", "close chapter"  
- "have X end by", "let X finish with", "make X conclude"
- ... endless variations

**LLM understands ALL naturally!**

### 2. Context-Aware Decisions
LLM can consider:
- Manuscript structure (chapters, paragraphs)
- Conversation history (previous edits)
- User's phrasing nuances
- Implied vs explicit intents

### 3. Consistent Philosophy
- Routing: Trust LLM intent classification ✅
- Fiction agent: Trust LLM operation selection ✅
- **No inconsistency!**

### 4. Less Code
- **Removed ~100 lines** of pattern matching
- **Simpler operation processing**
- **Easier to maintain**

---

## Trade-offs & Mitigations

### Potential Risk: Wrong Operation Choice
**Risk:** LLM might occasionally choose wrong operation type

**Mitigations:**
1. **Clear teaching** in system prompt with examples
2. **Prominent chapter boundary rules**
3. **Confidence scoring** (warn if < 0.7)
4. **User can regenerate** if wrong operation chosen
5. **Resolver validates** anchors (catches bad choices)

### Potential Risk: Less Deterministic
**Risk:** Same query might produce slightly different operations

**Mitigations:**
1. **System prompt consistency** (clear rules)
2. **Confidence thresholds** (flag uncertain operations)
3. **This is acceptable** - slight variations OK if semantically correct

---

## Testing Scenarios

### Scenario 1: "End Chapter 1 with a witty thought"
**Expected:**
- op_type: `insert_after_heading` OR `replace_range`
- anchor: Last paragraph of Chapter 1
- Does NOT include "## Chapter 2"
- Confidence: ≥ 0.8

**Why it works:**
- System prompt teaches "end with" means insert/append
- Chapter boundary rules prevent bleeding
- LLM chooses appropriate operation semantically

---

### Scenario 2: "Make the dialogue wittier"
**Expected:**
- op_type: `replace_range`
- anchor: Exact dialogue paragraph text
- scope: `paragraph`
- Confidence: ≥ 0.9 (exact text match)

**Why it works:**
- LLM understands "make X" means modify existing
- Chooses replace_range appropriately
- Provides good anchor (taught in examples)

---

### Scenario 3: "Write Chapter 5"
**Expected:**
- op_type: `replace_range` (if Chapter 5 exists) OR `insert_after_heading` (if new)
- anchor: Chapter 5 location or "## Chapter 4"
- scope: `chapter`
- Includes "## Chapter 5" heading

**Why it works:**
- LLM understands "write" means generate new content
- Chooses insert if new, replace if exists
- System adds chapter heading for chapter-scope insertions

---

### Scenario 4: "Have Peterson reflect on the irony at the chapter's conclusion"
**Expected:**
- op_type: `insert_after_heading` OR `replace_range`
- anchor: Last paragraph of current chapter
- Does NOT include next chapter heading

**Why it works:**
- **No "end with" keyword!**
- LLM understands "at chapter's conclusion" semantically
- Chooses appropriate operation without pattern matching
- This would FAIL with keyword approach!

---

## Comparison: Before vs After

| Aspect | Before (Pattern Matching) | After (LLM Understanding) |
|--------|--------------------------|---------------------------|
| **Routing** | Trust LLM ✅ | Trust LLM ✅ |
| **Intent Detection** | Pattern match ❌ | Trust LLM ✅ |
| **Operation Selection** | Based on intent ❌ | LLM chooses ✅ |
| **Edge Cases** | Must add keywords | Handles naturally |
| **Code Complexity** | ~700 lines | ~600 lines |
| **Maintainability** | Must update triggers | Self-adapting |
| **Consistency** | Mixed approach | Uniform philosophy |

---

## Roosevelt's Verdict

**BULLY!** We've achieved architectural consistency!

**The cavalry now operates intelligently at EVERY level:**
1. **Routing**: LLM intent classification (no pattern matching)
2. **Operation selection**: LLM semantic understanding (no pattern matching)
3. **Precision**: Resolver with progressive search (confidence-scored)

**By George!** Give the cavalry intelligence and training, not rigid marching orders!

**Trust the LLM to understand:**
- "End with X" → append/insert
- "Revise X" → replace
- "Add X" → insert or modify
- "Remove X" → delete
- "Have X conclude by Y" → append at end

**The LLM is smart enough!** 🏇

---

## Files Modified

- `llm-orchestrator/orchestrator/agents/fiction_editing_agent.py` - Complete refactor
  - Enhanced system prompt with operation teaching
  - Removed intent detection pattern matching
  - Removed mode-specific guidance
  - Simplified operation processing
  - Uniform resolver usage

---

**Last Updated:** October 29, 2025  
**Status:** Active - LLM-driven operation selection  
**Philosophy:** Trust the LLM, Teach the Structure







