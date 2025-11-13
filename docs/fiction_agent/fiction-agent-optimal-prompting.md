# Fiction Agent Optimal Prompting Strategy

**BULLY!** Roosevelt's "Trust the LLM, Teach the Structure" Doctrine

---

## Core Philosophy

**Don't pattern match intent - teach the LLM about operations and let it choose!**

The LLM should understand:
1. **What operations are available** (replace_range, insert_after_heading, delete_range)
2. **When to use each operation** (semantic understanding, not keywords)
3. **How to anchor properly** (chapter boundaries, exact text matching)
4. **The resolver will handle precision** (progressive search, confidence scoring)

---

## Proposed System Prompt (Simplified)

```
You are a MASTER NOVELIST editor/generator. You have three fundamental operations:

### 1. REPLACE_RANGE - Change Existing Text
**Use when:** User wants to revise, improve, change, or modify existing prose.
**Anchoring:** Provide 'original_text' with EXACT, VERBATIM text to replace (20-40 words).
**Examples:**
- "Make this dialogue wittier" → replace_range on dialogue paragraph
- "Improve the opening sentence" → replace_range on that sentence
- "Revise this scene for more tension" → replace_range on scene paragraphs

### 2. INSERT_AFTER_HEADING - Add New Content
**Use when:** User wants to add, append, or insert new content without replacing existing.
**Anchoring:** Provide 'anchor_text' with exact line/paragraph to insert after.
**Examples:**
- "End Chapter 1 with Peterson's thought" → insert_after last paragraph before next chapter
- "Add a scene transition here" → insert_after current paragraph
- "Insert a new paragraph about..." → insert_after specified location

### 3. DELETE_RANGE - Remove Content
**Use when:** User wants to remove, delete, or cut content.
**Anchoring:** Provide 'original_text' with EXACT text to delete.
**Examples:**
- "Remove this description" → delete_range on that paragraph
- "Cut the middle section" → delete_range on those paragraphs

---

## CHAPTER BOUNDARY RULES (CRITICAL!)

**Chapters are sacred boundaries marked by "## Chapter N" headings.**

### ✅ CORRECT - Ending Chapter 1:
```json
{
  "op_type": "insert_after_heading",
  "anchor_text": "But a billion dollars is an extraordinary amount to spend on nostalgia alone.",
  "text": "\n\nI nodded, already making mental notes about what to pack for a transatlantic crossing."
}
```

**OR:**
```json
{
  "op_type": "replace_range",
  "original_text": "\"I suspect nothing yet,\" Fleet replied. \"But a billion dollars is an extraordinary amount to spend on nostalgia alone.\"",
  "text": "\"I suspect nothing yet,\" Fleet replied. \"But a billion dollars is an extraordinary amount to spend on nostalgia alone.\"\n\nI nodded, already making mental notes about what to pack for a transatlantic crossing."
}
```

### ❌ WRONG - Bleeding into Chapter 2:
```json
{
  "op_type": "replace_range",
  "original_text": "But a billion dollars is an extraordinary amount to spend on nostalgia alone.\n\n## Chapter 2\n\nThe rest of the morning...",
  "text": "But a billion dollars...\n\nI nodded...\n\n## Chapter 2\n\nThe rest of the morning..."
}
```
**Problem:** Included next chapter's heading! This will delete the heading.

---

## TO ADD CONTENT AT CHAPTER END:

**Method 1 (INSERT):** Find last paragraph before next chapter heading, insert after it.
**Method 2 (REPLACE):** Find last paragraph, replace it with original + new content.

**Both methods work! Choose based on your semantic understanding of the request.**

---

## ANCHORING PRECISION

The system uses progressive search resolver with 4 strategies:
1. Exact match (confidence 1.0)
2. Normalized whitespace (confidence 0.9)
3. Sentence boundary (confidence 0.8)
4. Key phrase anchoring (confidence 0.7)

**Provide detailed anchors:**
- **original_text**: 20-40 words, complete sentences, EXACT verbatim text
- **anchor_text**: Exact line to insert after (heading or paragraph ending)
- **left_context + right_context**: Boundary text for disambiguation
- **occurrence_index**: If text appears multiple times (0=first, 1=second, etc.)

---

## SCOPE METADATA

Set appropriate scope:
- **"paragraph"**: Single paragraph or sentence edits
- **"chapter"**: Full chapter generation or major chapter rewrites
- **"multi_chapter"**: Operations spanning multiple chapters

The system uses this for display and validation, not operation restrictions.

---

## EXAMPLES BY QUERY TYPE

### "End Chapter 1 with a witty thought"
**Operation:** insert_after_heading (add new content)
**Anchor:** Last paragraph of Chapter 1 (before "## Chapter 2")
**Scope:** paragraph

### "Revise the dialogue to be wittier"
**Operation:** replace_range (change existing)
**Anchor:** Exact dialogue paragraph text
**Scope:** paragraph

### "Write Chapter 5 based on the outline"
**Operation:** replace_range OR insert_after_heading
**Anchor:** Chapter 5 location (after "## Chapter 4" or replace existing Chapter 5)
**Scope:** chapter

### "Add more tension to the confrontation scene"
**Operation:** replace_range (modify existing scene)
**Anchor:** Confrontation scene paragraph(s)
**Scope:** paragraph or chapter (depending on scene length)

### "Remove this description"
**Operation:** delete_range
**Anchor:** Exact description paragraph
**Scope:** paragraph

---

## NO PATTERN MATCHING NEEDED

The LLM understands:
- "End with X" → Likely insert_after or replace at end
- "Revise X" → Likely replace_range
- "Add X" → Likely insert_after or replace (contextual)
- "Remove X" → Likely delete_range
- "Write X" → Likely replace_range or insert_after (contextual)

**The LLM chooses based on semantic understanding, not keyword triggers!**

---

## IMPLEMENTATION CHANGES

### Remove from fiction_editing_agent.py:

```python
# REMOVE THIS HARDCODED INTENT DETECTION:
if any(k in lowered_req for k in ["end with", "finish with", ...]):
    edit_intent = "targeted_edit"
elif any(k in lowered_req for k in ["revise", "revision", ...]):
    edit_intent = "revision"
# ... etc
```

### Replace with:

```python
# Let the LLM choose operation type based on prompt understanding
# The resolver handles precision based on anchors provided
# No intent pattern matching needed!
```

### Enhanced System Prompt:

Include the "REPLACE_RANGE / INSERT_AFTER_HEADING / DELETE_RANGE" teaching section.
Include the "CHAPTER BOUNDARY RULES" section.
Remove mode-specific guidance (REVISION MODE, TARGETED EDIT MODE, etc.).

---

## BENEFITS

### ✅ Flexibility
- Handles edge cases naturally
- No manual trigger list maintenance
- Adapts to new phrasings automatically

### ✅ Semantic Understanding
- LLM understands intent from context
- Chooses operation type appropriately
- Respects chapter boundaries through understanding, not rules

### ✅ Consistency
- Same "trust the LLM" approach as routing
- Operation types are structural, not semantic
- Resolver handles precision, LLM handles intent

### ✅ Simplicity
- No complex intent detection code
- Fewer conditional branches
- Clearer mental model

---

## TRADE-OFFS

### Potential Risks:
- LLM might occasionally choose wrong operation type
- Requires good prompt engineering to teach operations clearly
- Less deterministic than hardcoded rules

### Mitigations:
- Clear operation type descriptions in prompt
- Examples for each operation type
- Chapter boundary rules prominently displayed
- Resolver validation (confidence scoring)
- User can always regenerate if wrong operation chosen

---

## TESTING SCENARIOS

### "End Chapter 1 with Peterson's witty thought"
**Expected:**
- op_type: "insert_after_heading" OR "replace_range"
- anchor: Last paragraph of Chapter 1
- Does NOT include "## Chapter 2"

### "Make the dialogue in paragraph 3 wittier"
**Expected:**
- op_type: "replace_range"
- anchor: Paragraph 3 dialogue text
- scope: "paragraph"

### "Write Chapter 5"
**Expected:**
- op_type: "replace_range" (if Chapter 5 exists) OR "insert_after_heading" (if new)
- anchor: Chapter 5 location or "## Chapter 4"
- scope: "chapter"

### "Add more tension before the reveal"
**Expected:**
- op_type: "replace_range" (modify existing scene)
- anchor: Scene before reveal
- scope: "paragraph" or "chapter"

---

## Roosevelt's Verdict

**BULLY!** Teach the LLM about structural operations, not semantic patterns!

**By George!** Trust the cavalry's intelligence - give them tools and training, not rigid marching orders!

The LLM is smart enough to understand:
- "End with X" means append/insert at end
- "Revise X" means replace existing
- "Add X" means insert new
- "Remove X" means delete

**No pattern matching needed - just clear operational teaching!** 🏇







