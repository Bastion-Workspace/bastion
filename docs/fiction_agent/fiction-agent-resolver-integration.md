# Fiction Editing Agent - Progressive Search Resolver Integration

**BULLY!** Roosevelt's "Hybrid Resolver Approach" - **Trust busting for imprecise edits!**

## Executive Summary

The Fiction Editing Agent (in `llm-orchestrator/orchestrator/agents/fiction_editing_agent.py`) now uses the same **Progressive Search Resolver** (`backend/utils/editor_operations_resolver.py`) that the Outline, Rules, and Character Development agents use. This provides:

1. **Surgical precision for revisions** - exact text targeting with confidence scoring
2. **Smart positioning for generation** - automatic insertion after last chapter
3. **Flexible targeting for edits** - balance between precision and usability

---

## The Resolver Service Architecture

### Shared Utility Pattern

**Location:** `backend/utils/editor_operations_resolver.py`

**Agents Using It:**
- ✅ **Outline Editing Agent** - `require_anchors=True` (strict)
- ✅ **Rules Editing Agent** - `require_anchors=True` (strict)
- ✅ **Character Development Agent** - `require_anchors=True` (strict)
- ✅ **Fiction Editing Agent** (NEW!) - `require_anchors` varies by mode (hybrid)

### Progressive Search Strategies

The resolver implements 4-level progressive search:

1. **Exact Match** (confidence 1.0)
   - Direct string matching with occurrence support
   - Fastest and most reliable

2. **Normalized Whitespace** (confidence 0.9)
   - Collapse all whitespace to single spaces
   - Handles formatting variations

3. **Sentence Boundary Match** (confidence 0.8)
   - Match first sentence, extend to expected length
   - Validates last few words present

4. **Key Phrase Anchoring** (confidence 0.7)
   - Match first 3 words + last 3 words
   - Spans content between anchors

### Anchor Types Supported

```python
{
    "original_text": "EXACT, VERBATIM text to replace (20-40 words)",
    "anchor_text": "Exact line to insert after (e.g., '## Chapter 5')",
    "left_context": "30-50 chars before target",
    "right_context": "30-50 chars after target",
    "occurrence_index": 0,  # 0=first, 1=second, etc.
}
```

---

## Fiction Agent's Hybrid Approach

### Mode-Based Resolver Configuration

#### 1. REVISION Mode (`revise`, `polish`, `tighten`, `improve`, `fix`)
```python
require_anchors=True  # ⚡ STRICT MODE
```

**Behavior:**
- Demands precise anchors (`original_text` or `left_context` + `right_context`)
- Fails loudly if anchors missing or weak
- Skips operations that can't be precisely targeted
- Strips chapter headings from replacement text
- Updates scope to 'paragraph'

**Use Case:** User asks to polish a specific paragraph or fix a sentence

**Example:**
```
User: "Polish the second paragraph to tighten the prose"
Agent: Provides exact 30-word anchor from that paragraph
Resolver: Finds exact match (confidence 1.0)
Result: Only that paragraph is modified
```

---

#### 2. TARGETED_EDIT Mode (`add`, `expand`, `more detail`, `include`)
```python
require_anchors=False  # 🎯 FLEXIBLE MODE
```

**Behavior:**
- Uses resolver for precision when anchors provided
- Falls back to paragraph scope if resolution fails
- Allows some imprecision (confidence ≥ 0.7)
- Strips chapter headings
- Updates scope to 'paragraph'

**Use Case:** User asks to add dialogue or expand a scene

**Example:**
```
User: "Add more tension to the confrontation scene"
Agent: Provides anchor to paragraph containing confrontation
Resolver: Finds match (confidence 0.8)
Fallback: Uses current paragraph if anchor fails
Result: That section is enhanced, not whole chapter rewritten
```

---

#### 3. GENERATION Mode (`write`, `generate`, `create`, `draft`)
```python
require_anchors=False  # 🏇 SMART POSITIONING
```

**Behavior:**
- Uses resolver with `anchor_text` for chapter positioning
- Automatically finds "## Chapter N" headings
- Inserts after previous chapter if target doesn't exist
- Appends after last chapter as fallback
- Adds chapter heading to generated content
- Preserves 'chapter' scope

**Use Case:** User asks to write a new chapter or generate opening scene

**Example:**
```
User: "Write Chapter 3"
Agent: Requests chapter 3 content
System: Finds "## Chapter 2" heading
Resolver: Positions insertion after Chapter 2 (confidence 1.0)
Fallback: Appends after last chapter if no chapters exist
Result: New chapter inserted at correct position
```

**Smart Positioning Logic:**
1. Check if Chapter N exists → Replace it
2. Find Chapter N-1 → Insert after it using `anchor_text="## Chapter N-1"`
3. No chapters exist → Insert after frontmatter
4. All else fails → Append to end

---

## LLM Prompt Updates

### System Prompt Additions

The agent's system prompt now includes:

```
=== ANCHORING RULES (CRITICAL FOR PRECISION!) ===

**For REVISE/DELETE Operations:**
- ALWAYS include 'original_text' with EXACT, VERBATIM text (20-40 words minimum)
- Include complete sentences with natural boundaries
- If phrase appears multiple times, set 'occurrence_index'
- ⚠️ NEVER include chapter headers (##) in original_text!

**For TARGETED EDIT Operations:**
- For replacing paragraph: Use 'original_text' with that paragraph's text
- For inserting between paragraphs: Use 'left_context'
- Prefer 'replace_range' on specific paragraph

**For GENERATION Operations:**
- Use 'insert_after_heading' with anchor_text='## Chapter N'
- System automatically positions after last chapter if no anchor provided

**Progressive Search Confidence:**
The system uses 4-level progressive search: exact match → normalized whitespace → 
sentence boundary → key phrase anchoring. Provide detailed anchors for best results.
```

### User Message Context

The agent now provides explicit guidance based on detected intent:

- **REVISION MODE** → "Make MINIMAL surgical edits only"
- **TARGETED EDIT MODE** → "Add/expand specific content while preserving the rest"
- **GENERATION MODE** → "Create complete new content"

---

## Error Handling & Fallbacks

### Revision Mode (Strict)
```python
except ValueError as e:
    if edit_intent == "revision":
        logger.error("❌ REVISION mode requires precise anchors! Skipping operation.")
        continue  # Skip operation entirely
```

**Philosophy:** Better to do nothing than make imprecise changes to prose.

### Targeted Edit Mode (Flexible)
```python
    elif edit_intent == "targeted_edit":
        logger.info("⚠️ TARGETED_EDIT: Falling back to paragraph scope")
        op.start = para_start
        op.end = para_end
        op.confidence = 0.5
```

**Philosophy:** If we can't find exact target, modify current paragraph as best guess.

### Generation Mode (Smart)
```python
    else:  # generation
        # Insert after last chapter or append to end
        if ch_ranges:
            op.start = ch_ranges[-1].end
            op.end = ch_ranges[-1].end
        else:
            op.start = fm_end_idx
            op.end = fm_end_idx
        op.confidence = 0.3
```

**Philosophy:** Always insert at sensible location - after last chapter or at document end.

---

## Benefits of This Approach

### 1. Precision Where It Matters
- Revisions target exact prose without rewriting entire chapters
- User selections are respected precisely
- Confidence scoring shows reliability of matches

### 2. Flexibility Where Needed
- Generation mode doesn't require manual positioning
- Targeted edits have fallbacks for usability
- Smart chapter detection for automatic insertion

### 3. Consistency Across Agents
- All editor-interactive agents use same resolver
- Shared code = shared improvements
- Proven patterns from outline/rules/character agents

### 4. User Experience
- "Revise chapter" → modifies specific paragraphs, not whole chapter
- "Write Chapter 5" → automatically inserts after Chapter 4
- "Add more detail here" → enhances current section only

---

## Testing Strategy

### Test Revision Mode
```
User: "Polish the dialogue in the second paragraph"
Expected: Only that paragraph's dialogue is refined
Verify: Chapter structure unchanged, other paragraphs untouched
```

### Test Targeted Edit Mode
```
User: "Add more tension to the confrontation"
Expected: Current scene enhanced with tension
Verify: Scene expanded but chapter not rewritten
```

### Test Generation Mode
```
User: "Write Chapter 5"
Expected: New chapter inserted after Chapter 4
Verify: Chapter 5 heading present, positioned correctly
```

### Test Smart Positioning
```
User: "Write Chapter 1" (in empty manuscript)
Expected: Chapter 1 inserted after frontmatter
Verify: Positioned at document start (after YAML)
```

---

## Comparison: Before vs After

### Before (Crude Anchoring)
```python
# Simple find() - first occurrence only
idx = manuscript.find(orig_text)
if idx != -1:
    start_ix = idx
    end_ix = idx + len(orig_text)
```

**Problems:**
- No confidence scoring
- No occurrence index support
- No whitespace normalization
- No fallback strategies
- Chapter rewrites for revisions

### After (Progressive Search Resolver)
```python
resolved_start, resolved_end, resolved_text, resolved_confidence = resolve_operation(
    manuscript,
    op_dict,
    selection=selection,
    frontmatter_end=fm_end_idx,
    require_anchors=(edit_intent == "revision"),
)
```

**Benefits:**
- 4-level progressive search
- Confidence scoring (0.0-1.0)
- Occurrence index support
- Whitespace normalization
- Sentence boundary matching
- Key phrase anchoring
- Smart fallbacks by mode
- Selection support

---

## Future Enhancements

### Potential Improvements

1. **Heading Hints for Generation**
   - Pass chapter context to resolver as `heading_hint`
   - Constrain search window to current chapter

2. **Confidence Thresholds**
   - Warn user if confidence < 0.8 for revisions
   - Request clarification for ambiguous anchors

3. **Multi-Operation Transactions**
   - Group related operations (e.g., "add dialogue to all scenes")
   - Validate all anchors before applying any

4. **Undo/Redo Support**
   - Store pre_hash for rollback
   - Confidence-based approval thresholds

---

## Roosevelt's Cavalry Charge Summary

**BULLY!** The Fiction Editing Agent now charges forward with the same precision as the Outline Agent's cavalry!

**Key Wins:**
- ✅ Surgical revisions (no more chapter rewrites)
- ✅ Smart chapter positioning (automatic insertion)
- ✅ Confidence-scored operations (know what's reliable)
- ✅ Shared resolver service (consistency across agents)
- ✅ Mode-based behavior (strict/flexible/smart)

**By George!** A well-organized editing system is like a well-organized cavalry charge - every operation knows its target and executes it perfectly! 🏇







