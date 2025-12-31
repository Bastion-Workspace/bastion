# Fiction Agent: Full Adjacent Chapters for Context

**BULLY!** Providing complete context for proper revision and generation! **By George!**

---

## Change Summary

**From:** Sending 1200-character excerpts of adjacent chapters  
**To:** Sending full adjacent chapters (1 before, 1 after) with clear labeling

---

## Why This Matters

### The Problem with Excerpts
**Previous approach:**
```python
f"Previous Chapter (excerpt):\n{prev_chapter_text[:1200]}\n\n"
f"Next Chapter (excerpt):\n{next_chapter_text[:1200]}\n\n"
```

**Issues:**
- LLM couldn't see full chapter transitions
- Missed tone/mood shifts that occur mid-chapter
- Couldn't properly assess if changes affect subsequent content
- Character emotional state might change after excerpt cutoff

### The Solution: Full Chapters
**New approach:**
```python
=== CHAPTER 1 (FOR CONTEXT - DO NOT EDIT) ===
[Full Chapter 1 text]

=== CHAPTER 2 (CURRENT WORK AREA) ===
[Full Chapter 2 text - this is what you edit]

=== CHAPTER 3 (FOR CONTEXT - DO NOT EDIT) ===
[Full Chapter 3 text]
```

**Benefits:**
- ✅ LLM sees complete tone/mood arcs in adjacent chapters
- ✅ Understands full character state transitions
- ✅ Can properly assess if edits create inconsistencies
- ✅ Knows exact context for chapter-ending additions
- ✅ Clear labeling prevents accidental edits to wrong chapter

---

## Implementation Details

### Context Structure

**What we send:**

1. **Previous Chapter (Chapter N-1)** - IF EXISTS
   - Full text (frontmatter stripped)
   - Labeled: "CHAPTER N-1 (FOR CONTEXT - DO NOT EDIT)"
   - Purpose: Tone, pacing, character state leading into current chapter

2. **Current Chapter (Chapter N)** - ALWAYS
   - Full text (frontmatter stripped)
   - Labeled: "CHAPTER N (CURRENT WORK AREA)"
   - Purpose: The chapter being edited

3. **Paragraph Around Cursor** - ALWAYS
   - Extracted from current chapter
   - Labeled: "PARAGRAPH AROUND CURSOR"
   - Purpose: Specific focus area

4. **Next Chapter (Chapter N+1)** - IF EXISTS
   - Full text (frontmatter stripped)
   - Labeled: "CHAPTER N+1 (FOR CONTEXT - DO NOT EDIT)"
   - Purpose: Transition checking, ensure edits don't create inconsistency

---

### Clear Chapter Labeling

**Code:**
```python
# Build context with full adjacent chapters for consistency analysis
prev_chapter_text = None
prev_chapter_label = None

if prev_c:
    prev_chapter_text = _strip_frontmatter_block(manuscript[prev_c.start:prev_c.end])
    prev_chapter_num = prev_c.chapter_number or "Previous"
    prev_chapter_label = f"Chapter {prev_chapter_num}" if prev_c.chapter_number else "Previous Chapter"

# Current chapter label
current_chapter_label = f"Chapter {current_chapter_number}" if current_chapter_number else "Current Chapter"

# Build message with clear sections
content = (
    "=== MANUSCRIPT CONTEXT ===\n"
    f"Primary file: {filename}\n"
    f"Working area: {current_chapter_label}\n"
    f"Cursor position: paragraph shown below\n\n"
    + (f"=== {prev_chapter_label.upper()} (FOR CONTEXT - DO NOT EDIT) ===\n{prev_chapter_text}\n\n" if prev_chapter_text else "")
    + f"=== {current_chapter_label.upper()} (CURRENT WORK AREA) ===\n{context_current_chapter_text}\n\n"
    + f"=== PARAGRAPH AROUND CURSOR ===\n{context_paragraph_text}\n\n"
    + (f"=== {next_chapter_label.upper()} (FOR CONTEXT - DO NOT EDIT) ===\n{next_chapter_text}\n\n" if next_chapter_text else "")
)
```

**Key features:**
- Clear section headers with chapter numbers
- Explicit "(FOR CONTEXT - DO NOT EDIT)" warnings
- Explicit "(CURRENT WORK AREA)" designation
- Cursor position specified
- Adjacent chapters only if they exist

---

### Enhanced System Prompt

**Added section explaining context structure:**

```
=== CRITICAL CONTENT BOUNDARIES ===

**Context Structure:**
You will receive:
- PREVIOUS CHAPTER (for context) - Full text for tone/continuity understanding
- CURRENT CHAPTER (work area) - The chapter you are editing
- PARAGRAPH AROUND CURSOR - Specific focus area within current chapter
- NEXT CHAPTER (for context) - Full text for transition/consistency checking

**Your operations must ONLY target CURRENT CHAPTER:**
- 'original_text' and 'anchor_text' must reference CURRENT CHAPTER text only
- Previous/Next chapters are for context (tone, transitions, continuity)
- NEVER create operations targeting text from Previous or Next chapters
- Reference materials (Outline, Rules, Style, Characters) are GUIDANCE only

**Use adjacent chapters to:**
- Ensure tone consistency when adding/revising content
- Check if ending current chapter affects beginning of next chapter
- Maintain character emotional state continuity
- Understand pacing and rhythm across chapter transitions

**Example:** If adding witty ending to Chapter 1, check if Chapter 2 opens with a tone
that would be inconsistent with that ending. If so, note this in your 'summary' field.
```

---

## Use Cases Enabled

### 1. Chapter-Ending Additions
**Query:** "End Chapter 1 with a witty thought from Peterson"

**LLM can now:**
- See full Chapter 1 to understand current tone
- See full Chapter 2 to check opening tone
- Assess if witty ending creates inconsistency with Chapter 2's serious opening
- Suggest tone adjustment or note potential issue in summary

**Before:** Only saw first 1200 chars of Chapter 2 - might miss tone shift

---

### 2. Character Emotional State
**Query:** "Add more tension to the confrontation scene"

**LLM can now:**
- See character emotional state at end of previous chapter
- Understand where character should be emotionally in current chapter
- See how character state continues in next chapter
- Ensure added tension flows naturally into next chapter's opening

**Before:** Might add tension that contradicts next chapter's calm opening

---

### 3. Plot Continuity
**Query:** "Revise the reveal to be more impactful"

**LLM can now:**
- See how previous chapter set up the reveal
- Understand full context of current reveal
- Check if next chapter references/builds on the reveal properly
- Ensure revision doesn't create plot holes

**Before:** Might revise reveal in way that breaks next chapter's references

---

### 4. Pacing Assessment
**Query:** "Improve pacing in the middle section"

**LLM can now:**
- See pacing in previous chapter
- Understand full pacing arc of current chapter
- See pacing in next chapter
- Ensure pacing changes flow naturally across transitions

**Before:** Only saw chapter beginnings/endings - missed full pacing arc

---

## Token Considerations

### Concern: More Tokens = Higher Cost?

**Analysis:**

**Typical chapter length:** 2,000-4,000 words (10,000-20,000 chars)

**Previous approach:**
- Prev excerpt: 1,200 chars
- Current chapter: 10,000-20,000 chars
- Next excerpt: 1,200 chars
- **Total:** ~12,400-22,400 chars

**New approach:**
- Prev chapter: 10,000-20,000 chars
- Current chapter: 10,000-20,000 chars
- Next chapter: 10,000-20,000 chars
- **Total:** ~30,000-60,000 chars

**Increase:** ~2-3x more tokens for adjacent chapters

**BUT:**
- Better context = better results = fewer regenerations
- Prevents inconsistencies that require follow-up fixes
- Most operations already have good context
- Only paid when chapters exist (new manuscripts have no adjacent chapters)

**Verdict:** Worth the token cost for quality improvement! ✅

---

## Example Message Structure

**Editing Chapter 2 of 3:**

```
=== MANUSCRIPT CONTEXT ===
Primary file: my-novel.md
Working area: Chapter 2
Cursor position: paragraph shown below

=== CHAPTER 1 (FOR CONTEXT - DO NOT EDIT) ===
The sun set over the harbor, painting the water in shades of crimson and gold. 
Fleet stood at the rail of his yacht, binoculars trained on the distant shore...

[Full Chapter 1 text - ~3000 words]

=== CHAPTER 2 (CURRENT WORK AREA) ===
Morning light found us still anchored in the bay. Fleet had been up all night, 
his telescope pointed at the merchant vessels clustered near the docks...

[Full Chapter 2 text - ~3000 words]

=== PARAGRAPH AROUND CURSOR ===
"You suspect nothing yet," Fleet replied. "But a billion dollars is an 
extraordinary amount to spend on nostalgia alone."

=== CHAPTER 3 (FOR CONTEXT - DO NOT EDIT) ===
The train to Southampton departed at noon. Fleet insisted on a private 
compartment, citing the need for "uninterrupted contemplation"...

[Full Chapter 3 text - ~3000 words]

=== CURRENT CHAPTER OUTLINE (beats to follow) ===
[Outline for Chapter 2]

=== RULES (universe constraints) ===
[Rules content]

=== STYLE GUIDE (voice and tone) ===
[Style guide content]

⚠️ CRITICAL: Your operations must target CHAPTER 2 ONLY. 
Adjacent chapters are for context (tone, transitions, continuity) - DO NOT edit them!

Provide a ManuscriptEdit JSON plan for the current work area.
```

---

## Comparison to Desktop App

**Desktop app approach:**
- Current chapter
- 2 chapters before
- 2 chapters after

**Our approach:**
- 1 chapter before
- Current chapter
- 1 chapter after

**Why 1 before/after is sufficient:**
- Most tone/continuity issues are with immediate neighbors
- Rare for changes in Chapter 2 to affect Chapter 5
- Keeps token usage reasonable
- Can expand to 2 before/after if needed later

---

## Safety Features

### 1. Explicit Warnings
```
⚠️ CRITICAL: Your operations must target CHAPTER 2 ONLY.
Adjacent chapters are for context - DO NOT edit them!
```

### 2. Clear Labeling
```
=== CHAPTER 1 (FOR CONTEXT - DO NOT EDIT) ===
=== CHAPTER 2 (CURRENT WORK AREA) ===
=== CHAPTER 3 (FOR CONTEXT - DO NOT EDIT) ===
```

### 3. System Prompt Reinforcement
```
**Your operations must ONLY target CURRENT CHAPTER:**
- 'original_text' and 'anchor_text' must reference CURRENT CHAPTER text only
- NEVER create operations targeting text from Previous or Next chapters
```

### 4. Content Boundaries Section
Explicitly teaches LLM:
- What is editable (current chapter)
- What is reference (adjacent chapters, outline, rules, etc.)
- How to use context appropriately

---

## Testing Scenarios

### Test 1: Chapter Ending Addition
```
Active: Chapter 1
Query: "End Chapter 1 with a witty thought from Peterson"

Expected:
- Operation targets Chapter 1 text only
- LLM notes if witty ending conflicts with Chapter 2's serious opening
- Suggests tone adjustment if needed
```

### Test 2: Character State Continuity
```
Active: Chapter 3
Query: "Revise the confrontation to show more anger"

Expected:
- LLM checks Chapter 2 for character's emotional buildup
- LLM checks Chapter 4 to ensure anger carries through appropriately
- Operation maintains continuity across chapter boundaries
```

### Test 3: Tone Consistency
```
Active: Chapter 2
Query: "Add more humor to lighten the mood"

Expected:
- LLM checks if Chapter 1 ended seriously (would make humor jarring)
- LLM checks if Chapter 3 continues serious tone (would make humor inconsistent)
- Notes in summary if humor creates tone whiplash
```

---

## Files Modified

**`llm-orchestrator/orchestrator/agents/fiction_editing_agent.py`**

**Changes:**
1. **Context building** (lines 357-402)
   - Full adjacent chapters instead of excerpts
   - Clear chapter labeling with numbers
   - Explicit "FOR CONTEXT - DO NOT EDIT" warnings
   - "CURRENT WORK AREA" designation

2. **System prompt** (lines 212-230)
   - Added "CRITICAL CONTENT BOUNDARIES" section
   - Context structure explanation
   - Usage guidelines for adjacent chapters
   - Example of tone consistency checking

---

## Roosevelt's Verdict

**BULLY!** Full adjacent chapters provide proper reconnaissance for the editing cavalry!

**The LLM can now:**
- ✅ See complete tone arcs across chapter transitions
- ✅ Understand full character emotional states
- ✅ Assess continuity implications of changes
- ✅ Provide better-informed revisions and additions

**By George!** No more blind guessing about what comes before and after! The cavalry now has complete battlefield intelligence! 🏇

---

**Last Updated:** October 29, 2025  
**Status:** Active - Full adjacent chapters for context  
**Token Impact:** ~2-3x increase for adjacent chapters (worth it for quality!)







