# Fiction Agent: Exact Text Matching Fix

**BULLY!** Fixing the anchor resolution failure! **By George!**

---

## The Problem

**User Query:** "Can we end Chapter 1 with a witty thought from Peterson?"

**LLM Response:** Great prose! But the edit couldn't be auto-inserted.

**Logs showed:**
```
❌ NO MATCH: revise operation failed to find target
🎯 RESOLVED: [99:99] confidence=0.00
⚠️ Low confidence anchor (0.00) - operation may be imprecise
```

**Root cause:** The LLM didn't provide EXACT anchor text for the `insert_after_heading` operation.

---

## What Happened

### The Manuscript (Last paragraph of Chapter 1):
```
"I suspect nothing yet," Fleet replied. "But a billion dollars is an extraordinary amount to spend on nostalgia alone."
```

### What the LLM Likely Provided (Incomplete):
```json
{
  "op_type": "insert_after_heading",
  "anchor_text": "But a billion dollars is an extraordinary amount to spend on nostalgia alone.",
  "text": "\n\nI nodded, already making mental notes..."
}
```

**Missing:** The dialogue attribution ("I suspect nothing yet," Fleet replied.) and the first part of the quote!

### What the Resolver Did:
1. Searched for the incomplete anchor text
2. Found NO EXACT MATCH (because manuscript has complete paragraph)
3. Tried normalized whitespace search - still no match
4. Tried sentence boundary search - still no match
5. Failed with confidence=0.00
6. Fell back to position [99:99] (basically nowhere)

**Result:** User got good prose but couldn't auto-insert it.

---

## The Fix

### Enhanced System Prompt with Prominent Warnings

**Added at the TOP of operations section:**

```
=== ⚠️ CRITICAL: EXACT TEXT MATCHING REQUIREMENT ===

**For ALL operations, anchor text MUST be 100% EXACT:**
- Copy COMPLETE paragraphs/sentences from manuscript (no shortening!)
- Include ALL dialogue tags, quotation marks, punctuation
- Match whitespace, line breaks, and formatting EXACTLY
- NEVER paraphrase, summarize, or "close enough" - must be VERBATIM
- Think: mentally COPY-PASTE the exact text from manuscript

**Why this matters:**
The system uses progressive search to find your anchor text. If your anchor doesn't
match EXACTLY, the system can't find it, and your edit will fail (confidence=0.00).
Even small differences (missing quote, incomplete sentence) cause failures.

**Example of FAILURE:**
Manuscript: '"I suspect nothing yet," Fleet replied. "But a billion dollars is an extraordinary amount to spend on nostalgia alone."'
Your anchor: "But a billion dollars is an extraordinary amount to spend on nostalgia alone."  ❌ INCOMPLETE!
Result: ❌ NO MATCH - confidence=0.00 - edit fails

**Example of SUCCESS:**
Manuscript: '"I suspect nothing yet," Fleet replied. "But a billion dollars is an extraordinary amount to spend on nostalgia alone."'
Your anchor: "\"I suspect nothing yet,\" Fleet replied. \"But a billion dollars is an extraordinary amount to spend on nostalgia alone.\""  ✅ COMPLETE!
Result: ✅ EXACT MATCH - confidence=1.00 - edit succeeds
```

---

### Updated `insert_after_heading` Guidance

**Enhanced the operation definition:**

```
**2. insert_after_heading**: Insert new text AFTER a specific location WITHOUT replacing
   USE WHEN: User wants to add, append, or insert new content (not replace existing)
   ANCHORING: Provide 'anchor_text' with EXACT, COMPLETE, VERBATIM paragraph/sentence to insert after
   ⚠️ CRITICAL: anchor_text must be 100% EXACT match from manuscript (not paraphrased!)
   EXAMPLES:
   - "End Chapter 1 with Peterson's thought" → insert_after last paragraph before '## Chapter 2'
     anchor_text: "\"I suspect nothing yet,\" Fleet replied. \"But a billion dollars is an extraordinary amount to spend on nostalgia alone.\""
   - "Add a scene transition" → insert_after current paragraph
     anchor_text: [EXACT text of current paragraph]
   
   ⚠️ CRITICAL FOR 'END CHAPTER' REQUESTS:
   When user says "end Chapter X with..." → ALWAYS use insert_after_heading
   - Find the LAST PARAGRAPH of target chapter (before next ## Chapter heading)
   - Copy that last paragraph EXACTLY as anchor_text (complete, verbatim, with punctuation)
   - Your new text will be inserted AFTER that paragraph
   - Do NOT use replace_range unless user explicitly asks to REVISE the existing ending
```

---

### Updated Chapter Ending Example

**Fixed the example to show COMPLETE paragraph:**

```
**Example - Ending Chapter 1:**
Last paragraph in manuscript (COMPLETE): '"I suspect nothing yet," Fleet replied. "But a billion dollars is an extraordinary amount to spend on nostalgia alone."'
Next line: "## Chapter 2"

✅ CORRECT (insert_after with COMPLETE paragraph as anchor):
{
  "op_type": "insert_after_heading",
  "anchor_text": "\"I suspect nothing yet,\" Fleet replied. \"But a billion dollars is an extraordinary amount to spend on nostalgia alone.\"",
  "text": "\n\nI nodded, already making mental notes about what to pack..."
}

⚠️ CRITICAL: Copy the ENTIRE last paragraph exactly as written! Include:
- All dialogue tags ("Fleet replied")
- All punctuation and quotation marks
- Complete sentences from start to end
- No paraphrasing or shortening!
```

**Before (Incomplete example):**
```json
{
  "anchor_text": "But a billion dollars is an extraordinary amount to spend on nostalgia alone."
}
```

**After (Complete example):**
```json
{
  "anchor_text": "\"I suspect nothing yet,\" Fleet replied. \"But a billion dollars is an extraordinary amount to spend on nostalgia alone.\""
}
```

---

## Why This Matters

### The Progressive Search System

The resolver uses 4 levels of matching:
1. **Exact match (1.0)** - Perfect character-for-character match
2. **Normalized whitespace (0.9)** - Allows whitespace differences
3. **Sentence boundary (0.8)** - Matches sentence starts/ends
4. **Key phrase (0.7)** - Matches first/last few words

**If you provide incomplete text:**
- Exact match fails (missing beginning of paragraph)
- Normalized fails (still missing text)
- Sentence boundary fails (doesn't start at sentence boundary)
- Key phrase might partially match but with low confidence
- **Result:** confidence < 0.7 → operation fails

**If you provide EXACT, COMPLETE text:**
- Exact match succeeds immediately
- **Result:** confidence = 1.0 → operation succeeds perfectly

---

## Expected Behavior After Fix

### User Query:
"Can we end Chapter 1 with a witty thought from Peterson?"

### LLM Should Now Return:
```json
{
  "target_filename": "Sidney Fleet 7 - Vintage Fleet.md",
  "operations": [{
    "op_type": "insert_after_heading",
    "anchor_text": "\"I suspect nothing yet,\" Fleet replied. \"But a billion dollars is an extraordinary amount to spend on nostalgia alone.\"",
    "text": "\n\nI nodded, already making mental notes about what to pack. At least this time I wouldn't have to worry about Fleet complaining about the quality of the coffee - though knowing him, he'd probably find something wrong with a billion-dollar ship's espresso machine within the first hour.",
    "occurrence_index": 0
  }],
  "scope": "paragraph",
  "summary": "Added witty ending to Chapter 1 from Peterson's perspective"
}
```

### Resolver Behavior:
```
✅ EXACT MATCH: Found anchor at position 1234:1350
🎯 RESOLVED: [1234:1350] confidence=1.00
📍 FICTION: insert_after_heading resolved [1234:1350] confidence=1.00
```

### User Experience:
- ✅ Good prose generated
- ✅ Anchor found with 100% confidence
- ✅ Edit can be auto-inserted
- ✅ User clicks "Apply" button
- ✅ Text inserted perfectly after last paragraph

---

## Key Changes Summary

### 1. Prominent Warning Section
- Added at TOP of operations guidance
- Explains EXACT text requirement
- Shows FAILURE vs SUCCESS example
- Uses the ACTUAL user scenario

### 2. Enhanced Operation Definitions
- Added "EXACT, COMPLETE, VERBATIM" to insert_after_heading
- Added explicit warning about 100% exact match
- Added concrete examples with complete paragraphs

### 3. Fixed Chapter Ending Example
- Shows COMPLETE last paragraph (including dialogue tags)
- Demonstrates correct anchor_text format
- Adds explicit checklist of what to include

### 4. Multiple Reinforcements
- "CRITICAL FOR 'END CHAPTER' REQUESTS" section
- Checklist in chapter boundaries section
- Concrete failure/success comparison

---

## Testing the Fix

### Test Case 1: End Chapter with Addition
```
User: "End Chapter 1 with a witty thought from Peterson"
Expected: LLM uses insert_after_heading with COMPLETE last paragraph as anchor
Result: Should resolve with confidence=1.00
```

### Test Case 2: Add Scene Transition
```
User: "Add a scene transition after the confrontation"
Expected: LLM uses insert_after_heading with COMPLETE confrontation paragraph
Result: Should resolve with confidence ≥ 0.9
```

### Test Case 3: Append to Chapter
```
User: "Add another paragraph to Chapter 2"
Expected: LLM uses insert_after_heading with last paragraph of Chapter 2
Result: Should resolve with confidence=1.00
```

---

## Files Modified

**`backend/services/langgraph_agents/fiction_editing_agent.py`**

**Changes:**
1. Added "⚠️ CRITICAL: EXACT TEXT MATCHING REQUIREMENT" section (lines 140-158)
   - Prominent warning at top
   - Failure vs success example
   - Explains why exact matching matters

2. Enhanced `insert_after_heading` definition (lines 165-183)
   - Added "EXACT, COMPLETE, VERBATIM" to anchoring description
   - Added critical warning about 100% exact match
   - Updated examples to show complete paragraphs
   - Added "CRITICAL FOR 'END CHAPTER' REQUESTS" subsection

3. Fixed chapter ending example (lines 196-211)
   - Shows COMPLETE last paragraph including dialogue tags
   - Demonstrates correct JSON format
   - Adds checklist of what to include

---

## Roosevelt's Verdict

**BULLY!** The LLM now has crystal-clear instructions about EXACT text matching!

**Key principles:**
- ✅ COMPLETE paragraphs, not fragments
- ✅ Include ALL dialogue tags and punctuation
- ✅ VERBATIM copy, never paraphrase
- ✅ Think "mentally copy-paste" from manuscript
- ✅ Concrete example showing FAILURE vs SUCCESS

**By George!** No more "close enough" anchor text! The cavalry must hit the EXACT target! 🏇

---

**Last Updated:** October 29, 2025  
**Status:** Active - Enhanced exact text matching guidance  
**Impact:** Should eliminate anchor resolution failures for insert operations







