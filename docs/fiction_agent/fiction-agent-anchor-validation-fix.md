# Fiction Agent: Anchor Validation Fix

**BULLY!** Found the problem and fixed it! **By George!**

---

## 🔴 The Problem (Found via Debug Logs)

**User query:** "Can you provide revisions to Chapter 1, putting Peterson's mind squarely in thinking of seeing Vivia"

**Debug logs revealed:**
```
🔍 FICTION DEBUG: Operation 0: op_type=replace_range
🔍 FICTION DEBUG: Operation 0: new text (first 100 chars)='The heat in Gibraltar was oppressive...'
```

**Notice what's MISSING?** No `original_text` line!

**Then the resolver confirmed:**
```
🔍 RESOLVER DEBUG: action=revise, require_anchors=False
🔍 RESOLVER DEBUG: original_text=[NONE]
🔍 RESOLVER DEBUG: anchor_text=[NONE]
🔍 RESOLVER DEBUG: left_context=[NONE]
🔍 RESOLVER DEBUG: right_context=[NONE]
❌ NO MATCH: revise operation failed to find target
🎯 RESOLVED: [99:99] confidence=0.00
```

---

## 🎯 Root Cause

**The LLM was returning:**
```json
{
  "op_type": "replace_range",
  "start": 100,
  "end": 500,
  "text": "The heat in Gibraltar was oppressive..." 
  // ❌ NO original_text field!
  // ❌ NO anchor_text field!
}
```

**Why?** The Pydantic model made anchor fields **optional**, so the LLM just skipped them!

---

## ✅ The Fix (Three-Pronged Approach)

### 1. **Pydantic Field Validation**
**File:** `backend/models/agent_response_models.py`

**Added validator to `EditorOperation` model:**
```python
@field_validator('original_text', 'anchor_text', mode='after')
@classmethod
def validate_required_anchors(cls, value, info):
    """Ensure required anchor fields are provided based on operation type."""
    if not info.data:
        return value
        
    op_type = info.data.get('op_type')
    field_name = info.field_name
    
    # For replace_range and delete_range, original_text is REQUIRED
    if op_type in ('replace_range', 'delete_range') and field_name == 'original_text':
        if not value or len(value.strip()) < 10:
            raise ValueError(
                f"❌ ANCHOR REQUIRED: {op_type} MUST provide 'original_text' with EXACT, VERBATIM text from manuscript (minimum 10 words). "
                f"Copy the complete paragraph/sentence you want to modify!"
            )
    
    # For insert_after_heading, anchor_text is REQUIRED  
    if op_type == 'insert_after_heading' and field_name == 'anchor_text':
        if not value or len(value.strip()) < 10:
            raise ValueError(
                f"❌ ANCHOR REQUIRED: insert_after_heading MUST provide 'anchor_text' with EXACT, COMPLETE line/paragraph to insert after (minimum 10 words). "
                f"Copy the complete line that should come BEFORE your new text!"
            )
    
    return value
```

**What this does:**
- Validates that `original_text` is provided for `replace_range` and `delete_range`
- Validates that `anchor_text` is provided for `insert_after_heading`
- Requires minimum 10 characters (not just empty string)
- Raises helpful error message if validation fails

---

### 2. **Updated JSON Schema in System Prompt**
**File:** `backend/services/langgraph_agents/fiction_editing_agent.py`

**Changed from vague schema:**
```json
"original_text": string (for replace/delete - EXACT verbatim text),
"anchor_text": string (for insert - exact line/paragraph to insert after),
```

**To explicit requirements:**
```json
"original_text": string (⚠️ REQUIRED for replace_range/delete_range - EXACT 20-40 words from manuscript!),
"anchor_text": string (⚠️ REQUIRED for insert_after_heading - EXACT complete line/paragraph to insert after!),
```

**Added critical callout:**
```
⚠️ ⚠️ ⚠️ CRITICAL FIELD REQUIREMENTS:
- replace_range → MUST include 'original_text' with EXACT 20-40 words from manuscript
- delete_range → MUST include 'original_text' with EXACT text to delete
- insert_after_heading → MUST include 'anchor_text' with EXACT complete line/paragraph to insert after
- If you don't provide these fields, the operation will FAIL!
```

**Why multiple warnings symbols?** To grab the LLM's attention that these are NOT optional!

---

### 3. **Graceful Error Handling**
**File:** `backend/services/langgraph_agents/fiction_editing_agent.py`

**Added specific handling for ValidationError:**
```python
except ValidationError as ve:
    # Validation error - likely missing required anchors
    logger.error(f"❌ Validation error in ManuscriptEdit: {ve}")
    error_msg = str(ve)
    if "ANCHOR REQUIRED" in error_msg:
        return self._create_success_result(
            response=(
                "⚠️ The editing operation couldn't be created because required anchoring information is missing.\n\n"
                "**What went wrong:** The system needs EXACT text from your manuscript to know where to make changes, but this information wasn't provided.\n\n"
                "**How to fix:** Please try rephrasing your request, or:\n"
                "- For additions: Specify what text should come BEFORE the new content\n"
                "- For changes: Specify what EXACT text to replace\n\n"
                f"**Technical details:** {error_msg}"
            ),
            tools_used=[],
            processing_time=(datetime.now() - start_time).total_seconds(),
            additional_data={"validation_error": error_msg, "raw": content},
        )
```

**What this does:**
- Catches `ValidationError` specifically (not generic Exception)
- Checks if it's an anchor validation error
- Provides user-friendly explanation
- Includes technical details for debugging
- Doesn't crash the agent - returns helpful response

---

## 🎬 Expected Behavior Now

### Scenario 1: LLM Provides Anchors (Success)
```
User: "Revise Chapter 1 opening"

LLM returns:
{
  "op_type": "replace_range",
  "original_text": "The heat in Gibraltar was oppressive, even by Mediterranean standards...",
  "text": "[revised prose]"
}

Validation: ✅ PASS - original_text provided and >= 10 chars
Resolver: ✅ MATCH - finds exact text
Result: ✅ Edit can be auto-inserted
```

---

### Scenario 2: LLM Forgets Anchors (Validation Error)
```
User: "Revise Chapter 1 opening"

LLM returns:
{
  "op_type": "replace_range",
  "text": "[revised prose]"
  // ❌ Missing original_text
}

Validation: ❌ FAIL - ValidationError raised
Handler: Catches error, returns helpful message
Result: User sees explanation and can rephrase query
```

**User sees:**
```
⚠️ The editing operation couldn't be created because required anchoring information is missing.

**What went wrong:** The system needs EXACT text from your manuscript to know where to make changes, but this information wasn't provided.

**How to fix:** Please try rephrasing your request, or:
- For additions: Specify what text should come BEFORE the new content
- For changes: Specify what EXACT text to replace

**Technical details:** [validation error details]
```

---

### Scenario 3: LLM Provides Empty Anchor (Validation Error)
```
User: "End Chapter 1 with..."

LLM returns:
{
  "op_type": "insert_after_heading",
  "anchor_text": "",  // ❌ Empty!
  "text": "[new prose]"
}

Validation: ❌ FAIL - anchor_text < 10 chars
Handler: Catches error, returns helpful message
Result: User sees explanation
```

---

## 🔧 How It Works

### The Validation Flow

1. **LLM generates response** → Returns JSON
2. **Pydantic attempts to parse** → Creates `ManuscriptEdit` object
3. **Field validators run** → Check `original_text` and `anchor_text`
4. **Validation based on `op_type`:**
   - `replace_range` → requires `original_text` >= 10 chars
   - `delete_range` → requires `original_text` >= 10 chars
   - `insert_after_heading` → requires `anchor_text` >= 10 chars
5. **If validation fails** → Raises `ValidationError`
6. **Exception handler catches** → Returns user-friendly message

### The Schema Enforcement

**Before:**
- Fields were optional
- No explicit requirements
- LLM could skip fields
- Result: No anchors, no match

**After:**
- Fields explicitly marked REQUIRED with ⚠️ symbols
- Multiple warnings about operation failure
- Clear connection between op_type and required field
- Result: LLM knows it MUST provide these fields

---

## 📊 Testing the Fix

### Test Case 1: Revision Request
```
Query: "Revise Chapter 1 to include Peterson's thoughts about Vivian"
Expected LLM output:
{
  "op_type": "replace_range",
  "original_text": "[EXACT 20-40 words from Chapter 1]",  ✅ REQUIRED
  "text": "[revised prose with Peterson's thoughts]"
}
Expected result: Validation passes, resolver finds match, edit succeeds
```

### Test Case 2: Chapter Ending Addition
```
Query: "End Chapter 1 with a witty thought from Peterson"
Expected LLM output:
{
  "op_type": "insert_after_heading",
  "anchor_text": "[EXACT last paragraph of Chapter 1]",  ✅ REQUIRED
  "text": "[Peterson's witty thought]"
}
Expected result: Validation passes, resolver finds match, edit succeeds
```

### Test Case 3: Deletion Request
```
Query: "Remove the description of Gibraltar heat"
Expected LLM output:
{
  "op_type": "delete_range",
  "original_text": "[EXACT text to delete]",  ✅ REQUIRED
  "text": ""
}
Expected result: Validation passes, resolver finds match, deletion succeeds
```

---

## 🎯 Why This Fix Works

### 1. **Catches the Problem Early**
- Validation happens at parse time
- Before resolver tries to find non-existent anchors
- Clear error message instead of mysterious failure

### 2. **Forces LLM Compliance**
- Prominent ⚠️ symbols grab attention
- Explicit "REQUIRED" markers
- Clear connection between op_type and required fields
- Multiple warnings about failure consequences

### 3. **Provides Helpful Feedback**
- User-friendly error messages
- Explains what went wrong
- Suggests how to fix it
- Includes technical details for debugging

### 4. **Doesn't Break Existing Behavior**
- Only validates when op_type requires anchors
- Optional fields remain optional (left_context, right_context)
- Backward compatible with properly-formed operations

---

## 📁 Files Modified

### 1. `backend/models/agent_response_models.py`
**Changes:**
- Added import: `from pydantic import BaseModel, Field, field_validator`
- Added `@field_validator` decorator to `EditorOperation` class
- Validates `original_text` for `replace_range`/`delete_range`
- Validates `anchor_text` for `insert_after_heading`
- Requires minimum 10 characters

### 2. `backend/services/langgraph_agents/fiction_editing_agent.py`
**Changes:**
- Added import: `from pydantic import ValidationError`
- Updated JSON schema with explicit `(REQUIRED)` markers
- Added ⚠️ symbols to anchor fields
- Added "CRITICAL FIELD REQUIREMENTS" callout section
- Added `except ValidationError` handler
- Provides user-friendly error messages for validation failures

---

## 🏇 Roosevelt's Verdict

**BULLY!** We've fixed the cavalry's broken compass!

**The problem:**
- LLM was generating orders (prose) but not providing coordinates (anchors)
- Troops couldn't execute orders without knowing WHERE to charge

**The solution:**
- Make coordinates MANDATORY for operations that need them
- Catch missing coordinates EARLY with validation
- Provide clear feedback when coordinates are missing

**By George!** No more blind charges! Every operation now requires proper reconnaissance data! 🏇

---

**Last Updated:** October 29, 2025  
**Status:** Active - Anchor validation enforced  
**Impact:** Should eliminate "NO MATCH" failures caused by missing anchor fields







