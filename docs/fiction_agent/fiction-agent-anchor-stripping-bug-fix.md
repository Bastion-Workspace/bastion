# Fiction Agent: Anchor Stripping Bug Fix

**By George!** Found the REAL problem! **BULLY!**

---

## 🔴 The Actual Bug

The validation wasn't failing because **the normalization code was STRIPPING OUT the anchor fields before validation could run**!

### The Smoking Gun (Lines 622-628)

**Before (BAD):**
```python
text_val = op.get("text") or ""
norm_ops.append({
    "op_type": op_type,
    "start": start_ix,
    "end": end_ix,
    "text": text_val,
    "pre_hash": ""  # ❌ ONLY THESE FIELDS - ANCHORS DISCARDED!
})
```

**What this did:**
1. LLM returns JSON with `original_text` field
2. Fallback normalization code processes the JSON
3. **Creates new dict with ONLY 5 fields** (op_type, start, end, text, pre_hash)
4. **Discards** `original_text`, `anchor_text`, and all anchor fields
5. Passes stripped dict to `ManuscriptEdit(**raw_obj)`
6. Pydantic sets missing fields to None (default)
7. Validation runs but... there's nothing to validate (fields were already stripped)

---

## ✅ The Fix (Two Parts)

### 1. Preserve Anchor Fields in Normalization

**File:** `backend/services/langgraph_agents/fiction_editing_agent.py`

**After (GOOD):**
```python
text_val = op.get("text") or ""

# **ROOSEVELT: PRESERVE ANCHOR FIELDS - Don't strip them out!**
normalized_op = {
    "op_type": op_type if op_type in ("replace_range", "delete_range", "insert_after_heading") else "replace_range",
    "start": max(0, min(len(manuscript), start_ix)),
    "end": max(0, min(len(manuscript), end_ix)),
    "text": text_val,
    "pre_hash": "",  # placeholder, will compute below
    # Preserve anchor fields for validation and resolver
    "original_text": op.get("original_text") or op.get("original"),  ✅
    "anchor_text": op.get("anchor_text"),  ✅
    "left_context": op.get("left_context"),  ✅
    "right_context": op.get("right_context"),  ✅
    "occurrence_index": op.get("occurrence_index", 0),  ✅
}
norm_ops.append(normalized_op)
```

**What this does:**
- **Preserves all anchor fields** from the LLM's JSON
- Passes them through to Pydantic validation
- Allows resolver to use them

---

### 2. Split Validators for Clarity

**File:** `backend/models/agent_response_models.py`

**Before:**
```python
@field_validator('original_text', 'anchor_text', mode='after')
@classmethod
def validate_required_anchors(cls, value, info):
    # One validator for both fields
    field_name = info.field_name  # This might not work correctly
    ...
```

**After:**
```python
@field_validator('original_text', mode='after')
@classmethod
def validate_original_text(cls, value, info):
    """Ensure original_text is provided for replace_range and delete_range."""
    if not info.data:
        return value
        
    op_type = info.data.get('op_type')
    
    # For replace_range and delete_range, original_text is REQUIRED
    if op_type in ('replace_range', 'delete_range'):
        if not value or (isinstance(value, str) and len(value.strip()) < 10):
            raise ValueError(
                f"❌ ANCHOR REQUIRED: {op_type} operations MUST provide 'original_text' field..."
            )
    
    return value

@field_validator('anchor_text', mode='after')
@classmethod
def validate_anchor_text(cls, value, info):
    """Ensure anchor_text is provided for insert_after_heading."""
    if not info.data:
        return value
        
    op_type = info.data.get('op_type')
    
    # For insert_after_heading, anchor_text is REQUIRED  
    if op_type == 'insert_after_heading':
        if not value or (isinstance(value, str) and len(value.strip()) < 10):
            raise ValueError(
                f"❌ ANCHOR REQUIRED: insert_after_heading operations MUST provide 'anchor_text' field..."
            )
    
    return value
```

**Why split?**
- Clearer intent (one validator per field)
- Better error messages
- More reliable in Pydantic v2

---

## 🎯 The Flow (Fixed)

### Before (Broken)
```
1. LLM returns: {"op_type": "replace_range", "original_text": "...", "text": "..."}
2. Normalization: Strip to {"op_type": "replace_range", "text": "..."}  ❌
3. Pydantic: Sets original_text = None (default)
4. Validation: Can't fail - field was already stripped!
5. Resolver: original_text=[NONE], can't find match
6. Result: ❌ NO MATCH confidence=0.00
```

### After (Fixed)
```
1. LLM returns: {"op_type": "replace_range", "original_text": "...", "text": "..."}
2. Normalization: Preserves {"op_type": "replace_range", "original_text": "...", "text": "..."}  ✅
3. Pydantic: original_text = "..."
4. Validation: Checks if original_text >= 10 chars  ✅
5. Resolver: original_text=[PRESENT], finds exact match
6. Result: ✅ EXACT MATCH confidence=1.00
```

### If LLM Doesn't Provide Anchor (Now Caught)
```
1. LLM returns: {"op_type": "replace_range", "text": "..."}  ❌ Missing original_text!
2. Normalization: Preserves {"op_type": "replace_range", "original_text": null, "text": "..."}
3. Pydantic: original_text = None
4. Validation: Fails! Raises ValueError  ✅
5. Exception handler: Returns user-friendly error message
6. User sees: "⚠️ Anchoring information is missing. Please try rephrasing..."
```

---

## 📊 Why Outline Agent Works

The user mentioned: "This works so well in our outline agent :'("

**Why outline agent works:**
- Probably doesn't have the same fallback normalization code
- Or has different/stricter normalization that preserves anchors
- Or uses structured output directly without stripping

**Why fiction agent didn't work:**
- Had overly-permissive fallback normalization
- Stripped anchor fields thinking they were "optional hints"
- Bypassed validation before it could run

---

## 🧪 Testing the Fix

### Test 1: LLM Provides Anchors
```
LLM returns:
{
  "op_type": "replace_range",
  "original_text": "The heat in Gibraltar was oppressive, even by Mediterranean standards...",
  "text": "[revised prose]"
}

Expected flow:
✅ Normalization preserves original_text
✅ Validation passes (original_text >= 10 chars)
✅ Resolver finds exact match
✅ Edit succeeds with confidence=1.00
```

### Test 2: LLM Forgets Anchors
```
LLM returns:
{
  "op_type": "replace_range",
  "text": "[revised prose]"
  // ❌ Missing original_text
}

Expected flow:
✅ Normalization preserves null original_text
❌ Validation fails with ValueError
✅ Exception handler catches error
✅ User sees helpful message
```

### Test 3: LLM Provides Empty Anchor
```
LLM returns:
{
  "op_type": "replace_range",
  "original_text": "",  // ❌ Empty!
  "text": "[revised prose]"
}

Expected flow:
✅ Normalization preserves empty string
❌ Validation fails (len < 10)
✅ Exception handler catches error
✅ User sees helpful message
```

---

## 📁 Files Modified

### 1. `backend/services/langgraph_agents/fiction_editing_agent.py`

**Lines 621-637:**
- Changed normalization dict construction
- Added **preservation of anchor fields**
- Added comment explaining why

**Before:** 5 fields preserved (op_type, start, end, text, pre_hash)  
**After:** 10 fields preserved (added original_text, anchor_text, left_context, right_context, occurrence_index)

---

### 2. `backend/models/agent_response_models.py`

**Lines 525-561:**
- Split single validator into two separate validators
- One for `original_text` (replace_range/delete_range)
- One for `anchor_text` (insert_after_heading)
- More explicit error messages

**Why split?**
- Clearer code
- Better error messages
- More reliable with Pydantic v2

---

## 🏇 Roosevelt's Verdict

**BULLY!** The problem was right under our noses!

**The bug:**
- Overly-helpful fallback code was **too helpful**
- Stripped out important anchor information
- Validation never got a chance to run

**The fix:**
- **Stop being too helpful** - preserve what the LLM provides
- Let validation do its job
- Catch missing anchors early with clear errors

**By George!** Sometimes the cavalry needs to trust the scout's map instead of throwing it away and guessing! 🏇

---

**Last Updated:** October 29, 2025  
**Status:** Active - Anchor fields now preserved through normalization  
**Impact:** Validation can now properly enforce required anchor fields







