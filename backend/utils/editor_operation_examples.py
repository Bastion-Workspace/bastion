"""
Roosevelt's Editor Operation Examples - The Big Stick Guide to Precise Edits

BULLY! This module provides crystal-clear examples for LLM agents on how to 
generate precise editor operations without gobbling up headers or structure!
"""

EDITOR_OPERATION_EXAMPLES = """
=== ROOSEVELT'S THREE-PRONGED EDITOR OPERATION GUIDE ===

**By George!** Follow these examples like a well-organized cavalry charge!

---
OPERATION 1: insert_after_heading
---
USE WHEN: Adding new content below a header WITHOUT removing the header

✅ CORRECT Example - Adding traits after a header:
{
    "op_type": "insert_after_heading",
    "start": 0,
    "end": 0,
    "anchor_text": "### Traits",
    "text": "- Analytical thinker\\n- Selfish in most matters\\n- Deeply protective of his sister Jill",
    "pre_hash": "",
    "note": "Adding character traits after Traits header"
}

✅ CORRECT Example - Adding relationship info after header:
{
    "op_type": "insert_after_heading",
    "start": 0,
    "end": 0,
    "anchor_text": "## Relationships",
    "text": "- **Relationship Type**: Sibling bond\\n- **Dynamics**: Jack cares deeply for Jill",
    "pre_hash": "",
    "note": "Adding relationship details"
}

❌ WRONG - Don't use replace_range with header included:
{
    "op_type": "replace_range",
    "original_text": "### Traits\\n- [To be developed]",  // ❌ NEVER include the header!
    "text": "- Analytical thinker",
    "note": "This will DELETE the header!"
}

---
OPERATION 2: replace_range
---
USE WHEN: Replacing existing content WITHOUT touching headers above it

✅ CORRECT Example - Replacing placeholder text:
{
    "op_type": "replace_range",
    "start": 0,
    "end": 0,
    "original_text": "- [To be developed based on story needs]",  // ONLY the content, NOT the header
    "text": "- Analytical thinker\\n- Selfish in most matters\\n- Deeply protective of his sister Jill",
    "pre_hash": "",
    "note": "Replacing placeholder with actual traits"
}

✅ CORRECT Example - Updating speech patterns:
{
    "op_type": "replace_range",
    "start": 0,
    "end": 0,
    "original_text": "- Speaks with a German accent\\n- [Specific speech patterns and vocabulary to be developed]\\n- [Common phrases or expressions to be added]",
    "text": "- Speaks with a German accent\\n- Uses German exclamations when excited (e.g., \\"Ausgezeichnet!\\", \\"Wunderbar!\\", \\"Mein Gott!\\")\\n- Analytical and precise in his word choice",
    "pre_hash": "",
    "note": "Adding specific speech pattern details"
}

❌ WRONG - Including header in original_text:
{
    "op_type": "replace_range",
    "original_text": "### Strengths\\n- [To be developed]",  // ❌ Don't include "### Strengths"!
    "text": "- Analytical mind\\n- Problem-solving abilities",
    "note": "This will DELETE the ### Strengths header!"
}

✅ CORRECT - Only include the content below the header:
{
    "op_type": "replace_range",
    "original_text": "- [To be developed]",  // ✅ Just the placeholder content
    "text": "- Analytical mind\\n- Problem-solving abilities",
    "pre_hash": "",
    "note": "Replacing placeholder with actual strengths"
}

---
OPERATION 3: delete_range
---
USE WHEN: Removing content (placeholder text, obsolete sections, etc.)

✅ CORRECT Example - Deleting placeholder text:
{
    "op_type": "delete_range",
    "start": 0,
    "end": 0,
    "original_text": "- [Connections to other characters to be established]",
    "text": "",
    "pre_hash": "",
    "note": "Removing placeholder relationship text"
}

✅ CORRECT Example - Removing outdated section:
{
    "op_type": "delete_range",
    "start": 0,
    "end": 0,
    "original_text": "- [Specific speech patterns and vocabulary to be developed]\\n- [Common phrases or expressions to be added]",
    "text": "",
    "pre_hash": "",
    "note": "Removing placeholder speech pattern notes"
}

---
=== CRITICAL RULES - MEMORIZE THESE! ===

1. **NEVER include headers in original_text for replace_range!**
   - Headers like "###" or "##" should NOT be in original_text
   - Only include the CONTENT below the header

2. **Use insert_after_heading for adding content below headers!**
   - Set anchor_text to the EXACT header line
   - This preserves the header structure

3. **Provide EXACT, VERBATIM text from the file!**
   - Copy text EXACTLY as it appears
   - Include punctuation, whitespace, line breaks
   - Minimum 10-20 words for reliable matching

4. **When in doubt, use insert_after_heading!**
   - Safer than replace_range for structured documents
   - Preserves existing structure

---
=== COMMON SCENARIOS ===

Scenario: User says "Add traits to the character"
✅ DO: Use insert_after_heading with anchor_text="### Traits"
❌ DON'T: Use replace_range with header included

Scenario: User says "Replace the placeholder text under Strengths"
✅ DO: Use replace_range with original_text="- [To be developed]"
❌ DON'T: Include "### Strengths" in original_text

Scenario: User says "Update the speech patterns"
✅ DO: Use replace_range targeting ONLY the placeholder lines
❌ DON'T: Include the "### Speech Patterns" header

Scenario: User says "Delete the placeholder relationships"
✅ DO: Use delete_range with original_text="- [Connections to be established]"
❌ DON'T: Delete the "## Relationships" header

---
=== REMEMBER ===

**A well-organized edit is like a well-organized cavalry charge - 
every operation knows its role and executes it perfectly!**

When you preserve headers and structure, you preserve the document's organization.
When you gobble up headers, you create chaos!

**Trust but verify** - always check your original_text doesn't include headers!
"""


def get_editor_operation_examples() -> str:
    """Return the comprehensive editor operation examples for agent prompts."""
    return EDITOR_OPERATION_EXAMPLES


def get_operation_type_guidance(request: str) -> str:
    """Provide specific guidance based on the request type.
    
    Args:
        request: The user's request or instruction
        
    Returns:
        Guidance string for the appropriate operation type
    """
    request_lower = request.lower()
    
    if any(word in request_lower for word in ["add", "insert", "create new", "add new"]):
        return """
        **GUIDANCE**: Use insert_after_heading operation
        - Set op_type: "insert_after_heading"
        - Set anchor_text to the EXACT header line (e.g., "### Traits", "## Background")
        - Set text to the new content to insert
        - This will preserve the header and add content below it
        """
    
    elif any(word in request_lower for word in ["replace", "update", "change", "revise"]):
        return """
        **GUIDANCE**: Use replace_range operation
        - Set op_type: "replace_range"
        - Set original_text to the EXACT content to replace (NOT including headers!)
        - Set text to the new replacement content
        - CRITICAL: Do NOT include header lines in original_text
        """
    
    elif any(word in request_lower for word in ["delete", "remove", "clear"]):
        return """
        **GUIDANCE**: Use delete_range operation
        - Set op_type: "delete_range"
        - Set original_text to the EXACT content to delete
        - Set text to "" (empty string)
        - This will remove the specified content while preserving structure
        """
    
    else:
        return """
        **GUIDANCE**: Determine appropriate operation type
        - insert_after_heading: For adding new content below headers
        - replace_range: For changing existing content (without touching headers)
        - delete_range: For removing content
        """


