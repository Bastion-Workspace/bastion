"""
Frontmatter Utilities - Reusable YAML frontmatter parsing and updating for all agents

This module provides robust frontmatter manipulation that preserves all existing
fields while allowing targeted updates. Use this for any agent that needs to
update document frontmatter without losing data.
"""

import re
import logging
from typing import Dict, Any, Optional, List, Union, Tuple

logger = logging.getLogger(__name__)


def strip_frontmatter_block(text: str) -> str:
    """Strip only the leading YAML frontmatter block from text. Uses \\A so scene breaks (---) in the body are never stripped."""
    try:
        return re.sub(r'\A---\s*\n[\s\S]*?\n---\s*\n', '', text)
    except Exception:
        return text


def frontmatter_end_index(text: str) -> int:
    """Return the end index of a leading YAML frontmatter block if present, else 0."""
    try:
        m = re.match(r'^(---\s*\n[\s\S]*?\n---\s*\n)', text, flags=re.MULTILINE)
        if m:
            return m.end()
        return 0
    except Exception:
        return 0


async def update_frontmatter_field(
    content: str,
    field_updates: Dict[str, Any],
    list_updates: Optional[Dict[str, List[str]]] = None,
    preserve_order: bool = True
) -> Tuple[str, bool]:
    """
    Update frontmatter fields while preserving all existing fields.
    
    This is the primary function agents should use to update frontmatter.
    It parses existing frontmatter as YAML, applies updates, and preserves
    all other fields.
    
    Args:
        content: Full document content with frontmatter
        field_updates: Dict of scalar field updates (e.g., {"title": "New Title"})
        list_updates: Dict of list field updates (e.g., {"files": ["./file1.md", "./file2.md"]})
        preserve_order: If True, preserve original field order when possible
    
    Returns:
        Tuple of (updated_content, success)
        - updated_content: Content with updated frontmatter
        - success: True if update succeeded, False otherwise
    
    Example:
        ```python
        # Update title and add a file to the files list
        new_content, success = await update_frontmatter_field(
            content=document_content,
            field_updates={"title": "Updated Title"},
            list_updates={"files": ["./new_file.md"]}
        )
        ```
    """
    try:
        import yaml
        
        # **CRITICAL**: First, remove ALL duplicate frontmatter blocks from content
        # This handles cases where previous updates created duplicates
        # We want to keep only the FIRST frontmatter block and the body content
        
        # Find the FIRST complete frontmatter block (from first --- to second ---)
        first_frontmatter_match = re.match(r'^---\s*\r?\n([\s\S]*?)\r?\n---\s*\r?\n', content)

        if first_frontmatter_match:
            # We found the first frontmatter block
            first_block_end = first_frontmatter_match.end()

            # Check if there are more frontmatter blocks after the first one
            remaining_content = content[first_block_end:]

            # Count frontmatter blocks before cleaning
            original_frontmatter_count = len(re.findall(r'^---\s*\r?\n', content, re.MULTILINE))

            # Remove ALL subsequent frontmatter blocks from the body
            # This regex matches complete frontmatter blocks (--- ... ---)
            cleaned_body = re.sub(r'^---\s*\r?\n[\s\S]*?\r?\n---\s*\r?\n', '', remaining_content, flags=re.MULTILINE)

            # Rebuild clean content with only the first frontmatter block
            first_frontmatter_block = content[:first_block_end]
            content = f"{first_frontmatter_block}{cleaned_body}"

            final_frontmatter_count = len(re.findall(r'^---\s*\r?\n', content, re.MULTILINE))
            if original_frontmatter_count > 2:
                removed_blocks = original_frontmatter_count - final_frontmatter_count
                logger.warning(f"🧹 Cleaned {removed_blocks} duplicate frontmatter blocks (was {original_frontmatter_count}, now {final_frontmatter_count})")
        
        # Extract frontmatter block and body (now content is clean)
        frontmatter_match = re.match(r"^---\s*\r?\n([\s\S]*?)\r?\n---\s*\r?\n", content)

        if not frontmatter_match:
            logger.warning("No frontmatter found in content after duplicate cleaning")
            # If no frontmatter, add basic frontmatter with the updates
            frontmatter_data = {}
            # Apply updates to empty frontmatter
            for key, value in (field_updates or {}).items():
                if value is not None:
                    frontmatter_data[key] = value
            for key, new_items in (list_updates or {}).items():
                if new_items:
                    frontmatter_data[key] = list(new_items)

            frontmatter_yaml = yaml.dump(frontmatter_data, default_flow_style=False, allow_unicode=True, sort_keys=not preserve_order).strip()
            updated_content = f"---\n{frontmatter_yaml}\n---\n{content}"
            return updated_content, True
        
        frontmatter_text = frontmatter_match.group(1)
        body = content[frontmatter_match.end():]
        
        # Final safety check: remove any remaining duplicate frontmatter blocks from body
        # This handles edge cases where duplicates might still exist
        original_body_len = len(body)
        # Remove ALL frontmatter blocks from body (they should not be there!)
        body = re.sub(r'^---\s*\r?\n[\s\S]*?\r?\n---\s*\r?\n', '', body, flags=re.MULTILINE)
        if len(body) < original_body_len:
            removed = original_body_len - len(body)
            logger.warning(f"⚠️ Removed {removed} chars of duplicate frontmatter from body (frontmatter should only be at the start!)")
        
        # Parse existing frontmatter as YAML
        try:
            frontmatter_data = yaml.safe_load(frontmatter_text) or {}
            if not isinstance(frontmatter_data, dict):
                logger.warning(f"YAML parsed to non-dict type: {type(frontmatter_data)}, using fallback")
                frontmatter_data = _parse_frontmatter_fallback(frontmatter_text)
            else:
                # Log field count for debugging
                field_count = len(frontmatter_data)
                logger.debug(f"Parsed frontmatter with {field_count} fields: {list(frontmatter_data.keys())}")
        except yaml.YAMLError as e:
            logger.warning(f"Failed to parse frontmatter as YAML: {e}, attempting fallback parsing")
            frontmatter_data = _parse_frontmatter_fallback(frontmatter_text)
        
        # Ensure frontmatter_data is a dict
        if not isinstance(frontmatter_data, dict):
            logger.error(f"Frontmatter parsing failed - got {type(frontmatter_data)}, using empty dict")
            frontmatter_data = {}
        
        # Apply scalar field updates
        for key, value in (field_updates or {}).items():
            if value is not None:
                frontmatter_data[key] = value
        
        # Apply list field updates - PRESERVE existing references!
        for key, new_items in (list_updates or {}).items():
            if not new_items:
                continue

            # Get existing list or create new one
            if key in frontmatter_data:
                existing_list = frontmatter_data[key]
                if isinstance(existing_list, list):
                    # Merge lists, avoiding duplicates - PRESERVE existing items
                    merged_list = existing_list.copy()  # Start with existing items
                    for item in new_items:
                        if item not in merged_list:
                            merged_list.append(item)
                    frontmatter_data[key] = merged_list
                elif isinstance(existing_list, str):
                    # Convert string to list and merge
                    frontmatter_data[key] = [existing_list] + [item for item in new_items if item != existing_list]
                else:
                    # Replace with list (fallback)
                    frontmatter_data[key] = list(new_items)
            else:
                # Create new list
                frontmatter_data[key] = list(new_items)
        
        # Validate that we still have all original fields (safety check)
        original_field_count = len(frontmatter_data)
        if original_field_count == 0 and frontmatter_text.strip():
            logger.warning(f"⚠️ Frontmatter data is empty after parsing - original had content. This may indicate a parsing issue.")
        
        # Serialize frontmatter back to YAML
        try:
            frontmatter_yaml = yaml.dump(
                frontmatter_data,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=not preserve_order  # Preserve order if requested
            ).strip()
            
            # Log final field count for debugging
            final_field_count = len(frontmatter_data)
            logger.debug(f"Serializing frontmatter with {final_field_count} fields: {list(frontmatter_data.keys())}")
        except Exception as dump_error:
            logger.error(f"Failed to serialize frontmatter to YAML: {dump_error}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return content, False
        
        # Rebuild content
        updated_content = f"---\n{frontmatter_yaml}\n---\n{body}"
        
        return updated_content, True
        
    except Exception as e:
        logger.error(f"Failed to update frontmatter: {e}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return content, False


def _parse_frontmatter_fallback(frontmatter_text: str) -> Dict[str, Any]:
    """
    Fallback parser for frontmatter that can't be parsed as YAML.
    
    Handles simple key-value pairs and basic lists.
    """
    data = {}
    lines = frontmatter_text.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue
        
        # Check for key-value pair
        if ':' in line and not line.strip().startswith('-'):
            parts = line.split(':', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                
                # Check if next lines are a list
                if not value and i + 1 < len(lines):
                    # Look ahead for list items
                    list_items = []
                    j = i + 1
                    while j < len(lines):
                        list_line = lines[j].strip()
                        if list_line.startswith('-'):
                            list_items.append(list_line[1:].strip())
                            j += 1
                        elif not list_line:
                            j += 1
                        else:
                            break
                    
                    if list_items:
                        data[key] = list_items
                        i = j
                        continue
                
                # Simple scalar value
                data[key] = value
        
        i += 1
    
    return data


async def add_to_frontmatter_list(
    content: str,
    list_key: str,
    new_items: List[str],
    also_update_files: bool = True
) -> Tuple[str, bool]:
    """
    Convenience function to add items to a frontmatter list field.
    
    This is a simplified wrapper around update_frontmatter_field for the
    common case of adding items to a list.
    
    Args:
        content: Full document content with frontmatter
        list_key: The list field to update (e.g., "components", "files")
        new_items: List of items to add (duplicates are avoided)
        also_update_files: If True, also add items to the "files" list
    
    Returns:
        Tuple of (updated_content, success)
    
    Example:
        ```python
        # Add a new component file reference
        new_content, success = await add_to_frontmatter_list(
            content=project_plan_content,
            list_key="components",
            new_items=["./new_component.md"]
        )
        ```
    """
    list_updates = {list_key: new_items}
    
    if also_update_files and list_key != "files":
        list_updates["files"] = new_items
    
    return await update_frontmatter_field(
        content=content,
        field_updates={},
        list_updates=list_updates
    )


async def parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """
    Parse frontmatter from document content.
    
    Returns the frontmatter as a dict and the body content.
    
    Args:
        content: Full document content with frontmatter
    
    Returns:
        Tuple of (frontmatter_dict, body_content)
        If no frontmatter found, returns ({}, content)
    
    Example:
        ```python
        frontmatter, body = await parse_frontmatter(document_content)
        title = frontmatter.get("title", "Untitled")
        files = frontmatter.get("files", [])
        ```
    """
    try:
        import yaml
        
        frontmatter_match = re.match(r"^---\s*\r?\n([\s\S]*?)\r?\n---\s*\r?\n", content)
        
        if not frontmatter_match:
            return {}, content
        
        frontmatter_text = frontmatter_match.group(1)
        body = content[frontmatter_match.end():]
        
        # Remove any duplicate frontmatter blocks from body
        body = re.sub(r'^---\s*\r?\n[\s\S]*?\r?\n---\s*\r?\n', '', body, flags=re.MULTILINE)
        
        try:
            frontmatter_data = yaml.safe_load(frontmatter_text) or {}
        except yaml.YAMLError:
            frontmatter_data = _parse_frontmatter_fallback(frontmatter_text)
        
        return frontmatter_data, body
        
    except Exception as e:
        logger.warning(f"Failed to parse frontmatter: {e}")
        return {}, content

