"""
Content Formatting Utilities - Reusable content formatting for all agents

This module provides content-type aware JSON-to-markdown conversion that can be
used by any agent that performs file editing. It intelligently formats different
content types (code, components, schematics, calculations, diagrams, prose) appropriately.

Used by agents for:
- Converting JSON structures to markdown before saving to files
- Preserving code blocks with syntax highlighting
- Preserving diagrams (Mermaid, ASCII art, image references)
- Preserving regular prose text
- Formatting tabular data as markdown tables
- Converting component lists to readable markdown lists

Supports multiple content types in the same document:
- Code blocks (```language ... ```)
- Diagrams (```mermaid ... ```, ASCII art, image references)
- Regular markdown prose (headings, paragraphs, lists, etc.)
- Structured data (JSON arrays/objects that need conversion)
"""

import json
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def convert_json_to_markdown(content: str, content_type: Optional[str] = None) -> str:
    """
    Convert any JSON arrays/objects in content to markdown format.
    Content-type aware: preserves code blocks, diagrams, prose text, converts structured data.
    
    This function can be used by any agent that needs to format structured content
    before saving to markdown files. It intelligently handles different content types:
    
    - **Code**: Preserves code blocks with syntax highlighting (```language ... ```)
    - **Diagrams**: Preserves Mermaid diagrams (```mermaid ... ```), ASCII art, image references
    - **Prose Text**: Preserves regular markdown (headings, paragraphs, lists, links)
    - **Components**: Converts JSON arrays of objects → Markdown lists
    - **Schematic/Circuit Data**: Converts to markdown tables if tabular, otherwise lists
    - **Calculations**: Preserves formulas and mathematical notation
    - **General**: Converts JSON to appropriate markdown format
    
    The function is smart about mixed content - it only converts JSON structures while
    preserving all existing markdown formatting, code blocks, and diagrams.
    
    Args:
        content: Content that may contain JSON structures mixed with markdown
        content_type: Type of content ("code", "components", "calculations", "diagrams", "general", etc.)
    
    Returns:
        Content with JSON converted to markdown format, all other content preserved
    
    Example:
        ```python
        from orchestrator.utils.content_formatting_utils import convert_json_to_markdown
        
        # Convert component JSON to markdown list
        json_content = '[{"name": "Teensy 4.1", "type": "microcontroller"}]'
        markdown = convert_json_to_markdown(json_content, content_type="components")
        # Returns: "- **name**: Teensy 4.1\n- **type**: microcontroller"
        
        # Mixed content with code and JSON
        mixed = '''Here's some code:
        ```cpp
        void setup() { }
        ```
        
        And some components:
        [{"name": "Resistor", "value": "10k"}]
        '''
        result = convert_json_to_markdown(mixed, content_type="general")
        # Code block preserved, JSON converted to list
        ```
    """
    # For code content, preserve code blocks - don't convert JSON that's part of code
    if content_type == "code":
        # Only convert if it's clearly not code (e.g., component lists in code context)
        # For now, preserve code blocks as-is
        return content
    
    # Preserve all existing markdown structures (code blocks, diagrams, images, etc.)
    # We'll extract these, convert JSON, then restore them
    preserved_blocks = []
    block_counter = 0
    
    # Pattern to match code blocks (including mermaid diagrams)
    code_block_pattern = r'```(\w+)?\s*\n([\s\S]*?)```'
    
    def preserve_block(match):
        nonlocal block_counter
        language = match.group(1) or ""
        block_content = match.group(2)
        placeholder = f"__PRESERVED_BLOCK_{block_counter}__"
        preserved_blocks.append({
            "placeholder": placeholder,
            "language": language,
            "content": block_content,
            "full_match": match.group(0)
        })
        block_counter += 1
        return placeholder
    
    # Preserve all code blocks (including mermaid diagrams)
    content_with_placeholders = re.sub(code_block_pattern, preserve_block, content, flags=re.MULTILINE)
    
    # Preserve image references (markdown image syntax)
    image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    images = []
    image_counter = 0
    
    def preserve_image(match):
        nonlocal image_counter
        alt_text = match.group(1)
        url = match.group(2)
        placeholder = f"__PRESERVED_IMAGE_{image_counter}__"
        images.append({
            "placeholder": placeholder,
            "alt": alt_text,
            "url": url,
            "full_match": match.group(0)
        })
        image_counter += 1
        return placeholder
    
    content_with_placeholders = re.sub(image_pattern, preserve_image, content_with_placeholders)
    
    # Preserve existing markdown tables (don't convert JSON that's already a table)
    table_pattern = r'\|[^\n]+\|\n\|[-\s|:]+\|[\s\S]*?(?=\n\n|\n[^|]|$)'
    tables = []
    table_counter = 0
    
    def preserve_table(match):
        nonlocal table_counter
        table_content = match.group(0)
        placeholder = f"__PRESERVED_TABLE_{table_counter}__"
        tables.append({
            "placeholder": placeholder,
            "content": table_content
        })
        table_counter += 1
        return placeholder
    
    content_with_placeholders = re.sub(table_pattern, preserve_table, content_with_placeholders, flags=re.MULTILINE)
    
    def json_to_markdown(data, is_schematic: bool = False):
        """Convert JSON data structure to markdown format"""
        # For schematic/circuit data, prefer tables if it's a list of objects with consistent keys
        if is_schematic and isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            # Try to format as table if all objects have same keys
            keys = set(data[0].keys())
            if all(set(item.keys()) == keys for item in data if isinstance(item, dict)):
                # Format as markdown table
                header = "| " + " | ".join(keys) + " |"
                separator = "| " + " | ".join(["---"] * len(keys)) + " |"
                rows = []
                for item in data:
                    row = "| " + " | ".join(str(item.get(k, "")) for k in keys) + " |"
                    rows.append(row)
                return "\n".join([header, separator] + rows)
        
        # If it's a list of objects, convert to markdown list format
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict):
                # List of objects - format each as a section
                result = []
                for item in data:
                    if isinstance(item, dict):
                        item_lines = []
                        for key, value in item.items():
                            if isinstance(value, list):
                                if value and isinstance(value[0], dict):
                                    # Nested list of objects - format recursively
                                    nested = []
                                    for nested_item in value:
                                        nested_lines = []
                                        for nk, nv in nested_item.items():
                                            nested_lines.append(f"    - **{nk}**: {nv}")
                                        nested.append("\n".join(nested_lines))
                                    value_str = "\n".join(nested)
                                else:
                                    value_str = ", ".join(str(v) for v in value)
                                item_lines.append(f"  - **{key}**: {value_str}")
                            else:
                                item_lines.append(f"  - **{key}**: {value}")
                        result.append("\n".join(item_lines))
                return "\n\n".join(result)
            else:
                # Simple list
                return "\n".join(f"- {item}" for item in data)
        
        # If it's a single object, convert to markdown list
        elif isinstance(data, dict):
            result = []
            for key, value in data.items():
                if isinstance(value, list):
                    if value and isinstance(value[0], dict):
                        # Nested list of objects
                        nested = []
                        for nested_item in value:
                            nested_lines = []
                            for nk, nv in nested_item.items():
                                nested_lines.append(f"  - **{nk}**: {nv}")
                            nested.append("\n".join(nested_lines))
                        value_str = "\n\n".join(nested)
                    else:
                        value_str = ", ".join(str(v) for v in value)
                    result.append(f"- **{key}**: {value_str}")
                else:
                    result.append(f"- **{key}**: {value}")
            return "\n".join(result)
        
        return None
    
    # Detect if this might be schematic/circuit connection data
    is_schematic = content_type in ["general", None] and any(
        keyword in content.lower() for keyword in ["connection", "pin", "wire", "schematic", "circuit", "route"]
    )
    
    # Now work with content that has preserved blocks removed
    # Pattern to match ```json code blocks (only if not already preserved)
    json_block_pattern = r'```json\s*(\[[\s\S]*?\]|{[\s\S]*?})\s*```'
    
    def replace_json_block(match):
        json_str = match.group(1)
        if not json_str or not json_str.strip():
            return match.group(0)
        
        try:
            data = json.loads(json_str.strip())
            markdown = json_to_markdown(data, is_schematic=is_schematic)
            if markdown:
                return markdown
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
        
        return match.group(0)  # Return original if conversion fails
    
    # Replace JSON code blocks in content with placeholders
    content_with_placeholders = re.sub(json_block_pattern, replace_json_block, content_with_placeholders, flags=re.MULTILINE | re.DOTALL)
    
    # Also try to match standalone JSON arrays/objects that span multiple lines
    # Look for lines that start with [ or { and try to parse complete JSON
    # But skip if it's part of a preserved block placeholder
    lines = content_with_placeholders.split('\n')
    result_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Skip if this is a placeholder (already preserved)
        if any(placeholder in line for placeholder in ["__PRESERVED_BLOCK_", "__PRESERVED_IMAGE_", "__PRESERVED_TABLE_"]):
            result_lines.append(line)
            i += 1
            continue
        
        # Check if this line starts a JSON array or object
        if stripped.startswith('[') or stripped.startswith('{'):
            # Try to collect complete JSON structure
            json_lines = [line]
            j = i + 1
            bracket_count = stripped.count('[') - stripped.count(']') + stripped.count('{') - stripped.count('}')
            
            while j < len(lines) and bracket_count > 0:
                json_lines.append(lines[j])
                bracket_count += lines[j].count('[') - lines[j].count(']') + lines[j].count('{') - lines[j].count('}')
                j += 1
            
            if bracket_count == 0:
                # Complete JSON structure found - try to parse and convert
                json_str = '\n'.join(json_lines)
                try:
                    data = json.loads(json_str)
                    markdown = json_to_markdown(data, is_schematic=is_schematic)
                    if markdown:
                        result_lines.append(markdown)
                        i = j
                        continue
                except json.JSONDecodeError:
                    pass
        
        result_lines.append(line)
        i += 1
    
    # Restore preserved blocks
    result = '\n'.join(result_lines)
    
    # Restore code blocks (including mermaid diagrams)
    for block in preserved_blocks:
        language_tag = f"{block['language']}\n" if block['language'] else ""
        result = result.replace(
            block['placeholder'],
            f"```{language_tag}{block['content']}```"
        )
    
    # Restore images
    for img in images:
        result = result.replace(
            img['placeholder'],
            f"![{img['alt']}]({img['url']})"
        )
    
    # Restore tables
    for table in tables:
        result = result.replace(
            table['placeholder'],
            table['content']
        )
    
    return result

