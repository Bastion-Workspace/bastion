"""
Org-mode utility functions for parsing and processing org files

Pure functions that can be reused across multiple agents/subgraphs.
"""

import re
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def parse_org_structure(content: str) -> List[Dict[str, Any]]:
    """
    Parse full org file structure at heading level
    
    Returns list of all headings with hierarchy information:
    [{
        "heading": "text",
        "level": 1,
        "line_number": 5,
        "start_position": 100,
        "end_position": 500,
        "tags": ["project", "someday"],
        "todo_state": "TODO" or None,
        "parent_path": ["Parent", "Grandparent"],
        "subtree_content": "content under this heading"
    }, ...]
    """
    structure = []
    
    if not content:
        return structure
    
    lines = content.split('\n')
    heading_pattern = re.compile(r'^(\*+)\s+(.+?)(?:\s+(:[a-zA-Z0-9_@:+-]+:))?\s*$')
    todo_pattern = re.compile(r'^(TODO|NEXT|STARTED|WAITING|HOLD|DONE|CANCELED|CANCELLED)\s+', re.IGNORECASE)
    
    current_pos = 0
    heading_stack = []  # Track parent hierarchy
    
    for line_num, line in enumerate(lines, 1):
        line_start = current_pos
        line_end = current_pos + len(line)
        
        match = heading_pattern.match(line)
        if match:
            stars = match.group(1)
            level = len(stars)
            heading_text = match.group(2).strip()
            tags_str = match.group(3) if match.group(3) else None
            
            # Extract TODO state
            todo_state = None
            todo_match = todo_pattern.match(heading_text)
            if todo_match:
                todo_state = todo_match.group(1).upper()
                heading_text = todo_pattern.sub('', heading_text).strip()
            
            # Extract tags
            tags = []
            if tags_str:
                tags = [t.strip() for t in tags_str.strip(':').split(':') if t.strip()]
            
            # Update heading stack to current level (remove deeper levels)
            while len(heading_stack) >= level:
                heading_stack.pop()
            
            # Calculate subtree end position (next heading at same or higher level)
            subtree_end = len(content)
            for j in range(line_num, len(lines)):
                next_line = lines[j]
                next_match = heading_pattern.match(next_line)
                if next_match:
                    next_level = len(next_match.group(1))
                    if next_level <= level:
                        # Found heading at same or higher level - subtree ends here
                        subtree_end = current_pos + sum(len(lines[k]) + 1 for k in range(line_num - 1, j))
                        break
            
            # Get subtree content
            subtree_start = line_end + 1  # Start after heading line
            subtree_content = content[subtree_start:subtree_end].strip()
            
            # Build parent path
            parent_path = heading_stack.copy()
            
            heading_info = {
                "heading": heading_text,
                "level": level,
                "line_number": line_num,
                "start_position": line_start,
                "end_position": subtree_end,
                "tags": tags,
                "todo_state": todo_state,
                "parent_path": parent_path,
                "subtree_content": subtree_content[:1000]  # Limit subtree content preview
            }
            
            structure.append(heading_info)
            
            # Add to stack
            heading_stack.append(heading_text)
        
        current_pos = line_end + 1  # +1 for newline
    
    return structure


def extract_tags_from_content(content: str, cursor_offset: int = -1) -> List[Dict[str, Any]]:
    """
    Extract tags from Org Mode headings in content
    
    Returns list of dicts with heading info and tags:
    [{"heading": "text", "level": 1, "tags": ["project", "someday"], "line": 5}, ...]
    """
    tags_info = []
    
    if not content:
        return tags_info
    
    lines = content.split('\n')
    
    # Pattern to match Org headings with tags: * Heading text :tag1:tag2:
    heading_pattern = re.compile(r'^(\*+)\s+(.+?)(?:\s+(:[a-zA-Z0-9_@:+-]+:))?\s*$')
    
    for line_num, line in enumerate(lines, 1):
        match = heading_pattern.match(line)
        if match:
            stars = match.group(1)
            heading_text = match.group(2).strip()
            tags_str = match.group(3) if match.group(3) else None
            
            # Remove TODO keywords from heading text
            heading_text = re.sub(r'^(TODO|NEXT|STARTED|WAITING|HOLD|DONE|CANCELED|CANCELLED)\s+', '', heading_text).strip()
            
            # Extract tags
            tags = []
            if tags_str:
                # Remove colons and split
                tags = [t.strip() for t in tags_str.strip(':').split(':') if t.strip()]
            
            # Only include headings with tags
            if tags:
                # Check if cursor is in this heading's range (if cursor_offset provided)
                is_cursor_heading = False
                if cursor_offset >= 0:
                    line_start = sum(len(lines[i]) + 1 for i in range(line_num - 1))
                    line_end = line_start + len(line)
                    is_cursor_heading = line_start <= cursor_offset <= line_end
                
                tags_info.append({
                    "heading": heading_text,
                    "level": len(stars),
                    "tags": tags,
                    "line": line_num,
                    "is_cursor_heading": is_cursor_heading
                })
    
    return tags_info


def detect_org_links(content: str) -> List[Dict[str, Any]]:
    """
    Detect Org-mode file links in content
    
    Supports:
    - [[file:path/to/file.md][Description]]
    - [[file:other.org]]
    - [[file:path/to/file.org::*Heading]]
    
    Returns links with position information for context filtering
    """
    links = []
    
    if not content:
        return links
    
    # Pattern for Org-mode file links: [[file:path][description]] or [[file:path]]
    # Also supports heading links: [[file:path::*Heading]]
    pattern = r'\[\[file:([^\]]+?)(?:\]\[([^\]]+?)\])?\]\]'
    
    for match in re.finditer(pattern, content):
        full_match = match.group(0)
        file_path = match.group(1)
        description = match.group(2) if match.group(2) else file_path
        match_start = match.start()
        match_end = match.end()
        
        # Check if it's a heading link (contains ::)
        heading = None
        if "::" in file_path:
            parts = file_path.split("::", 1)
            file_path = parts[0]
            heading = parts[1] if len(parts) > 1 else None
        
        links.append({
            "full_match": full_match,
            "file_path": file_path.strip(),
            "description": description.strip(),
            "heading": heading.strip() if heading else None,
            "position": match_start,  # Position in content for context filtering
            "end_position": match_end
        })
    
    return links


def find_heading_at_cursor(content: str, cursor_offset: int) -> Optional[Dict[str, Any]]:
    """
    Find the Org-mode heading that contains the cursor position
    
    Returns heading info with level, text, and position boundaries
    """
    if cursor_offset < 0 or cursor_offset >= len(content):
        return None
    
    # Find the heading that contains the cursor
    # Org headings start with * (one or more) followed by space
    heading_pattern = r'^(\*+)\s+(.+)$'
    
    lines = content.split('\n')
    current_pos = 0
    current_heading = None
    current_heading_start = 0
    current_heading_level = 0
    
    for i, line in enumerate(lines):
        line_start = current_pos
        line_end = current_pos + len(line)
        
        # Check if this line is a heading
        match = re.match(heading_pattern, line)
        if match:
            level = len(match.group(1))
            heading_text = match.group(2).strip()
            
            # This is a heading - update current heading
            current_heading = heading_text
            current_heading_start = line_start
            current_heading_level = level
        
        # Check if cursor is in this line
        if line_start <= cursor_offset <= line_end:
            # Cursor is in this line - return the current heading
            if current_heading:
                # Find where this heading's subtree ends (next heading of same or higher level)
                heading_end = len(content)
                for j in range(i + 1, len(lines)):
                    next_line = lines[j]
                    next_match = re.match(heading_pattern, next_line)
                    if next_match:
                        next_level = len(next_match.group(1))
                        if next_level <= current_heading_level:
                            # Found a heading at same or higher level - this subtree ends here
                            heading_end = current_pos + sum(len(lines[k]) + 1 for k in range(i + 1, j))
                            break
                
                return {
                    "heading": current_heading,
                    "level": current_heading_level,
                    "start_position": current_heading_start,
                    "end_position": heading_end,
                    "line_number": i
                }
            break
        
        current_pos = line_end + 1  # +1 for newline
    
    return current_heading if current_heading else None
