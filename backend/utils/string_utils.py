"""
String utility functions
Extracted from main.py for better modularity
"""

import re


def strip_yaml_frontmatter(content: str) -> str:
    """
    Strip YAML frontmatter from markdown content
    
    Removes YAML frontmatter between --- markers and returns cleaned content.
    """
    # Pattern to match YAML frontmatter between --- markers
    frontmatter_pattern = r'^---\s*\n(.*?)\n---\s*\n'
    
    # Remove the frontmatter and return the rest
    cleaned_content = re.sub(frontmatter_pattern, '', content, flags=re.DOTALL)
    
    # Clean up any leading/trailing whitespace
    return cleaned_content.strip()

