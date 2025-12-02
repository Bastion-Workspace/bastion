"""
Project Utilities

Pure utility functions for project-related content generation and filename handling.
These are stateless functions that don't require backend services.

Used by agents for:
- Generating filenames from content
- Generating titles from content
- Extracting descriptions from content
- Sanitizing filenames (preserving spaces, removing problematic characters)
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by preserving spaces and only removing problematic filesystem characters.
    
    Preserves spaces in filenames (filesystem supports spaces).
    Only removes truly problematic characters: / \ : * ? " < > |
    Trims leading/trailing spaces and dots (Windows restriction).
    
    Args:
        filename: Original filename (may contain spaces, special chars)
        
    Returns:
        Sanitized filename safe for filesystem use
        
    Example:
        >>> sanitize_filename("My Project - Component Spec.md")
        "My Project - Component Spec.md"
        >>> sanitize_filename("file:name*with?bad<chars>.md")
        "file_name_with_bad_chars_.md"
    """
    if not filename:
        return ""
    
    # Only remove problematic filesystem characters
    sanitized = "".join(
        c if c.isprintable() and c not in ['/', '\\', ':', '*', '?', '"', '<', '>', '|'] 
        else "_" 
        for c in filename
    )
    
    # Trim leading/trailing spaces and dots (Windows restriction)
    sanitized = sanitized.strip(' .')
    
    return sanitized


def generate_filename_from_content(content: str, content_type: str) -> str:
    """
    Generate a filename suggestion from content by extracting component/system names.
    
    Looks for capitalized words (likely proper nouns/component names) in content
    and uses them to generate a descriptive filename. Preserves spaces in filenames.
    
    Args:
        content: Content text to analyze
        content_type: Type of content (e.g., "component", "protocol", "schematic")
        
    Returns:
        Suggested filename with .md extension
        
    Example:
        >>> generate_filename_from_content("The Keyboard Matrix circuit uses...", "component")
        "Keyboard Matrix component.md"
        >>> generate_filename_from_content("General information about...", "specification")
        "specification details.md"
    """
    # Look for specific component/system names (capitalized words)
    capitalized_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
    
    if capitalized_words:
        # Use first significant capitalized phrase - preserve spaces in filename
        name = capitalized_words[0]
        # Sanitize but preserve spaces
        name = sanitize_filename(name)
        # Limit length
        name = name[:40]
        # Add content type
        return f"{name} {content_type}.md"
    
    # Fallback: use content type with descriptive name
    return f"{content_type} details.md"


def generate_title_from_content(content: str, content_type: str) -> str:
    """
    Generate a title suggestion from content by extracting component/system names.
    
    Args:
        content: Content text to analyze
        content_type: Type of content (e.g., "component", "protocol", "schematic")
        
    Returns:
        Suggested title
        
    Example:
        >>> generate_title_from_content("The Keyboard Matrix circuit uses...", "component")
        "Keyboard Matrix - Component"
        >>> generate_title_from_content("General information about...", "specification")
        "Specification Details"
    """
    # Look for specific component/system names
    capitalized_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
    
    if capitalized_words:
        return f"{capitalized_words[0]} - {content_type.title()}"
    
    # Fallback
    return f"{content_type.title()} Details"


def extract_description_from_content(content: str) -> str:
    """
    Extract a description from content (first sentence or summary).
    
    Attempts to extract the first sentence if it's a reasonable length (20-200 chars).
    Falls back to first 100 characters if no suitable sentence found.
    
    Args:
        content: Content text to extract description from
        
    Returns:
        Description string (first sentence or truncated content)
        
    Example:
        >>> extract_description_from_content("The keyboard scanning matrix uses a 8x8 grid. It connects...")
        "The keyboard scanning matrix uses a 8x8 grid."
        >>> extract_description_from_content("Short.")
        "Short...."
    """
    if not content:
        return ""
    
    # Get first sentence
    sentences = re.split(r'[.!?]\s+', content)
    if sentences:
        first_sentence = sentences[0].strip()
        # Use first sentence if it's a reasonable length
        if 20 <= len(first_sentence) <= 200:
            return first_sentence
    
    # Fallback: first 100 chars
    return content[:100].strip() + "..."


