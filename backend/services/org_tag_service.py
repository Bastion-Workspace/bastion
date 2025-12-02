"""
Org Tag Service - Add/Update Tags on Org Entries
Handles tagging org-mode headings with context-sensitive tag management
"""

import logging
import re
from typing import List, Optional, Set
from pathlib import Path

from config import settings
from models.org_tag_models import OrgTagRequest, OrgTagResponse

logger = logging.getLogger(__name__)


class OrgTagService:
    """
    Service for adding and managing tags on org-mode entries
    """
    
    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
    
    async def add_tags_to_entry(
        self,
        user_id: str,
        request: OrgTagRequest
    ) -> OrgTagResponse:
        """
        Add or update tags on an org-mode heading
        
        Args:
            user_id: User ID
            request: Tag request with file path, line number, and tags
            
        Returns:
            OrgTagResponse with success status and updated line
        """
        try:
            from services.database_manager.database_helpers import fetch_one
            
            # Get username for file path resolution
            row = await fetch_one("SELECT username FROM users WHERE user_id = $1", user_id)
            username = row['username'] if row else user_id
            
            # Resolve file path
            user_base_dir = self.upload_dir / "Users" / username
            file_path = user_base_dir / request.file_path
            
            if not file_path.exists():
                return OrgTagResponse(
                    success=False,
                    message=f"File not found: {request.file_path}"
                )
            
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Validate line number
            if request.line_number < 1 or request.line_number > len(lines):
                return OrgTagResponse(
                    success=False,
                    message=f"Invalid line number: {request.line_number} (file has {len(lines)} lines)"
                )
            
            # Get the line (convert to 0-indexed)
            line_idx = request.line_number - 1
            original_line = lines[line_idx].rstrip('\n')
            
            # Check if it's a heading
            if not re.match(r'^\*+\s+', original_line):
                return OrgTagResponse(
                    success=False,
                    message=f"Line {request.line_number} is not an org heading"
                )
            
            # Add/update tags
            updated_line, final_tags = self._add_tags_to_heading(
                original_line,
                request.tags,
                replace_existing=request.replace_existing
            )
            
            # Update the line
            lines[line_idx] = updated_line + '\n'
            
            # Write file back
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            logger.info(f"✅ TAG SUCCESS: Added tags {final_tags} to {request.file_path}:{request.line_number}")
            
            return OrgTagResponse(
                success=True,
                message=f"Tags added successfully",
                updated_line=updated_line,
                file_path=request.file_path,
                line_number=request.line_number,
                tags_applied=final_tags
            )
            
        except Exception as e:
            logger.error(f"❌ Tagging failed: {e}")
            return OrgTagResponse(
                success=False,
                message=f"Failed to add tags: {str(e)}"
            )
    
    def _add_tags_to_heading(
        self,
        heading_line: str,
        new_tags: List[str],
        replace_existing: bool = False
    ) -> tuple[str, List[str]]:
        """
        Add tags to a heading line, handling existing tags
        
        Args:
            heading_line: Original heading line
            new_tags: Tags to add
            replace_existing: If True, replace existing tags; if False, merge
            
        Returns:
            (updated_line, final_tags_list)
        """
        # Parse existing tags if any
        # Tags format: "* TODO Heading text                                :tag1:tag2:"
        tag_pattern = r'(\s+)(:[a-zA-Z0-9_@:]+:)\s*$'
        existing_match = re.search(tag_pattern, heading_line)
        
        if existing_match:
            # Extract existing tags
            existing_tags_str = existing_match.group(2)
            existing_tags = set(existing_tags_str.strip(':').split(':'))
            
            # Remove tags from line
            heading_without_tags = heading_line[:existing_match.start()]
        else:
            existing_tags = set()
            heading_without_tags = heading_line.rstrip()
        
        # Normalize new tags (ensure they start with : if needed)
        normalized_new_tags = []
        for tag in new_tags:
            tag = tag.strip()
            # Remove leading/trailing colons
            tag = tag.strip(':')
            if tag:
                normalized_new_tags.append(tag)
        
        # Determine final tags
        if replace_existing:
            final_tags = set(normalized_new_tags)
        else:
            # Merge existing and new tags
            final_tags = existing_tags | set(normalized_new_tags)
        
        # Build final tag string
        if final_tags:
            tags_str = ':' + ':'.join(sorted(final_tags)) + ':'
            
            # Calculate padding for right-alignment (standard column 77)
            # This matches the BeOrg style used in capture service
            padding_needed = max(1, 77 - len(heading_without_tags) - len(tags_str))
            updated_line = heading_without_tags + (' ' * padding_needed) + tags_str
        else:
            updated_line = heading_without_tags
        
        return updated_line, sorted(list(final_tags))


# Singleton instance
_org_tag_service: Optional[OrgTagService] = None


async def get_org_tag_service() -> OrgTagService:
    """Get or create the org tag service singleton"""
    global _org_tag_service
    
    if _org_tag_service is None:
        _org_tag_service = OrgTagService()
        logger.info("🚀 Org Tag Service initialized")
    
    return _org_tag_service











