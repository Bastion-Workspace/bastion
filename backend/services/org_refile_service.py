"""
Org Refile Service - Roosevelt's Moving Day!
Handles moving/refiling org-mode entries between files and headings
"""

import logging
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from config import settings

logger = logging.getLogger(__name__)


class OrgRefileService:
    """
    Service for refiling (moving) org-mode entries
    
    **BULLY!** Move those TODOs like a well-organized cavalry!
    """
    
    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
    
    async def discover_refile_targets(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Discover all potential refile targets for a user
        
        **By George!** Find all the places we can refile to!
        Respects user's refile_max_level setting
        
        Returns:
            List of targets with file, heading path, and display name
        """
        try:
            from services.database_manager.database_helpers import fetch_one
            from services.org_settings_service import get_org_settings_service
            
            # Get user's refile max level setting
            settings_service = await get_org_settings_service()
            settings = await settings_service.get_settings(user_id)
            max_level = settings.refile_max_level if settings else 2
            
            logger.info(f"🎯 Using refile_max_level: {max_level} (only showing headings up to {'*' * max_level})")
            
            # Get username
            row = await fetch_one("SELECT username FROM users WHERE user_id = $1", user_id)
            username = row['username'] if row else user_id
            
            # Find user's org directory
            user_base_dir = self.upload_dir / "Users" / username
            if not user_base_dir.exists():
                return []
            
            # Find all org files
            org_files = list(user_base_dir.rglob("*.org"))
            
            targets = []
            
            for org_file in org_files:
                relative_path = org_file.relative_to(user_base_dir)
                filename = org_file.name
                
                # Read file and parse headings
                try:
                    with open(org_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    headings = self._parse_headings_for_targets(content, max_level)
                    
                    # Add file root as a target
                    targets.append({
                        'file': str(relative_path),
                        'filename': filename,
                        'heading_path': [],
                        'heading_line': 0,
                        'display_name': f"📄 {filename}",
                        'level': 0
                    })
                    
                    # Add each heading as a target (filtered by max_level in _parse_headings_for_targets)
                    for heading in headings:
                        display_name = f"{'  ' * heading['level']}{heading['heading']}"
                        targets.append({
                            'file': str(relative_path),
                            'filename': filename,
                            'heading_path': heading['path'],
                            'heading_line': heading['line_number'],
                            'display_name': display_name,
                            'level': heading['level']
                        })
                
                except Exception as e:
                    logger.error(f"❌ Failed to parse {org_file}: {e}")
                    continue
            
            logger.info(f"✅ Discovered {len(targets)} refile targets for user {username}")
            return targets
        
        except Exception as e:
            logger.error(f"❌ Failed to discover refile targets: {e}")
            return []
    
    def _parse_headings_for_targets(self, content: str, max_level: int = 2) -> List[Dict[str, Any]]:
        """
        Parse org content to extract heading hierarchy
        
        Args:
            content: Org file content
            max_level: Maximum heading level to include (1 = only *, 2 = * and **, etc.)
        """
        headings = []
        lines = content.split('\n')
        
        heading_stack = []  # Track hierarchy
        
        for i, line in enumerate(lines, 1):
            # Match org headings
            match = re.match(r'^(\*+)\s+(.+)$', line)
            if match:
                level = len(match.group(1))
                
                # Skip headings beyond max_level
                if level > max_level:
                    continue
                
                heading_text = match.group(2).strip()
                
                # Remove TODO keywords from heading text for cleaner display
                heading_text = re.sub(r'^(TODO|NEXT|STARTED|WAITING|HOLD|DONE|CANCELED|CANCELLED)\s+', '', heading_text)
                # Remove tags at end
                heading_text = re.sub(r'\s+:[a-zA-Z0-9_@:]+:\s*$', '', heading_text)
                
                # Update stack to current level
                heading_stack = heading_stack[:level-1]
                heading_stack.append(heading_text)
                
                headings.append({
                    'level': level,
                    'heading': heading_text,
                    'line_number': i,
                    'path': list(heading_stack)
                })
        
        return headings
    
    async def refile_entry(
        self,
        user_id: str,
        source_file: str,
        source_line: int,
        target_file: str,
        target_heading_line: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Refile an org entry from one location to another
        
        **ROOSEVELT REFILE OPERATION!**
        
        Args:
            user_id: User ID
            source_file: Source file relative path
            source_line: Line number of entry to move
            target_file: Target file relative path
            target_heading_line: Line number of target heading (None = file root)
        
        Returns:
            Dict with success status and details
        """
        try:
            from services.database_manager.database_helpers import fetch_one
            
            # Get username
            row = await fetch_one("SELECT username FROM users WHERE user_id = $1", user_id)
            username = row['username'] if row else user_id
            
            user_base_dir = self.upload_dir / "Users" / username
            
            # Resolve file paths
            source_path = user_base_dir / source_file
            target_path = user_base_dir / target_file
            
            if not source_path.exists():
                raise ValueError(f"Source file not found: {source_file}")
            if not target_path.exists():
                raise ValueError(f"Target file not found: {target_file}")
            
            # Read both files
            with open(source_path, 'r', encoding='utf-8') as f:
                source_content = f.read()
            
            with open(target_path, 'r', encoding='utf-8') as f:
                target_content = f.read()
            
            # Extract the entry from source
            entry_lines, new_source_content = self._extract_entry(source_content, source_line)
            
            if not entry_lines:
                raise ValueError(f"No entry found at line {source_line}")
            
            # Insert entry into target
            new_target_content = self._insert_entry(target_content, entry_lines, target_heading_line)
            
            # Write both files
            with open(source_path, 'w', encoding='utf-8') as f:
                f.write(new_source_content)
            
            with open(target_path, 'w', encoding='utf-8') as f:
                f.write(new_target_content)
            
            logger.info(f"✅ REFILE: Moved entry from {source_file}:{source_line} to {target_file}")
            
            return {
                'success': True,
                'source_file': source_file,
                'target_file': target_file,
                'lines_moved': len(entry_lines)
            }
        
        except Exception as e:
            logger.error(f"❌ Refile failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _extract_entry(self, content: str, line_number: int) -> tuple:
        """
        Extract an org entry and return it along with content without the entry
        
        Returns:
            (entry_lines, remaining_content)
        """
        lines = content.split('\n')
        
        if line_number < 1 or line_number > len(lines):
            return [], content
        
        # Find the entry (line_number is 1-indexed)
        idx = line_number - 1
        start_line = lines[idx]
        
        # Check if this is a heading
        match = re.match(r'^(\*+)\s+', start_line)
        if not match:
            # Not a heading - just move this line
            entry_lines = [lines[idx]]
            new_lines = lines[:idx] + lines[idx+1:]
            return entry_lines, '\n'.join(new_lines)
        
        # It's a heading - find the entire subtree
        level = len(match.group(1))
        entry_lines = [start_line]
        
        # Collect all lines until we hit a same-or-higher level heading
        i = idx + 1
        while i < len(lines):
            line = lines[i]
            next_match = re.match(r'^(\*+)\s+', line)
            
            if next_match:
                next_level = len(next_match.group(1))
                if next_level <= level:
                    # Found a same or higher level heading - stop
                    break
            
            entry_lines.append(line)
            i += 1
        
        # Remove the entry from source
        new_lines = lines[:idx] + lines[i:]
        
        return entry_lines, '\n'.join(new_lines)
    
    def _adjust_heading_level(self, entry_lines: List[str], level_adjustment: int) -> List[str]:
        """
        Adjust the level of all headings in entry_lines
        
        Args:
            entry_lines: Lines to adjust
            level_adjustment: Number of stars to add (positive) or remove (negative)
        
        Returns:
            Adjusted lines
        """
        if level_adjustment == 0:
            return entry_lines
        
        adjusted_lines = []
        for line in entry_lines:
            match = re.match(r'^(\*+)(\s+.*)$', line)
            if match:
                stars = match.group(1)
                rest = match.group(2)
                new_level = len(stars) + level_adjustment
                if new_level > 0:
                    adjusted_lines.append('*' * new_level + rest)
                else:
                    # If adjustment would make it 0 or negative, keep at 1 star
                    adjusted_lines.append('*' + rest)
            else:
                adjusted_lines.append(line)
        
        return adjusted_lines
    
    def _insert_entry(
        self,
        content: str,
        entry_lines: List[str],
        target_heading_line: Optional[int] = None
    ) -> str:
        """
        Insert entry into target location
        
        If target_heading_line is None, append to end of file
        If target_heading_line is provided, insert as child of that heading
        """
        lines = content.split('\n')
        
        if target_heading_line is None:
            # Append to end of file - no level adjustment needed
            lines.extend([''] + entry_lines)
            return '\n'.join(lines)
        
        # Insert under specific heading as a child
        idx = target_heading_line - 1
        
        if idx < 0 or idx >= len(lines):
            # Invalid line - append to end
            lines.extend([''] + entry_lines)
            return '\n'.join(lines)
        
        # Find where to insert (after the target heading and its direct content)
        target_line = lines[idx]
        target_match = re.match(r'^(\*+)\s+', target_line)
        
        if not target_match:
            # Not a heading - insert after this line, no adjustment
            lines = lines[:idx+1] + [''] + entry_lines + lines[idx+1:]
            return '\n'.join(lines)
        
        target_level = len(target_match.group(1))
        
        # Adjust entry heading levels to be children of target
        # Get the level of the first heading in entry_lines
        entry_first_line = entry_lines[0] if entry_lines else ''
        entry_match = re.match(r'^(\*+)\s+', entry_first_line)
        
        if entry_match:
            entry_level = len(entry_match.group(1))
            # Make it one level deeper than target
            level_adjustment = (target_level + 1) - entry_level
            adjusted_entry_lines = self._adjust_heading_level(entry_lines, level_adjustment)
            
            logger.info(f"🔧 Adjusted heading level: {entry_level} stars → {entry_level + level_adjustment} stars (target level: {target_level})")
        else:
            # Not a heading, no adjustment
            adjusted_entry_lines = entry_lines
        
        # Find the end of this heading's content (before next same-level heading)
        insert_idx = idx + 1
        while insert_idx < len(lines):
            line = lines[insert_idx]
            next_match = re.match(r'^(\*+)\s+', line)
            
            if next_match:
                next_level = len(next_match.group(1))
                if next_level <= target_level:
                    # Found same or higher level - insert before this
                    break
            
            insert_idx += 1
        
        # Insert the adjusted entry
        lines = lines[:insert_idx] + [''] + adjusted_entry_lines + lines[insert_idx:]
        
        return '\n'.join(lines)


# Singleton instance
_org_refile_service = None


async def get_org_refile_service():
    """Get or create the org refile service singleton"""
    global _org_refile_service
    
    if _org_refile_service is None:
        _org_refile_service = OrgRefileService()
    
    return _org_refile_service

