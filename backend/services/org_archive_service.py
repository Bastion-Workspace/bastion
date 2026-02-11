"""
Org-Mode Archive Service - Roosevelt's "Clean Desk Policy"

**BULLY!** Archive completed tasks to keep your active files lean and mean!

This service handles:
- Archiving individual entries to {filename}_archive.org
- Bulk archiving all DONE items
- Maintaining file structure in archives
- Configurable archive locations
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class OrgArchiveService:
    """Service for archiving org-mode entries"""
    
    def __init__(self):
        self.archive_suffix = "_archive.org"
    
    async def archive_entry(
        self,
        user_id: str,
        source_file: str,
        line_number: int,
        archive_location: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Archive a single org entry to the archive file
        
        **ROOSEVELT DOCTRINE:** Move DONE tasks to the archive, keep active files clean!
        
        Args:
            user_id: User ID for file access
            source_file: Relative path to source file (e.g., "OrgMode/tasks.org")
            line_number: Line number of entry to archive (1-based)
            archive_location: Optional custom archive location (defaults to user's settings or {source}_archive.org)
        
        Returns:
            Dict with success status and details
        """
        try:
            logger.info(f"📦 ARCHIVE: User {user_id} archiving entry from {source_file} line {line_number}")
            
            # Get user's org directory
            from services.folder_service import FolderService
            folder_service = FolderService()
            
            # Resolve source file path
            source_path = await self._resolve_file_path(user_id, source_file, folder_service)
            if not source_path.exists():
                raise FileNotFoundError(f"Source file not found: {source_file}")
            
            # Determine archive file path
            if archive_location:
                # Resolve relative to source file's directory if it's a relative path
                archive_path = await self._resolve_file_path(user_id, archive_location, folder_service, base_path=source_path)
            else:
                # Check user's settings for default archive location
                archive_path = await self._get_archive_path(user_id, source_file, source_path)
            
            logger.info(f"📦 Archive source: {source_path}")
            logger.info(f"📦 Archive target: {archive_path}")
            
            # Read source file
            with open(source_path, 'r', encoding='utf-8') as f:
                source_lines = f.readlines()
            
            # Extract the entry
            entry_info = self._extract_entry(source_lines, line_number)
            if not entry_info:
                raise ValueError(f"No entry found at line {line_number}")
            
            entry_lines = entry_info['lines']
            entry_heading = entry_info['heading']
            
            logger.info(f"📦 Extracted entry: {entry_heading} ({len(entry_lines)} lines)")
            
            # Add archive metadata
            archived_entry = self._add_archive_metadata(entry_lines, source_file)
            
            # Ensure archive directory exists
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Read or create archive file
            if archive_path.exists():
                with open(archive_path, 'r', encoding='utf-8') as f:
                    archive_content = f.read()
            else:
                # Create new archive with header
                archive_content = f"# Archive of {source_path.name}\n\n"
                logger.info(f"📦 Creating new archive file: {archive_path}")
            
            # Append archived entry
            if not archive_content.endswith('\n\n'):
                archive_content += '\n' if archive_content.endswith('\n') else '\n\n'
            archive_content += archived_entry
            
            # Write archive file
            with open(archive_path, 'w', encoding='utf-8') as f:
                f.write(archive_content)
            
            logger.info(f"✅ Entry appended to archive: {archive_path}")
            
            # Remove entry from source file
            new_source_lines = (
                source_lines[:entry_info['start_line']] +
                source_lines[entry_info['end_line']:]
            )
            
            with open(source_path, 'w', encoding='utf-8') as f:
                f.writelines(new_source_lines)
            
            logger.info(f"✅ Entry removed from source: {source_path}")
            
            return {
                "success": True,
                "archived_heading": entry_heading,
                "archive_file": str(archive_path.relative_to(source_path.parent.parent)),
                "lines_archived": len(entry_lines)
            }
            
        except Exception as e:
            logger.error(f"❌ Archive failed: {e}", exc_info=True)
            raise
    
    async def archive_all_done(
        self,
        user_id: str,
        source_file: str,
        archive_location: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Archive all DONE items from a file
        
        **BULLY!** Clean house! Move all completed tasks to the archive!
        
        Args:
            user_id: User ID
            source_file: Source file path
            archive_location: Optional custom archive location
        
        Returns:
            Dict with count of archived items
        """
        try:
            logger.info(f"📦 BULK ARCHIVE: User {user_id} archiving all DONE from {source_file}")
            
            # Get user's org directory
            from services.folder_service import FolderService
            folder_service = FolderService()
            
            # Resolve source file path
            source_path = await self._resolve_file_path(user_id, source_file, folder_service)
            if not source_path.exists():
                raise FileNotFoundError(f"Source file not found: {source_file}")
            
            # Read source file
            with open(source_path, 'r', encoding='utf-8') as f:
                source_lines = f.readlines()
            
            # Find all DONE entries
            done_entries = self._find_done_entries(source_lines)
            
            if not done_entries:
                return {
                    "success": True,
                    "archived_count": 0,
                    "message": "No DONE items to archive"
                }
            
            logger.info(f"📦 Found {len(done_entries)} DONE entries to archive")
            
            # Determine archive file path
            if archive_location:
                # Resolve relative to source file's directory if it's a relative path
                archive_path = await self._resolve_file_path(user_id, archive_location, folder_service, base_path=source_path)
            else:
                # Check user's settings for default archive location
                archive_path = await self._get_archive_path(user_id, source_file, source_path)
            
            # Ensure archive directory exists
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Read or create archive file
            if archive_path.exists():
                with open(archive_path, 'r', encoding='utf-8') as f:
                    archive_content = f.read()
            else:
                archive_content = f"# Archive of {source_path.name}\n\n"
                logger.info(f"📦 Creating new archive file: {archive_path}")
            
            # Archive each entry (in reverse order to maintain line numbers)
            archived_headings = []
            for entry_info in reversed(done_entries):
                entry_lines = entry_info['lines']
                archived_entry = self._add_archive_metadata(entry_lines, source_file)
                
                if not archive_content.endswith('\n\n'):
                    archive_content += '\n' if archive_content.endswith('\n') else '\n\n'
                archive_content += archived_entry
                
                archived_headings.append(entry_info['heading'])
            
            # Write archive file
            with open(archive_path, 'w', encoding='utf-8') as f:
                f.write(archive_content)
            
            logger.info(f"✅ {len(done_entries)} entries appended to archive")
            
            # Remove all DONE entries from source (in reverse order)
            new_source_lines = source_lines.copy()
            for entry_info in reversed(done_entries):
                new_source_lines = (
                    new_source_lines[:entry_info['start_line']] +
                    new_source_lines[entry_info['end_line']:]
                )
            
            with open(source_path, 'w', encoding='utf-8') as f:
                f.writelines(new_source_lines)
            
            logger.info(f"✅ {len(done_entries)} entries removed from source")
            
            return {
                "success": True,
                "archived_count": len(done_entries),
                "archived_headings": archived_headings,
                "archive_file": str(archive_path.relative_to(source_path.parent.parent))
            }
            
        except Exception as e:
            logger.error(f"❌ Bulk archive failed: {e}", exc_info=True)
            raise
    
    def _parse_file_archive_directive(self, content: str) -> Optional[str]:
        """
        Parse #+ARCHIVE: directive from org file header
        
        Format: #+ARCHIVE: path/to/archive.org:: or #+ARCHIVE: path/to/archive.org::* Heading
        
        Returns the archive location string (without the :: and heading part)
        """
        if not content:
            return None
        
        lines = content.split('\n')
        # Look for #+ARCHIVE: in the header (before first heading)
        for line in lines:
            # Stop at first heading
            if re.match(r'^\*+\s+', line):
                break
            
            # Match #+ARCHIVE: directive
            match = re.match(r'^#\+ARCHIVE:\s+(.+?)(?:::|$)', line, re.IGNORECASE)
            if match:
                archive_location = match.group(1).strip()
                logger.info(f"📦 Found file-level #+ARCHIVE: directive: {archive_location}")
                return archive_location
        
        return None
    
    async def _get_archive_path(
        self,
        user_id: str,
        source_file: str,
        source_path: Path
    ) -> Path:
        """
        Get archive path with priority: file-level #+ARCHIVE: > settings > default
        
        Supports format specifiers:
        - %s = source filename without extension
        - %s_archive = source filename with _archive suffix
        """
        # Priority 1: Check file-level #+ARCHIVE: directive
        try:
            if source_path.exists():
                with open(source_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                
                file_archive = self._parse_file_archive_directive(file_content)
                if file_archive:
                    # Replace format specifiers
                    source_stem = source_path.stem
                    archive_location = file_archive.replace('%s', source_stem)
                    
                    # Resolve the archive path (relative to source file's directory if it's a relative path)
                    from services.folder_service import FolderService
                    folder_service = FolderService()
                    archive_path = await self._resolve_file_path(user_id, archive_location, folder_service, base_path=source_path)
                    
                    logger.info(f"📦 Using file-level #+ARCHIVE: directive: {archive_location}")
                    return archive_path
        except Exception as e:
            logger.warning(f"⚠️ Failed to parse file-level archive directive: {e}")
        
        # Priority 2: Check user's settings
        try:
            from services.org_settings_service import OrgSettingsService
            
            settings_service = OrgSettingsService()
            org_settings = await settings_service.get_settings(user_id)
            
            # Check if user has archive preferences configured
            if (org_settings.archive_preferences and 
                org_settings.archive_preferences.default_archive_location):
                
                archive_location = org_settings.archive_preferences.default_archive_location
                
                # Replace format specifiers
                source_stem = source_path.stem  # filename without extension
                archive_location = archive_location.replace('%s', source_stem)
                
                # Resolve the archive path (relative to source file's directory if it's a relative path)
                from services.folder_service import FolderService
                folder_service = FolderService()
                archive_path = await self._resolve_file_path(user_id, archive_location, folder_service, base_path=source_path)
                
                logger.info(f"📦 Using archive location from settings: {archive_location}")
                return archive_path
        except Exception as e:
            logger.warning(f"⚠️ Failed to get archive settings, using default: {e}")
        
        # Priority 3: Default: {filename}_archive.org in same directory
        return source_path.parent / f"{source_path.stem}{self.archive_suffix}"
    
    async def _resolve_file_path(
        self,
        user_id: str,
        relative_path: str,
        folder_service,
        base_path: Optional[Path] = None
    ) -> Path:
        """
        Resolve org file path to absolute path
        
        If base_path is provided and relative_path starts with ./ or ../ or is just a filename,
        resolve relative to base_path. Otherwise, resolve relative to user's base directory.
        
        Args:
            user_id: User ID
            relative_path: Path to resolve (e.g., "./archive.org", "../archive.org", "OrgMode/tasks.org")
            folder_service: Folder service instance
            base_path: Optional base path for relative resolution (typically source file's directory)
        """
        from config import settings
        from services.database_manager.database_helpers import fetch_one
        
        # Check if this is a relative path (starts with ./ or ../ or is just a filename without any /)
        # Paths like "OrgMode/archive.org" are relative to user's base directory, not source file
        is_relative = (
            relative_path.startswith('./') or 
            relative_path.startswith('../') or
            ('/' not in relative_path and not relative_path.startswith('/'))
        )
        
        if is_relative and base_path:
            # Resolve relative to source file's directory
            resolved_path = (base_path.parent / relative_path).resolve()
            logger.info(f"📦 Resolved relative path '{relative_path}' relative to {base_path.parent} -> {resolved_path}")
            return resolved_path
        
        # Otherwise, resolve relative to user's base directory
        row = await fetch_one("SELECT username FROM users WHERE user_id = $1", user_id)
        username = row['username'] if row else user_id
        
        # Construct user's org directory
        upload_dir = Path(settings.UPLOAD_DIR)
        user_base_dir = upload_dir / "Users" / username
        
        # Handle paths relative to user directory (e.g., "OrgMode/tasks.org" or "tasks.org")
        file_path = user_base_dir / relative_path
        
        return file_path
    
    def _extract_entry(self, lines: List[str], line_number: int) -> Optional[Dict[str, Any]]:
        """
        Extract an org entry (heading + content) from file lines
        
        Returns dict with:
            - lines: List of extracted lines
            - heading: Heading text
            - start_line: Starting line index (0-based)
            - end_line: Ending line index (0-based, exclusive)
        """
        if line_number < 1 or line_number > len(lines):
            return None
        
        # Convert to 0-based index
        start_idx = line_number - 1
        
        # Find the heading at or before this line
        heading_idx = start_idx
        while heading_idx >= 0:
            if re.match(r'^\*+\s+', lines[heading_idx]):
                break
            heading_idx -= 1
        
        if heading_idx < 0:
            return None
        
        # Get heading level
        heading_match = re.match(r'^(\*+)\s+(.*)', lines[heading_idx])
        if not heading_match:
            return None
        
        heading_level = len(heading_match.group(1))
        heading_text = heading_match.group(2).strip()
        
        # Find end of entry (next heading of same or higher level)
        end_idx = heading_idx + 1
        while end_idx < len(lines):
            match = re.match(r'^(\*+)\s+', lines[end_idx])
            if match and len(match.group(1)) <= heading_level:
                break
            end_idx += 1
        
        return {
            'lines': lines[heading_idx:end_idx],
            'heading': heading_text,
            'start_line': heading_idx,
            'end_line': end_idx
        }
    
    def _find_done_entries(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Find all entries with DONE state"""
        done_entries = []
        
        for idx, line in enumerate(lines):
            # Match heading with DONE state
            match = re.match(r'^(\*+)\s+(DONE|CANCELED|CANCELLED)\s+(.*)', line, re.IGNORECASE)
            if match:
                entry_info = self._extract_entry(lines, idx + 1)  # Convert to 1-based
                if entry_info:
                    done_entries.append(entry_info)
        
        return done_entries
    
    def _add_archive_metadata(self, entry_lines: List[str], source_file: str) -> str:
        """Add ARCHIVE_TIME property to entry"""
        lines = entry_lines.copy()
        
        # Get current timestamp
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %a %H:%M")
        
        # Check if entry has PROPERTIES drawer
        has_properties = False
        properties_end_idx = None
        
        for idx, line in enumerate(lines):
            if line.strip() == ':PROPERTIES:':
                has_properties = True
            elif has_properties and line.strip() == ':END:':
                properties_end_idx = idx
                break
        
        if has_properties and properties_end_idx:
            # Add ARCHIVE_TIME before :END:
            lines.insert(properties_end_idx, f":ARCHIVE_TIME: [{timestamp}]\n")
            lines.insert(properties_end_idx + 1, f":ARCHIVE_FILE: {source_file}\n")
        else:
            # Create new PROPERTIES drawer after heading
            properties_drawer = [
                ":PROPERTIES:\n",
                f":ARCHIVE_TIME: [{timestamp}]\n",
                f":ARCHIVE_FILE: {source_file}\n",
                ":END:\n"
            ]
            # Insert after heading line
            lines = [lines[0]] + properties_drawer + lines[1:]
        
        return ''.join(lines)


# Singleton instance
_org_archive_service_instance = None


async def get_org_archive_service() -> OrgArchiveService:
    """Get or create the OrgArchiveService singleton"""
    global _org_archive_service_instance
    if _org_archive_service_instance is None:
        _org_archive_service_instance = OrgArchiveService()
    return _org_archive_service_instance



