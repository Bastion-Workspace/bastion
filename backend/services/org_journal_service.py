"""
Org-Mode Journal Service
Handles journal entry capture with multiple organization strategies
"""

import asyncio
import logging
import re
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Optional, Tuple, List, Dict, Any
from zoneinfo import ZoneInfo

from config import settings
from models.org_capture_models import OrgCaptureRequest, OrgCaptureResponse
from models.org_settings_models import JournalPreferences, JournalOrganizationMode
from services.org_capture_service import expand_capture_placeholders

logger = logging.getLogger(__name__)


class OrgJournalService:
    """Service for managing org-mode journal entries"""

    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self._capture_lock = asyncio.Lock()

    async def capture_journal_entry(
        self,
        user_id: str,
        request: OrgCaptureRequest
    ) -> OrgCaptureResponse:
        """
        Main entry point - capture journal entry to appropriate file
        
        Args:
            user_id: User ID
            request: Capture request with content and entry_date
            
        Returns:
            OrgCaptureResponse with success status and preview
        """
        try:
            # Get username for file path
            from services.database_manager.database_helpers import fetch_one

            row = await fetch_one("SELECT username FROM users WHERE user_id = $1", user_id)
            username = row['username'] if row else user_id

            # Get journal preferences
            from services.org_settings_service import get_org_settings_service
            settings_service = await get_org_settings_service()
            org_settings = await settings_service.get_settings(user_id)
            journal_prefs = org_settings.journal_preferences

            if not journal_prefs.enabled:
                logger.warning(f"Journal disabled for user {user_id}, falling back to inbox")
                from services.org_capture_service import get_org_capture_service
                capture_service = await get_org_capture_service()
                return await capture_service.capture_to_inbox(user_id, request)

            # Determine entry date (use provided date or today)
            entry_date = await self._parse_entry_date(request.entry_date, user_id)

            # Determine target file path
            file_path = await self._determine_journal_file_path(
                user_id, username, entry_date, journal_prefs
            )

            # Snapshot before write for versioning (if document is in DB)
            doc_id = await self._document_id_for_path(file_path, username, user_id)
            if doc_id:
                try:
                    from services.document_version_service import snapshot_before_write
                    await snapshot_before_write(
                        doc_id, user_id, "quick_capture_journal", None, None
                    )
                except Exception as verr:
                    logger.debug(
                        "Version snapshot before journal capture failed (non-fatal): %s",
                        verr,
                    )

            async with self._capture_lock:
                # Serialize read-modify-write per process so sequential quick captures
                # do not overwrite each other (second write would lose first entry).
                await self._ensure_hierarchy_exists(
                    file_path, entry_date, journal_prefs.organization_mode, journal_prefs
                )

                entry_text = await self._append_journal_entry(
                    file_path, entry_date, request, journal_prefs, user_id
                )

                with open(file_path, 'r', encoding='utf-8') as f:
                    line_count = len(f.readlines())

            logger.info(f"JOURNAL CAPTURE: Added entry to {file_path.name}")

            # Update document_metadata.updated_at so the editor "Saved" metric reflects the write
            try:
                from services.folder_service import FolderService
                folder_service = FolderService()
                user_base_dir = self.upload_dir / "Users" / username
                journal_dir = file_path.parent
                folder_id = None
                if journal_dir != user_base_dir:
                    folder_id = await folder_service.get_folder_id_by_physical_path(journal_dir, user_id)
                doc_repo = folder_service.document_repository
                doc = await doc_repo.find_by_filename_and_context(
                    file_path.name, user_id, "user", folder_id
                )
                if doc:
                    await doc_repo.touch_updated_at(doc.document_id, user_id)
            except Exception as touch_err:
                logger.debug(f"Could not touch document updated_at after journal write: {touch_err}")

            return OrgCaptureResponse(
                success=True,
                message=f"Successfully captured to {file_path.name}",
                entry_preview=entry_text.strip(),
                file_path=str(file_path.relative_to(self.upload_dir)),
                line_number=line_count
            )

        except Exception as e:
            logger.error(f"❌ Journal capture failed: {e}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
            return OrgCaptureResponse(
                success=False,
                message=f"Failed to capture journal entry: {str(e)}",
                entry_preview=None,
                file_path=None,
                line_number=None
            )
    
    async def _parse_entry_date(
        self, entry_date_str: Optional[str], user_id: str
    ) -> date:
        """Parse entry date string or return today's date in user's timezone"""
        if entry_date_str:
            try:
                return datetime.strptime(entry_date_str, "%Y-%m-%d").date()
            except ValueError:
                logger.warning(f"Invalid date format: {entry_date_str}, using today")
        
        # Get user's timezone for "today"
        from services.settings_service import SettingsService
        settings_service = SettingsService()
        user_timezone = await settings_service.get_user_timezone(user_id)
        
        try:
            tz = ZoneInfo(user_timezone)
            return datetime.now(tz).date()
        except Exception as e:
            logger.warning(f"Invalid timezone {user_timezone}, using UTC: {e}")
            return datetime.now().date()
    
    async def _determine_journal_file_path(
        self,
        user_id: str,
        username: str,
        entry_date: date,
        journal_prefs: JournalPreferences
    ) -> Path:
        """Determine target file path based on organization mode and date"""
        user_base_dir = self.upload_dir / "Users" / username
        
        # Build journal location path
        if journal_prefs.journal_location:
            journal_dir = user_base_dir / journal_prefs.journal_location
        else:
            journal_dir = user_base_dir
        
        # Ensure journal directory exists
        journal_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine filename based on organization mode
        if journal_prefs.organization_mode == JournalOrganizationMode.MONOLITHIC:
            filename = "journal.org"
        elif journal_prefs.organization_mode == JournalOrganizationMode.YEARLY:
            filename = f"{entry_date.year} - Journal.org"
        elif journal_prefs.organization_mode == JournalOrganizationMode.MONTHLY:
            filename = f"{entry_date.strftime('%Y-%m')} - Journal.org"
        elif journal_prefs.organization_mode == JournalOrganizationMode.DAILY:
            filename = f"{entry_date.strftime('%Y-%m-%d')} - Journal.org"
        else:
            filename = "journal.org"  # Fallback
        
        return journal_dir / filename
    
    async def _ensure_hierarchy_exists(
        self,
        file_path: Path,
        entry_date: date,
        organization_mode: JournalOrganizationMode,
        journal_prefs: JournalPreferences
    ) -> None:
        """
        Create year/month/day headings if missing, inserting in chronological order
        
        This reads the file, checks for existing headings, and inserts
        missing hierarchy levels in the correct chronological position.
        """
        # If file doesn't exist, create it with frontmatter
        if not file_path.exists():
            file_title = self._generate_file_title(organization_mode, entry_date)
            
            # Get default tags from journal preferences for file frontmatter
            default_tags = journal_prefs.default_tags if journal_prefs.default_tags else ["journal"]
            filetags_str = ":" + ":".join(default_tags) + ":"
            
            frontmatter = f"#+TITLE: {file_title}\n#+FILETAGS: {filetags_str}\n\n"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(frontmatter)
            logger.info(f"📝 Created new journal file: {file_path.name}")
        
        # Read existing file content
        file_lines = []
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                file_lines = f.readlines()
        
        existing_content = ''.join(file_lines)
        
        # Parse existing headings
        existing_headings = self._parse_headings(existing_content)
        
        # Determine which headings we need
        needed_headings = self._get_needed_headings(entry_date, organization_mode)
        
        # Check what's missing
        missing_headings = []
        for level, heading_text in needed_headings:
            # Check if this heading already exists
            if not self._heading_exists(existing_headings, level, heading_text):
                missing_headings.append((level, heading_text))
        
        # If no missing headings, we're done
        if not missing_headings:
            return
        
        # Process missing headings in order (parent levels first)
        # This ensures month headings are created before day headings in yearly mode
        missing_headings_sorted = sorted(missing_headings, key=lambda x: x[0])  # Sort by level
        
        for level, heading_text in missing_headings_sorted:
            heading_line = f"{'*' * level} {heading_text}\n"
            
            # Extract date from heading text for chronological comparison
            # Format: "2026-01-15 Thursday" or "2026-01" or "2026"
            date_match = re.match(r'(\d{4}-\d{2}-\d{2})', heading_text)
            if date_match:
                target_date_str = date_match.group(1)
                try:
                    target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
                except ValueError:
                    # If date parsing fails, append to end
                    file_lines.append(heading_line)
                    continue
            else:
                # For year/month headings, try to parse
                year_month_match = re.match(r'(\d{4}-\d{2})', heading_text)
                if year_month_match:
                    target_date_str = year_month_match.group(1) + "-01"
                    try:
                        target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
                    except ValueError:
                        file_lines.append(heading_line)
                        continue
                else:
                    # Year-only format (e.g., "2024" for monolithic mode)
                    year_match = re.match(r'^(\d{4})$', heading_text)
                    if year_match:
                        year_str = year_match.group(1)
                        try:
                            target_date = datetime.strptime(f"{year_str}-01-01", "%Y-%m-%d").date()
                        except ValueError:
                            file_lines.append(heading_line)
                            continue
                    else:
                        # Unknown format - append to end
                        file_lines.append(heading_line)
                        continue
            
            # Find insertion point based on level and hierarchy
            insertion_point = len(file_lines)
            
            if level == 1:
                # Level 1 headings (years in monolithic mode, months in yearly mode, days in monthly mode)
                # Insert chronologically among other Level 1 headings
                for i, line in enumerate(file_lines):
                    # Skip blank lines when looking for headings
                    if line.strip() == '':
                        continue
                    
                    match = re.match(r'^(\*+)\s+(.+)$', line)
                    if match and len(match.group(1)) == level:
                        heading_text_existing = match.group(2).strip()
                        
                        # Try to extract date from existing heading
                        date_match_existing = re.match(r'(\d{4}-\d{2}-\d{2})', heading_text_existing)
                        if date_match_existing:
                            existing_date_str = date_match_existing.group(1)
                            try:
                                existing_date = datetime.strptime(existing_date_str, "%Y-%m-%d").date()
                                if existing_date > target_date:
                                    insertion_point = i
                                    break
                            except ValueError:
                                pass
                        else:
                            # Try year-month format
                            year_month_match_existing = re.match(r'(\d{4}-\d{2})', heading_text_existing)
                            if year_month_match_existing:
                                existing_date_str = year_month_match_existing.group(1) + "-01"
                                try:
                                    existing_date = datetime.strptime(existing_date_str, "%Y-%m-%d").date()
                                    if existing_date > target_date:
                                        insertion_point = i
                                        break
                                except ValueError:
                                    pass
                            else:
                                # Try year-only format (for monolithic mode)
                                year_match_existing = re.match(r'^(\d{4})$', heading_text_existing)
                                if year_match_existing:
                                    year_str_existing = year_match_existing.group(1)
                                    try:
                                        existing_date = datetime.strptime(f"{year_str_existing}-01-01", "%Y-%m-%d").date()
                                        if existing_date > target_date:
                                            insertion_point = i
                                            break
                                    except ValueError:
                                        pass
            else:
                # Level 2+ headings (days in yearly mode, entries in monthly mode)
                # Need to find parent heading first, then insert within that section
                parent_level = level - 1
                parent_heading_text = None
                
                # Determine parent heading text based on organization mode and level
                if organization_mode == JournalOrganizationMode.MONOLITHIC:
                    if level == 2:
                        # Level 2 (month) has Level 1 (year) as parent
                        year_str = str(entry_date.year)
                        parent_heading_text = year_str
                    elif level == 3:
                        # Level 3 (day) has Level 2 (month) as parent
                        month_str = entry_date.strftime('%Y-%m')
                        parent_heading_text = month_str
                    else:
                        # For other levels, use month as fallback
                        month_str = entry_date.strftime('%Y-%m')
                        parent_heading_text = month_str
                elif organization_mode == JournalOrganizationMode.YEARLY:
                    # Level 2 (day) has Level 1 (month) as parent
                    month_str = entry_date.strftime('%Y-%m')
                    parent_heading_text = month_str
                else:
                    # For monthly/daily mode, Level 2 entries go under Level 1 date headers
                    # Parent is the date header we're looking for
                    day_name = entry_date.strftime('%A')
                    date_str = entry_date.strftime('%Y-%m-%d')
                    parent_heading_text = f"{date_str} {day_name}"
                
                # Find the parent heading section
                parent_start = -1
                parent_end = len(file_lines)
                
                for i, line in enumerate(file_lines):
                    match = re.match(r'^(\*+)\s+(.+)$', line)
                    if match:
                        line_level = len(match.group(1))
                        line_text = match.group(2).strip()
                        
                        if line_level == parent_level and line_text == parent_heading_text:
                            parent_start = i
                        elif line_level <= parent_level and parent_start != -1:
                            # Found next heading at same or higher level - end of parent section
                            parent_end = i
                            break
                
                if parent_start == -1:
                    # Parent heading not found - this shouldn't happen if processing in order
                    # But fallback: append to end
                    logger.warning(f"Parent heading '{parent_heading_text}' not found for {heading_text}, appending to end")
                    file_lines.append(heading_line)
                    continue
                
                # Now find insertion point within parent section (chronologically)
                insertion_point = parent_start + 1
                
                # Skip any blank lines immediately after parent heading to find first content or heading
                while insertion_point < parent_end and file_lines[insertion_point].strip() == '':
                    insertion_point += 1
                
                # Track the last heading at our level we've seen
                last_heading_at_level = insertion_point - 1
                
                for i in range(insertion_point, parent_end):
                    line = file_lines[i]
                    match = re.match(r'^(\*+)\s+(.+)$', line)
                    if match:
                        line_level = len(match.group(1))
                        if line_level == level:
                            # Found a heading at our level
                            last_heading_at_level = i
                            heading_text_existing = match.group(2).strip()
                            
                            # Try to extract date from existing heading
                            date_match_existing = re.match(r'(\d{4}-\d{2}-\d{2})', heading_text_existing)
                            if date_match_existing:
                                existing_date_str = date_match_existing.group(1)
                                try:
                                    existing_date = datetime.strptime(existing_date_str, "%Y-%m-%d").date()
                                    if existing_date > target_date:
                                        # Insert right before this heading (headers should be adjacent)
                                        insertion_point = i
                                        break
                                    else:
                                        # Keep moving insertion point forward
                                        insertion_point = i + 1
                                except ValueError:
                                    pass
                        elif line_level < level:
                            # Found a heading at higher level - end of section
                            break
            
            # Ensure headers are directly adjacent - remove any blank lines between headers
            # If insertion_point points to a blank line, and the next non-blank line is a header at our level,
            # insert at the header position instead
            if insertion_point < len(file_lines):
                # Check if we're inserting before a blank line that's before a header
                if file_lines[insertion_point].strip() == '':
                    # Look ahead for the next non-blank line
                    for i in range(insertion_point + 1, min(insertion_point + 5, len(file_lines))):
                        if file_lines[i].strip() == '':
                            continue
                        match = re.match(r'^(\*+)\s+', file_lines[i])
                        if match:
                            line_level = len(match.group(1))
                            # If it's a header at our level or higher, insert at that position (headers should be adjacent)
                            if line_level <= level:
                                insertion_point = i
                                break
                        # If it's content, keep the blank line (content can be separated from headers)
                        break
            
            # Insert the heading at the correct position
            file_lines.insert(insertion_point, heading_line)
            logger.info(f"📝 Inserted missing heading: {heading_line.strip()}")
        
        # Write the file back with inserted headings
        if missing_headings:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(file_lines)
            hierarchy_list = [f"{'*' * level} {text}" for level, text in missing_headings]
            logger.info(f"📝 Created missing hierarchy: {hierarchy_list}")
    
    def _get_needed_headings(
        self, entry_date: date, organization_mode: JournalOrganizationMode
    ) -> list[Tuple[int, str]]:
        """Get list of (level, heading_text) tuples needed for this entry"""
        day_name = entry_date.strftime('%A')
        date_str = entry_date.strftime('%Y-%m-%d')
        month_str = entry_date.strftime('%Y-%m')
        year_str = str(entry_date.year)
        
        if organization_mode == JournalOrganizationMode.MONOLITHIC:
            return [
                (1, year_str),
                (2, month_str),
                (3, f"{date_str} {day_name}")
            ]
        elif organization_mode == JournalOrganizationMode.YEARLY:
            return [
                (1, month_str),
                (2, f"{date_str} {day_name}")
            ]
        elif organization_mode == JournalOrganizationMode.MONTHLY:
            return [
                (1, f"{date_str} {day_name}")
            ]
        elif organization_mode == JournalOrganizationMode.DAILY:
            return [
                (1, f"{date_str} {day_name}")
            ]
        else:
            return []
    
    def _parse_headings(self, content: str) -> list[Tuple[int, str]]:
        """Parse org-mode headings from content: returns list of (level, text)"""
        headings = []
        for line in content.split('\n'):
            match = re.match(r'^(\*+)\s+(.+)$', line)
            if match:
                level = len(match.group(1))
                text = match.group(2).strip()
                headings.append((level, text))
        return headings
    
    def _heading_exists(
        self, existing_headings: list[Tuple[int, str]], level: int, text: str
    ) -> bool:
        """Check if a heading exists at the given level with matching text"""
        for h_level, h_text in existing_headings:
            if h_level == level and h_text == text:
                return True
        return False
    
    def _find_last_heading_at_level(self, content: str, level: int) -> int:
        """Find position of last heading at given level (returns line number or -1)"""
        lines = content.split('\n')
        last_pos = -1
        for i, line in enumerate(lines):
            match = re.match(r'^(\*+)\s+', line)
            if match and len(match.group(1)) == level:
                last_pos = i
        return last_pos
    
    def _find_entry_end(self, file_lines: list[str], entry_header_line: int, entry_level: int, max_line: int) -> int:
        """
        Find where an entry's content ends.
        
        An entry consists of:
        - Header line (at entry_level)
        - Content lines (not headings, or headings at deeper levels)
        - Entry ends when we hit a heading at same or higher level, or max_line
        
        Returns the line number where the next entry should be inserted.
        """
        # Start after the header line
        current_line = entry_header_line + 1
        
        # Skip blank lines immediately after header
        while current_line < max_line and file_lines[current_line].strip() == '':
            current_line += 1
        
        # Now look for where content ends
        # Content ends when we hit a heading at entry_level or higher (same or higher level)
        for i in range(current_line, max_line):
            line = file_lines[i]
            match = re.match(r'^(\*+)\s+', line)
            if match:
                line_level = len(match.group(1))
                # If we find a heading at same or higher level, content ends here
                if line_level <= entry_level:
                    return i
        
        # If we didn't find another heading, content goes to max_line
        return max_line

    def _get_date_section_bounds(
        self,
        file_lines: List[str],
        entry_date: date,
        journal_prefs: JournalPreferences,
    ) -> Optional[Tuple[int, int, int, str]]:
        """
        Find the line bounds for a date's section in an already-read file.
        Returns (header_line_idx, content_start, content_end, header_line_text) or None.
        Content is file_lines[content_start:content_end] (body only, no header).
        """
        day_name = entry_date.strftime('%A')
        date_str = entry_date.strftime('%Y-%m-%d')
        if journal_prefs.organization_mode == JournalOrganizationMode.MONOLITHIC:
            header_level = 3
        elif journal_prefs.organization_mode == JournalOrganizationMode.YEARLY:
            header_level = 2
        else:
            header_level = 1
        target_header = f"{'*' * header_level} {date_str} {day_name}"
        date_header_line = -1
        header_line_text = ""
        for i, line in enumerate(file_lines):
            stripped = line.rstrip('\n').strip()
            if stripped == target_header:
                date_header_line = i
                header_line_text = line.rstrip('\n')
                break
        if date_header_line == -1:
            return None
        content_start = date_header_line + 1
        content_end = self._find_entry_end(file_lines, date_header_line, header_level, len(file_lines))
        return (date_header_line, content_start, content_end, header_line_text)

    async def _get_username_and_journal_prefs(self, user_id: str) -> Tuple[str, JournalPreferences]:
        """Get username and journal preferences; raises if journal disabled."""
        from services.database_manager.database_helpers import fetch_one
        from services.org_settings_service import get_org_settings_service
        row = await fetch_one("SELECT username FROM users WHERE user_id = $1", user_id)
        username = row['username'] if row else user_id
        settings_service = await get_org_settings_service()
        org_settings = await settings_service.get_settings(user_id)
        journal_prefs = org_settings.journal_preferences
        if not journal_prefs.enabled:
            raise ValueError("Journal is disabled for this user")
        return username, journal_prefs

    async def get_journal_entry(
        self, user_id: str, date_str: str
    ) -> Dict[str, Any]:
        """
        Read one date's journal entry. date_str is YYYY-MM-DD or "today".
        Returns dict with content, date, heading, document_id, file_path, has_content, error.
        """
        try:
            entry_date = await self._parse_entry_date(
                None if date_str.strip().lower() == "today" else date_str, user_id
            )
            username, journal_prefs = await self._get_username_and_journal_prefs(user_id)
            file_path = await self._determine_journal_file_path(
                user_id, username, entry_date, journal_prefs
            )
            if not file_path.exists():
                return {
                    "success": False,
                    "content": "",
                    "date": entry_date.strftime("%Y-%m-%d"),
                    "heading": "",
                    "document_id": None,
                    "file_path": None,
                    "has_content": False,
                    "error": "Journal file does not exist",
                }
            with open(file_path, 'r', encoding='utf-8') as f:
                file_lines = f.readlines()
            bounds = self._get_date_section_bounds(file_lines, entry_date, journal_prefs)
            if not bounds:
                return {
                    "success": True,
                    "content": "",
                    "date": entry_date.strftime("%Y-%m-%d"),
                    "heading": "",
                    "document_id": await self._document_id_for_path(file_path, username, user_id),
                    "file_path": str(file_path.relative_to(self.upload_dir)),
                    "has_content": False,
                    "error": None,
                }
            _, content_start, content_end, header_line_text = bounds
            content_lines = file_lines[content_start:content_end]
            content = ''.join(content_lines).rstrip()
            document_id = await self._document_id_for_path(file_path, username, user_id)
            return {
                "success": True,
                "content": content,
                "date": entry_date.strftime("%Y-%m-%d"),
                "heading": header_line_text,
                "document_id": document_id,
                "file_path": str(file_path.relative_to(self.upload_dir)),
                "has_content": bool(content.strip()),
                "error": None,
            }
        except ValueError as e:
            return {
                "success": False,
                "content": "",
                "date": date_str if date_str.strip().lower() != "today" else "",
                "heading": "",
                "document_id": None,
                "file_path": None,
                "has_content": False,
                "error": str(e),
            }
        except Exception as e:
            logger.exception("get_journal_entry failed")
            return {
                "success": False,
                "content": "",
                "date": "",
                "heading": "",
                "document_id": None,
                "file_path": None,
                "has_content": False,
                "error": str(e),
            }

    async def _document_id_for_path(
        self, file_path: Path, username: str, user_id: str
    ) -> Optional[str]:
        """Resolve document_id for a journal file path."""
        try:
            from services.folder_service import FolderService
            folder_service = FolderService()
            user_base_dir = self.upload_dir / "Users" / username
            journal_dir = file_path.parent
            folder_id = None
            if journal_dir != user_base_dir:
                folder_id = await folder_service.get_folder_id_by_physical_path(journal_dir, user_id)
            doc = await folder_service.document_repository.find_by_filename_and_context(
                file_path.name, user_id, "user", folder_id
            )
            return doc.document_id if doc else None
        except Exception:
            return None

    async def update_journal_entry(
        self, user_id: str, date_str: str, content: str, mode: str
    ) -> Dict[str, Any]:
        """
        Update a single date's journal section. mode is "replace" or "append".
        Only modifies that date's body; holds capture lock.
        """
        if mode not in ("replace", "append"):
            return {"success": False, "date": date_str, "error": f"Invalid mode: {mode}"}
        try:
            entry_date = await self._parse_entry_date(date_str, user_id)
            username, journal_prefs = await self._get_username_and_journal_prefs(user_id)
            file_path = await self._determine_journal_file_path(
                user_id, username, entry_date, journal_prefs
            )
            async with self._capture_lock:
                await self._ensure_hierarchy_exists(
                    file_path, entry_date, journal_prefs.organization_mode, journal_prefs
                )
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_lines = f.readlines()
                bounds = self._get_date_section_bounds(file_lines, entry_date, journal_prefs)
                if not bounds:
                    return {"success": False, "date": entry_date.strftime("%Y-%m-%d"), "error": "Date heading not found"}
                header_line_idx, content_start, content_end, _ = bounds
                if mode == "replace":
                    new_body = content.rstrip()
                    if not new_body.endswith('\n'):
                        new_body += '\n'
                    file_lines[content_start:content_end] = [new_body]
                else:
                    existing = ''.join(file_lines[content_start:content_end]).rstrip()
                    appended = content.rstrip()
                    combined = (existing + '\n\n' + appended).strip() + '\n' if (existing or appended) else ''
                    file_lines[content_start:content_end] = [combined]
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(file_lines)
                try:
                    await self._touch_document_updated_at(file_path, username, user_id)
                except Exception as touch_err:
                    logger.debug("Could not touch document updated_at: %s", touch_err)
            return {"success": True, "date": entry_date.strftime("%Y-%m-%d"), "error": None}
        except ValueError as e:
            return {"success": False, "date": date_str, "error": str(e)}
        except Exception as e:
            logger.exception("update_journal_entry failed")
            return {"success": False, "date": date_str, "error": str(e)}

    async def _touch_document_updated_at(
        self, file_path: Path, username: str, user_id: str
    ) -> None:
        """Update document_metadata.updated_at for a journal file."""
        from services.folder_service import FolderService
        folder_service = FolderService()
        user_base_dir = self.upload_dir / "Users" / username
        journal_dir = file_path.parent
        folder_id = None
        if journal_dir != user_base_dir:
            folder_id = await folder_service.get_folder_id_by_physical_path(journal_dir, user_id)
        doc = await folder_service.document_repository.find_by_filename_and_context(
            file_path.name, user_id, "user", folder_id
        )
        if doc:
            await folder_service.document_repository.touch_updated_at(doc.document_id, user_id)

    async def list_journal_entries(
        self,
        user_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        List journal entries in a date range with metadata (date, word_count, has_content).
        """
        try:
            username, journal_prefs = await self._get_username_and_journal_prefs(user_id)
            if start_date:
                try:
                    start_d = datetime.strptime(start_date, "%Y-%m-%d").date()
                except ValueError:
                    return {"success": False, "entries": [], "total": 0, "error": f"Invalid start_date: {start_date}"}
            else:
                start_d = date.today() - timedelta(days=31)
            if end_date:
                try:
                    end_d = datetime.strptime(end_date, "%Y-%m-%d").date()
                except ValueError:
                    return {"success": False, "entries": [], "total": 0, "error": f"Invalid end_date: {end_date}"}
            else:
                end_d = date.today()
            if start_d > end_d:
                return {"success": False, "entries": [], "total": 0, "error": "start_date must be before end_date"}
            entries: List[Dict[str, Any]] = []
            seen_files: Dict[Path, List[str]] = {}
            current = start_d
            while current <= end_d:
                file_path = await self._determine_journal_file_path(
                    user_id, username, current, journal_prefs
                )
                if not file_path.exists():
                    current += timedelta(days=1)
                    continue
                if file_path not in seen_files:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        seen_files[file_path] = f.readlines()
                file_lines = seen_files[file_path]
                bounds = self._get_date_section_bounds(file_lines, current, journal_prefs)
                if bounds:
                    _, cs, ce, _ = bounds
                    body = ''.join(file_lines[cs:ce]).strip()
                    word_count = len(body.split()) if body else 0
                    entries.append({
                        "date": current.strftime("%Y-%m-%d"),
                        "word_count": word_count,
                        "has_content": bool(body),
                    })
                current += timedelta(days=1)
            return {"success": True, "entries": entries, "total": len(entries), "error": None}
        except ValueError as e:
            return {"success": False, "entries": [], "total": 0, "error": str(e)}
        except Exception as e:
            logger.exception("list_journal_entries failed")
            return {"success": False, "entries": [], "total": 0, "error": str(e)}

    async def get_journal_entries(
        self,
        user_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_entries: int = 100,
    ) -> Dict[str, Any]:
        """
        Get full content of journal entries in a date range (for review/lookback).
        Returns list of {date, content, heading, has_content}; stops after max_entries.
        """
        try:
            username, journal_prefs = await self._get_username_and_journal_prefs(user_id)
            if start_date:
                try:
                    start_d = datetime.strptime(start_date, "%Y-%m-%d").date()
                except ValueError:
                    return {"success": False, "entries": [], "total": 0, "error": f"Invalid start_date: {start_date}"}
            else:
                start_d = date.today() - timedelta(days=31)
            if end_date:
                try:
                    end_d = datetime.strptime(end_date, "%Y-%m-%d").date()
                except ValueError:
                    return {"success": False, "entries": [], "total": 0, "error": f"Invalid end_date: {end_date}"}
            else:
                end_d = date.today()
            if start_d > end_d:
                return {"success": False, "entries": [], "total": 0, "error": "start_date must be before end_date"}
            cap = max(1, min(max_entries, 500))
            entries: List[Dict[str, Any]] = []
            seen_files: Dict[Path, List[str]] = {}
            current = start_d
            while current <= end_d and len(entries) < cap:
                file_path = await self._determine_journal_file_path(
                    user_id, username, current, journal_prefs
                )
                if not file_path.exists():
                    current += timedelta(days=1)
                    continue
                if file_path not in seen_files:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        seen_files[file_path] = f.readlines()
                file_lines = seen_files[file_path]
                bounds = self._get_date_section_bounds(file_lines, current, journal_prefs)
                if bounds:
                    _, cs, ce, header_line_text = bounds
                    body = ''.join(file_lines[cs:ce]).rstrip()
                    entries.append({
                        "date": current.strftime("%Y-%m-%d"),
                        "content": body,
                        "heading": header_line_text,
                        "has_content": bool(body.strip()),
                    })
                current += timedelta(days=1)
            return {"success": True, "entries": entries, "total": len(entries), "error": None}
        except ValueError as e:
            return {"success": False, "entries": [], "total": 0, "error": str(e)}
        except Exception as e:
            logger.exception("get_journal_entries failed")
            return {"success": False, "entries": [], "total": 0, "error": str(e)}

    async def search_journal_entries(
        self,
        user_id: str,
        query: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search within journal entry content in a date range. Returns list of {date, excerpt}."""
        try:
            username, journal_prefs = await self._get_username_and_journal_prefs(user_id)
            if start_date:
                try:
                    start_d = datetime.strptime(start_date, "%Y-%m-%d").date()
                except ValueError:
                    return {"success": False, "results": [], "count": 0, "error": f"Invalid start_date: {start_date}"}
            else:
                start_d = date.today() - timedelta(days=365)
            if end_date:
                try:
                    end_d = datetime.strptime(end_date, "%Y-%m-%d").date()
                except ValueError:
                    return {"success": False, "results": [], "count": 0, "error": f"Invalid end_date: {end_date}"}
            else:
                end_d = date.today()
            if start_d > end_d:
                return {"success": False, "results": [], "count": 0, "error": "start_date must be before end_date"}
            query_lower = query.lower()
            results: List[Dict[str, str]] = []
            seen_files: Dict[Path, List[str]] = {}
            current = start_d
            while current <= end_d:
                file_path = await self._determine_journal_file_path(
                    user_id, username, current, journal_prefs
                )
                if not file_path.exists():
                    current += timedelta(days=1)
                    continue
                if file_path not in seen_files:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        seen_files[file_path] = f.readlines()
                file_lines = seen_files[file_path]
                bounds = self._get_date_section_bounds(file_lines, current, journal_prefs)
                if bounds:
                    _, cs, ce, _ = bounds
                    body = ''.join(file_lines[cs:ce])
                    if query_lower in body.lower():
                        excerpt = body.strip()[:300]
                        if len(body.strip()) > 300:
                            excerpt += "..."
                        results.append({"date": current.strftime("%Y-%m-%d"), "excerpt": excerpt})
                current += timedelta(days=1)
            return {"success": True, "results": results, "count": len(results), "error": None}
        except ValueError as e:
            return {"success": False, "results": [], "count": 0, "error": str(e)}
        except Exception as e:
            logger.exception("search_journal_entries failed")
            return {"success": False, "results": [], "count": 0, "error": str(e)}

    async def _append_journal_entry(
        self,
        file_path: Path,
        entry_date: date,
        request: OrgCaptureRequest,
        journal_prefs: JournalPreferences,
        user_id: str
    ) -> str:
        """Format and insert journal entry under the appropriate date header"""
        # Get user's timezone for timestamp
        from services.settings_service import SettingsService
        settings_service = SettingsService()
        user_timezone = await settings_service.get_user_timezone(user_id)
        
        try:
            tz = ZoneInfo(user_timezone)
            now = datetime.now(tz)
        except Exception as e:
            logger.warning(f"Invalid timezone {user_timezone}, using UTC: {e}")
            now = datetime.now()
        
        timestamp = now.strftime("%Y-%m-%d %a %H:%M")
        
        # Build entry as body text under the date header only (no automatic subheading).
        # User can add their own * / ** headers in the content if they want.
        lines = []
        if journal_prefs.include_timestamps:
            lines.append(f":PROPERTIES:")
            lines.append(f":CREATED: [{timestamp}]")
            lines.append(f":END:")
        if request.content.strip():
            body = expand_capture_placeholders(request.content.strip(), now)
            lines.append(body)
        entry_text = '\n'.join(lines)
        
        # Read existing file content
        with open(file_path, 'r', encoding='utf-8') as f:
            file_lines = f.readlines()
        
        # Find the date header for this entry
        day_name = entry_date.strftime('%A')
        date_str = entry_date.strftime('%Y-%m-%d')
        
        # Determine what date header we're looking for based on organization mode
        if journal_prefs.organization_mode == JournalOrganizationMode.MONOLITHIC:
            target_header = f"* {date_str} {day_name}"
            header_level = 3
        elif journal_prefs.organization_mode == JournalOrganizationMode.YEARLY:
            target_header = f"* {date_str} {day_name}"
            header_level = 2
        elif journal_prefs.organization_mode == JournalOrganizationMode.MONTHLY:
            target_header = f"* {date_str} {day_name}"
            header_level = 1
        elif journal_prefs.organization_mode == JournalOrganizationMode.DAILY:
            target_header = f"* {date_str} {day_name}"
            header_level = 1
        else:
            target_header = f"* {date_str} {day_name}"
            header_level = 1
        
        # Find the line number where the date header is located
        date_header_line = -1
        for i, line in enumerate(file_lines):
            stripped = line.strip()
            if stripped == target_header:
                date_header_line = i
                break
        
        # If date header not found, append to end (shouldn't happen if _ensure_hierarchy_exists worked)
        if date_header_line == -1:
            logger.warning(f"Date header '{target_header}' not found, appending to end of file")
            with open(file_path, 'a', encoding='utf-8') as f:
                if file_lines and not file_lines[-1].endswith('\n'):
                    f.write('\n')
                f.write(entry_text)
                f.write('\n')
            return entry_text
        
        # Find insertion point: right after the date header, before any entries for that date
        # Look for the next heading at the same or higher level (which would be a different date)
        next_date_header_line = -1
        for i in range(date_header_line + 1, len(file_lines)):
            line = file_lines[i]
            # Check if this is a heading at the same or higher level as the date header
            match = re.match(r'^(\*+)\s+', line)
            if match:
                line_level = len(match.group(1))
                if line_level <= header_level:
                    # Found a heading at same or higher level - this is the next date section
                    next_date_header_line = i
                    break
        
        # Determine the search boundary (end of this date's section)
        search_end = next_date_header_line if next_date_header_line != -1 else len(file_lines)
        
        # Append at end of date section (no subheading; content only)
        insertion_line = search_end
        has_existing_content = search_end > date_header_line + 1
        separator = '\n' if has_existing_content else ''
        entry_with_newlines = separator + entry_text + '\n'
        file_lines.insert(insertion_line, entry_with_newlines)
        
        # Write the file back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(file_lines)
        
        return entry_text
    
    async def _validate_journal_location(
        self,
        user_id: str,
        username: str,
        journal_location: Optional[str]
    ) -> Tuple[bool, str]:
        """
        Validate that journal location folder exists
        
        Returns:
            (is_valid, error_message)
        """
        user_base_dir = self.upload_dir / "Users" / username
        
        if not journal_location:
            # Empty location means root - always valid
            return True, ""
        
        # Build full path
        journal_path = user_base_dir / journal_location
        
        # Check if it exists and is a directory
        if journal_path.exists():
            if journal_path.is_dir():
                return True, ""
            else:
                return False, f"Journal location exists but is not a directory: {journal_location}"
        else:
            return False, f"Journal location does not exist: {journal_location}. Please create the folder first."
    
    def _generate_file_title(
        self, organization_mode: JournalOrganizationMode, entry_date: date
    ) -> str:
        """Generate file title for frontmatter"""
        if organization_mode == JournalOrganizationMode.MONOLITHIC:
            return "Life Journal"
        elif organization_mode == JournalOrganizationMode.YEARLY:
            return f"{entry_date.year} Journal"
        elif organization_mode == JournalOrganizationMode.MONTHLY:
            return f"{entry_date.strftime('%B %Y')} Journal"
        elif organization_mode == JournalOrganizationMode.DAILY:
            return f"Journal - {entry_date.strftime('%A, %B %d, %Y')}"
        else:
            return "Journal"


# Singleton instance
_org_journal_service: Optional[OrgJournalService] = None


async def get_org_journal_service() -> OrgJournalService:
    """Get or create the org journal service instance"""
    global _org_journal_service
    
    if _org_journal_service is None:
        _org_journal_service = OrgJournalService()
        logger.info("Journal service initialized")
    
    return _org_journal_service
