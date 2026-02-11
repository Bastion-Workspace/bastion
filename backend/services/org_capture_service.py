"""
Org-Mode Quick Capture Service
Emacs-style quick capture to inbox.org
"""

import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Optional

from config import settings
from models.org_capture_models import OrgCaptureRequest, OrgCaptureResponse

logger = logging.getLogger(__name__)


def expand_capture_placeholders(content: str, now: datetime) -> str:
    """
    Expand Org-Mode capture template %-escapes in capture content.

    Supported (same semantics as Emacs org-capture):
    - %t   - Active timestamp, date only, e.g. [2025-02-06 Thu]
    - %T   - Active timestamp, date and time, e.g. [2025-02-06 Thu 14:30]
    - %u   - Inactive timestamp, date only, e.g. 2025-02-06 Thu
    - %U   - Inactive timestamp, date and time, e.g. 2025-02-06 Thu 14:30
    - %<FORMAT> - Custom format using Python strftime codes, e.g. %<%I:%M %p> → 02:30 PM

    For time-only in a journal (date is the day header), use %<%I:%M %p> for 12h or %<%H:%M> for 24h.
    """
    if not content:
        return content
    ts_datetime = now.strftime("%Y-%m-%d %a %H:%M")
    ts_date = now.strftime("%Y-%m-%d %a")
    out = content
    # %<FORMAT> first (custom strftime); use regex so we don't touch %t/%T/%u/%U inside
    def replace_format(match: re.Match) -> str:
        fmt = match.group(1)
        try:
            return now.strftime(fmt)
        except (ValueError, TypeError):
            return match.group(0)
    out = re.sub(r"%<([^>]+)>", replace_format, out)
    out = out.replace("%T", f"[{ts_datetime}]")
    out = out.replace("%t", f"[{ts_date}]")
    out = out.replace("%U", ts_datetime)
    out = out.replace("%u", ts_date)
    return out


class OrgCaptureService:
    """Service for quick-capturing content to inbox.org"""
    
    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
    
    async def _find_inbox_org(self, user_id: str, username: str) -> Path:
        """
        Find or create inbox.org for user
        
        **BULLY!** Smart discovery - find existing inbox or create new one!
        
        Strategy:
        1. Check user's org-mode settings for configured inbox_file
        2. If not configured, search user's entire directory tree for inbox.org
        3. If multiple found, use the first one
        4. If none found, create at Users/{username}/inbox.org
        5. Save discovered/created location to settings for future use
        """
        try:
            # Check org-mode settings for configured inbox
            from services.org_settings_service import get_org_settings_service
            
            org_settings_service = await get_org_settings_service()
            settings_obj = await org_settings_service.get_settings(user_id)
            
            user_base_dir = self.upload_dir / "Users" / username
            
            # If inbox_file is configured in settings, use it
            if settings_obj.inbox_file:
                inbox_path = user_base_dir / settings_obj.inbox_file
                if inbox_path.exists():
                    logger.info(f"📝 Using configured inbox: {inbox_path}")
                    return inbox_path
                else:
                    logger.warning(f"⚠️ Configured inbox not found: {inbox_path}, searching...")
            
            # Search for existing inbox.org in user's directory tree
            if user_base_dir.exists():
                logger.info(f"🔍 Searching for inbox.org in {user_base_dir}")
                inbox_files = list(user_base_dir.rglob("inbox.org"))
                
                if inbox_files:
                    # Sort by path depth (prefer deeper paths like OrgMode/inbox.org over root inbox.org)
                    # Then alphabetically for consistency
                    inbox_files.sort(key=lambda p: (-len(p.parts), str(p)))
                    
                    # Found existing inbox.org!
                    inbox_path = inbox_files[0]
                    logger.info(f"✅ Found existing inbox.org: {inbox_path}")
                    
                    if len(inbox_files) > 1:
                        logger.warning(f"⚠️ Multiple inbox.org files found, using first: {inbox_path}")
                        logger.warning(f"   Other locations: {[str(f) for f in inbox_files[1:]]}")
                    
                    # Save discovered location to settings
                    relative_path = inbox_path.relative_to(user_base_dir)
                    from models.org_settings_models import OrgModeSettingsUpdate
                    await org_settings_service.create_or_update_settings(
                        user_id,
                        OrgModeSettingsUpdate(inbox_file=str(relative_path))
                    )
                    logger.info(f"💾 Saved inbox location to settings: {relative_path}")
                    
                    return inbox_path
            
            # No existing inbox found - create new one at root of user directory
            if not user_base_dir.exists():
                user_base_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"📁 Created user directory: {user_base_dir}")
            
            inbox_path = user_base_dir / "inbox.org"
            inbox_path.touch()
            logger.info(f"📝 Created new inbox.org at {inbox_path}")
            
            # Save new location to settings
            from models.org_settings_models import OrgModeSettingsUpdate
            await org_settings_service.create_or_update_settings(
                user_id,
                OrgModeSettingsUpdate(inbox_file="inbox.org")
            )
            logger.info(f"💾 Saved inbox location to settings: inbox.org")
            
            return inbox_path
            
        except Exception as e:
            logger.error(f"❌ Error finding/creating inbox: {e}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
            
            # If we successfully found an inbox before the error, use it!
            # The error might be in settings save, not inbox discovery
            if 'inbox_path' in locals() and inbox_path and inbox_path.exists():
                logger.warning(f"⚠️ Using found inbox despite error: {inbox_path}")
                return inbox_path
            
            # Otherwise fallback to simple path
            user_base_dir = self.upload_dir / "Users" / username
            user_base_dir.mkdir(parents=True, exist_ok=True)
            fallback_path = user_base_dir / "inbox.org"
            logger.warning(f"⚠️ Fallback: Creating inbox at {fallback_path}")
            return fallback_path
    
    async def capture_to_inbox(
        self,
        user_id: str,
        request: OrgCaptureRequest
    ) -> OrgCaptureResponse:
        """
        Capture content to user's inbox.org
        
        **BULLY!** Quick capture like Emacs org-capture!
        
        Args:
            user_id: User ID
            request: Capture request with content and template
            
        Returns:
            OrgCaptureResponse with success status and preview
        """
        try:
            # Check if this is a journal entry and journal is enabled
            if request.template_type == "journal":
                from services.org_settings_service import get_org_settings_service
                settings_service = await get_org_settings_service()
                settings = await settings_service.get_settings(user_id)
                
                if settings.journal_preferences.enabled:
                    # Route to journal service
                    from services.org_journal_service import get_org_journal_service
                    journal_service = await get_org_journal_service()
                    return await journal_service.capture_journal_entry(user_id, request)
            
            # Otherwise, continue with regular inbox capture
            # Get username for file path
            from services.database_manager.database_helpers import fetch_one
            
            row = await fetch_one("SELECT username FROM users WHERE user_id = $1", user_id)
            username = row['username'] if row else user_id
            
            # Find or create inbox.org (smart discovery!)
            inbox_path = await self._find_inbox_org(user_id, username)
            
            # Format the entry based on template (with user's timezone)
            entry = await self._format_entry(request, user_id)
            
            # Append to inbox.org
            with open(inbox_path, 'a', encoding='utf-8') as f:
                # Add newline if file isn't empty
                if inbox_path.stat().st_size > 0:
                    f.write('\n')
                f.write(entry)
                f.write('\n')
            
            # Count lines to get approximate line number
            with open(inbox_path, 'r', encoding='utf-8') as f:
                line_count = len(f.readlines())
            
            logger.info(f"✅ CAPTURE SUCCESS: Added {request.template_type} to {inbox_path.name}")
            
            return OrgCaptureResponse(
                success=True,
                message=f"Successfully captured to {inbox_path.name}",
                entry_preview=entry.strip(),
                file_path=str(inbox_path.relative_to(self.upload_dir)),
                line_number=line_count
            )
            
        except Exception as e:
            logger.error(f"❌ Capture failed: {e}")
            return OrgCaptureResponse(
                success=False,
                message=f"Failed to capture: {str(e)}",
                entry_preview=None,
                file_path=None,
                line_number=None
            )

    async def capture_to_file(
        self,
        user_id: str,
        request: OrgCaptureRequest,
        target_file: str
    ) -> OrgCaptureResponse:
        """
        Capture content to a specific org file

        **BULLY!** Capture to any org file you want!

        Args:
            user_id: User ID
            request: Capture request with content and template
            target_file: Target org file name (e.g., "github.org")

        Returns:
            OrgCaptureResponse with success status and preview
        """
        try:
            # Get username for file path
            from services.database_manager.database_helpers import fetch_one

            row = await fetch_one("SELECT username FROM users WHERE user_id = $1", user_id)
            username = row['username'] if row else user_id

            # Build target file path
            user_dir = self.upload_dir / "Users" / username
            target_path = user_dir / target_file

            # Create user directory if it doesn't exist
            user_dir.mkdir(parents=True, exist_ok=True)

            # Format the entry based on template (with user's timezone)
            entry = await self._format_entry(request, user_id)

            # Append to target file
            with open(target_path, 'a', encoding='utf-8') as f:
                # Add newline if file isn't empty
                if target_path.exists() and target_path.stat().st_size > 0:
                    f.write('\n')
                f.write(entry)
                f.write('\n')

            # Count lines to get approximate line number
            with open(target_path, 'r', encoding='utf-8') as f:
                line_count = len(f.readlines())

            logger.info(f"✅ CAPTURE SUCCESS: Added {request.template_type} to {target_file}")

            return OrgCaptureResponse(
                success=True,
                message=f"Successfully captured to {target_file}",
                entry_preview=entry.strip(),
                file_path=str(target_path.relative_to(self.upload_dir)),
                line_number=line_count
            )

        except Exception as e:
            logger.error(f"❌ Capture to file failed: {e}")
            return OrgCaptureResponse(
                success=False,
                message=f"Failed to capture to file: {str(e)}",
                entry_preview=None,
                file_path=None,
                line_number=None
            )

    async def _format_entry(self, request: OrgCaptureRequest, user_id: str) -> str:
        """
        Format entry based on template type
        
        **By George!** We format like BeOrg - clean and simple with user's timezone!
        """
        # Get user's timezone preference
        from services.settings_service import SettingsService
        from zoneinfo import ZoneInfo
        
        settings_service = SettingsService()
        user_timezone = await settings_service.get_user_timezone(user_id)
        
        # Get current time in user's timezone
        try:
            tz = ZoneInfo(user_timezone)
            now = datetime.now(tz)
        except Exception as e:
            logger.warning(f"⚠️ Invalid timezone {user_timezone}, falling back to UTC: {e}")
            now = datetime.now()
        
        timestamp = now.strftime("%Y-%m-%d %a %H:%M")
        content = expand_capture_placeholders(request.content, now)
        
        # Use Level 1 headings like BeOrg
        heading_prefix = "*"
        
        # Format based on template type
        if request.template_type == "todo":
            # TODO entry format - BeOrg style
            todo_state = "TODO"
            
            # Add priority if specified
            if request.priority:
                content = f"[#{request.priority}] {content}"
            
            # Build heading: "* TODO Content"
            heading_parts = [heading_prefix, todo_state, content]
            
            # Add tags - right-aligned with spaces (BeOrg style)
            if request.tags:
                tags_str = ":" + ":".join(request.tags) + ":"
                # Calculate padding for right-alignment (standard is column 77)
                heading_without_tags = " ".join(heading_parts)
                padding_needed = max(1, 77 - len(heading_without_tags) - len(tags_str))
                heading = heading_without_tags + (" " * padding_needed) + tags_str
            else:
                heading = " ".join(heading_parts)
            
            lines = [heading]
            
            # Add scheduled/deadline if specified
            if request.scheduled:
                lines.append(f"SCHEDULED: <{request.scheduled}>")
            if request.deadline:
                lines.append(f"DEADLINE: <{request.deadline}>")
            
            # Add simple timestamp (BeOrg style - no properties drawer by default)
            lines.append(f"[{timestamp}]")
            
            return '\n'.join(lines)
        
        elif request.template_type == "journal":
            # Journal entry format - BeOrg style
            heading_base = f"{heading_prefix} Journal Entry"
            
            # Add tags - right-aligned
            if request.tags:
                tags_str = ":" + ":".join(request.tags) + ":"
            else:
                tags_str = ":journal:"
            
            padding_needed = max(1, 77 - len(heading_base) - len(tags_str))
            heading = heading_base + (" " * padding_needed) + tags_str
            
            lines = [heading, f"[{timestamp}]", "", content]
            return '\n'.join(lines)
        
        elif request.template_type == "meeting":
            # Meeting notes format - BeOrg style
            heading_base = f"{heading_prefix} Meeting: {content}"
            
            # Add tags - right-aligned
            if request.tags:
                tags_str = ":" + ":".join(request.tags) + ":"
            else:
                tags_str = ":meeting:"
            
            padding_needed = max(1, 77 - len(heading_base) - len(tags_str))
            heading = heading_base + (" " * padding_needed) + tags_str
            
            lines = [
                heading,
                f"[{timestamp}]",
                "",
                "** Attendees",
                "",
                "** Notes",
                "",
                "** Action Items"
            ]
            return '\n'.join(lines)
        
        else:  # note (default)
            # Simple note format - BeOrg style
            heading_base = f"{heading_prefix} {content}"
            
            # Add tags - right-aligned
            if request.tags:
                tags_str = ":" + ":".join(request.tags) + ":"
                padding_needed = max(1, 77 - len(heading_base) - len(tags_str))
                heading = heading_base + (" " * padding_needed) + tags_str
            else:
                heading = heading_base
            
            lines = [heading, f"[{timestamp}]"]
            return '\n'.join(lines)


# Singleton instance
_org_capture_service: Optional[OrgCaptureService] = None


async def get_org_capture_service() -> OrgCaptureService:
    """Get or create the org capture service instance"""
    global _org_capture_service
    
    if _org_capture_service is None:
        _org_capture_service = OrgCaptureService()
        logger.info("🚀 ROOSEVELT: Org Capture Service initialized")
    
    return _org_capture_service

