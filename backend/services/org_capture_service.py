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
        
        Resolve or create the user's inbox document.
        
        Strategy:
        1. Check user's org-mode settings for configured inbox_file
        2. If not configured, search user's entire directory tree for inbox.org
        3. If multiple found, use the first one
        4. If none found, create at Users/{username}/inbox.org
        5. Save discovered/created location to settings for future use
        """
        try:
            from services import ds_upload_library_fs as dsf
            from services.org_settings_service import get_org_settings_service

            org_settings_service = await get_org_settings_service()
            settings_obj = await org_settings_service.get_settings(user_id)

            user_base_dir = self.upload_dir / "Users" / username

            if settings_obj.inbox_file:
                inbox_path = user_base_dir / settings_obj.inbox_file
                if await dsf.exists(user_id, inbox_path):
                    logger.info(f"📝 Using configured inbox: {inbox_path}")
                    return inbox_path
                logger.warning(f"⚠️ Configured inbox not found: {inbox_path}, searching...")

            inbox_files: list[Path] = []
            if await dsf.is_dir(user_id, user_base_dir):
                logger.info(f"🔍 Searching for inbox.org under {user_base_dir} (document-service)")
                for p in await dsf.walk_org_files(user_id, username, include_archives=True):
                    if p.name.lower() != "inbox.org":
                        continue
                    if "/.versions/" in str(p).replace("\\", "/"):
                        continue
                    inbox_files.append(p)

            if inbox_files:
                inbox_files.sort(key=lambda p: (-len(p.parts), str(p)))
                inbox_path = inbox_files[0]
                logger.info(f"✅ Found existing inbox.org: {inbox_path}")
                if len(inbox_files) > 1:
                    logger.warning(f"⚠️ Multiple inbox.org files found, using first: {inbox_path}")
                    logger.warning(f"   Other locations: {[str(f) for f in inbox_files[1:]]}")

                relative_path = inbox_path.relative_to(user_base_dir)
                from models.org_settings_models import OrgModeSettingsUpdate

                await org_settings_service.create_or_update_settings(
                    user_id,
                    OrgModeSettingsUpdate(inbox_file=str(relative_path)),
                )
                logger.info(f"💾 Saved inbox location to settings: {relative_path}")
                return inbox_path

            inbox_path = user_base_dir / "inbox.org"
            if not await dsf.exists(user_id, inbox_path):
                await dsf.write_text(user_id, inbox_path, "")
            logger.info(f"📝 Ensured inbox.org at {inbox_path}")

            from models.org_settings_models import OrgModeSettingsUpdate

            await org_settings_service.create_or_update_settings(
                user_id,
                OrgModeSettingsUpdate(inbox_file="inbox.org"),
            )
            logger.info("💾 Saved inbox location to settings: inbox.org")
            return inbox_path

        except Exception as e:
            logger.error(f"❌ Error finding/creating inbox: {e}")
            import traceback

            logger.error(f"   Traceback: {traceback.format_exc()}")

            from services import ds_upload_library_fs as dsf

            if "inbox_path" in locals() and inbox_path and await dsf.exists(user_id, inbox_path):
                logger.warning(f"⚠️ Using found inbox despite error: {inbox_path}")
                return inbox_path

            user_base_dir = self.upload_dir / "Users" / username
            fallback_path = user_base_dir / "inbox.org"
            try:
                if not await dsf.exists(user_id, fallback_path):
                    await dsf.write_text(user_id, fallback_path, "")
            except Exception:
                pass
            logger.warning(f"⚠️ Fallback: inbox at {fallback_path}")
            return fallback_path

    async def _document_id_for_path(
        self, file_path: Path, user_id: str, username: str
    ) -> Optional[str]:
        """Resolve document_id for a file path (inbox or target file)."""
        try:
            from services.folder_service import FolderService
            folder_service = FolderService()
            user_base_dir = self.upload_dir / "Users" / username
            parent_dir = file_path.parent
            folder_id = None
            if parent_dir != user_base_dir:
                folder_id = await folder_service.get_folder_id_by_physical_path(
                    parent_dir, user_id
                )
            doc = await folder_service.document_repository.find_by_filename_and_context(
                file_path.name, user_id, "user", folder_id
            )
            return doc.document_id if doc else None
        except Exception:
            return None

    async def capture_to_inbox(
        self,
        user_id: str,
        request: OrgCaptureRequest
    ) -> OrgCaptureResponse:
        """
        Capture content to user's inbox.org
        
        Append a capture entry in org-capture style.
        
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

            # Snapshot before write for versioning (if document is in DB)
            doc_id = await self._document_id_for_path(inbox_path, user_id, username)
            if doc_id:
                try:
                    from services.document_version_service import snapshot_before_write
                    await snapshot_before_write(
                        doc_id, user_id, "quick_capture_inbox", None, None
                    )
                except Exception as verr:
                    logger.warning(
                        "Version snapshot before inbox capture failed (non-fatal): %s",
                        verr,
                    )

            from services import ds_upload_library_fs as dsf

            cur = ""
            try:
                cur = await dsf.read_text(user_id, inbox_path)
            except FileNotFoundError:
                pass
            suffix = (("\n" if cur.strip() else "") + entry + "\n")
            await dsf.append_text(user_id, inbox_path, suffix)

            line_count = len((await dsf.read_text(user_id, inbox_path)).splitlines())
            
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

        Append a capture entry to the specified org file.

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

            from services import ds_upload_library_fs as dsf

            # Format the entry based on template (with user's timezone)
            entry = await self._format_entry(request, user_id)

            # Snapshot before write for versioning (if document is in DB)
            doc_id = await self._document_id_for_path(target_path, user_id, username)
            if doc_id:
                try:
                    from services.document_version_service import snapshot_before_write
                    await snapshot_before_write(
                        doc_id, user_id, "quick_capture_file", None, None
                    )
                except Exception as verr:
                    logger.warning(
                        "Version snapshot before capture-to-file failed (non-fatal): %s",
                        verr,
                    )

            cur = ""
            try:
                cur = await dsf.read_text(user_id, target_path)
            except FileNotFoundError:
                pass
            suffix = (("\n" if cur.strip() else "") + entry + "\n")
            await dsf.append_text(user_id, target_path, suffix)

            line_count = len((await dsf.read_text(user_id, target_path)).splitlines())

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
        
        Format timestamps using the user's timezone (BeOrg-style simplicity).
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
            # Entry format aligns with org-capture; BeOrg-style refinements may follow.
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
        logger.info("Org Capture Service initialized")
    
    return _org_capture_service

