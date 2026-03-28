"""
Org Inbox Tools - Utilities to locate, create, and append to the user's org-mode inbox file.
Inbox path is resolved via org-mode settings only; no filesystem globbing.

Todo CRUD (list, create, update, toggle, delete, archive) has moved to the
universal todo API (backend/services/org_todo_service.py). Only inbox path
resolution and raw append/tag-index utilities remain here.
"""

import glob
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

from config import settings
from services.org_files_service import ensure_user_org_files

logger = logging.getLogger(__name__)


class OrgInboxTools:
    """
    File utilities to work with the user's org-mode inbox file.
    Strategy:
    - Resolve inbox path from org_settings_service (inbox_file setting)
    - If not configured, use ensure_user_org_files default (Users/{username}/Org/inbox.org)
    - Save resolved path to settings for future lookups
    - No glob fallback
    """

    def __init__(self, user_id: Optional[str] = None):
        self._global_upload_dir = Path(settings.UPLOAD_DIR)
        self._user_id = user_id

    async def _base_dir(self) -> Path:
        if self._user_id:
            info = await ensure_user_org_files(self._user_id)
            return Path(info["org_base_dir"])  # type: ignore[arg-type]
        return self._global_upload_dir

    async def _find_inbox_path(self) -> Optional[Path]:
        if not self._user_id:
            return None
        try:
            from services.org_settings_service import get_org_settings_service
            from models.org_settings_models import OrgModeSettingsUpdate

            org_settings_service = await get_org_settings_service()
            settings_obj = await org_settings_service.get_settings(self._user_id)
            base = await self._base_dir()
            user_base_dir = base.parent

            if settings_obj.inbox_file:
                inbox_path = user_base_dir / settings_obj.inbox_file
                if inbox_path.exists():
                    return inbox_path
                logger.warning("Configured inbox not found: %s", inbox_path)

            info = await ensure_user_org_files(self._user_id)
            inbox_path = Path(info["inbox_path"])
            relative_path = inbox_path.relative_to(user_base_dir)
            await org_settings_service.create_or_update_settings(
                self._user_id,
                OrgModeSettingsUpdate(inbox_file=str(relative_path)),
            )
            return inbox_path
        except Exception as e:
            logger.error("Failed to resolve inbox path: %s", e)
            return None

    async def ensure_inbox(self) -> Path:
        path = await self._find_inbox_path()
        if path:
            return path
        if self._user_id:
            info = await ensure_user_org_files(self._user_id)
            return Path(info["inbox_path"])
        base = await self._base_dir()
        base.mkdir(parents=True, exist_ok=True)
        path = base / "inbox.org"
        if not path.exists():
            path.write_text("* Inbox\n", encoding="utf-8")
        return path

    async def get_inbox_path(self) -> Optional[str]:
        path = await self._find_inbox_path()
        return str(path) if path else None

    async def append_text(self, content: str) -> Dict[str, Any]:
        path = await self.ensure_inbox()
        try:
            to_write = content if content.endswith("\n") else content + "\n"
            existing = path.read_text(encoding="utf-8")
            if existing and not existing.endswith("\n"):
                to_write = "\n" + to_write
            with path.open("a", encoding="utf-8") as f:
                f.write(to_write)
            return {"path": str(path), "appended_chars": len(to_write)}
        except Exception as e:
            logger.error(f"Failed to append text: {e}")
            return {"path": str(path), "error": str(e)}

    async def append_block(self, block: str) -> Dict[str, Any]:
        """Append a multi-line Org block and return inserted line range."""
        path = await self.ensure_inbox()
        try:
            block_text = block.strip("\n") + "\n"
            existing_lines = path.read_text(encoding="utf-8").splitlines()
            start_idx = len(existing_lines)
            with path.open("a", encoding="utf-8") as f:
                f.write(block_text)
            added_count = len(block_text.splitlines())
            end_idx = start_idx + added_count - 1
            return {
                "path": str(path),
                "line_start_index": start_idx,
                "line_end_index": end_idx,
                "written_lines": added_count
            }
        except Exception as e:
            logger.error(f"Failed to append block: {e}")
            return {"path": str(path), "error": str(e)}

    def _extract_tags_from_headline(self, line: str) -> List[str]:
        import re
        m = re.search(r"\s+:([A-Za-z0-9_@:+-]+):\s*$", line)
        if not m:
            return []
        tags = [t for t in m.group(1).split(":") if t]
        return tags

    async def index_tags(self) -> Dict[str, int]:
        """Scan .org files under uploads to build a tag frequency index."""
        base = await self._base_dir()
        counts: Dict[str, int] = {}
        try:
            if not base.exists():
                return counts
            for m in glob.glob(str(base / "**" / "*.org"), recursive=True):
                if "/.versions/" in m or "\\.versions\\" in m:
                    continue
                p = Path(m)
                try:
                    for line in p.read_text(encoding="utf-8").splitlines():
                        if line.lstrip().startswith("*"):
                            for t in self._extract_tags_from_headline(line):
                                counts[t] = counts.get(t, 0) + 1
                except Exception:
                    continue
            return counts
        except Exception as e:
            logger.error(f"Failed to index tags: {e}")
            return counts

    # ── Internal methods used by gRPC handlers (not exposed to LLM agents) ──

    async def _created_timestamp_line(self) -> str:
        """Format CREATED timestamp in user's timezone (org-mode style)."""
        if not self._user_id:
            return ""
        try:
            from datetime import datetime
            from zoneinfo import ZoneInfo
            from services.settings_service import settings_service
            user_timezone = await settings_service.get_user_timezone(self._user_id)
            tz = ZoneInfo(user_timezone)
            now = datetime.now(tz)
            ts = now.strftime("%Y-%m-%d %a %H:%M")
            return f"CREATED: [{ts}]\n"
        except Exception as e:
            logger.debug("Could not add CREATED timestamp: %s", e)
            return ""

    def _set_tags_on_headline(self, line: str, tags: List[str]) -> str:
        import re
        base = re.sub(r"\s+:([A-Za-z0-9_@:+-]+):\s*$", "", line)
        if tags:
            tag_suffix = ":" + ":".join(sorted(set(t.strip() for t in tags if t.strip()))) + ":"
            return f"{base} {tag_suffix}"
        return base

    async def list_items(self) -> Dict[str, Any]:
        """List task-like items from inbox.org. Used internally by gRPC AddOrgInboxItem handler."""
        path = await self.ensure_inbox()
        try:
            content = path.read_text(encoding="utf-8")
            lines = content.splitlines()
            items: List[Dict[str, Any]] = []
            for idx, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("- [") or stripped.startswith("- [x"):
                    done = "[x]" in stripped
                    text = stripped.split("]", 1)[-1].strip()
                    items.append({"line_index": idx, "done": done, "text": text})
                elif stripped.startswith("* TODO ") or stripped.startswith("* DONE ") or stripped.startswith("** TODO ") or stripped.startswith("** DONE "):
                    done = stripped.startswith("* DONE ") or stripped.startswith("** DONE ")
                    text = stripped.split(" ", 2)[-1]
                    items.append({"line_index": idx, "done": done, "text": text})
            return {"path": str(path), "items": items}
        except Exception as e:
            logger.error(f"Failed to list items: {e}")
            return {"path": str(path), "items": [], "error": str(e)}

    async def add_item(self, text: str, kind: str = "checkbox", body: Optional[str] = None) -> Dict[str, Any]:
        """Add a checkbox or TODO heading to inbox. Used internally by gRPC AddOrgInboxItem handler."""
        path = await self.ensure_inbox()
        if kind == "todo":
            line = f"* TODO {text}\n"
        else:
            line = f"- [ ] {text}\n"
        created_line = await self._created_timestamp_line()
        try:
            with path.open("a", encoding="utf-8") as f:
                existing = path.read_text(encoding="utf-8")
                if existing and not existing.endswith("\n"):
                    f.write("\n")
                f.write(line)
                if created_line:
                    f.write(created_line)
                if kind == "todo" and body and body.strip():
                    body_text = body.strip()
                    f.write(body_text + "\n" if not body_text.endswith("\n") else body_text)
            lines = path.read_text(encoding="utf-8").splitlines()
            added_index = len(lines) - 1
            if created_line:
                added_index = len(lines) - 2
            return {"path": str(path), "added": line.strip(), "line_index": added_index}
        except Exception as e:
            logger.error("Failed to add item: %s", e)
            return {"path": str(path), "error": str(e)}

    async def toggle_done(self, line_index: int) -> Dict[str, Any]:
        """Toggle done state of a task line. Used internally by gRPC ToggleOrgInboxItem handler."""
        path = await self.ensure_inbox()
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            if line_index < 0 or line_index >= len(lines):
                return {"path": str(path), "error": "line_index out of range"}
            line = lines[line_index]
            if "- [ ]" in line:
                lines[line_index] = line.replace("- [ ]", "- [x]", 1)
            elif "- [x]" in line:
                lines[line_index] = line.replace("- [x]", "- [ ]", 1)
            elif line.strip().startswith("* TODO "):
                lines[line_index] = line.replace("* TODO ", "* DONE ", 1)
            elif line.strip().startswith("** TODO "):
                lines[line_index] = line.replace("** TODO ", "** DONE ", 1)
            elif line.strip().startswith("* DONE "):
                lines[line_index] = line.replace("* DONE ", "* TODO ", 1)
            elif line.strip().startswith("** DONE "):
                lines[line_index] = line.replace("** DONE ", "** TODO ", 1)
            else:
                return {"path": str(path), "error": "line is not a task"}
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return {"path": str(path), "updated_index": line_index, "new_line": lines[line_index]}
        except Exception as e:
            logger.error(f"Failed to toggle item: {e}")
            return {"path": str(path), "error": str(e)}

    async def update_line(self, line_index: int, new_text: str) -> Dict[str, Any]:
        """Update the text of a task line. Used internally by gRPC UpdateOrgInboxItem handler."""
        path = await self.ensure_inbox()
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            if line_index < 0 or line_index >= len(lines):
                return {"path": str(path), "error": "line_index out of range"}
            prefix = ""
            stripped = lines[line_index].strip()
            if stripped.startswith("- [ ]"):
                prefix = "- [ ] "
            elif stripped.startswith("- [x]"):
                prefix = "- [x] "
            elif stripped.startswith("* TODO "):
                prefix = "* TODO "
            elif stripped.startswith("* DONE "):
                prefix = "* DONE "
            elif stripped.startswith("** TODO "):
                prefix = "** TODO "
            elif stripped.startswith("** DONE "):
                prefix = "** DONE "
            lines[line_index] = f"{prefix}{new_text}".rstrip()
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return {"path": str(path), "updated_index": line_index, "new_line": lines[line_index]}
        except Exception as e:
            logger.error(f"Failed to update line: {e}")
            return {"path": str(path), "error": str(e)}

    async def apply_tags(self, line_index: int, tags: List[str]) -> Dict[str, Any]:
        """Apply tags to a specific line. Used internally by gRPC ApplyOrgInboxTags handler."""
        path = await self.ensure_inbox()
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            if line_index < 0 or line_index >= len(lines):
                return {"path": str(path), "error": "line_index out of range"}
            line = lines[line_index]
            if line.lstrip().startswith("*"):
                new_line = self._set_tags_on_headline(line, tags)
                lines[line_index] = new_line
            else:
                tag_suffix = "  :" + ":".join(sorted(set(t.strip() for t in tags if t.strip()))) + ":"
                lines[line_index] = (line.rstrip() + tag_suffix)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return {"path": str(path), "updated_index": line_index, "new_line": lines[line_index]}
        except Exception as e:
            logger.error(f"Failed to apply tags: {e}")
            return {"path": str(path), "error": str(e)}

    async def set_schedule_and_repeater(self, line_index: int, scheduled: Optional[str], repeater: Optional[str]) -> Dict[str, Any]:
        """Set SCHEDULED and optional repeater. Used internally by gRPC SetOrgInboxSchedule handler."""
        path = await self.ensure_inbox()
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            if line_index < 0 or line_index >= len(lines):
                return {"path": str(path), "error": "line_index out of range"}
            if not lines[line_index].lstrip().startswith("*"):
                return {"path": str(path), "error": "line is not a headline"}
            insert_idx = line_index + 1
            sched_idx = None
            for idx in range(line_index + 1, min(line_index + 5, len(lines))):
                if lines[idx].strip().startswith("SCHEDULED:"):
                    sched_idx = idx
                    break
                if lines[idx].lstrip().startswith("*"):
                    break
            ts = scheduled or ""
            if repeater and ts:
                import re
                ts = re.sub(r"^<([^>]+)>$", lambda m: f"<{m.group(1)} {repeater}>", ts)
            content = f"SCHEDULED: {ts}".rstrip()
            if sched_idx is not None:
                lines[sched_idx] = content
            else:
                lines.insert(insert_idx, content)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return {"path": str(path), "updated_index": line_index, "scheduled_index": (sched_idx or insert_idx), "scheduled_line": content}
        except Exception as e:
            logger.error(f"Failed to set schedule/repeater: {e}")
            return {"path": str(path), "error": str(e)}


# Simple async wrappers for registry and gRPC handlers
_org_tools_instances: Dict[str, OrgInboxTools] = {}


async def _get_instance(user_id: Optional[str] = None) -> OrgInboxTools:
    global _org_tools_instances
    key = user_id or "__global__"
    if key not in _org_tools_instances:
        _org_tools_instances[key] = OrgInboxTools(user_id=user_id)
    return _org_tools_instances[key]


async def org_inbox_path(user_id: Optional[str] = None) -> str:
    inst = await _get_instance(user_id)
    path = await inst.get_inbox_path()
    return path or ""


async def org_inbox_append_text(content: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    inst = await _get_instance(user_id)
    return await inst.append_text(content=content)


async def org_inbox_append_block(block: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    inst = await _get_instance(user_id)
    return await inst.append_block(block=block)


async def org_inbox_index_tags(user_id: Optional[str] = None) -> Dict[str, int]:
    inst = await _get_instance(user_id)
    return await inst.index_tags()


# ── Internal wrappers used by gRPC handlers (not registered in tool registry) ──

async def org_inbox_list_items(user_id: Optional[str] = None) -> Dict[str, Any]:
    inst = await _get_instance(user_id)
    return await inst.list_items()


async def org_inbox_add_item(text: str, kind: str = "checkbox", user_id: Optional[str] = None) -> Dict[str, Any]:
    inst = await _get_instance(user_id)
    return await inst.add_item(text=text, kind=kind)


async def org_inbox_toggle_done(line_index: int, user_id: Optional[str] = None) -> Dict[str, Any]:
    inst = await _get_instance(user_id)
    return await inst.toggle_done(line_index=line_index)


async def org_inbox_update_line(line_index: int, new_text: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    inst = await _get_instance(user_id)
    return await inst.update_line(line_index=line_index, new_text=new_text)


async def org_inbox_apply_tags(line_index: int, tags: List[str], user_id: Optional[str] = None) -> Dict[str, Any]:
    inst = await _get_instance(user_id)
    return await inst.apply_tags(line_index=line_index, tags=tags)


async def org_inbox_set_schedule_and_repeater(line_index: int, scheduled: Optional[str], repeater: Optional[str], user_id: Optional[str] = None) -> Dict[str, Any]:
    inst = await _get_instance(user_id)
    return await inst.set_schedule_and_repeater(line_index=line_index, scheduled=scheduled, repeater=repeater)


async def org_inbox_archive_done(user_id: str) -> Dict[str, Any]:
    """
    Archive closed entries from inbox to global archive. Moves full entries (heading + subtree + properties).
    Only archives done states (DONE, CANCELLED, etc.) and - [x] checkboxes.
    """
    try:
        from services.org_todo_service import _resolve_done_states, OrgTodoService
        done_states = await _resolve_done_states(user_id)
        inst = await _get_instance(user_id)
        inbox_path = await inst.ensure_inbox()
        lines = inbox_path.read_text(encoding="utf-8").splitlines()
        entries = _parse_inbox_entries(lines)
        to_archive: List[List[str]] = []
        remaining: List[str] = []
        for entry_lines, state in entries:
            if state == "checkbox_done":
                to_archive.append(entry_lines)
            elif state and state.upper() in done_states:
                to_archive.append(entry_lines)
            else:
                remaining.extend(entry_lines)
        if not to_archive:
            return {
                "path": str(inbox_path),
                "archived_to": "",
                "archived_count": 0,
            }
        from services.org_files_service import get_user_archive_path
        archive_path = Path(await get_user_archive_path(user_id))
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        ts = await _format_archive_ts(user_id)
        source_label = str(inbox_path.name)
        with archive_path.open("a", encoding="utf-8") as f:
            for entry_lines in to_archive:
                block = OrgTodoService._upsert_properties_entry(entry_lines, "ARCHIVE_TIME", f"[{ts}]")
                block = OrgTodoService._upsert_properties_entry(block, "ARCHIVE_FILE", source_label)
                for line in block:
                    f.write(line + "\n" if not line.endswith("\n") else line)
        inbox_path.write_text("\n".join(remaining) + ("\n" if remaining else ""), encoding="utf-8")
        return {
            "path": str(inbox_path),
            "archived_to": str(archive_path),
            "archived_count": len(to_archive),
        }
    except Exception as e:
        logger.error("Failed to archive inbox items for %s: %s", user_id, e)
        return {"error": str(e)}


async def _format_archive_ts(user_id: str) -> str:
    """Format current time in user timezone for ARCHIVE_TIME property."""
    try:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        from services.settings_service import settings_service
        tz = ZoneInfo(await settings_service.get_user_timezone(user_id))
        return datetime.now(tz).strftime("%Y-%m-%d %a %H:%M")
    except Exception:
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %a %H:%M")


def _parse_inbox_entries(lines: List[str]) -> List[tuple]:
    """
    Parse lines into (entry_lines, state) tuples. Each entry is either a full headline subtree
    or a single checkbox line. state = todo state (e.g. DONE, CANCELLED) for headlines,
    'checkbox_done' / 'checkbox_open' for checkboxes, None for other (e.g. * Inbox).
    """
    result: List[tuple] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if re.match(r"^\*+\s+", stripped):
            level = 0
            for c in stripped:
                if c == "*":
                    level += 1
                else:
                    break
            j = i + 1
            while j < len(lines):
                next_line = lines[j]
                next_stripped = next_line.strip()
                if re.match(r"^\*+\s+", next_stripped):
                    next_level = 0
                    for c in next_stripped:
                        if c == "*":
                            next_level += 1
                        else:
                            break
                    if next_level <= level:
                        break
                j += 1
            entry_lines = lines[i:j]
            state = None
            m = re.match(r"^\*+\s+(\S+)\s", stripped)
            if m:
                state = m.group(1).upper()
            result.append((entry_lines, state))
            i = j
        elif stripped.startswith("- [x]"):
            result.append(([line], "checkbox_done"))
            i += 1
        elif stripped.startswith("- [ ]"):
            result.append(([line], "checkbox_open"))
            i += 1
        else:
            result.append(([line], None))
            i += 1
    return result
