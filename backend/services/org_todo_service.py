"""
Org Todo Service - Single service for all todo CRUD operations across any org file.
Org files on disk are the source of truth. Supports list, create, update, toggle, delete, archive.
"""

import logging
import html
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import settings

logger = logging.getLogger(__name__)

DEFAULT_TODO_STATES = ["TODO", "NEXT", "STARTED", "WAITING", "HOLD"]
DONE_STATES = {"DONE", "CANCELED", "CANCELLED"}

# Trailing org-mode tag(s) at end of headline text (e.g. " :@home:" or " :a::b:")
_RE_TRAILING_ORG_TAGS = re.compile(r"\s+:([A-Za-z0-9_@:+-]+):\s*$")

# Inactive timestamp on its own line (capture / creation time under a heading)
_RE_CAPTURE_TS_LINE = re.compile(r"^\s*\[(\d{4}-\d{2}-\d{2}[^\]]*)\]\s*$")

# Emacs org "Note taken on" blocks: header line then one or more indented continuation lines
_RE_NOTE_TAKEN = re.compile(
    r"(?:^|\n)\s*-\s+Note taken on\s+\[([^\]]+)\]\s*(?:\\\\|\\)?\s*\r?\n"
    r"((?:[ \t]+[^\n]+\r?\n?)+)",
    re.MULTILINE,
)

# PROPERTIES / LOGBOOK drawers (strip from body before preview)
_RE_ORG_DRAWER = re.compile(
    r":(?:PROPERTIES|LOGBOOK):\s*\n.*?:END:\s*",
    re.IGNORECASE | re.DOTALL,
)

# Planning lines often duplicated in body text
_RE_PLANNING_LINE = re.compile(
    r"^\s*(?:SCHEDULED|DEADLINE|CLOSED):\s*.*$",
    re.IGNORECASE | re.MULTILINE,
)


def _parse_todo_body(body: str, max_preview: int = 120) -> Dict[str, Any]:
    """
    Split org heading body into creation_timestamp, structured notes, and a prose-only preview.

    Avoids showing inactive capture timestamps and org note boilerplate as the only preview line.
    """
    if not body or not str(body).strip():
        return {
            "creation_timestamp": None,
            "notes": [],
            "body_preview": "",
        }

    text = str(body).strip()
    text = _RE_ORG_DRAWER.sub("\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    creation_timestamp: Optional[str] = None
    lines = text.split("\n")
    if lines:
        first = lines[0].strip()
        m0 = _RE_CAPTURE_TS_LINE.match(first)
        if m0:
            creation_timestamp = m0.group(1).strip()
            text = "\n".join(lines[1:]).strip()

    notes: List[Dict[str, str]] = []
    while True:
        m = _RE_NOTE_TAKEN.search(text)
        if not m:
            break
        note_body = " ".join(
            ln.strip()
            for ln in (m.group(2) or "").splitlines()
            if ln.strip()
        )
        notes.append(
            {
                "timestamp": (m.group(1) or "").strip(),
                "text": re.sub(r"\s+", " ", note_body).strip(),
            }
        )
        text = (text[: m.start()] + text[m.end() :]).strip()
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

    text = _RE_PLANNING_LINE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    preview = ""
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if _RE_CAPTURE_TS_LINE.match(line):
            continue
        if re.match(r"^\s*-\s+Note taken on\s+\[", line):
            continue
        preview = line
        break

    if len(preview) > max_preview:
        preview = preview[: max_preview - 3] + "..."

    return {
        "creation_timestamp": creation_timestamp,
        "notes": notes,
        "body_preview": preview,
    }


def _strip_trailing_org_tags_from_title(title: str) -> str:
    """Remove any trailing org-style :tag: from title so tags are only added via the tags parameter."""
    if not title or not title.strip():
        return title
    s = title.strip()
    while True:
        m = _RE_TRAILING_ORG_TAGS.search(s)
        if not m:
            break
        s = s[: m.start()].rstrip()
    return s


async def _resolve_done_states(user_id: str) -> set:
    """Resolve user's configured done states; always include DONE, CANCELED, CANCELLED so archiving moves all closed items."""
    try:
        from services.org_settings_service import get_org_settings_service
        svc = await get_org_settings_service()
        states = await svc.get_todo_states(user_id)
        result = {s.upper() for s in states.get("done", [])}
        result |= DONE_STATES
        return result
    except Exception:
        return DONE_STATES


async def _resolve_active_states(user_id: str) -> set:
    """Resolve user's configured active states from org settings; fallback to DEFAULT_TODO_STATES."""
    try:
        from services.org_settings_service import get_org_settings_service
        svc = await get_org_settings_service()
        states = await svc.get_todo_states(user_id)
        result = {s.upper() for s in states.get("active", [])}
        return result if result else set(DEFAULT_TODO_STATES)
    except Exception:
        return set(DEFAULT_TODO_STATES)


_org_todo_service: Optional["OrgTodoService"] = None


async def get_org_todo_service() -> "OrgTodoService":
    global _org_todo_service
    if _org_todo_service is None:
        _org_todo_service = OrgTodoService()
    return _org_todo_service


class OrgTodoService:
    """
    Universal todo operations on org files. List/create/update/toggle/delete/archive
    work on any org file; scope can be "all", "inbox", or a specific file path.
    Todo identifiers use file_path + line_number (0-based index).
    Toggle/update support org headings at any level (*, **, ***, ****, ...) and
    markdown checkboxes (- [ ] / - [x]).
    """

    def __init__(self) -> None:
        self._upload_dir = Path(settings.UPLOAD_DIR)

    async def _org_read(self, user_id: str, p: Path, *, utf8_sig: bool = False) -> str:
        from services import ds_upload_library_fs as dsf

        text = await dsf.read_text(user_id, p)
        if utf8_sig and text.startswith("\ufeff"):
            text = text[1:]
        return text

    async def _org_write(self, user_id: str, p: Path, content: str) -> None:
        from services import ds_upload_library_fs as dsf

        await dsf.write_text(user_id, p, content)

    async def _org_append(self, user_id: str, p: Path, suffix: str) -> None:
        from services import ds_upload_library_fs as dsf

        await dsf.append_text(user_id, p, suffix)

    async def _org_exists(self, user_id: str, p: Path) -> bool:
        from services import ds_upload_library_fs as dsf

        return await dsf.exists(user_id, p)

    async def _get_search_service(self):
        from services.org_search_service import get_org_search_service
        return await get_org_search_service()

    async def _get_inbox_path(self, user_id: str) -> str:
        from services.langgraph_tools.org_inbox_tools import _get_instance
        inst = await _get_instance(user_id)
        path = await inst.ensure_inbox()
        return str(path)

    async def _resolve_file_path(self, user_id: str, file_path: Optional[str]) -> Optional[Path]:
        if not file_path or not file_path.strip():
            return None
        path = Path(file_path.strip())
        if path.is_absolute():
            if str(path).startswith(str(self._upload_dir)):
                return path
            return None
        try:
            from services.database_manager.database_helpers import fetch_one
            row = await fetch_one("SELECT username FROM users WHERE user_id = $1", user_id)
        except Exception:
            row = None
        username = row["username"] if row else user_id
        base = self._upload_dir / "Users" / username
        resolved = (base / path).resolve()
        if not str(resolved).startswith(str(base)):
            return None
        return resolved

    async def _org_timestamp(self, user_id: str) -> str:
        """Format current time in user's timezone for org-mode timestamps (e.g. 2026-02-22 Sun 14:30)."""
        try:
            from datetime import datetime
            from zoneinfo import ZoneInfo
            from services.settings_service import settings_service
            user_timezone = await settings_service.get_user_timezone(user_id)
            tz = ZoneInfo(user_timezone)
            now = datetime.now(tz)
            return now.strftime("%Y-%m-%d %a %H:%M")
        except Exception as e:
            logger.debug("Could not get user timezone for org timestamp: %s", e)
            from datetime import datetime
            return datetime.now().strftime("%Y-%m-%d %a %H:%M")

    def _parse_closed_date(self, closed_str: Optional[str]) -> Optional[datetime]:
        """Parse org CLOSED timestamp string to datetime (UTC). Returns None if unparseable."""
        if not closed_str or not isinstance(closed_str, str):
            return None
        s = closed_str.strip().strip("[]")
        for fmt, size in (
            ("%Y-%m-%d %a %H:%M", 21),
            ("%Y-%m-%d %H:%M:%S", 19),
            ("%Y-%m-%d %H:%M", 16),
            ("%Y-%m-%d", 10),
        ):
            try:
                part = s[: size] if len(s) >= size else s
                dt = datetime.strptime(part, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except (ValueError, TypeError):
                continue
        return None

    async def list_todos(
        self,
        user_id: str,
        scope: str = "all",
        states: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        query: str = "",
        limit: int = 100,  # use <= 0 for no cap
        include_archives: bool = False,
        include_body: bool = False,
        closed_since_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        try:
            search = await self._get_search_service()
            if closed_since_days is not None and closed_since_days >= 0:
                todo_states = list(DONE_STATES)  # fetch done items only, then filter by closed date
            else:
                todo_states = states if states is not None else DEFAULT_TODO_STATES
            document_id_map = await search._get_document_id_map(user_id)
            if scope == "inbox":
                inbox_path = await self._get_inbox_path(user_id)
                org_files = [Path(inbox_path)]
            elif scope == "all":
                org_files = await search._find_user_org_files(user_id, include_archives=include_archives)
            else:
                resolved = await self._resolve_file_path(user_id, scope)
                if not resolved or not await self._org_exists(user_id, resolved):
                    return {"success": False, "error": f"File not found: {scope}", "results": [], "count": 0}
                org_files = [resolved]

            if not org_files:
                return {"success": True, "results": [], "count": 0, "files_searched": 0}

            all_results: List[Dict[str, Any]] = []
            for file_path in org_files:
                file_results = await search._search_org_file(
                    user_id=user_id,
                    file_path=file_path,
                    query=query,
                    tags=tags,
                    todo_states=todo_states,
                    include_content=True,
                )
                filename_stem = file_path.stem
                document_id = document_id_map.get(filename_stem)
                for r in file_results:
                    r["document_id"] = document_id
                    line_num = r.get("line_number", 0)
                    r["line_number"] = line_num - 1 if isinstance(line_num, int) and line_num > 0 else 0
                    body_raw = (r.get("body") or "").strip()
                    parsed = _parse_todo_body(body_raw)
                    r["body_preview"] = parsed["body_preview"]
                    r["creation_timestamp"] = parsed["creation_timestamp"]
                    r["notes"] = parsed["notes"]
                    if not include_body:
                        r.pop("body", None)
                all_results.extend(file_results)

            if closed_since_days is not None and closed_since_days >= 0:
                cutoff = datetime.now(timezone.utc) - timedelta(days=closed_since_days)
                all_results = [
                    r
                    for r in all_results
                    if (r.get("todo_state") or "").upper() in DONE_STATES
                    and self._parse_closed_date(r.get("closed")) is not None
                    and self._parse_closed_date(r.get("closed")) >= cutoff
                ]
            all_results.sort(key=lambda r: (-int(r.get("heading_match", False)), -r.get("match_count", 0)))
            # limit <= 0 means return all matches (agents and tools should see the full set).
            limited = all_results if limit <= 0 else all_results[:limit]
            return {
                "success": True,
                "results": limited,
                "count": len(limited),
                "total_matches": len(all_results),
                "files_searched": len(org_files),
                "filters": {"tags": tags, "todo_states": todo_states},
            }
        except Exception as e:
            logger.exception("list_todos failed")
            return {"success": False, "error": str(e), "results": [], "count": 0}

    async def create_todo(
        self,
        user_id: str,
        text: str,
        file_path: Optional[str] = None,
        state: str = "TODO",
        tags: Optional[List[str]] = None,
        scheduled: Optional[str] = None,
        deadline: Optional[str] = None,
        priority: Optional[str] = None,
        body: Optional[str] = None,
        heading_level: Optional[int] = None,
        insert_after_line_number: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not (text or "").strip():
            return {"success": False, "error": "text is required"}
        level = max(1, min(6, heading_level or 1))
        try:
            if file_path is None or (isinstance(file_path, str) and file_path.strip() == ""):
                from services.langgraph_tools.org_inbox_tools import _get_instance
                inst = await _get_instance(user_id)
                resolved = Path(await inst.ensure_inbox())
                header = None
            else:
                resolved = await self._resolve_file_path(user_id, file_path)
                if not resolved or not await self._org_exists(user_id, resolved):
                    return {"success": False, "error": f"File not found: {file_path}"}
                from utils.org_header_parser import parse_org_file_header, filetags_to_list
                file_content_for_header = await self._org_read(user_id, resolved, utf8_sig=True)
                header = parse_org_file_header(file_content_for_header)

            ts = await self._org_timestamp(user_id)
            effective_tags: Optional[List[str]] = tags if tags else None
            if effective_tags is None and header and header.filetags:
                effective_tags = filetags_to_list(header.filetags)
            tag_suffix = ""
            if effective_tags:
                tag_suffix = "  :" + ":".join(sorted(set(t.strip() for t in effective_tags if t.strip()))) + ":"
            effective_priority: Optional[str] = priority if (priority and str(priority).strip()) else None
            if effective_priority is None and header and header.priority:
                effective_priority = header.priority
            priority_str = ""
            if effective_priority and str(effective_priority).strip().upper() in ("A", "B", "C"):
                priority_str = f"[#{str(effective_priority).strip().upper()}] "
            stars = "*" * level
            # Strip any trailing org tags from title so we don't duplicate when agent puts tags in text and in tags param
            title_part = _strip_trailing_org_tags_from_title(text)
            line = f"{stars} {state.upper()} {priority_str}{title_part}{tag_suffix}\n"
            props_block = [":PROPERTIES:", f":CREATED:  [{ts}]", ":END:"]
            planning_lines: List[str] = []
            if scheduled and scheduled.strip():
                # Sometimes upstream content is HTML-escaped (e.g. "&lt;2026-04-25 Sat&gt;").
                # Org-mode requires literal angle brackets for active timestamps.
                sched = html.unescape(scheduled.strip())
                planning_lines.append(f"SCHEDULED: {sched}")
            if deadline and deadline.strip():
                dl = html.unescape(deadline.strip())
                planning_lines.append(f"DEADLINE: {dl}")
            body_block = ""
            if body and body.strip():
                body_block = body.strip() + "\n" if not body.strip().endswith("\n") else body.strip()

            new_block = [line.rstrip("\n")] + props_block + planning_lines + (body_block.splitlines() if body_block else [])

            if header is not None:
                lines = file_content_for_header.splitlines()
            else:
                lines = (await self._org_read(user_id, resolved)).splitlines()
            insert_at: Optional[int] = None
            if insert_after_line_number is not None and insert_after_line_number >= 0:
                insert_at = min(insert_after_line_number + 1, len(lines))

            if insert_at is not None:
                lines = lines[:insert_at] + new_block + lines[insert_at:]
                line_index = insert_at
                await self._org_write(user_id, resolved, "\n".join(lines) + "\n")
            else:
                line_index = len(lines)
                # Ensure new entry starts on its own line (never on same line as :END: or last line)
                await self._org_append(user_id, resolved, "\n" + "\n".join(new_block) + "\n")

            return {
                "success": True,
                "file_path": str(resolved),
                "line_number": line_index,
                "heading": line.strip(),
            }
        except Exception as e:
            logger.exception("create_todo failed")
            return {"success": False, "error": str(e)}

    def _find_line_index(self, lines: List[str], line_number: int, heading_text: Optional[str], window: int = 10) -> Optional[int]:
        n = len(lines)
        if line_number < 0 or line_number >= n:
            return None
        line = lines[line_number].strip()
        if heading_text is None or (heading_text.strip() in line or line.endswith(heading_text.strip())):
            return line_number
        for delta in range(1, window + 1):
            for idx in (line_number - delta, line_number + delta):
                if 0 <= idx < n and heading_text.strip() in lines[idx]:
                    return idx
        return line_number

    @staticmethod
    def _body_line_range(lines: List[str], headline_idx: int) -> Tuple[int, int]:
        """Return (start, end) so lines[start:end] is the body under the headline at headline_idx."""
        start = headline_idx + 1
        n = len(lines)
        if start >= n:
            return start, start
        end = start
        while end < n and not lines[end].strip().startswith("*"):
            end += 1
        return start, end

    @staticmethod
    def _get_heading_level(line: str) -> int:
        """Return number of leading * stars for an org headline, or 0 for non-headlines."""
        stripped = line.lstrip()
        if not stripped.startswith("*"):
            return 0
        count = 0
        for c in stripped:
            if c == "*":
                count += 1
            else:
                break
        return count

    @staticmethod
    def _extract_subtree(lines: List[str], idx: int) -> List[str]:
        """Return contiguous block lines[idx:end] where end is next heading at same or higher level, or EOF."""
        level = OrgTodoService._get_heading_level(lines[idx])
        i = idx + 1
        while i < len(lines):
            line = lines[i]
            if line.strip().startswith("*"):
                if OrgTodoService._get_heading_level(line) <= level:
                    break
            i += 1
        return lines[idx:i]

    @staticmethod
    def _upsert_properties_entry(block_lines: List[str], key: str, value: str) -> List[str]:
        """Insert or update :KEY: value inside :PROPERTIES:...:END: drawer. Creates drawer if absent."""
        prop_start: Optional[int] = None
        prop_end: Optional[int] = None
        key_line_idx: Optional[int] = None
        i = 0
        while i < len(block_lines):
            s = block_lines[i].strip()
            if s == ":PROPERTIES:":
                prop_start = i
                i += 1
                while i < len(block_lines):
                    t = block_lines[i].strip()
                    if t == ":END:":
                        prop_end = i
                        break
                    if t.startswith(f":{key}:"):
                        key_line_idx = i
                    i += 1
                break
            i += 1
        new_line = f":{key}: {value}"
        if prop_start is not None and prop_end is not None:
            if key_line_idx is not None:
                block_lines[key_line_idx] = new_line
            else:
                block_lines.insert(prop_end, new_line)
        else:
            insert_idx = 1
            while insert_idx < len(block_lines):
                s = block_lines[insert_idx].strip()
                if s.startswith("SCHEDULED:") or s.startswith("DEADLINE:") or s.startswith("CLOSED:"):
                    insert_idx += 1
                else:
                    break
            for line in reversed([":PROPERTIES:", new_line, ":END:"]):
                block_lines.insert(insert_idx, line)
        return block_lines

    async def toggle_todo(
        self,
        user_id: str,
        file_path: str,
        line_number: int,
        heading_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Toggle TODO <-> DONE (any org heading level) or - [ ] <-> - [x] (markdown)."""
        resolved = await self._resolve_file_path(user_id, file_path)
        if not resolved or not await self._org_exists(user_id, resolved):
            return {"success": False, "error": f"File not found: {file_path}"}
        try:
            lines = (await self._org_read(user_id, resolved)).splitlines()
            idx = self._find_line_index(lines, line_number, heading_text)
            if idx is None:
                return {"success": False, "error": "line_index out of range"}
            line = lines[idx]
            if "- [ ]" in line:
                lines[idx] = line.replace("- [ ]", "- [x]", 1)
            elif "- [x]" in line:
                lines[idx] = line.replace("- [x]", "- [ ]", 1)
            else:
                # Org-mode: any heading level with a configured todo state; cycle done <-> active
                done_states = await _resolve_done_states(user_id)
                active_states = await _resolve_active_states(user_id)
                all_known = done_states | active_states
                if not all_known:
                    return {"success": False, "error": "no todo states configured"}
                pattern = "|".join(re.escape(s) for s in all_known)
                m = re.match(rf"^(\s*\*+)\s+({pattern})(\s.*)?$", line.strip(), re.IGNORECASE)
                if m:
                    current = m.group(2).upper()
                    rest = m.group(3) or ""
                    leading = line[: len(line) - len(line.lstrip())]
                    if current in done_states:
                        new_state = sorted(active_states)[0]
                    else:
                        new_state = sorted(done_states)[0]
                    lines[idx] = leading + f"{m.group(1)} {new_state}{rest}"
                    # Add or remove CLOSED: planning line when moving to/from a done state
                    if new_state in done_states:
                        ts = await self._org_timestamp(user_id)
                        closed_line = f"CLOSED: [{ts}]"
                        insert_at = idx + 1
                        closed_idx = self._find_planning_line(lines, idx, "CLOSED:")
                        if closed_idx is not None:
                            lines[closed_idx] = closed_line
                        else:
                            lines.insert(insert_at, closed_line)
                    else:
                        closed_idx = self._find_planning_line(lines, idx, "CLOSED:")
                        if closed_idx is not None:
                            del lines[closed_idx]
                else:
                    return {"success": False, "error": "line is not a task (expected org TODO/DONE or markdown - [ ]/- [x])"}
            await self._org_write(user_id, resolved, "\n".join(lines) + "\n")
            return {"success": True, "file_path": str(resolved), "line_number": idx, "new_line": lines[idx]}
        except Exception as e:
            logger.exception("toggle_todo failed")
            return {"success": False, "error": str(e)}

    def _find_planning_line(self, lines: List[str], headline_idx: int, prefix: str) -> Optional[int]:
        """Return index of first line after headline that starts with prefix, within ~10 lines; else None."""
        for i in range(headline_idx + 1, min(headline_idx + 10, len(lines))):
            stripped = lines[i].strip()
            if stripped.startswith(prefix):
                return i
            if stripped.startswith("*"):
                break
        return None

    async def update_todo(
        self,
        user_id: str,
        file_path: str,
        line_number: int,
        heading_text: Optional[str] = None,
        new_state: Optional[str] = None,
        new_text: Optional[str] = None,
        add_tags: Optional[List[str]] = None,
        remove_tags: Optional[List[str]] = None,
        scheduled: Optional[str] = None,
        deadline: Optional[str] = None,
        priority: Optional[str] = None,
        new_body: Optional[str] = None,
    ) -> Dict[str, Any]:
        resolved = await self._resolve_file_path(user_id, file_path)
        if not resolved or not await self._org_exists(user_id, resolved):
            return {"success": False, "error": f"File not found: {file_path}"}
        try:
            lines = (await self._org_read(user_id, resolved)).splitlines()
            idx = self._find_line_index(lines, line_number, heading_text)
            if idx is None:
                return {"success": False, "error": "line_index out of range"}
            line = lines[idx]
            if not line.lstrip().startswith("*"):
                return {"success": False, "error": "line is not a headline"}

            if new_state is not None:
                # Replace TODO state word only; preserve asterisks, spacing, and rest of line (tags, etc.)
                line = re.sub(r"^(\s*\*+\s+)\S+(\s.*)?$", r"\1" + new_state.upper() + r"\2", line, count=1)

            if new_text is not None:
                stripped = line.strip()
                parts = stripped.split(" ", 2)
                prefix = (parts[0] + " " + parts[1] + " ") if len(parts) >= 2 else ""
                line = prefix + new_text.strip()

            if add_tags or remove_tags:
                m = re.search(r"\s+:([A-Za-z0-9_:+-]+):\s*$", line)
                current = set()
                if m:
                    current = set(t for t in m.group(1).split(":") if t)
                    line = re.sub(r"\s+:([A-Za-z0-9_:+-]+):\s*$", "", line)
                if add_tags:
                    current.update(t.strip() for t in add_tags if t.strip())
                if remove_tags:
                    current -= {t.strip() for t in remove_tags if t.strip()}
                if current:
                    line = line.rstrip() + "  :" + ":".join(sorted(current)) + ":"

            if priority is not None:
                p = priority.strip().upper()
                line = re.sub(r'\s*\[#[ABC]\]', '', line)
                if p in ('A', 'B', 'C'):
                    line = re.sub(
                        r'^(\s*\*+\s+\S+\s+)',
                        r'\1[#' + p + r'] ',
                        line,
                        count=1,
                    )
            lines[idx] = line

            if new_state is not None:
                done_set = await _resolve_done_states(user_id)
                if new_state.upper() in done_set:
                    ts = await self._org_timestamp(user_id)
                    closed_line = f"CLOSED: [{ts}]"
                    closed_idx = self._find_planning_line(lines, idx, "CLOSED:")
                    if closed_idx is not None:
                        lines[closed_idx] = closed_line
                    else:
                        lines.insert(idx + 1, closed_line)
                else:
                    closed_idx = self._find_planning_line(lines, idx, "CLOSED:")
                    if closed_idx is not None:
                        del lines[closed_idx]

            if new_body is not None:
                start, end = self._body_line_range(lines, idx)
                new_body_lines = new_body.strip().split("\n") if new_body.strip() else []
                lines = lines[:start] + new_body_lines + lines[end:]

            await self._org_write(user_id, resolved, "\n".join(lines) + "\n")
            return {"success": True, "file_path": str(resolved), "line_number": idx, "new_line": line}
        except Exception as e:
            logger.exception("update_todo failed")
            return {"success": False, "error": str(e)}

    async def delete_todo(
        self,
        user_id: str,
        file_path: str,
        line_number: int,
        heading_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        resolved = await self._resolve_file_path(user_id, file_path)
        if not resolved or not await self._org_exists(user_id, resolved):
            return {"success": False, "error": f"File not found: {file_path}"}
        try:
            lines = (await self._org_read(user_id, resolved)).splitlines()
            idx = self._find_line_index(lines, line_number, heading_text)
            if idx is None:
                return {"success": False, "error": "line_index out of range"}
            to_remove = [idx]
            for i in range(idx + 1, min(idx + 5, len(lines))):
                if lines[i].strip().startswith("SCHEDULED:") or lines[i].strip().startswith("DEADLINE:"):
                    to_remove.append(i)
                elif lines[i].strip().startswith("*"):
                    break
            for i in reversed(to_remove):
                del lines[i]
            await self._org_write(user_id, resolved, "\n".join(lines) + ("\n" if lines else ""))
            return {"success": True, "file_path": str(resolved), "deleted_line_count": len(to_remove)}
        except Exception as e:
            logger.exception("delete_todo failed")
            return {"success": False, "error": str(e)}

    async def archive_done(
        self, user_id: str, file_path: Optional[str] = None, preview_only: bool = False, line_number: Optional[int] = None
    ) -> Dict[str, Any]:
        try:
            if line_number is not None:
                if not file_path or not file_path.strip():
                    return {"success": False, "error": "file_path required when archiving a single entry by line_number"}
                from services.org_archive_service import get_org_archive_service
                archive_service = await get_org_archive_service()
                resolved = await self._resolve_file_path(user_id, file_path)
                if not resolved or not await self._org_exists(user_id, resolved):
                    return {"success": False, "error": f"File not found: {file_path}"}
                one_based = line_number + 1
                result = await archive_service.archive_entry(
                    user_id=user_id, source_file=file_path, line_number=one_based
                )
                if not result.get("success"):
                    return {"success": False, "error": result.get("error", "Archive failed")}
                rel_archive = result.get("archive_file", "")
                archive_full = (resolved.parent.parent / rel_archive) if rel_archive else resolved.parent / f"{resolved.stem}_archive.org"
                from utils.org_header_parser import parse_org_file_header
                header = parse_org_file_header(await self._org_read(user_id, resolved, utf8_sig=True))
                return {
                    "success": True,
                    "path": str(resolved),
                    "archived_to": str(archive_full),
                    "archived_count": result.get("lines_archived", 1),
                    "directive_found": header.archive is not None,
                    "directive_value": header.archive or "",
                    "archived_heading": result.get("archived_heading", ""),
                }
            if file_path is None or (isinstance(file_path, str) and file_path.strip() == ""):
                logger.info("archive_done: no file_path provided, archiving from inbox to global archive")
                from services.langgraph_tools.org_inbox_tools import org_inbox_archive_done
                result = await org_inbox_archive_done(user_id)
                result.setdefault("directive_found", False)
                result.setdefault("directive_value", "")
                return result
            resolved = await self._resolve_file_path(user_id, file_path)
            if not resolved or not await self._org_exists(user_id, resolved):
                return {"success": False, "error": f"File not found: {file_path}"}
            done_states = await _resolve_done_states(user_id)
            pattern = "|".join(re.escape(s) for s in done_states)
            done_re = re.compile(rf"^\*+\s+({pattern})\s", re.IGNORECASE)
            from services.org_archive_service import get_org_archive_service
            archive_service = await get_org_archive_service()
            archive_path = await archive_service._get_archive_path(user_id, file_path, resolved)
            file_content = await self._org_read(user_id, resolved, utf8_sig=True)
            from utils.org_header_parser import parse_org_file_header
            header = parse_org_file_header(file_content)
            directive_found = header.archive is not None
            directive_value = header.archive or ""
            if preview_only:
                lines = file_content.splitlines()
                archived_count = 0
                i = 0
                while i < len(lines):
                    stripped = lines[i].strip()
                    if done_re.match(stripped):
                        subtree = self._extract_subtree(lines, i)
                        archived_count += 1
                        i += len(subtree)
                    elif stripped.startswith("- [x]"):
                        archived_count += 1
                        i += 1
                    else:
                        i += 1
                return {
                    "success": True,
                    "path": str(resolved),
                    "archived_to": str(archive_path),
                    "archived_count": archived_count,
                    "directive_found": directive_found,
                    "directive_value": directive_value,
                }
            archive_path = Path(archive_path)
            ts = await self._org_timestamp(user_id)
            source_file_path = str(resolved)
            lines = file_content.splitlines()
            done_subtrees: List[str] = []
            archived_count = 0
            i = 0
            while i < len(lines):
                stripped = lines[i].strip()
                if done_re.match(stripped):
                    subtree = self._extract_subtree(lines, i)
                    subtree = self._upsert_properties_entry(subtree, "ARCHIVE_TIME", f"[{ts}]")
                    subtree = self._upsert_properties_entry(subtree, "ARCHIVE_FILE", source_file_path)
                    done_subtrees.extend(subtree)
                    del lines[i : i + len(subtree)]
                    archived_count += 1
                elif stripped.startswith("- [x]"):
                    subtree = [lines[i]]
                    subtree = self._upsert_properties_entry(subtree, "ARCHIVE_TIME", f"[{ts}]")
                    subtree = self._upsert_properties_entry(subtree, "ARCHIVE_FILE", source_file_path)
                    done_subtrees.extend(subtree)
                    del lines[i]
                    archived_count += 1
                else:
                    i += 1
            await self._org_write(user_id, resolved, "\n".join(lines) + ("\n" if lines else ""))
            archive_blob = "".join(
                (line + "\n" if not line.endswith("\n") else line) for line in done_subtrees
            )
            await self._org_append(user_id, archive_path, archive_blob)
            return {
                "success": True,
                "path": str(resolved),
                "archived_to": str(archive_path),
                "archived_count": archived_count,
                "directive_found": directive_found,
                "directive_value": directive_value,
            }
        except Exception as e:
            logger.exception("archive_done failed")
            return {"success": False, "error": str(e)}
