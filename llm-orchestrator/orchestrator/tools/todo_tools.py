"""
Universal Todo Tools - List, create, update, toggle, delete, archive todos across any org file.
Thin gRPC wrappers over backend OrgTodoService. I/O contracts for Agent Factory and automation engine.
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)


# ── List Todos ─────────────────────────────────────────────────────────────

class ListTodosInputs(BaseModel):
    """Required inputs for list_todos."""
    scope: str = Field(default="all", description="all, inbox, or org file path")


class ListTodosParams(BaseModel):
    """Optional parameters for list_todos."""
    states: Optional[List[str]] = Field(default=None, description="TODO states to include")
    tags: Optional[List[str]] = Field(default=None, description="Filter by tags")
    query: str = Field(default="", description="Search query")
    limit: int = Field(default=100, ge=1, le=500, description="Max results")
    include_archives: bool = Field(default=False, description="Include archive files")
    include_body: bool = Field(default=False, description="Include body/description text for each todo")
    closed_since_days: Optional[int] = Field(default=None, ge=0, description="Only DONE items closed in the last N days (e.g. 7 for last week)")


class ListTodosOutputs(BaseModel):
    """Outputs for list_todos."""
    success: bool = Field(description="Whether the request succeeded")
    results: List[Dict[str, Any]] = Field(
        description="Todo items: file_path, line_number (0-based), level, heading, todo_state, tags, scheduled, deadline, closed. Filter with states/tags/query params. Use file_path and line_number for update_todo/toggle_todo/delete_todo."
    )
    count: int = Field(description="Number of items returned")
    files_searched: int = Field(description="Number of org files searched")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def list_todos_tool(
    scope: str = "all",
    user_id: str = "system",
    states: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    query: str = "",
    limit: int = 100,
    include_archives: bool = False,
    include_body: bool = False,
    closed_since_days: Optional[int] = None,
) -> Dict[str, Any]:
    """List todos across org files. Use closed_since_days=7 for items completed in the last week. line_number in results is 0-based."""
    try:
        client = await get_backend_tool_client()
        result = await client.list_todos(
            user_id=user_id,
            scope=scope,
            states=states,
            tags=tags,
            query=query,
            limit=limit,
            include_archives=include_archives,
            include_body=include_body,
            closed_since_days=closed_since_days,
        )
        if not result.get("success"):
            err = result.get("error", "Unknown error")
            return {"success": False, "results": [], "count": 0, "files_searched": 0, "error": err, "formatted": err}
        results = result.get("results", [])
        count = result.get("count", 0)
        files_searched = result.get("files_searched", 0)
        lines = [f"Found {count} todo(s) across {files_searched} file(s). To toggle or update a todo, use its file_path and line_number (0-based)."]
        for i, r in enumerate(results[:15], 1):
            ln = r.get("line_number", 0)
            fp = r.get("file_path", "") or r.get("filename", "")
            closed = r.get("closed", "") or ""
            extra = f" closed={closed}" if closed else ""
            lines.append(f"{i}. [{r.get('todo_state', '')}] {r.get('heading', '')[:60]}{extra} — file_path={fp} line_number={ln}")
        if count > 15:
            lines.append(f"... and {count - 15} more.")
        return {
            "success": True,
            "results": results,
            "count": count,
            "files_searched": files_searched,
            "error": None,
            "formatted": "\n".join(lines),
        }
    except Exception as e:
        logger.exception("list_todos_tool failed")
        return {"success": False, "results": [], "count": 0, "files_searched": 0, "error": str(e), "formatted": str(e)}


# ── Create Todo ────────────────────────────────────────────────────────────

class CreateTodoInputs(BaseModel):
    """Required inputs for create_todo."""
    text: str = Field(
        description="Todo title only. Do not put tags (:tag:) or priority ([#A]) in the title; use the tags and priority parameters instead."
    )


class CreateTodoParams(BaseModel):
    """Optional parameters for create_todo."""
    file_path: Optional[str] = Field(default=None, description="Org file path (omit for inbox)")
    state: str = Field(default="TODO", description="Initial state (TODO, NEXT, etc.)")
    tags: Optional[List[str]] = Field(
        default=None,
        description="Tags for the heading (e.g. ['home', 'work']). Use this parameter; do not put :tag: in the title text.",
    )
    scheduled: Optional[str] = Field(default=None, description="Org timestamp (e.g. <2026-03-01 Sun>)")
    deadline: Optional[str] = Field(default=None, description="Org timestamp (e.g. <2026-03-15 Sun>)")
    priority: Optional[str] = Field(
        default=None,
        description="Priority A, B, or C. Use this parameter; do not put [#A] in the title text.",
    )
    body: Optional[str] = Field(default=None, description="Body/description text under the heading")
    heading_level: Optional[int] = Field(default=None, ge=1, le=6, description="Org heading level 1-6 (stars). Default 1.")
    insert_after_line_number: Optional[int] = Field(
        default=None,
        ge=0,
        description="0-based line index; insert new todo after this line (new heading appears on the next line). Omit to append at end of file. New entries are always placed on their own line, never on the same line as :END: or other metadata.",
    )


class CreateTodoOutputs(BaseModel):
    """Outputs for create_todo."""
    success: bool = Field(description="Whether the todo was created")
    file_path: Optional[str] = Field(default=None, description="Path to the org file")
    line_number: int = Field(description="0-based line index")
    heading: str = Field(description="Full heading line")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def create_todo_tool(
    text: str,
    user_id: str = "system",
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
    """Create a todo. Omit file_path for inbox. Put tags in the tags parameter and priority (A/B/C) in the priority parameter; do not embed :tag: or [#A] in the title text. Use heading_level 1-6 for org depth. Use insert_after_line_number (0-based) to insert after a specific line; omit to append at end. New entries are always written on a new line (never on the same line as :END: or the previous entry). list_todos returns line_number and level for update/toggle/delete."""
    try:
        client = await get_backend_tool_client()
        result = await client.create_todo(
            user_id=user_id,
            text=text,
            file_path=file_path,
            state=state,
            tags=tags,
            scheduled=scheduled,
            deadline=deadline,
            priority=priority,
            body=body,
            heading_level=heading_level,
            insert_after_line_number=insert_after_line_number,
        )
        if not result.get("success"):
            err = result.get("error", "Unknown error")
            return {"success": False, "file_path": None, "line_number": -1, "heading": "", "error": err, "formatted": err}
        return {
            "success": True,
            "file_path": result.get("file_path"),
            "line_number": result.get("line_number", 0),
            "heading": result.get("heading", ""),
            "error": None,
            "formatted": f"Created todo at line {result.get('line_number', 0)} in {result.get('file_path', 'inbox')}.",
        }
    except Exception as e:
        logger.exception("create_todo_tool failed")
        return {"success": False, "file_path": None, "line_number": -1, "heading": "", "error": str(e), "formatted": str(e)}


# ── Update Todo ─────────────────────────────────────────────────────────────

class UpdateTodoInputs(BaseModel):
    """Required inputs for update_todo."""
    file_path: str = Field(description="Org file path")
    line_number: int = Field(description="0-based line index from list_todos")
    new_state: Optional[str] = Field(default=None, description="New TODO state (TODO, DONE, etc.)")
    new_text: Optional[str] = Field(
        default=None,
        description="New heading title only. Do not include tags or priority here; use add_tags/remove_tags and priority.",
    )


class UpdateTodoParams(BaseModel):
    """Optional parameters for update_todo."""
    heading_text: Optional[str] = Field(default=None, description="Verify heading matches before updating")
    add_tags: Optional[List[str]] = Field(default=None, description="Tags to add (use this; do not put :tag: in new_text)")
    remove_tags: Optional[List[str]] = Field(default=None, description="Tags to remove")
    scheduled: Optional[str] = Field(default=None, description="Org timestamp (e.g. <2026-03-01 Sun>)")
    deadline: Optional[str] = Field(default=None, description="Org timestamp")
    priority: Optional[str] = Field(
        default=None,
        description="Set priority to A, B, or C. Use this parameter; do not put [#A] in new_text.",
    )
    new_body: Optional[str] = Field(default=None, description="Replace body/description under the heading")


class UpdateTodoOutputs(BaseModel):
    """Outputs for update_todo."""
    success: bool = Field(description="Whether the update succeeded")
    file_path: str = Field(description="Org file path")
    line_number: int = Field(description="Line index")
    new_line: str = Field(description="Updated line content")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def update_todo_tool(
    file_path: str,
    line_number: int,
    user_id: str = "system",
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
    """Update a todo. Use add_tags/remove_tags for tags and priority for A/B/C; do not embed :tag: or [#A] in new_text."""
    try:
        client = await get_backend_tool_client()
        result = await client.update_todo(
            user_id=user_id,
            file_path=file_path,
            line_number=line_number,
            heading_text=heading_text,
            new_state=new_state,
            new_text=new_text,
            add_tags=add_tags,
            remove_tags=remove_tags,
            scheduled=scheduled,
            deadline=deadline,
            priority=priority,
            new_body=new_body,
        )
        if not result.get("success"):
            err = result.get("error", "Unknown error")
            return {"success": False, "file_path": file_path, "line_number": line_number, "new_line": "", "error": err, "formatted": err}
        return {
            "success": True,
            "file_path": result.get("file_path", file_path),
            "line_number": result.get("line_number", line_number),
            "new_line": result.get("new_line", ""),
            "error": None,
            "formatted": f"Updated todo at line {result.get('line_number')} to {result.get('new_line', '')[:50]}.",
        }
    except Exception as e:
        logger.exception("update_todo_tool failed")
        return {"success": False, "file_path": file_path, "line_number": line_number, "new_line": "", "error": str(e), "formatted": str(e)}


# ── Toggle Todo ─────────────────────────────────────────────────────────────

class ToggleTodoInputs(BaseModel):
    """Required inputs for toggle_todo."""
    file_path: str = Field(description="Org file path")
    line_number: int = Field(description="0-based line index")


class ToggleTodoParams(BaseModel):
    """Optional parameters for toggle_todo."""
    heading_text: Optional[str] = Field(default=None, description="Verify heading matches")


class ToggleTodoOutputs(BaseModel):
    """Outputs for toggle_todo."""
    success: bool = Field(description="Whether the toggle succeeded")
    file_path: str = Field(description="Org file path")
    line_number: int = Field(description="Line index")
    new_line: str = Field(description="Line after toggle")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def toggle_todo_tool(
    file_path: str,
    line_number: int,
    user_id: str = "system",
    heading_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Toggle TODO <-> DONE (org headings at any level: *, **, ***, ...) or - [ ] <-> - [x] (markdown)."""
    try:
        client = await get_backend_tool_client()
        result = await client.toggle_todo(
            user_id=user_id,
            file_path=file_path,
            line_number=line_number,
            heading_text=heading_text,
        )
        if not result.get("success"):
            err = result.get("error", "Unknown error")
            return {"success": False, "file_path": file_path, "line_number": line_number, "new_line": "", "error": err, "formatted": err}
        return {
            "success": True,
            "file_path": result.get("file_path", file_path),
            "line_number": result.get("line_number", line_number),
            "new_line": result.get("new_line", ""),
            "error": None,
            "formatted": f"Toggled todo at line {result.get('line_number')} to {result.get('new_line', '')[:50]}.",
        }
    except Exception as e:
        logger.exception("toggle_todo_tool failed")
        return {"success": False, "file_path": file_path, "line_number": line_number, "new_line": "", "error": str(e), "formatted": str(e)}


# ── Delete Todo ─────────────────────────────────────────────────────────────

class DeleteTodoInputs(BaseModel):
    """Required inputs for delete_todo."""
    file_path: str = Field(description="Org file path")
    line_number: int = Field(description="0-based line index")


class DeleteTodoParams(BaseModel):
    """Optional parameters for delete_todo."""
    heading_text: Optional[str] = Field(default=None, description="Verify heading matches")


class DeleteTodoOutputs(BaseModel):
    """Outputs for delete_todo."""
    success: bool = Field(description="Whether the delete succeeded")
    file_path: str = Field(description="Org file path")
    deleted_line_count: int = Field(description="Number of lines removed")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def delete_todo_tool(
    file_path: str,
    line_number: int,
    user_id: str = "system",
    heading_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Delete the todo line (and any SCHEDULED/DEADLINE line below it)."""
    try:
        client = await get_backend_tool_client()
        result = await client.delete_todo(
            user_id=user_id,
            file_path=file_path,
            line_number=line_number,
            heading_text=heading_text,
        )
        if not result.get("success"):
            err = result.get("error", "Unknown error")
            return {"success": False, "file_path": file_path, "deleted_line_count": 0, "error": err, "formatted": err}
        return {
            "success": True,
            "file_path": result.get("file_path", file_path),
            "deleted_line_count": result.get("deleted_line_count", 0),
            "error": None,
            "formatted": f"Deleted todo at line {line_number} ({result.get('deleted_line_count', 0)} line(s) removed).",
        }
    except Exception as e:
        logger.exception("delete_todo_tool failed")
        return {"success": False, "file_path": file_path, "deleted_line_count": 0, "error": str(e), "formatted": str(e)}


# ── Archive Done ───────────────────────────────────────────────────────────

class ArchiveDoneInputs(BaseModel):
    """file_path: org file (omit for inbox). line_number: 0-based from list_todos; if set, archive only this entry (any state). Omit for bulk closed."""
    file_path: Optional[str] = Field(default=None, description="Org file path (omit for inbox)")
    line_number: Optional[int] = Field(default=None, ge=0, description="0-based line index of one entry to archive (any state). Omit to bulk-archive all closed items.")


class ArchiveDoneParams(BaseModel):
    """Optional parameters for archive_done."""
    preview_only: bool = Field(
        default=False,
        description="If true, resolve archive path and count items without writing. Use to confirm destination before archiving.",
    )


class ArchiveDoneOutputs(BaseModel):
    """Outputs for archive_done."""
    success: bool = Field(description="Whether the archive succeeded")
    path: str = Field(description="Source file path")
    archived_to: str = Field(description="Archive file path")
    archived_count: int = Field(description="Number of items archived")
    directive_found: bool = Field(description="Whether file had #+ARCHIVE: directive")
    directive_value: str = Field(description="Raw #+ARCHIVE: value if present (e.g. ./Archive/%s_archive.org)")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


def _format_archive_done_result(result: Dict[str, Any], preview_only: bool) -> str:
    count = result.get("archived_count", 0)
    archived_to = result.get("archived_to", "")
    directive_found = result.get("directive_found", False)
    directive_value = result.get("directive_value", "")
    single = count == 1
    heading = result.get("archived_heading", "")
    if single and heading:
        kind = f"entry '{heading}'"
    elif single:
        kind = "1 entry"
    else:
        kind = f"{count} closed item(s) (DONE/CANCELLED)"
    if preview_only:
        if directive_found:
            return (
                f"Preview: would archive {kind} to {archived_to} "
                f"(from #+ARCHIVE: directive '{directive_value}'). "
                "Call again without preview_only=True to execute."
            )
        return (
            f"Preview: would archive {kind} to {archived_to} "
            "(default — no #+ARCHIVE: directive found in file). "
            "You may want to confirm this destination with the user before proceeding."
        )
    if directive_found:
        return f"Archived {kind} to {archived_to} (#+ARCHIVE: directive)."
    return f"Archived {kind} to {archived_to} (default)."


async def archive_done_tool(
    user_id: str = "system",
    file_path: Optional[str] = None,
    line_number: Optional[int] = None,
    preview_only: bool = False,
) -> Dict[str, Any]:
    """Archive one entry or bulk closed items. To archive a specific entry (any state), provide file_path and line_number (0-based from list_todos). To bulk-archive all closed items, provide only file_path (or omit for inbox). Full entries (heading + subtree + properties) are always moved. Use preview_only=True to see destination and count without writing."""
    try:
        client = await get_backend_tool_client()
        result = await client.archive_done_todos(
            user_id=user_id, file_path=file_path, preview_only=preview_only, line_number=line_number
        )
        if not result.get("success"):
            err = result.get("error", "Unknown error")
            return {
                "success": False,
                "path": "",
                "archived_to": "",
                "archived_count": 0,
                "directive_found": False,
                "directive_value": "",
                "error": err,
                "formatted": err,
            }
        return {
            "success": True,
            "path": result.get("path", ""),
            "archived_to": result.get("archived_to", ""),
            "archived_count": result.get("archived_count", 0),
            "directive_found": result.get("directive_found", False),
            "directive_value": result.get("directive_value", ""),
            "error": None,
            "formatted": _format_archive_done_result(result, preview_only),
        }
    except Exception as e:
        logger.exception("archive_done_tool failed")
        return {
            "success": False,
            "path": "",
            "archived_to": "",
            "archived_count": 0,
            "directive_found": False,
            "directive_value": "",
            "error": str(e),
            "formatted": str(e),
        }


# ── Refile Todo ────────────────────────────────────────────────────────────

class RefileTodoInputs(BaseModel):
    """Required inputs for refile_todo."""
    source_file: str = Field(description="Org file path of the todo (from list_todos)")
    source_line: int = Field(description="0-based line index of the todo (from list_todos)")
    target_file: str = Field(description="Destination org file path")


class RefileTodoParams(BaseModel):
    """Optional parameters for refile_todo."""
    target_heading_line: Optional[int] = Field(default=None, ge=0, description="0-based line of target heading; omit to append at file root")


class RefileTodoOutputs(BaseModel):
    """Outputs for refile_todo."""
    success: bool = Field(description="Whether the refile succeeded")
    source_file: str = Field(description="Source file path")
    target_file: str = Field(description="Target file path")
    lines_moved: int = Field(description="Number of lines moved")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def refile_todo_tool(
    source_file: str,
    source_line: int,
    target_file: str,
    user_id: str = "system",
    target_heading_line: Optional[int] = None,
) -> Dict[str, Any]:
    """Move an org entry (any status — TODO, DONE, CANCELLED, etc.) and its full subtree (heading + body + properties) from one file to another. Use list_todos to get source_file and source_line; use discover_refile_targets or list_todos for target_file and optionally target_heading_line."""
    try:
        client = await get_backend_tool_client()
        result = await client.refile_todo(
            user_id=user_id,
            source_file=source_file,
            source_line=source_line,
            target_file=target_file,
            target_heading_line=target_heading_line,
        )
        if not result.get("success"):
            err = result.get("error", "Unknown error")
            return {
                "success": False,
                "source_file": result.get("source_file", source_file),
                "target_file": result.get("target_file", target_file),
                "lines_moved": result.get("lines_moved", 0),
                "error": err,
                "formatted": err,
            }
        lines_moved = result.get("lines_moved", 0)
        return {
            "success": True,
            "source_file": result.get("source_file", source_file),
            "target_file": result.get("target_file", target_file),
            "lines_moved": lines_moved,
            "error": None,
            "formatted": f"Refiled {lines_moved} line(s) from {result.get('source_file', source_file)} to {result.get('target_file', target_file)}.",
        }
    except Exception as e:
        logger.exception("refile_todo_tool failed")
        return {
            "success": False,
            "source_file": source_file,
            "target_file": target_file,
            "lines_moved": 0,
            "error": str(e),
            "formatted": str(e),
        }


# ── Discover Refile Targets ────────────────────────────────────────────────

class DiscoverRefileTargetsInputs(BaseModel):
    """No required inputs; user_id is injected."""


class DiscoverRefileTargetsOutputs(BaseModel):
    """Outputs for discover_refile_targets."""
    success: bool = Field(description="Whether the request succeeded")
    targets: List[Dict[str, Any]] = Field(description="List of refile destinations: file, filename, heading_path, heading_line (0-based), display_name, level")
    count: int = Field(description="Number of targets returned")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def discover_refile_targets_tool(user_id: str = "system") -> Dict[str, Any]:
    """List all org files and headings available as refile destinations. Use to present destination options to the user or to find the correct target_file/target_heading_line for refile_todo."""
    try:
        client = await get_backend_tool_client()
        result = await client.discover_refile_targets(user_id=user_id)
        if not result.get("success"):
            err = result.get("error", "Unknown error")
            return {"success": False, "targets": [], "count": 0, "error": err, "formatted": err}
        targets = result.get("targets", [])
        count = len(targets)
        lines = [f"Found {count} refile destination(s)."]
        for i, t in enumerate(targets[:20], 1):
            lines.append(f"{i}. {t.get('display_name', '')} — file={t.get('file', '')} heading_line={t.get('heading_line', 0)}")
        if count > 20:
            lines.append(f"... and {count - 20} more.")
        return {
            "success": True,
            "targets": targets,
            "count": count,
            "error": None,
            "formatted": "\n".join(lines),
        }
    except Exception as e:
        logger.exception("discover_refile_targets_tool failed")
        return {"success": False, "targets": [], "count": 0, "error": str(e), "formatted": str(e)}


# ── Register actions ───────────────────────────────────────────────────────

register_action(
    name="list_todos",
    category="org",
    description="List todos across org files (scope: all, inbox, or file path).",
    inputs_model=ListTodosInputs,
    params_model=ListTodosParams,
    outputs_model=ListTodosOutputs,
    tool_function=list_todos_tool,
)
register_action(
    name="create_todo",
    category="org",
    description=(
        "Create a todo in the inbox or a specific org file. New entries are always placed on their own line "
        "(after the last line or after insert_after_line_number); never on the same line as :END: or metadata. "
        "Use tags and priority parameters; do not put :tag: or [#A] in the title. Omit file_path for inbox. "
        "Optional: heading_level (1-6), insert_after_line_number (0-based, insert after that line), body, scheduled, deadline."
    ),
    inputs_model=CreateTodoInputs,
    params_model=CreateTodoParams,
    outputs_model=CreateTodoOutputs,
    tool_function=create_todo_tool,
)
register_action(
    name="update_todo",
    category="org",
    description="Update a todo (state, text, tags, schedule)",
    inputs_model=UpdateTodoInputs,
    params_model=UpdateTodoParams,
    outputs_model=UpdateTodoOutputs,
    tool_function=update_todo_tool,
)
register_action(
    name="toggle_todo",
    category="org",
    description="Toggle TODO <-> DONE for the given todo",
    inputs_model=ToggleTodoInputs,
    params_model=ToggleTodoParams,
    outputs_model=ToggleTodoOutputs,
    tool_function=toggle_todo_tool,
)
register_action(
    name="delete_todo",
    category="org",
    description="Delete the todo line and any SCHEDULED/DEADLINE line below it",
    inputs_model=DeleteTodoInputs,
    params_model=DeleteTodoParams,
    outputs_model=DeleteTodoOutputs,
    tool_function=delete_todo_tool,
)
register_action(
    name="archive_done",
    category="org",
    description="Archive one entry or bulk closed items. Single entry: provide file_path and line_number (0-based from list_todos) to archive that entry in any state. Bulk: omit line_number and provide file_path (or omit for inbox) to archive all closed items. Full entries (heading + subtree + properties) are always moved.",
    short_description="Archive one or bulk closed todo entries",
    inputs_model=ArchiveDoneInputs,
    params_model=ArchiveDoneParams,
    outputs_model=ArchiveDoneOutputs,
    tool_function=archive_done_tool,
)
register_action(
    name="refile_todo",
    category="org",
    description="Move a todo (and subtree) to another file or heading.",
    inputs_model=RefileTodoInputs,
    params_model=RefileTodoParams,
    outputs_model=RefileTodoOutputs,
    tool_function=refile_todo_tool,
)
register_action(
    name="discover_refile_targets",
    category="org",
    description="List org files and headings available as refile destinations.",
    inputs_model=DiscoverRefileTargetsInputs,
    params_model=None,
    outputs_model=DiscoverRefileTargetsOutputs,
    tool_function=discover_refile_targets_tool,
)
