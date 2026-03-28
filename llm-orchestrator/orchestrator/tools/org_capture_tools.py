"""
Org Capture Tools - Add items to the user's org-mode inbox via backend gRPC.

For todo management (list, update, toggle, delete, archive), use the universal
todo API in orchestrator.tools.todo_tools instead.
"""
import logging
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)


class AddOrgInboxItemInputs(BaseModel):
    """Required inputs for adding an org inbox item."""
    text: str = Field(description="Item text (e.g. 'Get groceries', 'Call mom')")


class AddOrgInboxItemParams(BaseModel):
    """Optional parameters."""
    kind: str = Field(default="todo", description="todo, note, checkbox, event, or contact")
    schedule: Optional[str] = Field(default=None, description="Optional org timestamp (e.g. <2026-03-01 Sun>)")
    tags: Optional[Union[List[str], str]] = Field(
        default=None,
        description="Tags for the item. Use this parameter; do not put :tag: in the title text (e.g. use tags=['home'], not 'Title :home:').",
    )


class AddOrgInboxItemOutputs(BaseModel):
    """Structured output from add_org_inbox_item."""
    success: bool = Field(description="Whether the item was added")
    todo_id: Optional[str] = Field(default=None, description="ID of the created item if available")
    file_path: Optional[str] = Field(default=None, description="Path to the org file if available")
    message: str = Field(description="Human-readable result message")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


def _to_tag_list(value: Optional[Union[List[str], str]]) -> Optional[List[str]]:
    """Accept comma-separated string or list for tags."""
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [s.strip() for s in value.split(",") if s.strip()] or None
    return None


async def add_org_inbox_item_tool(
    text: str = "",
    user_id: str = "system",
    kind: str = "todo",
    schedule: Optional[str] = None,
    tags: Optional[Union[List[str], str]] = None,
) -> Dict[str, Any]:
    """
    Add an item to the user's org-mode inbox. Supports kind: todo, note, checkbox, event, contact.
    New items are always appended on a new line (never on the same line as :END: or the previous entry).
    Put tags in the tags parameter; do not embed :tag: in the title text.
    Returns structured dict with success, message, and formatted.
    """
    try:
        if not text or not text.strip():
            msg = "Error: text is required."
            return {"success": False, "message": msg, "formatted": msg}
        tag_list = _to_tag_list(tags)
        logger.info("add_org_inbox_item: kind=%s text=%s", kind, text[:80])
        client = await get_backend_tool_client()
        result = await client.add_org_inbox_item(
            user_id=user_id,
            text=text.strip(),
            kind=kind,
            schedule=schedule,
            repeater=None,
            tags=tag_list,
        )
        if not result.get("success"):
            err = result.get("error", "Failed to add item.")
            return {"success": False, "message": err, "formatted": err}
        msg = result.get("message", "Item added to inbox.")
        return {
            "success": True,
            "todo_id": result.get("todo_id"),
            "file_path": result.get("file_path"),
            "message": msg,
            "formatted": msg,
        }
    except Exception as e:
        logger.error("add_org_inbox_item_tool error: %s", e)
        err = str(e)
        return {"success": False, "message": err, "formatted": f"Error: {err}"}


register_action(
    name="add_org_inbox_item",
    category="org",
    description=(
        "Quick-capture to org inbox (kind: todo, note, checkbox, event, contact). "
        "New items are always appended on a new line, never on the same line as :END: or the previous entry. "
        "Use create_todo for full todo creation (body, heading_level, insert position, scheduled, deadline)."
    ),
    inputs_model=AddOrgInboxItemInputs,
    params_model=AddOrgInboxItemParams,
    outputs_model=AddOrgInboxItemOutputs,
    tool_function=add_org_inbox_item_tool,
)


# ---- Journal capture ----

class CaptureJournalEntryInputs(BaseModel):
    """Required inputs for capturing a journal entry."""
    content: str = Field(description="Journal entry body text")


class CaptureJournalEntryParams(BaseModel):
    """Optional parameters for journal capture."""
    entry_date: Optional[str] = Field(default=None, description="Date for entry (YYYY-MM-DD); omit for today")
    title: Optional[str] = Field(default=None, description="Optional heading title")
    tags: Optional[Union[List[str], str]] = Field(default=None, description="Comma-separated or list of tags")


class CaptureJournalEntryOutputs(BaseModel):
    """Structured output from capture_journal_entry."""
    success: bool = Field(description="Whether the entry was captured")
    message: str = Field(description="Result message")
    entry_preview: Optional[str] = Field(default=None, description="Preview of the captured entry")
    file_path: Optional[str] = Field(default=None, description="Path to the journal file")
    document_id: Optional[str] = Field(default=None, description="Document ID if available")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def capture_journal_entry_tool(
    content: str = "",
    user_id: str = "system",
    entry_date: Optional[str] = None,
    title: Optional[str] = None,
    tags: Optional[Union[List[str], str]] = None,
) -> Dict[str, Any]:
    """
    Append an entry to the user's journal. Respects the user's journal organization
    preferences (monolithic, yearly, monthly, or daily files) and automatically
    creates date headings if missing. Use entry_date for back-dating. For editing
    an already-open journal file, use propose_document_edit instead.
    """
    try:
        if not content or not content.strip():
            msg = "Error: content is required."
            return {
                "success": False,
                "message": msg,
                "entry_preview": None,
                "file_path": None,
                "document_id": None,
                "formatted": msg,
            }
        tag_list = _to_tag_list(tags)
        logger.info("capture_journal_entry: content=%s", content[:80])
        client = await get_backend_tool_client()
        result = await client.capture_journal_entry(
            user_id=user_id,
            content=content.strip(),
            entry_date=entry_date,
            title=title,
            tags=tag_list,
        )
        if not result.get("success"):
            err = result.get("error", result.get("message", "Failed to capture."))
            return {
                "success": False,
                "message": result.get("message", err),
                "entry_preview": result.get("entry_preview"),
                "file_path": result.get("file_path"),
                "document_id": result.get("document_id"),
                "formatted": err,
            }
        msg = result.get("message", "Journal entry captured.")
        return {
            "success": True,
            "message": msg,
            "entry_preview": result.get("entry_preview"),
            "file_path": result.get("file_path"),
            "document_id": result.get("document_id"),
            "formatted": msg,
        }
    except Exception as e:
        logger.error("capture_journal_entry_tool error: %s", e)
        err = str(e)
        return {
            "success": False,
            "message": err,
            "entry_preview": None,
            "file_path": None,
            "document_id": None,
            "formatted": f"Error: {err}",
        }


register_action(
    name="capture_journal_entry",
    category="org",
    description=(
        "REQUIRED for adding new journal entries. Append an entry to the user's journal; "
        "optional date (YYYY-MM-DD) and title. Use this—not file create or document update—"
        "when the user asks to add or write a journal entry; it respects journal structure and avoids data loss."
    ),
    inputs_model=CaptureJournalEntryInputs,
    params_model=CaptureJournalEntryParams,
    outputs_model=CaptureJournalEntryOutputs,
    tool_function=capture_journal_entry_tool,
)


# ---- Journal read tools ----

class GetJournalEntryInputs(BaseModel):
    """Required inputs for reading a single journal entry."""
    date: str = Field(
        default="today",
        description="Date for the entry: 'today' or YYYY-MM-DD (e.g. 2026-03-04)",
    )


class GetJournalEntryOutputs(BaseModel):
    """Structured output from get_journal_entry."""
    success: bool = Field(description="Whether the read succeeded")
    content: str = Field(description="Body content of the journal entry for that date")
    date: str = Field(description="Date of the entry (YYYY-MM-DD)")
    heading: str = Field(description="Org heading line for that date")
    has_content: bool = Field(description="True if the entry has non-empty content")
    document_id: Optional[str] = Field(default=None, description="Document ID if available")
    file_path: Optional[str] = Field(default=None, description="Path to the journal file")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def get_journal_entry_tool(
    date: str = "today",
    user_id: str = "system",
) -> Dict[str, Any]:
    """
    Read one date's journal entry (section-aware). date: 'today' or YYYY-MM-DD.
    """
    try:
        client = await get_backend_tool_client()
        result = await client.get_journal_entry(user_id=user_id, date=date or "today")
        if not result.get("success"):
            err = result.get("error", "Failed to read journal entry.")
            return {
                "success": False,
                "content": "",
                "date": result.get("date", ""),
                "heading": "",
                "has_content": False,
                "document_id": None,
                "file_path": None,
                "formatted": err,
            }
        content = result.get("content", "")
        entry_date = result.get("date", "")
        has_content = result.get("has_content", False)
        if has_content:
            formatted = f"Journal entry for {entry_date}:\n\n{content}"
        else:
            formatted = f"No journal content for {entry_date}."
        return {
            "success": True,
            "content": content,
            "date": entry_date,
            "heading": result.get("heading", ""),
            "has_content": has_content,
            "document_id": result.get("document_id"),
            "file_path": result.get("file_path"),
            "formatted": formatted,
        }
    except Exception as e:
        logger.error("get_journal_entry_tool error: %s", e)
        err = str(e)
        return {
            "success": False,
            "content": "",
            "date": "",
            "heading": "",
            "has_content": False,
            "document_id": None,
            "file_path": None,
            "formatted": f"Error: {err}",
        }


register_action(
    name="get_journal_entry",
    category="org",
    description="Read one date's journal entry. Use date='today' or YYYY-MM-DD. Returns content, heading, and metadata.",
    inputs_model=GetJournalEntryInputs,
    params_model=None,
    outputs_model=GetJournalEntryOutputs,
    tool_function=get_journal_entry_tool,
)


class GetJournalEntriesParams(BaseModel):
    """Optional parameters for get_journal_entries."""
    start_date: Optional[str] = Field(default=None, description="Start of range (YYYY-MM-DD); default 31 days ago")
    end_date: Optional[str] = Field(default=None, description="End of range (YYYY-MM-DD); default today")
    max_entries: int = Field(default=30, ge=1, le=100, description="Max number of entries to return")


class GetJournalEntriesInputs(BaseModel):
    """No required inputs; use params for date range and limit."""
    pass


class GetJournalEntriesOutputs(BaseModel):
    """Structured output from get_journal_entries."""
    success: bool = Field(description="Whether the read succeeded")
    entries: List[Dict[str, Any]] = Field(
        description="List of {date, content, heading, has_content} for each entry"
    )
    total: int = Field(description="Number of entries returned")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def get_journal_entries_tool(
    user_id: str = "system",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_entries: int = 30,
) -> Dict[str, Any]:
    """
    Get full content of journal entries in a date range (for review or summarization).
    """
    try:
        client = await get_backend_tool_client()
        result = await client.get_journal_entries(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            max_entries=max(1, min(max_entries, 100)),
        )
        if not result.get("success"):
            err = result.get("error", "Failed to read journal entries.")
            return {"success": False, "entries": [], "total": 0, "formatted": err}
        entries = result.get("entries", [])
        total = result.get("total", 0)
        parts = [f"Found {total} journal entry(ies):"]
        for e in entries[:10]:
            d = e.get("date", "")
            has_content = e.get("has_content", False)
            content_preview = (e.get("content", "") or "")[:100].replace("\n", " ")
            if len((e.get("content") or "")) > 100:
                content_preview += "..."
            parts.append(f"- {d}: {'(empty)' if not has_content else content_preview}")
        if total > 10:
            parts.append(f"... and {total - 10} more.")
        return {
            "success": True,
            "entries": entries,
            "total": total,
            "formatted": "\n".join(parts),
        }
    except Exception as e:
        logger.error("get_journal_entries_tool error: %s", e)
        err = str(e)
        return {"success": False, "entries": [], "total": 0, "formatted": f"Error: {err}"}


register_action(
    name="get_journal_entries",
    category="org",
    description="Get full content of journal entries in a date range. Use for review or summarization.",
    inputs_model=GetJournalEntriesInputs,
    params_model=GetJournalEntriesParams,
    outputs_model=GetJournalEntriesOutputs,
    tool_function=get_journal_entries_tool,
)


class ListJournalEntriesParams(BaseModel):
    """Optional parameters for list_journal_entries."""
    start_date: Optional[str] = Field(default=None, description="Start of range (YYYY-MM-DD)")
    end_date: Optional[str] = Field(default=None, description="End of range (YYYY-MM-DD)")


class ListJournalEntriesInputs(BaseModel):
    """No required inputs; use params for date range."""
    pass


class ListJournalEntriesOutputs(BaseModel):
    """Structured output from list_journal_entries."""
    success: bool = Field(description="Whether the read succeeded")
    entries: List[Dict[str, Any]] = Field(
        description="List of {date, word_count, has_content} for each entry"
    )
    total: int = Field(description="Number of entries")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def list_journal_entries_tool(
    user_id: str = "system",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    List journal entries in a date range with metadata (word_count, has_content). No full content.
    """
    try:
        client = await get_backend_tool_client()
        result = await client.list_journal_entries(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
        )
        if not result.get("success"):
            err = result.get("error", "Failed to list journal entries.")
            return {"success": False, "entries": [], "total": 0, "formatted": err}
        entries = result.get("entries", [])
        total = result.get("total", 0)
        parts = [f"Journal entries in range: {total} total."]
        for e in entries[:15]:
            d = e.get("date", "")
            wc = e.get("word_count", 0)
            has_content = e.get("has_content", False)
            parts.append(f"- {d}: {wc} words" + (" (has content)" if has_content else " (empty)"))
        if total > 15:
            parts.append(f"... and {total - 15} more.")
        return {
            "success": True,
            "entries": entries,
            "total": total,
            "formatted": "\n".join(parts),
        }
    except Exception as e:
        logger.error("list_journal_entries_tool error: %s", e)
        err = str(e)
        return {"success": False, "entries": [], "total": 0, "formatted": f"Error: {err}"}


register_action(
    name="list_journal_entries",
    category="org",
    description="List journal entries in a date range with metadata (word count, has_content). Does not return full content.",
    inputs_model=ListJournalEntriesInputs,
    params_model=ListJournalEntriesParams,
    outputs_model=ListJournalEntriesOutputs,
    tool_function=list_journal_entries_tool,
)


class SearchJournalInputs(BaseModel):
    """Required inputs for searching journal."""
    query: str = Field(description="Search query (keyword or phrase to find in journal content)")


class SearchJournalParams(BaseModel):
    """Optional parameters for search_journal."""
    start_date: Optional[str] = Field(default=None, description="Start of range (YYYY-MM-DD)")
    end_date: Optional[str] = Field(default=None, description="End of range (YYYY-MM-DD)")


class SearchJournalOutputs(BaseModel):
    """Structured output from search_journal."""
    success: bool = Field(description="Whether the search succeeded")
    results: List[Dict[str, str]] = Field(
        description="List of {date, excerpt} for each matching entry"
    )
    count: int = Field(description="Number of matches")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def search_journal_tool(
    query: str = "",
    user_id: str = "system",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Search within journal entry content in a date range. Returns list of {date, excerpt}.
    """
    try:
        if not query or not query.strip():
            msg = "Error: query is required for journal search."
            return {"success": False, "results": [], "count": 0, "formatted": msg}
        client = await get_backend_tool_client()
        result = await client.search_journal(
            user_id=user_id,
            query=query.strip(),
            start_date=start_date,
            end_date=end_date,
        )
        if not result.get("success"):
            err = result.get("error", "Failed to search journal.")
            return {"success": False, "results": [], "count": 0, "formatted": err}
        results = result.get("results", [])
        count = result.get("count", 0)
        parts = [f"Found {count} journal entry(ies) matching '{query.strip()}':"]
        for r in results[:10]:
            ex = r.get("excerpt", "")
            parts.append(f"- {r.get('date', '')}: {ex[:150]}{'...' if len(ex) > 150 else ''}")
        if count > 10:
            parts.append(f"... and {count - 10} more.")
        return {
            "success": True,
            "results": results,
            "count": count,
            "formatted": "\n".join(parts),
        }
    except Exception as e:
        logger.error("search_journal_tool error: %s", e)
        err = str(e)
        return {"success": False, "results": [], "count": 0, "formatted": f"Error: {err}"}


register_action(
    name="search_journal",
    category="org",
    description="Search within journal entry content in a date range. Returns matching dates and excerpts.",
    inputs_model=SearchJournalInputs,
    params_model=SearchJournalParams,
    outputs_model=SearchJournalOutputs,
    tool_function=search_journal_tool,
)

