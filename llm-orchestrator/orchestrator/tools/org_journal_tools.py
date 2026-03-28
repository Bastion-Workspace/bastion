"""
Journal-specific tools: read, update, list, and search journal entries by date.

REQUIRED for journal processing: When the user asks to read, edit, add to, list, or
search their journal, you MUST use these tools (get_journal_entry, get_journal_entries,
update_journal_entry, capture_journal_entry, list_journal_entries, search_journal). Do NOT use generic
document tools (get_document_content, update_document_content, search_documents) on
journal files—they replace or read the whole file and can cause severe data loss.
These tools operate at the date-entry section level only.
"""
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)


# ---- get_journal_entry ----

class GetJournalEntryInputs(BaseModel):
    """Required inputs for reading a journal entry."""
    date: str = Field(description="Date of the entry: YYYY-MM-DD or 'today'")


class GetJournalEntryOutputs(BaseModel):
    """Structured output from get_journal_entry."""
    content: str = Field(description="Body content of the entry")
    date: str = Field(description="Entry date YYYY-MM-DD")
    heading: str = Field(description="Org heading line for the date")
    document_id: Optional[str] = Field(default=None, description="Document ID if resolved")
    file_path: Optional[str] = Field(default=None, description="Relative path to journal file")
    has_content: bool = Field(description="Whether the entry has any body content")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def get_journal_entry_tool(
    date: str = "today",
    user_id: str = "system",
) -> Dict[str, Any]:
    """
    Read one date's journal entry. Only that date's section is read; the rest of
    the file is not loaded. Use this instead of get_document_content when editing
    a specific journal day.
    """
    try:
        client = await get_backend_tool_client()
        result = await client.get_journal_entry(user_id=user_id, date=date or "today")
        if not result.get("success"):
            err = result.get("error", "Failed to read journal entry.")
            return {
                "content": "",
                "date": result.get("date", ""),
                "heading": "",
                "document_id": None,
                "file_path": None,
                "has_content": False,
                "formatted": err,
            }
        content = result.get("content", "")
        entry_date = result.get("date", "")
        heading = result.get("heading", "")
        has_content = result.get("has_content", False)
        if has_content:
            preview = content[:200].replace("\n", " ") + ("..." if len(content) > 200 else "")
            formatted = f"Journal entry for {entry_date}: {preview}"
        else:
            formatted = f"Journal entry for {entry_date}: (no content)"
        return {
            "content": content,
            "date": entry_date,
            "heading": heading,
            "document_id": result.get("document_id"),
            "file_path": result.get("file_path"),
            "has_content": has_content,
            "formatted": formatted,
        }
    except Exception as e:
        logger.exception("get_journal_entry_tool error: %s", e)
        err = str(e)
        return {
            "content": "",
            "date": "",
            "heading": "",
            "document_id": None,
            "file_path": None,
            "has_content": False,
            "formatted": f"Error: {err}",
        }


register_action(
    name="get_journal_entry",
    category="org",
    description=(
        "REQUIRED for reading journal entries. Read one date's journal entry (section-aware). "
        "Use this—not get_document_content—whenever you need to read or edit a journal day; "
        "generic document tools can cause data loss. Date: YYYY-MM-DD or 'today'."
    ),
    short_description="Read one date's journal entry",
    inputs_model=GetJournalEntryInputs,
    params_model=None,
    outputs_model=GetJournalEntryOutputs,
    tool_function=get_journal_entry_tool,
)


# ---- get_journal_entries (batch read for review / lookback) ----

class GetJournalEntriesInputs(BaseModel):
    """Inputs for reading a date range of journal entries (review/lookback)."""
    start_date: Optional[str] = Field(default=None, description="Start of range YYYY-MM-DD (default: 31 days ago)")
    end_date: Optional[str] = Field(default=None, description="End of range YYYY-MM-DD (default: today)")


class GetJournalEntriesParams(BaseModel):
    """Optional parameters."""
    max_entries: int = Field(default=100, description="Maximum number of entries to return (1-500, default 100)")


class JournalEntryWithContent(BaseModel):
    """One journal entry with full content."""
    date: str = Field(description="Entry date YYYY-MM-DD")
    content: str = Field(description="Full body content")
    heading: str = Field(description="Org heading line")
    has_content: bool = Field(description="Whether the entry has any body content")


class GetJournalEntriesOutputs(BaseModel):
    """Structured output from get_journal_entries."""
    entries: List[JournalEntryWithContent] = Field(description="Entries with full content (date, content, heading)")
    total: int = Field(description="Number of entries returned")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def get_journal_entries_tool(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_entries: int = 100,
    user_id: str = "system",
) -> Dict[str, Any]:
    """
    Read full content of multiple journal entries in a date range. Use for reviewing
    a series of dates, lookback (e.g. last 7 days), or all of a month (e.g. February).
    Returns entries with date, content, heading; capped by max_entries.
    """
    try:
        client = await get_backend_tool_client()
        result = await client.get_journal_entries(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            max_entries=max_entries,
        )
        if not result.get("success"):
            err = result.get("error", "Failed to read journal entries.")
            return {"entries": [], "total": 0, "formatted": err}
        raw = result.get("entries", [])
        entries = [JournalEntryWithContent(**e) for e in raw]
        total = result.get("total", 0)
        if not entries:
            formatted = "No journal entries in the requested date range."
        else:
            def _preview(ent: JournalEntryWithContent) -> str:
                text = (ent.content or "").replace("\n", " ").strip()
                if len(text) > 150:
                    return f"**{ent.date}** ({ent.heading or 'no heading'}): {text[:150]}..."
                return f"**{ent.date}** ({ent.heading or 'no heading'}): {text or '(no content)'}"
            parts = [_preview(e) for e in entries[:10]]
            formatted = f"Found {total} journal entries.\n\n" + "\n\n".join(parts)
            if total > 10:
                formatted += f"\n\n... and {total - 10} more entries."
        return {"entries": entries, "total": total, "formatted": formatted}
    except Exception as e:
        logger.exception("get_journal_entries_tool error: %s", e)
        return {"entries": [], "total": 0, "formatted": f"Error: {e}"}


register_action(
    name="get_journal_entries",
    category="org",
    description=(
        "REQUIRED for reading/reviewing a journal range or lookback. Get full content of journal entries "
        "for a date range in one call. Use for: 'review my journal', 'last 7 days', 'all of February', "
        "'show me this week'. Optional start_date, end_date (YYYY-MM-DD); optional max_entries (default 100, max 500). "
        "Do not use get_document_content or multiple get_journal_entry calls for ranges."
    ),
    short_description="Get journal entries for a date range",
    inputs_model=GetJournalEntriesInputs,
    params_model=GetJournalEntriesParams,
    outputs_model=GetJournalEntriesOutputs,
    tool_function=get_journal_entries_tool,
)


# ---- update_journal_entry ----

class UpdateJournalEntryInputs(BaseModel):
    """Required inputs for updating a journal entry."""
    date: str = Field(description="Entry date YYYY-MM-DD")
    content: str = Field(description="New body content (replace) or text to append")


class UpdateJournalEntryParams(BaseModel):
    """Optional parameters."""
    mode: str = Field(default="replace", description="'replace' to overwrite that day's body; 'append' to add after existing content")


class UpdateJournalEntryOutputs(BaseModel):
    """Structured output from update_journal_entry."""
    success: bool = Field(description="Whether the update succeeded")
    date: str = Field(description="Entry date that was updated")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def update_journal_entry_tool(
    date: str,
    content: str,
    mode: str = "replace",
    user_id: str = "system",
) -> Dict[str, Any]:
    """
    Update a single date's journal section only. Replaces or appends to that day's
    body without touching other dates. Use this instead of update_document_content
    to avoid overwriting the entire journal file.
    """
    try:
        if not date or not date.strip():
            return {"success": False, "date": "", "formatted": "Error: date is required (YYYY-MM-DD)."}
        if mode not in ("replace", "append"):
            return {"success": False, "date": date, "formatted": f"Error: mode must be 'replace' or 'append', got '{mode}'."}
        client = await get_backend_tool_client()
        result = await client.update_journal_entry(
            user_id=user_id,
            date=date.strip(),
            content=content or "",
            mode=mode,
        )
        if not result.get("success"):
            err = result.get("error", "Update failed.")
            return {"success": False, "date": result.get("date", date), "formatted": err}
        return {
            "success": True,
            "date": result.get("date", date),
            "formatted": f"Updated journal entry for {result.get('date', date)}.",
        }
    except Exception as e:
        logger.exception("update_journal_entry_tool error: %s", e)
        return {"success": False, "date": date, "formatted": f"Error: {e}"}


register_action(
    name="update_journal_entry",
    category="org",
    description=(
        "REQUIRED for editing journal entries. Replace or append to one date's journal section only. "
        "Use this—not update_document_content—whenever you edit existing journal content; "
        "generic document update replaces the entire file and causes severe data loss. Mode: 'replace' or 'append'."
    ),
    short_description="Replace or append to one date's journal entry",
    inputs_model=UpdateJournalEntryInputs,
    params_model=UpdateJournalEntryParams,
    outputs_model=UpdateJournalEntryOutputs,
    tool_function=update_journal_entry_tool,
)


# ---- list_journal_entries ----

class ListJournalEntriesInputs(BaseModel):
    """Inputs for listing journal entries (all optional for default range)."""
    start_date: Optional[str] = Field(default=None, description="Start of range YYYY-MM-DD")
    end_date: Optional[str] = Field(default=None, description="End of range YYYY-MM-DD")


class JournalEntryMeta(BaseModel):
    """Metadata for one journal entry."""
    date: str = Field(description="Entry date YYYY-MM-DD")
    word_count: int = Field(description="Word count of body")
    has_content: bool = Field(description="Whether the entry has any body content")


class ListJournalEntriesOutputs(BaseModel):
    """Structured output from list_journal_entries."""
    entries: List[JournalEntryMeta] = Field(description="List of entry metadata (date, word_count, has_content)")
    total: int = Field(description="Number of entries")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def list_journal_entries_tool(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    user_id: str = "system",
) -> Dict[str, Any]:
    """
    List journal entries in a date range with metadata (date, word_count, has_content).
    Does not return full content.
    """
    try:
        client = await get_backend_tool_client()
        result = await client.list_journal_entries(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
        )
        if not result.get("success"):
            err = result.get("error", "Failed to list entries.")
            return {"entries": [], "total": 0, "formatted": err}
        raw_entries = result.get("entries", [])
        entries = [JournalEntryMeta(**e) for e in raw_entries]
        total = result.get("total", 0)
        lines = [f"{e.date}: {e.word_count} words" for e in entries[:20]]
        if total > 20:
            lines.append(f"... and {total - 20} more")
        formatted = f"Found {total} journal entries.\n" + "\n".join(lines) if lines else f"Found {total} journal entries."
        return {"entries": entries, "total": total, "formatted": formatted}
    except Exception as e:
        logger.exception("list_journal_entries_tool error: %s", e)
        return {"entries": [], "total": 0, "formatted": f"Error: {e}"}


register_action(
    name="list_journal_entries",
    category="org",
    description=(
        "REQUIRED for listing journal entries. List journal entries in a date range with metadata "
        "(date, word_count, has_content). Use for discovering what dates have entries; "
        "do not use generic document search for journal files. Optional start_date, end_date (YYYY-MM-DD)."
    ),
    short_description="List journal entries in a date range",
    inputs_model=ListJournalEntriesInputs,
    params_model=None,
    outputs_model=ListJournalEntriesOutputs,
    tool_function=list_journal_entries_tool,
)


# ---- search_journal ----

class SearchJournalInputs(BaseModel):
    """Required inputs for searching journal entries."""
    query: str = Field(description="Search phrase (matched in entry body)")


class SearchJournalParams(BaseModel):
    """Optional parameters."""
    start_date: Optional[str] = Field(default=None, description="Start of range YYYY-MM-DD")
    end_date: Optional[str] = Field(default=None, description="End of range YYYY-MM-DD")


class JournalSearchResult(BaseModel):
    """One search hit."""
    date: str = Field(description="Entry date YYYY-MM-DD")
    excerpt: str = Field(description="Matching excerpt from the entry")


class SearchJournalOutputs(BaseModel):
    """Structured output from search_journal."""
    results: List[JournalSearchResult] = Field(description="List of matches with date and excerpt")
    count: int = Field(description="Number of matches")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def search_journal_tool(
    query: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    user_id: str = "system",
) -> Dict[str, Any]:
    """
    Search within journal entry content in a date range. Returns matching dates and excerpts.
    """
    try:
        if not query or not query.strip():
            return {"results": [], "count": 0, "formatted": "Error: query is required."}
        client = await get_backend_tool_client()
        result = await client.search_journal(
            user_id=user_id,
            query=query.strip(),
            start_date=start_date,
            end_date=end_date,
        )
        if not result.get("success"):
            err = result.get("error", "Search failed.")
            return {"results": [], "count": 0, "formatted": err}
        results = result.get("results", [])
        count = result.get("count", 0)
        lines = [f"{r['date']}: {r.get('excerpt', '')[:150]}..." for r in results[:10]]
        if count > 10:
            lines.append(f"... and {count - 10} more matches")
        formatted = f"Found {count} matching entries.\n" + "\n".join(lines) if lines else f"Found {count} matching entries."
        return {"results": results, "count": count, "formatted": formatted}
    except Exception as e:
        logger.exception("search_journal_tool error: %s", e)
        return {"results": [], "count": 0, "formatted": f"Error: {e}"}


register_action(
    name="search_journal",
    category="org",
    description=(
        "REQUIRED for searching journal content. Search within journal entry body text in a date range. "
        "Returns date and excerpt per match. Use this—not search_documents—for finding text in the journal; "
        "journal structure is date-based. Optional start_date, end_date (YYYY-MM-DD)."
    ),
    short_description="Search within journal entry text",
    inputs_model=SearchJournalInputs,
    params_model=SearchJournalParams,
    outputs_model=SearchJournalOutputs,
    tool_function=search_journal_tool,
)


JOURNAL_TOOLS = [
    get_journal_entry_tool,
    get_journal_entries_tool,
    update_journal_entry_tool,
    list_journal_entries_tool,
    search_journal_tool,
]
