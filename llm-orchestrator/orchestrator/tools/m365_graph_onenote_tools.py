"""Microsoft OneNote tools via Graph (connection-scoped)."""

import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.tools.m365_invoke_common import invoke_m365_graph
from orchestrator.utils.action_io_registry import register_action


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _format_notebooks(notebooks: List[Dict[str, Any]]) -> str:
    if not notebooks:
        return "OneNote: no notebooks found."
    lines = [
        "OneNote notebooks. Use the id values below for list_onenote_sections.",
        f"count: {len(notebooks)}",
        "",
    ]
    for i, nb in enumerate(notebooks, 1):
        lines.append(f'{i}. "{nb.get("display_name", "")}"')
        lines.append(f"   id: {nb.get('id', '')}")
        if nb.get("web_url"):
            lines.append(f"   web_url: {nb['web_url']}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _format_sections(sections: List[Dict[str, Any]], notebook_id: str) -> str:
    if not sections:
        return f"OneNote: no sections in notebook {notebook_id}."
    lines = [
        "OneNote sections. Use the id values below for list_onenote_pages.",
        f"notebook_id: {notebook_id}",
        f"count: {len(sections)}",
        "",
    ]
    for i, sec in enumerate(sections, 1):
        lines.append(f'{i}. "{sec.get("display_name", "")}"')
        lines.append(f"   id: {sec.get('id', '')}")
        if sec.get("web_url"):
            lines.append(f"   web_url: {sec['web_url']}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _format_pages(pages: List[Dict[str, Any]], section_id: str) -> str:
    if not pages:
        return f"OneNote: no pages in section {section_id}."
    lines = [
        "OneNote pages. Use the id values below for get_onenote_page_content.",
        f"section_id: {section_id}",
        f"count: {len(pages)}",
        "",
    ]
    for i, pg in enumerate(pages, 1):
        title = pg.get("title") or "(untitled)"
        lines.append(f'{i}. "{title}"')
        lines.append(f"   id: {pg.get('id', '')}")
        if pg.get("created_time"):
            lines.append(f"   created: {pg['created_time']}")
        lines.append("")
    return "\n".join(lines).rstrip()


_TAG_RE = re.compile(r"<[^>]+>")


def _format_page_content(html: str) -> str:
    if not html:
        return "OneNote page: empty content."
    plain = _TAG_RE.sub("", html).strip()
    preview = plain[:800] + ("…" if len(plain) > 800 else "")
    return (
        f"OneNote page content (HTML {len(html)} chars).\n"
        f"Text preview:\n{preview}"
    )


def _format_page_mutation(success: bool, page_id: str, error: str, title: str = "") -> str:
    if not success:
        return f"OneNote create_page failed: {error or 'unknown error'}"
    parts = ["OneNote page created."]
    if title:
        parts.append(f'  title: "{title}"')
    if page_id:
        parts.append(f"  page_id: {page_id}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Input / Output models
# ---------------------------------------------------------------------------

class ConnParams(BaseModel):
    connection_id: Optional[int] = Field(default=None, description="Microsoft 365 connection id")


class NotebooksOutputs(BaseModel):
    notebooks: List[Dict[str, Any]] = Field(default_factory=list)
    formatted: str = Field(description="Human-readable summary")


class ListOnenoteSectionsInputs(ConnParams):
    notebook_id: str = Field(
        description="Notebook id from list_onenote_notebooks (the long id string, not the display name)"
    )


class SectionsOutputs(BaseModel):
    sections: List[Dict[str, Any]] = Field(default_factory=list)
    formatted: str = Field(description="Human-readable summary")


class ListOnenotePagesInputs(ConnParams):
    section_id: str = Field(
        description="Section id from list_onenote_sections (the long id string, not the display name)"
    )
    top: int = Field(default=50, ge=1, le=200, description="Maximum pages to return (1-200)")


class PagesOutputs(BaseModel):
    pages: List[Dict[str, Any]] = Field(default_factory=list)
    formatted: str = Field(description="Human-readable summary")


class GetOnenotePageContentInputs(ConnParams):
    page_id: str = Field(
        description="Page id from list_onenote_pages (the long id string, not the page title)"
    )


class PageContentOutputs(BaseModel):
    html_content: str = Field(default="")
    formatted: str = Field(description="Human-readable summary")


class CreateOnenotePageInputs(ConnParams):
    section_id: str = Field(
        description="Section id from list_onenote_sections (the long id string, not the section name)"
    )
    html: str = Field(description="HTML body for the page")
    title: str = Field(default="", description="Optional title for the page")


class PageMutationOutputs(BaseModel):
    page_id: str = Field(default="")
    success: bool = Field(default=True)
    formatted: str = Field(description="Human-readable summary")


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

async def list_onenote_notebooks_tool(
    user_id: str = "system",
    connection_id: Optional[int] = None,
) -> Dict[str, Any]:
    out = await invoke_m365_graph("list_onenote_notebooks", user_id, connection_id, {})
    nbs = out.get("notebooks") or []
    nbs = nbs if isinstance(nbs, list) else []
    err = out.get("error")
    body = _format_notebooks(nbs)
    return {
        "notebooks": nbs,
        "formatted": (f"Error: {err}\n\n" if err else "") + body,
    }


async def list_onenote_sections_tool(
    user_id: str = "system",
    notebook_id: str = "",
    connection_id: Optional[int] = None,
) -> Dict[str, Any]:
    out = await invoke_m365_graph(
        "list_onenote_sections",
        user_id,
        connection_id,
        {"notebook_id": notebook_id},
    )
    sec = out.get("sections") or []
    sec = sec if isinstance(sec, list) else []
    err = out.get("error")
    body = _format_sections(sec, notebook_id)
    return {
        "sections": sec,
        "formatted": (f"Error: {err}\n\n" if err else "") + body,
    }


async def list_onenote_pages_tool(
    user_id: str = "system",
    section_id: str = "",
    top: int = 50,
    connection_id: Optional[int] = None,
) -> Dict[str, Any]:
    out = await invoke_m365_graph(
        "list_onenote_pages",
        user_id,
        connection_id,
        {"section_id": section_id, "top": top},
    )
    pages = out.get("pages") or []
    pages = pages if isinstance(pages, list) else []
    err = out.get("error")
    body = _format_pages(pages, section_id)
    return {
        "pages": pages,
        "formatted": (f"Error: {err}\n\n" if err else "") + body,
    }


async def get_onenote_page_content_tool(
    user_id: str = "system",
    page_id: str = "",
    connection_id: Optional[int] = None,
) -> Dict[str, Any]:
    out = await invoke_m365_graph(
        "get_onenote_page_content",
        user_id,
        connection_id,
        {"page_id": page_id},
    )
    html = out.get("html_content") or ""
    err = out.get("error")
    body = _format_page_content(html)
    return {
        "html_content": html,
        "formatted": (f"Error: {err}\n\n" if err else "") + body,
    }


async def create_onenote_page_tool(
    user_id: str = "system",
    section_id: str = "",
    html: str = "",
    title: str = "",
    connection_id: Optional[int] = None,
) -> Dict[str, Any]:
    out = await invoke_m365_graph(
        "create_onenote_page",
        user_id,
        connection_id,
        {"section_id": section_id, "html": html, "title": title},
    )
    ok = out.get("success", False)
    pid = out.get("page_id") or ""
    err = out.get("error") or ""
    return {
        "page_id": pid,
        "success": ok,
        "formatted": _format_page_mutation(ok, pid, err, title=title),
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

register_action(
    name="list_onenote_notebooks",
    category="notes",
    description="List OneNote notebooks",
    inputs_model=ConnParams,
    outputs_model=NotebooksOutputs,
    tool_function=list_onenote_notebooks_tool,
)
register_action(
    name="list_onenote_sections",
    category="notes",
    description="List sections in a OneNote notebook",
    inputs_model=ListOnenoteSectionsInputs,
    outputs_model=SectionsOutputs,
    tool_function=list_onenote_sections_tool,
)
register_action(
    name="list_onenote_pages",
    category="notes",
    description="List pages in a OneNote section",
    inputs_model=ListOnenotePagesInputs,
    outputs_model=PagesOutputs,
    tool_function=list_onenote_pages_tool,
)
register_action(
    name="get_onenote_page_content",
    category="notes",
    description="Get OneNote page HTML content",
    inputs_model=GetOnenotePageContentInputs,
    outputs_model=PageContentOutputs,
    tool_function=get_onenote_page_content_tool,
)
register_action(
    name="create_onenote_page",
    category="notes",
    description="Create a OneNote page in a section",
    inputs_model=CreateOnenotePageInputs,
    outputs_model=PageMutationOutputs,
    tool_function=create_onenote_page_tool,
)
