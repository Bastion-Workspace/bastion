"""
Editor navigation tools - Format-agnostic section browsing and search.

Local implementations (Zone 1). Operate on editor content injected at runtime
via _editor_content and _pipeline_metadata.  Used by the editor-navigation skill
to let agents dynamically pull sections, search content, and navigate reference
files from the active editor beyond the static cursor-based context window.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.utils.action_io_registry import register_action
from orchestrator.utils.frontmatter_utils import frontmatter_end_index
from orchestrator.utils.section_scoping import (
    SectionRange,
    find_heading_sections,
    _heading_plain_text,
)
from orchestrator.utils.org_utils import parse_org_structure

logger = logging.getLogger(__name__)

_NO_CONTENT_MSG = "No editor content available. Open a file and try again."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_editor_format(pipeline_metadata: Optional[Dict] = None) -> str:
    """Return 'org' or 'markdown' based on active editor filename."""
    editor = (pipeline_metadata or {}).get("shared_memory", {}).get("active_editor", {})
    filename = (editor.get("filename") or "").lower()
    if filename.endswith(".org"):
        return "org"
    return "markdown"


def _parse_sections_normalized(
    text: str,
    max_level: int = 3,
    fmt: str = "markdown",
) -> List[Dict[str, Any]]:
    """
    Parse heading sections into a uniform list regardless of format.

    Each entry: {index, heading, level, char_count, start, end}.
    """
    if not text:
        return []

    if fmt == "org":
        org = parse_org_structure(text)
        sections: List[Dict[str, Any]] = []
        for idx, h in enumerate(org):
            start = h.get("start_position", 0)
            end = h.get("end_position", len(text))
            sections.append({
                "index": idx,
                "heading": h.get("heading", ""),
                "level": h.get("level", 1),
                "char_count": end - start,
                "start": start,
                "end": end,
            })
        return sections

    ranges = find_heading_sections(text, max_level=max_level)
    return [
        {
            "index": idx,
            "heading": r.heading_text,
            "level": r.level,
            "char_count": r.end - r.start,
            "start": r.start,
            "end": r.end,
        }
        for idx, r in enumerate(ranges)
    ]


def _match_section_by_heading(
    sections: List[Dict[str, Any]],
    heading_query: str,
) -> Optional[Dict[str, Any]]:
    """Case-insensitive substring match on plain heading text."""
    q = heading_query.strip().lower()
    if not q:
        return None
    for s in sections:
        plain = _heading_plain_text(s["heading"]).lower()
        if q in plain:
            return s
    return None


def _section_content(text: str, section: Dict[str, Any]) -> str:
    return text[section["start"]:section["end"]].strip()


def _find_section(
    sections: List[Dict[str, Any]],
    heading: str = "",
    index: int = -1,
) -> Optional[Dict[str, Any]]:
    """Find a section by heading substring or index."""
    if heading:
        return _match_section_by_heading(sections, heading)
    if 0 <= index < len(sections):
        return sections[index]
    return None


# ---------------------------------------------------------------------------
# I/O Models
# ---------------------------------------------------------------------------

class ListSectionsInputs(BaseModel):
    """Inputs for editor_list_sections."""
    max_level: int = Field(default=3, description="Heading depth (1-6). Default 3.")


class ListSectionsOutputs(BaseModel):
    """Outputs for editor_list_sections."""
    sections: List[Dict[str, Any]] = Field(description="Section list with heading, level, char_count")
    total_sections: int = Field(description="Number of sections")
    total_chars: int = Field(description="Total document characters")
    formatted: str = Field(description="Human-readable TOC with char counts")


class GetSectionInputs(BaseModel):
    """Inputs for editor_get_section."""
    heading: str = Field(default="", description="Heading text to search for (case-insensitive substring match)")
    index: int = Field(default=-1, description="Section index (0-based). Use heading OR index.")
    include_adjacent: bool = Field(default=False, description="Also return previous and next sections")
    max_level: int = Field(default=3, description="Heading depth (1-6)")


class GetSectionOutputs(BaseModel):
    """Outputs for editor_get_section."""
    heading: str = Field(description="Matched heading line")
    content: str = Field(description="Full section content")
    index: int = Field(description="Section index")
    char_count: int = Field(description="Section character count")
    previous_heading: str = Field(default="", description="Previous section heading")
    next_heading: str = Field(default="", description="Next section heading")
    previous_content: str = Field(default="", description="Previous section content (when include_adjacent)")
    next_content: str = Field(default="", description="Next section content (when include_adjacent)")
    formatted: str = Field(description="Human-readable summary")


class GetSectionsInputs(BaseModel):
    """Inputs for editor_get_sections (batch)."""
    headings: Optional[List[str]] = Field(default=None, description="Heading substrings to find")
    indices: Optional[List[int]] = Field(default=None, description="Section indices to retrieve")
    max_level: int = Field(default=3, description="Heading depth (1-6)")


class GetSectionsOutputs(BaseModel):
    """Outputs for editor_get_sections."""
    sections: List[Dict[str, Any]] = Field(description="Retrieved sections")
    total_chars: int = Field(description="Combined character count")
    formatted: str = Field(description="Human-readable summary")


class SearchContentInputs(BaseModel):
    """Inputs for editor_search_content."""
    query: str = Field(description="Text to search for (case-insensitive)")
    max_results: int = Field(default=10, description="Maximum matches to return")


class SearchContentOutputs(BaseModel):
    """Outputs for editor_search_content."""
    matches: List[Dict[str, Any]] = Field(description="Match objects with context")
    total_matches: int = Field(description="Total number of matches found")
    sections_with_matches: List[str] = Field(description="Unique section headings containing matches")
    formatted: str = Field(description="Human-readable summary")


class GetRefSectionInputs(BaseModel):
    """Inputs for editor_get_ref_section."""
    ref_category: str = Field(description="Reference category (e.g. 'outline', 'rules', 'style')")
    heading: str = Field(default="", description="Heading text to find (case-insensitive substring)")
    index: int = Field(default=-1, description="Section index (0-based). Use heading OR index.")
    max_level: int = Field(default=3, description="Heading depth (1-6)")


class GetRefSectionOutputs(BaseModel):
    """Outputs for editor_get_ref_section."""
    heading: str = Field(description="Matched heading line")
    content: str = Field(description="Full section content")
    index: int = Field(description="Section index in the reference file")
    char_count: int = Field(description="Section character count")
    ref_category: str = Field(description="Reference category used")
    formatted: str = Field(description="Human-readable summary")


# ---------------------------------------------------------------------------
# Tool Functions
# ---------------------------------------------------------------------------

def editor_list_sections_tool(
    max_level: int = 3,
    _editor_content: str = "",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    List all heading-based sections in the active editor document.
    Returns a table of contents with heading text, level, and character count per section.
    Use this to see the document structure before pulling specific sections.
    """
    if not (_editor_content or "").strip():
        return {"sections": [], "total_sections": 0, "total_chars": 0, "formatted": _NO_CONTENT_MSG}
    try:
        fmt = _detect_editor_format(_pipeline_metadata)
        sections = _parse_sections_normalized(_editor_content, max_level=max_level, fmt=fmt)
        total_chars = len(_editor_content)
        fm_end = frontmatter_end_index(_editor_content)
        body_after_fm = _editor_content[fm_end:].strip()
        if not sections and not body_after_fm and fm_end > 0:
            formatted = (
                f"0 section(s), {total_chars:,} chars total.\n\n"
                f"Document body is empty (frontmatter only, {total_chars:,} chars). "
                "Use append or insert_after_heading with patch_file to add initial content."
            )
            return {
                "sections": sections,
                "total_sections": len(sections),
                "total_chars": total_chars,
                "formatted": formatted,
            }
        lines = [f"{len(sections)} section(s), {total_chars:,} chars total:"]
        for s in sections:
            lines.append(f"  {s['index']:>3}. {s['heading']}  ({s['char_count']:,} chars)")
        return {
            "sections": sections,
            "total_sections": len(sections),
            "total_chars": total_chars,
            "formatted": "\n".join(lines),
        }
    except Exception as e:
        logger.exception("editor_list_sections_tool failed")
        return {"sections": [], "total_sections": 0, "total_chars": 0, "formatted": f"Error: {e}"}


def editor_get_section_tool(
    heading: str = "",
    index: int = -1,
    include_adjacent: bool = False,
    max_level: int = 3,
    _editor_content: str = "",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Retrieve a single section by heading name (substring match) or index.
    Optionally includes the previous and next sections for surrounding context.
    """
    empty: Dict[str, Any] = {
        "heading": "", "content": "", "index": -1, "char_count": 0,
        "previous_heading": "", "next_heading": "",
        "previous_content": "", "next_content": "",
        "formatted": "",
    }
    if not (_editor_content or "").strip():
        return {**empty, "formatted": _NO_CONTENT_MSG}
    if not heading and index < 0:
        return {**empty, "formatted": "Provide heading (substring) or index (>= 0)."}
    try:
        fmt = _detect_editor_format(_pipeline_metadata)
        sections = _parse_sections_normalized(_editor_content, max_level=max_level, fmt=fmt)
        if not sections:
            return {**empty, "formatted": "No sections found in document."}
        match = _find_section(sections, heading=heading, index=index)
        if not match:
            target = heading if heading else f"index {index}"
            return {**empty, "formatted": f"Section not found: {target}"}
        idx = match["index"]
        content = _section_content(_editor_content, match)
        prev_heading = sections[idx - 1]["heading"] if idx > 0 else ""
        next_heading = sections[idx + 1]["heading"] if idx + 1 < len(sections) else ""
        prev_content = ""
        next_content = ""
        if include_adjacent:
            if idx > 0:
                prev_content = _section_content(_editor_content, sections[idx - 1])
            if idx + 1 < len(sections):
                next_content = _section_content(_editor_content, sections[idx + 1])
        fmt_parts = [f"{match['heading']}  ({match['char_count']:,} chars, index {idx})", "", content]
        return {
            "heading": match["heading"],
            "content": content,
            "index": idx,
            "char_count": match["char_count"],
            "previous_heading": prev_heading,
            "next_heading": next_heading,
            "previous_content": prev_content,
            "next_content": next_content,
            "formatted": "\n".join(fmt_parts),
        }
    except Exception as e:
        logger.exception("editor_get_section_tool failed")
        return {**empty, "formatted": f"Error: {e}"}


def editor_get_sections_tool(
    headings: Optional[List[str]] = None,
    indices: Optional[List[int]] = None,
    max_level: int = 3,
    _editor_content: str = "",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Retrieve multiple sections at once by heading names and/or indices.
    More efficient than multiple single-section calls.
    """
    empty: Dict[str, Any] = {"sections": [], "total_chars": 0, "formatted": ""}
    if not (_editor_content or "").strip():
        return {**empty, "formatted": _NO_CONTENT_MSG}
    if not headings and not indices:
        return {**empty, "formatted": "Provide at least one heading or index."}
    try:
        fmt = _detect_editor_format(_pipeline_metadata)
        all_sections = _parse_sections_normalized(_editor_content, max_level=max_level, fmt=fmt)
        if not all_sections:
            return {**empty, "formatted": "No sections found in document."}

        found: List[Dict[str, Any]] = []
        seen_indices: set = set()

        for h in (headings or []):
            match = _match_section_by_heading(all_sections, h)
            if match and match["index"] not in seen_indices:
                seen_indices.add(match["index"])
                found.append({
                    "heading": match["heading"],
                    "content": _section_content(_editor_content, match),
                    "index": match["index"],
                    "char_count": match["char_count"],
                })
        for i in (indices or []):
            if 0 <= i < len(all_sections) and i not in seen_indices:
                seen_indices.add(i)
                s = all_sections[i]
                found.append({
                    "heading": s["heading"],
                    "content": _section_content(_editor_content, s),
                    "index": s["index"],
                    "char_count": s["char_count"],
                })

        found.sort(key=lambda x: x["index"])
        total_chars = sum(s["char_count"] for s in found)
        lines = [f"Retrieved {len(found)} section(s), {total_chars:,} chars total:"]
        for s in found:
            lines.append(f"\n--- {s['heading']} (index {s['index']}, {s['char_count']:,} chars) ---")
            lines.append(s["content"])
        return {
            "sections": found,
            "total_chars": total_chars,
            "formatted": "\n".join(lines),
        }
    except Exception as e:
        logger.exception("editor_get_sections_tool failed")
        return {**empty, "formatted": f"Error: {e}"}


def editor_search_content_tool(
    query: str = "",
    max_results: int = 10,
    _editor_content: str = "",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Search for a term or phrase in the active editor document (case-insensitive).
    Returns matching snippets with surrounding context and the section they appear in.
    """
    empty: Dict[str, Any] = {
        "matches": [], "total_matches": 0,
        "sections_with_matches": [], "formatted": "",
    }
    if not (_editor_content or "").strip():
        return {**empty, "formatted": _NO_CONTENT_MSG}
    if not (query or "").strip():
        return {**empty, "formatted": "query is required."}
    try:
        fmt = _detect_editor_format(_pipeline_metadata)
        sections = _parse_sections_normalized(_editor_content, max_level=3, fmt=fmt)

        content_lower = _editor_content.lower()
        query_lower = query.strip().lower()
        context_window = 200

        matches: List[Dict[str, Any]] = []
        seen: set = set()
        start_pos = 0
        while True:
            pos = content_lower.find(query_lower, start_pos)
            if pos < 0:
                break
            end = pos + len(query_lower)
            if (pos, end) not in seen:
                seen.add((pos, end))
                ctx_before = _editor_content[max(0, pos - context_window):pos]
                ctx_after = _editor_content[end:end + context_window]
                sec_heading = ""
                sec_index = -1
                for s in sections:
                    if s["start"] <= pos < s["end"]:
                        sec_heading = s["heading"]
                        sec_index = s["index"]
                        break
                matches.append({
                    "match_text": _editor_content[pos:end],
                    "context_before": ctx_before,
                    "context_after": ctx_after,
                    "section_heading": sec_heading,
                    "section_index": sec_index,
                    "char_offset": pos,
                })
            start_pos = end
            if len(matches) >= max_results:
                break

        sec_set: List[str] = list(dict.fromkeys(
            m["section_heading"] for m in matches if m["section_heading"]
        ))
        lines = [f"Found {len(matches)} match(es) for \"{query}\":"]
        for i, m in enumerate(matches, 1):
            sec_label = f" in {m['section_heading']}" if m["section_heading"] else ""
            preview = (m["context_before"][-60:] + ">>>" + m["match_text"] + "<<<" + m["context_after"][:60]).replace("\n", " ")
            lines.append(f"  {i}. offset {m['char_offset']}{sec_label}")
            lines.append(f"     ...{preview}...")
        return {
            "matches": matches,
            "total_matches": len(matches),
            "sections_with_matches": sec_set,
            "formatted": "\n".join(lines),
        }
    except Exception as e:
        logger.exception("editor_search_content_tool failed")
        return {**empty, "formatted": f"Error: {e}"}


async def editor_get_ref_section_tool(
    ref_category: str = "",
    heading: str = "",
    index: int = -1,
    max_level: int = 3,
    _editor_content: str = "",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
    user_id: str = "system",
) -> Dict[str, Any]:
    """
    Retrieve a section from a referenced file (ref_* in frontmatter).
    ref_category is the part after 'ref_' (e.g. 'outline', 'rules', 'style').
    Finds the ref path from frontmatter, loads the file, then extracts the section.
    """
    empty: Dict[str, Any] = {
        "heading": "", "content": "", "index": -1, "char_count": 0,
        "ref_category": ref_category, "formatted": "",
    }
    if not (ref_category or "").strip():
        return {**empty, "formatted": "ref_category is required (e.g. 'outline', 'rules', 'style')."}
    if not heading and index < 0:
        return {**empty, "formatted": "Provide heading (substring) or index (>= 0)."}

    try:
        editor = (_pipeline_metadata or {}).get("shared_memory", {}).get("active_editor", {})
        frontmatter = editor.get("frontmatter") or {}

        ref_key = f"ref_{ref_category.strip()}"
        ref_path = frontmatter.get(ref_key)
        if not ref_path:
            available = [k for k in frontmatter if k.startswith("ref_")]
            avail_str = ", ".join(available) if available else "none"
            return {**empty, "formatted": f"No '{ref_key}' in frontmatter. Available ref keys: {avail_str}"}

        if isinstance(ref_path, list):
            ref_path = ref_path[0] if ref_path else ""
        ref_path = str(ref_path).strip()
        if not ref_path:
            return {**empty, "formatted": f"'{ref_key}' is empty in frontmatter."}

        from orchestrator.tools.reference_file_loader import load_file_by_path
        result = await load_file_by_path(
            ref_path=ref_path,
            user_id=user_id,
            active_editor=editor,
        )
        if not result or not result.get("found"):
            return {**empty, "formatted": f"Could not load reference file: {ref_path}"}

        ref_content = result.get("content") or ""
        if not ref_content.strip():
            return {**empty, "formatted": f"Reference file '{ref_path}' is empty."}

        ref_filename = (result.get("filename") or ref_path).lower()
        ref_fmt = "org" if ref_filename.endswith(".org") else "markdown"
        sections = _parse_sections_normalized(ref_content, max_level=max_level, fmt=ref_fmt)
        if not sections:
            return {**empty, "formatted": f"No sections found in reference file '{ref_path}'."}

        match = _find_section(sections, heading=heading, index=index)
        if not match:
            target = heading if heading else f"index {index}"
            toc = "\n".join(f"  {s['index']}. {s['heading']}" for s in sections)
            return {**empty, "formatted": f"Section not found: {target}\n\nAvailable sections:\n{toc}"}

        content = ref_content[match["start"]:match["end"]].strip()
        return {
            "heading": match["heading"],
            "content": content,
            "index": match["index"],
            "char_count": match["char_count"],
            "ref_category": ref_category.strip(),
            "formatted": f"[ref_{ref_category}] {match['heading']} ({match['char_count']:,} chars)\n\n{content}",
        }
    except Exception as e:
        logger.exception("editor_get_ref_section_tool failed")
        return {**empty, "formatted": f"Error: {e}"}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

register_action(
    name="editor_list_sections",
    category="editor",
    description="List all heading-based sections in the active editor (TOC with char counts)",
    inputs_model=ListSectionsInputs,
    outputs_model=ListSectionsOutputs,
    tool_function=editor_list_sections_tool,
)
register_action(
    name="editor_get_section",
    category="editor",
    description="Retrieve a single section from the active editor by heading name or index",
    inputs_model=GetSectionInputs,
    outputs_model=GetSectionOutputs,
    tool_function=editor_get_section_tool,
)
register_action(
    name="editor_get_sections",
    category="editor",
    description="Retrieve multiple sections at once by heading names and/or indices (batch)",
    inputs_model=GetSectionsInputs,
    outputs_model=GetSectionsOutputs,
    tool_function=editor_get_sections_tool,
)
register_action(
    name="editor_search_content",
    category="editor",
    description="Search for a term or phrase in the active editor document with context",
    inputs_model=SearchContentInputs,
    outputs_model=SearchContentOutputs,
    tool_function=editor_search_content_tool,
)
register_action(
    name="editor_get_ref_section",
    category="editor",
    description="Retrieve a section from a referenced file (ref_* in frontmatter) by heading or index",
    inputs_model=GetRefSectionInputs,
    outputs_model=GetRefSectionOutputs,
    tool_function=editor_get_ref_section_tool,
)
