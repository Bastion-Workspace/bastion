"""
Tool Pack Registry - Planner-oriented groupings of tools for plan step augmentation.

Tool packs are named groups of tool function names. The planner can attach pack names
to plan steps; the automation engine merges pack tools with the skill's tools at execution.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ToolPack(BaseModel):
    """A named group of tools that can be attached to a plan step for augmentation."""

    name: str
    description: str = Field(description="For LLM planner context")
    tools: List[str] = Field(description="Function names in orchestrator.tools")


TOOL_PACKS: Dict[str, ToolPack] = {
    "text_transforms": ToolPack(
        name="text_transforms",
        description="Text manipulation: summarize, extract, format conversion, merge, compare",
        tools=[
            "summarize_text_tool",
            "extract_structured_data_tool",
            "transform_format_tool",
            "merge_texts_tool",
            "compare_texts_tool",
        ],
    ),
    "session_memory": ToolPack(
        name="session_memory",
        description="Ephemeral clipboard for passing data between plan steps",
        tools=["clipboard_store_tool", "clipboard_get_tool"],
    ),
    "discovery": ToolPack(
        name="discovery",
        description="Search documents and web for information",
        tools=[
            "search_documents_tool",
            "search_web_tool",
            "expand_query_tool",
            "search_conversation_cache_tool",
        ],
    ),
    "knowledge": ToolPack(
        name="knowledge",
        description="Retrieve and extract content from specific documents",
        tools=[
            "get_document_content_tool",
            "search_within_document_tool",
            "search_segments_across_documents_tool",
        ],
    ),
    "document_management": ToolPack(
        name="document_management",
        description="Create, append to, and read metadata of documents",
        tools=[
            "create_document_tool",
            "append_to_document_tool",
            "get_document_metadata_tool",
        ],
    ),
    "file_management": ToolPack(
        name="file_management",
        description="Create and organize files and folders",
        tools=["list_folders_tool", "create_user_file_tool", "create_user_folder_tool"],
    ),
    "org_management": ToolPack(
        name="org_management",
        description="Org-mode inbox and task management",
        tools=[
            "add_org_inbox_item_tool",
            "list_org_todos_tool",
            "parse_org_structure_tool",
            "search_org_headings_tool",
        ],
    ),
    "math": ToolPack(
        name="math",
        description="Math calculations, formulas, and unit conversions",
        tools=[
            "calculate_expression_tool",
            "evaluate_formula_tool",
            "convert_units_tool",
        ],
    ),
}


def get_all_packs() -> List[ToolPack]:
    """Return all tool packs."""
    return list(TOOL_PACKS.values())


def get_pack(name: str) -> Optional[ToolPack]:
    """Return the pack with the given name, or None."""
    return TOOL_PACKS.get(name)


def resolve_pack_tools(pack_names: List[str]) -> List[str]:
    """
    Resolve one or more pack names to a deduplicated list of tool names.
    Order is preserved (first pack's tools, then second, etc.), with duplicates removed.
    """
    seen: set = set()
    result: List[str] = []
    for pack_name in pack_names:
        pack = TOOL_PACKS.get(pack_name)
        if not pack:
            continue
        for tool_name in pack.tools:
            if tool_name not in seen:
                seen.add(tool_name)
                result.append(tool_name)
    return result
