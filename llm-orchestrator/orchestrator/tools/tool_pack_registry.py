"""
Tool Pack Registry - Conceptual organization of tools into "Expeditionary Kits"

This module defines Tool Packs (conceptual groupings) that help agents understand
which tools are related and should be loaded together for specific tasks.
"""

from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass


class ToolPack(str, Enum):
    """Tool Pack categories - conceptual groupings of related tools"""
    DISCOVERY = "discovery"
    KNOWLEDGE = "knowledge"
    ANALYTICAL = "analytical"
    ACTION_STRUCTURAL = "action_structural"
    ORG_MANAGEMENT = "org_management"
    ENGINEERING_VIZ = "engineering_viz"


@dataclass
class ToolMetadata:
    """Metadata for a single tool"""
    name: str
    description: str
    pack: ToolPack
    semantic_keywords: List[str]  # Keywords for semantic matching
    parameters: Optional[Dict] = None


# Tool Pack Definitions
TOOL_PACKS: Dict[ToolPack, List[ToolMetadata]] = {
    ToolPack.DISCOVERY: [
        ToolMetadata(
            name="search_documents_tool",
            description="Search local documents and knowledge base",
            pack=ToolPack.DISCOVERY,
            semantic_keywords=["search", "find", "documents", "local", "knowledge base", "lookup"]
        ),
        ToolMetadata(
            name="search_web_tool",
            description="Search the web for current information",
            pack=ToolPack.DISCOVERY,
            semantic_keywords=["web", "internet", "online", "current", "latest", "recent", "browse"]
        ),
        ToolMetadata(
            name="expand_query_tool",
            description="Generate alternative search queries for better recall",
            pack=ToolPack.DISCOVERY,
            semantic_keywords=["expand", "variations", "alternative", "query", "semantic"]
        ),
        ToolMetadata(
            name="search_conversation_cache_tool",
            description="Search previous conversation history for cached results",
            pack=ToolPack.DISCOVERY,
            semantic_keywords=["cache", "history", "previous", "conversation", "past", "remembered"]
        )
    ],
    
    ToolPack.KNOWLEDGE: [
        ToolMetadata(
            name="get_document_content_tool",
            description="Retrieve full document content by ID",
            pack=ToolPack.KNOWLEDGE,
            semantic_keywords=["get", "retrieve", "read", "content", "document", "full text"]
        ),
        ToolMetadata(
            name="search_within_document_tool",
            description="Search for specific content within a document",
            pack=ToolPack.KNOWLEDGE,
            semantic_keywords=["within", "inside", "document", "section", "part", "specific"]
        ),
        ToolMetadata(
            name="search_segments_across_documents_tool",
            description="Search for segments across multiple documents",
            pack=ToolPack.KNOWLEDGE,
            semantic_keywords=["segments", "across", "multiple", "cross-document", "sections"]
        ),
        ToolMetadata(
            name="extract_relevant_content_section",
            description="Extract relevant content sections from documents",
            pack=ToolPack.KNOWLEDGE,
            semantic_keywords=["extract", "relevant", "section", "portion", "excerpt"]
        )
    ],
    
    ToolPack.ANALYTICAL: [
        ToolMetadata(
            name="analyze_information_needs_tool",
            description="Analyze what information is needed to answer a query",
            pack=ToolPack.ANALYTICAL,
            semantic_keywords=["analyze", "information", "needs", "gap", "what do I need", "requirements"]
        ),
        ToolMetadata(
            name="generate_project_aware_queries_tool",
            description="Generate queries aware of project context",
            pack=ToolPack.ANALYTICAL,
            semantic_keywords=["project", "context", "aware", "generate", "queries", "intelligent"]
        ),
        ToolMetadata(
            name="calculate_expression_tool",
            description="Perform mathematical calculations",
            pack=ToolPack.ANALYTICAL,
            semantic_keywords=["calculate", "math", "compute", "expression", "arithmetic", "numbers"]
        ),
        ToolMetadata(
            name="evaluate_formula_tool",
            description="Evaluate mathematical formulas",
            pack=ToolPack.ANALYTICAL,
            semantic_keywords=["formula", "evaluate", "equation", "solve", "math"]
        ),
        ToolMetadata(
            name="convert_units_tool",
            description="Convert between different units of measurement",
            pack=ToolPack.ANALYTICAL,
            semantic_keywords=["convert", "units", "measurement", "transform", "change units"]
        )
    ],
    
    ToolPack.ACTION_STRUCTURAL: [
        ToolMetadata(
            name="create_user_file",
            description="Create a new file in user's documents",
            pack=ToolPack.ACTION_STRUCTURAL,
            semantic_keywords=["create", "file", "new", "write", "save", "document"]
        ),
        ToolMetadata(
            name="create_user_folder",
            description="Create a new folder in user's documents",
            pack=ToolPack.ACTION_STRUCTURAL,
            semantic_keywords=["create", "folder", "directory", "organize", "structure"]
        ),
        ToolMetadata(
            name="update_document_metadata",
            description="Update document metadata and frontmatter",
            pack=ToolPack.ACTION_STRUCTURAL,
            semantic_keywords=["update", "metadata", "frontmatter", "edit", "modify", "change"]
        ),
        ToolMetadata(
            name="send_room_message",
            description="Send a message to a chat room",
            pack=ToolPack.ACTION_STRUCTURAL,
            semantic_keywords=["send", "message", "chat", "room", "communicate", "notify"]
        ),
        ToolMetadata(
            name="get_user_rooms",
            description="Get list of user's chat rooms",
            pack=ToolPack.ACTION_STRUCTURAL,
            semantic_keywords=["rooms", "chat", "list", "get", "user", "conversations"]
        )
    ],
    
    ToolPack.ORG_MANAGEMENT: [
        ToolMetadata(
            name="org_inbox_list_items",
            description="List task items from org-mode inbox",
            pack=ToolPack.ORG_MANAGEMENT,
            semantic_keywords=["org", "inbox", "list", "tasks", "items", "todo"]
        ),
        ToolMetadata(
            name="org_inbox_add_item",
            description="Add a new item to org-mode inbox",
            pack=ToolPack.ORG_MANAGEMENT,
            semantic_keywords=["org", "inbox", "add", "create", "task", "item"]
        ),
        ToolMetadata(
            name="search_org_files",
            description="Search across all org-mode files",
            pack=ToolPack.ORG_MANAGEMENT,
            semantic_keywords=["org", "search", "files", "find", "org-mode"]
        ),
        ToolMetadata(
            name="list_org_todos",
            description="List all TODO items from org files",
            pack=ToolPack.ORG_MANAGEMENT,
            semantic_keywords=["org", "todo", "list", "tasks", "items"]
        ),
        ToolMetadata(
            name="search_org_by_tag",
            description="Search org files by tag",
            pack=ToolPack.ORG_MANAGEMENT,
            semantic_keywords=["org", "tag", "search", "filter", "find"]
        )
    ],
    
    ToolPack.ENGINEERING_VIZ: [
        ToolMetadata(
            name="create_chart_tool",
            description="Create charts and visualizations from data",
            pack=ToolPack.ENGINEERING_VIZ,
            semantic_keywords=["chart", "graph", "visualize", "plot", "diagram", "data visualization"]
        ),
        ToolMetadata(
            name="design_system_component_tool",
            description="Design system components",
            pack=ToolPack.ENGINEERING_VIZ,
            semantic_keywords=["design", "system", "component", "architecture", "model"]
        ),
        ToolMetadata(
            name="simulate_system_failure_tool",
            description="Simulate system failure scenarios",
            pack=ToolPack.ENGINEERING_VIZ,
            semantic_keywords=["simulate", "failure", "system", "test", "scenario"]
        ),
        ToolMetadata(
            name="get_system_topology_tool",
            description="Get system topology information",
            pack=ToolPack.ENGINEERING_VIZ,
            semantic_keywords=["topology", "system", "structure", "architecture", "layout"]
        )
    ]
}


def get_all_tools() -> List[ToolMetadata]:
    """Get all tools from all packs"""
    all_tools = []
    for pack_tools in TOOL_PACKS.values():
        all_tools.extend(pack_tools)
    return all_tools


def get_tools_by_pack(pack: ToolPack) -> List[ToolMetadata]:
    """Get all tools in a specific pack"""
    return TOOL_PACKS.get(pack, [])


def get_tool_by_name(tool_name: str) -> Optional[ToolMetadata]:
    """Get tool metadata by name"""
    for pack_tools in TOOL_PACKS.values():
        for tool in pack_tools:
            if tool.name == tool_name:
                return tool
    return None


def get_pack_for_tool(tool_name: str) -> Optional[ToolPack]:
    """Get the pack that contains a specific tool"""
    tool = get_tool_by_name(tool_name)
    return tool.pack if tool else None
