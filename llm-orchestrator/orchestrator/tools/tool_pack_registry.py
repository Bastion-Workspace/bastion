"""
Tool Pack Registry - DEPRECATED.

Skills are now the single unit of capability assignment (Skills-First Architecture).
This registry is retained for backward compatibility with stored playbooks that still
reference tool_packs. New code should use skill_ids exclusively.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

# GitHub OAuth tools (registry names). Keep in sync with backend/api/agent_factory_api.py _GITHUB_TOOL_ROWS.
GITHUB_PACK_WRITE_TOOLS: frozenset = frozenset(
    {
        "github_create_issue",
        "github_create_issue_comment",
        "github_create_pr_review",
    }
)
GITHUB_PACK_ALL_TOOLS: Tuple[str, ...] = (
    "github_list_repos",
    "github_get_repo",
    "github_list_issues",
    "github_get_issue",
    "github_list_issue_comments",
    "github_list_pulls",
    "github_get_pull",
    "github_get_pull_diff",
    "github_list_pull_reviews",
    "github_list_pull_comments",
    "github_list_commits",
    "github_get_commit",
    "github_compare_refs",
    "github_get_file_content",
    "github_list_branches",
    "github_search_code",
    "github_create_issue",
    "github_create_issue_comment",
    "github_create_pr_review",
)
GITHUB_PACK_READ_TOOLS: Tuple[str, ...] = tuple(
    t for t in GITHUB_PACK_ALL_TOOLS if t not in GITHUB_PACK_WRITE_TOOLS
)


# Tool function names that were renamed in the Action I/O Registry for list/get consistency
_TOOL_TO_ACTION_RENAMES: Dict[str, str] = {
    "get_emails_tool": "list_emails",
    "get_email_folders_tool": "list_email_folders",
    "get_contacts_tool": "list_contacts",
    "get_rss_articles_tool": "list_rss_articles",
}


def _tool_name_to_registry_name(tool_name: str) -> str:
    """Map tool function name to Action I/O Registry action name (strip trailing _tool)."""
    if tool_name in _TOOL_TO_ACTION_RENAMES:
        return _TOOL_TO_ACTION_RENAMES[tool_name]
    return tool_name[:-5] if tool_name.endswith("_tool") else tool_name


class ToolPack(BaseModel):
    """A named group of tools that can be attached to a plan step for augmentation."""

    name: str
    description: str = Field(description="For LLM planner context")
    tools: List[str] = Field(description="Function names in orchestrator.tools")
    read_tools: Optional[List[str]] = Field(
        default=None,
        description="Read-only subset; when mode=read only these are resolved. If None, defaults to tools.",
    )

    def get_aggregate_outputs(self) -> List[Dict[str, Any]]:
        """
        Derive output fields from the Action I/O Registry for all tools in this pack.
        Returns a list of {name, type, description, source_action} for Workflow Composer.
        Tools not yet registered are skipped. Duplicate field names are disambiguated with source_action prefix.
        """
        from orchestrator.utils.action_io_registry import get_action

        seen: Dict[str, int] = {}
        fields: List[Dict[str, Any]] = []
        for tool_name in self.tools:
            action_name = _tool_name_to_registry_name(tool_name)
            contract = get_action(action_name)
            if not contract:
                continue
            for f in contract.get_output_fields():
                name = f.get("name", "")
                if name in seen:
                    seen[name] += 1
                    display_name = f"{action_name}.{name}"
                else:
                    seen[name] = 1
                    display_name = name
                fields.append({
                    "name": display_name,
                    "type": f.get("type", "any"),
                    "description": f.get("description", ""),
                    "source_action": action_name,
                })
        return fields


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
        read_tools=[
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
        read_tools=["clipboard_get_tool"],
    ),
    "planning": ToolPack(
        name="planning",
        description="Self-managed task planning: create, track, and adapt a multi-step plan",
        tools=[
            "create_plan_tool",
            "get_plan_tool",
            "update_plan_step_tool",
            "add_plan_step_tool",
        ],
        read_tools=["get_plan_tool"],
    ),
    "discovery": ToolPack(
        name="discovery",
        description="Use when you do NOT yet know which document contains the answer. Searches across the knowledge base (semantic), by tags, images, and the web. Use knowledge pack once you have a document_id.",
        tools=[
            "search_documents_tool",
            "search_by_tags_tool",
            "search_images_tool",
            "search_web_tool",
            "enhance_query_tool",
        ],
        read_tools=[
            "search_documents_tool",
            "search_by_tags_tool",
            "search_images_tool",
            "search_web_tool",
            "enhance_query_tool",
        ],
    ),
    "knowledge": ToolPack(
        name="knowledge",
        description="Use when you already have a document_id, path, or filename and need to read or search within it. Includes: full content retrieval, path resolution, in-document text search, and multi-document segment search. Use discovery pack first if you don't know where the content lives.",
        tools=[
            "get_document_content_tool",
            "find_document_by_path_tool",
            "search_within_document_tool",
            "search_segments_across_documents_tool",
        ],
        read_tools=[
            "get_document_content_tool",
            "find_document_by_path_tool",
            "search_within_document_tool",
            "search_segments_across_documents_tool",
        ],
    ),
    "knowledge_graph": ToolPack(
        name="knowledge_graph",
        description="Use when the query is entity-driven: find documents mentioning a person, organisation, or location; traverse entity co-occurrence relationships. Complements discovery and knowledge packs for structured entity queries.",
        tools=[
            "find_documents_by_entities_tool",
            "find_related_documents_by_entities_tool",
            "find_co_occurring_entities_tool",
            "search_entities_tool",
            "get_entity_tool",
        ],
        read_tools=[
            "find_documents_by_entities_tool",
            "find_related_documents_by_entities_tool",
            "find_co_occurring_entities_tool",
            "search_entities_tool",
            "get_entity_tool",
        ],
    ),
    "rss": ToolPack(
        name="rss",
        description="RSS: list feeds (with unread_count), get/search articles (read/star/import flags, unread/starred filters), list starred across all feeds, mark read/unread, set star, refresh, unread counts, pause/resume polling",
        tools=[
            "list_rss_feeds_tool",
            "add_rss_feed_tool",
            "refresh_rss_feed_tool",
            "get_rss_articles_tool",
            "search_rss_tool",
            "list_starred_rss_articles_tool",
            "delete_rss_feed_tool",
            "mark_article_read_tool",
            "mark_article_unread_tool",
            "set_article_starred_tool",
            "get_unread_counts_tool",
            "toggle_feed_active_tool",
        ],
        read_tools=[
            "list_rss_feeds_tool",
            "get_rss_articles_tool",
            "search_rss_tool",
            "list_starred_rss_articles_tool",
            "get_unread_counts_tool",
        ],
    ),
    "document_management": ToolPack(
        name="document_management",
        description="Create typed documents with frontmatter templates (project, fiction, electronics, outline, character, etc.), update content, and read metadata. For raw/untyped files use the file_management pack.",
        tools=[
            "create_typed_document_tool",
            "update_document_content_tool",
            "get_document_metadata_tool",
        ],
        read_tools=["get_document_metadata_tool"],
    ),
    "file_management": ToolPack(
        name="file_management",
        description="Create raw files and organize folders. For typed documents with templates (project, fiction, electronics, etc.) use the document_management pack instead. patch_file and append_to_file propose edits for user approval.",
        tools=[
            "list_folders_tool",
            "create_user_file_tool",
            "create_user_folder_tool",
            "patch_file_tool",
            "append_to_file_tool",
        ],
        read_tools=["list_folders_tool"],
    ),
    "org_management": ToolPack(
        name="org_management",
        description="Org-mode file structure parsing and headings search (read-only)",
        tools=[
            "parse_org_structure_tool",
            "search_org_headings_tool",
            "get_org_statistics_tool",
        ],
        read_tools=[
            "parse_org_structure_tool",
            "search_org_headings_tool",
            "get_org_statistics_tool",
        ],
    ),
    "task_management": ToolPack(
        name="task_management",
        description="Universal todo list, create, update, toggle, delete, archive, refile across any org file",
        tools=[
            "list_todos_tool",
            "create_todo_tool",
            "update_todo_tool",
            "toggle_todo_tool",
            "delete_todo_tool",
            "archive_done_tool",
            "refile_todo_tool",
            "discover_refile_targets_tool",
        ],
        read_tools=["list_todos_tool", "discover_refile_targets_tool"],
    ),
    "math": ToolPack(
        name="math",
        description="Math calculations, named formula evaluation (HVAC/electrical/construction), and unit conversions. Formula names are listed in evaluate_formula's description; call list_available_formulas for parameter details.",
        tools=[
            "calculate_expression_tool",
            "evaluate_formula_tool",
            "convert_units_tool",
            "list_available_formulas_tool",
        ],
        read_tools=[
            "calculate_expression_tool",
            "evaluate_formula_tool",
            "convert_units_tool",
            "list_available_formulas_tool",
        ],
    ),
    "utility": ToolPack(
        name="utility",
        description="State management and data manipulation: counters, dates, booleans, lists",
        tools=[
            "adjust_number_tool",
            "adjust_date_tool",
            "parse_date_tool",
            "compare_dates_tool",
            "set_value_tool",
            "toggle_boolean_tool",
            "append_to_list_tool",
            "get_list_length_tool",
        ],
        read_tools=["parse_date_tool", "compare_dates_tool", "get_list_length_tool"],
    ),
    "contacts": ToolPack(
        name="contacts",
        description="O365 and org-mode contacts: list, get by ID, create, update, delete (unified when include_org)",
        tools=[
            "get_contacts_tool",
            "get_contact_by_id_tool",
            "create_contact_tool",
            "update_contact_tool",
            "delete_contact_tool",
        ],
        read_tools=["get_contacts_tool", "get_contact_by_id_tool"],
    ),
    "notifications": ToolPack(
        name="notifications",
        description="Send messages and reminders to the user via in-app, Telegram, Discord, or email. "
                    "Use notify_user for the high-level path (respects user preferences); "
                    "use send_channel_message for a specific channel.",
        tools=["notify_user_tool", "send_channel_message_tool", "schedule_reminder_tool"],
        read_tools=[],
    ),
    "email": ToolPack(
        name="email",
        description="Read, search, send, draft, and manage emails via O365/Microsoft Graph. "
                    "Use list_email_folders to discover folder names before listing.",
        tools=[
            "get_emails_tool",
            "search_emails_tool",
            "get_email_thread_tool",
            "read_email_tool",
            "send_email_tool",
            "reply_to_email_tool",
            "create_draft_tool",
            "move_email_tool",
            "update_email_tool",
            "get_email_folders_tool",
            "get_email_statistics_tool",
        ],
        read_tools=[
            "get_emails_tool",
            "search_emails_tool",
            "get_email_thread_tool",
            "read_email_tool",
            "get_email_folders_tool",
            "get_email_statistics_tool",
        ],
    ),
    "calendar": ToolPack(
        name="calendar",
        description="Read and manage O365 calendar events. Use list_calendars first to get "
                    "calendar IDs, then get_calendar_events with ISO 8601 date ranges.",
        tools=[
            "list_calendars_tool",
            "get_calendar_events_tool",
            "get_event_by_id_tool",
            "create_event_tool",
            "update_event_tool",
            "delete_event_tool",
        ],
        read_tools=["list_calendars_tool", "get_calendar_events_tool", "get_event_by_id_tool"],
    ),
    "todo": ToolPack(
        name="todo",
        description="Microsoft To Do (Graph). Use the same Microsoft 365 email connection as other M365 packs; "
                    "list_todo_lists first, then pass each list's id (from the response) to get_todo_tasks and mutations—not display titles.",
        tools=[
            "list_todo_lists_tool",
            "get_todo_tasks_tool",
            "create_todo_task_tool",
            "update_todo_task_tool",
            "delete_todo_task_tool",
        ],
        read_tools=["list_todo_lists_tool", "get_todo_tasks_tool"],
    ),
    "files": ToolPack(
        name="files",
        description="OneDrive: browse, search, upload, and manage files. list_drive_items for root or folder; "
                    "get_onedrive_file_content returns base64 for small files.",
        tools=[
            "list_drive_items_tool",
            "get_drive_item_tool",
            "search_drive_tool",
            "get_onedrive_file_content_tool",
            "upload_onedrive_file_tool",
            "create_drive_folder_tool",
            "move_drive_item_tool",
            "delete_drive_item_tool",
        ],
        read_tools=[
            "list_drive_items_tool",
            "get_drive_item_tool",
            "search_drive_tool",
            "get_onedrive_file_content_tool",
        ],
    ),
    "onenote": ToolPack(
        name="onenote",
        description="OneNote: list notebooks, sections, pages; read or create pages (HTML).",
        tools=[
            "list_onenote_notebooks_tool",
            "list_onenote_sections_tool",
            "list_onenote_pages_tool",
            "get_onenote_page_content_tool",
            "create_onenote_page_tool",
        ],
        read_tools=[
            "list_onenote_notebooks_tool",
            "list_onenote_sections_tool",
            "list_onenote_pages_tool",
            "get_onenote_page_content_tool",
        ],
    ),
    "planner": ToolPack(
        name="planner",
        description="Microsoft Planner: list plans, then tasks; create, update, or delete tasks.",
        tools=[
            "list_planner_plans_tool",
            "get_planner_tasks_tool",
            "create_planner_task_tool",
            "update_planner_task_tool",
            "delete_planner_task_tool",
        ],
        read_tools=["list_planner_plans_tool", "get_planner_tasks_tool"],
    ),
    "github": ToolPack(
        name="github",
        description="GitHub API via OAuth: repositories, issues, pull requests, diffs, commits, code search. "
                    "Connect GitHub in Settings and bind accounts here; choose this pack on playbook steps with connection IDs.",
        tools=list(GITHUB_PACK_ALL_TOOLS),
        read_tools=list(GITHUB_PACK_READ_TOOLS),
    ),
    "gitea": ToolPack(
        name="gitea",
        description="Gitea API via personal access token (Settings → External connections). "
                    "Same tools as the GitHub pack; bind Gitea connection IDs on playbook steps.",
        tools=list(GITHUB_PACK_ALL_TOOLS),
        read_tools=list(GITHUB_PACK_READ_TOOLS),
    ),
    "navigation": ToolPack(
        name="navigation",
        description="Save and manage named locations; compute and save routes between them.",
        tools=[
            "create_location_tool",
            "list_locations_tool",
            "delete_location_tool",
            "compute_route_tool",
            "save_route_tool",
            "list_saved_routes_tool",
        ],
        read_tools=["list_locations_tool", "compute_route_tool", "list_saved_routes_tool"],
    ),
    "data_workspace": ToolPack(
        name="data_workspace",
        description="Query tabular data stored in Data Workspaces. Use list_data_workspaces "
                    "to find workspace IDs, get_workspace_schema to inspect tables, "
                    "then query_data_workspace with SQL or natural language.",
        tools=[
            "list_data_workspaces_tool",
            "get_workspace_schema_tool",
            "resolve_workspace_link_tool",
            "query_data_workspace_tool",
            "create_workspace_table_tool",
            "insert_workspace_rows_tool",
            "update_workspace_rows_tool",
            "delete_workspace_rows_tool",
        ],
        read_tools=[
            "list_data_workspaces_tool",
            "get_workspace_schema_tool",
            "resolve_workspace_link_tool",
            "query_data_workspace_tool",
        ],
    ),
    "image_generation": ToolPack(
        name="image_generation",
        description="Generate images from text descriptions using the configured image model.",
        tools=["generate_image_tool"],
        read_tools=[],
    ),
    "visualization": ToolPack(
        name="visualization",
        description="Create charts (bar, line, pie, scatter) from data. Returns a rendered image or embed code.",
        tools=["create_chart_tool"],
        read_tools=[],
    ),
    "data_connection_builder": ToolPack(
        name="data_connection_builder",
        description="Analyze APIs and websites, build and test data connections, bulk scrape URLs for content and images; build control panes for status bar",
        tools=[
            "probe_api_endpoint_tool",
            "analyze_openapi_spec_tool",
            "draft_connector_definition_tool",
            "validate_connector_definition_tool",
            "test_connector_endpoint_tool",
            "create_data_connector_tool",
            "list_data_connectors_tool",
            "update_data_connector_tool",
            "bulk_scrape_urls_tool",
            "get_bulk_scrape_status_tool",
            "bind_data_source_to_agent_tool",
            "list_control_panes_tool",
            "get_connector_endpoints_tool",
            "create_control_pane_tool",
            "update_control_pane_tool",
            "delete_control_pane_tool",
            "execute_control_action_tool",
            "crawl_web_content_tool",
            "search_web_tool",
        ],
        read_tools=[
            "probe_api_endpoint_tool",
            "analyze_openapi_spec_tool",
            "validate_connector_definition_tool",
            "test_connector_endpoint_tool",
            "list_data_connectors_tool",
            "get_bulk_scrape_status_tool",
            "list_control_panes_tool",
            "get_connector_endpoints_tool",
            "crawl_web_content_tool",
            "search_web_tool",
        ],
    ),
    "browser": ToolPack(
        name="browser",
        description="Browser automation: open persistent sessions, navigate, click, fill, extract, "
                    "inspect page structure, screenshot, download files, and close sessions. "
                    "Use browser_inspect to discover CSS selectors before building click/fill/extract steps.",
        tools=[
            "browser_open_session_tool",
            "browser_navigate_tool",
            "browser_click_tool",
            "browser_fill_tool",
            "browser_wait_tool",
            "browser_scroll_tool",
            "browser_extract_tool",
            "browser_inspect_tool",
            "browser_screenshot_tool",
            "browser_download_file_tool",
            "browser_close_session_tool",
        ],
        read_tools=[
            "browser_wait_tool",
            "browser_extract_tool",
            "browser_inspect_tool",
            "browser_screenshot_tool",
            "browser_download_file_tool",
        ],
    ),
    "local_device": ToolPack(
        name="local_device",
        description="All tools from the user's connected Bastion Local Proxy device. Dynamically filtered at execution time to only include capabilities the device actually provides.",
        tools=[
            "local_screenshot_tool",
            "local_clipboard_read_tool",
            "local_clipboard_write_tool",
            "local_system_info_tool",
            "local_desktop_notify_tool",
            "local_shell_execute_tool",
            "local_read_file_tool",
            "local_list_directory_tool",
            "local_write_file_tool",
            "local_list_processes_tool",
            "local_open_url_tool",
        ],
        read_tools=[
            "local_screenshot_tool",
            "local_clipboard_read_tool",
            "local_system_info_tool",
            "local_read_file_tool",
            "local_list_directory_tool",
            "local_list_processes_tool",
        ],
    ),
    "code_workspace": ToolPack(
        name="code_workspace",
        description="Coding workspace tools: file tree, content search, git info, and workspace root selection (local proxy-backed).",
        tools=[
            "code_open_workspace_tool",
            "code_file_tree_tool",
            "code_search_files_tool",
            "code_git_info_tool",
            "local_read_file_tool",
            "local_write_file_tool",
            "local_list_directory_tool",
            "local_shell_execute_tool",
        ],
        read_tools=[
            "code_open_workspace_tool",
            "code_file_tree_tool",
            "code_search_files_tool",
            "code_git_info_tool",
            "local_read_file_tool",
            "local_list_directory_tool",
        ],
    ),
    "team_tools": ToolPack(
        name="team_tools",
        description="Autonomous agent line tools: messaging, tasks, goals, governance. Auto-injected when the agent runs in a team context.",
        tools=[
            "send_to_agent",
            "start_agent_conversation",
            "halt_agent_conversation",
            "read_team_timeline",
            "read_my_messages",
            "get_team_status_board",
            "write_to_workspace",
            "read_workspace",
            "create_task_for_agent",
            "check_my_tasks",
            "update_task_status",
            "escalate_task",
            "list_team_goals",
            "report_goal_progress",
            "delegate_goal_to_tasks",
            "propose_hire",
            "propose_strategy_change",
            "get_agent_run_history",
            "propose_action",
            "vote_on_proposal",
            "tally_proposals",
        ],
        read_tools=[
            "read_team_timeline",
            "read_my_messages",
            "get_team_status_board",
            "read_workspace",
            "check_my_tasks",
            "list_team_goals",
            "get_agent_run_history",
            "tally_proposals",
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


def resolve_pack_tools_with_mode(
    pack_entries: List[Union[str, Dict[str, str]]],
) -> List[str]:
    """
    Resolve packs with optional access mode. Each entry is either a string (full access,
    backward compat) or {"pack": "name", "mode": "read"|"full"}.
    Returns a deduplicated list of tool names in order.
    """
    seen: set = set()
    result: List[str] = []
    for entry in pack_entries or []:
        if isinstance(entry, str):
            pack_name = entry
            mode = "full"
        elif isinstance(entry, dict):
            pack_name = entry.get("pack") or entry.get("name")
            mode = (entry.get("mode") or "full").lower()
            if not pack_name:
                continue
        else:
            continue
        pack = TOOL_PACKS.get(pack_name)
        if not pack:
            continue
        if mode == "read":
            tools_to_add = pack.read_tools if pack.read_tools is not None else pack.tools
        else:
            tools_to_add = pack.tools
        for tool_name in tools_to_add:
            if tool_name not in seen:
                seen.add(tool_name)
                result.append(tool_name)
    return result
