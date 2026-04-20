"""
Declarative registry: connection capability keys, tool→prefix mapping, scoped prefixes.
Mirrors backend/services/provider_capability_registry.py for provider/capability keys only;
tool name lists match llm-orchestrator/orchestrator/engines/tool_resolution (registry names).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from orchestrator.tools.tool_pack_registry import (  # noqa: E402
    GITHUB_PACK_ALL_TOOLS,
    GITHUB_PACK_WRITE_TOOLS,
)

@dataclass(frozen=True)
class ProviderCapability:
    key: str
    tool_prefix: str
    tool_names: Tuple[str, ...]
    write_tools: FrozenSet[str] = frozenset()


@dataclass(frozen=True)
class ProviderDefinition:
    provider: str
    connection_type: str
    capabilities: Tuple[ProviderCapability, ...]
    multi_service: bool = False


_REGISTRY: Dict[Tuple[str, str], ProviderDefinition] = {}


def _cap(
    key: str,
    tools: Tuple[str, ...],
    *,
    write: FrozenSet[str] = frozenset(),
) -> ProviderCapability:
    return ProviderCapability(key=key, tool_prefix=key, tool_names=tools, write_tools=write)


def _register(defn: ProviderDefinition) -> None:
    _REGISTRY[(defn.provider.lower(), defn.connection_type.lower())] = defn


_EMAIL_WRITE = frozenset({"send_email", "reply_to_email"})
_CAL_WRITE = frozenset({"create_event", "update_event", "delete_event"})
_CONTACTS_WRITE = frozenset(
    {"create_contact", "update_contact", "delete_contact"}
)
_TODO_WRITE = frozenset(
    {"create_todo_task", "update_todo_task", "delete_todo_task"}
)
_FILES_WRITE = frozenset(
    {
        "upload_onedrive_file",
        "create_drive_folder",
        "move_drive_item",
        "delete_drive_item",
    }
)
_ONENOTE_WRITE = frozenset({"create_onenote_page"})
_PLANNER_WRITE = frozenset(
    {"create_planner_task", "update_planner_task", "delete_planner_task"}
)
_DEVOPS_WRITE = frozenset(
    {"create_devops_work_item", "update_devops_work_item", "add_devops_work_item_comment"}
)

_M365_CAPS: Tuple[ProviderCapability, ...] = (
    _cap(
        "email",
        (
            "list_emails",
            "search_emails",
            "list_email_folders",
            "read_email",
            "get_email_thread",
            "get_email_statistics",
            "send_email",
            "reply_to_email",
        ),
        write=_EMAIL_WRITE,
    ),
    _cap(
        "calendar",
        (
            "list_calendars",
            "get_calendar_events",
            "get_event_by_id",
            "create_event",
            "update_event",
            "delete_event",
        ),
        write=_CAL_WRITE,
    ),
    _cap(
        "contacts",
        (
            "list_contacts",
            "get_contact_by_id",
            "create_contact",
            "update_contact",
            "delete_contact",
            "search_contacts",
        ),
        write=_CONTACTS_WRITE,
    ),
    _cap(
        "todo",
        (
            "list_todo_lists",
            "get_todo_tasks",
            "create_todo_task",
            "update_todo_task",
            "delete_todo_task",
        ),
        write=_TODO_WRITE,
    ),
    _cap(
        "files",
        (
            "list_drive_items",
            "get_drive_item",
            "search_drive",
            "get_onedrive_file_content",
            "upload_onedrive_file",
            "create_drive_folder",
            "move_drive_item",
            "delete_drive_item",
        ),
        write=_FILES_WRITE,
    ),
    _cap(
        "onenote",
        (
            "list_onenote_notebooks",
            "list_onenote_sections",
            "list_onenote_pages",
            "get_onenote_page_content",
            "create_onenote_page",
        ),
        write=_ONENOTE_WRITE,
    ),
    _cap(
        "planner",
        (
            "list_planner_plans",
            "get_planner_tasks",
            "create_planner_task",
            "update_planner_task",
            "delete_planner_task",
        ),
        write=_PLANNER_WRITE,
    ),
    _cap(
        "devops",
        (
            "list_devops_projects",
            "list_devops_teams",
            "list_devops_team_members",
            "query_devops_work_items",
            "get_devops_work_item",
            "list_devops_iterations",
            "get_devops_iteration_work_items",
            "list_devops_boards",
            "get_devops_board_columns",
            "list_devops_repos",
            "list_devops_pull_requests",
            "list_devops_pipelines",
            "get_devops_pipeline_runs",
            "create_devops_work_item",
            "update_devops_work_item",
            "add_devops_work_item_comment",
        ),
        write=_DEVOPS_WRITE,
    ),
)

_register(
    ProviderDefinition(
        provider="microsoft",
        connection_type="email",
        capabilities=_M365_CAPS,
        multi_service=True,
    )
)

_register(
    ProviderDefinition(
        provider="github",
        connection_type="code_platform",
        capabilities=(
            ProviderCapability(
                key="github",
                tool_prefix="github",
                tool_names=tuple(GITHUB_PACK_ALL_TOOLS),
                write_tools=frozenset(GITHUB_PACK_WRITE_TOOLS),
            ),
        ),
        multi_service=False,
    )
)
_register(
    ProviderDefinition(
        provider="gitea",
        connection_type="code_platform",
        capabilities=(
            ProviderCapability(
                key="gitea",
                tool_prefix="gitea",
                tool_names=tuple(GITHUB_PACK_ALL_TOOLS),
                write_tools=frozenset(GITHUB_PACK_WRITE_TOOLS),
            ),
        ),
        multi_service=False,
    )
)


def get_provider_definition(provider: str, connection_type: str) -> Optional[ProviderDefinition]:
    return _REGISTRY.get(((provider or "").strip().lower(), (connection_type or "").strip().lower()))


@lru_cache(maxsize=1)
def build_tool_to_prefix_map() -> Dict[str, str]:
    m: Dict[str, str] = {}
    for defn in _REGISTRY.values():
        for cap in defn.capabilities:
            for tn in cap.tool_names:
                m[tn] = cap.tool_prefix
    return m


@lru_cache(maxsize=1)
def get_all_scoped_prefixes() -> Tuple[str, ...]:
    s: Set[str] = set()
    for defn in _REGISTRY.values():
        for cap in defn.capabilities:
            s.add(cap.tool_prefix)
    return tuple(sorted(s))


@lru_cache(maxsize=1)
def get_all_capability_keys() -> FrozenSet[str]:
    s: Set[str] = set()
    for defn in _REGISTRY.values():
        for cap in defn.capabilities:
            s.add(cap.key)
    return frozenset(s)


# Prefixes that use email/calendar style sanitization in pipeline_executor (not gh_/mcp).
@lru_cache(maxsize=1)
def get_m365_style_sanitize_prefixes() -> FrozenSet[str]:
    return frozenset(get_all_scoped_prefixes()) - frozenset({"github", "gitea"})


MULTI_ACCOUNT_MANIFEST_TYPES: FrozenSet[str] = frozenset({
    "email",
    "calendar",
    "contacts",
    "todo",
    "files",
    "onenote",
    "planner",
    "devops",
})
