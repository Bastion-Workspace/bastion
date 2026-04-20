"""
Per-connection Workflow Composer actions for Microsoft 365 (contacts + Graph workloads).

Uses the same connection id as the user's Microsoft email row; action names are
``<prefix>:<connection_id>:<registry_tool_name>``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from services.database_manager.database_helpers import fetch_all

CONTACTS_CATEGORY_TOOL_NAMES = frozenset(
    {
        "list_contacts",
        "get_contact_by_id",
        "create_contact",
        "update_contact",
        "delete_contact",
        "search_contacts",
    }
)

# (tool_name, description_template, input_fields, output_fields)
CONTACTS_TOOL_SPECS: Sequence[Tuple[str, str, List[Dict[str, Any]], List[Dict[str, Any]]]] = (
    (
        "list_contacts",
        "List contacts from {account}",
        [
            {"name": "folder_id", "type": "text", "description": "O365 folder id", "required": False, "default": ""},
            {"name": "top", "type": "number", "description": "Max results", "required": False, "default": 100},
            {"name": "sources", "type": "text", "description": "all, microsoft, org, caldav", "required": False, "default": "all"},
        ],
        [
            {"name": "contacts", "type": "list[record]", "description": "Contacts"},
            {"name": "count", "type": "number", "description": "Count"},
            {"name": "formatted", "type": "text", "description": "Summary"},
        ],
    ),
    (
        "get_contact_by_id",
        "Get contact by id from {account}",
        [{"name": "contact_id", "type": "text", "description": "Contact id", "required": True}],
        [{"name": "formatted", "type": "text", "description": "Summary"}],
    ),
    (
        "create_contact",
        "Create contact in {account}",
        [
            {"name": "display_name", "type": "text", "description": "Display name", "required": False, "default": ""},
            {"name": "given_name", "type": "text", "description": "Given name", "required": False, "default": ""},
            {"name": "surname", "type": "text", "description": "Surname", "required": False, "default": ""},
        ],
        [{"name": "success", "type": "boolean", "description": "Ok"}, {"name": "formatted", "type": "text", "description": "Summary"}],
    ),
    (
        "update_contact",
        "Update contact in {account}",
        [
            {"name": "contact_id", "type": "text", "description": "Contact id", "required": True},
        ],
        [{"name": "success", "type": "boolean", "description": "Ok"}, {"name": "formatted", "type": "text", "description": "Summary"}],
    ),
    (
        "delete_contact",
        "Delete contact in {account}",
        [{"name": "contact_id", "type": "text", "description": "Contact id", "required": True}],
        [{"name": "success", "type": "boolean", "description": "Ok"}, {"name": "formatted", "type": "text", "description": "Summary"}],
    ),
    (
        "search_contacts",
        "Search contacts in {account}",
        [
            {"name": "query", "type": "text", "description": "Search query", "required": True},
            {"name": "sources", "type": "text", "description": "all, microsoft, org", "required": False, "default": "all"},
            {"name": "top", "type": "number", "description": "Max results", "required": False, "default": 20},
        ],
        [
            {"name": "contacts", "type": "list[record]", "description": "Matches"},
            {"name": "formatted", "type": "text", "description": "Summary"},
        ],
    ),
)

M365_TODO_SPECS: Sequence[Tuple[str, str, List[Dict[str, Any]], List[Dict[str, Any]]]] = (
    ("list_todo_lists", "List Microsoft To Do lists ({account})", [], [
        {"name": "lists", "type": "list[record]", "description": "Lists"},
        {"name": "formatted", "type": "text", "description": "Summary"},
    ]),
    (
        "get_todo_tasks",
        "List tasks in a To Do list ({account})",
        [
            {"name": "list_id", "type": "text", "description": "List id", "required": True},
            {"name": "top", "type": "number", "description": "Max tasks", "required": False, "default": 50},
        ],
        [{"name": "tasks", "type": "list[record]", "description": "Tasks"}, {"name": "formatted", "type": "text", "description": "Summary"}],
    ),
    (
        "create_todo_task",
        "Create To Do task ({account})",
        [
            {"name": "list_id", "type": "text", "description": "List id", "required": True},
            {"name": "title", "type": "text", "description": "Title", "required": True},
        ],
        [{"name": "task_id", "type": "text", "description": "New task id"}, {"name": "formatted", "type": "text", "description": "Summary"}],
    ),
    (
        "update_todo_task",
        "Update To Do task ({account})",
        [
            {"name": "list_id", "type": "text", "description": "List id", "required": True},
            {"name": "task_id", "type": "text", "description": "Task id", "required": True},
        ],
        [{"name": "success", "type": "boolean", "description": "Ok"}, {"name": "formatted", "type": "text", "description": "Summary"}],
    ),
    (
        "delete_todo_task",
        "Delete To Do task ({account})",
        [
            {"name": "list_id", "type": "text", "description": "List id", "required": True},
            {"name": "task_id", "type": "text", "description": "Task id", "required": True},
        ],
        [{"name": "success", "type": "boolean", "description": "Ok"}, {"name": "formatted", "type": "text", "description": "Summary"}],
    ),
)

M365_FILES_SPECS: Sequence[Tuple[str, str, List[Dict[str, Any]], List[Dict[str, Any]]]] = (
    (
        "list_drive_items",
        "List OneDrive items ({account})",
        [
            {"name": "parent_item_id", "type": "text", "description": "Folder id; empty = root", "required": False, "default": ""},
            {"name": "top", "type": "number", "description": "Max items", "required": False, "default": 50},
        ],
        [{"name": "items", "type": "list[record]", "description": "Items"}, {"name": "formatted", "type": "text", "description": "Summary"}],
    ),
    (
        "get_drive_item",
        "Get OneDrive item metadata ({account})",
        [{"name": "item_id", "type": "text", "description": "Item id", "required": True}],
        [{"name": "item", "type": "record", "description": "Item"}, {"name": "formatted", "type": "text", "description": "Summary"}],
    ),
    (
        "search_drive",
        "Search OneDrive ({account})",
        [
            {"name": "query", "type": "text", "description": "Query", "required": True},
            {"name": "top", "type": "number", "description": "Max results", "required": False, "default": 25},
        ],
        [{"name": "items", "type": "list[record]", "description": "Items"}, {"name": "formatted", "type": "text", "description": "Summary"}],
    ),
    (
        "get_onedrive_file_content",
        "Download OneDrive file ({account})",
        [{"name": "item_id", "type": "text", "description": "Item id", "required": True}],
        [
            {"name": "content_base64", "type": "text", "description": "Base64 content"},
            {"name": "mime_type", "type": "text", "description": "Mime type"},
            {"name": "formatted", "type": "text", "description": "Summary"},
        ],
    ),
    (
        "upload_onedrive_file",
        "Upload file to OneDrive ({account})",
        [
            {"name": "parent_item_id", "type": "text", "description": "Parent folder id", "required": False, "default": ""},
            {"name": "name", "type": "text", "description": "File name", "required": True},
            {"name": "content_base64", "type": "text", "description": "Base64 body", "required": True},
        ],
        [{"name": "item_id", "type": "text", "description": "New item id"}, {"name": "formatted", "type": "text", "description": "Summary"}],
    ),
    (
        "create_drive_folder",
        "Create OneDrive folder ({account})",
        [
            {"name": "parent_item_id", "type": "text", "description": "Parent folder id", "required": False, "default": ""},
            {"name": "name", "type": "text", "description": "Folder name", "required": True},
        ],
        [{"name": "item_id", "type": "text", "description": "New folder id"}, {"name": "formatted", "type": "text", "description": "Summary"}],
    ),
    (
        "move_drive_item",
        "Move OneDrive item ({account})",
        [
            {"name": "item_id", "type": "text", "description": "Item id", "required": True},
            {"name": "new_parent_item_id", "type": "text", "description": "Destination folder id", "required": True},
        ],
        [{"name": "success", "type": "boolean", "description": "Ok"}, {"name": "formatted", "type": "text", "description": "Summary"}],
    ),
    (
        "delete_drive_item",
        "Delete OneDrive item ({account})",
        [{"name": "item_id", "type": "text", "description": "Item id", "required": True}],
        [{"name": "success", "type": "boolean", "description": "Ok"}, {"name": "formatted", "type": "text", "description": "Summary"}],
    ),
)

M365_ONENOTE_SPECS: Sequence[Tuple[str, str, List[Dict[str, Any]], List[Dict[str, Any]]]] = (
    ("list_onenote_notebooks", "List OneNote notebooks ({account})", [], [
        {"name": "notebooks", "type": "list[record]", "description": "Notebooks"},
        {"name": "formatted", "type": "text", "description": "Summary"},
    ]),
    (
        "list_onenote_sections",
        "List OneNote sections ({account})",
        [{"name": "notebook_id", "type": "text", "description": "Notebook id", "required": True}],
        [{"name": "sections", "type": "list[record]", "description": "Sections"}, {"name": "formatted", "type": "text", "description": "Summary"}],
    ),
    (
        "list_onenote_pages",
        "List OneNote pages ({account})",
        [
            {"name": "section_id", "type": "text", "description": "Section id", "required": True},
            {"name": "top", "type": "number", "description": "Max pages", "required": False, "default": 50},
        ],
        [{"name": "pages", "type": "list[record]", "description": "Pages"}, {"name": "formatted", "type": "text", "description": "Summary"}],
    ),
    (
        "get_onenote_page_content",
        "Get OneNote page HTML ({account})",
        [{"name": "page_id", "type": "text", "description": "Page id", "required": True}],
        [{"name": "html_content", "type": "text", "description": "HTML"}, {"name": "formatted", "type": "text", "description": "Summary"}],
    ),
    (
        "create_onenote_page",
        "Create OneNote page ({account})",
        [
            {"name": "section_id", "type": "text", "description": "Section id", "required": True},
            {"name": "html", "type": "text", "description": "HTML body", "required": True},
        ],
        [{"name": "page_id", "type": "text", "description": "New page id"}, {"name": "formatted", "type": "text", "description": "Summary"}],
    ),
)

M365_PLANNER_SPECS: Sequence[Tuple[str, str, List[Dict[str, Any]], List[Dict[str, Any]]]] = (
    ("list_planner_plans", "List Microsoft Planner plans ({account})", [], [
        {"name": "plans", "type": "list[record]", "description": "Plans"},
        {"name": "formatted", "type": "text", "description": "Summary"},
    ]),
    (
        "get_planner_tasks",
        "List Planner tasks ({account})",
        [{"name": "plan_id", "type": "text", "description": "Plan id", "required": True}],
        [{"name": "tasks", "type": "list[record]", "description": "Tasks"}, {"name": "formatted", "type": "text", "description": "Summary"}],
    ),
    (
        "create_planner_task",
        "Create Planner task ({account})",
        [
            {"name": "plan_id", "type": "text", "description": "Plan id", "required": True},
            {"name": "title", "type": "text", "description": "Title", "required": True},
        ],
        [{"name": "task_id", "type": "text", "description": "New task id"}, {"name": "formatted", "type": "text", "description": "Summary"}],
    ),
    (
        "update_planner_task",
        "Update Planner task ({account})",
        [{"name": "task_id", "type": "text", "description": "Task id", "required": True}],
        [{"name": "success", "type": "boolean", "description": "Ok"}, {"name": "formatted", "type": "text", "description": "Summary"}],
    ),
    (
        "delete_planner_task",
        "Delete Planner task ({account})",
        [
            {"name": "task_id", "type": "text", "description": "Task id", "required": True},
            {"name": "etag", "type": "text", "description": "Task etag", "required": False, "default": ""},
        ],
        [{"name": "success", "type": "boolean", "description": "Ok"}, {"name": "formatted", "type": "text", "description": "Summary"}],
    ),
)


def _registry_names_from_specs(specs: Sequence[Tuple[str, str, Any, Any]]) -> frozenset:
    return frozenset(t[0] for t in specs)


TODO_REGISTRY_NAMES = _registry_names_from_specs(M365_TODO_SPECS)
FILES_REGISTRY_NAMES = _registry_names_from_specs(M365_FILES_SPECS)
ONENOTE_REGISTRY_NAMES = _registry_names_from_specs(M365_ONENOTE_SPECS)
PLANNER_REGISTRY_NAMES = _registry_names_from_specs(M365_PLANNER_SPECS)

M365_SCOPED_REGISTRY_NAMES = (
    TODO_REGISTRY_NAMES | FILES_REGISTRY_NAMES | ONENOTE_REGISTRY_NAMES | PLANNER_REGISTRY_NAMES
)


async def _microsoft_email_rows(user_id: str) -> List[Dict[str, Any]]:
    rows = await fetch_all(
        """
        SELECT id AS connection_id, account_identifier, display_name, provider
        FROM external_connections
        WHERE user_id = $1 AND connection_type = 'email' AND is_active = true
          AND LOWER(TRIM(COALESCE(provider, ''))) = 'microsoft'
        ORDER BY id
        """,
        user_id,
    )
    return list(rows or [])


def _expand_specs(
    rows: List[Dict[str, Any]],
    name_prefix: str,
    category: str,
    specs: Sequence[Tuple[str, str, List[Dict[str, Any]], List[Dict[str, Any]]]],
    account_template: str = "{account}",
) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    for row in rows:
        connection_id = int(row["connection_id"])
        account = (row.get("display_name") or row.get("account_identifier") or "").strip() or row.get(
            "account_identifier"
        ) or f"connection {connection_id}"
        prov = (row.get("provider") or "").strip()
        if prov:
            account = f"{account} ({prov})"
        for tool_name, desc_tpl, input_fields, output_fields in specs:
            action_name = f"{name_prefix}:{connection_id}:{tool_name}"
            description = desc_tpl.replace(account_template, account)
            actions.append(
                {
                    "name": action_name,
                    "category": category,
                    "description": description,
                    "input_schema": {
                        "type": "object",
                        "properties": {f["name"]: {"type": "string"} for f in input_fields},
                    },
                    "params_schema": {},
                    "output_schema": {"type": "object", "properties": {"formatted": {"type": "string"}}},
                    "input_fields": input_fields,
                    "output_fields": output_fields,
                }
            )
    return actions


async def get_contacts_actions_for_user(user_id: str) -> List[Dict[str, Any]]:
    rows = await _microsoft_email_rows(user_id)
    return _expand_specs(rows, "contacts", "contacts", CONTACTS_TOOL_SPECS, account_template="{account}")


async def get_m365_todo_actions_for_user(user_id: str) -> List[Dict[str, Any]]:
    return _expand_specs(await _microsoft_email_rows(user_id), "todo", "todo", M365_TODO_SPECS)


async def get_m365_files_actions_for_user(user_id: str) -> List[Dict[str, Any]]:
    return _expand_specs(await _microsoft_email_rows(user_id), "files", "files", M365_FILES_SPECS)


async def get_m365_onenote_actions_for_user(user_id: str) -> List[Dict[str, Any]]:
    return _expand_specs(await _microsoft_email_rows(user_id), "onenote", "onenote", M365_ONENOTE_SPECS)


async def get_m365_planner_actions_for_user(user_id: str) -> List[Dict[str, Any]]:
    return _expand_specs(await _microsoft_email_rows(user_id), "planner", "planner", M365_PLANNER_SPECS)
