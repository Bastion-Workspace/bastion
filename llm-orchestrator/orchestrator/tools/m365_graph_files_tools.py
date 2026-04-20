"""Microsoft OneDrive / Files tools via Graph (connection-scoped)."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.tools.m365_invoke_common import invoke_m365_graph
from orchestrator.utils.action_io_registry import register_action


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _format_drive_items(items: List[Dict[str, Any]], parent_item_id: str) -> str:
    parent = parent_item_id or "root"
    if not items:
        return f"OneDrive: no items in folder (parent={parent})."
    lines = [
        "OneDrive items. Use the id values below for subsequent operations.",
        f"parent: {parent}",
        f"count: {len(items)}",
        "",
    ]
    for i, it in enumerate(items, 1):
        name = it.get("name", "")
        kind = "folder" if it.get("is_folder") else "file"
        size = it.get("size") or 0
        modified = it.get("last_modified", "")
        lines.append(f'{i}. "{name}" ({kind})')
        lines.append(f"   id: {it.get('id', '')}")
        if not it.get("is_folder"):
            lines.append(f"   mime_type: {it.get('mime_type', '')}")
            lines.append(f"   size: {size}")
        if modified:
            lines.append(f"   last_modified: {modified}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _format_drive_item(item: Dict[str, Any]) -> str:
    if not item:
        return "OneDrive item: not found."
    name = item.get("name", "")
    kind = "folder" if item.get("is_folder") else "file"
    lines = [
        f'OneDrive item: "{name}" ({kind})',
        f"  id: {item.get('id', '')}",
    ]
    if not item.get("is_folder"):
        lines.append(f"  mime_type: {item.get('mime_type', '')}")
        lines.append(f"  size: {item.get('size', 0)}")
    if item.get("web_url"):
        lines.append(f"  web_url: {item['web_url']}")
    if item.get("parent_id"):
        lines.append(f"  parent_id: {item['parent_id']}")
    if item.get("last_modified"):
        lines.append(f"  last_modified: {item['last_modified']}")
    return "\n".join(lines)


def _format_search_results(items: List[Dict[str, Any]], query: str) -> str:
    if not items:
        return f'OneDrive search "{query}": no results.'
    lines = [
        f'OneDrive search results for "{query}". Use these ids for follow-up operations.',
        f"count: {len(items)}",
        "",
    ]
    for i, it in enumerate(items, 1):
        name = it.get("name", "")
        kind = "folder" if it.get("is_folder") else "file"
        lines.append(f'{i}. "{name}" ({kind})')
        lines.append(f"   id: {it.get('id', '')}")
        if it.get("mime_type"):
            lines.append(f"   mime_type: {it['mime_type']}")
        if it.get("size"):
            lines.append(f"   size: {it['size']}")
        if it.get("web_url"):
            lines.append(f"   web_url: {it['web_url']}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _format_file_content(mime_type: str, content_b64: str) -> str:
    size_chars = len(content_b64) if content_b64 else 0
    approx_bytes = int(size_chars * 0.75)
    trunc = " (truncated)" if size_chars >= 12000 else ""
    return (
        f"OneDrive file content downloaded.\n"
        f"  mime_type: {mime_type or 'unknown'}\n"
        f"  base64_length: {size_chars}{trunc}\n"
        f"  approx_bytes: {approx_bytes}"
    )


def _format_file_mutation(operation: str, success: bool, item_id: str, error: str, name: str = "") -> str:
    if not success:
        return f"OneDrive {operation} failed: {error or 'unknown error'}"
    parts = [f"OneDrive {operation} succeeded."]
    if name:
        parts.append(f'  name: "{name}"')
    if item_id:
        parts.append(f"  item_id: {item_id}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Input / Output models
# ---------------------------------------------------------------------------

class ConnParams(BaseModel):
    connection_id: Optional[int] = Field(default=None, description="Microsoft 365 connection id")


class ListDriveItemsInputs(ConnParams):
    parent_item_id: str = Field(
        default="",
        description="Folder item id from a previous list_drive_items or search_drive call. "
        "Leave empty for the root of OneDrive."
    )
    top: int = Field(default=50, ge=1, le=200, description="Maximum items to return (1-200)")


class DriveItemsOutputs(BaseModel):
    items: List[Dict[str, Any]] = Field(default_factory=list)
    formatted: str = Field(description="Human-readable summary")


class GetDriveItemInputs(ConnParams):
    item_id: str = Field(
        description="The item id from list_drive_items or search_drive (the long id string, not the file name)"
    )


class DriveItemOutputs(BaseModel):
    item: Dict[str, Any] = Field(default_factory=dict)
    formatted: str = Field(description="Human-readable summary")


class SearchDriveInputs(ConnParams):
    query: str = Field(description="Search query (file name, keyword, or phrase)")
    top: int = Field(default=25, ge=1, le=100, description="Maximum results (1-100)")


class GetOnedriveFileContentInputs(ConnParams):
    item_id: str = Field(
        description="The item id from list_drive_items or search_drive (the long id string, not the file name)"
    )


class FileContentOutputs(BaseModel):
    content_base64: str = Field(default="")
    mime_type: str = Field(default="")
    formatted: str = Field(description="Human-readable summary")


class UploadOnedriveFileInputs(ConnParams):
    parent_item_id: str = Field(
        default="",
        description="Destination folder item id from list_drive_items. Leave empty for root."
    )
    name: str = Field(description="File name including extension (e.g. 'report.pdf')")
    content_base64: str = Field(description="File bytes encoded as base64")
    mime_type: str = Field(default="application/octet-stream", description="MIME type of the file")


class FileMutationOutputs(BaseModel):
    item_id: str = Field(default="")
    success: bool = Field(default=True)
    formatted: str = Field(description="Human-readable summary")


class CreateDriveFolderInputs(ConnParams):
    parent_item_id: str = Field(
        default="",
        description="Parent folder item id from list_drive_items. Leave empty for root."
    )
    name: str = Field(description="Folder name to create")


class MoveDriveItemInputs(ConnParams):
    item_id: str = Field(
        description="The item id to move (from list_drive_items or search_drive)"
    )
    new_parent_item_id: str = Field(
        description="Destination folder item id from list_drive_items"
    )


class OkOutputs(BaseModel):
    success: bool = Field(default=True)
    formatted: str = Field(description="Human-readable summary")


class DeleteDriveItemInputs(ConnParams):
    item_id: str = Field(
        description="The item id to delete (from list_drive_items or search_drive)"
    )


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

async def list_drive_items_tool(
    user_id: str = "system",
    parent_item_id: str = "",
    top: int = 50,
    connection_id: Optional[int] = None,
) -> Dict[str, Any]:
    out = await invoke_m365_graph(
        "list_drive_items",
        user_id,
        connection_id,
        {"parent_item_id": parent_item_id, "top": top},
    )
    items = out.get("items") or []
    items = items if isinstance(items, list) else []
    err = out.get("error")
    body = _format_drive_items(items, parent_item_id)
    return {
        "items": items,
        "formatted": (f"Error: {err}\n\n" if err else "") + body,
    }


async def get_drive_item_tool(
    user_id: str = "system",
    item_id: str = "",
    connection_id: Optional[int] = None,
) -> Dict[str, Any]:
    out = await invoke_m365_graph(
        "get_drive_item", user_id, connection_id, {"item_id": item_id}
    )
    item = out.get("item") or {}
    item = item if isinstance(item, dict) else {}
    err = out.get("error")
    body = _format_drive_item(item)
    return {
        "item": item,
        "formatted": (f"Error: {err}\n\n" if err else "") + body,
    }


async def search_drive_tool(
    user_id: str = "system",
    query: str = "",
    top: int = 25,
    connection_id: Optional[int] = None,
) -> Dict[str, Any]:
    out = await invoke_m365_graph(
        "search_drive", user_id, connection_id, {"query": query, "top": top}
    )
    items = out.get("items") or []
    items = items if isinstance(items, list) else []
    err = out.get("error")
    body = _format_search_results(items, query)
    return {
        "items": items,
        "formatted": (f"Error: {err}\n\n" if err else "") + body,
    }


async def get_onedrive_file_content_tool(
    user_id: str = "system",
    item_id: str = "",
    connection_id: Optional[int] = None,
) -> Dict[str, Any]:
    out = await invoke_m365_graph(
        "get_file_content", user_id, connection_id, {"item_id": item_id}
    )
    content_b64 = out.get("content_base64") or ""
    mime = out.get("mime_type") or ""
    err = out.get("error")
    body = _format_file_content(mime, content_b64)
    return {
        "content_base64": content_b64,
        "mime_type": mime,
        "formatted": (f"Error: {err}\n\n" if err else "") + body,
    }


async def upload_onedrive_file_tool(
    user_id: str = "system",
    parent_item_id: str = "",
    name: str = "file.bin",
    content_base64: str = "",
    mime_type: str = "application/octet-stream",
    connection_id: Optional[int] = None,
) -> Dict[str, Any]:
    out = await invoke_m365_graph(
        "upload_file",
        user_id,
        connection_id,
        {
            "parent_item_id": parent_item_id,
            "name": name,
            "content_base64": content_base64,
            "mime_type": mime_type,
        },
    )
    ok = out.get("success", False)
    iid = out.get("item_id") or ""
    err = out.get("error") or ""
    return {
        "item_id": iid,
        "success": ok,
        "formatted": _format_file_mutation("upload", ok, iid, err, name=name),
    }


async def create_drive_folder_tool(
    user_id: str = "system",
    parent_item_id: str = "",
    name: str = "New folder",
    connection_id: Optional[int] = None,
) -> Dict[str, Any]:
    out = await invoke_m365_graph(
        "create_drive_folder",
        user_id,
        connection_id,
        {"parent_item_id": parent_item_id, "name": name},
    )
    ok = out.get("success", False)
    iid = out.get("item_id") or ""
    err = out.get("error") or ""
    return {
        "item_id": iid,
        "success": ok,
        "formatted": _format_file_mutation("create_folder", ok, iid, err, name=name),
    }


async def move_drive_item_tool(
    user_id: str = "system",
    item_id: str = "",
    new_parent_item_id: str = "",
    connection_id: Optional[int] = None,
) -> Dict[str, Any]:
    out = await invoke_m365_graph(
        "move_drive_item",
        user_id,
        connection_id,
        {"item_id": item_id, "new_parent_item_id": new_parent_item_id},
    )
    ok = out.get("success", False)
    err = out.get("error") or ""
    return {
        "success": ok,
        "formatted": _format_file_mutation("move", ok, item_id, err),
    }


async def delete_drive_item_tool(
    user_id: str = "system",
    item_id: str = "",
    connection_id: Optional[int] = None,
) -> Dict[str, Any]:
    out = await invoke_m365_graph(
        "delete_drive_item", user_id, connection_id, {"item_id": item_id}
    )
    ok = out.get("success", False)
    err = out.get("error") or ""
    return {
        "success": ok,
        "formatted": _format_file_mutation("delete", ok, item_id, err),
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

register_action(
    name="list_drive_items",
    category="files",
    description="List items in OneDrive (root or folder)",
    inputs_model=ListDriveItemsInputs,
    outputs_model=DriveItemsOutputs,
    tool_function=list_drive_items_tool,
)
register_action(
    name="get_drive_item",
    category="files",
    description="Get OneDrive item metadata",
    inputs_model=GetDriveItemInputs,
    outputs_model=DriveItemOutputs,
    tool_function=get_drive_item_tool,
)
register_action(
    name="search_drive",
    category="files",
    description="Search OneDrive",
    inputs_model=SearchDriveInputs,
    outputs_model=DriveItemsOutputs,
    tool_function=search_drive_tool,
)
register_action(
    name="get_onedrive_file_content",
    category="files",
    description="Download OneDrive file content (base64)",
    inputs_model=GetOnedriveFileContentInputs,
    outputs_model=FileContentOutputs,
    tool_function=get_onedrive_file_content_tool,
)
register_action(
    name="upload_onedrive_file",
    category="files",
    description="Upload a file to OneDrive",
    inputs_model=UploadOnedriveFileInputs,
    outputs_model=FileMutationOutputs,
    tool_function=upload_onedrive_file_tool,
)
register_action(
    name="create_drive_folder",
    category="files",
    description="Create a folder in OneDrive",
    inputs_model=CreateDriveFolderInputs,
    outputs_model=FileMutationOutputs,
    tool_function=create_drive_folder_tool,
)
register_action(
    name="move_drive_item",
    category="files",
    description="Move a OneDrive item to another folder",
    inputs_model=MoveDriveItemInputs,
    outputs_model=OkOutputs,
    tool_function=move_drive_item_tool,
)
register_action(
    name="delete_drive_item",
    category="files",
    description="Delete a OneDrive item",
    inputs_model=DeleteDriveItemInputs,
    outputs_model=OkOutputs,
    tool_function=delete_drive_item_tool,
)
