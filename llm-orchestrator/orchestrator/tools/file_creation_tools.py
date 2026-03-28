"""
File Creation Tools for LLM Orchestrator Agents
Wrapper tools for creating files and folders in user's My Documents section
"""

import logging
from typing import Dict, Any, Optional, List

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)


# ── I/O models for file creation tools ──────────────────────────────────────

class ListFoldersInputs(BaseModel):
    """No required inputs (user_id is engine-injected)."""
    pass


class ListFoldersOutputs(BaseModel):
    """Outputs for list_folders_tool."""
    folders: List[Dict[str, Any]] = Field(description="List of folder dicts with folder_id, name, path, document_count")
    count: int = Field(description="Number of folders")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class CreateUserFileInputs(BaseModel):
    """Required inputs for create_user_file_tool."""
    filename: str = Field(description="Name of the file to create")
    content: str = Field(description="File content")


class CreateUserFileParams(BaseModel):
    """Optional parameters."""
    folder_id: Optional[str] = Field(default=None, description="Folder ID")
    folder_path: Optional[str] = Field(default=None, description="Folder path")
    title: Optional[str] = Field(default=None, description="Document title")


class CreateUserFileOutputs(BaseModel):
    """Outputs for create_user_file_tool."""
    success: bool = Field(description="Whether the file was created")
    document_id: Optional[str] = Field(default=None, description="Document ID")
    filename: Optional[str] = Field(default=None, description="Filename created")
    file_type: Optional[str] = Field(default=None, description="File extension or type")
    title: Optional[str] = Field(default=None, description="Document title")
    folder_path: Optional[str] = Field(default=None, description="Folder path")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class CreateUserFolderInputs(BaseModel):
    """Required inputs for create_user_folder_tool."""
    folder_name: str = Field(description="Name of the folder to create")


class CreateUserFolderParams(BaseModel):
    """Optional parameters."""
    parent_folder_id: Optional[str] = Field(default=None, description="Parent folder ID")
    parent_folder_path: Optional[str] = Field(default=None, description="Parent folder path")


class CreateUserFolderOutputs(BaseModel):
    """Outputs for create_user_folder_tool."""
    success: bool = Field(description="Whether the folder was created")
    folder_id: Optional[str] = Field(default=None, description="Folder ID")
    name: Optional[str] = Field(default=None, description="Folder name")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


def _build_folder_paths(folders: list) -> Dict[str, str]:
    """Build path string for each folder_id from flat list (parent_folder_id, name)."""
    by_id = {f["folder_id"]: f for f in folders}
    paths = {}

    def path_for(folder_id: str) -> str:
        if folder_id in paths:
            return paths[folder_id]
        f = by_id.get(folder_id)
        if not f:
            return ""
        parent_id = f.get("parent_folder_id")
        if not parent_id or parent_id not in by_id:
            paths[folder_id] = f["name"]
            return paths[folder_id]
        parent_path = path_for(parent_id)
        paths[folder_id] = f"{parent_path}/{f['name']}" if parent_path else f["name"]
        return paths[folder_id]

    for f in folders:
        path_for(f["folder_id"])
    return paths


async def list_folders_tool(user_id: str = "system") -> Dict[str, Any]:
    """
    List all folders in the user's document tree.
    Returns structured dict with folders list, count, and formatted.
    """
    try:
        client = await get_backend_tool_client()
        folders = await client.get_folder_tree(user_id=user_id)
        if not folders:
            return {"folders": [], "count": 0, "formatted": "No folders found."}
        paths = _build_folder_paths(folders)
        lines = []
        folder_list = []
        for f in sorted(folders, key=lambda x: paths.get(x["folder_id"], x["name"])):
            path = paths.get(f["folder_id"], f["name"])
            count = f.get("document_count", 0)
            lines.append(f"{path} ({count} docs)")
            folder_list.append({**f, "path": path})
        return {"folders": folder_list, "count": len(folder_list), "formatted": "\n".join(lines)}
    except Exception as e:
        logger.error(f"List folders tool error: {e}")
        err = str(e)
        return {"folders": [], "count": 0, "formatted": f"Failed to list folders: {err}"}


async def create_user_file_tool(
    filename: str,
    content: str,
    folder_id: Optional[str] = None,
    folder_path: Optional[str] = None,
    title: Optional[str] = None,
    tags: Optional[list] = None,
    category: Optional[str] = None,
    user_id: str = "system"
) -> Dict[str, Any]:
    """
    Create a file in the user's My Documents section.
    Returns structured dict with success, document_id, title, folder_path, and formatted.
    """
    try:
        logger.info(f"Creating user file: {filename} for user {user_id}")

        client = await get_backend_tool_client()
        response = await client.create_user_file(
            filename=filename,
            content=content,
            user_id=user_id,
            folder_id=folder_id,
            folder_path=folder_path,
            title=title,
            tags=tags,
            category=category
        )

        if response.get("success"):
            created_filename = response.get("filename", filename)
            folder_path_display = folder_path or response.get("folder_id", "My Documents")
            file_type = (created_filename.split(".")[-1] if "." in created_filename else None) or response.get("file_type")
            msg = f"Created file: {created_filename} in folder {folder_path_display} (document_id: {response.get('document_id', '')})"
            logger.info("Created user file: %s (document_id: %s)", created_filename, response.get("document_id"))
            return {
                "success": True,
                "document_id": response.get("document_id"),
                "filename": created_filename,
                "file_type": file_type,
                "title": response.get("title", filename),
                "folder_path": folder_path_display,
                "formatted": msg,
            }
        err = response.get("error", "Unknown error")
        logger.warning("Failed to create user file: %s", err)
        return {"success": False, "document_id": None, "filename": None, "file_type": None, "title": None, "folder_path": None, "error": err, "formatted": f"Failed to create file: {err}"}

    except Exception as e:
        logger.error("File creation tool error: %s", e)
        err = str(e)
        return {"success": False, "document_id": None, "filename": None, "file_type": None, "title": None, "folder_path": None, "error": err, "formatted": f"Failed to create file: {err}"}


register_action(
    name="list_folders",
    category="file",
    description="List all folders in the user's document tree",
    inputs_model=ListFoldersInputs,
    outputs_model=ListFoldersOutputs,
    tool_function=list_folders_tool,
)
register_action(
    name="create_user_file",
    category="file",
    description="Create a raw file with content. Use create_typed_document for templates.",
    inputs_model=CreateUserFileInputs,
    params_model=CreateUserFileParams,
    outputs_model=CreateUserFileOutputs,
    tool_function=create_user_file_tool,
    retriable=False,
)


async def create_user_folder_tool(
    folder_name: str,
    parent_folder_id: Optional[str] = None,
    parent_folder_path: Optional[str] = None,
    user_id: str = "system"
) -> Dict[str, Any]:
    """
    Create a folder in the user's My Documents section.
    Returns structured dict with success, folder_id, name, and formatted.
    """
    try:
        logger.info(f"Creating user folder: {folder_name} for user {user_id}")

        client = await get_backend_tool_client()
        response = await client.create_user_folder(
            folder_name=folder_name,
            user_id=user_id,
            parent_folder_id=parent_folder_id,
            parent_folder_path=parent_folder_path
        )

        if response.get("success"):
            msg = f"Created folder: {response.get('folder_name', folder_name)} (folder_id: {response.get('folder_id', '')})"
            logger.info(f"Created user folder: {response.get('folder_name')} (folder_id: {response.get('folder_id')})")
            return {
                "success": True,
                "folder_id": response.get("folder_id"),
                "name": response.get("folder_name", folder_name),
                "formatted": msg
            }
        err = response.get("error", "Unknown error")
        logger.warning(f"Failed to create user folder: {err}")
        return {"success": False, "error": err, "formatted": f"Failed to create folder: {err}"}

    except Exception as e:
        logger.error(f"Folder creation tool error: {e}")
        err = str(e)
        return {"success": False, "error": err, "formatted": f"Failed to create folder: {err}"}


register_action(
    name="create_user_folder",
    category="file",
    description="Create a folder in the user's My Documents section",
    inputs_model=CreateUserFolderInputs,
    params_model=CreateUserFolderParams,
    outputs_model=CreateUserFolderOutputs,
    tool_function=create_user_folder_tool,
    retriable=False,
)

