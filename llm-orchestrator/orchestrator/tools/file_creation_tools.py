"""
File Creation Tools for LLM Orchestrator Agents
Wrapper tools for creating files and folders in user's My Documents section
"""

import logging
from typing import Dict, Any, Optional

from orchestrator.backend_tool_client import get_backend_tool_client

logger = logging.getLogger(__name__)


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


async def list_folders_tool(user_id: str = "system") -> str:
    """
    List all folders in the user's document tree.

    Args:
        user_id: User ID (injected by engine).

    Returns:
        Formatted list of folders with names, paths, and document counts.
    """
    try:
        client = await get_backend_tool_client()
        folders = await client.get_folder_tree(user_id=user_id)
        if not folders:
            return "No folders found."
        paths = _build_folder_paths(folders)
        lines = []
        for f in sorted(folders, key=lambda x: paths.get(x["folder_id"], x["name"])):
            path = paths.get(f["folder_id"], f["name"])
            count = f.get("document_count", 0)
            lines.append(f"{path} ({count} docs)")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"List folders tool error: {e}")
        return f"Failed to list folders: {e}"


async def create_user_file_tool(
    filename: str,
    content: str,
    folder_id: Optional[str] = None,
    folder_path: Optional[str] = None,
    title: Optional[str] = None,
    tags: Optional[list] = None,
    category: Optional[str] = None,
    user_id: str = "system"
) -> str:
    """
    Create a file in the user's My Documents section

    Args:
        filename: Name of the file to create (e.g., "sensor_spec.md", "circuit_diagram.txt")
        content: File content as string
        folder_id: Optional folder ID to place file in (must be user's folder)
        folder_path: Optional folder path (e.g., "Projects/Electronics") - will create if needed
        title: Optional document title (defaults to filename)
        tags: Optional list of tags for the document
        category: Optional category for the document
        user_id: User ID (required - must match the user making the request)

    Returns:
        Human-readable success or failure message.
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
            folder_path_display = folder_path or response.get("folder_id", "My Documents")
            msg = f"Created file: {response.get('filename', filename)} in folder {folder_path_display} (document_id: {response.get('document_id', '')})"
            logger.info(f"Created user file: {response.get('filename')} (document_id: {response.get('document_id')})")
            return msg
        err = response.get("error", "Unknown error")
        logger.warning(f"Failed to create user file: {err}")
        return f"Failed to create file: {err}"

    except Exception as e:
        logger.error(f"File creation tool error: {e}")
        return f"Failed to create file: {e}"


async def create_user_folder_tool(
    folder_name: str,
    parent_folder_id: Optional[str] = None,
    parent_folder_path: Optional[str] = None,
    user_id: str = "system"
) -> str:
    """
    Create a folder in the user's My Documents section

    Args:
        folder_name: Name of the folder to create (e.g., "Electronics Projects", "Components")
        parent_folder_id: Optional parent folder ID (must be user's folder)
        parent_folder_path: Optional parent folder path (e.g., "Projects") - will resolve to folder_id
        user_id: User ID (required - must match the user making the request)

    Returns:
        Human-readable success or failure message.
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
            return msg
        err = response.get("error", "Unknown error")
        logger.warning(f"Failed to create user folder: {err}")
        return f"Failed to create folder: {err}"

    except Exception as e:
        logger.error(f"Folder creation tool error: {e}")
        return f"Failed to create folder: {e}"

