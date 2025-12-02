"""
File Creation Tools for LLM Orchestrator Agents
Wrapper tools for creating files and folders in user's My Documents section
"""

import logging
from typing import Dict, Any, Optional

from orchestrator.backend_tool_client import get_backend_tool_client

logger = logging.getLogger(__name__)


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
        Dict with success, document_id, filename, folder_id, and message
    """
    try:
        logger.info(f"Creating user file: {filename} for user {user_id}")
        
        # Get backend client
        client = await get_backend_tool_client()
        
        # Call backend client method
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
            logger.info(f"✅ Created user file: {response.get('filename')} (document_id: {response.get('document_id')})")
            return response
        else:
            logger.warning(f"⚠️ Failed to create user file: {response.get('error')}")
            return response
        
    except Exception as e:
        logger.error(f"❌ File creation tool error: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Error creating file: {str(e)}"
        }


async def create_user_folder_tool(
    folder_name: str,
    parent_folder_id: Optional[str] = None,
    parent_folder_path: Optional[str] = None,
    user_id: str = "system"
) -> Dict[str, Any]:
    """
    Create a folder in the user's My Documents section
    
    Args:
        folder_name: Name of the folder to create (e.g., "Electronics Projects", "Components")
        parent_folder_id: Optional parent folder ID (must be user's folder)
        parent_folder_path: Optional parent folder path (e.g., "Projects") - will resolve to folder_id
        user_id: User ID (required - must match the user making the request)
    
    Returns:
        Dict with success, folder_id, folder_name, parent_folder_id, and message
    """
    try:
        logger.info(f"Creating user folder: {folder_name} for user {user_id}")
        
        # Get backend client
        client = await get_backend_tool_client()
        
        # Call backend client method
        response = await client.create_user_folder(
            folder_name=folder_name,
            user_id=user_id,
            parent_folder_id=parent_folder_id,
            parent_folder_path=parent_folder_path
        )
        
        if response.get("success"):
            logger.info(f"✅ Created user folder: {response.get('folder_name')} (folder_id: {response.get('folder_id')})")
            return response
        else:
            logger.warning(f"⚠️ Failed to create user folder: {response.get('error')}")
            return response
        
    except Exception as e:
        logger.error(f"❌ Folder creation tool error: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Error creating folder: {str(e)}"
        }

