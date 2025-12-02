"""
Document Editing Tools for LLM Orchestrator Agents
Wrapper tools for updating document titles, frontmatter, and content
"""

import logging
import os
from typing import Dict, Any, Optional, List
import httpx

from orchestrator.backend_tool_client import get_backend_tool_client

logger = logging.getLogger(__name__)


async def update_document_metadata_tool(
    document_id: str,
    title: Optional[str] = None,
    frontmatter_type: Optional[str] = None,
    user_id: str = "system"
) -> Dict[str, Any]:
    """
    Update document title and/or frontmatter type
    
    Args:
        document_id: Document ID to update
        title: Optional new title (updates both database metadata and frontmatter)
        frontmatter_type: Optional frontmatter type (e.g., "electronics", "fiction", "rules")
        user_id: User ID (required - must match document owner)
    
    Returns:
        Dict with success, document_id, updated_fields, and message
    """
    try:
        logger.info(f"Updating document metadata: {document_id} (title={title}, type={frontmatter_type})")
        
        # Get backend client
        client = await get_backend_tool_client()
        
        # Call backend client method
        response = await client.update_document_metadata(
            document_id=document_id,
            user_id=user_id,
            title=title,
            frontmatter_type=frontmatter_type
        )
        
        if response.get("success"):
            logger.info(f"✅ Updated document metadata: {response.get('updated_fields')}")
            return response
        else:
            logger.warning(f"⚠️ Failed to update document metadata: {response.get('error')}")
            return response
        
    except Exception as e:
        logger.error(f"❌ Document editing tool error: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Error updating document: {str(e)}"
        }


async def update_document_content_tool(
    document_id: str,
    content: str,
    user_id: str = "system",
    append: bool = False
) -> Dict[str, Any]:
    """
    Update document content (append or replace)
    
    Args:
        document_id: Document ID to update
        content: New content to add (if append=True) or replace entire content (if append=False)
        user_id: User ID (required - must match document owner)
        append: If True, append content to existing; if False, replace entire content
    
    Returns:
        Dict with success, document_id, content_length, and message
    """
    try:
        logger.info(f"Updating document content: {document_id} (append={append}, content_length={len(content)})")
        
        # Get backend client
        client = await get_backend_tool_client()
        
        # Call backend client method via gRPC
        response = await client.update_document_content(
            document_id=document_id,
            content=content,
            user_id=user_id,
            append=append
        )
        
        if response.get("success"):
            logger.info(f"✅ Updated document content: {response.get('content_length')} chars")
            return response
        else:
            logger.warning(f"⚠️ Failed to update document content: {response.get('error')}")
            return response
        
    except Exception as e:
        logger.error(f"❌ Document content update error: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Error updating document content: {str(e)}"
        }


async def propose_document_edit_tool(
    document_id: str,
    edit_type: str,
    operations: Optional[List[Dict[str, Any]]] = None,
    content_edit: Optional[Dict[str, Any]] = None,
    agent_name: str = "unknown",
    summary: str = "",
    requires_preview: bool = True,
    user_id: str = "system"
) -> Dict[str, Any]:
    """
    Propose a document edit for user review (universal edit proposal system)
    
    Args:
        document_id: Document ID to edit
        edit_type: "operations" or "content"
        operations: List of EditorOperation dicts (for operation-based edits)
        content_edit: ContentEdit dict (for content-based edits)
        agent_name: Name of proposing agent
        summary: Human-readable summary of proposed changes
        requires_preview: If False and edit is small, frontend may auto-apply
        user_id: User ID (required - must match document owner)
    
    Returns:
        Dict with success, proposal_id, document_id, and message
    """
    try:
        logger.info(f"Proposing document edit: {document_id} (type={edit_type}, agent={agent_name})")
        
        # Get backend client
        client = await get_backend_tool_client()
        
        # Call backend client method via gRPC
        response = await client.propose_document_edit(
            document_id=document_id,
            edit_type=edit_type,
            operations=operations,
            content_edit=content_edit,
            agent_name=agent_name,
            summary=summary,
            requires_preview=requires_preview,
            user_id=user_id
        )
        
        if response.get("success"):
            logger.info(f"✅ Document edit proposal created: {response.get('proposal_id')}")
            return response
        else:
            logger.warning(f"⚠️ Failed to propose document edit: {response.get('error')}")
            return response
        
    except Exception as e:
        logger.error(f"❌ Document edit proposal error: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Error proposing document edit: {str(e)}"
        }


async def apply_operations_directly_tool(
    document_id: str,
    operations: List[Dict[str, Any]],
    user_id: str = "system",
    agent_name: str = "unknown"
) -> Dict[str, Any]:
    """
    Apply operations directly to a document file without creating a proposal.
    
    **SECURITY**: Only allowed for specific agents (electronics_agent) editing referenced files.
    This is a restricted operation - use with caution!
    
    Args:
        document_id: Document ID to edit
        operations: List of EditorOperation dicts to apply
        user_id: User ID (required - must match document owner)
        agent_name: Name of agent requesting this operation (for security check)
    
    Returns:
        Dict with success, document_id, applied_count, and message
    """
    try:
        logger.info(f"Applying operations directly: {document_id} (agent: {agent_name}, {len(operations)} operations)")
        
        # Get backend client
        client = await get_backend_tool_client()
        
        # Call backend client method via gRPC
        # Note: We'll need to add this method to the backend client
        # For now, we'll use a workaround by creating a temporary proposal and applying it immediately
        # Actually, let's add it to the backend client properly
        response = await client.apply_operations_directly(
            document_id=document_id,
            operations=operations,
            user_id=user_id,
            agent_name=agent_name
        )
        
        if response.get("success"):
            logger.info(f"✅ Applied operations directly: {response.get('applied_count')} operation(s)")
            return response
        else:
            logger.warning(f"⚠️ Failed to apply operations directly: {response.get('error')}")
            return response
        
    except Exception as e:
        logger.error(f"❌ Operations application error: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Error applying operations directly: {str(e)}"
        }


async def apply_document_edit_proposal_tool(
    proposal_id: str,
    selected_operation_indices: Optional[List[int]] = None,
    user_id: str = "system"
) -> Dict[str, Any]:
    """
    Apply an approved document edit proposal
    
    Args:
        proposal_id: ID of proposal to apply
        selected_operation_indices: Which operations to apply (None = all, only for operation-based edits)
        user_id: User ID (required - must match proposal owner)
    
    Returns:
        Dict with success, document_id, applied_count, and message
    """
    try:
        logger.info(f"Applying document edit proposal: {proposal_id}")
        
        # Get backend client
        client = await get_backend_tool_client()
        
        # Call backend client method via gRPC
        response = await client.apply_document_edit_proposal(
            proposal_id=proposal_id,
            selected_operation_indices=selected_operation_indices,
            user_id=user_id
        )
        
        if response.get("success"):
            logger.info(f"✅ Applied document edit proposal: {response.get('applied_count')} edit(s)")
            return response
        else:
            logger.warning(f"⚠️ Failed to apply document edit proposal: {response.get('error')}")
            return response
        
    except Exception as e:
        logger.error(f"❌ Document edit proposal application error: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Error applying document edit proposal: {str(e)}"
        }

