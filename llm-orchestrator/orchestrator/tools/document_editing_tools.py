"""
Document Editing Tools for LLM Orchestrator Agents
Wrapper tools for updating document titles, frontmatter, and content
"""

import logging
from typing import Dict, Any, Optional, List

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)


# ── I/O models for document editing tools ───────────────────────────────────

class UpdateDocumentMetadataInputs(BaseModel):
    """Required inputs for update_document_metadata_tool."""
    document_id: str = Field(description="Document ID to update")


class UpdateDocumentMetadataParams(BaseModel):
    """Optional parameters."""
    title: Optional[str] = Field(default=None, description="New title")
    frontmatter_type: Optional[str] = Field(default=None, description="Frontmatter type e.g. electronics, fiction, rules")


class UpdateDocumentMetadataOutputs(BaseModel):
    """Typed outputs for update_document_metadata_tool."""
    success: bool = Field(description="Whether the update succeeded")
    document_id: str = Field(description="Document ID")
    updated_fields: Optional[Dict[str, Any]] = Field(default=None, description="Fields that were updated")
    message: Optional[str] = Field(default=None, description="Status message")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class UpdateDocumentContentInputs(BaseModel):
    """Required inputs for update_document_content_tool."""
    document_id: str = Field(description="Document ID to update")
    content: str = Field(description="New content (append or replace)")


class UpdateDocumentContentParams(BaseModel):
    """Optional parameters."""
    append: bool = Field(default=False, description="If True append; if False replace entire content")


class UpdateDocumentContentOutputs(BaseModel):
    """Typed outputs for update_document_content_tool."""
    success: bool = Field(description="Whether the update succeeded")
    document_id: str = Field(description="Document ID")
    content_length: Optional[int] = Field(default=None, description="Content length after update")
    message: Optional[str] = Field(default=None, description="Status message")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class ProposeDocumentEditInputs(BaseModel):
    """Required inputs for propose_document_edit_tool."""
    document_id: str = Field(description="Document ID to edit")
    edit_type: str = Field(description="Edit type: operations or content")


class ProposeDocumentEditParams(BaseModel):
    """Optional parameters."""
    operations: Optional[List[Dict[str, Any]]] = Field(default=None, description="EditorOperation list for operation-based edits")
    content_edit: Optional[Dict[str, Any]] = Field(default=None, description="ContentEdit for content-based edits")
    agent_name: str = Field(default="unknown", description="Proposing agent name")
    summary: str = Field(default="", description="Human-readable summary of changes")
    requires_preview: bool = Field(default=True, description="Whether frontend must show preview")


class ProposeDocumentEditOutputs(BaseModel):
    """Typed outputs for propose_document_edit_tool."""
    success: bool = Field(description="Whether the proposal was created")
    proposal_id: Optional[str] = Field(default=None, description="Proposal ID for approval")
    document_id: str = Field(description="Document ID")
    message: Optional[str] = Field(default=None, description="Status message")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class ApplyOperationsDirectlyInputs(BaseModel):
    """Required inputs for apply_operations_directly_tool."""
    document_id: str = Field(description="Document ID to edit")
    operations: List[Dict[str, Any]] = Field(description="List of EditorOperation dicts to apply")


class ApplyOperationsDirectlyParams(BaseModel):
    """Optional parameters."""
    agent_name: str = Field(default="unknown", description="Agent requesting the operation")


class ApplyOperationsDirectlyOutputs(BaseModel):
    """Typed outputs for apply_operations_directly_tool."""
    success: bool = Field(description="Whether operations were applied")
    document_id: str = Field(description="Document ID")
    applied_count: Optional[int] = Field(default=None, description="Number of operations applied")
    message: Optional[str] = Field(default=None, description="Status message")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class ApplyDocumentEditProposalInputs(BaseModel):
    """Required inputs for apply_document_edit_proposal_tool."""
    proposal_id: str = Field(description="ID of proposal to apply")


class ApplyDocumentEditProposalParams(BaseModel):
    """Optional parameters."""
    selected_operation_indices: Optional[List[int]] = Field(default=None, description="Which operations to apply (None = all)")


class ApplyDocumentEditProposalOutputs(BaseModel):
    """Typed outputs for apply_document_edit_proposal_tool."""
    success: bool = Field(description="Whether the proposal was applied")
    document_id: Optional[str] = Field(default=None, description="Document ID")
    proposal_id: Optional[str] = Field(default=None, description="Proposal ID that was applied")
    applied_count: Optional[int] = Field(default=None, description="Number of edits applied")
    message: Optional[str] = Field(default=None, description="Status message")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


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
            formatted = f"Updated document metadata for {document_id}. Updated fields: {response.get('updated_fields', [])}"
            return {**response, "formatted": formatted}
        else:
            logger.warning(f"⚠️ Failed to update document metadata: {response.get('error')}")
            formatted = f"Failed to update document metadata: {response.get('error', 'Unknown error')}"
            return {**response, "formatted": formatted}
        
    except Exception as e:
        logger.error(f"❌ Document editing tool error: {e}")
        return {
            "success": False,
            "document_id": document_id,
            "error": str(e),
            "message": f"Error updating document: {str(e)}",
            "formatted": f"Error updating document: {str(e)}"
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
            formatted = f"Updated document content for {document_id}. Length: {response.get('content_length', 0)} chars (append={append})"
            return {**response, "formatted": formatted}
        else:
            logger.warning(f"⚠️ Failed to update document content: {response.get('error')}")
            formatted = f"Failed to update document content: {response.get('error', 'Unknown error')}"
            return {**response, "formatted": formatted}
        
    except Exception as e:
        logger.error(f"❌ Document content update error: {e}")
        return {
            "success": False,
            "document_id": document_id,
            "content_length": None,
            "error": str(e),
            "message": f"Error updating document content: {str(e)}",
            "formatted": f"Error updating document content: {str(e)}"
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
    Propose a document edit for user review (universal edit proposal system).
    For org-mode files: do not use for TODO state, tags, or priority changes — use
    list_todos and update_todo_tool (or toggle_todo_tool) so changes apply directly.
    
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
            formatted = f"Document edit proposal created for {document_id}. Proposal ID: {response.get('proposal_id')}. Summary: {summary or 'No summary'}"
            return {
                "success": True,
                "proposal_id": response.get("proposal_id"),
                "document_id": response.get("document_id", document_id),
                "message": response.get("message"),
                "error": None,
                "formatted": formatted,
            }
        logger.warning(f"⚠️ Failed to propose document edit: {response.get('error')}")
        formatted = f"Failed to propose document edit: {response.get('error', 'Unknown error')}"
        return {
            "success": False,
            "proposal_id": response.get("proposal_id"),
            "document_id": response.get("document_id", document_id),
            "message": response.get("message"),
            "error": response.get("error"),
            "formatted": formatted,
        }
        
    except Exception as e:
        logger.error(f"❌ Document edit proposal error: {e}")
        return {
            "success": False,
            "proposal_id": None,
            "document_id": document_id,
            "message": f"Error proposing document edit: {str(e)}",
            "error": str(e),
            "formatted": f"Error proposing document edit: {str(e)}",
        }


async def apply_operations_directly_tool(
    document_id: str,
    operations: List[Dict[str, Any]],
    user_id: str = "system",
    agent_name: str = "unknown"
) -> Dict[str, Any]:
    """
    Apply operations directly to a document file without creating a proposal.
    
    **SECURITY**: Only allowed for trusted agent names enforced by the backend (e.g. project_content_manager).
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
            formatted = f"Applied {response.get('applied_count', 0)} operation(s) directly to document {document_id}"
            return {**response, "formatted": formatted}
        else:
            logger.warning(f"⚠️ Failed to apply operations directly: {response.get('error')}")
            formatted = f"Failed to apply operations: {response.get('error', 'Unknown error')}"
            return {**response, "formatted": formatted}
        
    except Exception as e:
        logger.error(f"❌ Operations application error: {e}")
        return {
            "success": False,
            "document_id": document_id,
            "error": str(e),
            "message": f"Error applying operations directly: {str(e)}",
            "formatted": f"Error applying operations directly: {str(e)}"
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
            formatted = f"Applied document edit proposal {proposal_id}. Applied {response.get('applied_count', 0)} edit(s) to document {response.get('document_id', '')}"
            return {
                "success": True,
                "document_id": response.get("document_id"),
                "proposal_id": response.get("proposal_id", proposal_id),
                "applied_count": response.get("applied_count"),
                "message": response.get("message"),
                "error": None,
                "formatted": formatted,
            }
        logger.warning(f"⚠️ Failed to apply document edit proposal: {response.get('error')}")
        formatted = f"Failed to apply document edit proposal: {response.get('error', 'Unknown error')}"
        return {
            "success": False,
            "document_id": response.get("document_id"),
            "proposal_id": response.get("proposal_id", proposal_id),
            "applied_count": response.get("applied_count"),
            "message": response.get("message"),
            "error": response.get("error"),
            "formatted": formatted,
        }
        
    except Exception as e:
        logger.error(f"❌ Document edit proposal application error: {e}")
        return {
            "success": False,
            "document_id": None,
            "proposal_id": proposal_id,
            "applied_count": None,
            "message": f"Error applying document edit proposal: {str(e)}",
            "error": str(e),
            "formatted": f"Error applying document edit proposal: {str(e)}",
        }


# ── I/O models for list/get/reject proposals ─────────────────────────────────

class ListDocumentProposalsInputs(BaseModel):
    """Required inputs for list_document_proposals_tool."""
    document_id: str = Field(description="Document ID to list proposals for")


class ListDocumentProposalsOutputs(BaseModel):
    """Typed outputs for list_document_proposals_tool."""
    success: bool = Field(description="Whether the call succeeded")
    proposals: List[Dict[str, Any]] = Field(default_factory=list, description="List of proposal summaries")
    count: int = Field(default=0, description="Number of proposals")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class GetProposalDetailsInputs(BaseModel):
    """Required inputs for get_proposal_details_tool."""
    proposal_id: str = Field(description="ID of the proposal to fetch")


class GetProposalDetailsOutputs(BaseModel):
    """Typed outputs for get_proposal_details_tool."""
    success: bool = Field(description="Whether the call succeeded")
    proposal_id: Optional[str] = Field(default=None, description="Proposal ID")
    document_id: Optional[str] = Field(default=None, description="Document ID")
    edit_type: Optional[str] = Field(default=None, description="operations or content")
    operations: Optional[List[Dict[str, Any]]] = Field(default=None, description="List of EditorOperation dicts")
    content_edit: Optional[Dict[str, Any]] = Field(default=None, description="ContentEdit dict if content-based")
    agent_name: Optional[str] = Field(default=None, description="Proposing agent name")
    summary: Optional[str] = Field(default=None, description="Summary of changes")
    created_at: Optional[str] = Field(default=None, description="Creation timestamp")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class RejectDocumentProposalInputs(BaseModel):
    """Required inputs for reject_document_proposal_tool."""
    proposal_id: str = Field(description="ID of the proposal to reject (delete)")


class RejectDocumentProposalOutputs(BaseModel):
    """Typed outputs for reject_document_proposal_tool."""
    success: bool = Field(description="Whether the proposal was rejected")
    proposal_id: Optional[str] = Field(default=None, description="Proposal ID that was rejected")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def list_document_proposals_tool(
    document_id: str,
    user_id: str = "system"
) -> Dict[str, Any]:
    """
    List pending document edit proposals for a document.
    Use this to see what proposals exist before applying or rejecting.
    """
    try:
        client = await get_backend_tool_client()
        result = await client.list_document_proposals(document_id=document_id, user_id=user_id)
        if not result.get("success"):
            err = result.get("error", "Unknown error")
            return {
                "success": False,
                "proposals": [],
                "count": 0,
                "error": err,
                "formatted": f"Failed to list proposals: {err}",
            }
        proposals = result.get("proposals", [])
        parts = [f"Found {len(proposals)} pending proposal(s):"]
        for i, p in enumerate(proposals, 1):
            created = p.get("created_at", "")[:19] if p.get("created_at") else ""
            parts.append(
                f"{i}. {p.get('proposal_id', '')} — {p.get('edit_type', '')} ({p.get('operations_count', 0)} op(s)) — "
                f"\"{p.get('summary', '')}\" (created {created})"
            )
        return {
            "success": True,
            "proposals": proposals,
            "count": len(proposals),
            "error": None,
            "formatted": "\n".join(parts),
        }
    except Exception as e:
        logger.error(f"List document proposals error: {e}")
        return {
            "success": False,
            "proposals": [],
            "count": 0,
            "error": str(e),
            "formatted": f"Error listing proposals: {str(e)}",
        }


async def get_proposal_details_tool(
    proposal_id: str,
    user_id: str = "system"
) -> Dict[str, Any]:
    """
    Get full details of a document edit proposal (operations, summary, etc.).
    Use this to inspect what a proposal would change before applying or rejecting.
    """
    try:
        client = await get_backend_tool_client()
        result = await client.get_document_edit_proposal(proposal_id=proposal_id, user_id=user_id)
        if not result.get("success"):
            err = result.get("error", "Proposal not found")
            return {
                "success": False,
                "error": err,
                "formatted": f"Failed to get proposal: {err}",
            }
        ops = result.get("operations") or []
        parts = [
            f"Proposal {result.get('proposal_id', '')} for document {result.get('document_id', '')}",
            f"Type: {result.get('edit_type', '')}, Agent: {result.get('agent_name', '')}",
            f"Summary: {result.get('summary', '')}",
            f"Created: {result.get('created_at', '')}",
            f"Operations ({len(ops)}):",
        ]
        for i, op in enumerate(ops[:20], 1):
            parts.append(f"  {i}. {op.get('op_type', '')} — {op.get('original_text', '')[:60]}... → {op.get('text', '')[:60]}...")
        if len(ops) > 20:
            parts.append(f"  ... and {len(ops) - 20} more")
        return {
            "success": True,
            "proposal_id": result.get("proposal_id"),
            "document_id": result.get("document_id"),
            "edit_type": result.get("edit_type"),
            "operations": ops,
            "content_edit": result.get("content_edit"),
            "agent_name": result.get("agent_name"),
            "summary": result.get("summary"),
            "created_at": result.get("created_at"),
            "error": None,
            "formatted": "\n".join(parts),
        }
    except Exception as e:
        logger.error(f"Get proposal details error: {e}")
        return {
            "success": False,
            "error": str(e),
            "formatted": f"Error getting proposal details: {str(e)}",
        }


async def reject_document_proposal_tool(
    proposal_id: str,
    user_id: str = "system"
) -> Dict[str, Any]:
    """
    Reject (delete) a document edit proposal without applying it.
    Use when the user or agent decides not to apply a proposal.
    """
    try:
        client = await get_backend_tool_client()
        result = await client.reject_document_edit_proposal(proposal_id=proposal_id, user_id=user_id)
        if result.get("success"):
            formatted = f"Rejected (deleted) proposal {proposal_id}"
        else:
            formatted = f"Failed to reject proposal: {result.get('error', 'Unknown error')}"
        return {
            "success": result.get("success", False),
            "proposal_id": proposal_id,
            "error": result.get("error"),
            "formatted": formatted,
        }
    except Exception as e:
        logger.error(f"Reject document proposal error: {e}")
        return {
            "success": False,
            "proposal_id": proposal_id,
            "error": str(e),
            "formatted": f"Error rejecting proposal: {str(e)}",
        }


# ── Registry ─────────────────────────────────────────────────────────────────
# Only register tools that should appear in Agent Factory. Internal plumbing
# (update_document_metadata, propose_document_edit, apply_operations_directly,
# apply_document_edit_proposal) is used by built-in agents and patch_file but
# not exposed in the tool list.

register_action(
    name="update_document_content",
    category="document",
    description="Update document content (append or replace)",
    inputs_model=UpdateDocumentContentInputs,
    params_model=UpdateDocumentContentParams,
    outputs_model=UpdateDocumentContentOutputs,
    tool_function=update_document_content_tool,
    retriable=False,
)

register_action(
    name="list_document_proposals",
    category="document",
    description="List pending document edit proposals for a document",
    inputs_model=ListDocumentProposalsInputs,
    params_model=None,
    outputs_model=ListDocumentProposalsOutputs,
    tool_function=list_document_proposals_tool,
)

register_action(
    name="get_proposal_details",
    category="document",
    description="Get full details of a document edit proposal (operations, summary)",
    inputs_model=GetProposalDetailsInputs,
    params_model=None,
    outputs_model=GetProposalDetailsOutputs,
    tool_function=get_proposal_details_tool,
)

register_action(
    name="reject_document_proposal",
    category="document",
    description="Reject (delete) a document edit proposal without applying it",
    inputs_model=RejectDocumentProposalInputs,
    params_model=None,
    outputs_model=RejectDocumentProposalOutputs,
    tool_function=reject_document_proposal_tool,
    retriable=False,
)

