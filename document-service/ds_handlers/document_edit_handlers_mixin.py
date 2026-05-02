"""gRPC handlers for Document Edit and Conversation operations."""

import json
import logging

import grpc
from protos import tool_service_pb2

logger = logging.getLogger(__name__)


def _write_initiator_from_update_content_request(request) -> str:
    """Map gRPC UpdateDocumentContentRequest to update_document_content_tool write_initiator."""
    if "write_initiator" not in request.DESCRIPTOR.fields_by_name:
        return "agent_tool"
    try:
        if request.HasField("write_initiator") and request.write_initiator == "user_api":
            return "user_api"
    except (AttributeError, ValueError):
        pass
    return "agent_tool"


class DocumentEditHandlersMixin:
    """Mixin providing Document Edit gRPC handlers.

    Mixed into ToolServiceImplementation; accesses self._get_search_service(),
    self._get_document_repo(), etc. via standard Python MRO.
    """

    async def UpdateDocumentMetadata(
        self,
        request: tool_service_pb2.UpdateDocumentMetadataRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.UpdateDocumentMetadataResponse:
        """Update document title and/or frontmatter type"""
        try:
            logger.info(f"UpdateDocumentMetadata: user={request.user_id}, doc={request.document_id}, title={request.title}, type={request.frontmatter_type}")
            
            # Import document editing tool
            from ds_langgraph_tools.document_editing_tools import update_document_metadata_tool
            
            # Execute metadata update
            result = await update_document_metadata_tool(
                document_id=request.document_id,
                title=request.title if request.title else None,
                frontmatter_type=request.frontmatter_type if request.frontmatter_type else None,
                user_id=request.user_id
            )
            
            # Build response
            if result.get("success"):
                response = tool_service_pb2.UpdateDocumentMetadataResponse(
                    success=True,
                    document_id=result.get("document_id", request.document_id),
                    updated_fields=result.get("updated_fields", []),
                    message=result.get("message", "Document metadata updated successfully")
                )
                logger.info(f"UpdateDocumentMetadata: Success - updated {len(response.updated_fields)} field(s)")
            else:
                response = tool_service_pb2.UpdateDocumentMetadataResponse(
                    success=False,
                    document_id=request.document_id,
                    message=result.get("message", "Document metadata update failed"),
                    error=result.get("error", "Unknown error")
                )
                logger.warning(f"UpdateDocumentMetadata: Failed - {response.error}")
            
            return response
            
        except Exception as e:
            logger.error(f"UpdateDocumentMetadata error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Document metadata update failed: {str(e)}")
    
    async def UpdateDocumentContent(
        self,
        request: tool_service_pb2.UpdateDocumentContentRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.UpdateDocumentContentResponse:
        """Update document content (append or replace)"""
        try:
            write_initiator = _write_initiator_from_update_content_request(request)
            logger.info(
                f"UpdateDocumentContent: user={request.user_id}, doc={request.document_id}, "
                f"append={request.append}, content_length={len(request.content)}, "
                f"write_initiator={write_initiator}"
            )
            
            # Import document editing tool
            from ds_langgraph_tools.document_editing_tools import update_document_content_tool
            
            # Execute content update
            result = await update_document_content_tool(
                document_id=request.document_id,
                content=request.content,
                user_id=request.user_id,
                append=request.append,
                write_initiator=write_initiator,
            )
            
            # Build response
            if result.get("success"):
                response = tool_service_pb2.UpdateDocumentContentResponse(
                    success=True,
                    document_id=result.get("document_id", request.document_id),
                    content_length=result.get("content_length", len(request.content)),
                    message=result.get("message", "Document content updated successfully")
                )
                logger.info(f"UpdateDocumentContent: Success - updated content ({response.content_length} chars)")
            else:
                response = tool_service_pb2.UpdateDocumentContentResponse(
                    success=False,
                    document_id=request.document_id,
                    message=result.get("message", "Document content update failed"),
                    error=result.get("error", "Unknown error")
                )
                logger.warning(f"UpdateDocumentContent: Failed - {response.error}")
            
            return response
            
        except Exception as e:
            logger.error(f"UpdateDocumentContent error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Document content update failed: {str(e)}")
    
    async def ProposeDocumentEdit(
        self,
        request: tool_service_pb2.ProposeDocumentEditRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ProposeDocumentEditResponse:
        """Propose a document edit for user review"""
        try:
            logger.info(f"ProposeDocumentEdit: user={request.user_id}, doc={request.document_id}, type={request.edit_type}, agent={request.agent_name}")
            
            # Import document editing tool
            from ds_langgraph_tools.document_editing_tools import propose_document_edit_tool
            
            # Convert proto operations to dicts
            operations = None
            if request.edit_type == "operations" and request.operations:
                operations = []
                for op_proto in request.operations:
                    op_dict = {
                        "op_type": op_proto.op_type,
                        "start": op_proto.start,
                        "end": op_proto.end,
                        "text": op_proto.text,
                        "pre_hash": op_proto.pre_hash,
                        "original_text": op_proto.original_text if op_proto.HasField("original_text") else None,
                        "anchor_text": op_proto.anchor_text if op_proto.HasField("anchor_text") else None,
                        "left_context": op_proto.left_context if op_proto.HasField("left_context") else None,
                        "right_context": op_proto.right_context if op_proto.HasField("right_context") else None,
                        "occurrence_index": op_proto.occurrence_index if op_proto.HasField("occurrence_index") else None,
                        "note": op_proto.note if op_proto.HasField("note") else None,
                        "confidence": op_proto.confidence if op_proto.HasField("confidence") else None,
                        "search_text": op_proto.search_text if op_proto.HasField("search_text") else None,
                    }
                    operations.append(op_dict)
            
            # Convert proto content_edit to dict
            content_edit = None
            if request.edit_type == "content" and request.HasField("content_edit"):
                ce_proto = request.content_edit
                content_edit = {
                    "edit_mode": ce_proto.edit_mode,
                    "content": ce_proto.content,
                    "insert_position": ce_proto.insert_position if ce_proto.HasField("insert_position") else None,
                    "note": ce_proto.note if ce_proto.HasField("note") else None
                }
            
            # Execute proposal
            result = await propose_document_edit_tool(
                document_id=request.document_id,
                edit_type=request.edit_type,
                operations=operations,
                content_edit=content_edit,
                agent_name=request.agent_name,
                summary=request.summary,
                requires_preview=request.requires_preview,
                user_id=request.user_id
            )
            
            # Build response
            if result.get("success"):
                response = tool_service_pb2.ProposeDocumentEditResponse(
                    success=True,
                    proposal_id=result.get("proposal_id", ""),
                    document_id=result.get("document_id", request.document_id),
                    message=result.get("message", "Document edit proposal created successfully")
                )
                logger.info(f"ProposeDocumentEdit: Success - proposal_id={response.proposal_id}")
            else:
                response = tool_service_pb2.ProposeDocumentEditResponse(
                    success=False,
                    proposal_id="",
                    document_id=request.document_id,
                    message=result.get("message", "Document edit proposal failed"),
                    error=result.get("error", "Unknown error")
                )
                logger.warning(f"ProposeDocumentEdit: Failed - {response.error}")
            
            return response
            
        except Exception as e:
            logger.error(f"ProposeDocumentEdit error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Document edit proposal failed: {str(e)}")
    
    async def ApplyOperationsDirectly(
        self,
        request: tool_service_pb2.ApplyOperationsDirectlyRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ApplyOperationsDirectlyResponse:
        """Apply operations directly to a document (for authorized agents only)"""
        try:
            playbook_auto_apply = bool(getattr(request, "playbook_auto_apply", False))
            logger.info(
                "ApplyOperationsDirectly: user=%s doc=%s agent=%s ops=%s playbook_auto_apply=%s",
                request.user_id,
                request.document_id,
                request.agent_name,
                len(request.operations),
                playbook_auto_apply,
            )
            
            # Import document editing tool
            from ds_langgraph_tools.document_editing_tools import apply_operations_directly
            
            # Convert proto operations to dicts
            operations = []
            for op_proto in request.operations:
                op_dict = {
                    "op_type": op_proto.op_type,
                    "start": op_proto.start,
                    "end": op_proto.end,
                    "text": op_proto.text,
                    "pre_hash": op_proto.pre_hash,
                    "original_text": op_proto.original_text if op_proto.HasField("original_text") else None,
                    "anchor_text": op_proto.anchor_text if op_proto.HasField("anchor_text") else None,
                    "left_context": op_proto.left_context if op_proto.HasField("left_context") else None,
                    "right_context": op_proto.right_context if op_proto.HasField("right_context") else None,
                    "occurrence_index": op_proto.occurrence_index if op_proto.HasField("occurrence_index") else None,
                    "note": op_proto.note if op_proto.HasField("note") else None,
                    "confidence": op_proto.confidence if op_proto.HasField("confidence") else None
                }
                operations.append(op_dict)
            
            # Execute direct operation application
            result = await apply_operations_directly(
                document_id=request.document_id,
                operations=operations,
                user_id=request.user_id,
                agent_name=request.agent_name,
                playbook_auto_apply=playbook_auto_apply,
            )
            
            # Build response
            if result.get("success"):
                response = tool_service_pb2.ApplyOperationsDirectlyResponse(
                    success=True,
                    document_id=result.get("document_id", request.document_id),
                    applied_count=result.get("applied_count", len(operations)),
                    message=result.get("message", "Operations applied successfully")
                )
                logger.info(f"ApplyOperationsDirectly: Success - {result.get('applied_count')} operations applied")
                return response
            else:
                response = tool_service_pb2.ApplyOperationsDirectlyResponse(
                    success=False,
                    document_id=request.document_id,
                    applied_count=0,
                    message=result.get("message", "Failed to apply operations"),
                    error=result.get("error", "Unknown error")
                )
                logger.warning(f"ApplyOperationsDirectly: Failed - {result.get('error')}")
                return response
                
        except Exception as e:
            logger.error(f"ApplyOperationsDirectly error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Direct operation application failed: {str(e)}")
    
    async def ApplyDocumentEditProposal(
        self,
        request: tool_service_pb2.ApplyDocumentEditProposalRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ApplyDocumentEditProposalResponse:
        """Apply an approved document edit proposal"""
        try:
            logger.info(f"ApplyDocumentEditProposal: user={request.user_id}, proposal={request.proposal_id}, selected_ops={len(request.selected_operation_indices)}")
            
            # Import document editing tool
            from ds_langgraph_tools.document_editing_tools import apply_document_edit_proposal
            
            # Convert repeated int32 to list
            selected_indices = list(request.selected_operation_indices) if request.selected_operation_indices else None
            
            # Execute proposal application
            result = await apply_document_edit_proposal(
                proposal_id=request.proposal_id,
                selected_operation_indices=selected_indices,
                user_id=request.user_id
            )
            
            # Build response
            if result.get("success"):
                response = tool_service_pb2.ApplyDocumentEditProposalResponse(
                    success=True,
                    document_id=result.get("document_id", ""),
                    applied_count=result.get("applied_count", 0),
                    message=result.get("message", "Document edit proposal applied successfully")
                )
                logger.info(f"ApplyDocumentEditProposal: Success - applied {response.applied_count} edit(s)")
            else:
                response = tool_service_pb2.ApplyDocumentEditProposalResponse(
                    success=False,
                    document_id="",
                    applied_count=0,
                    message=result.get("message", "Document edit proposal application failed"),
                    error=result.get("error", "Unknown error")
                )
                logger.warning(f"ApplyDocumentEditProposal: Failed - {response.error}")
            
            return response
            
        except Exception as e:
            logger.error(f"ApplyDocumentEditProposal error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Document edit proposal application failed: {str(e)}")

    async def ListDocumentProposals(
        self,
        request: tool_service_pb2.ListDocumentProposalsRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ListDocumentProposalsResponse:
        """List pending document edit proposals for a document."""
        try:
            from ds_langgraph_tools.document_editing_tools import list_pending_proposals_for_document
            list_result = await list_pending_proposals_for_document(
                document_id=request.document_id,
                user_id=request.user_id
            )
            proposals_raw = list_result["proposals"]
            summaries = []
            for p in proposals_raw:
                ops = p.get("operations") or []
                count = len(ops) if isinstance(ops, list) else 0
                if p.get("edit_type") == "content" and p.get("content_edit"):
                    count = 1
                summaries.append(tool_service_pb2.ProposalSummary(
                    proposal_id=p.get("proposal_id", ""),
                    document_id=p.get("document_id", ""),
                    edit_type=p.get("edit_type", ""),
                    agent_name=p.get("agent_name", ""),
                    summary=p.get("summary", ""),
                    operations_count=count,
                    created_at=p.get("created_at") or "",
                    expires_at=p.get("expires_at") or ""
                ))
            return tool_service_pb2.ListDocumentProposalsResponse(
                success=True,
                proposals=summaries
            )
        except Exception as e:
            logger.error(f"ListDocumentProposals error: {e}")
            return tool_service_pb2.ListDocumentProposalsResponse(
                success=False,
                error=str(e)
            )

    async def GetDocumentEditProposal(
        self,
        request: tool_service_pb2.GetDocumentEditProposalRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.GetDocumentEditProposalResponse:
        """Get full details of a document edit proposal."""
        try:
            import json
            from ds_langgraph_tools.document_editing_tools import get_document_edit_proposal
            proposal = await get_document_edit_proposal(
                proposal_id=request.proposal_id,
                user_id=request.user_id
            )
            if not proposal:
                return tool_service_pb2.GetDocumentEditProposalResponse(
                    success=False,
                    error="Proposal not found"
                )
            operations_json = json.dumps(proposal.get("operations") or [])
            content_edit = proposal.get("content_edit")
            content_edit_json = json.dumps(content_edit) if content_edit is not None else ""
            return tool_service_pb2.GetDocumentEditProposalResponse(
                success=True,
                proposal_id=proposal.get("proposal_id", ""),
                document_id=proposal.get("document_id", ""),
                edit_type=proposal.get("edit_type", ""),
                operations_json=operations_json,
                content_edit_json=content_edit_json,
                agent_name=proposal.get("agent_name", ""),
                summary=proposal.get("summary", ""),
                created_at=proposal.get("created_at") or ""
            )
        except Exception as e:
            logger.error(f"GetDocumentEditProposal error: {e}")
            return tool_service_pb2.GetDocumentEditProposalResponse(
                success=False,
                error=str(e)
            )

    async def RejectDocumentEditProposal(
        self,
        request: tool_service_pb2.RejectDocumentEditProposalRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.RejectDocumentEditProposalResponse:
        """Reject (delete) a document edit proposal."""
        try:
            from ds_db.database_manager.database_helpers import execute
            from ds_langgraph_tools.document_editing_tools import get_document_edit_proposal
            proposal = await get_document_edit_proposal(
                proposal_id=request.proposal_id,
                user_id=request.user_id
            )
            if not proposal:
                return tool_service_pb2.RejectDocumentEditProposalResponse(
                    success=False,
                    error="Proposal not found"
                )
            if proposal["user_id"] != request.user_id:
                return tool_service_pb2.RejectDocumentEditProposalResponse(
                    success=False,
                    error="Access denied"
                )
            await execute(
                "DELETE FROM document_edit_proposals WHERE proposal_id = $1::uuid",
                request.proposal_id,
                rls_context={"user_id": request.user_id, "user_role": "user"}
            )
            return tool_service_pb2.RejectDocumentEditProposalResponse(success=True)
        except Exception as e:
            logger.error(f"RejectDocumentEditProposal error: {e}")
            return tool_service_pb2.RejectDocumentEditProposalResponse(
                success=False,
                error=str(e)
            )

