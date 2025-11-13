"""
gRPC Context Gatherer - Comprehensive Context Assembly for llm-orchestrator

Gathers all necessary context from backend state and assembles it into
gRPC proto messages for sending to llm-orchestrator.

This is the SINGLE SOURCE OF TRUTH for what context gets sent to llm-orchestrator.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from protos import orchestrator_pb2
from langchain_core.messages import BaseMessage
from services.prompt_service import prompt_service

logger = logging.getLogger(__name__)


class GRPCContextGatherer:
    """
    Assembles comprehensive context for llm-orchestrator requests
    
    Responsibilities:
    - Extract conversation history from LangGraph state
    - Build user persona from settings
    - Conditionally add editor context (when on editor page)
    - Conditionally add pipeline context (when on pipeline page)
    - Add permission grants from shared memory
    - Add pending operations from state
    - Add routing locks and other context
    """
    
    def __init__(self):
        self.prompt_service = prompt_service
    
    async def build_chat_request(
        self,
        query: str,
        user_id: str,
        conversation_id: str,
        session_id: str = "default",
        request_context: Optional[Dict[str, Any]] = None,
        state: Optional[Dict[str, Any]] = None,
        agent_type: Optional[str] = None,
        routing_reason: Optional[str] = None
    ) -> orchestrator_pb2.ChatRequest:
        """
        Build comprehensive ChatRequest for llm-orchestrator
        
        Args:
            query: Current user query
            user_id: User UUID
            conversation_id: Conversation UUID
            session_id: Session identifier
            request_context: Frontend request context (active_editor, pipeline, etc.)
            state: LangGraph conversation state (if available)
            agent_type: Optional explicit agent routing
            routing_reason: Optional reason for routing decision
            
        Returns:
            Fully populated ChatRequest proto message
        """
        try:
            logger.info(f"🔧 CONTEXT GATHERER: Building gRPC request for user {user_id}")
            
            # Create base request with core fields
            grpc_request = orchestrator_pb2.ChatRequest(
                user_id=user_id,
                conversation_id=conversation_id,
                query=query,
                session_id=session_id
            )
            
            # Add routing control if specified
            if agent_type:
                grpc_request.agent_type = agent_type
                logger.info(f"🎯 CONTEXT GATHERER: Explicit routing to {agent_type}")
            
            if routing_reason:
                grpc_request.routing_reason = routing_reason
            
            # Initialize request context if not provided
            request_context = request_context or {}
            
            # === CONVERSATION HISTORY ===
            await self._add_conversation_history(grpc_request, state, conversation_id, user_id)
            
            # === USER PERSONA ===
            await self._add_user_persona(grpc_request, user_id)
            
            # === EDITOR CONTEXT (Conditional) ===
            await self._add_editor_context(grpc_request, request_context)
            
            # === PIPELINE CONTEXT (Conditional) ===
            await self._add_pipeline_context(grpc_request, request_context)
            
            # === PERMISSION GRANTS (Conditional) ===
            await self._add_permission_grants(grpc_request, state)
            
            # === PENDING OPERATIONS (Conditional) ===
            await self._add_pending_operations(grpc_request, state)
            
            # === ROUTING LOCKS (Conditional) ===
            await self._add_routing_locks(grpc_request, request_context, state)
            
            # === CHECKPOINTING (Conditional) ===
            await self._add_checkpoint_info(grpc_request, request_context)
            
            # Log summary
            self._log_context_summary(grpc_request)
            
            return grpc_request
            
        except Exception as e:
            logger.error(f"❌ CONTEXT GATHERER: Failed to build request: {e}")
            # Return minimal valid request
            return orchestrator_pb2.ChatRequest(
                user_id=user_id,
                conversation_id=conversation_id,
                query=query,
                session_id=session_id
            )
    
    async def _add_conversation_history(
        self,
        grpc_request: orchestrator_pb2.ChatRequest,
        state: Optional[Dict[str, Any]],
        conversation_id: str,
        user_id: str
    ) -> None:
        """Add conversation history to request"""
        try:
            # Extract messages from state if available
            messages = []
            if state and "messages" in state:
                messages = state["messages"]
            
            # Limit to last 20 messages for context window
            recent_messages = messages[-20:] if len(messages) > 20 else messages
            
            for msg in recent_messages:
                if hasattr(msg, 'content') and hasattr(msg, 'type'):
                    # LangChain message format
                    role = "user" if msg.type == "human" else "assistant"
                    
                    grpc_request.conversation_history.append(
                        orchestrator_pb2.ConversationMessage(
                            role=role,
                            content=msg.content,
                            timestamp=datetime.now().isoformat()
                        )
                    )
            
            if len(grpc_request.conversation_history) > 0:
                logger.info(f"✅ CONTEXT: Added {len(grpc_request.conversation_history)} messages to history")
            
        except Exception as e:
            logger.warning(f"⚠️ CONTEXT: Failed to add conversation history: {e}")
    
    async def _add_user_persona(
        self,
        grpc_request: orchestrator_pb2.ChatRequest,
        user_id: str
    ) -> None:
        """Add user persona and preferences"""
        try:
            # Get user settings
            user_settings = await self.prompt_service.get_user_settings_for_service(user_id)
            
            if user_settings:
                grpc_request.persona.CopyFrom(
                    orchestrator_pb2.UserPersona(
                        ai_name=user_settings.ai_name or "Kodex",
                        persona_style=user_settings.persona_style.value if user_settings.persona_style else "professional",
                        political_bias=user_settings.political_bias.value if user_settings.political_bias else "neutral",
                        timezone=user_settings.timezone if hasattr(user_settings, 'timezone') else "UTC"
                    )
                )
                logger.info(f"✅ CONTEXT: Added persona (ai_name={user_settings.ai_name})")
            
        except Exception as e:
            logger.warning(f"⚠️ CONTEXT: Failed to add persona: {e}")
    
    async def _add_editor_context(
        self,
        grpc_request: orchestrator_pb2.ChatRequest,
        request_context: Dict[str, Any]
    ) -> None:
        """Add active editor context (for fiction editing, proofreading, etc.)"""
        try:
            active_editor = request_context.get("active_editor")
            editor_preference = request_context.get("editor_preference", "prefer")
            
            # Skip if user said to ignore editor
            if editor_preference == "ignore":
                return
            
            # Skip if no editor context
            if not active_editor:
                return
            
            # Validate editor context
            if not isinstance(active_editor, dict):
                return
            
            if not active_editor.get("is_editable"):
                return
            
            filename = active_editor.get("filename", "")
            if not filename.endswith(".md"):
                return
            
            # Parse frontmatter
            frontmatter_data = active_editor.get("frontmatter", {})
            frontmatter = orchestrator_pb2.EditorFrontmatter(
                type=frontmatter_data.get("type", ""),
                title=frontmatter_data.get("title", ""),
                author=frontmatter_data.get("author", ""),
                status=frontmatter_data.get("status", "")
            )
            
            # Add tags
            tags = frontmatter_data.get("tags", [])
            if tags:
                frontmatter.tags.extend(tags)
            
            # Add custom fields
            for key, value in frontmatter_data.items():
                if key not in ["type", "title", "author", "tags", "status"]:
                    frontmatter.custom_fields[key] = str(value)
            
            # Build editor message
            grpc_request.active_editor.CopyFrom(
                orchestrator_pb2.ActiveEditor(
                    is_editable=True,
                    filename=filename,
                    language=active_editor.get("language", "markdown"),
                    content=active_editor.get("content", ""),
                    content_length=len(active_editor.get("content", "")),
                    frontmatter=frontmatter,
                    editor_preference=editor_preference
                )
            )
            
            logger.info(f"✅ CONTEXT: Added editor context (file={filename}, type={frontmatter.type}, {len(active_editor.get('content', ''))} chars)")
            
        except Exception as e:
            logger.warning(f"⚠️ CONTEXT: Failed to add editor context: {e}")
    
    async def _add_pipeline_context(
        self,
        grpc_request: orchestrator_pb2.ChatRequest,
        request_context: Dict[str, Any]
    ) -> None:
        """Add pipeline execution context"""
        try:
            pipeline_preference = request_context.get("pipeline_preference")
            active_pipeline_id = request_context.get("active_pipeline_id")
            
            # Skip if user said to ignore pipelines
            if pipeline_preference == "ignore":
                return
            
            # Skip if no pipeline context
            if not active_pipeline_id:
                return
            
            # Build pipeline message
            grpc_request.pipeline_context.CopyFrom(
                orchestrator_pb2.PipelineContext(
                    pipeline_preference=pipeline_preference or "prefer",
                    active_pipeline_id=active_pipeline_id,
                    pipeline_name=""  # Could fetch from DB if needed
                )
            )
            
            logger.info(f"✅ CONTEXT: Added pipeline context (id={active_pipeline_id})")
            
        except Exception as e:
            logger.warning(f"⚠️ CONTEXT: Failed to add pipeline context: {e}")
    
    async def _add_permission_grants(
        self,
        grpc_request: orchestrator_pb2.ChatRequest,
        state: Optional[Dict[str, Any]]
    ) -> None:
        """Add HITL permission grants"""
        try:
            if not state:
                return
            
            shared_memory = state.get("shared_memory", {})
            
            # Check if any permissions exist
            has_permissions = any([
                shared_memory.get("web_search_permission"),
                shared_memory.get("web_crawl_permission"),
                shared_memory.get("file_write_permission"),
                shared_memory.get("external_api_permission")
            ])
            
            if not has_permissions:
                return
            
            # Build permission grants
            grpc_request.permission_grants.CopyFrom(
                orchestrator_pb2.PermissionGrants(
                    web_search_permission=shared_memory.get("web_search_permission", False),
                    web_crawl_permission=shared_memory.get("web_crawl_permission", False),
                    file_write_permission=shared_memory.get("file_write_permission", False),
                    external_api_permission=shared_memory.get("external_api_permission", False)
                )
            )
            
            granted = [k.replace("_permission", "") for k, v in shared_memory.items() if k.endswith("_permission") and v]
            logger.info(f"✅ CONTEXT: Added permission grants ({', '.join(granted)})")
            
        except Exception as e:
            logger.warning(f"⚠️ CONTEXT: Failed to add permission grants: {e}")
    
    async def _add_pending_operations(
        self,
        grpc_request: orchestrator_pb2.ChatRequest,
        state: Optional[Dict[str, Any]]
    ) -> None:
        """Add pending operations awaiting user approval"""
        try:
            if not state:
                return
            
            pending_ops = state.get("pending_operations", [])
            
            if not pending_ops:
                return
            
            for op in pending_ops:
                if not isinstance(op, dict):
                    continue
                
                grpc_request.pending_operations.append(
                    orchestrator_pb2.PendingOperationInfo(
                        id=op.get("id", ""),
                        type=op.get("type", ""),
                        summary=op.get("summary", ""),
                        permission_required=op.get("permission_required", False),
                        status=op.get("status", "pending"),
                        created_at=op.get("created_at", datetime.now().isoformat())
                    )
                )
            
            logger.info(f"✅ CONTEXT: Added {len(grpc_request.pending_operations)} pending operations")
            
        except Exception as e:
            logger.warning(f"⚠️ CONTEXT: Failed to add pending operations: {e}")
    
    async def _add_routing_locks(
        self,
        grpc_request: orchestrator_pb2.ChatRequest,
        request_context: Dict[str, Any],
        state: Optional[Dict[str, Any]]
    ) -> None:
        """Add routing locks for dedicated agent sessions"""
        try:
            # Check request context first
            locked_agent = request_context.get("locked_agent")
            
            # Check shared memory as fallback
            if not locked_agent and state:
                shared_memory = state.get("shared_memory", {})
                locked_agent = shared_memory.get("locked_agent")
            
            if locked_agent:
                grpc_request.locked_agent = locked_agent
                logger.info(f"✅ CONTEXT: Added routing lock (agent={locked_agent})")
            
        except Exception as e:
            logger.warning(f"⚠️ CONTEXT: Failed to add routing lock: {e}")
    
    async def _add_checkpoint_info(
        self,
        grpc_request: orchestrator_pb2.ChatRequest,
        request_context: Dict[str, Any]
    ) -> None:
        """Add checkpoint info for conversation branching"""
        try:
            base_checkpoint_id = request_context.get("base_checkpoint_id")
            
            if base_checkpoint_id:
                grpc_request.base_checkpoint_id = base_checkpoint_id
                logger.info(f"✅ CONTEXT: Added checkpoint branching (checkpoint={base_checkpoint_id})")
            
        except Exception as e:
            logger.warning(f"⚠️ CONTEXT: Failed to add checkpoint info: {e}")
    
    def _log_context_summary(self, grpc_request: orchestrator_pb2.ChatRequest) -> None:
        """Log summary of what context was included"""
        context_items = []
        
        if len(grpc_request.conversation_history) > 0:
            context_items.append(f"history({len(grpc_request.conversation_history)})")
        
        if grpc_request.HasField("persona"):
            context_items.append("persona")
        
        if grpc_request.HasField("active_editor"):
            context_items.append(f"editor({grpc_request.active_editor.filename})")
        
        if grpc_request.HasField("pipeline_context"):
            context_items.append("pipeline")
        
        if grpc_request.HasField("permission_grants"):
            context_items.append("permissions")
        
        if len(grpc_request.pending_operations) > 0:
            context_items.append(f"pending_ops({len(grpc_request.pending_operations)})")
        
        if grpc_request.locked_agent:
            context_items.append(f"locked({grpc_request.locked_agent})")
        
        if grpc_request.agent_type:
            context_items.append(f"route({grpc_request.agent_type})")
        
        logger.info(f"📦 CONTEXT SUMMARY: {', '.join(context_items) if context_items else 'minimal'}")


# Singleton instance
_context_gatherer: Optional[GRPCContextGatherer] = None


def get_context_gatherer() -> GRPCContextGatherer:
    """Get singleton context gatherer instance"""
    global _context_gatherer
    if _context_gatherer is None:
        _context_gatherer = GRPCContextGatherer()
    return _context_gatherer

