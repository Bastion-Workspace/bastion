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

            # Check if models are properly configured
            models_configured = request_context.get("models_configured", True)  # Default to True for backward compatibility
            if not models_configured:
                grpc_request.metadata["models_not_configured_warning"] = "AI models are using system fallback defaults. Consider configuring specific models in Settings > AI Models for better performance and consistency."
                logger.warning("⚠️ CONTEXT GATHERER: Models not explicitly configured - using fallbacks")
            
            # Initialize request context if not provided
            request_context = request_context or {}
            if request_context.get("user_chat_model"):
                grpc_request.metadata["user_chat_model"] = request_context["user_chat_model"]

            # === CONVERSATION HISTORY + ATTACHMENTS (load messages once when not in state) ===
            messages_data: Optional[Dict[str, Any]] = None
            if not (state and "messages" in state):
                try:
                    from services.conversation_service import ConversationService
                    conversation_service = ConversationService()
                    conversation_service.set_current_user(user_id)
                    messages_data = await conversation_service.get_conversation_messages(
                        conversation_id=conversation_id,
                        user_id=user_id,
                        skip=0,
                        limit=20,
                    )
                    if messages_data and messages_data.get("messages"):
                        logger.info(
                            "CONTEXT: Loaded %s messages from database for conversation %s",
                            len(messages_data["messages"]),
                            conversation_id,
                        )
                except Exception as db_error:
                    logger.warning("CONTEXT: Failed to load conversation messages: %s", db_error)

            await self._add_conversation_history(
                grpc_request, state, conversation_id, user_id, messages_data=messages_data
            )
            await self._add_attachments(
                grpc_request, conversation_id, user_id, messages_data=messages_data
            )
            
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
            
            # === IMAGE DESCRIPTION CONTEXT (when agent_type is image_description) ===
            if agent_type == "image_description":
                self._add_image_description_context(grpc_request, request_context)
            # === DOCUMENT DESCRIPTION CONTEXT (when agent_type is document_description) ===
            if agent_type == "document_description":
                self._add_document_description_context(grpc_request, request_context)
            
            # === PRIMARY AGENT SELECTED (for conversation continuity) ===
            await self._add_primary_agent_selected(grpc_request, state)
            
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
        user_id: str,
        messages_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add conversation history to request. Uses state['messages'] if present, else messages_data if provided, else loads from DB."""
        try:
            messages = []

            if state and "messages" in state:
                messages = state["messages"]
            elif messages_data and "messages" in messages_data:
                from langchain_core.messages import HumanMessage, AIMessage
                for msg_dict in messages_data["messages"]:
                    role = msg_dict.get("message_type", "user")
                    content = msg_dict.get("content", "")

                    if role == "user":
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        messages.append(AIMessage(content=content))
                logger.info("CONTEXT: Added %s messages to history (from shared messages_data)", len(messages))
            else:
                try:
                    from services.conversation_service import ConversationService
                    conversation_service = ConversationService()
                    conversation_service.set_current_user(user_id)
                    loaded = await conversation_service.get_conversation_messages(
                        conversation_id=conversation_id,
                        user_id=user_id,
                        skip=0,
                        limit=20,
                    )
                    if loaded and "messages" in loaded:
                        from langchain_core.messages import HumanMessage, AIMessage
                        for msg_dict in loaded["messages"]:
                            role = msg_dict.get("message_type", "user")
                            content = msg_dict.get("content", "")

                            if role == "user":
                                messages.append(HumanMessage(content=content))
                            elif role == "assistant":
                                messages.append(AIMessage(content=content))
                        logger.info("CONTEXT: Loaded %s messages from database for conversation %s", len(messages), conversation_id)
                except Exception as db_error:
                    logger.warning("CONTEXT: Failed to load conversation history from database: %s", db_error)
            
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
    
    async def _add_attachments(
        self,
        grpc_request: orchestrator_pb2.ChatRequest,
        conversation_id: str,
        user_id: str,
        messages_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Extract attachments from latest user message and add to request metadata"""
        try:
            attachments = await self._extract_latest_message_attachments(
                conversation_id=conversation_id,
                user_id=user_id,
                messages_data=messages_data,
            )
            
            if attachments:
                import json
                grpc_request.metadata["attachments"] = json.dumps(attachments)
                attached_images = [{"data": a.get("base64_data"), "base64": a.get("base64_data")} for a in attachments if a.get("base64_data")]
                if attached_images:
                    grpc_request.metadata["attached_images"] = json.dumps(attached_images)
                logger.info(f"✅ CONTEXT: Added {len(attachments)} attachment(s) from latest message")
            else:
                logger.debug("📎 CONTEXT: No attachments found in latest message")
                
        except Exception as e:
            logger.warning(f"⚠️ CONTEXT: Failed to extract attachments: {e}")
    
    async def _extract_latest_message_attachments(
        self,
        conversation_id: str,
        user_id: str,
        messages_data: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract attachments from the latest user message in the conversation.
        Uses messages_data when provided (avoids duplicate load); otherwise loads from DB.
        Queries conversation_message_attachments when available; falls back to metadata_json.
        For images, reads file and adds base64_data for orchestrator vision/face analysis.
        """
        try:
            import base64
            import json
            from pathlib import Path
            from services.conversation_service import ConversationService
            from services.database_manager.database_helpers import fetch_all

            if not messages_data or "messages" not in messages_data:
                conversation_service = ConversationService()
                conversation_service.set_current_user(user_id)
                messages_data = await conversation_service.get_conversation_messages(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    skip=0,
                    limit=20,
                )

            if not messages_data or "messages" not in messages_data:
                return []

            messages = messages_data["messages"]
            latest_message_id = None

            for msg in reversed(messages):
                if msg.get("message_type") == "user" or msg.get("role") == "user":
                    latest_message_id = msg.get("message_id")
                    break

            if not latest_message_id:
                return []

            attachments = []

            try:
                rows = await fetch_all(
                    """
                    SELECT attachment_id, filename, content_type, file_size, file_path,
                           uploaded_at, is_image, metadata_json
                    FROM conversation_message_attachments
                    WHERE message_id = $1
                    ORDER BY uploaded_at DESC
                    """,
                    latest_message_id,
                )
                if rows:
                    for row in rows:
                        file_path = row.get("file_path")
                        content_type = row.get("content_type", "")
                        is_image = row.get("is_image", content_type.startswith("image/"))
                        base64_data = None
                        if is_image and file_path and Path(file_path).exists():
                            try:
                                with open(file_path, "rb") as f:
                                    base64_data = base64.b64encode(f.read()).decode("utf-8")
                            except Exception as e:
                                logger.warning(f"Failed to read image for attachment: {e}")
                        uploaded_at = row.get("uploaded_at")
                        attachments.append({
                            "attachment_id": str(row.get("attachment_id", "")),
                            "filename": row.get("filename", ""),
                            "content_type": content_type,
                            "file_size": row.get("file_size", 0),
                            "file_path": file_path,
                            "base64_data": base64_data,
                            "uploaded_at": uploaded_at.isoformat() if hasattr(uploaded_at, "isoformat") else str(uploaded_at),
                        })
            except Exception as e:
                logger.debug(f"message_attachments table not available or query failed: {e}")

            if attachments:
                logger.info(f"Found {len(attachments)} attachment(s) from conversation_message_attachments")
                return attachments

            metadata_json = None
            for msg in reversed(messages):
                if msg.get("message_type") == "user" or msg.get("role") == "user":
                    metadata_json = msg.get("metadata_json", {})
                    if isinstance(metadata_json, str):
                        metadata_json = json.loads(metadata_json)
                    break

            if not metadata_json:
                return []

            attachments = metadata_json.get("attachments", [])
            if not attachments:
                return []

            result = []
            for att in attachments:
                storage_path = att.get("storage_path") or att.get("file_path")
                content_type = att.get("content_type", "")
                is_image = content_type.startswith("image/") if content_type else False
                base64_data = att.get("base64_data") or att.get("data")
                if is_image and storage_path and not base64_data:
                    try:
                        if Path(storage_path).exists():
                            with open(storage_path, "rb") as f:
                                base64_data = base64.b64encode(f.read()).decode("utf-8")
                    except Exception as e:
                        logger.warning(f"Failed to read image at {storage_path}: {e}")
                result.append({
                    "attachment_id": att.get("attachment_id", ""),
                    "filename": att.get("filename", ""),
                    "content_type": content_type or "application/octet-stream",
                    "file_size": att.get("file_size") or att.get("size_bytes", 0),
                    "file_path": storage_path,
                    "base64_data": base64_data,
                    "uploaded_at": att.get("uploaded_at", ""),
                })
            logger.info(f"Found {len(result)} attachment(s) in latest user message metadata")
            return result

        except Exception as e:
            logger.warning(f"Failed to extract attachments from latest message: {e}")
            return []
    
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
            
            # Add user preferred name and AI context
            from services.settings_service import settings_service
            try:
                preferred_name = await settings_service.get_user_preferred_name(user_id)
                ai_context = await settings_service.get_user_ai_context(user_id)
                
                if preferred_name:
                    grpc_request.metadata["user_preferred_name"] = preferred_name
                    logger.info(f"✅ CONTEXT: Added preferred name: {preferred_name}")
                
                if ai_context:
                    grpc_request.metadata["user_ai_context"] = ai_context
                    logger.info(f"✅ CONTEXT: Added AI context ({len(ai_context)} chars)")
            except Exception as e:
                logger.warning(f"⚠️ CONTEXT: Failed to add user context: {e}")
            
            # Add model preferences from settings service (or keep override from request_context, e.g. external chat)
            try:
                chat_model = grpc_request.metadata.get("user_chat_model") or await settings_service.get_llm_model()
                fast_model = await settings_service.get_classification_model()
                image_model = await settings_service.get_image_generation_model()
                
                if chat_model:
                    grpc_request.metadata["user_chat_model"] = chat_model
                    logger.info(f"🎯 SENDING TO ORCHESTRATOR: user_chat_model = {chat_model}")
                if fast_model:
                    grpc_request.metadata["user_fast_model"] = fast_model
                if image_model:
                    grpc_request.metadata["user_image_model"] = image_model
                
                logger.info(f"✅ CONTEXT: Added model preferences (chat={chat_model}, fast={fast_model}, image={image_model})")
            except Exception as e:
                logger.warning(f"⚠️ CONTEXT: Failed to add model preferences: {e}")
            
            # Add user timezone to metadata (for datetime context in agents)
            try:
                user_timezone = await settings_service.get_user_timezone(user_id)
                if user_timezone:
                    grpc_request.metadata["user_timezone"] = user_timezone
                    logger.info(f"🌍 CONTEXT: Added user timezone to metadata: {user_timezone}")
            except Exception as e:
                logger.warning(f"⚠️ CONTEXT: Failed to add user timezone: {e}")
            
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
            
            logger.info(f"🔍 EDITOR CONTEXT CHECK: request_context keys={list(request_context.keys())}, active_editor={active_editor is not None}, editor_preference={editor_preference}")
            
            # CRITICAL: Always send editor_preference in metadata, even when active_editor is not sent
            # This ensures the orchestrator can block editor-gated agents when preference is 'ignore'
            if editor_preference:
                grpc_request.metadata["editor_preference"] = editor_preference
                logger.info(f"📝 EDITOR PREFERENCE: Added to metadata = '{editor_preference}'")
            
            # Skip if user said to ignore editor (don't send active_editor, but editor_preference is already in metadata)
            if editor_preference == "ignore":
                logger.info(f"⚠️ EDITOR CONTEXT: Skipping active_editor - editor_preference is 'ignore' (but editor_preference sent in metadata)")
                return
            
            # Skip if no editor context
            if not active_editor:
                logger.info(f"⚠️ EDITOR CONTEXT: Skipping - no active_editor in request_context")
                return
            
            # Validate editor context
            if not isinstance(active_editor, dict):
                logger.warning(f"⚠️ EDITOR CONTEXT: Skipping - active_editor is not a dict (type={type(active_editor)})")
                return
            
            # CRITICAL: Reject if is_editable is False or missing - this ensures stale editor data is cleared
            # EXCEPTION: Reference documents (type: reference) should ALWAYS be sent even if not editable
            # because reference_agent needs to READ them (journals, logs, etc.)
            is_editable = active_editor.get("is_editable")
            frontmatter_type = active_editor.get("frontmatter", {}).get("type", "").strip().lower()
            
            if (not is_editable or is_editable is False) and frontmatter_type != "reference":
                logger.info(f"⚠️ EDITOR CONTEXT: Skipping - active_editor.is_editable is False or missing (editor tab likely closed)")
                return
            elif frontmatter_type == "reference" and not is_editable:
                logger.info(f"📚 EDITOR CONTEXT: Including reference document even though not editable (reference_agent needs to read it)")
            
            filename = active_editor.get("filename", "")
            if not (filename.endswith(".md") or filename.endswith(".org")):
                logger.warning(f"⚠️ EDITOR CONTEXT: Skipping - filename '{filename}' does not end with .md or .org")
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
            custom_fields_added = []
            for key, value in frontmatter_data.items():
                if key not in ["type", "title", "author", "tags", "status"]:
                    frontmatter.custom_fields[key] = str(value)
                    custom_fields_added.append(key)
                    if key in ["files", "components", "protocols", "schematics", "specifications"]:
                        logger.info(f"🔍 ADDING CUSTOM FIELD: {key} = {str(value)[:200]} (original type: {type(value).__name__})")
            if custom_fields_added:
                logger.info(f"✅ CONTEXT: Added {len(custom_fields_added)} custom frontmatter field(s): {custom_fields_added}")
            else:
                logger.info(f"⚠️ CONTEXT: No custom frontmatter fields found (frontmatter keys: {list(frontmatter_data.keys())})")
            
            # Build editor message
            canonical_path = active_editor.get("canonical_path") or active_editor.get("canonicalPath") or ""
            if canonical_path:
                logger.info(f"✅ CONTEXT: Active editor canonical_path: {canonical_path}")
            else:
                logger.warning(f"⚠️ CONTEXT: Active editor has no canonical_path - relative references may fail!")
            
            # Extract cursor and selection state
            cursor_offset = active_editor.get("cursor_offset", -1)
            selection_start = active_editor.get("selection_start", -1)
            selection_end = active_editor.get("selection_end", -1)
            
            # Extract document metadata
            document_id = active_editor.get("document_id") or ""
            folder_id = active_editor.get("folder_id") or ""
            file_path = active_editor.get("file_path") or ""
            
            # Log cursor state for debugging
            if cursor_offset >= 0:
                logger.info(f"✅ CONTEXT: Cursor detected at offset {cursor_offset}")
            if selection_start >= 0 and selection_end > selection_start:
                logger.info(f"✅ CONTEXT: Selection detected from {selection_start} to {selection_end}")
            
            grpc_request.active_editor.CopyFrom(
                orchestrator_pb2.ActiveEditor(
                    is_editable=is_editable if is_editable is not None else True,
                    filename=filename,
                    language=active_editor.get("language", "markdown"),
                    content=active_editor.get("content", ""),
                    content_length=len(active_editor.get("content", "")),
                    frontmatter=frontmatter,
                    editor_preference=editor_preference,
                    canonical_path=canonical_path,
                    cursor_offset=cursor_offset,
                    selection_start=selection_start,
                    selection_end=selection_end,
                    document_id=document_id,
                    folder_id=folder_id,
                    file_path=file_path
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

    def _add_image_description_context(
        self,
        grpc_request: orchestrator_pb2.ChatRequest,
        request_context: Dict[str, Any]
    ) -> None:
        """Add image description context (image_base64, image_analysis_model) for image_description agent."""
        try:
            image_base64 = request_context.get("image_base64")
            image_analysis_model = request_context.get("image_analysis_model")
            if image_base64:
                grpc_request.metadata["image_base64"] = image_base64
                logger.info("CONTEXT: Added image_base64 for image description")
            if image_analysis_model:
                grpc_request.metadata["image_analysis_model"] = image_analysis_model
                logger.info(f"CONTEXT: Added image_analysis_model = {image_analysis_model}")
        except Exception as e:
            logger.warning(f"CONTEXT: Failed to add image description context: {e}")

    def _add_document_description_context(
        self,
        grpc_request: orchestrator_pb2.ChatRequest,
        request_context: Dict[str, Any]
    ) -> None:
        """Add document description context (document_content, document_analysis_model) for document_description agent."""
        try:
            document_content = request_context.get("document_content")
            document_analysis_model = request_context.get("document_analysis_model") or request_context.get("image_analysis_model")
            if document_content is not None:
                grpc_request.metadata["document_content"] = document_content
                logger.info("CONTEXT: Added document_content for document description")
            if document_analysis_model:
                grpc_request.metadata["document_analysis_model"] = document_analysis_model
                logger.info(f"CONTEXT: Added document_analysis_model = {document_analysis_model}")
        except Exception as e:
            logger.warning(f"CONTEXT: Failed to add document description context: {e}")
    
    async def _add_primary_agent_selected(
        self,
        grpc_request: orchestrator_pb2.ChatRequest,
        state: Optional[Dict[str, Any]]
    ) -> None:
        """
        Load agent routing metadata from backend conversation database
        
        This provides agent continuity information (primary_agent_selected, last_agent)
        to the orchestrator for proper routing decisions. Backend database is the
        source of truth for conversation-level agent metadata.
        """
        try:
            # Load agent metadata from backend database
            from services.conversation_service import ConversationService
            conversation_service = ConversationService()
            
            agent_metadata = await conversation_service.get_agent_metadata(
                conversation_id=grpc_request.conversation_id,
                user_id=grpc_request.user_id
            )
            
            if agent_metadata:
                # Add to metadata for orchestrator
                primary_agent = agent_metadata.get("primary_agent_selected")
                last_agent = agent_metadata.get("last_agent")
                
                if primary_agent:
                    grpc_request.metadata["primary_agent_selected"] = primary_agent
                    logger.info(f"📋 CONTEXT: Loaded primary_agent_selected from backend DB: {primary_agent}")
                
                if last_agent:
                    grpc_request.metadata["last_agent"] = last_agent
                    logger.debug(f"📋 CONTEXT: Loaded last_agent from backend DB: {last_agent}")
            else:
                logger.debug(f"📋 CONTEXT: No agent metadata in backend DB (new conversation or first request)")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load agent metadata from backend: {e}")
    
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

