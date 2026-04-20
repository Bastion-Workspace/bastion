"""
LLM Orchestrator gRPC Service Implementation
Handles incoming gRPC requests for LLM orchestration
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import AsyncIterator, Optional, Dict, Any, List

import grpc
from protos import orchestrator_pb2, orchestrator_pb2_grpc
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)


def _merge_request_persona_ai_name(
    merged_metadata: Dict[str, Any], request_metadata: Dict[str, Any]
) -> None:
    """If the route did not attach persona_ai_name, use the request-level persona (user default)."""
    p = request_metadata.get("persona") or {}
    if isinstance(p, dict):
        pan = (p.get("ai_name") or "").strip()
        if pan:
            merged_metadata.setdefault("persona_ai_name", pan)


# In-memory cache for conversation-level metadata (primary_agent_selected)
# This bridges the gap between different agents' checkpoints
# Key: conversation_id, Value: {"primary_agent_selected": str, "last_agent": str, "timestamp": float}
_conversation_metadata_cache = {}


class OrchestratorGRPCService(orchestrator_pb2_grpc.OrchestratorServiceServicer):
    """
    gRPC service implementation for LLM Orchestrator.

    Chat and routed skills run through Agent Factory (CustomAgentRunner + playbooks).
    """
    
    def __init__(self):
        self.is_initialized = False
        logger.info("Initializing OrchestratorGRPCService...")

    @staticmethod
    def _strip_duplicate_trailing_user_message(
        messages: List[Any],
        request_query: str,
    ) -> List[Any]:
        """
        Remove trailing user message when it duplicates the current request (UI vs external-connector).

        External connectors persist the user turn before the orchestrator call; history then includes
        that message. UI path typically does not. Uses content match and/or sequence_number vs max
        assistant sequence to avoid stripping the wrong turn when text repeats.
        """
        if not messages or not isinstance(messages[-1], HumanMessage):
            return messages
        last = messages[-1]
        last_seq = (getattr(last, "additional_kwargs", None) or {}).get("sequence_number") or 0
        max_ai_seq = 0
        for m in messages:
            if isinstance(m, AIMessage):
                s = (getattr(m, "additional_kwargs", None) or {}).get("sequence_number") or 0
                if s > max_ai_seq:
                    max_ai_seq = s
        if last.content == request_query:
            return messages[:-1]
        if last_seq and last_seq > max_ai_seq:
            return messages[:-1]
        return messages

    def _extract_shared_memory(self, request: orchestrator_pb2.ChatRequest, existing_shared_memory: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract shared_memory from proto request (active_editor, permissions, pipeline context, etc.)
        
        This is a centralized helper to avoid duplication across agent handlers.
        Used by both intent classification and agent processing.
        
        Args:
            request: gRPC ChatRequest proto
            existing_shared_memory: Optional existing shared_memory dict to merge into
            
        Returns:
            Dict with shared_memory fields extracted from request
        """
        shared_memory = existing_shared_memory.copy() if existing_shared_memory else {}
        
        # Extract active_editor
        # ROBUST CHECK: Check HasField first, but fallback to checking fields if False
        has_active_editor = request.HasField("active_editor")
        if not has_active_editor:
            # Fallback: check if essential fields are non-empty
            ae = request.active_editor
            if ae.filename or ae.content or ae.document_id:
                has_active_editor = True
                logger.info(f"🔍 SHARED MEMORY EXTRACTION: HasField('active_editor') was False but fields are present")
        
        logger.info(f"🔍 SHARED MEMORY EXTRACTION: has_active_editor={has_active_editor}")
        
        if has_active_editor:
            logger.info(f"✅ ACTIVE EDITOR RECEIVED: filename={request.active_editor.filename}, type={request.active_editor.frontmatter.type}, content_length={len(request.active_editor.content)}")
            # Parse custom_fields, converting stringified lists back to actual lists
            # The backend converts YAML lists to strings (e.g., "['./file1.md', './file2.md']")
            # We need to parse them back for reference file loading to work
            frontmatter_custom = {}
            custom_fields_count = len(request.active_editor.frontmatter.custom_fields)
            logger.info(f"🔍 CUSTOM FIELDS: Found {custom_fields_count} custom field(s) in proto")
            if custom_fields_count > 0:
                logger.info(f"🔍 CUSTOM FIELDS KEYS: {list(request.active_editor.frontmatter.custom_fields.keys())}")
            for key, value in request.active_editor.frontmatter.custom_fields.items():
                # Debug: Log what we're trying to parse
                if key in ["files", "components", "protocols", "schematics", "specifications"]:
                    logger.info(f"🔍 PARSING CUSTOM FIELD: {key} = {value} (type: {type(value).__name__})")
                
                # Try to parse stringified lists (Python repr format or JSON)
                if isinstance(value, str):
                    value_stripped = value.strip()
                    
                    # Try Python list format: "['./file1.md', './file2.md']"
                    if value_stripped.startswith('[') and value_stripped.endswith(']'):
                        try:
                            import ast
                            parsed = ast.literal_eval(value)
                            if isinstance(parsed, list):
                                frontmatter_custom[key] = parsed
                                logger.info(f"✅ PARSED {key} as Python list: {len(parsed)} items")
                                continue
                        except (ValueError, SyntaxError) as e:
                            logger.debug(f"⚠️ Failed to parse {key} as Python list: {e}")
                        # Try JSON parsing as fallback for list-like strings
                        try:
                            import json
                            parsed = json.loads(value)
                            if isinstance(parsed, list):
                                frontmatter_custom[key] = parsed
                                logger.info(f"✅ PARSED {key} as JSON list: {len(parsed)} items")
                                continue
                        except (ValueError, json.JSONDecodeError):
                            pass  # Fall through to keep as string
                    
                    # Try YAML list format (newline-separated): "- ./file1.md\n- ./file2.md"
                    elif '\n' in value and value_stripped.startswith('-'):
                        try:
                            import yaml
                            parsed = yaml.safe_load(value)
                            if isinstance(parsed, list):
                                frontmatter_custom[key] = parsed
                                logger.info(f"✅ PARSED {key} as YAML list: {len(parsed)} items")
                                continue
                        except (yaml.YAMLError, ValueError) as e:
                            logger.debug(f"⚠️ Failed to parse {key} as YAML: {e}")
                    
                    # For plain strings (like "../Characters/Nick.md"), don't try to parse as JSON
                    # This avoids unnecessary error logs for simple file paths
                
                # If not a list, keep as string
                frontmatter_custom[key] = value
                if key in ["files", "components", "protocols", "schematics", "specifications"]:
                    logger.warning(f"⚠️ {key} kept as string (not parsed as list): {value[:100]}")
            
            # Extract canonical_path from proto (backend sends it from frontend)
            canonical_path = request.active_editor.canonical_path if request.active_editor.canonical_path else None
            if canonical_path:
                logger.info(f"📄 Active editor canonical_path: {canonical_path}")
            else:
                logger.warning(f"⚠️ Active editor has no canonical_path - relative references may fail!")

            # Extract cursor and selection state
            # Backend always sets these fields (even to -1 if not available)
            cursor_offset = request.active_editor.cursor_offset if request.active_editor.cursor_offset >= 0 else -1
            selection_start = request.active_editor.selection_start if request.active_editor.selection_start >= 0 else -1
            selection_end = request.active_editor.selection_end if request.active_editor.selection_end >= 0 else -1
            
            # Extract document metadata
            document_id = request.active_editor.document_id if request.active_editor.document_id else None
            folder_id = request.active_editor.folder_id if request.active_editor.folder_id else None
            file_path = request.active_editor.file_path if request.active_editor.file_path else request.active_editor.filename
            
            # 🔒 LOCK target_document_id at request start to prevent race conditions
            # This captures which document was active when the user sent the message
            # Even if editor_ctx_cache changes mid-request (tab switch/shutdown), we use this locked value
            if document_id:
                shared_memory["target_document_id"] = document_id
                logger.info(f"🔒 LOCKED target_document_id at request start: {document_id}")
            
            # Log cursor state for debugging
            if cursor_offset >= 0:
                logger.info(f"✅ CONTEXT: Cursor detected at offset {cursor_offset}")
            if selection_start >= 0 and selection_end > selection_start:
                logger.info(f"✅ CONTEXT: Selection detected from {selection_start} to {selection_end}")
            
            shared_memory["active_editor"] = {
                "is_editable": request.active_editor.is_editable,
                "filename": request.active_editor.filename,
                "file_path": file_path,
                "canonical_path": canonical_path,  # Full filesystem path for resolving relative references
                "language": request.active_editor.language,
                "content": request.active_editor.content,
                "cursor_offset": cursor_offset,
                "selection_start": selection_start,
                "selection_end": selection_end,
                "document_id": document_id,
                "folder_id": folder_id,
                "frontmatter": {
                    "type": request.active_editor.frontmatter.type,
                    "title": request.active_editor.frontmatter.title,
                    "author": request.active_editor.frontmatter.author,
                    "tags": list(request.active_editor.frontmatter.tags),
                    "status": request.active_editor.frontmatter.status,
                    **frontmatter_custom
                }
            }
            
            # Extract editor_preference from active_editor proto (CRITICAL for editor-gated agent routing)
            if hasattr(request.active_editor, 'editor_preference') and request.active_editor.editor_preference:
                shared_memory["editor_preference"] = request.active_editor.editor_preference
                logger.info(f"📝 EDITOR PREFERENCE: Extracted from active_editor = '{request.active_editor.editor_preference}'")
        
        # Extract editor_preference from metadata as fallback (if not in active_editor)
        if "editor_preference" not in shared_memory and request.metadata and "editor_preference" in request.metadata:
            shared_memory["editor_preference"] = request.metadata["editor_preference"]
            logger.info(f"📝 EDITOR PREFERENCE: Extracted from metadata = '{request.metadata['editor_preference']}'")
        
        # Default to 'prefer' if not provided
        if "editor_preference" not in shared_memory:
            shared_memory["editor_preference"] = "prefer"
            logger.debug(f"📝 EDITOR PREFERENCE: Defaulting to 'prefer' (not provided in request)")

        # Extract active chat artifact context (artifact drawer)
        if request.HasField("active_artifact"):
            aa = request.active_artifact
            if aa.code:
                shared_memory["active_artifact"] = {
                    "artifact_type": aa.artifact_type,
                    "title": aa.title,
                    "code": aa.code,
                    "language": aa.language,
                }

        # Extract active data workspace context
        if request.HasField("active_data_workspace"):
            dw = request.active_data_workspace
            schema = [
                {"name": c.name, "type": c.type, "description": c.description}
                for c in dw.columns
            ]
            visible_rows = []
            if dw.visible_rows_json:
                try:
                    visible_rows = json.loads(dw.visible_rows_json)
                except (TypeError, ValueError):
                    pass
            shared_memory["active_data_workspace"] = {
                "workspace_id": dw.workspace_id,
                "workspace_name": dw.workspace_name,
                "database_id": dw.database_id,
                "database_name": dw.database_name,
                "table_id": dw.table_id,
                "table_name": dw.table_name,
                "row_count": dw.row_count,
                "schema": schema,
                "visible_rows": visible_rows,
                "visible_row_count": dw.visible_row_count,
            }

        # Extract pipeline context
        if request.HasField("pipeline_context"):
            shared_memory["active_pipeline_id"] = request.pipeline_context.active_pipeline_id
            shared_memory["pipeline_preference"] = request.pipeline_context.pipeline_preference
        
        # Extract permission grants
        if request.HasField("permission_grants"):
            if request.permission_grants.web_search_permission:
                shared_memory["web_search_permission"] = True
            if request.permission_grants.web_crawl_permission:
                shared_memory["web_crawl_permission"] = True
            if request.permission_grants.file_write_permission:
                shared_memory["file_write_permission"] = True
            if request.permission_grants.external_api_permission:
                shared_memory["external_api_permission"] = True
        
        # Extract model preferences from request.metadata (CRITICAL for user model selection)
        if request.metadata:
            if "user_chat_model" in request.metadata:
                shared_memory["user_chat_model"] = request.metadata["user_chat_model"]
                logger.info(f"🎯 EXTRACTED user_chat_model from metadata: {request.metadata['user_chat_model']}")
            if "user_fast_model" in request.metadata:
                shared_memory["user_fast_model"] = request.metadata["user_fast_model"]
                logger.debug(f"🎯 EXTRACTED user_fast_model from metadata: {request.metadata['user_fast_model']}")
            if "user_image_model" in request.metadata:
                shared_memory["user_image_model"] = request.metadata["user_image_model"]
                logger.debug(f"🎯 EXTRACTED user_image_model from metadata: {request.metadata['user_image_model']}")
            if "user_timezone" in request.metadata:
                shared_memory["user_timezone"] = request.metadata["user_timezone"]
                logger.info(f"🌍 EXTRACTED user_timezone from metadata: {request.metadata['user_timezone']}")
            
            # Extract attachments from metadata
            if "attachments" in request.metadata:
                try:
                    import json
                    attachments_json = request.metadata["attachments"]
                    if isinstance(attachments_json, str):
                        attachments = json.loads(attachments_json)
                    else:
                        attachments = attachments_json
                    
                    # Separate attachments by type
                    shared_memory["attached_images"] = [
                        att for att in attachments 
                        if att.get("content_type", "").startswith("image/")
                    ]
                    shared_memory["attached_documents"] = [
                        att for att in attachments 
                        if att.get("content_type") in ["application/pdf", "text/plain", "text/markdown"]
                    ]
                    shared_memory["attached_audio"] = [
                        att for att in attachments 
                        if att.get("content_type", "").startswith("audio/")
                    ]
                    
                    if shared_memory.get("attached_images") or shared_memory.get("attached_documents") or shared_memory.get("attached_audio"):
                        logger.info(f"📎 SHARED MEMORY: Added {len(shared_memory.get('attached_images', []))} image(s), {len(shared_memory.get('attached_documents', []))} document(s), {len(shared_memory.get('attached_audio', []))} audio file(s)")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to parse attachments from metadata: {e}")
            if "image_base64" in request.metadata:
                shared_memory["image_base64"] = request.metadata["image_base64"]
                logger.debug("📎 SHARED MEMORY: Added image_base64 for image description routing")
        
        # Fallback: use persona timezone so custom agents get correct date/time when metadata lacks user_timezone
        if "user_timezone" not in shared_memory and request.HasField("persona"):
            shared_memory["user_timezone"] = request.persona.timezone or "UTC"
            logger.debug("EXTRACTED user_timezone from persona (fallback)")
        
        return shared_memory
    
    def _extract_conversation_context(self, request: orchestrator_pb2.ChatRequest) -> dict:
        """
        Extract conversation context from proto request for intent classification
        
        Builds context dict matching backend structure for 1:1 parity.
        """
        context = {
            "messages": [],
            "shared_memory": {},
            "conversation_intelligence": {}
        }
        
        # Extract conversation history
        last_assistant_message = None
        for msg in request.conversation_history:
            context["messages"].append({
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp
            })
            # Track last assistant message for intent classifier context
            if msg.role == "assistant":
                last_assistant_message = msg.content
        
        # Store last agent response in shared_memory for intent classifier
        if last_assistant_message:
            context["shared_memory"]["last_response"] = last_assistant_message
            logger.debug(f"📋 Stored last agent response ({len(last_assistant_message)} chars) for intent classifier context")
        
        # Extract primary_agent_selected from metadata if provided (for conversation continuity)
        # Note: This is checked BEFORE checkpoint merge, so it's expected to be None for new conversations
        # The checkpoint shared_memory will be merged later and will contain primary_agent_selected if it exists
        if request.metadata and "primary_agent_selected" in request.metadata:
            context["shared_memory"]["primary_agent_selected"] = request.metadata["primary_agent_selected"]
            logger.info(f"📋 CONTEXT: Extracted primary_agent_selected from metadata: {request.metadata['primary_agent_selected']}")
        else:
            # This is expected for new conversations - checkpoint will have it if conversation exists
            logger.debug(f"📋 CONTEXT: No primary_agent_selected in metadata (will check checkpoint shared_memory)")
        
        # Extract last_agent from metadata if provided (for conversation continuity)
        if request.metadata and "last_agent" in request.metadata:
            context["shared_memory"]["last_agent"] = request.metadata["last_agent"]
            logger.info(f"📋 CONTEXT: Extracted last_agent from metadata: {request.metadata['last_agent']}")
        else:
            logger.debug(f"📋 CONTEXT: No last_agent in metadata")
        
        # Extract user_chat_model from metadata (for title generation and agent model selection)
        if request.metadata and "user_chat_model" in request.metadata:
            context["shared_memory"]["user_chat_model"] = request.metadata["user_chat_model"]
            logger.debug(f"📋 CONTEXT: Extracted user_chat_model from metadata: {request.metadata['user_chat_model']}")
        
        # Use centralized shared_memory extraction
        context["shared_memory"] = self._extract_shared_memory(request, context["shared_memory"])
        
        # Extract conversation intelligence (if provided)
        if request.HasField("conversation_intelligence"):
            # This would be populated if backend sends it
            # For now, basic structure
            context["conversation_intelligence"] = {
                "agent_outputs": {}
            }
        
        return context

    async def _generate_and_yield_title(
        self,
        request: orchestrator_pb2.ChatRequest,
        user_message: str,
        response_text: str,
    ):
        """
        Generate a conversation title via TitleGenerationService and schedule DB update.
        Returns a ChatChunk with type="title" for the caller to yield.
        Only used for first turn (caller checks not request.conversation_history).
        """
        try:
            from orchestrator.services.title_generation_service import get_title_generation_service
            title_service = get_title_generation_service()
            generated_title = await title_service.generate_title(
                user_message=user_message or "",
                agent_response=response_text or None,
            )
            if not generated_title:
                return None
            try:
                from orchestrator.backend_tool_client import get_backend_tool_client
                backend_client = await get_backend_tool_client()
                asyncio.create_task(
                    backend_client.update_conversation_title(
                        conversation_id=request.conversation_id,
                        title=generated_title,
                        user_id=request.user_id,
                    )
                )
            except Exception as e:
                logger.warning("Failed to queue title update: %s", e)
            return orchestrator_pb2.ChatChunk(
                type="title",
                message=generated_title,
                timestamp=datetime.now().isoformat(),
                agent_name="system",
            )
        except Exception as e:
            logger.warning("Title generation failed: %s", e)
            return None
    
    async def _load_checkpoint_shared_memory(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load shared_memory from checkpoint state for conversation continuity
        
        This ensures primary_agent_selected and other continuity data is available
        for intent classification before the agent processes the request.
        
        Args:
            metadata: Metadata dict with user_id and conversation_id
            
        Returns:
            Dict with shared_memory from checkpoint, or empty dict if not found
        """
        try:
            from orchestrator.agents.custom_agent_runner import CustomAgentRunner

            runner = CustomAgentRunner()
            config = runner._get_checkpoint_config(metadata)
            workflow = await runner._get_workflow()
            shared_memory = await runner._load_checkpoint_shared_memory(workflow, config)
            return shared_memory

        except Exception as e:
            logger.debug(f"⚠️ Failed to load checkpoint shared_memory in gRPC service: {e}")
            return {}
    
    async def _llm_reselect_skill(
        self,
        query: str,
        rejected_skills: set,
        registry,
        metadata: Dict[str, Any],
        conversation_context: Dict[str, Any],
        editor_context=None,
    ) -> Optional[str]:
        """Run LLM skill selection excluding already-rejected skills."""
        try:
            from orchestrator.routes.route_selector import llm_select_route_ranked
            shared = conversation_context.get("shared_memory") or {}
            has_image = bool(shared.get("attached_images") or shared.get("image_base64"))
            eligible, _ = registry.filter_eligible(
                query=query,
                editor_context=editor_context,
                conversation_context=conversation_context,
                editor_preference=shared.get("editor_preference", "prefer"),
                has_image_context=has_image,
            )
            eligible = [s for s in eligible if s.name not in rejected_skills]
            if not eligible:
                return None
            routing_result = await llm_select_route_ranked(
                eligible,
                query,
                editor_context=editor_context,
                conversation_context=conversation_context,
                metadata=metadata,
            )
            selected = routing_result.primary
            if selected and selected not in rejected_skills:
                logger.info("LLM re-selection picked: %s (after rejections: %s)", selected, rejected_skills)
                fallback = [s for s in routing_result.fallback_stack if s not in rejected_skills]
                sm = metadata.get("shared_memory") or {}
                sm["routing_fallback_stack"] = fallback
                return selected
            return None
        except Exception as e:
            logger.warning("LLM re-selection failed: %s", e)
            return None

    def _save_agent_identity_to_cache(self, agent_result: Dict[str, Any], conversation_id: str) -> None:
        """
        Save the agent's identity (primary_agent_selected) to the in-memory cache.
        
        This bridges the gap between different agents' checkpoints, ensuring
        conversation continuity when switching between agents.
        
        Args:
            agent_result: Result dict from agent.process() containing shared_memory
            conversation_id: The conversation ID
        """
        try:
            if not conversation_id:
                return
            if not isinstance(agent_result, dict):
                logger.debug(f"Skipping agent identity cache: agent_result is not a dict (type={type(agent_result).__name__})")
                return

            # Extract primary_agent_selected and optional agent_profile_id from agent's result
            result_shared_memory = agent_result.get("shared_memory", {})
            primary_agent = result_shared_memory.get("primary_agent_selected")
            last_agent = result_shared_memory.get("last_agent")
            agent_profile_id = result_shared_memory.get("agent_profile_id")

            if not primary_agent and not agent_profile_id:
                # Agent didn't set primary_agent_selected or agent_profile_id, skip
                return

            # Store in cache
            import time
            cached = {
                "primary_agent_selected": primary_agent or "custom_agent",
                "last_agent": (last_agent or primary_agent or "custom_agent"),
                "timestamp": time.time(),
            }
            if agent_profile_id:
                cached["agent_profile_id"] = agent_profile_id
            _conversation_metadata_cache[conversation_id] = cached

            logger.info(f"✅ CACHED AGENT IDENTITY: primary_agent_selected = '{cached['primary_agent_selected']}', last_agent = '{cached['last_agent']}' (conversation: {conversation_id})")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save agent identity to cache: {e}")
    
    def _load_agent_identity_from_cache(self, conversation_id: str) -> Dict[str, Any]:
        """
        Load agent identity from the in-memory cache.
        
        This provides a fallback when checkpoint loading doesn't give us
        the most recent agent identity (e.g., when a different agent ran last).
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            Dict with primary_agent_selected and last_agent, or empty dict
        """
        try:
            if not conversation_id:
                return {}
            
            cached = _conversation_metadata_cache.get(conversation_id, {})
            if cached:
                logger.debug(f"📦 LOADED FROM CACHE: primary_agent_selected = '{cached.get('primary_agent_selected')}' (conversation: {conversation_id})")
            
            return cached
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load agent identity from cache: {e}")
            return {}
    
    async def StreamChat(
        self,
        request: orchestrator_pb2.ChatRequest,
        context: grpc.aio.ServicerContext
    ) -> AsyncIterator[orchestrator_pb2.ChatChunk]:
        """
        Stream chat responses back to client
        
        Dispatches through the unified orchestrator (routes, skills, Agent Factory playbooks).
        Includes cancellation support - detects client disconnect and cancels operations
        """
        # Create cancellation token for this request
        cancellation_token = asyncio.Event()
        
        # Monitor for client disconnect
        async def monitor_cancellation():
            """Monitor gRPC context for client disconnect"""
            while not context.cancelled():
                await asyncio.sleep(0.1)  # Check every 100ms
            # Client disconnected - signal cancellation
            if not cancellation_token.is_set():
                logger.info("🛑 Client disconnected - signalling cancellation")
                cancellation_token.set()
        
        # Start cancellation monitor
        monitor_task = asyncio.create_task(monitor_cancellation())
        
        try:
            query_preview = request.query[:100] if len(request.query) > 100 else request.query
            logger.debug(f"StreamChat request from user {request.user_id}: {query_preview}")

            # Parse metadata into dictionary
            metadata = dict(request.metadata) if request.metadata else {}
            logger.debug(f"🔍 RECEIVED METADATA: user_chat_model = {metadata.get('user_chat_model')}")

            # Check for model configuration warning
            model_warning = metadata.get("models_not_configured_warning")
            if model_warning:
                logger.warning(f"⚠️ MODEL CONFIG WARNING: {model_warning}")
                # Yield a warning message to the user
                yield orchestrator_pb2.ChatChunk(
                    type="warning",
                    message=f"⚠️ Model Configuration: {model_warning}",
                    timestamp=datetime.now().isoformat(),
                    agent_name="system"
                )
            
            # Add required fields for checkpointing (conversation_id, user_id)
            metadata["conversation_id"] = request.conversation_id
            metadata["user_id"] = request.user_id
            
            # Extract persona from proto request and add to metadata (for all agents)
            if request.HasField("persona"):
                persona_dict = {
                    "ai_name": request.persona.ai_name if request.persona.ai_name else "Alex",
                    "persona_style": request.persona.persona_style if request.persona.persona_style else "professional",
                    "political_bias": request.persona.political_bias if request.persona.political_bias else "neutral",
                    "timezone": request.persona.timezone if request.persona.timezone else "UTC"
                }
                if request.persona.custom_preferences:
                    persona_dict["custom_preferences"] = dict(request.persona.custom_preferences)
                metadata["persona"] = persona_dict
                logger.debug(f"✅ PERSONA: Extracted persona for agents (ai_name={persona_dict['ai_name']}, style={persona_dict['persona_style']})")
            else:
                # Default persona if not provided
                metadata["persona"] = {
                    "ai_name": "Alex",
                    "persona_style": "professional",
                    "political_bias": "neutral",
                    "timezone": "UTC"
                }
                logger.debug("📋 PERSONA: No persona provided, using defaults")
            
            # Load checkpoint shared_memory for conversation continuity (primary_agent_selected, etc.)
            checkpoint_shared_memory = await self._load_checkpoint_shared_memory(metadata)
            
            # Also load from cache (bridges gap between different agents' checkpoints)
            # Cache takes priority over checkpoint for agent identity
            cached_agent_identity = self._load_agent_identity_from_cache(request.conversation_id)
            if cached_agent_identity:
                # Merge cache into checkpoint, cache values take precedence
                if not checkpoint_shared_memory:
                    checkpoint_shared_memory = {}
                checkpoint_shared_memory["primary_agent_selected"] = cached_agent_identity.get("primary_agent_selected")
                checkpoint_shared_memory["last_agent"] = cached_agent_identity.get("last_agent")
                if cached_agent_identity.get("agent_profile_id"):
                    checkpoint_shared_memory["agent_profile_id"] = cached_agent_identity["agent_profile_id"]
                logger.debug(f"📚 Merged cache into checkpoint: primary_agent={cached_agent_identity.get('primary_agent_selected')}")
            
            # Build conversation context from proto fields for intent classification
            conversation_context = self._extract_conversation_context(request)
            
            # Extract shared_memory from request (includes editor_preference)
            request_shared_memory = self._extract_shared_memory(request)
            
            # 🔒 Extract target_document_id as top-level metadata field for easy agent access
            # This locked document ID prevents race conditions during tab switches
            if "target_document_id" in request_shared_memory:
                metadata["target_document_id"] = request_shared_memory["target_document_id"]
                logger.debug(f"🔒 Added target_document_id to metadata: {metadata['target_document_id']}")
            
            # Merge checkpoint shared_memory into context for intent classifier
            # Note: active_editor should NOT be in checkpoint (cleared by agents before save)
            # But we defensively clear it here if it somehow persists (safety net)
            if checkpoint_shared_memory:
                # Defensive: Clear active_editor from checkpoint if present (shouldn't be there)
                # Agents clear it before checkpoint save, but this is a safety net
                if "active_editor" in checkpoint_shared_memory:
                    logger.debug(f"📝 EDITOR: Found active_editor in checkpoint (shouldn't be there) - clearing defensively")
                    checkpoint_shared_memory = checkpoint_shared_memory.copy()
                    del checkpoint_shared_memory["active_editor"]
                
                # CRITICAL: editor_preference is request-scoped, not conversation-scoped
                # Clear it from checkpoint so current request's value always wins
                if "editor_preference" in checkpoint_shared_memory:
                    logger.debug(f"📝 EDITOR PREFERENCE: Clearing from checkpoint (request-scoped, not conversation-scoped)")
                    checkpoint_shared_memory = checkpoint_shared_memory.copy()
                    del checkpoint_shared_memory["editor_preference"]
                
                conversation_context["shared_memory"].update(checkpoint_shared_memory)
                # Log specifically about agent continuity for debugging
                primary_agent = checkpoint_shared_memory.get("primary_agent_selected")
                last_agent = checkpoint_shared_memory.get("last_agent")
                if primary_agent or last_agent:
                    logger.debug(f"📚 Loaded agent continuity from checkpoint: primary_agent={primary_agent}, last_agent={last_agent}")
                else:
                    logger.debug(f"📚 Merged checkpoint shared_memory (no agent continuity): {list(checkpoint_shared_memory.keys())}")
            
            # Log editor_preference before merge
            request_editor_pref = request_shared_memory.get("editor_preference", "not_set")
            logger.debug(f"📝 EDITOR PREFERENCE: From request = '{request_editor_pref}'")
            
            # Merge request shared_memory (current editor_preference takes precedence)
            conversation_context["shared_memory"].update(request_shared_memory)
            
            # Log final editor_preference after merge
            final_editor_pref = conversation_context["shared_memory"].get("editor_preference", "not_set")
            logger.debug(f"📝 EDITOR PREFERENCE: Final (after merge) = '{final_editor_pref}'")
            
            # Determine which agent to use via skill discovery (or intent classification fallback when disabled)
            primary_agent_name = None
            agent_type = None  # Initialize to None - will be set by routing logic or skill discovery
            discovered_route = None  # Set by short-circuit routes or route discovery
            custom_agent_profile_id = metadata.get("agent_profile_id")  # Agent Factory: route to custom agent when set
            if not custom_agent_profile_id:
                conv_shared = conversation_context.get("shared_memory", {})
                if conv_shared.get("primary_agent_selected") == "custom_agent" and conv_shared.get("agent_profile_id"):
                    custom_agent_profile_id = conv_shared["agent_profile_id"]
                    metadata["agent_profile_id"] = custom_agent_profile_id
                    logger.info("Restored agent_profile_id from conversation context (sticky routing): %s", custom_agent_profile_id)
            from orchestrator.engines.unified_dispatch import _ensure_routes_loaded
            from orchestrator.routes import get_route_registry
            _ensure_routes_loaded()
            registry = get_route_registry()

            # SHORT-CIRCUIT ROUTING: Strip "/help" prefix and continue with normal discovery
            query_lower = request.query.lower().strip()
            if query_lower.startswith("/help"):
                cleaned_query = request.query[5:].strip()
                request.query = cleaned_query if cleaned_query else "What can you help me with?"
                logger.debug("Stripped /help prefix; proceeding with normal skill discovery")
            if request.agent_type and request.agent_type != "auto":
                # Explicit agent routing provided by backend
                agent_type = request.agent_type
                primary_agent_name = agent_type
                discovered_route = registry.get(agent_type)
                logger.info(f"EXPLICIT ROUTING: {agent_type} (reason: {request.routing_reason or 'not specified'})")
            
            # If agent_type is still None, run skill discovery
            editor_context = None
            if agent_type is None:
                if not custom_agent_profile_id:
                    try:
                        from orchestrator.routes.definitions import load_auto_routable_agents
                        await load_auto_routable_agents(request.user_id)
                    except Exception as e:
                        logger.warning("load_auto_routable_agents failed: %s", e)
                active_editor = (request_shared_memory or {}).get("active_editor") or conversation_context.get("shared_memory", {}).get("active_editor")
                if active_editor:
                    fm = (active_editor.get("frontmatter") or {})
                    editor_context = {
                        "type": (fm.get("type") or "").strip().lower() or None,
                        "filename": active_editor.get("filename"),
                        "language": active_editor.get("language"),
                    }
                    if not editor_context["type"] and (editor_context.get("filename") or "").lower().endswith(".org"):
                        editor_context["type"] = "org"
                shared = conversation_context.get("shared_memory") or {}
                has_image = bool(shared.get("attached_images") or shared.get("image_base64"))
                eligible, instant_route = registry.filter_eligible(
                    query=request.query,
                    editor_context=editor_context,
                    conversation_context=conversation_context,
                    editor_preference=shared.get("editor_preference", "prefer"),
                    has_image_context=has_image,
                )
                # When user has a document open and editor_preference is "prefer", pin to the
                # single editor skill dedicated to that document type so we skip compound plans
                # (e.g. outline open -> outline_editing only).
                if instant_route is None and eligible and editor_context and shared.get("editor_preference") == "prefer":
                    from orchestrator.routes.route_schema import EngineType
                    editor_type = (editor_context.get("type") or "").strip().lower()
                    if editor_type:
                        dedicated = [
                            s for s in eligible
                            if s.engine == EngineType.CUSTOM_AGENT and (s.editor_types or []) == [editor_type]
                        ]
                        if len(dedicated) == 1:
                            instant_route = dedicated[0].name
                            logger.info(
                                "Skill discovery: %s (editor-type pin: type=%s)",
                                instant_route,
                                editor_type,
                            )
                if instant_route:
                    agent_type = instant_route
                    discovered_route = registry.get(agent_type)
                    primary_agent_name = f"{agent_type}_agent" if not agent_type.endswith("_agent") else agent_type
                    logger.info("Skill discovery: %s (instant)", agent_type)
                elif not custom_agent_profile_id:
                    default_profile_id = metadata.get("default_agent_profile_id")
                    if default_profile_id:
                        custom_agent_profile_id = default_profile_id
                        metadata["agent_profile_id"] = default_profile_id
                        primary_agent_name = "custom_agent"
                        logger.info("Default agent profile: %s", default_profile_id)
                    else:
                        agent_type = "chat"
                        discovered_route = registry.get("chat")
                        primary_agent_name = "chat_agent"
                        logger.info("No default agent, fallback to chat")
                else:
                    agent_type = "chat"
                    discovered_route = registry.get("chat")
                    primary_agent_name = "chat_agent"
                    logger.info("Route discovery: no eligible routes, fallback to chat")
                if "shared_memory" not in metadata:
                    metadata["shared_memory"] = {}
                metadata["shared_memory"]["primary_agent_selected"] = primary_agent_name or agent_type
                conversation_context["shared_memory"]["primary_agent_selected"] = primary_agent_name or agent_type

            if not discovered_route:
                discovered_route = registry.get("chat")
                primary_agent_name = "chat_agent"

            if custom_agent_profile_id:
                discovered_route = None

            # Parse conversation history for agent (sequence_number / created_at for robust merge)
            messages = []
            for msg in request.conversation_history:
                kwargs: Dict[str, Any] = {}
                meta = dict(msg.metadata) if msg.metadata else {}
                seq_raw = (meta.get("sequence_number") or "").strip()
                if seq_raw:
                    try:
                        kwargs["sequence_number"] = int(seq_raw)
                    except ValueError:
                        pass
                ts = (getattr(msg, "timestamp", None) or "").strip()
                if ts:
                    kwargs["created_at"] = ts
                if msg.role == "user":
                    messages.append(HumanMessage(content=msg.content, additional_kwargs=kwargs))
                elif msg.role == "assistant":
                    tcs = (meta.get("tool_call_summary") or "").strip()
                    if tcs:
                        kwargs["tool_call_summary"] = tcs
                    messages.append(AIMessage(content=msg.content, additional_kwargs=kwargs))

            # External connectors persist the user message before calling the orchestrator, so history
            # can include the current turn. Strip trailing duplicate user (content match or seq-only).
            messages = self._strip_duplicate_trailing_user_message(messages, request.query or "")

            # Skill dispatch: unified dispatcher
            line_dispatch_on = (
                metadata.get("line_dispatch_mode") == "true" and metadata.get("ceo_profile_id")
            )
            if line_dispatch_on:
                from orchestrator.engines.unified_dispatch import get_unified_dispatcher

                dispatcher = get_unified_dispatcher()
                shared_memory = self._extract_shared_memory(request, metadata.get("shared_memory", {}))
                dispatch_metadata = {
                    "user_id": request.user_id,
                    "conversation_id": request.conversation_id,
                    "shared_memory": shared_memory,
                    **{k: v for k, v in metadata.items() if k != "shared_memory"},
                }
                if not request.conversation_history:
                    placeholder_title = (request.query or "New conversation").strip()[:50]
                    if placeholder_title:
                        yield orchestrator_pb2.ChatChunk(
                            type="title",
                            message=placeholder_title,
                            timestamp=datetime.now().isoformat(),
                            agent_name="system",
                        )
                        try:
                            from orchestrator.backend_tool_client import get_backend_tool_client

                            backend_client = await get_backend_tool_client()
                            asyncio.create_task(
                                backend_client.update_conversation_title(
                                    conversation_id=request.conversation_id,
                                    title=placeholder_title,
                                    user_id=request.user_id,
                                )
                            )
                        except Exception as e:
                            logger.warning("Failed to queue placeholder title update (line dispatch): %s", e)
                start_time = datetime.now()
                pending_complete_chunk = None
                accumulated_content = ""
                async for chunk in dispatcher.dispatch_line(
                    request.query,
                    dispatch_metadata,
                    messages,
                    cancellation_token,
                ):
                    if chunk.type == "complete":
                        pending_complete_chunk = chunk
                        continue
                    if chunk.type == "content" and chunk.message:
                        accumulated_content += chunk.message
                    yield chunk
                if pending_complete_chunk is not None:
                    duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
                    merged_metadata = (
                        dict(pending_complete_chunk.metadata) if pending_complete_chunk.metadata else {}
                    )
                    merged_metadata["duration_ms"] = str(duration_ms)
                    _merge_request_persona_ai_name(merged_metadata, metadata)
                    yield orchestrator_pb2.ChatChunk(
                        type=pending_complete_chunk.type,
                        message=pending_complete_chunk.message,
                        timestamp=pending_complete_chunk.timestamp,
                        agent_name=pending_complete_chunk.agent_name,
                        metadata=merged_metadata,
                        tools_used=list(pending_complete_chunk.tools_used),
                    )
                self._save_agent_identity_to_cache(
                    {
                        "shared_memory": {
                            "primary_agent_selected": "line_dispatch",
                            "last_agent": "line_dispatch",
                        }
                    },
                    request.conversation_id,
                )
                return

            if custom_agent_profile_id:
                from orchestrator.engines.unified_dispatch import get_unified_dispatcher
                dispatcher = get_unified_dispatcher()
                shared_memory = self._extract_shared_memory(request, metadata.get("shared_memory", {}))
                dispatch_metadata = {
                    "user_id": request.user_id,
                    "conversation_id": request.conversation_id,
                    "shared_memory": shared_memory,
                    **{k: v for k, v in metadata.items() if k != "shared_memory"},
                }
                if not request.conversation_history:
                    placeholder_title = (request.query or "New conversation").strip()[:50]
                    if placeholder_title:
                        yield orchestrator_pb2.ChatChunk(
                            type="title",
                            message=placeholder_title,
                            timestamp=datetime.now().isoformat(),
                            agent_name="system",
                        )
                        try:
                            from orchestrator.backend_tool_client import get_backend_tool_client
                            backend_client = await get_backend_tool_client()
                            asyncio.create_task(
                                backend_client.update_conversation_title(
                                    conversation_id=request.conversation_id,
                                    title=placeholder_title,
                                    user_id=request.user_id,
                                )
                            )
                        except Exception as e:
                            logger.warning("Failed to queue placeholder title update: %s", e)
                start_time = datetime.now()
                pending_complete_chunk = None
                accumulated_content = ""
                async for chunk in dispatcher.dispatch_custom_agent(
                    custom_agent_profile_id,
                    request.query,
                    dispatch_metadata,
                    messages,
                    cancellation_token,
                ):
                    if chunk.type == "complete":
                        pending_complete_chunk = chunk
                        continue
                    if chunk.type == "content" and chunk.message:
                        accumulated_content += chunk.message
                    yield chunk
                if pending_complete_chunk is not None:
                    duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
                    merged_metadata = (
                        dict(pending_complete_chunk.metadata) if pending_complete_chunk.metadata else {}
                    )
                    merged_metadata["duration_ms"] = str(duration_ms)
                    _merge_request_persona_ai_name(merged_metadata, metadata)
                    yield orchestrator_pb2.ChatChunk(
                        type=pending_complete_chunk.type,
                        message=pending_complete_chunk.message,
                        timestamp=pending_complete_chunk.timestamp,
                        agent_name=pending_complete_chunk.agent_name,
                        metadata=merged_metadata,
                        tools_used=list(pending_complete_chunk.tools_used),
                    )
                    if not request.conversation_history:
                        # Skip LLM title generation for Agent Line runs (detected via line_id/team_id in metadata)
                        if not (metadata.get("line_id") or metadata.get("team_id")):
                            title_chunk = await self._generate_and_yield_title(
                                request, request.query or "", accumulated_content
                            )
                            if title_chunk:
                                yield title_chunk
                self._save_agent_identity_to_cache(
                    {
                        "shared_memory": {
                            "primary_agent_selected": "custom_agent",
                            "last_agent": "custom_agent",
                            "agent_profile_id": custom_agent_profile_id,
                        }
                    },
                    request.conversation_id,
                )
                return
            if discovered_route:
                if not request.conversation_history:
                    title = (request.query or "New conversation").strip()[:50]
                    if title:
                        yield orchestrator_pb2.ChatChunk(
                            type="title",
                            message=title,
                            timestamp=datetime.now().isoformat(),
                            agent_name="system",
                        )
                        try:
                            from orchestrator.backend_tool_client import get_backend_tool_client
                            backend_client = await get_backend_tool_client()
                            asyncio.create_task(
                                backend_client.update_conversation_title(
                                    conversation_id=request.conversation_id,
                                    title=title,
                                    user_id=request.user_id
                                )
                            )
                        except Exception as e:
                            logger.warning("Failed to queue title update: %s", e)
                shared_memory = self._extract_shared_memory(request, metadata.get("shared_memory", {}))
                shared_memory.setdefault("routing_fallback_stack", [])
                dispatch_metadata = {
                    "user_id": request.user_id,
                    "conversation_id": request.conversation_id,
                    "shared_memory": shared_memory,
                    **{k: v for k, v in metadata.items() if k != "shared_memory"},
                }
                from orchestrator.engines.unified_dispatch import get_unified_dispatcher
                dispatcher = get_unified_dispatcher()
                start_time = datetime.now()
                max_retries = 2
                current_discovered_route = discovered_route
                current_skill_name = discovered_route.name
                rejected_skills = set()

                for attempt in range(max_retries + 1):
                    if attempt > 0:
                        fallback_stack = dispatch_metadata.get("shared_memory", {}).get("routing_fallback_stack", [])
                        next_skill = None
                        while fallback_stack and next_skill is None:
                            candidate = fallback_stack.pop(0)
                            if candidate not in rejected_skills:
                                next_skill = candidate
                        if next_skill is None:
                            next_skill = await self._llm_reselect_skill(
                                request.query, rejected_skills, registry, metadata,
                                conversation_context, editor_context,
                            )
                        if next_skill is None:
                            logger.info("All fallbacks exhausted; ending dispatch")
                            break
                        current_discovered_route = registry.get(next_skill) or registry.get("chat")
                        current_skill_name = current_discovered_route.name if current_discovered_route else "chat"
                        logger.info("Skill rejected, retrying with: %s (attempt %d)", current_skill_name, attempt)

                    rejected = False
                    pending_complete_chunk = None
                    accumulated_content = ""
                    async for chunk in dispatcher.dispatch(
                        current_skill_name, request.query, dispatch_metadata, messages, cancellation_token
                    ):
                        if chunk.type == "rejected":
                            rejected = True
                            rejected_skills.add(current_skill_name)
                            break
                        if chunk.type == "complete":
                            if pending_complete_chunk is not None:
                                yield pending_complete_chunk
                            pending_complete_chunk = chunk
                            continue
                        if chunk.type == "content" and chunk.message:
                            accumulated_content += chunk.message
                        yield chunk

                    if not rejected:
                        if pending_complete_chunk is not None:
                            import json
                            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
                            merged_metadata = (
                                dict(pending_complete_chunk.metadata) if pending_complete_chunk.metadata else {}
                            )
                            merged_metadata["duration_ms"] = str(duration_ms)
                            merged_metadata["skills_used"] = json.dumps([current_discovered_route.name])
                            _merge_request_persona_ai_name(merged_metadata, metadata)
                            yield orchestrator_pb2.ChatChunk(
                                type=pending_complete_chunk.type,
                                message=pending_complete_chunk.message,
                                timestamp=pending_complete_chunk.timestamp,
                                agent_name=pending_complete_chunk.agent_name,
                                metadata=merged_metadata,
                                tools_used=list(pending_complete_chunk.tools_used),
                            )
                            if not request.conversation_history:
                                # Skip LLM title generation for Agent Line runs (detected via line_id/team_id in metadata)
                                if not (metadata.get("line_id") or metadata.get("team_id")):
                                    title_chunk = await self._generate_and_yield_title(
                                        request, request.query or "", accumulated_content
                                    )
                                    if title_chunk:
                                        yield title_chunk
                        agent_name = primary_agent_name or current_discovered_route.name
                        self._save_agent_identity_to_cache(
                            {"shared_memory": {"primary_agent_selected": agent_name, "last_agent": agent_name}},
                            request.conversation_id,
                        )
                        return

                agent_name = primary_agent_name or current_discovered_route.name
                self._save_agent_identity_to_cache(
                    {"shared_memory": {"primary_agent_selected": agent_name, "last_agent": agent_name}},
                    request.conversation_id,
                )
                return

            
        except asyncio.CancelledError:
            logger.info("🛑 StreamChat cancelled by client")
            yield orchestrator_pb2.ChatChunk(
                type="error",
                message="Operation cancelled by user",
                timestamp=datetime.now().isoformat(),
                agent_name="system"
            )
        except Exception as e:
            logger.error(f"Error in StreamChat: {e}")
            import traceback
            traceback.print_exc()
            yield orchestrator_pb2.ChatChunk(
                type="error",
                message=f"Error: {str(e)}",
                timestamp=datetime.now().isoformat(),
                agent_name="system"
            )
        finally:
            # Clean up cancellation monitor
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
    
    async def HealthCheck(
        self,
        request: orchestrator_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext
    ) -> orchestrator_pb2.HealthCheckResponse:
        """Health check endpoint"""
        return orchestrator_pb2.HealthCheckResponse(
            status="healthy",
            details={
                "phase": "6",
                "service": "llm-orchestrator",
                "status": "multi_agent_active",
                "agents": "custom_agent_factory,weather,image_generation,rss,org,substack,podcast_script",
                "features": "agent_factory_playbooks,web_search,document_search,caching,conversation,formatting,weather_forecasts,image_generation,rss_management,org_management,article_generation,podcast_script_generation,org_project_capture,cross_document_synthesis"
            }
        )

    async def GetActions(
        self,
        request: orchestrator_pb2.GetActionsRequest,
        context: grpc.aio.ServicerContext
    ) -> orchestrator_pb2.GetActionsResponse:
        """Return all registered action I/O contracts for Agent Factory Workflow Composer."""
        try:
            import orchestrator.tools  # Ensures all register_action() calls have run
            from orchestrator.utils.action_io_registry import get_all_actions
            actions = get_all_actions()
            payload = []
            typed_actions = []
            for name, contract in actions.items():
                payload.append({
                    "name": name,
                    "category": contract.category,
                    "description": contract.description,
                    "short_description": getattr(contract, "short_description", None),
                    "input_schema": contract.get_input_schema(),
                    "params_schema": contract.get_params_schema(),
                    "output_schema": contract.get_output_schema(),
                    "input_fields": contract.get_input_fields(),
                    "output_fields": contract.get_output_fields(),
                })
                inputs = [
                    orchestrator_pb2.ActionField(
                        name=f.get("name", ""),
                        type=f.get("type", "text"),
                        description=f.get("description", ""),
                        required=f.get("required", False),
                        default_value=str(f["default"]) if f.get("default") is not None else "",
                    )
                    for f in contract.get_input_fields()
                ]
                outputs = [
                    orchestrator_pb2.ActionField(
                        name=f.get("name", ""),
                        type=f.get("type", "text"),
                        description=f.get("description", ""),
                        required=False,
                        default_value="",
                    )
                    for f in contract.get_output_fields()
                ]
                params = []
                if contract.params_model:
                    params_schema = contract.get_params_schema()
                    for prop, meta in (params_schema.get("properties") or {}).items():
                        params.append(orchestrator_pb2.ActionField(
                            name=prop,
                            type=meta.get("type", "text"),
                            description=meta.get("description", ""),
                            required=prop in (params_schema.get("required") or []),
                            default_value="",
                        ))
                typed_actions.append(orchestrator_pb2.ActionContract(
                    name=name,
                    category=contract.category,
                    description=contract.description,
                    inputs=inputs,
                    outputs=outputs,
                    params=params,
                ))
            return orchestrator_pb2.GetActionsResponse(actions_json=json.dumps(payload), actions=typed_actions)
        except Exception as e:
            logger.exception("GetActions failed")
            return orchestrator_pb2.GetActionsResponse(actions_json=json.dumps({"error": str(e), "actions": []}))

    async def GetPlugins(
        self,
        request: orchestrator_pb2.GetPluginsRequest,
        context: grpc.aio.ServicerContext
    ) -> orchestrator_pb2.GetPluginsResponse:
        """Return available Agent Factory plugins and their connection requirements."""
        try:
            from orchestrator.plugins.plugin_loader import get_plugin_loader
            loader = get_plugin_loader()
            names = loader.discover_plugins()
            plugins = []
            for name in names:
                plugin = loader.get_plugin(name)
                if not plugin:
                    continue
                req = plugin.get_connection_requirements()
                connection_requirements = [
                    orchestrator_pb2.PluginConnectionField(key=k, label=v)
                    for k, v in req.items()
                ]
                plugins.append(orchestrator_pb2.PluginInfo(
                    name=plugin.plugin_name,
                    version=plugin.plugin_version,
                    connection_requirements=connection_requirements,
                ))
            return orchestrator_pb2.GetPluginsResponse(plugins=plugins)
        except Exception as e:
            logger.exception("GetPlugins failed")
            return orchestrator_pb2.GetPluginsResponse(plugins=[])