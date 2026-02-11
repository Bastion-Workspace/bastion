"""
LLM Orchestrator gRPC Service Implementation
Handles incoming gRPC requests for LLM orchestration
"""

import asyncio
import logging
from datetime import datetime
from typing import AsyncIterator, Optional, Dict, Any, List

import grpc
from protos import orchestrator_pb2, orchestrator_pb2_grpc
from langchain_core.messages import HumanMessage, AIMessage

try:
    from config.settings import settings
except ImportError:
    settings = None

logger = logging.getLogger(__name__)


# In-memory cache for conversation-level metadata (primary_agent_selected)
# This bridges the gap between different agents' checkpoints
# Key: conversation_id, Value: {"primary_agent_selected": str, "last_agent": str, "timestamp": float}
_conversation_metadata_cache = {}


class OrchestratorGRPCService(orchestrator_pb2_grpc.OrchestratorServiceServicer):
    """
    gRPC service implementation for LLM Orchestrator
    
    Phase 5: Full sophisticated research agent with multi-round workflow!
    """
    
    def __init__(self):
        self.is_initialized = False
        logger.info("Initializing OrchestratorGRPCService...")

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
    
    def _is_first_user_message(self, conversation_history) -> bool:
        """
        Check if this is the first user message in the conversation
        
        Args:
            conversation_history: List of conversation messages from request
            
        Returns:
            True if this is the first user message, False otherwise
        """
        # Count user messages in history (current message is not in history yet)
        user_message_count = sum(1 for msg in conversation_history if msg.role == "user")
        return user_message_count == 0
    
    def _extract_response_text(self, result: Any) -> str:
        """
        Extract response text from agent result
        
        Handles different response formats from different agents.
        Works with both dict and string results.
        
        Args:
            result: Agent result (dict, string, or other)
            
        Returns:
            Response text string
        """
        # If result is not a dict, convert to string
        if not isinstance(result, dict):
            return str(result)
        
        # Try response field first (for agents like dictionary that structure response in response.message)
        response = result.get("response", "")
        if isinstance(response, dict):
            # Check for message field first (dictionary agent format)
            if "message" in response:
                return response.get("message", "")
            # Fallback to response field (other agents)
            if "response" in response:
                return response.get("response", "")
        if isinstance(response, str):
            return response
        
        # Try messages (for agents that use messages as primary response)
        agent_messages = result.get("messages", [])
        if agent_messages:
            last_message = agent_messages[-1]
            if hasattr(last_message, 'content'):
                return last_message.content
            return str(last_message)
        
        # Fallback
        return "Response generated"
    
    def _is_standard_format(self, result: Any) -> bool:
        """
        Check if result is in standard AgentResponse format
        
        Args:
            result: Agent result to check
            
        Returns:
            True if result matches standard format, False otherwise
        """
        if not isinstance(result, dict):
            return False
        
        # Check for required fields
        required_fields = ["response", "task_status", "agent_type", "timestamp"]
        return all(field in result for field in required_fields)
    
    def _extract_editor_operations_unified(self, result: Any) -> Optional[List[Dict[str, Any]]]:
        """
        Extract editor_operations from agent result (unified fallback chain)
        
        Checks all known locations where agents might place editor_operations.
        Supports both legacy and standard formats.
        
        Args:
            result: Agent result dictionary
            
        Returns:
            List of editor operations or None if not found
        """
        if not isinstance(result, dict):
            logger.warning(f"🔍 EXTRACT EDITOR OPS: result is not a dict (type: {type(result)})")
            return None
        
        # DEBUG: Log what we're checking
        logger.info(f"🔍 EXTRACT EDITOR OPS: result keys: {list(result.keys())}")
        logger.info(f"🔍 EXTRACT EDITOR OPS: result.get('editor_operations') = {result.get('editor_operations')}")
        logger.info(f"🔍 EXTRACT EDITOR OPS: type = {type(result.get('editor_operations'))}")
        if result.get("editor_operations"):
            logger.info(f"🔍 EXTRACT EDITOR OPS: length = {len(result.get('editor_operations'))}")
        
        # Check all known locations in priority order
        # CRITICAL: Use 'is not None' instead of 'or' to handle empty lists correctly
        ops = result.get("editor_operations")
        if ops is not None:
            logger.info(f"✅ EXTRACT EDITOR OPS: Found at result level ({len(ops)} ops)")
            return ops
        
        ops = result.get("agent_results", {}).get("editor_operations")
        if ops is not None:
            logger.info(f"✅ EXTRACT EDITOR OPS: Found at agent_results level ({len(ops)} ops)")
            return ops
        
        if isinstance(result.get("response"), dict):
            ops = result.get("response", {}).get("editor_operations")
            if ops is not None:
                logger.info(f"✅ EXTRACT EDITOR OPS: Found at response level ({len(ops)} ops)")
                return ops
        
        logger.warning(f"⚠️ EXTRACT EDITOR OPS: Not found in any location")
        return None
    
    def _extract_manuscript_edit_unified(self, result: Any) -> Optional[Dict[str, Any]]:
        """
        Extract manuscript_edit from agent result (unified fallback chain)
        
        Checks all known locations where agents might place manuscript_edit.
        Supports both legacy and standard formats.
        
        Args:
            result: Agent result dictionary
            
        Returns:
            Manuscript edit metadata dict or None if not found
        """
        if not isinstance(result, dict):
            logger.warning(f"🔍 EXTRACT MANUSCRIPT EDIT: result is not a dict (type: {type(result)})")
            return None
        
        # DEBUG: Log what we're checking
        logger.info(f"🔍 EXTRACT MANUSCRIPT EDIT: result keys: {list(result.keys())}")
        logger.info(f"🔍 EXTRACT MANUSCRIPT EDIT: result.get('manuscript_edit') = {result.get('manuscript_edit')}")
        
        # Check all known locations in priority order
        # CRITICAL: Use 'is not None' instead of 'or' to handle empty dicts correctly
        edit = result.get("manuscript_edit")
        if edit is not None:
            logger.info(f"✅ EXTRACT MANUSCRIPT EDIT: Found at result level")
            return edit
        
        edit = result.get("agent_results", {}).get("manuscript_edit")
        if edit is not None:
            logger.info(f"✅ EXTRACT MANUSCRIPT EDIT: Found at agent_results level")
            return edit
        
        if isinstance(result.get("response"), dict):
            edit = result.get("response", {}).get("manuscript_edit")
            if edit is not None:
                logger.info(f"✅ EXTRACT MANUSCRIPT EDIT: Found at response level")
                return edit
        
        logger.warning(f"⚠️ EXTRACT MANUSCRIPT EDIT: Not found in any location")
        return None
    
    def _extract_response_unified(self, result: Any, agent_type: str = "unknown") -> Dict[str, Any]:
        """
        Unified response extraction supporting both old and new formats
        
        This method normalizes agent responses into a standard structure.
        It first checks if the result is already in standard format, then
        falls back to legacy extraction patterns.
        
        Args:
            result: Agent result (dict, string, or other)
            agent_type: Agent identifier (for fallback when not in result)
            
        Returns:
            Normalized response dictionary with standard fields
        """
        from orchestrator.models.agent_response_contract import AgentResponse
        
        # If result is already in standard format, use it directly
        if self._is_standard_format(result):
            return result
        
        # Try to create AgentResponse from legacy format
        if isinstance(result, dict):
            try:
                # Use the from_legacy_format helper
                standard_response = AgentResponse.from_legacy_format(result, agent_type)
                return standard_response.dict(exclude_none=True)
            except Exception as e:
                logger.warning(f"Failed to normalize response to standard format: {e}")
        
        # Fallback: Manual extraction for edge cases
        response_text = self._extract_response_text(result)
        
        # Extract task_status
        task_status = result.get("task_status", "complete") if isinstance(result, dict) else "complete"
        
        # Extract agent_type
        extracted_agent_type = (
            result.get("agent_type") or
            result.get("agent_results", {}).get("agent_type") if isinstance(result, dict) else
            agent_type
        )
        
        # Extract timestamp
        from datetime import datetime
        timestamp = (
            result.get("timestamp") or
            datetime.now().isoformat()
        )
        
        # Extract optional fields
        editor_operations = self._extract_editor_operations_unified(result) if isinstance(result, dict) else None
        manuscript_edit = self._extract_manuscript_edit_unified(result) if isinstance(result, dict) else None
        
        return {
            "response": response_text,
            "task_status": task_status,
            "agent_type": extracted_agent_type,
            "timestamp": timestamp,
            "editor_operations": editor_operations,
            "manuscript_edit": manuscript_edit,
            # Preserve any other fields from original result
            **(result if isinstance(result, dict) else {})
        }
    
    async def _process_agent_with_cancellation(
        self,
        agent,
        query: str,
        metadata: Dict[str, Any],
        messages: List[Any],
        cancellation_token: asyncio.Event
    ) -> Dict[str, Any]:
        """
        Process agent request with cancellation support
        
        Wraps agent.process() with cancellation token handling.
        If agent supports process_with_cancellation(), uses that; otherwise falls back to standard process().
        
        Args:
            agent: Agent instance to process request
            query: User query
            metadata: Metadata dictionary
            messages: Conversation messages
            cancellation_token: Cancellation event token
            
        Returns:
            Agent result dictionary
        """
        # Check if agent supports cancellation-aware processing
        if hasattr(agent, 'process_with_cancellation'):
            result = await agent.process_with_cancellation(
                query=query,
                metadata=metadata,
                messages=messages,
                cancellation_token=cancellation_token
            )
        else:
            # Fallback to standard process with cancellation monitoring
            process_task = asyncio.create_task(
                agent.process(query=query, metadata=metadata, messages=messages)
            )
            
            # Wait for either completion or cancellation
            done, pending = await asyncio.wait(
                [process_task, asyncio.create_task(cancellation_token.wait())],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
            
            # Check if cancellation was requested
            if cancellation_token.is_set():
                process_task.cancel()
                try:
                    await process_task
                except asyncio.CancelledError:
                    pass
                raise asyncio.CancelledError("Operation cancelled")
            
            # Return result
            result = await process_task
        
        # Save agent identity to cache for conversation continuity
        # Cache serves as optimization layer; backend will also save when storing response
        conversation_id = metadata.get("conversation_id")
        if conversation_id:
            self._save_agent_identity_to_cache(result, conversation_id)
        
        return result
    
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
            from orchestrator.engines.unified_dispatch import get_unified_dispatcher
            chat_agent = get_unified_dispatcher()._get_conversational_engine()._get_agent()
            if not chat_agent:
                return {}
            config = chat_agent._get_checkpoint_config(metadata)
            workflow = await chat_agent._get_workflow()
            shared_memory = await chat_agent._load_checkpoint_shared_memory(workflow, config)
            return shared_memory
            
        except Exception as e:
            logger.debug(f"⚠️ Failed to load checkpoint shared_memory in gRPC service: {e}")
            return {}
    
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

            # Extract primary_agent_selected from agent's result
            result_shared_memory = agent_result.get("shared_memory", {})
            primary_agent = result_shared_memory.get("primary_agent_selected")
            last_agent = result_shared_memory.get("last_agent")
            
            if not primary_agent:
                # Agent didn't set primary_agent_selected, skip
                return
            
            # Store in cache
            import time
            _conversation_metadata_cache[conversation_id] = {
                "primary_agent_selected": primary_agent,
                "last_agent": last_agent or primary_agent,
                "timestamp": time.time()
            }
            
            logger.info(f"✅ CACHED AGENT IDENTITY: primary_agent_selected = '{primary_agent}', last_agent = '{last_agent}' (conversation: {conversation_id})")
            
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
        
        Supports multiple agent types: research, chat, help, weather, image_generation, rss, org, etc.
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
            discovered_skill = None  # Set by short-circuit routes or skill discovery
            from orchestrator.skills import get_skill_registry, load_all_skills
            load_all_skills()
            registry = get_skill_registry()

            # SHORT-CIRCUIT ROUTING: Check for "/help" prefix for instant help routing
            query_lower = request.query.lower().strip()
            if query_lower.startswith("/help"):
                agent_type = "help"
                discovered_skill = registry.get("help")
                primary_agent_name = "help_agent"
                # Strip "/help" prefix from query (with optional space after)
                # Handle both "/help" and "/help " (with space)
                cleaned_query = request.query[5:].strip()  # Remove "/help" (5 chars)
                # If no query after "/help", use empty string (help agent will show general help)
                if not cleaned_query:
                    cleaned_query = ""
                # Update request.query for agent processing
                request.query = cleaned_query
                logger.debug(f"❓ SHORT-CIRCUIT ROUTING: Help agent (query starts with '/help', cleaned: '{cleaned_query[:50]}...' if cleaned_query else 'empty')")
                
                # Ensure shared_memory is in metadata and conversation_context
                if "shared_memory" not in metadata:
                    metadata["shared_memory"] = {}
                
                # Merge the extracted request shared_memory (including active_editor!)
                metadata["shared_memory"].update(request_shared_memory)
                metadata["shared_memory"]["primary_agent_selected"] = agent_type
                
                if "shared_memory" not in conversation_context:
                    conversation_context["shared_memory"] = {}
                conversation_context["shared_memory"].update(request_shared_memory)
                conversation_context["shared_memory"]["primary_agent_selected"] = agent_type
                
                if checkpoint_shared_memory:
                    checkpoint_shared_memory["primary_agent_selected"] = agent_type
                
                logger.info(f"📋 SET primary_agent_selected: {agent_type} (and merged {len(request_shared_memory)} shared_memory keys)")
            # SHORT-CIRCUIT ROUTING: Check for "/define" prefix for instant dictionary routing
            elif query_lower.startswith("/define"):
                agent_type = "dictionary"
                discovered_skill = registry.get("dictionary")
                primary_agent_name = "dictionary_agent"
                # Strip "/define" prefix from query (with optional space after)
                # Handle both "/define" and "/define " (with space)
                cleaned_query = request.query[7:].strip()  # Remove "/define" (7 chars)
                # If no query after "/define", use empty string (dictionary agent will handle it)
                if not cleaned_query:
                    cleaned_query = ""
                # Update request.query for agent processing
                request.query = cleaned_query
                logger.info(f"📖 SHORT-CIRCUIT ROUTING: Dictionary agent (query starts with '/define', cleaned: '{cleaned_query[:50]}...' if cleaned_query else 'empty')")
                
                # Ensure shared_memory is in metadata and conversation_context
                if "shared_memory" not in metadata:
                    metadata["shared_memory"] = {}
                
                # Merge the extracted request shared_memory (including active_editor!)
                metadata["shared_memory"].update(request_shared_memory)
                metadata["shared_memory"]["primary_agent_selected"] = agent_type
                
                if "shared_memory" not in conversation_context:
                    conversation_context["shared_memory"] = {}
                conversation_context["shared_memory"].update(request_shared_memory)
                conversation_context["shared_memory"]["primary_agent_selected"] = agent_type
                
                if checkpoint_shared_memory:
                    checkpoint_shared_memory["primary_agent_selected"] = agent_type
                
                logger.info(f"📋 SET primary_agent_selected: {agent_type} (and merged {len(request_shared_memory)} shared_memory keys)")
            elif request.agent_type and request.agent_type != "auto":
                # Explicit agent routing provided by backend
                agent_type = request.agent_type
                primary_agent_name = agent_type
                discovered_skill = registry.get(agent_type)  # registry.get() accepts "help_agent" -> help skill
                logger.info(f"EXPLICIT ROUTING: {agent_type} (reason: {request.routing_reason or 'not specified'})")
            
            # If agent_type is still None, run skill discovery (compound-aware)
            compound_plan = None
            if agent_type is None:
                from orchestrator.skills.skill_llm_selector import llm_select_skill, llm_select_skill_or_plan
                active_editor = (request_shared_memory or {}).get("active_editor") or conversation_context.get("shared_memory", {}).get("active_editor")
                editor_context = None
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
                # (e.g. outline open -> outline_editing only, not story_analysis + outline_editing).
                # Exception: for fiction, if the query explicitly asks for story analysis (e.g.
                # "Analyze the story"), do not pin to fiction_editing so skill discovery can
                # route to the story_analysis conversational skill.
                if instant_route is None and eligible and editor_context and shared.get("editor_preference") == "prefer":
                    from orchestrator.skills.skill_schema import EngineType
                    editor_type = (editor_context.get("type") or "").strip().lower()
                    if editor_type:
                        skip_editor_pin = False
                        route_to_story_analysis = False
                        if editor_type == "fiction":
                            query_lower = (request.query or "").strip().lower()
                            explicit_analysis_keywords = [
                                "analyze", "critique", "review", "assess", "evaluate", "examine", "study"
                            ]
                            if any(kw in query_lower for kw in explicit_analysis_keywords):
                                skip_editor_pin = True
                                route_to_story_analysis = any(
                                    s.name == "story_analysis" for s in eligible
                                )
                                if route_to_story_analysis:
                                    instant_route = "story_analysis"
                                    logger.info(
                                        "Skill discovery: routing to story_analysis (explicit analysis query with fiction open)"
                                    )
                                else:
                                    logger.info(
                                        "Skill discovery: skipping editor-type pin (fiction) for explicit analysis query"
                                    )
                        if not skip_editor_pin and not route_to_story_analysis:
                            dedicated = [
                                s for s in eligible
                                if s.engine == EngineType.EDITOR and (s.editor_types or []) == [editor_type]
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
                    discovered_skill = registry.get(agent_type)
                    primary_agent_name = f"{agent_type}_agent" if not agent_type.endswith("_agent") else agent_type
                    logger.info("Skill discovery: %s (instant)", agent_type)
                elif eligible:
                    plan = await llm_select_skill_or_plan(
                        eligible,
                        request.query,
                        editor_context=editor_context,
                        conversation_context=conversation_context,
                        metadata=metadata,
                    )
                    if plan is None:
                        selected_name = await llm_select_skill(
                            eligible,
                            request.query,
                            editor_context=editor_context,
                            conversation_context=conversation_context,
                            metadata=metadata,
                        )
                        agent_type = selected_name or "chat"
                        discovered_skill = registry.get(agent_type)
                        primary_agent_name = f"{agent_type}_agent" if not agent_type.endswith("_agent") else agent_type
                        logger.info("Skill discovery: %s (fallback)", agent_type)
                    elif plan.is_compound and len(plan.steps) > 1:
                        compound_plan = plan
                        agent_type = "compound"
                        primary_agent_name = "compound_agent"
                        discovered_skill = None
                        step_skills = [s.skill_name for s in plan.steps]
                        logger.info(
                            "Skill discovery: compound plan (%d steps): %s",
                            len(plan.steps),
                            ", ".join(step_skills),
                        )
                        if plan.reasoning:
                            logger.info("Compound plan reasoning: %s", plan.reasoning.strip())
                        for s in plan.steps:
                            logger.info(
                                "  step %s: skill=%s | sub_query=%s",
                                s.step_id,
                                s.skill_name,
                                (s.sub_query or "").strip() or "(none)",
                            )
                    else:
                        agent_type = (plan.skill or "chat")
                        discovered_skill = registry.get(agent_type)
                        primary_agent_name = f"{agent_type}_agent" if not agent_type.endswith("_agent") else agent_type
                        logger.info("Skill discovery: %s", agent_type)
                else:
                    agent_type = "chat"
                    discovered_skill = registry.get("chat")
                    primary_agent_name = "chat_agent"
                    logger.info("Skill discovery: no eligible skills, fallback to chat")
                if "shared_memory" not in metadata:
                    metadata["shared_memory"] = {}
                metadata["shared_memory"]["primary_agent_selected"] = primary_agent_name or agent_type
                conversation_context["shared_memory"]["primary_agent_selected"] = primary_agent_name or agent_type

            if not discovered_skill:
                discovered_skill = registry.get("chat")
                primary_agent_name = "chat_agent"

            # Parse conversation history for agent
            messages = []
            for msg in request.conversation_history:
                if msg.role == "user":
                    messages.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    messages.append(AIMessage(content=msg.content))

            # Skill dispatch: compound plan or unified dispatcher
            if compound_plan:
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
                dispatch_metadata = {
                    "user_id": request.user_id,
                    "conversation_id": request.conversation_id,
                    "shared_memory": shared_memory,
                    **{k: v for k, v in metadata.items() if k != "shared_memory"},
                }
                from orchestrator.engines.plan_engine import PlanEngine
                plan_engine = PlanEngine()
                async for chunk in plan_engine.execute_plan(
                    compound_plan, request.query, dispatch_metadata, messages, cancellation_token
                ):
                    yield chunk
                self._save_agent_identity_to_cache(
                    {"shared_memory": {"primary_agent_selected": "compound_agent", "last_agent": "compound_agent"}},
                    request.conversation_id,
                )
                return
            if discovered_skill:
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
                dispatch_metadata = {
                    "user_id": request.user_id,
                    "conversation_id": request.conversation_id,
                    "shared_memory": shared_memory,
                    **{k: v for k, v in metadata.items() if k != "shared_memory"},
                }
                from orchestrator.engines.unified_dispatch import get_unified_dispatcher
                dispatcher = get_unified_dispatcher()
                async for chunk in dispatcher.dispatch(
                    discovered_skill.name, request.query, dispatch_metadata, messages, cancellation_token
                ):
                    yield chunk
                agent_name = primary_agent_name or discovered_skill.name
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
    
    async def StartTask(
        self,
        request: orchestrator_pb2.TaskRequest,
        context: grpc.aio.ServicerContext
    ) -> orchestrator_pb2.TaskResponse:
        """
        Start async task processing
        
        Phase 1: Stub implementation
        """
        logger.info(f"StartTask request from user {request.user_id}")
        
        return orchestrator_pb2.TaskResponse(
            task_id=f"task_{datetime.now().timestamp()}",
            status="queued",
            message="Phase 1: Task queued (full implementation in Phase 2)"
        )
    
    async def GetTaskStatus(
        self,
        request: orchestrator_pb2.TaskStatusRequest,
        context: grpc.aio.ServicerContext
    ) -> orchestrator_pb2.TaskStatusResponse:
        """Get status of async task"""
        logger.info(f"GetTaskStatus request for task {request.task_id}")
        
        return orchestrator_pb2.TaskStatusResponse(
            task_id=request.task_id,
            status="completed",
            result="Phase 1: Stub response",
            error_message=""
        )
    
    async def ApprovePermission(
        self,
        request: orchestrator_pb2.PermissionApproval,
        context: grpc.aio.ServicerContext
    ) -> orchestrator_pb2.ApprovalResponse:
        """Handle permission approval (HITL)"""
        logger.info(f"ApprovePermission from user {request.user_id}: {request.approval_decision}")
        
        return orchestrator_pb2.ApprovalResponse(
            success=True,
            message="Phase 1: Permission recorded",
            next_action="continue"
        )
    
    async def GetPendingPermissions(
        self,
        request: orchestrator_pb2.PermissionRequest,
        context: grpc.aio.ServicerContext
    ) -> orchestrator_pb2.PermissionList:
        """Get list of pending permissions"""
        logger.info(f"GetPendingPermissions for user {request.user_id}")
        
        return orchestrator_pb2.PermissionList(
            permissions=[]  # Phase 1: No pending permissions
        )
    
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
                "agents": "research,chat,help,weather,image_generation,rss,org,substack,podcast_script",
                "features": "multi_round_research,query_expansion,gap_analysis,web_search,caching,conversation,formatting,weather_forecasts,image_generation,rss_management,org_management,article_generation,podcast_script_generation,org_project_capture,cross_document_synthesis"
            }
        )

    async def StartTask(
        self,
        request: orchestrator_pb2.TaskRequest,
        context: grpc.aio.ServicerContext
    ) -> orchestrator_pb2.TaskResponse:
        """
        Start async task processing
        
        Phase 1: Stub implementation
        """
        logger.info(f"StartTask request from user {request.user_id}")
        
        return orchestrator_pb2.TaskResponse(
            task_id=f"task_{datetime.now().timestamp()}",
            status="queued",
            message="Phase 1: Task queued (full implementation in Phase 2)"
        )
    
    async def GetTaskStatus(
        self,
        request: orchestrator_pb2.TaskStatusRequest,
        context: grpc.aio.ServicerContext
    ) -> orchestrator_pb2.TaskStatusResponse:
        """Get status of async task"""
        logger.info(f"GetTaskStatus request for task {request.task_id}")
        
        return orchestrator_pb2.TaskStatusResponse(
            task_id=request.task_id,
            status="completed",
            result="Phase 1: Stub response",
            error_message=""
        )
    
    async def ApprovePermission(
        self,
        request: orchestrator_pb2.PermissionApproval,
        context: grpc.aio.ServicerContext
    ) -> orchestrator_pb2.ApprovalResponse:
        """Handle permission approval (HITL)"""
        logger.info(f"ApprovePermission from user {request.user_id}: {request.approval_decision}")
        
        return orchestrator_pb2.ApprovalResponse(
            success=True,
            message="Phase 1: Permission recorded",
            next_action="continue"
        )
    
    async def GetPendingPermissions(
        self,
        request: orchestrator_pb2.PermissionRequest,
        context: grpc.aio.ServicerContext
    ) -> orchestrator_pb2.PermissionList:
        """Get list of pending permissions"""
        logger.info(f"GetPendingPermissions for user {request.user_id}")
        
        return orchestrator_pb2.PermissionList(
            permissions=[]  # Phase 1: No pending permissions
        )
    
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
                "agents": "research,chat,help,weather,image_generation,rss,org,substack,podcast_script",
                "features": "multi_round_research,query_expansion,gap_analysis,web_search,caching,conversation,formatting,weather_forecasts,image_generation,rss_management,org_management,article_generation,podcast_script_generation,org_project_capture,cross_document_synthesis"
            }
        )