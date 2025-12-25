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
from orchestrator.agents import (
    get_full_research_agent,
    ChatAgent,
    DictionaryAgent,
    DataFormattingAgent,
    get_weather_agent,
    get_image_generation_agent,
    # FactCheckingAgent removed - not actively used
    get_rss_agent,
    get_org_inbox_agent,
    get_substack_agent,
    get_podcast_script_agent,
    get_org_project_agent,
    get_entertainment_agent,
    get_electronics_agent,
    get_character_development_agent,
    get_content_analysis_agent,
    get_fiction_editing_agent,
    get_outline_editing_agent,
    get_story_analysis_agent,
    get_site_crawl_agent,
    get_rules_editing_agent,
    get_style_editing_agent,
    get_proofreading_agent,
    get_general_project_agent,
    get_reference_agent,
    get_knowledge_builder_agent
)
from orchestrator.services import get_intent_classifier
from langchain_core.messages import HumanMessage, AIMessage

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
        self.research_agent = None
        self.chat_agent = None
        self.dictionary_agent = None
        self.data_formatting_agent = None
        self.help_agent = None
        self.weather_agent = None
        self.image_generation_agent = None
        # FactCheckingAgent removed - not actively used
        self.rss_agent = None
        self.org_inbox_agent = None
        self.substack_agent = None
        self.podcast_script_agent = None
        self.org_project_agent = None
        self.entertainment_agent = None
        self.electronics_agent = None
        self.character_development_agent = None
        self.content_analysis_agent = None
        self.fiction_editing_agent = None
        self.outline_editing_agent = None
        self.story_analysis_agent = None
        self.site_crawl_agent = None
        self.rules_editing_agent = None
        self.style_editing_agent = None
        self.proofreading_agent = None
        self.general_project_agent = None
        self.reference_agent = None
        self.knowledge_builder_agent = None
        logger.info("Initializing OrchestratorGRPCService...")
    
    def _ensure_agents_loaded(self):
        """Lazy load agents"""
        agents_loaded = 0
        total_agents = 27  # Total number of agents to load
        
        if self.research_agent is None:
            self.research_agent = get_full_research_agent()
            agents_loaded += 1
        
        if self.chat_agent is None:
            self.chat_agent = ChatAgent()
            agents_loaded += 1
        
        if self.dictionary_agent is None:
            self.dictionary_agent = DictionaryAgent()
            agents_loaded += 1
        
        if self.data_formatting_agent is None:
            self.data_formatting_agent = DataFormattingAgent()
            agents_loaded += 1
        
        if self.help_agent is None:
            from orchestrator.agents import HelpAgent
            self.help_agent = HelpAgent()
            agents_loaded += 1
        
        if self.weather_agent is None:
            self.weather_agent = get_weather_agent()
            agents_loaded += 1
        
        if self.image_generation_agent is None:
            self.image_generation_agent = get_image_generation_agent()
            agents_loaded += 1
        
        # FactCheckingAgent removed - not actively used
        
        if self.rss_agent is None:
            self.rss_agent = get_rss_agent()
            agents_loaded += 1
        
        if self.org_inbox_agent is None:
            self.org_inbox_agent = get_org_inbox_agent()
            agents_loaded += 1
        
        if self.substack_agent is None:
            self.substack_agent = get_substack_agent()
            agents_loaded += 1
        
        if self.podcast_script_agent is None:
            self.podcast_script_agent = get_podcast_script_agent()
            agents_loaded += 1
        
        if self.org_project_agent is None:
            self.org_project_agent = get_org_project_agent()
            agents_loaded += 1
        
        if self.entertainment_agent is None:
            self.entertainment_agent = get_entertainment_agent()
            agents_loaded += 1

        if self.electronics_agent is None:
            self.electronics_agent = get_electronics_agent()
            agents_loaded += 1
        
        if self.character_development_agent is None:
            self.character_development_agent = get_character_development_agent()
            agents_loaded += 1
        
        if self.content_analysis_agent is None:
            self.content_analysis_agent = get_content_analysis_agent()
            agents_loaded += 1
        
        if self.fiction_editing_agent is None:
            self.fiction_editing_agent = get_fiction_editing_agent()
            agents_loaded += 1
        
        if self.outline_editing_agent is None:
            self.outline_editing_agent = get_outline_editing_agent()
            agents_loaded += 1
        
        if self.story_analysis_agent is None:
            self.story_analysis_agent = get_story_analysis_agent()
            agents_loaded += 1
        
        if self.site_crawl_agent is None:
            self.site_crawl_agent = get_site_crawl_agent()
            agents_loaded += 1
        
        if self.rules_editing_agent is None:
            self.rules_editing_agent = get_rules_editing_agent()
            agents_loaded += 1
        
        if self.style_editing_agent is None:
            self.style_editing_agent = get_style_editing_agent()
            agents_loaded += 1
        
        if self.proofreading_agent is None:
            self.proofreading_agent = get_proofreading_agent()
            agents_loaded += 1
        
        if self.general_project_agent is None:
            self.general_project_agent = get_general_project_agent()
            agents_loaded += 1
        
        if self.reference_agent is None:
            self.reference_agent = get_reference_agent()
            agents_loaded += 1
        
        if self.knowledge_builder_agent is None:
            self.knowledge_builder_agent = get_knowledge_builder_agent()
            agents_loaded += 1
        
        if agents_loaded > 0:
            logger.info(f"✅ Loaded {agents_loaded}/{total_agents} agents")
    
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
        logger.info(f"🔍 SHARED MEMORY EXTRACTION: HasField('active_editor')={request.HasField('active_editor')}")
        if request.HasField("active_editor"):
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
                    # Try Python list format: "['./file1.md', './file2.md']"
                    if value.strip().startswith('[') and value.strip().endswith(']'):
                        try:
                            import ast
                            parsed = ast.literal_eval(value)
                            if isinstance(parsed, list):
                                frontmatter_custom[key] = parsed
                                logger.info(f"✅ PARSED {key} as Python list: {len(parsed)} items")
                                continue
                        except (ValueError, SyntaxError) as e:
                            logger.debug(f"⚠️ Failed to parse {key} as Python list: {e}")
                    # Try JSON format: '["file1.md", "file2.md"]'
                    try:
                        import json
                        parsed = json.loads(value)
                        if isinstance(parsed, list):
                            frontmatter_custom[key] = parsed
                            logger.info(f"✅ PARSED {key} as JSON list: {len(parsed)} items")
                            continue
                    except (ValueError, json.JSONDecodeError) as e:
                        logger.debug(f"⚠️ Failed to parse {key} as JSON: {e}")
                    # Try YAML list format (newline-separated): "- ./file1.md\n- ./file2.md"
                    if '\n' in value and value.strip().startswith('-'):
                        try:
                            import yaml
                            parsed = yaml.safe_load(value)
                            if isinstance(parsed, list):
                                frontmatter_custom[key] = parsed
                                logger.info(f"✅ PARSED {key} as YAML list: {len(parsed)} items")
                                continue
                        except (yaml.YAMLError, ValueError) as e:
                            logger.debug(f"⚠️ Failed to parse {key} as YAML: {e}")
                
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
        
        # Try messages first (most common format)
        agent_messages = result.get("messages", [])
        if agent_messages:
            last_message = agent_messages[-1]
            if hasattr(last_message, 'content'):
                return last_message.content
            return str(last_message)
        
        # Try response field
        response = result.get("response", "")
        if isinstance(response, dict):
            return response.get("response", "") or response.get("message", "")
        if isinstance(response, str):
            return response
        
        # Fallback
        return "Response generated"
    
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
            # Ensure chat_agent is loaded (it has the workflow we need)
            self._ensure_agents_loaded()
            
            if not self.chat_agent:
                return {}
            
            # Get checkpoint config
            config = self.chat_agent._get_checkpoint_config(metadata)
            
            # Get workflow (will initialize if needed)
            workflow = await self.chat_agent._get_workflow()
            
            # Load shared_memory from checkpoint
            shared_memory = await self.chat_agent._load_checkpoint_shared_memory(workflow, config)
            
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
        
        Supports multiple agent types: research, chat, data_formatting
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
            logger.info(f"📨 StreamChat request from user {request.user_id}: {query_preview}")
            
            # Load agents
            self._ensure_agents_loaded()
            
            # Parse metadata into dictionary
            metadata = dict(request.metadata) if request.metadata else {}
            logger.info(f"🔍 RECEIVED METADATA: user_chat_model = {metadata.get('user_chat_model')}")

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
                logger.info(f"✅ PERSONA: Extracted persona for agents (ai_name={persona_dict['ai_name']}, style={persona_dict['persona_style']})")
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
                logger.info(f"📚 Merged cache into checkpoint: primary_agent={cached_agent_identity.get('primary_agent_selected')}")
            
            # Build conversation context from proto fields for intent classification
            conversation_context = self._extract_conversation_context(request)
            
            # Extract shared_memory from request (includes editor_preference)
            request_shared_memory = self._extract_shared_memory(request)
            
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
                    logger.info(f"📚 Loaded agent continuity from checkpoint: primary_agent={primary_agent}, last_agent={last_agent}")
                else:
                    logger.info(f"📚 Merged checkpoint shared_memory (no agent continuity): {list(checkpoint_shared_memory.keys())}")
            
            # Log editor_preference before merge
            request_editor_pref = request_shared_memory.get("editor_preference", "not_set")
            logger.info(f"📝 EDITOR PREFERENCE: From request = '{request_editor_pref}'")
            
            # Merge request shared_memory (current editor_preference takes precedence)
            conversation_context["shared_memory"].update(request_shared_memory)
            
            # Log final editor_preference after merge
            final_editor_pref = conversation_context["shared_memory"].get("editor_preference", "not_set")
            logger.info(f"📝 EDITOR PREFERENCE: Final (after merge) = '{final_editor_pref}'")
            
            # Determine which agent to use via intent classification
            primary_agent_name = None
            
            # SHORT-CIRCUIT ROUTING: Check for "/help" prefix for instant help routing
            query_lower = request.query.lower().strip()
            if query_lower.startswith("/help"):
                agent_type = "help_agent"
                primary_agent_name = agent_type
                # Strip "/help" prefix from query (with optional space after)
                # Handle both "/help" and "/help " (with space)
                cleaned_query = request.query[5:].strip()  # Remove "/help" (5 chars)
                # If no query after "/help", use empty string (help agent will show general help)
                if not cleaned_query:
                    cleaned_query = ""
                # Update request.query for agent processing
                request.query = cleaned_query
                logger.info(f"❓ SHORT-CIRCUIT ROUTING: Help agent (query starts with '/help', cleaned: '{cleaned_query[:50]}...' if cleaned_query else 'empty')")
                # Set primary_agent_selected in shared_memory for conversation continuity
                if "shared_memory" not in metadata:
                    metadata["shared_memory"] = {}
                metadata["shared_memory"]["primary_agent_selected"] = agent_type
                if "shared_memory" not in conversation_context:
                    conversation_context["shared_memory"] = {}
                conversation_context["shared_memory"]["primary_agent_selected"] = agent_type
                if checkpoint_shared_memory:
                    checkpoint_shared_memory["primary_agent_selected"] = agent_type
                logger.info(f"📋 SET primary_agent_selected: {agent_type} (for conversation continuity)")
            # SHORT-CIRCUIT ROUTING: Check for "define:" prefix for instant dictionary routing
            elif query_lower.startswith("define:"):
                agent_type = "dictionary_agent"
                primary_agent_name = agent_type
                logger.info(f"📖 SHORT-CIRCUIT ROUTING: Dictionary agent (query starts with 'define:')")
            elif request.agent_type and request.agent_type != "auto":
                # Explicit agent routing provided by backend
                agent_type = request.agent_type
                primary_agent_name = agent_type  # Store for next intent classification
                logger.info(f"🎯 EXPLICIT ROUTING: {agent_type} (reason: {request.routing_reason or 'not specified'})")
            else:
                # Run intent classification to determine agent
                logger.info(f"🎯 INTENT CLASSIFICATION: Running for query: {query_preview}")
                intent_classifier = get_intent_classifier()
                intent_result = await intent_classifier.classify_intent(
                    user_message=request.query,
                    conversation_context=conversation_context
                )
                agent_type = intent_result.target_agent
                primary_agent_name = agent_type  # Store for next intent classification
                logger.info(f"✅ INTENT CLASSIFICATION: → {agent_type} (action: {intent_result.action_intent}, confidence: {intent_result.confidence})")
                if intent_result.reasoning:
                    logger.info(f"💡 REASONING: {intent_result.reasoning}")
                
                # IMPORTANT: We set primary_agent_selected TENTATIVELY here based on routing
                # But we'll update it AFTER the agent completes with the actual agent's value
                # This ensures the agent's own identity takes precedence
                if "shared_memory" not in metadata:
                    metadata["shared_memory"] = {}
                # Set tentatively for this request (agent may override)
                metadata["shared_memory"]["primary_agent_selected"] = agent_type
                conversation_context["shared_memory"]["primary_agent_selected"] = agent_type
                logger.info(f"📋 TENTATIVE primary_agent_selected: {agent_type} (agent may override)")
                
                # Handle conversation title from intent classification (for new conversations)
                if intent_result.conversation_title:
                    logger.info(f"🔤 INTENT CLASSIFICATION generated title: {intent_result.conversation_title}")
                    
                    # Send title immediately via stream so frontend can update UI right away
                    yield orchestrator_pb2.ChatChunk(
                        type="title",
                        message=intent_result.conversation_title,
                        timestamp=datetime.now().isoformat(),
                        agent_name="intent_classifier"
                    )
                    
                    # Update conversation title asynchronously (non-blocking)
                    try:
                        from orchestrator.backend_tool_client import get_backend_tool_client
                        backend_client = await get_backend_tool_client()
                        # Run in background - don't await to avoid blocking response
                        asyncio.create_task(
                            backend_client.update_conversation_title(
                                conversation_id=request.conversation_id,
                                title=intent_result.conversation_title,
                                user_id=request.user_id
                            )
                        )
                        logger.info(f"🔤 Queued title update: {intent_result.conversation_title}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to queue title update: {e}")
            
            # Parse conversation history for agent
            messages = []
            for msg in request.conversation_history:
                if msg.role == "user":
                    messages.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    messages.append(AIMessage(content=msg.content))
            
            # Convert agent name from intent classifier format to routing format
            # Intent classifier returns names like "research_agent", routing uses "research"
            def normalize_agent_name(agent_name: str) -> str:
                """Convert agent name from intent classifier format to routing format"""
                # Handle special cases first
                special_cases = {
                    "combined_proofread_and_analyze": "chat",  # Not implemented, fallback to chat
                    "pipeline_agent": "data_formatting",  # Pipeline functionality handled by data_formatting
                    "website_crawler_agent": "site_crawl",  # Different naming convention
                    "dictionary_agent": "dictionary",  # Short-circuit dictionary agent
                }
                
                if agent_name in special_cases:
                    return special_cases[agent_name]
                
                # Standard conversion: remove "_agent" suffix
                if agent_name.endswith("_agent"):
                    return agent_name[:-6]  # Remove "_agent" (6 characters)
                
                # Already in short format, return as-is
                return agent_name
            
            # Normalize agent name for routing
            original_agent_type = agent_type
            agent_type = normalize_agent_name(agent_type)
            
            # Log if conversion occurred (for debugging, but no "not migrated" message)
            if original_agent_type != agent_type:
                logger.debug(f"🔄 Agent name normalized: {original_agent_type} → {agent_type}")
            
            # Route to appropriate agent
            if agent_type == "chat":
                yield orchestrator_pb2.ChatChunk(
                    type="status",
                    message="Chat agent processing your message...",
                    timestamp=datetime.now().isoformat(),
                    agent_name="chat_agent"
                )
                
                # Build metadata with shared_memory (STANDARD PATTERN)
                shared_memory = self._extract_shared_memory(request, metadata.get("shared_memory", {}))
                
                chat_metadata = {
                    "user_id": request.user_id,
                    "conversation_id": request.conversation_id,
                    "shared_memory": shared_memory,
                    **{k: v for k, v in metadata.items() if k != "shared_memory"}
                }
                
                # Check for cancellation before processing
                if cancellation_token.is_set():
                    raise asyncio.CancelledError("Operation cancelled")
                
                result = await self._process_agent_with_cancellation(
                    agent=self.chat_agent,
                    query=request.query,
                    metadata=chat_metadata,
                    messages=messages,
                    cancellation_token=cancellation_token
                )
                
                response_text = result.get("response", "No response generated")
                
                yield orchestrator_pb2.ChatChunk(
                    type="content",
                    message=response_text,
                    timestamp=datetime.now().isoformat(),
                    agent_name="chat_agent"
                )
                
                # Title generation is now handled by intent classifier (parallel, faster)
                
                yield orchestrator_pb2.ChatChunk(
                    type="complete",
                    message=f"Chat complete (status: {result.get('task_status', 'complete')})",
                    timestamp=datetime.now().isoformat(),
                    agent_name="system"
                )
                
            elif agent_type == "help":
                yield orchestrator_pb2.ChatChunk(
                    type="status",
                    message="Help agent loading documentation...",
                    timestamp=datetime.now().isoformat(),
                    agent_name="help_agent"
                )
                
                # Build metadata with shared_memory (STANDARD PATTERN)
                shared_memory = self._extract_shared_memory(request, metadata.get("shared_memory", {}))
                
                help_metadata = {
                    "user_id": request.user_id,
                    "conversation_id": request.conversation_id,
                    "shared_memory": shared_memory,
                    **{k: v for k, v in metadata.items() if k != "shared_memory"}
                }
                
                if cancellation_token.is_set():
                    raise asyncio.CancelledError("Operation cancelled")
                
                result = await self._process_agent_with_cancellation(
                    agent=self.help_agent,
                    query=request.query,
                    metadata=help_metadata,
                    messages=messages,
                    cancellation_token=cancellation_token
                )
                
                response_text = self._extract_response_text(result)
                
                yield orchestrator_pb2.ChatChunk(
                    type="content",
                    message=response_text,
                    timestamp=datetime.now().isoformat(),
                    agent_name="help_agent"
                )
                
                yield orchestrator_pb2.ChatChunk(
                    type="complete",
                    message=f"Help complete (status: {result.get('task_status', 'complete')})",
                    timestamp=datetime.now().isoformat(),
                    agent_name="system"
                )
            
            elif agent_type == "dictionary":
                yield orchestrator_pb2.ChatChunk(
                    type="status",
                    message="Dictionary agent looking up word definition...",
                    timestamp=datetime.now().isoformat(),
                    agent_name="dictionary_agent"
                )
                
                # Build metadata with shared_memory (STANDARD PATTERN)
                shared_memory = self._extract_shared_memory(request, metadata.get("shared_memory", {}))
                
                dictionary_metadata = {
                    "user_id": request.user_id,
                    "conversation_id": request.conversation_id,
                    "shared_memory": shared_memory,
                    **{k: v for k, v in metadata.items() if k != "shared_memory"}
                }
                
                if cancellation_token.is_set():
                    raise asyncio.CancelledError("Operation cancelled")
                
                result = await self._process_agent_with_cancellation(
                    agent=self.dictionary_agent,
                    query=request.query,
                    metadata=dictionary_metadata,
                    messages=messages,
                    cancellation_token=cancellation_token
                )
                
                response_text = self._extract_response_text(result)
                
                yield orchestrator_pb2.ChatChunk(
                    type="content",
                    message=response_text,
                    timestamp=datetime.now().isoformat(),
                    agent_name="dictionary_agent"
                )
                
                # Title generation is now handled by intent classifier (parallel, faster)
                
                yield orchestrator_pb2.ChatChunk(
                    type="complete",
                    message=f"Dictionary lookup complete (word found: {result.get('found', False)})",
                    timestamp=datetime.now().isoformat(),
                    agent_name="system"
                )
            
            elif agent_type == "data_formatting":
                yield orchestrator_pb2.ChatChunk(
                    type="status",
                    message="Data formatting agent organizing your data...",
                    timestamp=datetime.now().isoformat(),
                    agent_name="data_formatting_agent"
                )
                
                # Build metadata with shared_memory (STANDARD PATTERN)
                shared_memory = self._extract_shared_memory(request, metadata.get("shared_memory", {}))
                
                data_formatting_metadata = {
                    "user_id": request.user_id,
                    "conversation_id": request.conversation_id,
                    "shared_memory": shared_memory,
                    **{k: v for k, v in metadata.items() if k != "shared_memory"}
                }
                
                if cancellation_token.is_set():
                    raise asyncio.CancelledError("Operation cancelled")
                
                result = await self._process_agent_with_cancellation(
                    agent=self.data_formatting_agent,
                    query=request.query,
                    metadata=data_formatting_metadata,
                    messages=messages,
                    cancellation_token=cancellation_token
                )
                
                yield orchestrator_pb2.ChatChunk(
                    type="content",
                    message=result.get("response", "No formatted output generated"),
                    timestamp=datetime.now().isoformat(),
                    agent_name="data_formatting_agent"
                )
                
                yield orchestrator_pb2.ChatChunk(
                    type="complete",
                    message=f"Formatting complete (type: {result.get('format_type', 'structured_text')})",
                    timestamp=datetime.now().isoformat(),
                    agent_name="system"
                )
            
            elif agent_type == "weather":
                yield orchestrator_pb2.ChatChunk(
                    type="status",
                    message="Weather agent checking meteorological conditions...",
                    timestamp=datetime.now().isoformat(),
                    agent_name="weather_agent"
                )
                
                # Build metadata with shared_memory (STANDARD PATTERN)
                shared_memory = self._extract_shared_memory(request, metadata.get("shared_memory", {}))
                
                weather_metadata = {
                    "user_id": request.user_id,
                    "conversation_id": request.conversation_id,
                    "shared_memory": shared_memory,
                    **{k: v for k, v in metadata.items() if k != "shared_memory"}
                }
                
                if cancellation_token.is_set():
                    raise asyncio.CancelledError("Operation cancelled")
                
                result = await self._process_agent_with_cancellation(
                    agent=self.weather_agent,
                    query=request.query,
                    metadata=weather_metadata,
                    messages=messages,
                    cancellation_token=cancellation_token
                )
                
                # Extract response from result (matches BaseAgent response format)
                response_messages = result.get("messages", [])
                if response_messages:
                    response_text = response_messages[-1].content if hasattr(response_messages[-1], 'content') else str(response_messages[-1])
                else:
                    # Fallback to response text if no messages
                    response_text = result.get("response", {}).get("message", "No weather data available") if isinstance(result.get("response"), dict) else result.get("response", "No weather data available")
                
                yield orchestrator_pb2.ChatChunk(
                    type="content",
                    message=response_text,
                    timestamp=datetime.now().isoformat(),
                    agent_name="weather_agent"
                )
                
                # Generate title asynchronously if this is the first message
                # Title generation is now handled by intent classifier (parallel, faster)
                
                is_complete = result.get("is_complete", True)
                status_msg = "Weather check complete" if is_complete else "Weather data incomplete"
                
                yield orchestrator_pb2.ChatChunk(
                    type="complete",
                    message=status_msg,
                    timestamp=datetime.now().isoformat(),
                    agent_name="system"
                )
            
            elif agent_type == "image_generation":
                yield orchestrator_pb2.ChatChunk(
                    type="status",
                    message="Image generation agent creating your images...",
                    timestamp=datetime.now().isoformat(),
                    agent_name="image_generation_agent"
                )
                
                # Build metadata for image generation agent
                shared_memory = self._extract_shared_memory(request, metadata.get("shared_memory", {}))
                
                image_metadata = {
                    "user_id": request.user_id,
                    "conversation_id": request.conversation_id,
                    "shared_memory": shared_memory,  # Extracted, not empty
                    "persona": request.persona if request.HasField("persona") else None,
                    **{k: v for k, v in metadata.items() if k != "shared_memory"}  # Merge other metadata fields
                }
                
                result = await self.image_generation_agent.process(
                    query=request.query,
                    metadata=image_metadata,
                    messages=messages
                )
                
                # Extract response from result
                response_messages = result.get("messages", [])
                response_text = response_messages[-1].content if response_messages else result.get("response", "No images generated")
                
                yield orchestrator_pb2.ChatChunk(
                    type="content",
                    message=response_text,
                    timestamp=datetime.now().isoformat(),
                    agent_name="image_generation_agent"
                )
                
                is_complete = result.get("is_complete", True)
                status_msg = "Image generation complete" if is_complete else "Image generation incomplete"
                
                yield orchestrator_pb2.ChatChunk(
                    type="complete",
                    message=status_msg,
                    timestamp=datetime.now().isoformat(),
                    agent_name="system"
                )
            
            # FactCheckingAgent removed - not actively used
            
            elif agent_type == "rss":
                yield orchestrator_pb2.ChatChunk(
                    type="status",
                    message="RSS agent managing feeds...",
                    timestamp=datetime.now().isoformat(),
                    agent_name="rss_agent"
                )
                
                # Build metadata for RSS agent
                shared_memory = self._extract_shared_memory(request, metadata.get("shared_memory", {}))
                
                rss_metadata = {
                    "user_id": request.user_id,
                    "conversation_id": request.conversation_id,
                    "shared_memory": shared_memory  # Extracted, not empty
                }
                
                result = await self.rss_agent.process(
                    query=request.query,
                    metadata=rss_metadata,
                    messages=messages
                )
                
                # Extract response from result
                response_messages = result.get("messages", [])
                response_text = response_messages[-1].content if response_messages else result.get("response", "No RSS results available")
                
                yield orchestrator_pb2.ChatChunk(
                    type="content",
                    message=response_text,
                    timestamp=datetime.now().isoformat(),
                    agent_name="rss_agent"
                )
                
                is_complete = result.get("is_complete", True)
                agent_results = result.get("agent_results", {})
                rss_operations = agent_results.get("rss_operations", [])
                feeds_processed = len(rss_operations)
                
                status_msg = f"RSS operation complete: {feeds_processed} operations processed"
                
                yield orchestrator_pb2.ChatChunk(
                    type="complete",
                    message=status_msg,
                    timestamp=datetime.now().isoformat(),
                    agent_name="system"
                )
            
            elif agent_type == "entertainment":
                yield orchestrator_pb2.ChatChunk(
                    type="status",
                    message="🎬 Entertainment agent searching movies and TV shows...",
                    timestamp=datetime.now().isoformat(),
                    agent_name="entertainment_agent"
                )
                
                # Build metadata with user_id and shared_memory
                shared_memory = self._extract_shared_memory(request, metadata.get("shared_memory", {}))
                
                entertainment_metadata = {
                    "user_id": request.user_id,
                    "shared_memory": shared_memory,  # Extracted properly
                    **{k: v for k, v in metadata.items() if k != "shared_memory"}
                }
                
                result = await self.entertainment_agent.process(
                    query=request.query,
                    metadata=entertainment_metadata,
                    messages=messages
                )
                
                yield orchestrator_pb2.ChatChunk(
                    type="content",
                    message=result.get("response", "No entertainment content found"),
                    timestamp=datetime.now().isoformat(),
                    agent_name="entertainment_agent"
                )
                
                # Include content type and confidence in complete message
                content_type = result.get("content_type", "mixed")
                confidence = result.get("confidence", 0.0)
                items_count = len(result.get("items_found", []))
                
                status_msg = f"Entertainment search complete: {items_count} items found (type: {content_type}, confidence: {confidence:.2f})"

                yield orchestrator_pb2.ChatChunk(
                    type="complete",
                    message=status_msg,
                    timestamp=datetime.now().isoformat(),
                    agent_name="system"
                )

            elif agent_type == "electronics":
                yield orchestrator_pb2.ChatChunk(
                    type="status",
                    message="🔌 Electronics agent designing circuits and generating code...",
                    timestamp=datetime.now().isoformat(),
                    agent_name="electronics_agent"
                )

                # Build metadata with user_id and shared_memory (using centralized extraction)
                shared_memory = self._extract_shared_memory(request, metadata.get("shared_memory", {}))
                
                # Log shared_memory contents for debugging
                logger.info(f"🔍 ELECTRONICS AGENT STATE: shared_memory keys={list(shared_memory.keys())}")
                if "active_editor" in shared_memory:
                    ae = shared_memory["active_editor"]
                    logger.info(f"✅ ACTIVE EDITOR IN STATE: filename={ae.get('filename')}, type={ae.get('frontmatter', {}).get('type')}, content_length={len(ae.get('content', ''))}")
                else:
                    logger.info(f"⚠️ NO ACTIVE EDITOR IN SHARED_MEMORY")
                
                electronics_metadata = {
                    "user_id": request.user_id,
                    "shared_memory": shared_memory,
                    **{k: v for k, v in metadata.items() if k != "shared_memory"}  # Merge other metadata fields
                }

                if cancellation_token.is_set():
                    raise asyncio.CancelledError("Operation cancelled")
                
                result = await self._process_agent_with_cancellation(
                    agent=self.electronics_agent,
                    query=request.query,
                    metadata=electronics_metadata,
                    messages=messages,
                    cancellation_token=cancellation_token
                )

                # Extract response text - handle various response structures
                response_text = None
                if isinstance(result, dict):
                    response_text = result.get("response")
                    # If response is still a dict, try to extract text from it
                    if isinstance(response_text, dict):
                        response_text = response_text.get("response") or str(response_text)
                    # Ensure we have a string
                    if not isinstance(response_text, str):
                        response_text = str(response_text) if response_text else None
                elif isinstance(result, str):
                    response_text = result
                else:
                    response_text = str(result)
                
                # Fallback if we still don't have text
                if not response_text or response_text.strip() == "":
                    response_text = "Electronics design assistance complete"
                
                # Check for editor operations (editing mode)
                editor_operations = result.get("editor_operations") or result.get("response", {}).get("editor_operations") if isinstance(result.get("response"), dict) else None
                manuscript_edit = result.get("manuscript_edit") or result.get("response", {}).get("manuscript_edit") if isinstance(result.get("response"), dict) else None
                
                if editor_operations:
                    # Send editor operations as separate chunk
                    import json
                    editor_ops_data = {
                        "operations": editor_operations,
                        "manuscript_edit": manuscript_edit
                    }
                    yield orchestrator_pb2.ChatChunk(
                        type="editor_operations",
                        message=json.dumps(editor_ops_data),
                        timestamp=datetime.now().isoformat(),
                        agent_name="electronics_agent"
                    )
                    yield orchestrator_pb2.ChatChunk(
                        type="content",
                        message=response_text,
                        timestamp=datetime.now().isoformat(),
                        agent_name="electronics_agent"
                    )
                else:
                    # Generation mode: send content normally
                    yield orchestrator_pb2.ChatChunk(
                        type="content",
                        message=response_text,
                        timestamp=datetime.now().isoformat(),
                        agent_name="electronics_agent"
                    )

                # Include design type and confidence in complete message
                design_type = result.get("design_type", "general")
                confidence = result.get("confidence", 0.0)
                components_count = len(result.get("components", []))
                code_snippets_count = len(result.get("code_snippets", []))

                status_msg = f"Electronics design complete: {design_type} (components: {components_count}, code: {code_snippets_count}, confidence: {confidence:.2f})"

                yield orchestrator_pb2.ChatChunk(
                    type="complete",
                    message=status_msg,
                    timestamp=datetime.now().isoformat(),
                    agent_name="system"
                )

            elif agent_type == "general_project":
                yield orchestrator_pb2.ChatChunk(
                    type="status",
                    message="General project agent planning and managing your project...",
                    timestamp=datetime.now().isoformat(),
                    agent_name="general_project_agent"
                )

                # Build metadata with user_id and shared_memory
                shared_memory = self._extract_shared_memory(request, metadata.get("shared_memory", {}))
                
                logger.info(f"General project agent state: shared_memory keys={list(shared_memory.keys())}")
                if "active_editor" in shared_memory:
                    ae = shared_memory["active_editor"]
                    logger.info(f"Active editor in state: filename={ae.get('filename')}, type={ae.get('frontmatter', {}).get('type')}, content_length={len(ae.get('content', ''))}")
                else:
                    logger.info(f"No active editor in shared_memory")
                
                general_project_metadata = {
                    "user_id": request.user_id,
                    "shared_memory": shared_memory,
                    **{k: v for k, v in metadata.items() if k != "shared_memory"}
                }

                if cancellation_token.is_set():
                    raise asyncio.CancelledError("Operation cancelled")
                
                result = await self._process_agent_with_cancellation(
                    agent=self.general_project_agent,
                    query=request.query,
                    metadata=general_project_metadata,
                    messages=messages,
                    cancellation_token=cancellation_token
                )

                # Extract response text - handle various response structures
                response_text = None
                if isinstance(result, dict):
                    response_text = result.get("response")
                    # If response is still a dict, try to extract text from it
                    if isinstance(response_text, dict):
                        response_text = response_text.get("response") or str(response_text)
                    # Ensure we have a string
                    if not isinstance(response_text, str):
                        response_text = str(response_text) if response_text else None
                elif isinstance(result, str):
                    response_text = result
                else:
                    response_text = str(result)
                
                # Fallback if we still don't have text
                if not response_text or response_text.strip() == "":
                    response_text = "General project assistance complete"
                
                # Check for editor operations (editing mode)
                editor_operations = result.get("editor_operations") or result.get("response", {}).get("editor_operations") if isinstance(result.get("response"), dict) else None
                manuscript_edit = result.get("manuscript_edit") or result.get("response", {}).get("manuscript_edit") if isinstance(result.get("response"), dict) else None
                
                if editor_operations:
                    # Send editor operations as separate chunk
                    import json
                    editor_ops_data = {
                        "operations": editor_operations,
                        "manuscript_edit": manuscript_edit
                    }
                    yield orchestrator_pb2.ChatChunk(
                        type="editor_operations",
                        message=json.dumps(editor_ops_data),
                        timestamp=datetime.now().isoformat(),
                        agent_name="general_project_agent"
                    )
                    yield orchestrator_pb2.ChatChunk(
                        type="content",
                        message=response_text,
                        timestamp=datetime.now().isoformat(),
                        agent_name="general_project_agent"
                    )
                else:
                    # Generation mode: send content normally
                    yield orchestrator_pb2.ChatChunk(
                        type="content",
                        message=response_text,
                        timestamp=datetime.now().isoformat(),
                        agent_name="general_project_agent"
                    )

                query_type = result.get("query_type", "general") if isinstance(result, dict) else "general"
                confidence = result.get("confidence", 0.0) if isinstance(result, dict) else 0.0

                status_msg = f"General project complete: {query_type} (confidence: {confidence:.2f})"

                yield orchestrator_pb2.ChatChunk(
                    type="complete",
                    message=status_msg,
                    timestamp=datetime.now().isoformat(),
                    agent_name="system"
                )

            elif agent_type == "reference":
                yield orchestrator_pb2.ChatChunk(
                    type="status",
                    message="📚 Reference agent analyzing your document...",
                    timestamp=datetime.now().isoformat(),
                    agent_name="reference_agent"
                )

                # Build metadata with user_id and shared_memory
                shared_memory = self._extract_shared_memory(request, metadata.get("shared_memory", {}))
                
                logger.info(f"Reference agent state: shared_memory keys={list(shared_memory.keys())}")
                if "active_editor" in shared_memory:
                    ae = shared_memory["active_editor"]
                    logger.info(f"Active editor in state: filename={ae.get('filename')}, type={ae.get('frontmatter', {}).get('type')}, content_length={len(ae.get('content', ''))}")
                else:
                    logger.info(f"No active editor in shared_memory")
                
                reference_metadata = {
                    "user_id": request.user_id,
                    "shared_memory": shared_memory,
                    **{k: v for k, v in metadata.items() if k != "shared_memory"}
                }

                if cancellation_token.is_set():
                    raise asyncio.CancelledError("Operation cancelled")
                
                result = await self._process_agent_with_cancellation(
                    agent=self.reference_agent,
                    query=request.query,
                    metadata=reference_metadata,
                    messages=messages,
                    cancellation_token=cancellation_token
                )

                # Extract response text - handle various response structures
                response_text = None
                if isinstance(result, dict):
                    response_text = result.get("response")
                    # If response is still a dict, try to extract text from it
                    if isinstance(response_text, dict):
                        response_text = response_text.get("response") or str(response_text)
                    # Ensure we have a string
                    if not isinstance(response_text, str):
                        response_text = str(response_text) if response_text else None
                elif isinstance(result, str):
                    response_text = result
                else:
                    response_text = str(result)
                
                # Fallback if we still don't have text
                if not response_text or response_text.strip() == "":
                    response_text = "Reference analysis complete"
                
                yield orchestrator_pb2.ChatChunk(
                    type="content",
                    message=response_text,
                    timestamp=datetime.now().isoformat(),
                    agent_name="reference_agent"
                )

                complexity_level = result.get("complexity_level", "simple_qa") if isinstance(result, dict) else "simple_qa"
                confidence = result.get("confidence", 0.0) if isinstance(result, dict) else 0.0
                research_used = result.get("research_used", False) if isinstance(result, dict) else False

                status_msg = f"Reference analysis complete: {complexity_level} (confidence: {confidence:.2f}, research: {research_used})"

                yield orchestrator_pb2.ChatChunk(
                    type="complete",
                    message=status_msg,
                    timestamp=datetime.now().isoformat(),
                    agent_name="system"
                )

            elif agent_type == "character_development":
                yield orchestrator_pb2.ChatChunk(
                    type="status",
                    message="Character development agent processing character edits...",
                    timestamp=datetime.now().isoformat(),
                    agent_name="character_development_agent"
                )
                
                # Build metadata with user_id and shared_memory
                shared_memory = self._extract_shared_memory(request, metadata.get("shared_memory", {}))
                
                character_metadata = {
                    "user_id": request.user_id,
                    "shared_memory": shared_memory,
                    **{k: v for k, v in metadata.items() if k != "shared_memory"}  # Merge other metadata fields
                }
                
                result = await self.character_development_agent.process(
                    query=request.query,
                    metadata=character_metadata,
                    messages=messages
                )
                
                # Extract response from result
                if isinstance(result, dict):
                    agent_messages = result.get("messages", [])
                    if agent_messages:
                        for msg in agent_messages:
                            if hasattr(msg, 'content'):
                                yield orchestrator_pb2.ChatChunk(
                                    type="content",
                                    message=msg.content,
                                    timestamp=datetime.now().isoformat(),
                                    agent_name="character_development_agent"
                                )
                    else:
                        response_text = result.get("response", "Character development complete")
                        yield orchestrator_pb2.ChatChunk(
                            type="content",
                            message=response_text,
                            timestamp=datetime.now().isoformat(),
                            agent_name="character_development_agent"
                        )
                    
                    # Include editor operations in metadata if available
                    agent_results = result.get("agent_results", {})
                    editor_operations = result.get("editor_operations") or agent_results.get("editor_operations")
                    manuscript_edit = result.get("manuscript_edit") or agent_results.get("manuscript_edit")
                    
                    logger.info(f"📥 GRPC SERVICE (character_development_agent): Received result keys: {list(result.keys())}")
                    logger.info(f"📥 GRPC SERVICE: editor_operations from top level: {len(result.get('editor_operations', []))}, from agent_results: {len(agent_results.get('editor_operations', []))}, final: {len(editor_operations) if editor_operations else 0}")
                    
                    if editor_operations:
                        logger.info(f"✅ GRPC SERVICE: Sending {len(editor_operations)} editor operation(s) to frontend")
                        # Send editor operations as separate chunk
                        import json
                        editor_ops_data = {
                            "operations": editor_operations,
                            "manuscript_edit": manuscript_edit
                        }
                        logger.info(f"✅ Sending {len(editor_operations)} editor_operations to frontend (first op: {editor_operations[0] if editor_operations else None})")
                        logger.info(f"🔍 Editor operations data structure: operations_count={len(editor_operations)}, has_manuscript_edit={bool(manuscript_edit)}")
                        if editor_operations:
                            logger.info(f"🔍 First operation details: op_type={editor_operations[0].get('op_type') if editor_operations else 'none'}, start={editor_operations[0].get('start') if editor_operations else 'none'}, end={editor_operations[0].get('end') if editor_operations else 'none'}, has_text={bool(editor_operations[0].get('text')) if editor_operations else False}, text_preview={editor_operations[0].get('text', '')[:100] if editor_operations and editor_operations[0].get('text') else 'N/A'}")
                        yield orchestrator_pb2.ChatChunk(
                            type="editor_operations",
                            message=json.dumps(editor_ops_data),
                            timestamp=datetime.now().isoformat(),
                            agent_name="character_development_agent"
                        )
                        logger.info(f"✅ Sent editor_operations chunk to frontend (type='editor_operations', message_length={len(json.dumps(editor_ops_data))})")
                        yield orchestrator_pb2.ChatChunk(
                            type="complete",
                            message="Character edit plan ready",
                            timestamp=datetime.now().isoformat(),
                            agent_name="system"
                        )
                    else:
                        logger.warning(f"⚠️ GRPC SERVICE: No editor_operations found in result (result keys: {list(result.keys())}, agent_results keys: {list(agent_results.keys())})")
                        yield orchestrator_pb2.ChatChunk(
                            type="complete",
                            message="Character development complete",
                            timestamp=datetime.now().isoformat(),
                            agent_name="system"
                        )
                else:
                    yield orchestrator_pb2.ChatChunk(
                        type="content",
                        message=str(result),
                        timestamp=datetime.now().isoformat(),
                        agent_name="character_development_agent"
                    )
                    yield orchestrator_pb2.ChatChunk(
                        type="complete",
                        message="Character development complete",
                        timestamp=datetime.now().isoformat(),
                        agent_name="system"
                    )

            elif agent_type == "content_analysis":
                yield orchestrator_pb2.ChatChunk(
                    type="status",
                    message="Content analysis agent analyzing documents...",
                    timestamp=datetime.now().isoformat(),
                    agent_name="content_analysis_agent"
                )
                
                # Build metadata with shared_memory
                shared_memory = self._extract_shared_memory(request, metadata.get("shared_memory", {}))
                
                content_analysis_metadata = {
                    "user_id": request.user_id,
                    "conversation_id": request.conversation_id,
                    "shared_memory": shared_memory,  # Extracted properly
                    **{k: v for k, v in metadata.items() if k != "shared_memory"}
                }
                
                result = await self.content_analysis_agent.process(
                    query=request.query,
                    metadata=content_analysis_metadata,
                    messages=messages
                )
                
                # Extract response text from nested structure
                response_obj = result.get("response", {})
                if isinstance(response_obj, dict):
                    response_text = response_obj.get("response", "Content analysis complete")
                else:
                    response_text = str(response_obj) if response_obj else "Content analysis complete"
                
                yield orchestrator_pb2.ChatChunk(
                    type="content",
                    message=response_text,
                    timestamp=datetime.now().isoformat(),
                    agent_name="content_analysis_agent"
                )
                
                yield orchestrator_pb2.ChatChunk(
                    type="complete",
                    message=f"Content analysis complete (status: {result.get('task_status', 'complete')})",
                    timestamp=datetime.now().isoformat(),
                    agent_name="system"
                )

            elif agent_type == "fiction_editing":
                yield orchestrator_pb2.ChatChunk(
                    type="status",
                    message="Fiction editing agent processing manuscript edits...",
                    timestamp=datetime.now().isoformat(),
                    agent_name="fiction_editing_agent"
                )
                
                # Build metadata with user_id and shared_memory
                # Ensure primary_agent_selected is included from metadata
                base_shared_memory = metadata.get("shared_memory", {})
                shared_memory = self._extract_shared_memory(request, base_shared_memory)
                # Ensure primary_agent_selected is preserved (set by intent classification)
                if "primary_agent_selected" in base_shared_memory:
                    shared_memory["primary_agent_selected"] = base_shared_memory["primary_agent_selected"]
                
                fiction_metadata = {
                    "user_id": request.user_id,
                    "shared_memory": shared_memory,
                    **{k: v for k, v in metadata.items() if k != "shared_memory"}  # Merge other metadata fields
                }
                
                result = await self.fiction_editing_agent.process(
                    query=request.query,
                    metadata=fiction_metadata,
                    messages=messages
                )
                
                # Extract response from result
                if isinstance(result, dict):
                    # Include editor operations in metadata if available
                    agent_results = result.get("agent_results", {})
                    editor_operations = result.get("editor_operations") or agent_results.get("editor_operations")
                    manuscript_edit = result.get("manuscript_edit") or agent_results.get("manuscript_edit")
                    
                    # Debug logging for editor operations
                    logger.info(f"🔍 Fiction editing agent result structure: has_editor_ops={bool(result.get('editor_operations'))}, agent_results_ops={bool(agent_results.get('editor_operations'))}, final_ops={bool(editor_operations)}, ops_count={len(editor_operations) if editor_operations else 0}")
                    
                    # Always send content chunk first (for chat sidebar)
                    agent_messages = result.get("messages", [])
                    response_text = None
                    
                    if agent_messages:
                        for msg in agent_messages:
                            if hasattr(msg, 'content'):
                                response_text = msg.content
                                yield orchestrator_pb2.ChatChunk(
                                    type="content",
                                    message=response_text,
                                    timestamp=datetime.now().isoformat(),
                                    agent_name="fiction_editing_agent"
                                )
                    else:
                        # Extract response text from result
                        response_data = result.get("response", {})
                        logger.info(f"🔍 Response extraction: response_data type={type(response_data)}, is_dict={isinstance(response_data, dict)}, keys={list(response_data.keys()) if isinstance(response_data, dict) else 'N/A'}")
                        # Handle both string and dict responses
                        if isinstance(response_data, str):
                            response_text = response_data
                            logger.info(f"🔍 Response extraction: response_data is string, length={len(response_text)}")
                        elif isinstance(response_data, dict):
                            response_text = response_data.get("response", "")
                            logger.info(f"🔍 Response extraction: extracted from dict, response_text length={len(response_text)}, preview={response_text[:100] if response_text else 'EMPTY'}")
                            # Fallback: if response key is empty, use summary from manuscript_edit
                            if not response_text and manuscript_edit:
                                response_text = manuscript_edit.get("summary", "")
                                logger.info(f"🔍 Response extraction: used summary from manuscript_edit, length={len(response_text)}")
                        else:
                            response_text = ""
                            logger.warning(f"🔍 Response extraction: response_data is neither string nor dict: {type(response_data)}")
                        
                        # Fallback: if still no response text, use summary from manuscript_edit
                        if not response_text and manuscript_edit:
                            response_text = manuscript_edit.get("summary", "")
                        
                        # Final fallback: generic message
                        if not response_text:
                            op_count = len(editor_operations) if editor_operations else 0
                            if op_count > 0:
                                response_text = f"Generated {op_count} edit(s) to the manuscript."
                            else:
                                response_text = "Fiction editing complete"
                        
                        # Always send content chunk (even if editor_operations exist)
                        yield orchestrator_pb2.ChatChunk(
                            type="content",
                            message=response_text,
                            timestamp=datetime.now().isoformat(),
                            agent_name="fiction_editing_agent"
                        )
                        logger.info(f"✅ Sent content chunk to frontend (response_text length: {len(response_text)})")
                    
                    # Then send editor operations if available
                    if editor_operations:
                        # Send editor operations as separate chunk
                        import json
                        editor_ops_data = {
                            "operations": editor_operations,
                            "manuscript_edit": manuscript_edit
                        }
                        logger.info(f"✅ Sending {len(editor_operations)} editor_operations to frontend (first op: {editor_operations[0] if editor_operations else None})")
                        logger.info(f"🔍 Editor operations data structure: operations_count={len(editor_operations)}, has_manuscript_edit={bool(manuscript_edit)}")
                        logger.info(f"🔍 First operation details: op_type={editor_operations[0].get('op_type') if editor_operations else 'none'}, start={editor_operations[0].get('start') if editor_operations else 'none'}, end={editor_operations[0].get('end') if editor_operations else 'none'}, has_text={bool(editor_operations[0].get('text')) if editor_operations else False}")
                        yield orchestrator_pb2.ChatChunk(
                            type="editor_operations",
                            message=json.dumps(editor_ops_data),
                            timestamp=datetime.now().isoformat(),
                            agent_name="fiction_editing_agent"
                        )
                        logger.info(f"✅ Sent editor_operations chunk to frontend (type='editor_operations', message_length={len(json.dumps(editor_ops_data))})")
                    else:
                        logger.warning(f"⚠️ No editor_operations to send (result keys: {list(result.keys()) if isinstance(result, dict) else 'not a dict'})")
                    
                        yield orchestrator_pb2.ChatChunk(
                            type="complete",
                            message="Fiction edit plan ready",
                            timestamp=datetime.now().isoformat(),
                            agent_name="system"
                        )
                else:
                    yield orchestrator_pb2.ChatChunk(
                        type="content",
                        message=str(result),
                        timestamp=datetime.now().isoformat(),
                        agent_name="fiction_editing_agent"
                    )
                    yield orchestrator_pb2.ChatChunk(
                        type="complete",
                        message="Fiction editing complete",
                        timestamp=datetime.now().isoformat(),
                        agent_name="system"
                    )

            elif agent_type == "outline_editing":
                yield orchestrator_pb2.ChatChunk(
                    type="status",
                    message="Outline editing agent processing outline edits...",
                    timestamp=datetime.now().isoformat(),
                    agent_name="outline_editing_agent"
                )
                
                # Build metadata with user_id and shared_memory
                shared_memory = self._extract_shared_memory(request, metadata.get("shared_memory", {}))
                
                outline_metadata = {
                    "user_id": request.user_id,
                    "shared_memory": shared_memory,
                    **{k: v for k, v in metadata.items() if k != "shared_memory"}  # Merge other metadata fields
                }
                logger.info(f"🔍 OUTLINE METADATA: user_chat_model = {outline_metadata.get('user_chat_model')}, shared_memory keys = {list(shared_memory.keys())}")
                
                result = await self.outline_editing_agent.process(
                    query=request.query,
                    metadata=outline_metadata,
                    messages=messages
                )
                
                # Extract response text for title generation (handles both dict and string results)
                response_text = self._extract_response_text(result)
                
                # Extract response from result and yield chunks
                # Check for editor operations first (regardless of message structure)
                editor_operations = result.get("editor_operations")
                manuscript_edit = result.get("manuscript_edit")
                
                # Debug logging for editor operations extraction
                logger.info(f"🔍 Outline agent result structure: top_level_ops={bool(editor_operations)}, result_keys={list(result.keys()) if isinstance(result, dict) else 'not a dict'}")
                
                # Also check agent_results and response dict
                if not editor_operations:
                    agent_results = result.get("agent_results", {})
                    editor_operations = agent_results.get("editor_operations")
                    manuscript_edit = manuscript_edit or agent_results.get("manuscript_edit")
                    logger.info(f"🔍 Checked agent_results: ops={bool(editor_operations)}")
                
                if not editor_operations:
                    response_obj = result.get("response", {})
                    if isinstance(response_obj, dict):
                        editor_operations = response_obj.get("editor_operations")
                        manuscript_edit = manuscript_edit or response_obj.get("manuscript_edit")
                        logger.info(f"🔍 Checked response dict: ops={bool(editor_operations)}, response_keys={list(response_obj.keys()) if isinstance(response_obj, dict) else 'not a dict'}")
                
                # Final check - log what we found
                if editor_operations:
                    logger.info(f"✅ Found {len(editor_operations)} editor operations to send")
                else:
                    logger.warning(f"⚠️ No editor_operations found in result structure")
                
                if isinstance(result, dict):
                    agent_messages = result.get("messages", [])
                    if agent_messages:
                        # Stream messages as they come
                        for msg in agent_messages:
                            if hasattr(msg, 'content'):
                                yield orchestrator_pb2.ChatChunk(
                                    type="content",
                                    message=msg.content,
                                    timestamp=datetime.now().isoformat(),
                                    agent_name="outline_editing_agent"
                                )
                        # Update response_text from last message if we have messages
                        if agent_messages:
                            last_msg = agent_messages[-1]
                            if hasattr(last_msg, 'content'):
                                response_text = last_msg.content
                            else:
                                response_text = str(last_msg)
                    else:
                        # No messages, extract from response field
                        response_obj = result.get("response", "")
                        if isinstance(response_obj, dict):
                            response_text = response_obj.get("response", "Outline editing complete")
                        elif isinstance(response_obj, str):
                            response_text = response_obj
                        else:
                            response_text = "Outline editing complete"
                        
                        # Send content (editor operations sent separately below if present)
                        yield orchestrator_pb2.ChatChunk(
                            type="content",
                            message=response_text,
                            timestamp=datetime.now().isoformat(),
                            agent_name="outline_editing_agent"
                        )
                
                else:
                    # Result is a string or other non-dict type
                    response_text = str(result)
                    yield orchestrator_pb2.ChatChunk(
                        type="content",
                        message=response_text,
                        timestamp=datetime.now().isoformat(),
                        agent_name="outline_editing_agent"
                    )
                
                # Send editor operations if present (after content, regardless of message structure)
                if editor_operations and len(editor_operations) > 0:
                    import json
                    editor_ops_data = {
                        "operations": editor_operations,
                        "manuscript_edit": manuscript_edit
                    }
                    logger.info(f"✅ Sending {len(editor_operations)} editor operations to frontend")
                    yield orchestrator_pb2.ChatChunk(
                        type="editor_operations",
                        message=json.dumps(editor_ops_data),
                        timestamp=datetime.now().isoformat(),
                        agent_name="outline_editing_agent"
                    )
                elif editor_operations is not None:
                    logger.warning(f"⚠️ editor_operations is empty list (length={len(editor_operations)})")
                else:
                    logger.warning(f"⚠️ editor_operations is None - not sending operations chunk")
                
                # Generate title asynchronously if this is the first message
                # Title generation is now handled by intent classifier (parallel, faster)
                
                yield orchestrator_pb2.ChatChunk(
                    type="complete",
                    message="Outline editing complete",
                    timestamp=datetime.now().isoformat(),
                    agent_name="system"
                )

            elif agent_type == "rules_editing":
                yield orchestrator_pb2.ChatChunk(
                    type="status",
                    message="Rules editing agent processing rules edits...",
                    timestamp=datetime.now().isoformat(),
                    agent_name="rules_editing_agent"
                )
                
                # Build metadata with user_id and shared_memory
                shared_memory = self._extract_shared_memory(request, metadata.get("shared_memory", {}))
                
                rules_metadata = {
                    "user_id": request.user_id,
                    "shared_memory": shared_memory,
                    **{k: v for k, v in metadata.items() if k != "shared_memory"}  # Merge other metadata fields
                }
                
                result = await self.rules_editing_agent.process(
                    query=request.query,
                    metadata=rules_metadata,
                    messages=messages
                )
                
                # Extract response from result
                if isinstance(result, dict):
                    agent_messages = result.get("messages", [])
                    if agent_messages:
                        for msg in agent_messages:
                            if hasattr(msg, 'content'):
                                yield orchestrator_pb2.ChatChunk(
                                    type="content",
                                    message=msg.content,
                                    timestamp=datetime.now().isoformat(),
                                    agent_name="rules_editing_agent"
                                )
                    else:
                        response_data = result.get("response", "Rules editing complete")
                        # Handle both string and dict responses
                        if isinstance(response_data, str):
                            response_text = response_data
                        elif isinstance(response_data, dict):
                            response_text = response_data.get("response", "Rules editing complete")
                        else:
                            response_text = "Rules editing complete"
                        yield orchestrator_pb2.ChatChunk(
                            type="content",
                            message=response_text,
                            timestamp=datetime.now().isoformat(),
                            agent_name="rules_editing_agent"
                        )
                    
                    # Include editor operations in metadata if available
                    agent_results = result.get("agent_results", {})
                    editor_operations = result.get("editor_operations") or agent_results.get("editor_operations")
                    manuscript_edit = result.get("manuscript_edit") or agent_results.get("manuscript_edit")
                    
                    if editor_operations:
                        # Send editor operations as separate chunk
                        import json
                        editor_ops_data = {
                            "operations": editor_operations,
                            "manuscript_edit": manuscript_edit
                        }
                        yield orchestrator_pb2.ChatChunk(
                            type="editor_operations",
                            message=json.dumps(editor_ops_data),
                            timestamp=datetime.now().isoformat(),
                            agent_name="rules_editing_agent"
                        )
                        yield orchestrator_pb2.ChatChunk(
                            type="complete",
                            message="Rules edit plan ready",
                            timestamp=datetime.now().isoformat(),
                            agent_name="system"
                        )
                    else:
                        yield orchestrator_pb2.ChatChunk(
                            type="complete",
                            message="Rules editing complete",
                            timestamp=datetime.now().isoformat(),
                            agent_name="system"
                        )
                else:
                    yield orchestrator_pb2.ChatChunk(
                        type="content",
                        message=str(result),
                        timestamp=datetime.now().isoformat(),
                        agent_name="rules_editing_agent"
                    )
                    yield orchestrator_pb2.ChatChunk(
                        type="complete",
                        message="Rules editing complete",
                        timestamp=datetime.now().isoformat(),
                        agent_name="system"
                    )

            elif agent_type == "style_editing":
                yield orchestrator_pb2.ChatChunk(
                    type="status",
                    message="Style editing agent processing style guide edits...",
                    timestamp=datetime.now().isoformat(),
                    agent_name="style_editing_agent"
                )
                
                # Build metadata with user_id and shared_memory
                shared_memory = self._extract_shared_memory(request, metadata.get("shared_memory", {}))
                
                style_metadata = {
                    "user_id": request.user_id,
                    "shared_memory": shared_memory,
                    **{k: v for k, v in metadata.items() if k != "shared_memory"}  # Merge other metadata fields
                }
                
                result = await self.style_editing_agent.process(
                    query=request.query,
                    metadata=style_metadata,
                    messages=messages
                )
                
                # Extract response from result
                if isinstance(result, dict):
                    agent_messages = result.get("messages", [])
                    if agent_messages:
                        for msg in agent_messages:
                            if hasattr(msg, 'content'):
                                yield orchestrator_pb2.ChatChunk(
                                    type="content",
                                    message=msg.content,
                                    timestamp=datetime.now().isoformat(),
                                    agent_name="style_editing_agent"
                                )
                    else:
                        response_data = result.get("response", "Style editing complete")
                        # Handle both string and dict responses
                        if isinstance(response_data, str):
                            response_text = response_data
                        elif isinstance(response_data, dict):
                            response_text = response_data.get("response", "Style editing complete")
                        else:
                            response_text = "Style editing complete"
                        yield orchestrator_pb2.ChatChunk(
                            type="content",
                            message=response_text,
                            timestamp=datetime.now().isoformat(),
                            agent_name="style_editing_agent"
                        )
                    
                    # Include editor operations in metadata if available
                    agent_results = result.get("agent_results", {})
                    editor_operations = result.get("editor_operations") or agent_results.get("editor_operations")
                    manuscript_edit = result.get("manuscript_edit") or agent_results.get("manuscript_edit")
                    
                    if editor_operations:
                        # Send editor operations as separate chunk
                        import json
                        editor_ops_data = {
                            "operations": editor_operations,
                            "manuscript_edit": manuscript_edit
                        }
                        yield orchestrator_pb2.ChatChunk(
                            type="editor_operations",
                            message=json.dumps(editor_ops_data),
                            timestamp=datetime.now().isoformat(),
                            agent_name="style_editing_agent"
                        )
                        yield orchestrator_pb2.ChatChunk(
                            type="complete",
                            message="Style edit plan ready",
                            timestamp=datetime.now().isoformat(),
                            agent_name="system"
                        )
                    else:
                        yield orchestrator_pb2.ChatChunk(
                            type="complete",
                            message="Style editing complete",
                            timestamp=datetime.now().isoformat(),
                            agent_name="system"
                        )
                else:
                    yield orchestrator_pb2.ChatChunk(
                        type="content",
                        message=str(result),
                        timestamp=datetime.now().isoformat(),
                        agent_name="style_editing_agent"
                    )
                    yield orchestrator_pb2.ChatChunk(
                        type="complete",
                        message="Style editing complete",
                        timestamp=datetime.now().isoformat(),
                        agent_name="system"
                    )

            elif agent_type == "proofreading":
                yield orchestrator_pb2.ChatChunk(
                    type="status",
                    message="Proofreading agent analyzing manuscript...",
                    timestamp=datetime.now().isoformat(),
                    agent_name="proofreading_agent"
                )
                
                # Build metadata with user_id and shared_memory
                shared_memory = self._extract_shared_memory(request, metadata.get("shared_memory", {}))
                
                proofreading_metadata = {
                    "user_id": request.user_id,
                    "shared_memory": shared_memory,
                    **{k: v for k, v in metadata.items() if k != "shared_memory"}  # Merge other metadata fields
                }
                
                result = await self.proofreading_agent.process(
                    query=request.query,
                    metadata=proofreading_metadata,
                    messages=messages
                )
                
                # Extract response from result
                if isinstance(result, dict):
                    agent_messages = result.get("messages", [])
                    if agent_messages:
                        for msg in agent_messages:
                            if hasattr(msg, 'content'):
                                yield orchestrator_pb2.ChatChunk(
                                    type="content",
                                    message=msg.content,
                                    timestamp=datetime.now().isoformat(),
                                    agent_name="proofreading_agent"
                                )
                    else:
                        response_data = result.get("response", "Proofreading complete")
                        # Handle both string and dict responses
                        if isinstance(response_data, str):
                            response_text = response_data
                        elif isinstance(response_data, dict):
                            response_text = response_data.get("response", "Proofreading complete")
                        else:
                            response_text = "Proofreading complete"
                        yield orchestrator_pb2.ChatChunk(
                            type="content",
                            message=response_text,
                            timestamp=datetime.now().isoformat(),
                            agent_name="proofreading_agent"
                        )
                    
                    # Include editor operations in metadata if available
                    agent_results = result.get("agent_results", {})
                    editor_operations = result.get("editor_operations") or agent_results.get("editor_operations")
                    manuscript_edit = result.get("manuscript_edit") or agent_results.get("manuscript_edit")
                    
                    if editor_operations:
                        # Send editor operations as separate chunk
                        import json
                        editor_ops_data = {
                            "operations": editor_operations,
                            "manuscript_edit": manuscript_edit
                        }
                        yield orchestrator_pb2.ChatChunk(
                            type="editor_operations",
                            message=json.dumps(editor_ops_data),
                            timestamp=datetime.now().isoformat(),
                            agent_name="proofreading_agent"
                        )
                        yield orchestrator_pb2.ChatChunk(
                            type="complete",
                            message="Proofreading corrections ready",
                            timestamp=datetime.now().isoformat(),
                            agent_name="system"
                        )
                    else:
                        yield orchestrator_pb2.ChatChunk(
                            type="complete",
                            message="Proofreading complete",
                            timestamp=datetime.now().isoformat(),
                            agent_name="system"
                        )
                else:
                    yield orchestrator_pb2.ChatChunk(
                        type="content",
                        message=str(result),
                        timestamp=datetime.now().isoformat(),
                        agent_name="proofreading_agent"
                    )
                    yield orchestrator_pb2.ChatChunk(
                        type="complete",
                        message="Proofreading complete",
                        timestamp=datetime.now().isoformat(),
                        agent_name="system"
                    )

            elif agent_type == "story_analysis":
                yield orchestrator_pb2.ChatChunk(
                    type="status",
                    message="Story analysis agent analyzing manuscript...",
                    timestamp=datetime.now().isoformat(),
                    agent_name="story_analysis_agent"
                )
                
                # Build metadata with shared_memory (STANDARD PATTERN)
                shared_memory = self._extract_shared_memory(request, metadata.get("shared_memory", {}))
                
                story_analysis_metadata = {
                    "user_id": request.user_id,
                    "conversation_id": request.conversation_id,
                    "shared_memory": shared_memory,
                    **{k: v for k, v in metadata.items() if k != "shared_memory"}
                }
                
                result = await self.story_analysis_agent.process(
                    query=request.query,
                    metadata=story_analysis_metadata,
                    messages=messages
                )
                
                # Extract response text from nested structure
                # result is the response dict from process() which contains {"task_status": ..., "response": analysis_text, ...}
                logger.info(f"🔍 Story analysis result structure: type={type(result)}, is_dict={isinstance(result, dict)}, keys={list(result.keys()) if isinstance(result, dict) else 'N/A'}")
                
                if isinstance(result, dict):
                    # The result dict has a "response" key containing the actual analysis text
                    response_text = result.get("response", "")
                    if isinstance(response_text, dict):
                        # If response is nested, extract the actual text
                        response_text = response_text.get("response", "Story analysis complete")
                    elif not response_text:
                        # Fallback to structured_response if available
                        structured = result.get("structured_response", {})
                        response_text = structured.get("analysis_text", "Story analysis complete")
                    
                    logger.info(f"🔍 Story analysis response extraction: response_text length={len(response_text) if response_text else 0}, preview={response_text[:200] if response_text else 'EMPTY'}")
                else:
                    response_text = str(result) if result else "Story analysis complete"
                    logger.warning(f"🔍 Story analysis result is not a dict: {type(result)}")
                
                if not response_text or response_text == "Story analysis complete":
                    logger.warning(f"⚠️ Story analysis response is empty or default - result keys: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}")
                
                yield orchestrator_pb2.ChatChunk(
                    type="content",
                    message=response_text,
                    timestamp=datetime.now().isoformat(),
                    agent_name="story_analysis_agent"
                )
                
                yield orchestrator_pb2.ChatChunk(
                    type="complete",
                    message=f"Story analysis complete (status: {result.get('task_status', 'complete')})",
                    timestamp=datetime.now().isoformat(),
                    agent_name="system"
                )

            elif agent_type == "site_crawl":
                yield orchestrator_pb2.ChatChunk(
                    type="status",
                    message="Site crawl agent crawling website...",
                    timestamp=datetime.now().isoformat(),
                    agent_name="site_crawl_agent"
                )
                
                # Build metadata with shared_memory (STANDARD PATTERN)
                shared_memory = self._extract_shared_memory(request, metadata.get("shared_memory", {}))
                
                site_crawl_metadata = {
                    "user_id": request.user_id,
                    "conversation_id": request.conversation_id,
                    "shared_memory": shared_memory,
                    **{k: v for k, v in metadata.items() if k != "shared_memory"}
                }
                
                result = await self.site_crawl_agent.process(
                    query=request.query,
                    metadata=site_crawl_metadata,
                    messages=messages
                )
                
                # Extract response text from nested structure
                response_obj = result.get("response", {})
                if isinstance(response_obj, dict):
                    response_text = response_obj.get("response", "Site crawl complete")
                else:
                    response_text = str(response_obj) if response_obj else "Site crawl complete"
                
                yield orchestrator_pb2.ChatChunk(
                    type="content",
                    message=response_text,
                    timestamp=datetime.now().isoformat(),
                    agent_name="site_crawl_agent"
                )
                
                yield orchestrator_pb2.ChatChunk(
                    type="complete",
                    message=f"Site crawl complete (status: {result.get('task_status', 'complete')})",
                    timestamp=datetime.now().isoformat(),
                    agent_name="system"
                )

            elif agent_type == "org_inbox":
                yield orchestrator_pb2.ChatChunk(
                    type="status",
                    message="Org inbox agent managing inbox...",
                    timestamp=datetime.now().isoformat(),
                    agent_name="org_inbox_agent"
                )
                
                # Build metadata for org inbox agent (STANDARD PATTERN)
                shared_memory = self._extract_shared_memory(request, metadata.get("shared_memory", {}))
                
                org_inbox_metadata = {
                    "user_id": request.user_id,
                    "conversation_id": request.conversation_id,
                    "shared_memory": shared_memory,  # Extracted properly
                    **{k: v for k, v in metadata.items() if k != "shared_memory"}
                }
                
                result = await self.org_inbox_agent.process(
                    query=request.query,
                    metadata=org_inbox_metadata,
                    messages=messages
                )
                
                # Extract response from result
                response_messages = result.get("messages", [])
                response_text = response_messages[-1].content if response_messages else result.get("response", "No org inbox results available")
                
                yield orchestrator_pb2.ChatChunk(
                    type="content",
                    message=response_text,
                    timestamp=datetime.now().isoformat(),
                    agent_name="org_inbox_agent"
                )
                
                is_complete = result.get("is_complete", True)
                agent_results = result.get("agent_results", {})
                org_inbox_result = agent_results.get("org_inbox", {})
                action = org_inbox_result.get("action", "unknown")
                
                status_msg = f"Org inbox operation complete: {action}"
                
                yield orchestrator_pb2.ChatChunk(
                    type="complete",
                    message=status_msg,
                    timestamp=datetime.now().isoformat(),
                    agent_name="system"
                )
            
            elif agent_type == "substack":
                yield orchestrator_pb2.ChatChunk(
                    type="status",
                    message="Substack agent generating article...",
                    timestamp=datetime.now().isoformat(),
                    agent_name="substack_agent"
                )
                
                # Build metadata with shared_memory (STANDARD PATTERN)
                shared_memory = self._extract_shared_memory(request, metadata.get("shared_memory", {}))
                
                substack_metadata = {
                    "user_id": request.user_id,
                    "conversation_id": request.conversation_id,
                    "shared_memory": shared_memory,
                    **{k: v for k, v in metadata.items() if k != "shared_memory"}
                }
                
                result = await self.substack_agent.process(
                    query=request.query,
                    metadata=substack_metadata,
                    messages=messages
                )
                
                # Extract response
                response_messages = result.get("messages", [])
                response_text = response_messages[-1].content if response_messages else result.get("response", "No article generated")
                agent_results = result.get("agent_results", {})
                
                # Check for editor operations (editing mode)
                editor_operations = result.get("editor_operations") or agent_results.get("editor_operations")
                manuscript_edit = result.get("manuscript_edit") or agent_results.get("manuscript_edit")
                
                if editor_operations:
                    # Send editor operations as separate chunk
                    import json
                    editor_ops_data = {
                        "operations": editor_operations,
                        "manuscript_edit": manuscript_edit
                    }
                    yield orchestrator_pb2.ChatChunk(
                        type="editor_operations",
                        message=json.dumps(editor_ops_data),
                        timestamp=datetime.now().isoformat(),
                        agent_name="substack_agent"
                    )
                    yield orchestrator_pb2.ChatChunk(
                        type="content",
                        message=response_text,
                        timestamp=datetime.now().isoformat(),
                        agent_name="substack_agent"
                    )
                else:
                    # Generation mode: send content normally
                    yield orchestrator_pb2.ChatChunk(
                        type="content",
                        message=response_text,
                        timestamp=datetime.now().isoformat(),
                        agent_name="substack_agent"
                    )
                
                yield orchestrator_pb2.ChatChunk(
                    type="complete",
                    message="Article generation complete",
                    timestamp=datetime.now().isoformat(),
                    agent_name="system"
                )
            
            elif agent_type == "podcast_script":
                yield orchestrator_pb2.ChatChunk(
                    type="status",
                    message="Podcast script agent generating script...",
                    timestamp=datetime.now().isoformat(),
                    agent_name="podcast_script_agent"
                )
                
                # Build metadata with shared_memory (STANDARD PATTERN)
                shared_memory = self._extract_shared_memory(request, metadata.get("shared_memory", {}))
                
                podcast_metadata = {
                    "user_id": request.user_id,
                    "conversation_id": request.conversation_id,
                    "shared_memory": shared_memory,
                    **{k: v for k, v in metadata.items() if k != "shared_memory"}
                }
                
                result = await self.podcast_script_agent.process(
                    query=request.query,
                    metadata=podcast_metadata,
                    messages=messages
                )
                
                # Extract response
                response_messages = result.get("messages", [])
                response_text = response_messages[-1].content if response_messages else result.get("response", "No script generated")
                agent_results = result.get("agent_results", {})
                
                # Check for editor operations (editing mode)
                editor_operations = result.get("editor_operations") or agent_results.get("editor_operations")
                manuscript_edit = result.get("manuscript_edit") or agent_results.get("manuscript_edit")
                
                if editor_operations:
                    # Send editor operations as separate chunk
                    import json
                    editor_ops_data = {
                        "operations": editor_operations,
                        "manuscript_edit": manuscript_edit
                    }
                    yield orchestrator_pb2.ChatChunk(
                        type="editor_operations",
                        message=json.dumps(editor_ops_data),
                        timestamp=datetime.now().isoformat(),
                        agent_name="podcast_script_agent"
                    )
                    yield orchestrator_pb2.ChatChunk(
                        type="content",
                        message=response_text,
                        timestamp=datetime.now().isoformat(),
                        agent_name="podcast_script_agent"
                    )
                else:
                    # Generation mode: send content normally
                    yield orchestrator_pb2.ChatChunk(
                        type="content",
                        message=response_text,
                        timestamp=datetime.now().isoformat(),
                        agent_name="podcast_script_agent"
                    )
                
                yield orchestrator_pb2.ChatChunk(
                    type="complete",
                    message="Podcast script generation complete",
                    timestamp=datetime.now().isoformat(),
                    agent_name="system"
                )
            
            elif agent_type == "org_project":
                yield orchestrator_pb2.ChatChunk(
                    type="status",
                    message="Org project agent capturing project...",
                    timestamp=datetime.now().isoformat(),
                    agent_name="org_project_agent"
                )
                
                # Build metadata with shared_memory (STANDARD PATTERN)
                # Extract base shared_memory first, then add pending operations
                shared_memory = self._extract_shared_memory(request, metadata.get("shared_memory", {}))
                
                # Check if there's pending project capture in conversation intelligence
                if request.conversation_intelligence and request.conversation_intelligence.pending_operations:
                    for op in request.conversation_intelligence.pending_operations:
                        if op.operation_type == "project_capture":
                            # Restore pending project state
                            import json
                            try:
                                pending_data = json.loads(op.operation_data) if op.operation_data else {}
                                shared_memory["pending_project_capture"] = pending_data
                            except:
                                pass
                
                org_project_metadata = {
                    "user_id": request.user_id,
                    "conversation_id": request.conversation_id,
                    "shared_memory": shared_memory,
                    **{k: v for k, v in metadata.items() if k != "shared_memory"}
                }
                
                result = await self.org_project_agent.process(
                    query=request.query,
                    metadata=org_project_metadata,
                    messages=messages
                )
                
                # Extract response
                response_messages = result.get("messages", [])
                response_text = response_messages[-1].content if response_messages else result.get("response", "No project captured")
                
                yield orchestrator_pb2.ChatChunk(
                    type="content",
                    message=response_text,
                    timestamp=datetime.now().isoformat(),
                    agent_name="org_project_agent"
                )
                
                # Check if awaiting user input or complete
                agent_results = result.get("agent_results", {})
                task_status = agent_results.get("task_status", "complete")
                
                if task_status in ["permission_required", "incomplete"]:
                    status_msg = "Project preview ready for confirmation" if task_status == "permission_required" else "Additional details needed"
                else:
                    status_msg = "Project capture complete"
                
                yield orchestrator_pb2.ChatChunk(
                    type="complete",
                    message=status_msg,
                    timestamp=datetime.now().isoformat(),
                    agent_name="system"
                )
                
            elif agent_type == "knowledge_builder_agent":
                yield orchestrator_pb2.ChatChunk(
                    type="status",
                    message="Knowledge Builder: Investigating truth and compiling findings...",
                    timestamp=datetime.now().isoformat(),
                    agent_name="knowledge_builder_agent"
                )
                
                # Build metadata with shared_memory
                shared_memory = self._extract_shared_memory(request, metadata.get("shared_memory", {}))
                
                kb_metadata = {
                    "user_id": request.user_id,
                    "conversation_id": request.conversation_id,
                    "shared_memory": shared_memory,
                    **{k: v for k, v in metadata.items() if k != "shared_memory"}
                }
                
                result = await self._process_agent_with_cancellation(
                    agent=self.knowledge_builder_agent,
                    query=request.query,
                    metadata=kb_metadata,
                    messages=messages,
                    cancellation_token=cancellation_token
                )
                
                # Extract response
                response_obj = result.get("response", {})
                if isinstance(response_obj, dict):
                    response_text = response_obj.get("response", "Knowledge document created")
                else:
                    response_text = str(response_obj) if response_obj else "Knowledge document created"
                
                yield orchestrator_pb2.ChatChunk(
                    type="content",
                    message=response_text,
                    timestamp=datetime.now().isoformat(),
                    agent_name="knowledge_builder_agent"
                )
                
                yield orchestrator_pb2.ChatChunk(
                    type="complete",
                    message="Knowledge document saved successfully",
                    timestamp=datetime.now().isoformat(),
                    agent_name="system"
                )
            
            else:  # Default to research
                yield orchestrator_pb2.ChatChunk(
                    type="status",
                    message="Starting sophisticated multi-round research...",
                    timestamp=datetime.now().isoformat(),
                    agent_name="research_agent"
                )
                
                if cancellation_token.is_set():
                    raise asyncio.CancelledError("Operation cancelled")
                
                # Build metadata with shared_memory for research agent
                shared_memory = self._extract_shared_memory(request, metadata.get("shared_memory", {}))
                logger.info(f"🔍 RESEARCH AGENT: shared_memory keys={list(shared_memory.keys())}, user_chat_model={shared_memory.get('user_chat_model')}")
                
                # Prepare research agent metadata with shared_memory (includes user_chat_model)
                research_metadata = {
                    "user_id": request.user_id,
                    "conversation_id": request.conversation_id,
                    "shared_memory": shared_memory,  # Includes user_chat_model
                    **{k: v for k, v in metadata.items() if k != "shared_memory"}
                }
                
                # Research agent uses .process() method (not .research()) to load checkpoint messages
                # This ensures conversation history is available for context
                research_task = asyncio.create_task(
                    self._process_agent_with_cancellation(
                        agent=self.research_agent,
                        query=request.query,
                        metadata=research_metadata,
                        messages=messages,
                        cancellation_token=cancellation_token
                    )
                )
                
                # Wait for either completion or cancellation
                done, pending = await asyncio.wait(
                    [research_task, asyncio.create_task(cancellation_token.wait())],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                
                # Check if cancellation was requested
                if cancellation_token.is_set():
                    research_task.cancel()
                    try:
                        await research_task
                    except asyncio.CancelledError:
                        pass
                    raise asyncio.CancelledError("Operation cancelled")
                
                result = await research_task
                
                # Extract response from process() result format
                # process() returns: {"response": response_text, "task_status": ..., "agent_type": ..., "sources_used": ...}
                response_obj = result.get("response", {})
                if isinstance(response_obj, dict):
                    final_response = response_obj.get("response", "Research complete")
                else:
                    final_response = str(response_obj) if response_obj else "Research complete"
                
                sources_used = result.get("sources_used", [])
                sources_msg = f" - used sources: {', '.join(sources_used)}" if sources_used else ""
                
                yield orchestrator_pb2.ChatChunk(
                    type="status",
                    message=f"Research complete{sources_msg}",
                    timestamp=datetime.now().isoformat(),
                    agent_name="research_agent"
                )
                
                yield orchestrator_pb2.ChatChunk(
                    type="content",
                    message=final_response,
                    timestamp=datetime.now().isoformat(),
                    agent_name="research_agent"
                )
                
                yield orchestrator_pb2.ChatChunk(
                    type="complete",
                    message="Research complete",
                    timestamp=datetime.now().isoformat(),
                    agent_name="system"
                )
            
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
                "agents": "research,chat,data_formatting,help,weather,image_generation,rss,org_inbox,substack,podcast_script,org_project",
                "features": "multi_round_research,query_expansion,gap_analysis,web_search,caching,conversation,formatting,weather_forecasts,image_generation,rss_management,org_inbox_management,article_generation,podcast_script_generation,org_project_capture"
            }
        )

