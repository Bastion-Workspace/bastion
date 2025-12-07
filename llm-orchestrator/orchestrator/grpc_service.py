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
    get_proofreading_agent,
    get_general_project_agent
)
from orchestrator.services import get_intent_classifier
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)


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
        self.proofreading_agent = None
        self.general_project_agent = None
        logger.info("Initializing OrchestratorGRPCService...")
    
    def _ensure_agents_loaded(self):
        """Lazy load agents"""
        agents_loaded = 0
        total_agents = 25  # Total number of agents to load
        
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
        
        if self.proofreading_agent is None:
            self.proofreading_agent = get_proofreading_agent()
            agents_loaded += 1
        
        if self.general_project_agent is None:
            self.general_project_agent = get_general_project_agent()
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

            shared_memory["active_editor"] = {
                "is_editable": request.active_editor.is_editable,
                "filename": request.active_editor.filename,
                "file_path": request.active_editor.filename,  # Proto doesn't have file_path, use filename
                "canonical_path": canonical_path,  # Full filesystem path for resolving relative references
                "language": request.active_editor.language,
                "content": request.active_editor.content,
                "frontmatter": {
                    "type": request.active_editor.frontmatter.type,
                    "title": request.active_editor.frontmatter.title,
                    "author": request.active_editor.frontmatter.author,
                    "tags": list(request.active_editor.frontmatter.tags),
                    "status": request.active_editor.frontmatter.status,
                    **frontmatter_custom
                }
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
            return await agent.process_with_cancellation(
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
            return await process_task
    
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
            
            # Build conversation context from proto fields for intent classification
            conversation_context = self._extract_conversation_context(request)
            
            # Merge checkpoint shared_memory into context for intent classifier
            if checkpoint_shared_memory:
                conversation_context["shared_memory"].update(checkpoint_shared_memory)
                # Log specifically about agent continuity for debugging
                primary_agent = checkpoint_shared_memory.get("primary_agent_selected")
                last_agent = checkpoint_shared_memory.get("last_agent")
                if primary_agent or last_agent:
                    logger.info(f"📚 Loaded agent continuity from checkpoint: primary_agent={primary_agent}, last_agent={last_agent}")
                else:
                    logger.info(f"📚 Merged checkpoint shared_memory (no agent continuity): {list(checkpoint_shared_memory.keys())}")
            
            # Determine which agent to use via intent classification
            primary_agent_name = None
            
            # SHORT-CIRCUIT ROUTING: Check for "define:" prefix for instant dictionary routing
            if request.query.lower().strip().startswith("define:"):
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
                    agent_name="orchestrator"
                )
                
                # Check for cancellation before processing
                if cancellation_token.is_set():
                    raise asyncio.CancelledError("Operation cancelled")
                
                result = await self._process_agent_with_cancellation(
                    agent=self.chat_agent,
                    query=request.query,
                    metadata=metadata,
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
                    agent_name="orchestrator"
                )
                
                if cancellation_token.is_set():
                    raise asyncio.CancelledError("Operation cancelled")
                
                result = await self._process_agent_with_cancellation(
                    agent=self.help_agent,
                    query=request.query,
                    metadata=metadata,
                    messages=messages,
                    cancellation_token=cancellation_token
                )
                
                yield orchestrator_pb2.ChatChunk(
                    type="content",
                    message=result.get("response", "No help content available"),
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
                    agent_name="orchestrator"
                )
                
                if cancellation_token.is_set():
                    raise asyncio.CancelledError("Operation cancelled")
                
                result = await self._process_agent_with_cancellation(
                    agent=self.dictionary_agent,
                    query=request.query,
                    metadata=metadata,
                    messages=messages,
                    cancellation_token=cancellation_token
                )
                
                response_text = result.get("message", "No definition available")
                
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
                    agent_name="orchestrator"
                )
                
                if cancellation_token.is_set():
                    raise asyncio.CancelledError("Operation cancelled")
                
                result = await self._process_agent_with_cancellation(
                    agent=self.data_formatting_agent,
                    query=request.query,
                    metadata=metadata,
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
                    agent_name="orchestrator"
                )
                
                # Process using BaseAgent pattern (query, metadata, messages)
                # metadata already contains persona from top of StreamChat
                if cancellation_token.is_set():
                    raise asyncio.CancelledError("Operation cancelled")
                
                result = await self._process_agent_with_cancellation(
                    agent=self.weather_agent,
                    query=request.query,
                    metadata=metadata,
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
                    agent_name="orchestrator"
                )
                
                # Build metadata for image generation agent
                image_metadata = {
                    "user_id": request.user_id,
                    "conversation_id": request.conversation_id,
                    "shared_memory": {},
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
                    agent_name="orchestrator"
                )
                
                # Build metadata for RSS agent
                rss_metadata = {
                    "user_id": request.user_id,
                    "conversation_id": request.conversation_id,
                    "shared_memory": {}
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
                    agent_name="orchestrator"
                )
                
                # Build metadata with user_id
                entertainment_metadata = {
                    "user_id": request.user_id,
                    **metadata
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
                    agent_name="orchestrator"
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
                    agent_name="orchestrator"
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

            elif agent_type == "character_development":
                yield orchestrator_pb2.ChatChunk(
                    type="status",
                    message="Character development agent processing character edits...",
                    timestamp=datetime.now().isoformat(),
                    agent_name="orchestrator"
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
                            agent_name="character_development_agent"
                        )
                        yield orchestrator_pb2.ChatChunk(
                            type="complete",
                            message="Character edit plan ready",
                            timestamp=datetime.now().isoformat(),
                            agent_name="system"
                        )
                    else:
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
                    agent_name="orchestrator"
                )
                
                result = await self.content_analysis_agent.process(
                    query=request.query,
                    metadata=metadata,
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
                    agent_name="orchestrator"
                )
                
                # Build metadata with user_id and shared_memory
                shared_memory = self._extract_shared_memory(request, metadata.get("shared_memory", {}))
                
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
                    agent_messages = result.get("messages", [])
                    if agent_messages:
                        for msg in agent_messages:
                            if hasattr(msg, 'content'):
                                yield orchestrator_pb2.ChatChunk(
                                    type="content",
                                    message=msg.content,
                                    timestamp=datetime.now().isoformat(),
                                    agent_name="fiction_editing_agent"
                                )
                    else:
                        response_data = result.get("response", "Fiction editing complete")
                        # Handle both string and dict responses
                        if isinstance(response_data, str):
                            response_text = response_data
                        elif isinstance(response_data, dict):
                            response_text = response_data.get("response", "Fiction editing complete")
                        else:
                            response_text = "Fiction editing complete"
                        yield orchestrator_pb2.ChatChunk(
                            type="content",
                            message=response_text,
                            timestamp=datetime.now().isoformat(),
                            agent_name="fiction_editing_agent"
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
                            agent_name="fiction_editing_agent"
                        )
                        yield orchestrator_pb2.ChatChunk(
                            type="complete",
                            message="Fiction edit plan ready",
                            timestamp=datetime.now().isoformat(),
                            agent_name="system"
                        )
                    else:
                        yield orchestrator_pb2.ChatChunk(
                            type="complete",
                            message="Fiction editing complete",
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
                    agent_name="orchestrator"
                )
                
                # Build metadata with user_id and shared_memory
                shared_memory = self._extract_shared_memory(request, metadata.get("shared_memory", {}))
                
                outline_metadata = {
                    "user_id": request.user_id,
                    "shared_memory": shared_memory,
                    **{k: v for k, v in metadata.items() if k != "shared_memory"}  # Merge other metadata fields
                }
                
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
                
                # Also check agent_results and response dict
                if not editor_operations:
                    agent_results = result.get("agent_results", {})
                    editor_operations = agent_results.get("editor_operations")
                    manuscript_edit = manuscript_edit or agent_results.get("manuscript_edit")
                
                if not editor_operations:
                    response_obj = result.get("response", {})
                    if isinstance(response_obj, dict):
                        editor_operations = response_obj.get("editor_operations")
                        manuscript_edit = manuscript_edit or response_obj.get("manuscript_edit")
                
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
                if editor_operations:
                    import json
                    editor_ops_data = {
                        "operations": editor_operations,
                        "manuscript_edit": manuscript_edit
                    }
                    yield orchestrator_pb2.ChatChunk(
                        type="editor_operations",
                        message=json.dumps(editor_ops_data),
                        timestamp=datetime.now().isoformat(),
                        agent_name="outline_editing_agent"
                    )
                
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
                    agent_name="orchestrator"
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

            elif agent_type == "proofreading":
                yield orchestrator_pb2.ChatChunk(
                    type="status",
                    message="Proofreading agent analyzing manuscript...",
                    timestamp=datetime.now().isoformat(),
                    agent_name="orchestrator"
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
                    agent_name="orchestrator"
                )
                
                result = await self.story_analysis_agent.process(
                    query=request.query,
                    metadata=metadata,
                    messages=messages
                )
                
                # Extract response text from nested structure
                response_obj = result.get("response", {})
                if isinstance(response_obj, dict):
                    response_text = response_obj.get("response", "Story analysis complete")
                else:
                    response_text = str(response_obj) if response_obj else "Story analysis complete"
                
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
                    agent_name="orchestrator"
                )
                
                result = await self.site_crawl_agent.process(
                    query=request.query,
                    metadata=metadata,
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
                    agent_name="orchestrator"
                )
                
                # Build metadata for org inbox agent
                org_inbox_metadata = {
                    "user_id": request.user_id,
                    "conversation_id": request.conversation_id,
                    "shared_memory": {},
                    "persona": {}  # Add persona if available from context
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
                    agent_name="orchestrator"
                )
                
                # Build metadata with shared_memory (using centralized extraction)
                shared_memory = self._extract_shared_memory(request)
                
                substack_metadata = {
                    "user_id": request.user_id,
                    "conversation_id": request.conversation_id,
                    "shared_memory": shared_memory,
                    "persona": {}
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
                    agent_name="orchestrator"
                )
                
                # Build metadata with shared_memory (using centralized extraction)
                shared_memory = self._extract_shared_memory(request)
                
                podcast_metadata = {
                    "user_id": request.user_id,
                    "conversation_id": request.conversation_id,
                    "shared_memory": shared_memory,
                    "persona": {}
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
                    agent_name="orchestrator"
                )
                
                # Build state dict with shared_memory for pending state
                shared_memory = {}
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
                    "persona": {}
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
                
            else:  # Default to research
                yield orchestrator_pb2.ChatChunk(
                    type="status",
                    message="Starting sophisticated multi-round research...",
                    timestamp=datetime.now().isoformat(),
                    agent_name="orchestrator"
                )
                
                if cancellation_token.is_set():
                    raise asyncio.CancelledError("Operation cancelled")
                
                # Research agent uses .research() method, wrap with cancellation
                research_task = asyncio.create_task(
                    self.research_agent.research(
                        query=request.query,
                        conversation_id=request.conversation_id
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
                
                current_round = result.get("current_round", "")
                sources_used = result.get("sources_used", [])
                
                yield orchestrator_pb2.ChatChunk(
                    type="status",
                    message=f"Research complete - used sources: {', '.join(sources_used)}",
                    timestamp=datetime.now().isoformat(),
                    agent_name="research_agent"
                )
                
                if result.get("cache_hit"):
                    yield orchestrator_pb2.ChatChunk(
                        type="status",
                        message="Used cached research from previous conversation",
                        timestamp=datetime.now().isoformat(),
                        agent_name="research_agent"
                    )
                
                final_response = result.get("final_response", "No response generated")
                yield orchestrator_pb2.ChatChunk(
                    type="content",
                    message=final_response,
                    timestamp=datetime.now().isoformat(),
                    agent_name="research_agent"
                )
                
                # Generate title asynchronously if this is the first message
                # Title generation is now handled by intent classifier (parallel, faster)
                
                completion_msg = f"Multi-round research complete ({current_round})"
                yield orchestrator_pb2.ChatChunk(
                    type="complete",
                    message=completion_msg,
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

