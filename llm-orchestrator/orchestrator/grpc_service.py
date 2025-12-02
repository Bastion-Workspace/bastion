"""
LLM Orchestrator gRPC Service Implementation
Handles incoming gRPC requests for LLM orchestration
"""

import logging
from datetime import datetime
from typing import AsyncIterator, Optional, Dict, Any

import grpc
from protos import orchestrator_pb2, orchestrator_pb2_grpc
from orchestrator.agents import (
    get_full_research_agent,
    ChatAgent,
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
    get_report_formatting_agent,
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
        self.report_formatting_agent = None
        self.general_project_agent = None
        logger.info("Initializing OrchestratorGRPCService...")
    
    def _ensure_agents_loaded(self):
        """Lazy load agents"""
        if self.research_agent is None:
            self.research_agent = get_full_research_agent()
            logger.info("✅ Full research agent loaded")
        
        if self.chat_agent is None:
            self.chat_agent = ChatAgent()
            logger.info("✅ Chat agent loaded")
        
        if self.data_formatting_agent is None:
            self.data_formatting_agent = DataFormattingAgent()
            logger.info("✅ Data formatting agent loaded")
        
        if self.help_agent is None:
            from orchestrator.agents import HelpAgent
            self.help_agent = HelpAgent()
            logger.info("✅ Help agent loaded")
        
        if self.weather_agent is None:
            self.weather_agent = get_weather_agent()
            logger.info("✅ Weather agent loaded")
        
        if self.image_generation_agent is None:
            self.image_generation_agent = get_image_generation_agent()
            logger.info("✅ Image generation agent loaded")
        
        # FactCheckingAgent removed - not actively used
        
        if self.rss_agent is None:
            self.rss_agent = get_rss_agent()
            logger.info("✅ RSS agent loaded")
        
        if self.org_inbox_agent is None:
            self.org_inbox_agent = get_org_inbox_agent()
            logger.info("✅ Org inbox agent loaded")
        
        if self.substack_agent is None:
            self.substack_agent = get_substack_agent()
            logger.info("✅ Substack agent loaded")
        
        if self.podcast_script_agent is None:
            self.podcast_script_agent = get_podcast_script_agent()
            logger.info("✅ Podcast script agent loaded")
        
        if self.org_project_agent is None:
            self.org_project_agent = get_org_project_agent()
            logger.info("✅ Org project agent loaded")
        
        if self.entertainment_agent is None:
            self.entertainment_agent = get_entertainment_agent()
            logger.info("✅ Entertainment agent loaded")

        if self.electronics_agent is None:
            self.electronics_agent = get_electronics_agent()
            logger.info("✅ Electronics agent loaded")
        
        if self.character_development_agent is None:
            self.character_development_agent = get_character_development_agent()
            logger.info("✅ Character development agent loaded")
        
        if self.content_analysis_agent is None:
            self.content_analysis_agent = get_content_analysis_agent()
            logger.info("✅ Content analysis agent loaded")
        
        if self.fiction_editing_agent is None:
            self.fiction_editing_agent = get_fiction_editing_agent()
            logger.info("✅ Fiction editing agent loaded")
        
        if self.outline_editing_agent is None:
            self.outline_editing_agent = get_outline_editing_agent()
            logger.info("✅ Outline editing agent loaded")
        
        if self.story_analysis_agent is None:
            self.story_analysis_agent = get_story_analysis_agent()
            logger.info("✅ Story analysis agent loaded")
        
        if self.site_crawl_agent is None:
            self.site_crawl_agent = get_site_crawl_agent()
            logger.info("✅ Site crawl agent loaded")
        
        if self.rules_editing_agent is None:
            self.rules_editing_agent = get_rules_editing_agent()
            logger.info("✅ Rules editing agent loaded")
        
        if self.proofreading_agent is None:
            self.proofreading_agent = get_proofreading_agent()
            logger.info("✅ Proofreading agent loaded")
        
        if self.report_formatting_agent is None:
            self.report_formatting_agent = get_report_formatting_agent()
            logger.info("✅ Report formatting agent loaded")
        
        if self.general_project_agent is None:
            self.general_project_agent = get_general_project_agent()
            logger.info("✅ General project agent loaded")
    
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
        """
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
                    "ai_name": request.persona.ai_name if request.persona.ai_name else "Codex",
                    "persona_style": request.persona.persona_style if request.persona.persona_style else "professional",
                    "political_bias": request.persona.political_bias if request.persona.political_bias else "neutral",
                    "timezone": request.persona.timezone if request.persona.timezone else "UTC"
                }
                metadata["persona"] = persona_dict
                logger.info(f"✅ PERSONA: Extracted persona for agents (ai_name={persona_dict['ai_name']}, style={persona_dict['persona_style']})")
            else:
                # Default persona if not provided
                metadata["persona"] = {
                    "ai_name": "Codex",
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
                logger.info(f"📚 Merged checkpoint shared_memory into context: {list(checkpoint_shared_memory.keys())}")
            
            # Determine which agent to use via intent classification
            primary_agent_name = None
            if request.agent_type and request.agent_type != "auto":
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
            
            # Parse conversation history for agent
            messages = []
            for msg in request.conversation_history:
                if msg.role == "user":
                    messages.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    messages.append(AIMessage(content=msg.content))
            
            # Map specialized agents to available migrated agents (fallback)
            # TODO: Remove this mapping as more agents get migrated
            agent_mapping = {
                # Migrated agents
                "chat_agent": "chat",
                "research_agent": "research",
                "data_formatting_agent": "data_formatting",
                "help_agent": "help",
                "weather_agent": "weather",
                "image_generation_agent": "image_generation",
                # FactCheckingAgent removed - not actively used
                "rss_agent": "rss",
                "org_inbox_agent": "org_inbox",
                "entertainment_agent": "entertainment",
                "electronics_agent": "electronics",
                "character_development_agent": "character_development",
                
                # Migrated agents (continued)
                "content_analysis_agent": "content_analysis",
                "fiction_editing_agent": "fiction_editing",
                "outline_editing_agent": "outline_editing",
                "story_analysis_agent": "story_analysis",
                "site_crawl_agent": "site_crawl",
                "rules_editing_agent": "rules_editing",
                "proofreading_agent": "proofreading",
                "report_formatting_agent": "report_formatting",
                "general_project_agent": "general_project",
                
                # Unmigrated agents - map to closest available agent
                "proofreading_agent": "chat",
                "podcast_script_agent": "chat",
                "substack_agent": "chat",
                "org_inbox_agent": "chat",
                "org_project_agent": "chat",
                "website_crawler_agent": "research",
                "pipeline_agent": "data_formatting",
                "image_generation_agent": "chat",
                # WargamingAgent removed - not fully fleshed out
                # SysMLAgent removed - not fully fleshed out
                "rss_agent": "chat",
                "combined_proofread_and_analyze": "chat"
            }
            
            # Map to available agent if not migrated yet
            if agent_type in agent_mapping:
                mapped_agent = agent_mapping[agent_type]
                if mapped_agent != agent_type:
                    logger.info(f"🔄 AGENT MAPPING: {agent_type} → {mapped_agent} (not migrated yet)")
                agent_type = mapped_agent
            else:
                # Unknown agent - default to chat
                logger.warning(f"⚠️ Unknown agent type: {agent_type}, falling back to chat")
                agent_type = "chat"
            
            # Route to appropriate agent
            if agent_type == "chat":
                yield orchestrator_pb2.ChatChunk(
                    type="status",
                    message="Chat agent processing your message...",
                    timestamp=datetime.now().isoformat(),
                    agent_name="orchestrator"
                )
                
                result = await self.chat_agent.process(
                    query=request.query,
                    metadata=metadata,
                    messages=messages
                )
                
                yield orchestrator_pb2.ChatChunk(
                    type="content",
                    message=result.get("response", "No response generated"),
                    timestamp=datetime.now().isoformat(),
                    agent_name="chat_agent"
                )
                
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
                
                result = await self.help_agent.process(
                    query=request.query,
                    metadata=metadata,
                    messages=messages
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
            
            elif agent_type == "data_formatting":
                yield orchestrator_pb2.ChatChunk(
                    type="status",
                    message="Data formatting agent organizing your data...",
                    timestamp=datetime.now().isoformat(),
                    agent_name="orchestrator"
                )
                
                result = await self.data_formatting_agent.process(
                    query=request.query,
                    metadata=metadata,
                    messages=messages
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
                result = await self.weather_agent.process(
                    query=request.query,
                    metadata=metadata,
                    messages=messages
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
                
                # Build state dict for image generation agent
                # Include metadata with model preferences
                state = {
                    "messages": messages + [HumanMessage(content=request.query)],
                    "user_id": request.user_id,
                    "conversation_id": request.conversation_id,
                    "metadata": metadata,  # Includes user_image_model
                    "shared_memory": {},
                    "persona": request.persona if request.HasField("persona") else None
                }
                
                result = await self.image_generation_agent.process(state)
                
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
                
                # Build state dict for RSS agent
                state = {
                    "messages": messages + [HumanMessage(content=request.query)],
                    "user_id": request.user_id,
                    "conversation_id": request.conversation_id,
                    "shared_memory": {}
                }
                
                result = await self.rss_agent.process(state)
                
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

                result = await self.electronics_agent.process(
                    query=request.query,
                    metadata=electronics_metadata,
                    messages=messages
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

                result = await self.general_project_agent.process(
                    query=request.query,
                    metadata=general_project_metadata,
                    messages=messages
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
                
                # Build state dict for character development agent
                state = {
                    "messages": messages,
                    "user_id": request.user_id,
                    "shared_memory": context.get("shared_memory", {}),
                    "metadata": metadata
                }
                
                result = await self.character_development_agent.process(state)
                
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
                    if agent_results.get("editor_operations"):
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
                
                # Build state dict for fiction editing agent
                state = {
                    "messages": messages,
                    "user_id": request.user_id,
                    "shared_memory": context.get("shared_memory", {}),
                    "metadata": metadata
                }
                
                result = await self.fiction_editing_agent.process(state)
                
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
                        response_text = result.get("response", {}).get("response", "Fiction editing complete")
                        if isinstance(response_text, dict):
                            response_text = response_text.get("response", "Fiction editing complete")
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
                
                # Build state dict for outline editing agent
                state = {
                    "messages": messages,
                    "user_id": request.user_id,
                    "shared_memory": context.get("shared_memory", {}),
                    "metadata": metadata
                }
                
                result = await self.outline_editing_agent.process(state)
                
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
                                    agent_name="outline_editing_agent"
                                )
                    else:
                        response_text = result.get("response", {}).get("response", "Outline editing complete")
                        if isinstance(response_text, dict):
                            response_text = response_text.get("response", "Outline editing complete")
                        yield orchestrator_pb2.ChatChunk(
                            type="content",
                            message=response_text,
                            timestamp=datetime.now().isoformat(),
                            agent_name="outline_editing_agent"
                        )
                    
                    # Include editor operations in metadata if available
                    agent_results = result.get("agent_results", {})
                    editor_operations = result.get("editor_operations") or agent_results.get("editor_operations")
                    manuscript_edit = result.get("manuscript_edit") or agent_results.get("manuscript_edit")
                    
                    if editor_operations:
                        yield orchestrator_pb2.ChatChunk(
                            type="complete",
                            message="Outline edit plan ready",
                            timestamp=datetime.now().isoformat(),
                            agent_name="system"
                        )
                    else:
                        yield orchestrator_pb2.ChatChunk(
                            type="complete",
                            message="Outline editing complete",
                            timestamp=datetime.now().isoformat(),
                            agent_name="system"
                        )
                else:
                    yield orchestrator_pb2.ChatChunk(
                        type="content",
                        message=str(result),
                        timestamp=datetime.now().isoformat(),
                        agent_name="outline_editing_agent"
                    )
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
                
                # Build state dict for rules editing agent
                state = {
                    "messages": messages,
                    "user_id": request.user_id,
                    "shared_memory": context.get("shared_memory", {}),
                    "metadata": metadata
                }
                
                result = await self.rules_editing_agent.process(state)
                
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
                        response_text = result.get("response", {}).get("response", "Rules editing complete")
                        if isinstance(response_text, dict):
                            response_text = response_text.get("response", "Rules editing complete")
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
                
                # Build state dict for proofreading agent
                state = {
                    "messages": messages,
                    "user_id": request.user_id,
                    "shared_memory": context.get("shared_memory", {}),
                    "metadata": metadata
                }
                
                result = await self.proofreading_agent.process(state)
                
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
                        response_text = result.get("response", {}).get("response", "Proofreading complete")
                        if isinstance(response_text, dict):
                            response_text = response_text.get("response", "Proofreading complete")
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

            elif agent_type == "report_formatting":
                yield orchestrator_pb2.ChatChunk(
                    type="status",
                    message="Report formatting agent formatting research results...",
                    timestamp=datetime.now().isoformat(),
                    agent_name="orchestrator"
                )
                
                # Build state dict for report formatting agent
                state = {
                    "query": request.query,
                    "messages": messages,
                    "user_id": request.user_id,
                    "shared_memory": context.get("shared_memory", {}),
                    "metadata": metadata
                }
                
                result = await self.report_formatting_agent.process(state)
                
                # Extract response from result
                if isinstance(result, dict):
                    response_text = result.get("response", {}).get("response", "Report formatting complete")
                    if isinstance(response_text, dict):
                        response_text = response_text.get("response", "Report formatting complete")
                    yield orchestrator_pb2.ChatChunk(
                        type="content",
                        message=response_text,
                        timestamp=datetime.now().isoformat(),
                        agent_name="report_formatting_agent"
                    )
                    
                    yield orchestrator_pb2.ChatChunk(
                        type="complete",
                        message="Report formatting complete",
                        timestamp=datetime.now().isoformat(),
                        agent_name="system"
                    )
                else:
                    yield orchestrator_pb2.ChatChunk(
                        type="content",
                        message=str(result),
                        timestamp=datetime.now().isoformat(),
                        agent_name="report_formatting_agent"
                    )
                    yield orchestrator_pb2.ChatChunk(
                        type="complete",
                        message="Report formatting complete",
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
                
                # Build state dict for org inbox agent
                state = {
                    "messages": messages + [HumanMessage(content=request.query)],
                    "user_id": request.user_id,
                    "conversation_id": request.conversation_id,
                    "shared_memory": {},
                    "persona": {}  # Add persona if available from context
                }
                
                result = await self.org_inbox_agent.process(state)
                
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
                
                # Build state dict with shared_memory (using centralized extraction)
                shared_memory = self._extract_shared_memory(request)
                
                state = {
                    "messages": messages + [HumanMessage(content=request.query)],
                    "user_id": request.user_id,
                    "conversation_id": request.conversation_id,
                    "shared_memory": shared_memory,
                    "persona": {}
                }
                
                result = await self.substack_agent.process(state)
                
                # Extract response
                response_messages = result.get("messages", [])
                response_text = response_messages[-1].content if response_messages else result.get("response", "No article generated")
                
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
                
                # Build state dict with shared_memory (using centralized extraction)
                shared_memory = self._extract_shared_memory(request)
                
                state = {
                    "messages": messages + [HumanMessage(content=request.query)],
                    "user_id": request.user_id,
                    "conversation_id": request.conversation_id,
                    "shared_memory": shared_memory,
                    "persona": {}
                }
                
                result = await self.podcast_script_agent.process(state)
                
                # Extract response
                response_messages = result.get("messages", [])
                response_text = response_messages[-1].content if response_messages else result.get("response", "No script generated")
                
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
                
                state = {
                    "messages": messages + [HumanMessage(content=request.query)],
                    "user_id": request.user_id,
                    "conversation_id": request.conversation_id,
                    "shared_memory": shared_memory,
                    "persona": {}
                }
                
                result = await self.org_project_agent.process(state)
                
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
                
                result = await self.research_agent.research(
                    query=request.query,
                    conversation_id=request.conversation_id
                )
                
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
                
                completion_msg = f"Multi-round research complete ({current_round})"
                yield orchestrator_pb2.ChatChunk(
                    type="complete",
                    message=completion_msg,
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

