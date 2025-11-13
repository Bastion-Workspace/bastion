"""
LLM Orchestrator gRPC Service Implementation
Handles incoming gRPC requests for LLM orchestration
"""

import logging
from datetime import datetime
from typing import AsyncIterator

import grpc
from protos import orchestrator_pb2, orchestrator_pb2_grpc
from orchestrator.agents import (
    get_full_research_agent,
    ChatAgent,
    DataFormattingAgent,
    get_weather_agent,
    get_image_generation_agent,
    get_fact_checking_agent,
    get_rss_agent,
    get_org_inbox_agent,
    get_substack_agent,
    get_podcast_script_agent,
    get_org_project_agent,
    get_entertainment_agent
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
        self.fact_checking_agent = None
        self.rss_agent = None
        self.org_inbox_agent = None
        self.substack_agent = None
        self.podcast_script_agent = None
        self.org_project_agent = None
        self.entertainment_agent = None
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
        
        if self.fact_checking_agent is None:
            self.fact_checking_agent = get_fact_checking_agent()
            logger.info("✅ Fact-checking agent loaded")
        
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
        for msg in request.conversation_history:
            context["messages"].append({
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp
            })
        
        # Extract shared memory fields
        if request.HasField("active_editor"):
            context["shared_memory"]["active_editor"] = {
                "is_editable": request.active_editor.is_editable,
                "filename": request.active_editor.filename,
                "language": request.active_editor.language,
                "content": request.active_editor.content,
                "frontmatter": {
                    "type": request.active_editor.frontmatter.type,
                    "title": request.active_editor.frontmatter.title,
                    "author": request.active_editor.frontmatter.author,
                    "tags": list(request.active_editor.frontmatter.tags),
                    "status": request.active_editor.frontmatter.status,
                    **dict(request.active_editor.frontmatter.custom_fields)
                }
            }
        
        if request.HasField("pipeline_context"):
            context["shared_memory"]["active_pipeline_id"] = request.pipeline_context.active_pipeline_id
            context["shared_memory"]["pipeline_preference"] = request.pipeline_context.pipeline_preference
        
        # Extract permission grants
        if request.HasField("permission_grants"):
            if request.permission_grants.web_search_permission:
                context["shared_memory"]["web_search_permission"] = True
            if request.permission_grants.web_crawl_permission:
                context["shared_memory"]["web_crawl_permission"] = True
            if request.permission_grants.file_write_permission:
                context["shared_memory"]["file_write_permission"] = True
            if request.permission_grants.external_api_permission:
                context["shared_memory"]["external_api_permission"] = True
        
        # Extract conversation intelligence (if provided)
        if request.HasField("conversation_intelligence"):
            # This would be populated if backend sends it
            # For now, basic structure
            context["conversation_intelligence"] = {
                "agent_outputs": {}
            }
        
        return context
    
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
            
            # Build conversation context from proto fields for intent classification
            conversation_context = self._extract_conversation_context(request)
            
            # Determine which agent to use via intent classification
            if request.agent_type and request.agent_type != "auto":
                # Explicit agent routing provided by backend
                agent_type = request.agent_type
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
                "fact_checking_agent": "fact_checking",
                "rss_agent": "rss",
                "org_inbox_agent": "org_inbox",
                "entertainment_agent": "entertainment",
                
                # Unmigrated agents - map to closest available agent
                "fiction_editing_agent": "chat",  # Will migrate soon
                "story_analysis_agent": "chat",
                "content_analysis_agent": "chat",
                "proofreading_agent": "chat",
                "outline_editing_agent": "chat",
                "rules_editing_agent": "chat",
                "character_development_agent": "chat",
                "podcast_script_agent": "chat",
                "substack_agent": "chat",
                "org_inbox_agent": "chat",
                "org_project_agent": "chat",
                "website_crawler_agent": "research",
                "pipeline_agent": "data_formatting",
                "image_generation_agent": "chat",
                "wargaming_agent": "chat",
                "sysml_agent": "chat",
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
                
                # Build state dict for weather agent
                state = {
                    "messages": messages + [HumanMessage(content=request.query)],
                    "user_id": request.user_id,
                    "conversation_id": request.conversation_id,
                    "shared_memory": {},
                    "persona": request.persona.style if request.HasField("persona") else "casual"
                }
                
                result = await self.weather_agent.process(state)
                
                # Extract response from result
                response_messages = result.get("messages", [])
                response_text = response_messages[-1].content if response_messages else result.get("response", "No weather data available")
                
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
                state = {
                    "messages": messages + [HumanMessage(content=request.query)],
                    "user_id": request.user_id,
                    "conversation_id": request.conversation_id,
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
            
            elif agent_type == "fact_checking":
                yield orchestrator_pb2.ChatChunk(
                    type="status",
                    message="Fact-checking agent verifying claims...",
                    timestamp=datetime.now().isoformat(),
                    agent_name="orchestrator"
                )
                
                # Build state dict for fact-checking agent
                state = {
                    "messages": messages + [HumanMessage(content=request.query)],
                    "user_id": request.user_id,
                    "conversation_id": request.conversation_id,
                    "shared_memory": {},
                    "content": request.query  # Content to fact-check
                }
                
                result = await self.fact_checking_agent.process(state)
                
                # Extract response from result
                response_messages = result.get("messages", [])
                response_text = response_messages[-1].content if response_messages else result.get("response", "No fact-checking results available")
                
                yield orchestrator_pb2.ChatChunk(
                    type="content",
                    message=response_text,
                    timestamp=datetime.now().isoformat(),
                    agent_name="fact_checking_agent"
                )
                
                is_complete = result.get("is_complete", True)
                agent_results = result.get("agent_results", {})
                fact_check_results = agent_results.get("fact_checking_results", {})
                claims_verified = fact_check_results.get("claims_verified", 0)
                claims_found = fact_check_results.get("claims_found", 0)
                
                status_msg = f"Fact-checking complete: {claims_verified}/{claims_found} claims verified"
                
                yield orchestrator_pb2.ChatChunk(
                    type="complete",
                    message=status_msg,
                    timestamp=datetime.now().isoformat(),
                    agent_name="system"
                )
            
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
                
                # Build state dict with active editor from shared_memory
                shared_memory = {}
                if request.active_editor.content or request.active_editor.file_path:
                    shared_memory["active_editor"] = {
                        "content": request.active_editor.content,
                        "file_path": request.active_editor.file_path,
                        "language": request.active_editor.language,
                        "frontmatter": dict(request.active_editor.frontmatter) if request.active_editor.frontmatter else {}
                    }
                
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
                
                # Build state dict with active editor from shared_memory
                shared_memory = {}
                if request.active_editor.content or request.active_editor.file_path:
                    shared_memory["active_editor"] = {
                        "content": request.active_editor.content,
                        "file_path": request.active_editor.file_path,
                        "language": request.active_editor.language,
                        "frontmatter": dict(request.active_editor.frontmatter) if request.active_editor.frontmatter else {}
                    }
                
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
                "agents": "research,chat,data_formatting,help,weather,image_generation,fact_checking,rss,org_inbox,substack,podcast_script,org_project",
                "features": "multi_round_research,query_expansion,gap_analysis,web_search,caching,conversation,formatting,weather_forecasts,image_generation,fact_verification,rss_management,org_inbox_management,article_generation,podcast_script_generation,org_project_capture"
            }
        )

