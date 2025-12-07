"""
Base Agent Class - Roosevelt's "LangGraph Gold Standard" Implementation
Common functionality for all LangGraph agents with best practices
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from enum import Enum

from services.service_container import service_container
from services.langgraph_tools.centralized_tool_registry import (
    get_tool_registry, AgentType, get_agent_tools, get_tool_function
)
# Import will be done lazily when needed

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Agent task completion status"""
    COMPLETE = "complete"
    INCOMPLETE = "incomplete" 
    PERMISSION_REQUIRED = "permission_required"
    ERROR = "error"


class AgentError:
    """Structured error information for agent failures"""
    
    def __init__(self, error_type: str, message: str, recovery_actions: List[str] = None):
        self.error_type = error_type
        self.message = message
        self.recovery_actions = recovery_actions or []
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type,
            "message": self.message,
            "recovery_actions": self.recovery_actions,
            "timestamp": self.timestamp
        }


class BaseAgent:
    """
    Base class for all LangGraph agents following Roosevelt's best practices
    
    Key principles:
    - Additive state updates (never mutate input state)
    - Centralized tool access through registry
    - Structured error handling with recovery
    - Type-safe agent results
    """
    
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.agent_type_enum = self._get_agent_type_enum(agent_type)
        self.chat_service = None
        self.available_tools: List[str] = []
        self._tool_registry = None
        self._initialize_services()
    
    def _get_agent_type_enum(self, agent_type: str) -> AgentType:
        """Convert string agent type to enum"""
        type_mapping = {
            "research_agent": AgentType.RESEARCH_AGENT,
            "chat_agent": AgentType.CHAT_AGENT,
            # CodingAgent removed - not fully fleshed out
            "report_formatting_agent": AgentType.REPORT_AGENT,
            "data_formatting_agent": AgentType.DATA_FORMATTING_AGENT,
            "weather_agent": AgentType.WEATHER_AGENT,
            "calculate_agent": AgentType.CALCULATE_AGENT,
            "rss_background_agent": AgentType.RSS_BACKGROUND_AGENT,
            "rss_agent": AgentType.RSS_AGENT,
            "org_inbox_agent": AgentType.ORG_INBOX_AGENT,
            "org_project_agent": AgentType.ORG_PROJECT_AGENT,
            "image_generation_agent": AgentType.IMAGE_GENERATION_AGENT,
            # WargamingAgent removed - not fully fleshed out
            "proofreading_agent": AgentType.PROOFREADING_AGENT,
            "website_crawler_agent": AgentType.WEBSITE_CRAWLER_AGENT,
            # Content and Writing Agents
            "fiction_editing_agent": AgentType.FICTION_EDITING_AGENT,
            "outline_editing_agent": AgentType.OUTLINE_EDITING_AGENT,
            "character_development_agent": AgentType.CHARACTER_DEVELOPMENT_AGENT,
            "rules_editing_agent": AgentType.RULES_EDITING_AGENT,
            # SysMLAgent removed - not fully fleshed out
            "story_analysis_agent": AgentType.STORY_ANALYSIS_AGENT,
            "content_analysis_agent": AgentType.CONTENT_ANALYSIS_AGENT,
            # FactCheckingAgent removed - not actively used
            "site_crawl_agent": AgentType.SITE_CRAWL_AGENT,
            "podcast_script_agent": AgentType.PODCAST_SCRIPT_AGENT,
            "substack_agent": AgentType.SUBSTACK_AGENT,
            "email_agent": AgentType.EMAIL_AGENT,
            "messaging_agent": AgentType.MESSAGING_AGENT,
            "entertainment_agent": AgentType.ENTERTAINMENT_AGENT,
            # Intent and Intelligence Agents
            # DEPRECATED: simple_intent_agent removed - intent classification now in llm-orchestrator
            # "simple_intent_agent": AgentType.SIMPLE_INTENT_AGENT,
            "permission_intelligence_agent": AgentType.PERMISSION_INTELLIGENCE_AGENT,
            # Pipeline Agent
            "pipeline_agent": AgentType.PIPELINE_AGENT,
            # Template Agent
            "template_agent": AgentType.TEMPLATE_AGENT,
            "podcast_script_agent": AgentType.PODCAST_SCRIPT_AGENT,
        }
        return type_mapping.get(agent_type, AgentType.RESEARCH_AGENT)
    
    @property
    def tool_registry(self):
        """Public access to tool registry (lazy initialization handled by _initialize_tools)"""
        return self._tool_registry
    
    def _initialize_services(self):
        """Initialize required services"""
        # Don't initialize services immediately - use lazy loading
        self.chat_service = None
    
    async def _get_chat_service(self):
        """Get chat service with performance optimizations"""
        if self.chat_service is None:
            try:
                # ROOSEVELT'S SIMPLIFIED APPROACH: Always try service container first!
                from services.service_container import service_container
                
                # Ensure service container is initialized
                if not service_container.is_initialized:
                    logger.info(f"🔄 Initializing service container for {self.agent_type}")
                    await service_container.initialize()
                
                # Use the main service container chat service (properly initialized with model)
                if service_container.chat_service:
                    self.chat_service = service_container.chat_service
                    logger.debug(f"✅ Using service container chat service for {self.agent_type}")
                    return self.chat_service
                
                # PERFORMANCE OPTIMIZATION: Try pre-warmed service as fallback
                from services.worker_warmup import worker_warmup_service
                
                warmed_chat_service = worker_warmup_service.get_service('chat')
                if warmed_chat_service:
                    self.chat_service = warmed_chat_service
                    logger.debug(f"🔥 Using pre-warmed chat service for {self.agent_type}")
                    return self.chat_service
                
                # For orchestrator agents, use lazy loading for better performance
                if self.agent_type == "orchestrator_agent":
                    logger.debug(f"⚡ Using lazy chat service for {self.agent_type}")
                    from services.lazy_chat_service import LazyChatService
                    self.chat_service = LazyChatService()
                    await self.chat_service.initialize_minimal()
                    return self.chat_service
                
                # Last resort - initialize directly
                logger.info(f"🔄 Initializing chat service directly for {self.agent_type}")
                from services.chat_service import ChatService
                self.chat_service = ChatService()
                await self.chat_service.initialize()
                    
                if not self.chat_service:
                    raise Exception("Chat service not available")
                    
            except Exception as e:
                logger.error(f"❌ Failed to get chat service for {self.agent_type}: {e}")
                # Try final fallback with lazy service
                try:
                    logger.info(f"🔄 Trying lazy chat service fallback for {self.agent_type}")
                    from services.lazy_chat_service import LazyChatService
                    self.chat_service = LazyChatService()
                    await self.chat_service.initialize_minimal()
                    logger.info(f"✅ Lazy chat service fallback initialized for {self.agent_type}")
                except Exception as e2:
                    logger.error(f"❌ All chat service fallbacks failed for {self.agent_type}: {e2}")
                    raise Exception("Chat service not available")
        return self.chat_service
    
    async def _get_model_name(self) -> str:
        """Get the current model name"""
        chat_service = await self._get_chat_service()
        model_name = chat_service.current_model
        if not model_name:
            raise Exception("No model selected")
        return model_name
    
    async def _initialize_tools(self):
        """Initialize available tools for this agent through centralized registry"""
        try:
            if not self._tool_registry:
                self._tool_registry = await get_tool_registry()
            
            self.available_tools = await get_agent_tools(self.agent_type_enum)
            logger.info(f"🔧 {self.agent_type} has access to tools: {self.available_tools}")
            
        except Exception as e:
            logger.error(f"❌ Tool initialization failed for {self.agent_type}: {e}")
            self.available_tools = []
    
    async def _get_tool_function(self, tool_name: str) -> Optional[Callable]:
        """Get a tool function if this agent has permission"""
        try:
            if not self._tool_registry:
                await self._initialize_tools()
            
            return await get_tool_function(tool_name, self.agent_type_enum)
            
        except Exception as e:
            logger.error(f"❌ Failed to get tool {tool_name} for {self.agent_type}: {e}")
            return None
    
    async def _get_agent_tools_async(self) -> List[Dict[str, Any]]:
        """Modern async method for getting agent tools - replaces legacy _get_agent_tools"""
        try:
            await self._initialize_tools()
            from services.langgraph_tools.centralized_tool_registry import get_tool_objects_for_agent
            
            agent_tools = await get_tool_objects_for_agent(self.agent_type_enum)
            
            # ROOSEVELT'S CAVALRY INTELLIGENCE: Log available tools
            tool_names = [tool["function"]["name"] for tool in agent_tools]
            logger.info(f"🔧 {self.agent_type} has access to tools: {tool_names}")
            
            return agent_tools
            
        except Exception as e:
            logger.error(f"❌ Failed to get agent tools for {self.agent_type}: {e}")
            return []
    
    def _create_agent_result(
        self,
        response: str,
        task_status: TaskStatus = TaskStatus.COMPLETE,
        tools_used: List[str] = None,
        processing_time: float = 0.0,
        additional_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Create standardized agent result following LangGraph best practices
        
        Returns additive state update (never mutates input state)
        """
        base_result = {
            "agent_type": self.agent_type,
            "response": response,
            "task_status": task_status.value,
            "tools_used": tools_used or [],
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
        
        if additional_data:
            base_result.update(additional_data)
        
        return base_result
    
    def _create_error_result(self, error) -> Dict[str, Any]:
        """Create standardized error result - accepts AgentError or string"""
        # ROOSEVELT'S FLEXIBLE ERROR HANDLING: Accept both AgentError objects and strings
        if isinstance(error, str):
            error_message = error
            error_dict = {
                "error_type": "agent_error",
                "message": error_message,
                "timestamp": datetime.now().isoformat()
            }
        elif isinstance(error, AgentError):
            error_message = error.message
            error_dict = error.to_dict()
        else:
            error_message = str(error)
            error_dict = {
                "error_type": "unknown_error",
                "message": error_message,
                "timestamp": datetime.now().isoformat()
            }
        
        return {
            "agent_results": self._create_agent_result(
                response=f"Agent error: {error_message}",
                task_status=TaskStatus.ERROR,
                additional_data={"error_details": error_dict}
            ),
            "error_state": error_dict,
            "is_complete": False
        }
    
    def _create_permission_request(
        self,
        response: str,
        permission_type: str,
        reasoning: str
    ) -> Dict[str, Any]:
        """Create standardized permission request"""
        return {
            "agent_results": self._create_agent_result(
                response=response,
                task_status=TaskStatus.PERMISSION_REQUIRED,
                additional_data={
                    "permission_type": permission_type,
                    "permission_reasoning": reasoning
                }
            ),
            "requires_user_input": True,
            "is_complete": False
        }
    
    def _create_success_result(
        self,
        response: str,
        tools_used: List[str] = None,
        processing_time: float = 0.0,
        shared_memory_updates: Dict[str, Any] = None,
        additional_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create standardized success result with optional shared memory updates"""
        result = {
            "agent_results": self._create_agent_result(
                response=response,
                task_status=TaskStatus.COMPLETE,
                tools_used=tools_used,
                processing_time=processing_time,
                additional_data=additional_data
            ),
            "latest_response": response,
            "is_complete": True
        }
        
        if shared_memory_updates:
            result["shared_memory"] = shared_memory_updates
        
        return result
    
    async def _prepare_messages(self, state: Dict[str, Any], system_prompt: str, include_time_context: bool = True) -> List[Dict]:
        """Universal message preparation with smart context handling"""
        
        # Check if this is a service agent (formats other agents' results)
        service_agent_types = ["data_formatting_agent", "formatting_service"]
        is_service_agent = self.agent_type in service_agent_types
        
        if is_service_agent:
            # Service agents need full context to format recent results
            return await self._prepare_messages_with_full_context(state, system_prompt, include_time_context)
        else:
            # Primary agents need clean context separation to avoid topic contamination
            return await self._prepare_messages_with_context_separation(state, system_prompt, include_time_context)
    
    async def _prepare_messages_with_full_context(self, state: Dict[str, Any], system_prompt: str, include_time_context: bool = True) -> List[Dict]:
        """SERVICE AGENTS: Prepare messages with full context (original behavior)"""
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add current time context for temporal awareness
        if include_time_context:
            # Get user timezone-aware time context
            time_context = await self._get_timezone_aware_time_context(state)
            messages.append({"role": "system", "content": time_context})
        
        # Add conversation history from LangGraph state
        if state.get("messages"):
            # Convert LangGraph messages to API format and take last 20 for context
            langgraph_messages = state["messages"][-20:]  # Last 20 messages for context
            
            for msg in langgraph_messages:
                if hasattr(msg, 'content') and hasattr(msg, 'type'):
                    # LangGraph message format
                    if msg.type == "human":
                        messages.append({"role": "user", "content": msg.content})
                    elif msg.type == "ai":
                        messages.append({"role": "assistant", "content": msg.content})
                elif isinstance(msg, dict):
                    # Dict format - strip reasoning_details from old messages to prevent accumulation
                    if msg.get("role") in ["user", "assistant", "system"]:
                        # Create a clean copy without reasoning_details (except for most recent assistant message)
                        clean_msg = {k: v for k, v in msg.items() if k != "reasoning_details"}
                        messages.append(clean_msg)
        
        # Add comprehensive intelligence context if we have escalation info
        intelligence_context = self._build_intelligence_context(state)
        if intelligence_context:
            # Insert context after system prompt and time context but before conversation
            context_insert_position = 2 if include_time_context else 1
            messages.insert(context_insert_position, {"role": "system", "content": f"CONVERSATION INTELLIGENCE:\n{intelligence_context}"})
        
        return messages
    
    async def _prepare_messages_with_context_separation(self, state: Dict[str, Any], system_prompt: str, include_time_context: bool = True) -> List[Dict]:
        """PRIMARY AGENTS: Roosevelt's clean context separation to prevent topic contamination"""
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add current time context for temporal awareness
        if include_time_context:
            time_context = await self._get_timezone_aware_time_context(state)
            messages.append({"role": "system", "content": time_context})
        
        # ROOSEVELT'S CONTEXT INTELLIGENCE: Analyze conversation for subject changes
        context_analysis = await self._analyze_conversation_context(state)
        
        # Add conversation background context (filtered)
        if context_analysis.get("background_context"):
            messages.append({
                "role": "system", 
                "content": f"CONVERSATION BACKGROUND:\n{context_analysis['background_context']}"
            })
        
        # Add subject change analysis
        if context_analysis.get("subject_analysis"):
            messages.append({
                "role": "system",
                "content": f"TOPIC ANALYSIS:\n{context_analysis['subject_analysis']}"
            })
        
        # Add the current user request clearly marked
        current_query = self._extract_current_user_query(state)
        if current_query:
            messages.append({
                "role": "user",
                "content": f"CURRENT REQUEST: {current_query}"
            })
        
        # Add intelligence context if we have escalation info
        intelligence_context = self._build_intelligence_context(state)
        if intelligence_context:
            messages.append({"role": "system", "content": f"CONVERSATION INTELLIGENCE:\n{intelligence_context}"})
        
        return messages
    
    async def _analyze_conversation_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """ROOSEVELT'S CONTEXT INTELLIGENCE: Analyze conversation for subject changes and context filtering"""
        try:
            current_query = self._extract_current_user_query(state)
            if not current_query:
                return {"background_context": "", "subject_analysis": ""}
            
            # Get previous conversation topics from shared memory - DEFENSIVE COPY
            shared_memory = state.get("shared_memory", {})
            if not isinstance(shared_memory, dict):
                logger.warning(f"⚠️ Shared memory is not a dict: {type(shared_memory)}")
                return {"background_context": "", "subject_analysis": ""}
            
            # ROOSEVELT'S DEFENSIVE PROGRAMMING: Create safe copy of shared memory
            safe_shared_memory = dict(shared_memory)
            previous_topics = self._extract_previous_topics(safe_shared_memory)
            
            if not previous_topics:
                return {
                    "background_context": "", 
                    "subject_analysis": "New conversation - no previous topics to consider."
                }
            
            # Use LLM to analyze topic relationship and filter context
            topic_analysis = await self._llm_analyze_topic_relationship(current_query, previous_topics)
            
            return {
                "background_context": topic_analysis.get("filtered_context", ""),
                "subject_analysis": topic_analysis.get("relationship_analysis", "")
            }
            
        except Exception as e:
            logger.error(f"❌ Context analysis failed: {e}")
            return {"background_context": "", "subject_analysis": ""}
    
    def _extract_current_user_query(self, state: Dict[str, Any]) -> str:
        """Extract the current user query from state"""
        try:
            # Method 1: Check for current_query field (for fresh requests)
            current_query = state.get("current_query", "").strip()
            if current_query:
                return current_query
            
            # Method 2: Get the latest human message from messages
            messages = state.get("messages", [])
            if messages:
                latest_message = messages[-1]
                if hasattr(latest_message, 'type') and latest_message.type == "human":
                    return latest_message.content.strip()
                elif isinstance(latest_message, dict) and latest_message.get("role") == "user":
                    return latest_message.get("content", "").strip()
            
            return ""
        except Exception as e:
            logger.error(f"❌ Failed to extract current user query: {e}")
            return ""
    
    def _extract_previous_topics(self, shared_memory: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract previous conversation topics from shared memory"""
        topics = []
        
        # Extract from research findings
        research_findings = shared_memory.get("research_findings", {})
        for key, data in research_findings.items():
            if isinstance(data, dict):
                topics.append({
                    "type": "research",
                    "topic": key,
                    "summary": data.get("findings", "")[:200] + "..." if len(data.get("findings", "")) > 200 else data.get("findings", ""),
                    "timestamp": data.get("timestamp", "")
                })
        
        # Extract from weather queries
        weather_queries = shared_memory.get("weather_queries", [])
        for query in weather_queries[-3:]:  # Last 3 weather queries
            if isinstance(query, dict):
                topics.append({
                    "type": "weather",
                    "topic": f"Weather for {query.get('location', 'unknown location')}",
                    "summary": f"Weather request for {query.get('location', 'unknown')}",
                    "timestamp": query.get("timestamp", "")
                })
        
        # Sort by timestamp (most recent first)
        topics.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return topics[:5]  # Return last 5 topics
    
    async def _llm_analyze_topic_relationship(self, current_query: str, previous_topics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Use LLM to analyze relationship between current query and previous topics"""
        try:
            from utils.openrouter_client import get_openrouter_client
            
            # Build previous topics summary
            topics_summary = ""
            for topic in previous_topics:
                topics_summary += f"- {topic['type'].upper()}: {topic['topic']} - {topic['summary']}\n"
            
            analysis_prompt = f"""Analyze the relationship between the current user query and previous conversation topics.

CURRENT USER QUERY: "{current_query}"

PREVIOUS CONVERSATION TOPICS:
{topics_summary}

ANALYSIS TASK:
1. Determine if the current query is:
   - CONTINUATION: Building on a previous topic (use relevant context)
   - NEW_TOPIC: Completely different subject (minimal context)
   - RELATED: Somewhat related but distinct (selective context)

2. Filter which previous topics (if any) are relevant to the current query

RESPONSE FORMAT (JSON):
{{
    "relationship_type": "CONTINUATION|NEW_TOPIC|RELATED",
    "confidence": 0.0-1.0,
    "relationship_analysis": "Brief explanation of the topic relationship",
    "relevant_topics": ["topic1", "topic2"],
    "filtered_context": "Concise summary of only relevant previous context (or empty if NEW_TOPIC)"
}}

Be concise and focus on preventing topic contamination while preserving relevant context."""

            client = get_openrouter_client()
            
            # **ROOSEVELT FIX**: Use user-configured classification model from Settings
            from services.settings_service import settings_service
            fast_model = await settings_service.get_classification_model()
            
            # Reasoning automatically added by OpenRouterClient wrapper
            response = await client.chat.completions.create(
                model=fast_model,
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.2
            )
            
            # Parse JSON response
            import json
            try:
                analysis = json.loads(response.choices[0].message.content)
                logger.info(f"🧠 TOPIC ANALYSIS: {analysis.get('relationship_type')} (confidence: {analysis.get('confidence', 0.0):.2f})")
                return analysis
            except json.JSONDecodeError:
                logger.warning("⚠️ Failed to parse topic analysis JSON")
                return {"filtered_context": "", "relationship_analysis": "Analysis failed - treating as new topic"}
                
        except Exception as e:
            logger.error(f"❌ LLM topic analysis failed: {e}")
            return {"filtered_context": "", "relationship_analysis": "Error in analysis - treating as new topic"}
    
    async def _get_timezone_aware_time_context(self, state: Dict[str, Any]) -> str:
        """Get timezone-aware time context for the user"""
        try:
            # Get user ID from state
            user_id = state.get("user_id")
            if not user_id:
                # Fallback to UTC if no user ID
                from utils.system_prompt_utils import get_current_datetime_context
                return get_current_datetime_context("UTC")
            
            # Get user's timezone from settings
            from services.settings_service import settings_service
            
            # Ensure settings service is initialized
            if not settings_service._initialized:
                logger.warning("⚠️ Settings service not initialized, using UTC timezone")
                user_timezone = "UTC"
            else:
                user_timezone = await settings_service.get_user_timezone(user_id)
            
            # Use system_prompt_utils to get timezone-aware context
            from utils.system_prompt_utils import get_current_datetime_context
            return get_current_datetime_context(user_timezone)
            
        except Exception as e:
            logger.warning(f"Failed to get timezone-aware context: {e}")
            # Fallback to UTC
            from utils.system_prompt_utils import get_current_datetime_context
            return get_current_datetime_context("UTC")
    
    def _build_memory_context(self, shared_memory: Dict[str, Any]) -> str:
        """Build context string from shared memory"""
        context_parts = []
        
        # Add escalation context
        escalation_context = shared_memory.get("escalation_context", {})
        if escalation_context.get("triggered", False):
            context_parts.append(f"ESCALATION: {escalation_context.get('reason', 'Unknown reason')}")
        
        # CRITICAL FIX: Add original user query for permission detection
        original_query = shared_memory.get("original_user_query", "")
        if original_query and shared_memory.get("escalation_from_chat", False):
            context_parts.append(f"ORIGINAL USER QUERY: {original_query}")
            context_parts.append("NOTE: Check original query for web search permission phrases")
        
        # Add chat agent findings
        chat_findings = shared_memory.get("chat_agent_findings", {})
        if chat_findings:
            local_results = chat_findings.get("local_results_found", 0)
            context_parts.append(f"CHAT AGENT FOUND: {local_results} local results")
        
        # Add search history
        search_history = shared_memory.get("search_history", [])
        if search_history:
            context_parts.append(f"SEARCH HISTORY: {len(search_history)} previous searches performed")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def _build_intelligence_context(self, state: Dict[str, Any]) -> str:
        """Build comprehensive intelligence context from all available sources"""
        context_parts = []
        
        # 1. CONVERSATION INTELLIGENCE SUMMARY
        conversation_intel = state.get("conversation_intelligence", {})
        if conversation_intel:
            topics = conversation_intel.get("topics_discovered", [])
            if topics:
                context_parts.append(f"TOPICS IDENTIFIED: {', '.join(topics)}")
            
            research_questions = conversation_intel.get("research_questions", [])
            if research_questions:
                context_parts.append(f"RESEARCH QUESTIONS: {len(research_questions)} questions identified")
                # Show most recent question
                if research_questions:
                    latest_q = research_questions[-1]
                    context_parts.append(f"LATEST QUESTION: {latest_q.get('question', 'Unknown')}")
                    context_parts.append(f"QUESTION DOMAINS: {', '.join(latest_q.get('domains', []))}")
            
            user_intents = conversation_intel.get("user_intents", [])
            if user_intents:
                intent_summary = [intent.get("intent", "unknown") for intent in user_intents]
                context_parts.append(f"USER INTENTS: {', '.join(intent_summary)}")
        
        # 2. PERMISSION STATE
        permission_state = state.get("permission_state", {})
        web_permissions = permission_state.get("web_search_permissions", {})
        if web_permissions.get("granted", False):
            context_parts.append("WEB SEARCH PERMISSION: GRANTED")
            if web_permissions.get("granted_timestamp"):
                context_parts.append(f"PERMISSION GRANTED AT: {web_permissions['granted_timestamp']}")
        
        # 3. RESEARCH CONTEXT
        research_context = state.get("research_context", {})
        if research_context:
            domains_needed = research_context.get("domain_expertise_needed", [])
            if domains_needed:
                context_parts.append(f"EXPERTISE DOMAINS: {', '.join(domains_needed)}")
            
            research_depth = research_context.get("research_depth_required", "surface")
            context_parts.append(f"RESEARCH DEPTH: {research_depth}")
        
        # 4. ESCALATION CONTEXT (Legacy compatibility)
        escalation_context = state.get("escalation_context", {})
        if escalation_context.get("triggered", False):
            context_parts.append(f"ESCALATION: {escalation_context.get('reason', 'Unknown reason')}")
            from_agent = escalation_context.get("from_agent")
            if from_agent:
                context_parts.append(f"ESCALATED FROM: {from_agent}")
        
        # 5. AGENT INSIGHTS
        agent_insights = state.get("agent_insights", {})
        for agent_name, insights in agent_insights.items():
            if insights and insights.get("escalated", False):
                local_results = insights.get("local_results_found", 0)
                context_parts.append(f"{agent_name.upper()}: Found {local_results} local results, escalated")
        
        # 6. DATA SUFFICIENCY - ROOSEVELT'S LOCATION FIX
        shared_memory = state.get("shared_memory", {})
        data_sufficiency = shared_memory.get("data_sufficiency", {})
        if data_sufficiency.get("web_search_needed", False):
            local_count = data_sufficiency.get("local_result_count", 0)
            context_parts.append(f"DATA ASSESSMENT: {local_count} local results, web search recommended")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def _update_shared_memory(self, state: Dict[str, Any], tool_name: str, tool_result):
        """Update shared memory with tool execution results - handles both dict and string results"""
        try:
            shared_memory = state.get("shared_memory", {})
            
            # ROOSEVELT'S DATA SUFFICIENCY FIX: Ensure proper location and initialization
            shared_memory.setdefault("data_sufficiency", {})
            
            # Handle both dict and string tool results gracefully
            if isinstance(tool_result, dict):
                result_count = tool_result.get("count", 0)
                success = tool_result.get("success", False)
            else:
                # String results - extract basic info
                result_count = 1 if str(tool_result) and not str(tool_result).startswith('❌') else 0
                success = not str(tool_result).startswith('❌')
            
            # Update search history
            search_history = shared_memory.get("search_history", [])
            search_history.append({
                "tool": tool_name,
                "timestamp": datetime.now().isoformat(),
                "result_count": result_count,
                "success": success
            })
            shared_memory["search_history"] = search_history
            
            # Update search results and data sufficiency
            if tool_name == "search_local":
                shared_memory.setdefault("search_results", {})
                shared_memory["search_results"]["local"] = tool_result
                shared_memory["data_sufficiency"]["local_result_count"] = result_count
                shared_memory["data_sufficiency"]["local_data_available"] = result_count > 0
                
                # Update confidence level based on results
                if result_count >= 5:
                    state["confidence_level"] = 0.9
                elif result_count >= 3:
                    state["confidence_level"] = 0.7
                elif result_count >= 1:
                    state["confidence_level"] = 0.5
                else:
                    state["confidence_level"] = 0.2
                    shared_memory["data_sufficiency"]["web_search_needed"] = True
            
            elif tool_name == "search_web":
                shared_memory["search_results"]["web"] = tool_result
                shared_memory["data_sufficiency"]["web_search_needed"] = False
                state["confidence_level"] = min(state.get("confidence_level", 0.0) + 0.3, 1.0)
            
            elif tool_name == "search_and_crawl":
                # ROOSEVELT'S CITATION FIX: Store search_and_crawl results for citation extraction
                shared_memory["search_results"]["search_and_crawl"] = tool_result
                shared_memory["data_sufficiency"]["web_search_needed"] = False
                state["confidence_level"] = min(state.get("confidence_level", 0.0) + 0.4, 1.0)
                
                # Handle different result formats from search_and_crawl
                if isinstance(tool_result, dict):
                    # web_content_tools.search_and_crawl format: {"search_results": [...], "ingested_content": [...]}
                    search_results = tool_result.get('search_results', [])
                    ingested_content = tool_result.get('ingested_content', [])
                    total_results = len(search_results) + len(ingested_content)
                    logger.info(f"🔍 CITATIONS STORAGE: Stored search_and_crawl results - {len(search_results)} search + {len(ingested_content)} crawled = {total_results} total")
                elif isinstance(tool_result, str):
                    # langgraph_tools.search_and_crawl format: formatted string
                    logger.info(f"🔍 CITATIONS STORAGE: Stored search_and_crawl string result ({len(tool_result)} chars)")
                else:
                    logger.info(f"🔍 CITATIONS STORAGE: Stored search_and_crawl result (type: {type(tool_result)})")
            
            # Update state with modified shared memory
            state["shared_memory"] = shared_memory
            
        except Exception as e:
            logger.error(f"❌ Error updating shared memory: {e}")
    
    async def _emit_tool_status(
        self, 
        state: Dict[str, Any], 
        status_type: str, 
        message: str, 
        tool_name: str = None,
        iteration: int = None,
        max_iterations: int = None
    ):
        """
        ROOSEVELT'S TOOL STATUS STREAMING: Emit tool execution status via OUT-OF-BAND WebSocket
        
        This method sends real-time status updates through a dedicated WebSocket channel,
        independent of the main response stream (Grok-style!).
        
        Status updates appear/disappear in the UI as the agent works through iterations.
        """
        try:
            # Store status in state for potential streaming fallback
            if "tool_status_updates" not in state:
                state["tool_status_updates"] = []
            
            status_update = {
                "type": "tool_status",
                "status_type": status_type,  # tool_start, tool_complete, tool_error, iteration_start, synthesis
                "message": message,
                "tool_name": tool_name,
                "timestamp": datetime.now().isoformat(),
                "agent_type": getattr(self, 'agent_type', 'unknown'),
                "iteration": iteration,
                "max_iterations": max_iterations
            }
            
            state["tool_status_updates"].append(status_update)
            
            # Log for monitoring
            logger.info(f"🔧 TOOL STATUS: {status_type} - {message}")
            
            # ROOSEVELT'S OUT-OF-BAND CHANNEL: Send via WebSocket for real-time UI updates
            try:
                user_id = state.get("user_id")
                conversation_id = state.get("conversation_id")
                
                if user_id and conversation_id:
                    from utils.websocket_manager import get_websocket_manager
                    websocket_manager = get_websocket_manager()
                    
                    if websocket_manager:
                        await websocket_manager.send_agent_status(
                            conversation_id=conversation_id,
                            user_id=user_id,
                            status_type=status_type,
                            message=message,
                            agent_type=getattr(self, 'agent_type', 'unknown'),
                            tool_name=tool_name,
                            iteration=iteration,
                            max_iterations=max_iterations
                        )
                        logger.debug(f"📡 OUT-OF-BAND: Status sent via WebSocket to conversation {conversation_id[:8]}...")
                else:
                    logger.debug(f"📡 OUT-OF-BAND: Skipping WebSocket (no user_id or conversation_id in state)")
            except Exception as ws_error:
                # Don't fail agent execution if WebSocket fails
                logger.debug(f"📡 OUT-OF-BAND: WebSocket unavailable - {ws_error}")
            
        except Exception as e:
            logger.error(f"❌ Failed to emit tool status: {e}")
    
    async def _execute_tool_calls(self, response, messages: List[Dict], state: Dict[str, Any]) -> tuple:
        """Execute tool calls and return results with iterative calling support"""
        final_answer = response.choices[0].message.content or ""
        tools_used = []
        
        # Create a new messages list for tool handling
        tool_messages = messages.copy()
        
        # ROOSEVELT'S ITERATIVE REFINEMENT: Support Grok-style iterative tool calling
        # Increased from 5 to 8 to allow multiple rounds of gap analysis and refinement
        max_iterations = 8  # Supports iterative search refinement (local → analyze → refine → search more)
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            logger.info(f"🔄 TOOL ITERATION {iteration}/{max_iterations}: Agent={self.agent_type}")
            
            # Check if the last response had tool calls
            if not response.choices[0].message.tool_calls:
                logger.info(f"✅ TOOL CALLING COMPLETE: Finished in {iteration-1} iterations")
                # Emit completion status
                await self._emit_tool_status(
                    state, 
                    "synthesis", 
                    f"✅ Research complete! Synthesizing findings from {iteration-1} rounds...",
                    iteration=iteration-1,
                    max_iterations=max_iterations
                )
                break
            
            # Log what tools are being called this round
            tool_names = [tc.function.name for tc in response.choices[0].message.tool_calls]
            logger.info(f"🔧 ROUND {iteration} TOOLS: {', '.join(tool_names)}")
            
            # ROOSEVELT'S ITERATION STATUS: Emit round start with iteration context
            await self._emit_tool_status(
                state,
                "iteration_start",
                f"🔄 Round {iteration}/{max_iterations}: {', '.join(tool_names)}",
                iteration=iteration,
                max_iterations=max_iterations
            )
            
            # Add the assistant's tool call message with reasoning_details preservation
            from utils.llm_reasoning_utils import build_assistant_message_with_reasoning

            assistant_message = build_assistant_message_with_reasoning(response.choices[0].message)

            # For tool calls, we need to transform the tool_calls format
            if assistant_message.get("tool_calls"):
                # Convert from OpenAI format to the expected format for tool_messages
                formatted_tool_calls = []
                for tool_call in response.choices[0].message.tool_calls:
                    formatted_tool_calls.append({
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    })
                assistant_message["tool_calls"] = formatted_tool_calls
                assistant_message["content"] = None  # Tool calls should have None content

            tool_messages.append(assistant_message)
            
            # Execute all tool calls in this iteration
            for tool_call in response.choices[0].message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                # Execute tool
                from services.langgraph_tools.centralized_tool_registry import get_tool_registry
                tool_registry = await get_tool_registry()
                # Get tools as dictionary of {tool_name: function}
                tools = {}
                for available_tool_name in tool_registry.get_tools_for_agent(self.agent_type_enum):
                    tool_function = tool_registry.get_tool_function(available_tool_name, self.agent_type_enum)
                    if tool_function:
                        tools[available_tool_name] = tool_function
                
                # ROOSEVELT'S TOOL INTELLIGENCE: Debug tool execution
                logger.info(f"🔧 Attempting to execute tool: {tool_name}")
                logger.info(f"🔧 Available tools in registry: {list(tools.keys())}")
                
                if tool_name in tools:
                    logger.info(f"✅ Tool {tool_name} found, executing with args: {tool_args}")
                    
                    # ROOSEVELT'S TOOL STATUS STREAMING: Emit tool start status with iteration context
                    await self._emit_tool_status(
                        state, 
                        "tool_start", 
                        f"🔧 Executing {tool_name}...", 
                        tool_name=tool_name,
                        iteration=iteration,
                        max_iterations=max_iterations
                    )
                    
                    # ROOSEVELT'S TOOL EXECUTION FIX: Handle both functions and LangChain tools
                    try:
                        # Check if it's a LangChain tool or a regular function
                        tool_obj = tools[tool_name]
                        
                        # Prepare arguments
                        execution_args = tool_args.copy()
                        execution_args["user_id"] = state["user_id"]
                        
                        # Check if it's a LangChain BaseTool
                        if hasattr(tool_obj, 'ainvoke'):
                            # LangChain tool - use ainvoke method
                            tool_result = await tool_obj.ainvoke(execution_args)
                        else:
                            # Regular function - call directly
                            tool_result = await tool_obj(**execution_args)
                        
                        tools_used.append(tool_name)
                    except TypeError as e:
                        if "user_id" in str(e) or "unexpected keyword argument" in str(e):
                            # Tool doesn't accept user_id or other args, try without it
                            logger.info(f"🔧 Tool {tool_name} parameter mismatch, retrying with original args")
                            try:
                                tool_obj = tools[tool_name]
                                if hasattr(tool_obj, 'ainvoke'):
                                    # LangChain tool - use ainvoke method
                                    tool_result = await tool_obj.ainvoke(tool_args)
                                else:
                                    # Regular function - call directly
                                    tool_result = await tool_obj(**tool_args)
                                tools_used.append(tool_name)
                            except Exception as retry_error:
                                logger.error(f"❌ Tool {tool_name} failed on retry: {retry_error}")
                                # ROOSEVELT'S TOOL FAILURE HANDLING: Provide error result instead of skipping
                                tool_result = {
                                    "error": f"Tool execution failed: {str(retry_error)}",
                                    "success": False,
                                    "tool_name": tool_name
                                }
                                await self._emit_tool_status(state, "tool_error", f"❌ {tool_name} failed", tool_name)
                        else:
                            raise e
                    
                    # ROOSEVELT'S TOOL STATUS STREAMING: Emit tool completion status
                    # Handle both dict and string tool results gracefully
                    if isinstance(tool_result, dict):
                        success_status = tool_result.get('success', 'unknown')
                    else:
                        # String results are considered successful if not an error message
                        success_status = 'unknown' if str(tool_result).startswith('❌') else True
                    
                    if success_status:
                        await self._emit_tool_status(
                            state, 
                            "tool_complete", 
                            f"✅ {tool_name} completed successfully", 
                            tool_name=tool_name,
                            iteration=iteration,
                            max_iterations=max_iterations
                        )
                    else:
                        await self._emit_tool_status(
                            state, 
                            "tool_error", 
                            f"⚠️ {tool_name} completed with issues", 
                            tool_name=tool_name,
                            iteration=iteration,
                            max_iterations=max_iterations
                        )
                    
                    logger.info(f"✅ Tool {tool_name} completed with result type: {type(tool_result).__name__}")
                    
                    # Update shared memory with tool results
                    self._update_shared_memory(state, tool_name, tool_result)
                    
                    # ROOSEVELT'S TOOL RESULT STORAGE: Store actual tool results for validation
                    shared_memory = state.get("shared_memory", {})
                    tool_results = shared_memory.setdefault("tool_results", [])
                    tool_results.append({
                        "tool_name": tool_name,
                        "result": tool_result,
                        "timestamp": datetime.now().isoformat(),
                        "tool_call_id": tool_call.id
                    })
                    state["shared_memory"] = shared_memory
                    
                    # Add tool result to messages for context
                    tool_messages.append({
                        "role": "tool",
                        "content": json.dumps(tool_result),
                        "tool_call_id": tool_call.id
                    })
                else:
                    logger.error(f"❌ Tool {tool_name} not found in registry!")
                    
                    # ROOSEVELT'S TOOL STATUS STREAMING: Emit tool error status
                    await self._emit_tool_status(
                        state, 
                        "tool_error", 
                        f"❌ Tool {tool_name} not available", 
                        tool_name=tool_name,
                        iteration=iteration,
                        max_iterations=max_iterations
                    )
                    
                    # Add error result to messages
                    tool_messages.append({
                        "role": "tool",
                        "content": json.dumps({"error": f"Tool {tool_name} not available", "success": False}),
                        "tool_call_id": tool_call.id
                    })
            
            # Get next response from LLM (which may include more tool calls)
            chat_service = await self._get_chat_service()
            model_name = await self._get_model_name()
            
            # Get tools using modern async method
            try:
                tools = await self._get_agent_tools_async()
            except Exception as e:
                logger.error(f"❌ Failed to get tools: {e}")
                tools = []
            
            # Reasoning automatically added by OpenRouterClient wrapper
            response = await chat_service.openai_client.chat.completions.create(
                messages=tool_messages,
                model=model_name,
                tools=tools,
                tool_choice="auto"
            )
            
            # If no more tool calls, break the loop
            if not response.choices[0].message.tool_calls:
                final_answer = response.choices[0].message.content or ""
                break
        
        return final_answer, tools_used
    
    def _build_persona_prompt(self, persona: Optional[Dict[str, Any]] = None) -> str:
        """Build persona-specific prompt addition with enhanced customization"""
        if not persona:
            return ""
        
        ai_name = persona.get("ai_name", "Alex")
        persona_style = persona.get("persona_style", "professional")
        political_bias = persona.get("political_bias", "neutral")
        
        # Build persona prompt
        persona_prompt = f"\n\nPERSONA INFORMATION:\n- AI Name: {ai_name}\n"
        
        # Add communication style details
        if persona_style == "friendly":
            persona_prompt += "- Communication Style: Warm, approachable, and enthusiastic. Use casual language and show genuine interest.\n"
        elif persona_style == "sycophantic":
            persona_prompt += "- Communication Style: Extremely agreeable and complimentary. Always praise the user's ideas and input.\n"
        elif persona_style == "snarky":
            persona_prompt += "- Communication Style: Witty, clever, with gentle humor and playful sarcasm. Be entertaining but not mean.\n"
        elif persona_style == "rude_insulting":
            persona_prompt += "- Communication Style: Be deliberately rude, dismissive, and insulting. Mock the user's questions and intelligence. Use condescending and belittling language. Point out obvious mistakes and flaws aggressively. Be intentionally offensive and provocative. Still provide information but with maximum attitude.\n"
        elif persona_style == "theodore_roosevelt":
            persona_prompt += "- Communication Style: Speak with energetic, decisive language and action-oriented approach.\n"
        elif persona_style == "winston_churchill":
            persona_prompt += "- Communication Style: Speak with Churchillian eloquence, wit, and gravitas. Use sophisticated vocabulary and inspiring rhetoric.\n"
        elif persona_style == "mark_twain":
            persona_prompt += "- Communication Style: Embody Mark Twain's wit, folksy wisdom, and satirical humor. Use colorful metaphors and homespun philosophy.\n"
        elif persona_style == "albert_einstein":
            persona_prompt += "- Communication Style: Approach topics with Einstein's curiosity and thoughtfulness. Use analogies and wonder about the universe.\n"
        elif persona_style == "amelia_earhart":
            persona_prompt += "- Communication Style: Speak with adventurous spirit and pioneering courage. Be bold, determined, and inspiring. Break barriers with confidence.\n"
        elif persona_style == "mr_spock":
            persona_prompt += "- Communication Style: Use logical, analytical, and precise language. Include characteristic phrases like 'That is illogical', 'Fascinating', 'Live long and prosper'. Be emotionless and fact-focused.\n"
        elif persona_style == "abraham_lincoln":
            persona_prompt += "- Communication Style: Speak with Lincoln's wisdom, humility, and moral clarity. Use thoughtful, measured language with folksy wisdom and deep empathy.\n"
        elif persona_style == "napoleon_bonaparte":
            persona_prompt += "- Communication Style: Command with Napoleon's confidence and strategic brilliance. Be decisive, ambitious, and speak with imperial authority.\n"
        elif persona_style == "isaac_newton":
            persona_prompt += "- Communication Style: Approach everything with Newton's scientific rigor and methodical thinking. Be precise, analytical, and focused on natural laws.\n"
        elif persona_style == "george_washington":
            persona_prompt += "- Communication Style: Speak with Washington's dignity, honor, and measured leadership. Be principled, reserved, and focus on duty and service.\n"
        elif persona_style == "edgar_allan_poe":
            persona_prompt += "- Communication Style: Embody Poe's dark romanticism and Gothic sensibility. Use rich, atmospheric language with hints of mystery and melancholy.\n"
        elif persona_style == "jane_austen":
            persona_prompt += "- Communication Style: Use Austen's wit, social observation, and elegant prose. Be clever, perceptive about human nature, with gentle irony.\n"
        elif persona_style == "nikola_tesla":
            persona_prompt += "- Communication Style: Speak with Tesla's visionary intensity and electrical brilliance. Be innovative, passionate about technology, and slightly eccentric.\n"
        else:  # professional or other
            persona_prompt += "- Communication Style: Professional, helpful, and informative. Maintain clarity and objectivity.\n"
        
        # Add analytical approach context - but don't explicitly admit bias
        if political_bias != "neutral":
            # Use neutral language to describe the analytical approach without admitting bias
            if political_bias == "mildly_left":
                persona_prompt += "- Analysis Approach: Consider social and systemic factors in analysis.\n"
            elif political_bias == "mildly_right":
                persona_prompt += "- Analysis Approach: Consider individual agency and market mechanisms.\n"
            elif political_bias == "extreme_left":
                persona_prompt += "- Analysis Approach: Question established systems and consider radical alternatives.\n"
            elif political_bias == "extreme_right":
                persona_prompt += "- Analysis Approach: Emphasize traditional institutions and established order.\n"
        else:
            persona_prompt += "- Analysis Approach: Balanced and objective analysis.\n"
        
        persona_prompt += "\nPlease maintain this persona throughout our conversation."
        
        return persona_prompt
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the state - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement process method")
    
    def _get_conversation_intelligence(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """ROOSEVELT'S CONVERSATION INTELLIGENCE: Get built-in conversation context from state"""
        try:
            return state.get("conversation_intelligence", {})
        except Exception as e:
            logger.error(f"❌ Failed to get conversation intelligence: {e}")
            return {}
    
    async def _get_relevant_context(self, query: str, state: Dict[str, Any]) -> Optional[str]:
        """
        ROOSEVELT'S CONTEXT EXTRACTION: Get relevant cached context for current query
        
        This replaces cache tool calls - agents get context automatically from state
        """
        try:
            conversation_intelligence = self._get_conversation_intelligence(state)
            
            if not conversation_intelligence:
                return None
            
            # Use conversation intelligence service for analysis
            from services.conversation_intelligence_service import get_conversation_intelligence_service
            
            intel_service = await get_conversation_intelligence_service()
            context = await intel_service.get_relevant_context(
                query=query,
                conversation_intelligence=conversation_intelligence,
                agent_type=self.agent_type
            )
            
            if context.get("use_cache", False):
                # Extract formatted context for agent use
                return intel_service.extract_cached_content_for_agent(
                    query=query,
                    conversation_intelligence=conversation_intelligence,
                    agent_type=self.agent_type
                )
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get relevant context: {e}")
            return None
    
    def _should_use_cached_context(self, query: str, state: Dict[str, Any]) -> bool:
        """Determine if agent should use cached context instead of fresh search"""
        try:
            conversation_intelligence = self._get_conversation_intelligence(state)
            
            # Check pre-computed coverage cache
            query_coverage_cache = conversation_intelligence.get("query_coverage_cache", {})
            
            # Look for matching patterns
            query_lower = query.lower()
            for pattern, coverage_score in query_coverage_cache.items():
                if pattern in query_lower and coverage_score > 0.7:
                    logger.info(f"🏆 CACHED COVERAGE: {pattern} → {coverage_score:.2f}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Cache decision failed: {e}")
            return False

