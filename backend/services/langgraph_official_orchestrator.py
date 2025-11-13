"""
LangGraph Official Orchestrator - Roosevelt's "Gold Standard" Implementation
Follows LangGraph's official HITL patterns: interrupt_before, external checkpointing
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Import agents and services
from services.langgraph_agents.clean_research_agent import CleanResearchAgent
from services.langgraph_agents.content_analysis_agent import ContentAnalysisAgent
from services.langgraph_agents.report_formatting_agent import ReportFormattingAgent
from services.langgraph_agents.chat_agent import ChatAgent
from services.langgraph_agents.permission_intelligence_agent import PermissionIntelligenceAgent
from services.langgraph_agents.rss_agent import RSSAgent
from services.langgraph_agents.data_formatting_agent import DataFormattingAgent
from services.langgraph_agents.simple_intent_agent import SimpleIntentAgent
from services.langgraph_enhanced_state import ConversationState
from services.langgraph_postgres_checkpointer import get_async_postgres_saver
from models.shared_memory_models import SharedMemory, validate_shared_memory, merge_shared_memory

logger = logging.getLogger(__name__)


class PermissionRequest(TypedDict):
    """Structure for permission requests in HITL checkpoints"""
    operation_type: Literal["web_search", "web_crawl", "data_modification"]
    query: str
    reasoning: str
    estimated_cost: Optional[str]
    safety_level: Literal["low", "medium", "high"]


class LangGraphOfficialOrchestrator:
    """
    Roosevelt's "Gold Standard" LangGraph Orchestrator
    Follows official LangGraph HITL patterns with interrupt_before
    """
    
    def __init__(self):
        self.graph = None
        self.postgres_checkpointer = None
        self.is_initialized = False
        
        # Initialize stateless agents
        self.clean_research_agent = CleanResearchAgent()
        self.report_formatting_agent = ReportFormattingAgent()
        self.chat_agent = ChatAgent()
        self.permission_intelligence_agent = PermissionIntelligenceAgent()
        self.rss_agent = RSSAgent()
        self.data_formatting_agent = DataFormattingAgent()  # ROOSEVELT'S TABLE SPECIALIST
        self.intent_classification_agent = SimpleIntentAgent()  # ROOSEVELT'S LEAN ROUTING COMMAND CENTER
        self.weather_agent = None  # ROOSEVELT'S METEOROLOGICAL INTELLIGENCE - Lazy loaded
        self.content_analysis_agent = ContentAnalysisAgent()
        from services.langgraph_agents.story_analysis_agent import StoryAnalysisAgent
        self.story_analysis_agent = StoryAnalysisAgent()
        from services.langgraph_agents.fiction_editing_agent import FictionEditingAgent
        self.fiction_editing_agent = FictionEditingAgent()
        from services.langgraph_agents.sysml_agent import SysMLAgent
        self.sysml_agent = SysMLAgent()
        from services.langgraph_agents.proofreading_agent import ProofreadingAgent
        self.proofreading_agent = ProofreadingAgent()
        from services.langgraph_agents.org_inbox_agent import OrgInboxAgent
        self.org_inbox_agent = OrgInboxAgent()
        # NEW: Org Project Agent
        from services.langgraph_agents.org_project_agent import OrgProjectAgent
        self.org_project_agent = OrgProjectAgent()
        from services.langgraph_agents.image_generation_agent import ImageGenerationAgent
        self.image_generation_agent = ImageGenerationAgent()
        from services.langgraph_agents.wargaming_agent import WargamingAgent
        self.wargaming_agent = WargamingAgent()
        # NEW: Rules editing agent
        from services.langgraph_agents.rules_editing_agent import RulesEditingAgent
        self.rules_editing_agent = RulesEditingAgent()
        # NEW: Podcast script agent
        from services.langgraph_agents.podcast_script_agent import PodcastScriptAgent
        self.podcast_script_agent = PodcastScriptAgent()
        # NEW: Substack agent
        from services.langgraph_agents.substack_agent import SubstackAgent
        self.substack_agent = SubstackAgent()
        # NEW: Character development agent
        from services.langgraph_agents.character_development_agent import CharacterDevelopmentAgent
        self.character_development_agent = CharacterDevelopmentAgent()
        # NEW: Outline editing agent
        from services.langgraph_agents.outline_editing_agent import OutlineEditingAgent
        self.outline_editing_agent = OutlineEditingAgent()
        # NEW: Site crawl agent (query-driven research)
        from services.langgraph_agents.site_crawl_agent import SiteCrawlAgent
        self.site_crawl_agent = SiteCrawlAgent()
        # NEW: Website crawler agent (storage-driven ingestion)
        from services.langgraph_agents.website_crawler_agent import WebsiteCrawlerAgent
        self.website_crawler_agent = WebsiteCrawlerAgent()
        # NEW: Entertainment agent
        from services.langgraph_agents.entertainment_agent import EntertainmentAgent
        self.entertainment_agent = EntertainmentAgent()
        
        # ROOSEVELT'S CLEAN RESEARCH AGENT - No complex initialization needed!
        self.use_clean_research = True
    
    async def initialize(self):
        """Initialize the LangGraph with official patterns"""
        try:
            logger.info("🚀 Initializing LangGraph Official Orchestrator with PostgreSQL persistence...")
            
            # Initialize PostgreSQL checkpointer with LangGraph-native pattern
            self.checkpointer = await get_async_postgres_saver()
            logger.info("✅ PostgreSQL checkpointer initialized")
            

            
            # Create state graph
            self.graph = StateGraph(ConversationState)
            
            # === CORE NODES ===
            self.graph.add_node("intent_classifier", self._intent_classifier_node)
            self.graph.add_node("research_agent", self._research_agent_node)
            self.graph.add_node("report_agent", self._report_formatting_agent_node)
            self.graph.add_node("chat_agent", self._chat_agent_node)
            self.graph.add_node("data_formatting_agent", self._data_formatting_agent_node)  # ROOSEVELT'S TABLE SPECIALIST
            self.graph.add_node("weather_agent", self._weather_agent_node)  # ROOSEVELT'S METEOROLOGICAL INTELLIGENCE
            self.graph.add_node("rss_agent", self._rss_agent_node)
            self.graph.add_node("rss_metadata_request", self._rss_metadata_request_node)
            self.graph.add_node("content_analysis_agent", self._content_analysis_agent_node)
            self.graph.add_node("story_analysis_agent", self._story_analysis_agent_node)
            self.graph.add_node("site_crawl_agent", self._site_crawl_agent_node)
            self.graph.add_node("website_crawler_agent", self._website_crawler_agent_node)
            self.graph.add_node("fact_checking_agent", self._fact_checking_agent_node)
            self.graph.add_node("fiction_editing_agent", self._fiction_editing_agent_node)
            self.graph.add_node("rules_editing_agent", self._rules_editing_agent_node)
            self.graph.add_node("character_development_agent", self._character_development_agent_node)
            self.graph.add_node("outline_editing_agent", self._outline_editing_agent_node)
            self.graph.add_node("sysml_agent", self._sysml_agent_node)
            self.graph.add_node("proofreading_agent", self._proofreading_agent_node)
            self.graph.add_node("podcast_script_agent", self._podcast_script_agent_node)
            self.graph.add_node("substack_agent", self._substack_agent_node)
            self.graph.add_node("org_inbox_agent", self._org_inbox_agent_node)
            self.graph.add_node("org_project_agent", self._org_project_agent_node)
            self.graph.add_node("image_generation_agent", self._image_generation_agent_node)
            self.graph.add_node("wargaming_agent", self._wargaming_agent_node)
            self.graph.add_node("messaging_agent", self._messaging_agent_node)  # BULLY! Messaging cavalry!
            self.graph.add_node("entertainment_agent", self._entertainment_agent_node)  # BULLY! Entertainment cavalry!
            
            # === HITL PERMISSION NODES ===
            self.graph.add_node("web_search_permission", self._web_search_permission_node)
            
            # === COLLABORATION NODES ===
            # ROOSEVELT'S NATURAL COLLABORATION: Removed collaboration suggestion and handler nodes
            # Let agents handle collaboration decisions with full conversation context
            
            # === UTILITY NODES ===
            self.graph.add_node("final_response", self._final_response_node)
            self.graph.add_node("update_metadata", self._update_metadata_node)
            # Combined node: run proofreading + analysis in parallel then synthesize
            self.graph.add_node("combined_proofread_and_analyze", self._combined_proofread_and_analyze_node)
            
            # === ENTRY POINT ===
            self.graph.set_entry_point("intent_classifier")
            
            # === CONDITIONAL ROUTING ===
            self.graph.add_conditional_edges(
                "intent_classifier",
                self._route_from_intent,
                {
                    "research_agent": "research_agent",
                    "site_crawl_agent": "site_crawl_agent",
                    "website_crawler_agent": "website_crawler_agent",
                    "chat_agent": "chat_agent",
                    "data_formatting_agent": "data_formatting_agent",  # ROOSEVELT'S TABLE SPECIALIST
                    "weather_agent": "weather_agent",  # ROOSEVELT'S METEOROLOGICAL INTELLIGENCE
                    "rss_agent": "rss_agent",
                    "org_inbox_agent": "org_inbox_agent",
                    "org_project_agent": "org_project_agent",
                    "content_analysis_agent": "content_analysis_agent",
                    "story_analysis_agent": "story_analysis_agent",
                    "fact_checking_agent": "fact_checking_agent",
                    "fiction_editing_agent": "fiction_editing_agent",
                    "rules_editing_agent": "rules_editing_agent",
                    "character_development_agent": "character_development_agent",
                    "outline_editing_agent": "outline_editing_agent",
                    "sysml_agent": "sysml_agent",
                    "proofreading_agent": "proofreading_agent",
                    "podcast_script_agent": "podcast_script_agent",
                    "substack_agent": "substack_agent",
                    "image_generation_agent": "image_generation_agent",
                    "wargaming_agent": "wargaming_agent",
                    "entertainment_agent": "entertainment_agent",
                    "combined_proofread_and_analyze": "combined_proofread_and_analyze",
                    "end": END
                }
            )
            
            self.graph.add_conditional_edges(
                "research_agent",
                self._route_from_research,
                {
                    "template_report": "report_agent",
                    "data_formatting_agent": "data_formatting_agent",  # ROOSEVELT'S SMART ROUTING
                    # ROOSEVELT'S NATURAL COLLABORATION: Removed collaboration routing
                    "final_response": "final_response",
                    "end": END
                }
            )
            
            # Report agent always goes to final response
            self.graph.add_edge("report_agent", "final_response")
            # Org inbox agent goes to final response
            self.graph.add_edge("org_inbox_agent", "final_response")
            # Org project agent goes to final response
            self.graph.add_edge("org_project_agent", "final_response")
            
            self.graph.add_conditional_edges(
                "chat_agent",
                self._route_from_chat,
                {
                    "final_response": "final_response"
                }
            )
            # Data formatting agent always goes to final response
            self.graph.add_edge("data_formatting_agent", "final_response")
            # Weather agent always goes to final response  
            self.graph.add_edge("weather_agent", "final_response")
            self.graph.add_edge("content_analysis_agent", "final_response")
            self.graph.add_edge("story_analysis_agent", "final_response")
            self.graph.add_edge("fact_checking_agent", "final_response")
            self.graph.add_edge("fiction_editing_agent", "final_response")
            self.graph.add_edge("sysml_agent", "final_response")
            self.graph.add_edge("proofreading_agent", "final_response")
            self.graph.add_edge("combined_proofread_and_analyze", "final_response")
            self.graph.add_edge("image_generation_agent", "final_response")
            self.graph.add_edge("wargaming_agent", "final_response")
            self.graph.add_edge("podcast_script_agent", "final_response")
            self.graph.add_edge("substack_agent", "final_response")
            # ROOSEVELT'S NATURAL COLLABORATION: Removed collaboration suggestion and handler edges
            # Let agents handle collaboration decisions with full conversation context
            
            # ROOSEVELT'S EDITING AGENTS WITH CLARIFICATION SUPPORT
            # These agents can request clarification mid-workflow
            self.graph.add_edge("rules_editing_agent", "final_response")
            self.graph.add_edge("character_development_agent", "final_response")
            self.graph.add_conditional_edges(
                "outline_editing_agent",
                self._route_from_outline,
                {
                    "final_response": "final_response",
                    "end": END
                }
            )
            
            self.graph.add_conditional_edges(
                "rss_agent",
                self._route_from_rss,
                {
                    "metadata_request": "rss_metadata_request",
                    "final_response": "final_response",
                    "end": END
                }
            )
            # REMOVED: web_search_permission routing - research agent now does comprehensive search directly
            self.graph.add_edge("final_response", "update_metadata")
            self.graph.add_edge("update_metadata", END)
            
            # === COMPILE WITH POSTGRESQL PERSISTENCE AND HITL BREAKPOINTS ===
            # ROOSEVELT'S STATIC HITL PATTERN: Using interrupt_before for proper HITL
            # **ROOSEVELT FIX**: recursion_limit removed from compile() - it's now specified in invoke/stream calls
            self.graph = self.graph.compile(
                checkpointer=self.checkpointer,
                interrupt_before=[
                    "rss_metadata_request" # RSS metadata request for HITL
                    # REMOVED: web_search_permission - research agent now does comprehensive search directly
                ]
            )
            
            self.is_initialized = True
            logger.info("✅ LangGraph Official Orchestrator initialized with gold standard patterns")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LangGraph Official Orchestrator: {e}")
            raise
    
    # === ROUTING FUNCTIONS (Pure Functions) ===
    
    def _route_from_intent(self, state: ConversationState) -> str:
        from services.orchestrator_routing import route_from_intent
        return route_from_intent(state)
    
    def _route_from_outline(self, state: ConversationState) -> str:
        """Route from outline editing agent - check if clarification needed."""
        from services.orchestrator_routing import route_from_outline
        return route_from_outline(state)
    
    def _route_from_research(self, state: ConversationState) -> str:
        from services.orchestrator_routing import route_from_research
        return route_from_research(state)
    
    def _route_from_permission(self, state: ConversationState) -> str:
        from services.orchestrator_routing import route_from_permission
        return route_from_permission(state)
    
    def _route_from_rss(self, state: ConversationState) -> str:
        from services.orchestrator_routing import route_from_rss
        return route_from_rss(state)
    
    # ROOSEVELT'S NATURAL COLLABORATION: Removed collaboration routing function
    # Let agents handle collaboration decisions with full conversation context
    
    def _route_from_chat(self, state: ConversationState) -> str:
        from services.orchestrator_routing import route_from_chat
        return route_from_chat(state)
    
    # === NODE IMPLEMENTATIONS ===
    
    async def _intent_classifier_node(self, state: ConversationState) -> Dict[str, Any]:
        from services.orchestrator_nodes import intent_classifier_node
        return await intent_classifier_node(self, state)
    
    async def _research_agent_node(self, state: ConversationState) -> Dict[str, Any]:
        from services.orchestrator_nodes import agent_node
        return await agent_node(self, state, "clean_research_agent", "latest_response")
    
    async def _report_formatting_agent_node(self, state: ConversationState) -> Dict[str, Any]:
        from services.orchestrator_nodes import agent_node
        return await agent_node(self, state, "report_formatting_agent", "latest_response")
    
    async def _chat_agent_node(self, state: ConversationState) -> Dict[str, Any]:
        from services.orchestrator_nodes import agent_node
        return await agent_node(self, state, "chat_agent", "chat_agent_response")
    
    async def _data_formatting_agent_node(self, state: ConversationState) -> Dict[str, Any]:
        from services.orchestrator_nodes import agent_node
        return await agent_node(self, state, "data_formatting_agent", "data_formatting_response")
    
    async def _content_analysis_agent_node(self, state: ConversationState) -> Dict[str, Any]:
        from services.orchestrator_nodes import agent_node
        return await agent_node(self, state, "content_analysis_agent", "latest_response")

    async def _story_analysis_agent_node(self, state: ConversationState) -> Dict[str, Any]:
        from services.orchestrator_nodes import agent_node
        return await agent_node(self, state, "story_analysis_agent", "latest_response")

    async def _site_crawl_agent_node(self, state: ConversationState) -> Dict[str, Any]:
        from services.orchestrator_nodes import agent_node
        return await agent_node(self, state, "site_crawl_agent", "latest_response")

    async def _website_crawler_agent_node(self, state: ConversationState) -> Dict[str, Any]:
        from services.orchestrator_nodes import agent_node
        return await agent_node(self, state, "website_crawler_agent", "latest_response")

    async def _fact_checking_agent_node(self, state: ConversationState) -> Dict[str, Any]:
        from services.orchestrator_nodes import agent_node
        return await agent_node(self, state, "fact_checking_agent", "latest_response")

    async def _fiction_editing_agent_node(self, state: ConversationState) -> Dict[str, Any]:
        from services.orchestrator_nodes import agent_node
        return await agent_node(self, state, "fiction_editing_agent", "latest_response")

    async def _rules_editing_agent_node(self, state: ConversationState) -> Dict[str, Any]:
        from services.orchestrator_nodes import agent_node
        return await agent_node(self, state, "rules_editing_agent", "latest_response")

    async def _character_development_agent_node(self, state: ConversationState) -> Dict[str, Any]:
        from services.orchestrator_nodes import agent_node
        return await agent_node(self, state, "character_development_agent", "latest_response")

    async def _outline_editing_agent_node(self, state: ConversationState) -> Dict[str, Any]:
        from services.orchestrator_nodes import agent_node
        return await agent_node(self, state, "outline_editing_agent", "latest_response")

    async def _sysml_agent_node(self, state: ConversationState) -> Dict[str, Any]:
        from services.orchestrator_nodes import agent_node
        return await agent_node(self, state, "sysml_agent", "latest_response")

    async def _proofreading_agent_node(self, state: ConversationState) -> Dict[str, Any]:
        from services.orchestrator_nodes import agent_node
        return await agent_node(self, state, "proofreading_agent", "latest_response")

    async def _podcast_script_agent_node(self, state: ConversationState) -> Dict[str, Any]:
        from services.orchestrator_nodes import agent_node
        return await agent_node(self, state, "podcast_script_agent", "latest_response")

    async def _substack_agent_node(self, state: ConversationState) -> Dict[str, Any]:
        from services.orchestrator_nodes import agent_node
        return await agent_node(self, state, "substack_agent", "latest_response")

    async def _combined_proofread_and_analyze_node(self, state: ConversationState) -> Dict[str, Any]:
        try:
            # ROOSEVELT'S PARALLEL EXECUTION: Run both agents with original state
            # This ensures both agents get the same user message without contamination
            from services.orchestrator_nodes import agent_node
            import asyncio
            
            # Create clean state copies for parallel execution
            proofreading_state = dict(state)
            analysis_state = dict(state)
            
            # Run both agents in parallel with original state
            proofreading_task = agent_node(self, proofreading_state, "proofreading_agent", "latest_response")
            analysis_task = agent_node(self, analysis_state, "content_analysis_agent", "latest_response")
            
            # Wait for both to complete
            s1, s2 = await asyncio.gather(proofreading_task, analysis_task)
            # Build combined chat message when both succeeded
            try:
                pr_sr = ((s1.get("agent_results", {}) or {}).get("structured_response", {}) or {})
                pr_count = len((pr_sr.get("corrections") or [])) if isinstance(pr_sr, dict) else 0
                pr_response = s1.get("latest_response", "") or ""
                analysis_text = s2.get("latest_response", "") or ""
                
                # Build comprehensive combined response
                combined_parts = ["## Combined: Proofreading + Analysis\n"]
                
                if pr_count > 0:
                    combined_parts.append(f"**Proofreading: {pr_count} correction(s) suggested.**\n")
                    combined_parts.append("### Proofreading Results:\n")
                    combined_parts.append(pr_response)
                    combined_parts.append("\n---\n")
                else:
                    combined_parts.append("**Proofreading: No corrections needed.**\n\n---\n")
                
                combined_parts.append("### Content Analysis:\n")
                combined_parts.append(analysis_text)
                
                combined_message = "\n".join(combined_parts)
                
                # Merge both agent results into final state
                final_state = {
                    **state,  # Preserve original state
                    "latest_response": combined_message,
                    "is_complete": True,
                    "agent_results": {
                        "proofreading": s1.get("agent_results", {}),
                        "analysis": s2.get("agent_results", {}),
                        "combined": True
                    }
                }
                
                # ROOSEVELT'S EDITOR INTEGRATION: Pass through editor operations from proofreading
                if s1.get("editor_operations"):
                    final_state["editor_operations"] = s1["editor_operations"]
                if s1.get("manuscript_edit"):
                    final_state["manuscript_edit"] = s1["manuscript_edit"]
                
                # Preserve messages from both agents for conversation history
                final_messages = list(state.get("messages", []))
                if s1.get("latest_response"):
                    from langchain_core.messages import AIMessage
                    final_messages.append(AIMessage(content=f"[Proofreading] {s1['latest_response']}"))
                if s2.get("latest_response"):
                    from langchain_core.messages import AIMessage
                    final_messages.append(AIMessage(content=f"[Analysis] {s2['latest_response']}"))
                final_messages.append(AIMessage(content=combined_message))
                final_state["messages"] = final_messages
                
                return final_state
            except Exception:
                pass
            return s2
        except Exception as e:
            return { **state, "latest_response": f"Combined proofreading+analysis failed: {e}", "is_complete": True }

    async def _image_generation_agent_node(self, state: ConversationState) -> Dict[str, Any]:
        from services.orchestrator_nodes import agent_node
        return await agent_node(self, state, "image_generation_agent", "latest_response")

    async def _wargaming_agent_node(self, state: ConversationState) -> Dict[str, Any]:
        from services.orchestrator_nodes import agent_node
        return await agent_node(self, state, "wargaming_agent", "latest_response")
    
    async def _messaging_agent_node(self, state: ConversationState) -> Dict[str, Any]:
        """BULLY! Messaging cavalry node"""
        from services.orchestrator_nodes import agent_node
        return await agent_node(self, state, "messaging_agent", "latest_response")
    
    async def _entertainment_agent_node(self, state: ConversationState) -> Dict[str, Any]:
        """BULLY! Entertainment cavalry node"""
        from services.orchestrator_nodes import agent_node
        return await agent_node(self, state, "entertainment_agent", "latest_response")
    
    async def _weather_agent_node(self, state: ConversationState) -> Dict[str, Any]:
        from services.orchestrator_nodes import weather_agent_node
        return await weather_agent_node(self, state)
    
    async def _rss_agent_node(self, state: ConversationState) -> Dict[str, Any]:
        from services.orchestrator_nodes import agent_node
        return await agent_node(self, state, "rss_agent", "rss_agent_response")
    
    async def _rss_metadata_request_node(self, state: ConversationState) -> Dict[str, Any]:
        from services.orchestrator_nodes import rss_metadata_request_node
        return await rss_metadata_request_node(self, state)
    
    async def _org_inbox_agent_node(self, state: ConversationState) -> Dict[str, Any]:
        from services.orchestrator_nodes import agent_node
        return await agent_node(self, state, "org_inbox_agent", "org_inbox_response")
    
    async def _org_project_agent_node(self, state: ConversationState) -> Dict[str, Any]:
        from services.orchestrator_nodes import agent_node
        return await agent_node(self, state, "org_project_agent", "org_project_response")
    
    async def _parse_rss_metadata_response(self, user_message: str, metadata_operations: List[Dict[str, Any]]) -> bool:
        """Parse user's metadata response for RSS operations"""
        try:
            user_message_lower = user_message.lower()
            
            # Check for common metadata response patterns
            metadata_indicators = [
                "title:", "category:", "as", "in category", 
                "call it", "name it", "put it in"
            ]
            
            has_metadata = any(indicator in user_message_lower for indicator in metadata_indicators)
            
            if has_metadata:
                logger.info("🛑 RSS METADATA REQUEST: Detected metadata in user response")
                return True
            
            # Check for simple confirmations
            confirmation_words = ["yes", "ok", "sure", "fine", "good", "use that"]
            is_confirmation = any(word in user_message_lower for word in confirmation_words)
            
            if is_confirmation:
                logger.info("🛑 RSS METADATA REQUEST: Detected confirmation in user response")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ RSS metadata response parsing error: {e}")
            return False
    
    def _build_rss_metadata_request(self, metadata_operations: List[Dict[str, Any]]) -> str:
        """Build metadata request message for RSS operations"""
        try:
            if not metadata_operations:
                return "No metadata required."
            
            request_parts = []
            
            for operation in metadata_operations:
                feed_url = operation.get("feed_url", "unknown")
                missing_metadata = operation.get("missing_metadata", [])
                suggested_title = operation.get("suggested_title", "RSS Feed")
                suggested_category = operation.get("suggested_category", "other")
                available_categories = operation.get("available_categories", [])
                
                request_parts.append(f"For the RSS feed at {feed_url}:")
                
                if "title" in missing_metadata:
                    request_parts.append(f"  • Please provide a title (suggested: '{suggested_title}')")
                
                if "category" in missing_metadata:
                    categories_str = ", ".join(available_categories)
                    request_parts.append(f"  • Please provide a category (suggested: '{suggested_category}', available: {categories_str})")
            
            return "\n".join(request_parts)
            
        except Exception as e:
            logger.error(f"❌ RSS metadata request building error: {e}")
            return "Please provide the required metadata for RSS feed creation."
    
    async def _web_search_permission_node(self, state: ConversationState) -> Dict[str, Any]:
        from services.orchestrator_nodes import web_search_permission_node
        return await web_search_permission_node(self, state)
    
    # ROOSEVELT'S NATURAL COLLABORATION: Removed collaboration suggestion and handler nodes
    # Let agents handle collaboration decisions with full conversation context
    
    async def _final_response_node(self, state: ConversationState) -> Dict[str, Any]:
        from services.orchestrator_nodes import final_response_node
        return await final_response_node(state)
    
    async def _update_metadata_node(self, state: ConversationState) -> Dict[str, Any]:
        """Update conversation metadata within LangGraph workflow for proper persistence"""
        try:
            from datetime import datetime
            
            # ROOSEVELT'S LANGGRAPH FIX: Return ONLY metadata updates, preserve existing state
            updates = {}
            
            # Get the first user message for title generation
            messages = state.get("messages", [])
            user_message = None
            for msg in messages:
                if hasattr(msg, 'type') and msg.type == "human":
                    user_message = msg.content
                    break
            
            # Always update the last activity timestamp
            updates["conversation_updated_at"] = datetime.now().isoformat()
            
            # Generate title if this is a new conversation (no title yet)
            if not state.get("conversation_title") and user_message and len(user_message.strip()) > 0:
                title = await self._generate_conversation_title(user_message)
                updates["conversation_title"] = title
                logger.info(f"✅ METADATA NODE: Generated conversation title: '{title}'")
                
                # Also initialize other metadata fields
                if not state.get("conversation_created_at"):
                    updates["conversation_created_at"] = datetime.now().isoformat()
                if not state.get("conversation_tags"):
                    updates["conversation_tags"] = []
                if not state.get("conversation_description"):
                    updates["conversation_description"] = None
                if not state.get("is_pinned"):
                    updates["is_pinned"] = False
                if not state.get("is_archived"):
                    updates["is_archived"] = False
            
            # Update conversation topic based on the query
            if user_message and len(user_message.strip()) > 0:
                # Simple topic extraction - could be enhanced with LLM
                topic = user_message[:50] + "..." if len(user_message) > 50 else user_message
                updates["conversation_topic"] = topic.strip()
            
            logger.info(f"✅ METADATA NODE: Returning metadata updates for LangGraph state preservation")
            logger.info(f"🎯 METADATA NODE: Update keys: {list(updates.keys())}")
            return updates
            
        except Exception as e:
            logger.warning(f"⚠️ METADATA NODE: Failed to update conversation metadata: {e}")
            # Don't fail the conversation for metadata issues - return empty updates
            return {}
    
        # === PERMISSION HANDLING (Only in Permission Node) ===
    
    # === CONVERSATION HISTORY LOADING ===
    
    # ROOSEVELT'S LANGGRAPH TRUST: Removed custom conversation loading
    # LangGraph's MemorySaver will handle conversation persistence automatically!
    
    # === UTILITY METHODS ===
    
    def _get_conversation_config(self, conversation_id: str, user_id: str, base_checkpoint_id: Optional[str] = None) -> Dict[str, Any]:
        """
        ROOSEVELT'S CONVERSATION ISOLATION: Get consistent LangGraph config
        Uses namespaced thread_id = "{user_id}:{conversation_id}" for strict per-user isolation.
        """
        from services.orchestrator_utils import normalize_thread_id, validate_thread_id
        thread_id = normalize_thread_id(user_id, conversation_id)
        validate_thread_id(user_id, thread_id)
        # **ROOSEVELT FIX**: recursion_limit now goes in config dict, not compile()
        # Generous limit of 50 prevents runaway graph traversal while allowing complex workflows
        config: Dict[str, Any] = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 50
        }
        # If a base checkpoint is provided, include it to branch from that point
        if base_checkpoint_id:
            config["configurable"]["checkpoint_id"] = base_checkpoint_id
        return config
    

    def _convert_to_agent_state(self, state: ConversationState, agent_name: str) -> Dict[str, Any]:
        """Convert orchestrator state to PURE agent format - only what agents need"""
        # ROOSEVELT'S PERMISSION PRESERVATION: Keep raw shared_memory to preserve permission flags
        raw_shared_memory = state.get("shared_memory", {})
        
        # CRITICAL FIX: Don't validate shared_memory as it strips permission fields!
        # Just use the raw shared_memory to preserve web_search_permission
        if isinstance(raw_shared_memory, dict):
            shared_memory = raw_shared_memory  # Use raw to preserve all permission fields
        else:
            shared_memory = {}
        
        logger.info(f"🔧 STATE CONVERSION: web_search_permission = {shared_memory.get('web_search_permission')}")
        
        # ROOSEVELT'S PURE AGENT PRINCIPLE: Only pass what agents need!
        # NOTE: user_id is needed for personalized tool access (user's documents in Qdrant)
        return {
            "messages": state["messages"],
            "user_id": state["user_id"],  # Needed for personalized tool access
            "current_query": self._get_latest_user_message(state),
            "persona": state.get("persona"),
            "active_agent": agent_name,
            "shared_memory": shared_memory,
            "conversation_topic": state.get("conversation_topic")
        }
    
    def _build_conversation_context_for_intent_classifier(self, state: ConversationState) -> Dict[str, Any]:
        from services.orchestrator_utils import build_conversation_context_for_intent_classifier
        return build_conversation_context_for_intent_classifier(self, state)
    
    def _analyze_active_agent_context(self, messages: List, shared_memory: Dict[str, Any]) -> Dict[str, Any]:
        from services.orchestrator_utils import analyze_active_agent_context
        return analyze_active_agent_context(messages, shared_memory)
    
    async def _get_enhanced_routing_decision(self, intent_result: Dict[str, Any], conversation_context: Dict[str, Any], state: ConversationState) -> Dict[str, Any]:
        """ROOSEVELT'S ENHANCED ROUTING: Combine intent classification with agent intelligence"""
        try:
            from services.agent_intelligence_network import get_agent_network
            
            # If a direct target_agent is provided, honor it
            direct_agent = intent_result.get("target_agent")
            if isinstance(direct_agent, str) and direct_agent.strip():
                logger.info(f"🎯 ENHANCED ROUTING: Direct agent specified → {direct_agent}")
                return {
                    "primary_agent": direct_agent,
                    "confidence": intent_result.get("confidence", 0.8),
                    "reasoning": "Direct routing from simple classifier",
                    "agent_suggestions": []
                }

            agent_network = get_agent_network()
            intent_type = intent_result.get("intent_type", "")
            
            # Get capable agents for this intent type
            logger.info(f"🎯 ENHANCED ROUTING: Checking agents for intent_type='{intent_type}' (type: {type(intent_type)})")
            capable_agents = agent_network.get_agents_by_intent_type(intent_type)
            
            if not capable_agents:
                logger.warning(f"⚠️ ENHANCED ROUTING FALLBACK: No agents found for intent type: {intent_type}")
                logger.warning(f"⚠️ Available agents: {[agent.agent_type for agent in agent_network.get_available_agents()]}")
                fallback_recommendation = intent_result.get("target_agent", "chat_agent")
                logger.warning(f"⚠️ Using fallback recommendation: {fallback_recommendation}")
                return {
                    "primary_agent": fallback_recommendation,
                    "confidence": intent_result.get("confidence", 0.5),
                    "reasoning": f"Fallback routing for {intent_type} - no agents found in intelligence network",
                    "agent_suggestions": []
                }
            
            # Calculate routing confidence for each capable agent
            agent_scores = []
            for agent_info in capable_agents:
                confidence = agent_network.calculate_agent_routing_confidence(
                    agent_info, intent_result, conversation_context
                )
                agent_scores.append({
                    "agent_type": agent_info.agent_type,
                    "confidence": confidence,
                    "reasoning": f"Intent match + context analysis for {agent_info.display_name}"
                })
            
            # Sort by confidence
            agent_scores.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Get the best agent
            best_agent = agent_scores[0] if agent_scores else None
            
            if best_agent:
                logger.info(f"🎯 ENHANCED ROUTING SUCCESS: {intent_type} → {best_agent['agent_type']} (confidence: {best_agent['confidence']:.2f})")
                logger.info(f"🎯 ENHANCED ROUTING: Found {len(capable_agents)} capable agents, selected best match")
                # ROOSEVELT'S AGENT SCORES: Format without nested f-strings to avoid syntax issues
                agent_score_list = [(agent['agent_type'], f"{agent['confidence']:.2f}") for agent in agent_scores[:3]]
                logger.info(f"🎯 ENHANCED ROUTING: Agent scores: {agent_score_list}")
                
                return {
                    "primary_agent": best_agent["agent_type"],
                    "confidence": best_agent["confidence"],
                    "reasoning": best_agent["reasoning"],
                    "agent_suggestions": agent_scores[:3],  # Top 3 suggestions
                    "intent_result": intent_result,
                    "context_analysis": conversation_context.get("active_agent_context", {})
                }
            else:
                # Fallback to original intent routing
                return {
                    "primary_agent": intent_result.get("target_agent", "chat_agent"),
                    "confidence": intent_result.get("confidence", 0.5),
                    "reasoning": "Fallback to intent classifier routing",
                    "agent_suggestions": []
                }
                
        except Exception as e:
            logger.error(f"❌ Enhanced routing decision failed: {e}")
            # Fallback to original intent routing
            return {
                "primary_agent": intent_result.get("target_agent", "chat_agent"),
                "confidence": intent_result.get("confidence", 0.5),
                "reasoning": f"Error in enhanced routing: {str(e)}",
                "agent_suggestions": []
            }
    
    def _get_latest_user_message(self, state: ConversationState) -> str:
        from services.orchestrator_utils import get_latest_user_message
        return get_latest_user_message(state)
    
    async def process_user_query(
        self,
        user_message: str,
        user_id: str,
        conversation_id: str,
        persona: Optional[Dict[str, Any]] = None,
        extra_shared_memory: Optional[Dict[str, Any]] = None,
        base_checkpoint_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process user query through official LangGraph HITL pattern with streaming
        """
        try:
            logger.info(f"🎯 OFFICIAL ORCHESTRATOR: Processing query through LangGraph with HITL streaming")
            
            # ROOSEVELT'S LANGGRAPH CHECKPOINT PATTERN: Check for existing checkpoint first!
            config = self._get_conversation_config(conversation_id, user_id, base_checkpoint_id)
            
            # Check if we're resuming from checkpoint (HITL pattern or continuing conversation)
            try:
                current_state = await self.graph.aget_state(config)
                if current_state.values and current_state.next:
                    # RESUME FROM CHECKPOINT - HITL scenario with interruption
                    logger.info(f"🔄 LANGGRAPH: Resuming from checkpoint at node(s): {current_state.next}")
                    logger.info(f"🔄 LANGGRAPH: Adding new message to checkpoint state")
                    
                    # Update the existing state with new message
                    await self.graph.aupdate_state(config, {"messages": current_state.values["messages"] + [HumanMessage(content=user_message)]})
                    # Optionally merge extra shared memory for this turn
                    try:
                        if extra_shared_memory:
                            existing_sm = current_state.values.get("shared_memory", {}) or {}
                            merged_sm = existing_sm.copy()
                            merged_sm.update(extra_shared_memory)
                            await self.graph.aupdate_state(config, {"shared_memory": merged_sm})
                            logger.info("🔧 Merged extra_shared_memory into checkpoint state")
                    except Exception as _:
                        logger.warning("⚠️ Failed to merge extra_shared_memory into checkpoint state")
                    
                    # Use None as input to resume from checkpoint
                    input_data = None
                elif current_state.values and current_state.values.get("messages"):
                    # ROOSEVELT'S PERMISSION DETECTION: Check for pending permission requests
                    shared_memory = current_state.values.get("shared_memory", {})
                    agent_results = current_state.values.get("agent_results", {})
                    
                    # More precise permission detection - avoid false positives
                    has_pending_permission = (
                        shared_memory.get("web_search_permission") == "pending" or
                        (agent_results.get("permission_needed") is True and 
                         agent_results.get("task_status") == "permission_required") or
                        (agent_results.get("structured_response", {}).get("task_status") == "permission_required")
                    )
                    
                    # ROOSEVELT'S TEMPLATE CONFIRMATION DETECTION: Check for pending template confirmations
                    has_pending_template_confirmation = (
                        agent_results.get("awaiting_confirmation") is True and
                        agent_results.get("suggested_template_id") is not None
                    )
                    
                    # Don't treat as permission if already granted
                    if shared_memory.get("web_search_permission") is True:
                        has_pending_permission = False
                    
                    logger.info(f"🔍 PERMISSION CHECK: has_pending_permission={has_pending_permission}, user_message='{user_message.lower().strip()}'")
                    
                    # Use Permission Intelligence Agent for permission analysis
                    if has_pending_permission:
                        logger.info(f"🧠 PERMISSION CHECK: Using AI to analyze user intent for: '{user_message}'")
                        temp_state = {
                            "messages": [HumanMessage(content=user_message)],
                            "current_query": user_message
                        }
                        permission_analysis_state = await self.permission_intelligence_agent.process(temp_state)
                        permission_results = permission_analysis_state.get("agent_results", {})
                        permission_data = permission_results.get("additional_data", {}).get("permission_analysis", {})
                        
                        if permission_data.get("permission_status") == "granted":
                            # ROOSEVELT'S HITL RESUMPTION: Use LangGraph's continue method for permission approval
                            logger.info(f"✅ PERMISSION GRANTED: {permission_data.get('reasoning', 'User approval detected')} - resuming LangGraph with existing state")
                            
                            # CRITICAL: Use LangGraph's built-in continue() method to resume from checkpoint
                            # This preserves the original query and shared memory
                            try:
                                logger.info(f"🔄 LANGGRAPH RESUME: Using continue() to resume interrupted conversation")
                                
                                # Initialize resumption tracking variables
                                final_state = None
                                response_state = None
                                interrupted = False
                                
                                # ROOSEVELT'S PERMISSION SETTING: PRESERVE existing shared memory and ADD permission flag
                                existing_shared_memory = current_state.values.get("shared_memory", {})
                                # CRITICAL: Preserve all existing shared memory (original query, expanded queries, etc.)
                                preserved_shared_memory = existing_shared_memory.copy()
                                preserved_shared_memory["web_search_permission"] = "granted"
                                
                                resume_input = {
                                    "messages": [HumanMessage(content=user_message)],
                                    "shared_memory": preserved_shared_memory
                                }
                                logger.info(f"🔑 PERMISSION FLAG: Setting web_search_permission=granted while preserving existing shared memory")
                                logger.info(f"📚 PRESERVED QUERIES: original_user_query={existing_shared_memory.get('original_user_query', 'N/A')[:50]}...")
                                logger.info(f"📚 PRESERVED EXPANSIONS: {len(existing_shared_memory.get('expanded_queries', {}).get('expanded_queries', []))} expanded queries")
                                
                                # Continue with user approval message and permission flag - LangGraph will restore context
                                async for event in self.graph.astream(resume_input, config):
                                    # Process resumption events the same way as normal streaming
                                    logger.info(f"🎯 LANGGRAPH RESUME EVENT: {list(event.keys()) if isinstance(event, dict) else type(event)}")
                                    
                                    if isinstance(event, dict):
                                        if "__interrupt__" in event:
                                            logger.info(f"🛑 LANGGRAPH INTERRUPTION DETECTED: HITL checkpoint reached")
                                            interrupted = True
                                            break
                                        
                                        # Update final_state from each node
                                        for node_name, node_result in event.items():
                                            if node_name != "__interrupt__" and isinstance(node_result, dict):
                                                logger.info(f"🎯 LANGGRAPH RESUME STATE: Updated final_state from node: {[node_name]}")
                                                final_state = node_result
                                                if node_name in ["final_response", "research_agent", "chat_agent"]:
                                                    response_state = node_result
                                
                                # Extract the final response
                                if final_state:
                                    logger.info(f"🎯 LANGGRAPH RESUME: Extracted state from resumption")
                                    latest_response = final_state.get("latest_response", final_state.get("agent_results", {}).get("response", ""))
                                    
                                    return {
                                        "response": latest_response,
                                        "is_complete": final_state.get("is_complete", not interrupted),
                                        "interrupted": interrupted,
                                        "permission_request": interrupted,
                                        "conversation_id": conversation_id
                                    }
                                else:
                                    logger.warning(f"⚠️ LANGGRAPH RESUME: No final state received")
                                    return {
                                        "response": "Permission granted, but failed to resume research.",
                                        "is_complete": False,
                                        "interrupted": False,
                                        "permission_request": False,
                                        "conversation_id": conversation_id
                                    }
                                    
                            except Exception as resume_error:
                                logger.error(f"❌ LANGGRAPH RESUME FAILED: {resume_error}")
                                # Fall back to treating as new conversation
                                input_data = {"messages": [HumanMessage(content=user_message)]}
                        else:
                            logger.info(f"❌ PERMISSION DENIED: {permission_data.get('reasoning', 'User did not approve')} - not resuming")
                            # Don't resume - treat as new conversation  
                            input_data = {"messages": [HumanMessage(content=user_message)]}
                        
                    elif has_pending_template_confirmation:
                        # Use Permission Intelligence Agent for template confirmation too
                        logger.info(f"🧠 TEMPLATE CONFIRMATION: Using AI to analyze user intent for: '{user_message}'")
                        temp_state = {
                            "messages": [HumanMessage(content=user_message)],
                            "current_query": user_message
                        }
                        permission_analysis_state = await self.permission_intelligence_agent.process(temp_state)
                        permission_results = permission_analysis_state.get("agent_results", {})
                        permission_data = permission_results.get("additional_data", {}).get("permission_analysis", {})
                        
                        if permission_data.get("permission_status") == "granted":
                            # TEMPLATE CONFIRMATION RESPONSE - Convert to template execution command
                            logger.info(f"🔄 LANGGRAPH: Detected template confirmation response, converting to execution command")
                        
                        suggested_template_id = agent_results.get("suggested_template_id")
                        template_execution_command = f"TEMPLATE_CONFIRMED:accept|{suggested_template_id}"
                        
                        # Update state with template execution command
                        updated_messages = current_state.values["messages"] + [HumanMessage(content=template_execution_command)]
                        
                        input_data = {
                            "user_id": user_id,
                            "conversation_id": conversation_id,
                            "messages": updated_messages,
                            "shared_memory": current_state.values.get("shared_memory", {}),
                            "persona": persona,
                            "current_query": template_execution_command
                        }
                        
                    else:
                        # CONTINUE CONVERSATION - has existing messages but no interruption
                        logger.info(f"🔄 LANGGRAPH: Continuing existing conversation with {len(current_state.values['messages'])} previous messages")
                        
                        # Load existing conversation state and add new message
                        existing_messages = current_state.values["messages"]
                        existing_shared_memory = current_state.values.get("shared_memory", {})
                        
                        # ROOSEVELT'S CLEAN ARCHITECTURE: Use existing shared memory for continuing conversations  
                        # LangGraph's proper thread isolation ensures conversations remain separate
                        shared_memory_to_use = existing_shared_memory
                        if extra_shared_memory and isinstance(extra_shared_memory, dict):
                            temp = shared_memory_to_use.copy() if isinstance(shared_memory_to_use, dict) else {}
                            temp.update(extra_shared_memory)
                            shared_memory_to_use = temp
                        
                        input_data = {
                            "user_id": user_id,
                            "conversation_id": conversation_id,
                            "messages": existing_messages + [HumanMessage(content=user_message)],
                            "shared_memory": shared_memory_to_use,
                            "persona": persona  # ROOSEVELT'S PERSONA FIX: Include persona in continued conversations
                        }
                else:
                    # FRESH START - LangGraph will automatically load from checkpoints if conversation exists
                    logger.info(f"🆕 LANGGRAPH: Starting fresh or resuming from LangGraph checkpoints")
                    
                    # ROOSEVELT'S FRESH CONVERSATION GUARANTEE: Always start with clean shared memory for new conversations
                    logger.info(f"🧹 ROOSEVELT'S FRESH CONVERSATION: Ensuring clean shared memory for conversation {conversation_id}")
                    
                    # Start with just the new user message - LangGraph will handle history automatically
                    # Start with clean shared memory and merge any extras
                    base_shared_memory: Dict[str, Any] = {}
                    if extra_shared_memory and isinstance(extra_shared_memory, dict):
                        base_shared_memory.update(extra_shared_memory)
                    input_data = {
                        "user_id": user_id,
                        "conversation_id": conversation_id,
                        "messages": [HumanMessage(content=user_message)],
                        "shared_memory": base_shared_memory,
                        "persona": persona  # ROOSEVELT'S PERSONA FIX: Include persona in fresh conversations
                    }
                

            except Exception as e:
                # ROOSEVELT'S SMART FALLBACK: Only create fresh conversation for truly new conversations
                logger.info(f"🎯 LANGGRAPH: Checkpoint loading failed for conversation {conversation_id}")
                if "aget_tuple" in str(e) or "get_next_version" in str(e) or "not found" in str(e).lower():
                    logger.debug(f"🔧 LangGraph checkpointer: New conversation (expected): {e}")
                    is_truly_new_conversation = True
                else:
                    logger.warning(f"⚠️ LANGGRAPH: Unexpected checkpoint loading error: {e}")
                    is_truly_new_conversation = False
                
                if is_truly_new_conversation:
                    # ROOSEVELT'S FRESH START: Only for genuinely new conversations
                    logger.info(f"🧹 ROOSEVELT'S FRESH START: Clean shared memory for new conversation {conversation_id}")
                    
                    base_shared_memory: Dict[str, Any] = {}
                    if extra_shared_memory and isinstance(extra_shared_memory, dict):
                        base_shared_memory.update(extra_shared_memory)
                    input_data = {
                        "user_id": user_id,
                        "conversation_id": conversation_id,
                        "messages": [HumanMessage(content=user_message)],
                        "shared_memory": base_shared_memory,
                        "persona": persona
                    }
                else:
                    # ROOSEVELT'S RECOVERY MODE: Try to preserve any existing state
                    logger.warning(f"🔄 LANGGRAPH RECOVERY: Checkpoint error but trying to preserve context")
                    
                    # Try to get state without checkpoint loading
                    try:
                        recovery_state = await self.graph.aget_state(config)
                        if recovery_state and recovery_state.values:
                            existing_shared_memory = recovery_state.values.get("shared_memory", {})
                            existing_messages = recovery_state.values.get("messages", [])
                            logger.info(f"🔄 RECOVERY: Found existing state with {len(existing_messages)} messages")
                            
                            temp_sm = existing_shared_memory if isinstance(existing_shared_memory, dict) else {}
                            if extra_shared_memory and isinstance(extra_shared_memory, dict):
                                temp_sm = temp_sm.copy()
                                temp_sm.update(extra_shared_memory)
                            input_data = {
                                "user_id": user_id,
                                "conversation_id": conversation_id,
                                "messages": existing_messages + [HumanMessage(content=user_message)],
                                "shared_memory": temp_sm,
                                "persona": persona
                            }
                        else:
                            # Last resort fallback
                            base_shared_memory: Dict[str, Any] = {}
                            if extra_shared_memory and isinstance(extra_shared_memory, dict):
                                base_shared_memory.update(extra_shared_memory)
                            input_data = {
                                "user_id": user_id,
                                "conversation_id": conversation_id,
                                "messages": [HumanMessage(content=user_message)],
                                "shared_memory": base_shared_memory,
                                "persona": persona
                            }
                    except:
                        # Last resort fallback
                        base_shared_memory: Dict[str, Any] = {}
                        if extra_shared_memory and isinstance(extra_shared_memory, dict):
                            base_shared_memory.update(extra_shared_memory)
                        input_data = {
                            "user_id": user_id,
                            "conversation_id": conversation_id,
                            "messages": [HumanMessage(content=user_message)],
                            "shared_memory": base_shared_memory,
                            "persona": persona
                        }
            
            # If editor_preference is present in extra_shared_memory, propagate to shared_memory
            try:
                if extra_shared_memory and isinstance(extra_shared_memory, dict):
                    pref = extra_shared_memory.get("editor_preference")
                    if pref:
                        sm = input_data.get("shared_memory", {}) or {}
                        sm["editor_preference"] = pref
                        input_data["shared_memory"] = sm
            except Exception:
                pass

            logger.info(f"🎯 LANGGRAPH: Starting streaming with thread_id: {conversation_id}")
            
            final_state = None
            response_state = None  # ROOSEVELT'S FIX: Preserve the actual response state
            interrupted = False
            async for event in self.graph.astream(input_data, config):
                logger.info(f"🎯 LANGGRAPH STREAM EVENT: {list(event.keys()) if isinstance(event, dict) else type(event)}")
                
                # Handle different event types
                if isinstance(event, dict):
                    if "__interrupt__" in event:
                        logger.info("🛑 LANGGRAPH INTERRUPTION DETECTED: HITL checkpoint reached")
                        interrupted = True
                        # DON'T overwrite final_state with interrupt event
                    else:
                        # Regular node execution event: {"node_name": state}
                        node_names = list(event.keys())
                        
                        # ROOSEVELT'S RESPONSE PRESERVATION: Keep response state separate from metadata
                        if any(node in ['final_response', 'chat_agent'] for node in node_names):
                            # This contains the actual response - preserve it!
                            response_state = event
                            logger.info(f"🎯 LANGGRAPH RESPONSE STATE: Preserved from node: {node_names}")
                        
                        # Always update final_state for other processing
                        final_state = event
                        logger.info(f"🎯 LANGGRAPH STATE: Updated final_state from node: {node_names}")
                elif isinstance(event, tuple) or isinstance(event, list):
                    # Some LangGraph events are tuples/lists - log but continue
                    logger.info(f"🎯 LANGGRAPH STREAM: Non-dict event {type(event)}: {event}")
                else:
                    logger.info(f"🎯 LANGGRAPH STREAM: Unknown event type {type(event)}: {event}")
            
            if final_state is None:
                logger.error("❌ No valid state events received from LangGraph stream")
                return {
                    "status": "error",
                    "response": "No response from orchestrator",
                    "conversation_id": conversation_id
                }
            
            # ROOSEVELT'S STREAMING FIX: Extract state from preserved response state or final state
            actual_state = None
            
            # First, try to use the preserved response state (contains actual responses)
            if response_state is not None:
                if isinstance(response_state, dict):
                    # Try direct access first (response state might be the state directly)
                    if "messages" in response_state and "latest_response" in response_state:
                        actual_state = response_state
                        logger.info("🎯 LANGGRAPH: Using response_state directly as actual_state")
                    else:
                        # Try extracting from nested event structure
                        for node_name, state_data in response_state.items():
                            if node_name != "__interrupt__" and isinstance(state_data, dict):
                                actual_state = state_data
                                logger.info(f"🎯 LANGGRAPH: Extracted state from response node: {node_name}")
                                break
            
            # Fallback to final_state if response_state didn't work
            if actual_state is None and isinstance(final_state, dict):
                # Try direct access first (final state might be the state directly)
                if "messages" in final_state and "latest_response" in final_state:
                    actual_state = final_state
                    logger.info("🎯 LANGGRAPH: Using final_state directly as actual_state (fallback)")
                else:
                    # Try extracting from nested event structure
                    for node_name, state_data in final_state.items():
                        if node_name != "__interrupt__" and isinstance(state_data, dict):
                            actual_state = state_data
                            logger.info(f"🎯 LANGGRAPH: Extracted state from final node: {node_name} (fallback)")
                            break
            
            if actual_state is None:
                logger.error("❌ Could not extract state from LangGraph event")
                logger.error(f"❌ Final state structure: {final_state}")
                return {
                    "status": "error", 
                    "response": "Could not extract state from orchestrator",
                    "conversation_id": conversation_id
                }
            
            logger.info(f"🎯 LANGGRAPH: Completed streaming with final state keys: {list(actual_state.keys())}")
            
            response = actual_state.get("latest_response", "") or actual_state.get("chat_agent_response", "")
            logger.info(f"🎯 RESPONSE EXTRACTION: latest_response='{actual_state.get('latest_response', '')[:50]}...', chat_agent_response='{actual_state.get('chat_agent_response', '')[:50]}...'")
            is_complete = actual_state.get("is_complete", False)
            
            # ROOSEVELT'S PERMISSION REQUEST EXTRACTION: Check both top-level and agent_results
            permission_request = actual_state.get("permission_request") or actual_state.get("agent_results", {}).get("permission_request")
            
            # If we have a permission request, use its message as the response
            if permission_request and isinstance(permission_request, dict) and "message" in permission_request:
                response = permission_request["message"]
                logger.info("🛡️ PERMISSION REQUEST: Using permission request message as response")
            
            logger.info(f"🎯 PROCESS_USER_QUERY: Final response: '{response[:100]}...' (length: {len(response)})")
            logger.info(f"🎯 PROCESS_USER_QUERY: Is complete: {is_complete}")
            logger.info(f"🎯 PROCESS_USER_QUERY: Interrupted: {interrupted}")
            logger.info(f"🎯 PROCESS_USER_QUERY: Permission request: {bool(permission_request)}")
            
            # ROOSEVELT'S METADATA MANAGEMENT: Metadata now handled within LangGraph workflow
            
            # ROOSEVELT'S PURE LANGGRAPH: No dual persistence - LangGraph AsyncPostgresSaver handles ALL persistence
            logger.info(f"✅ LangGraph AsyncPostgresSaver automatically persisted conversation {conversation_id}")
            

            
            # Determine status based on interruption and completion
            if interrupted or permission_request:
                status = "interrupted"  # HITL interruption detected
            elif is_complete:
                status = "complete"
            else:
                status = "processing"  # Still running
            
            return {
                "status": status,
                "response": response,
                "conversation_id": conversation_id,
                "interrupted": interrupted,  # ROOSEVELT'S FIX: Include interrupted flag
                "permission_request": permission_request,
                "final_state": actual_state
            }
            
        except Exception as e:
            logger.error(f"❌ Official orchestrator error: {e}")
            
            # ROOSEVELT'S CONNECTION RECOVERY: Handle PostgreSQL connection issues
            if "connection is closed" in str(e).lower() or "connection" in str(e).lower():
                logger.warning("🔄 PostgreSQL connection issue detected, attempting recovery...")
                try:
                    # Reset the checkpointer and try again (once)
                    from services.langgraph_postgres_checkpointer import reset_postgres_checkpointer
                    await reset_postgres_checkpointer()
                    
                    # Re-initialize this orchestrator instance
                    await self.initialize()
                    
                    # Retry the operation once
                    logger.info("🔄 Retrying orchestrator operation with fresh connection")
                    return await self._retry_with_fresh_connection(user_message, user_id, conversation_id, persona)
                    
                except Exception as retry_error:
                    logger.error(f"❌ Connection recovery failed: {retry_error}")
                    return {
                        "status": "error",
                        "response": f"Database connection error. Please try again. ({str(e)})",
                        "conversation_id": conversation_id
                    }
            
            return {
                "status": "error",
                "response": f"Orchestrator error: {str(e)}",
                "conversation_id": conversation_id
            }
    
    async def _retry_with_fresh_connection(
        self,
        user_message: str,
        user_id: str,
        conversation_id: str,
        persona: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Retry operation with a fresh database connection"""
        try:
            logger.info("🔄 RETRY: Starting fresh operation with new connection")
            
            # Build config for fresh attempt
            config = self._get_conversation_config(conversation_id, user_id)
            
            # Get current state with fresh checkpointer
            current_state = await self.graph.aget_state(config)
            
            # Determine input based on current state
            if current_state.values and current_state.values.get("messages"):
                # Continue existing conversation
                existing_messages = current_state.values["messages"]
                existing_shared_memory = current_state.values.get("shared_memory", {})
                
                input_data = {
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "messages": existing_messages + [HumanMessage(content=user_message)],
                    "shared_memory": existing_shared_memory,
                    "persona": persona
                }
            else:
                # Fresh start
                input_data = {
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "messages": [HumanMessage(content=user_message)],
                    "shared_memory": {},
                    "persona": persona
                }
            
            # Execute the graph with fresh connection
            events = []
            async for event in self.graph.astream(input_data, config, stream_mode="values"):
                events.append(event)
            
            # Process final result
            final_state = events[-1] if events else {}
            response = final_state.get("latest_response", "Connection recovered successfully")
            is_complete = final_state.get("is_complete", True)
            
            logger.info("✅ RETRY: Operation completed successfully with fresh connection")
            
            return {
                "status": "complete" if is_complete else "processing",
                "response": response,
                "conversation_id": conversation_id,
                "final_state": final_state
            }
            
        except Exception as e:
            logger.error(f"❌ RETRY: Failed even with fresh connection: {e}")
            return {
                "status": "error",
                "response": f"Connection retry failed: {str(e)}",
                "conversation_id": conversation_id
            }
    
    async def _update_conversation_metadata(self, state: Dict[str, Any], user_message: str, is_complete: bool) -> None:
        from services.orchestrator_utils import update_conversation_metadata
        await update_conversation_metadata(self, state, user_message, is_complete)
    
    async def _generate_conversation_title(self, user_message: str) -> str:
        from services.orchestrator_utils import generate_conversation_title
        return await generate_conversation_title(user_message)
    
    def get_smart_intent_classifier(self):
        """Deprecated: SmartIntentClassifier is no longer used."""
        return None
    
    async def continue_conversation(self, query: str, conversation_id: str, user_id: str) -> Dict[str, Any]:
        """Continue an existing conversation with a new message"""
        try:
            logger.info(f"🔄 Continuing conversation {conversation_id} with: {query[:50]}...")
            
            # Use the same process_user_query method for consistency
            return await self.process_user_query(query, conversation_id, user_id)
            
        except Exception as e:
            logger.error(f"❌ Continue conversation failed: {e}")
            raise
    
    async def get_conversation_state(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get the current state of a conversation from checkpoints"""
        try:
            if not self.checkpointer:
                await self.initialize()
            
            # Get latest checkpoint for conversation using LangGraph's built-in method  
            # ROOSEVELT'S SIMPLIFIED CONFIG: Use standard thread_id pattern for state access
            # Note: recursion_limit not needed for get_state (only for invoke/stream)
            config = {"configurable": {"thread_id": conversation_id}}
            
            # Use the graph's get_state method which properly handles the checkpointer
            current_state = await self.graph.aget_state(config)
            if current_state and current_state.values:
                return current_state.values
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to get conversation state: {e}")
            return None


# Global instance
_official_orchestrator_instance: Optional[LangGraphOfficialOrchestrator] = None

async def get_official_orchestrator(enable_structured: bool = True) -> LangGraphOfficialOrchestrator:
    """
    Singleton pattern for the LangGraph Official Orchestrator
    ROOSEVELT'S STRUCTURED OUTPUT INTEGRATION
    """
    global _official_orchestrator_instance
    if _official_orchestrator_instance is None:
        _official_orchestrator_instance = LangGraphOfficialOrchestrator()
        await _official_orchestrator_instance.initialize()
        logger.info(f"🎯 ORCHESTRATOR: Initialized with structured research {'ENABLED' if enable_structured else 'DISABLED'}")
    
    # Clean research agent is always enabled - no configuration needed!
    return _official_orchestrator_instance
