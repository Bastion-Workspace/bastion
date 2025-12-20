"""
Agent Intelligence Network - Roosevelt's "Inter-Agent Command Center"
Provides agent discovery, capability mapping, and collaboration coordination
"""

import logging
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class AgentCapability(str, Enum):
    """Core capabilities that agents can provide"""
    WEATHER_INFORMATION = "weather_information"
    RESEARCH_INTELLIGENCE = "research_intelligence"
    DATA_FORMATTING = "data_formatting"
    DOCUMENT_ANALYSIS = "document_analysis"
    WEB_SEARCH = "web_search"
    LOCAL_SEARCH = "local_search"
    MATHEMATICAL_COMPUTATION = "mathematical_computation"
    CONVERSATION = "conversation"
    RSS_MANAGEMENT = "rss_management"
    CODE_GENERATION = "code_generation"
    ORG_INBOX_MANAGEMENT = "org_inbox_management"
    PIPELINE_DESIGN = "pipeline_design"
    EMAIL_SENDING = "email_sending"



class CollaborationPermission(str, Enum):
    """ROOSEVELT'S COLLABORATION PERMISSION TAXONOMY"""
    AUTO_USE = "auto_use"              # Execute automatically when beneficial (e.g., Data Formatting)
    SUGGEST_ONLY = "suggest_only"      # Suggest to user and wait for approval (e.g., Weather, Research)
    HITL_REQUIRED = "hitl_required"    # Always require explicit human permission (e.g., Web search)
    NEVER_AUTO = "never_auto"          # Never suggest or auto-execute (e.g., Admin functions)


@dataclass
class AgentInfo:
    """Information about an available agent"""
    agent_type: str
    display_name: str
    description: str
    capabilities: List[AgentCapability]
    specialties: List[str]
    handoff_triggers: List[str]  # Scenarios where this agent should be suggested
    collaboration_permission: CollaborationPermission  # How this agent can be invoked by others
    

class AgentIntelligenceNetwork:
    """
    Roosevelt's Agent Command Center
    Central registry for agent discovery and collaboration coordination
    """
    
    def __init__(self):
        self._agent_registry: Dict[str, AgentInfo] = {}
        self._collaboration_patterns: Dict[str, List[str]] = {}
        self._initialize_agent_registry()
    
    def _initialize_agent_registry(self):
        """Initialize the registry with all available agents and their capabilities"""
        
        # Weather Agent
        self._agent_registry["weather_agent"] = AgentInfo(
            agent_type="weather_agent",
            display_name="Weather Intelligence Agent",
            description="Provides current conditions, forecasts, and weather-based recommendations",
            capabilities=[AgentCapability.WEATHER_INFORMATION],
            specialties=[
                "Current weather conditions",
                "Weather forecasts (1-5 days)",
                "Activity recommendations based on weather",
                "Travel weather planning",
                "Severe weather alerts"
            ],
            handoff_triggers=[
                "location-based research results",
                "travel planning queries",
                "outdoor activity research",
                "event planning with venues",
                "agricultural/seasonal queries"
            ],
            collaboration_permission=CollaborationPermission.SUGGEST_ONLY  # Weather requires user approval
        )
        
        # Research Agent
        self._agent_registry["research_agent"] = AgentInfo(
            agent_type="research_agent", 
            display_name="Research Intelligence Agent",
            description="Comprehensive research using local documents and web sources",
            capabilities=[
                AgentCapability.RESEARCH_INTELLIGENCE,
                AgentCapability.WEB_SEARCH,
                AgentCapability.LOCAL_SEARCH,
                AgentCapability.DOCUMENT_ANALYSIS
            ],
            specialties=[
                "Factual information research",
                "Document analysis and summarization", 
                "Web search and content extraction",
                "Historical and current event research",
                "Scientific and technical research",
                "Travel and location information",
                "Safety protocols and procedures"
            ],
            handoff_triggers=[
                "travel weather requests",
                "outdoor activity weather", 
                "event weather planning",
                "severe weather alerts",
                "location-based queries",
                "activity safety concerns"
            ],
            collaboration_permission=CollaborationPermission.SUGGEST_ONLY  # Research requires user approval
        )
        
        # Data Formatting Agent
        self._agent_registry["data_formatting_agent"] = AgentInfo(
            agent_type="data_formatting_agent",
            display_name="Data Formatting Specialist", 
            description="Transforms research data into organized tables, charts, and structured formats",
            capabilities=[AgentCapability.DATA_FORMATTING],
            specialties=[
                "Markdown table creation",
                "Timeline and chronological formatting",
                "Visual timeline creation",
                "Data organization and structuring",
                "Comparison tables",
                "Summary formatting",
                "Report structuring"
            ],
            handoff_triggers=[
                "research results with multiple data points",
                "timeline creation requests",
                "chronological organization needs",
                "comparison queries",
                "list organization requests",
                "summary table needs"
            ],
            collaboration_permission=CollaborationPermission.AUTO_USE  # BULLY! Auto-execute when beneficial
        )
        
        # Chat Agent
        self._agent_registry["chat_agent"] = AgentInfo(
            agent_type="chat_agent",
            display_name="Conversation Agent",
            description="Handles general conversation and simple queries",
            capabilities=[AgentCapability.CONVERSATION],
            specialties=[
                "General conversation",
                "Simple questions",
                "Clarification requests",
                "Casual interaction"
            ],
            handoff_triggers=[],  # Chat agent typically doesn't initiate handoffs
            collaboration_permission=CollaborationPermission.SUGGEST_ONLY  # Chat requires approval for handoffs
        )
        
        # RSS Agent  
        self._agent_registry["rss_agent"] = AgentInfo(
            agent_type="rss_agent",
            display_name="RSS News Agent",
            description="Manages RSS feeds and news updates",
            capabilities=[AgentCapability.RSS_MANAGEMENT],
            specialties=[
                "RSS feed management",
                "News updates",
                "Feed summaries"
            ],
            handoff_triggers=[],
            collaboration_permission=CollaborationPermission.SUGGEST_ONLY  # RSS requires user approval
        )

        # ORG_INBOX_AGENT removed - migrated to llm-orchestrator gRPC service
        # ORG_PROJECT_AGENT removed - migrated to llm-orchestrator gRPC service

        # Pipeline Designer Agent
        self._agent_registry["pipeline_agent"] = AgentInfo(
            agent_type="pipeline_agent",
            display_name="Pipeline Designer Agent",
            description="Design-time pipeline modeling for AWS Glue/Lambda ETL",
            capabilities=[AgentCapability.PIPELINE_DESIGN],
            specialties=[
                "AWS Glue jobs and crawlers",
                "Lambda transforms",
                "S3 data flows",
                "Athena query sinks"
            ],
            handoff_triggers=[],
            collaboration_permission=CollaborationPermission.SUGGEST_ONLY
        )

        # EMAIL_AGENT removed - not used, functionality not migrated
        # self._agent_registry["email_agent"] = AgentInfo(
        #     agent_type="email_agent",
        #     display_name="Email Agent",
        #     description="Drafts and sends emails with user approval and conversation context",
        #     capabilities=[AgentCapability.EMAIL_SENDING],
        #     specialties=[
        #         "Email drafting",
        #         "Context-aware email composition",
        #         "Professional email formatting",
        #         "Email sending with approval"
        #     ],
        #     handoff_triggers=[
        #         "send email",
        #         "email to",
        #         "compose email",
        #         "draft email"
        #     ],
        #     collaboration_permission=CollaborationPermission.SUGGEST_ONLY
        # )
        
        # Initialize collaboration patterns
        self._initialize_collaboration_patterns()
    
    def _initialize_collaboration_patterns(self):
        """Define common collaboration patterns between agents"""
        
        # Weather → Research patterns
        self._collaboration_patterns["weather_to_research"] = [
            "travel_planning",
            "outdoor_activities", 
            "event_planning",
            "severe_weather_response",
            "agricultural_planning"
        ]
        
        # Research → Weather patterns
        self._collaboration_patterns["research_to_weather"] = [
            "location_research",
            "travel_destinations",
            "outdoor_venues",
            "event_locations",
            "seasonal_activities"
        ]
        
        # Research → Data Formatting patterns
        self._collaboration_patterns["research_to_formatting"] = [
            "multiple_data_sources",
            "comparison_research",
            "statistical_data",
            "list_organization"
        ]
    
    def get_available_agents(self) -> List[AgentInfo]:
        """Get list of all available agents"""
        return list(self._agent_registry.values())
    
    def get_agent_info(self, agent_type: str) -> Optional[AgentInfo]:
        """Get information about a specific agent"""
        return self._agent_registry.get(agent_type)
    
    def get_agents_by_capability(self, capability: AgentCapability) -> List[AgentInfo]:
        """Find agents that have a specific capability"""
        return [
            agent for agent in self._agent_registry.values()
            if capability in agent.capabilities
        ]
    
    def get_agents_by_intent_type(self, intent_type: str) -> List[AgentInfo]:
        """ROOSEVELT'S INTENT-TO-AGENT MAPPING: Get agents that can handle specific intent types"""
        # Handle both enum and string intent types - ROBUST CONVERSION
        if hasattr(intent_type, 'value'):
            # It's an enum, use the value
            intent_str = intent_type.value.upper()
        else:
            # It's already a string, normalize it
            intent_str = str(intent_type).replace("IntentType.", "").upper()
        
        intent_to_capability_mapping = {
            "WEATHER": [AgentCapability.WEATHER_INFORMATION],
            "RESEARCH": [AgentCapability.RESEARCH_INTELLIGENCE, AgentCapability.WEB_SEARCH, AgentCapability.DOCUMENT_ANALYSIS],
            "DATA_FORMATTING": [AgentCapability.DATA_FORMATTING],
            "CALCULATE": [AgentCapability.MATHEMATICAL_COMPUTATION],
            "CODING": [AgentCapability.CODE_GENERATION],
            "RSS": [AgentCapability.RSS_MANAGEMENT],
            "CHAT": [AgentCapability.CONVERSATION],
            "ORG_INBOX": [AgentCapability.ORG_INBOX_MANAGEMENT],
            "MAPPING": [AgentCapability.RESEARCH_INTELLIGENCE],  # Future mapping agent would have its own capability
            "COLLABORATION_RESPONSE": [AgentCapability.CONVERSATION],  # Handled by chat/orchestrator
            "PERMISSION_GRANT": [AgentCapability.RESEARCH_INTELLIGENCE],  # Continue research
            "PERMISSION_CANCEL": [AgentCapability.CONVERSATION],  # Cancel and return to chat
            "EMAIL": [AgentCapability.EMAIL_SENDING]
        }
        
        logger.info(f"🎯 AGENT MAPPING: intent_type={intent_type} → intent_str={intent_str}")
        required_capabilities = intent_to_capability_mapping.get(intent_str, [])
        if not required_capabilities:
            logger.warning(f"❌ No capabilities found for intent: {intent_str}")
            return []
        
        logger.info(f"🎯 Required capabilities for {intent_str}: {required_capabilities}")
        
        capable_agents = []
        for capability in required_capabilities:
            agents_with_capability = self.get_agents_by_capability(capability)
            logger.info(f"🎯 Agents with {capability}: {[agent.agent_type for agent in agents_with_capability]}")
            capable_agents.extend(agents_with_capability)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_agents = []
        for agent in capable_agents:
            if agent.agent_type not in seen:
                seen.add(agent.agent_type)
                unique_agents.append(agent)
        
        logger.info(f"✅ Final capable agents for {intent_str}: {[agent.agent_type for agent in unique_agents]}")
        return unique_agents
    
    def calculate_agent_routing_confidence(self, agent_info: AgentInfo, intent_result: Dict[str, Any], conversation_context: Dict[str, Any]) -> float:
        """ROOSEVELT'S ROUTING CONFIDENCE: Calculate how well an agent matches the current context"""
        base_confidence = 0.5
        
        # Intent alignment bonus - NORMALIZE INTENT TYPE
        intent_type_raw = intent_result.get("intent_type", "")
        # Normalize intent type (handle both enum and string)
        if hasattr(intent_type_raw, 'value'):
            intent_type = intent_type_raw.value.upper()
        else:
            intent_type = str(intent_type_raw).replace("IntentType.", "").upper()
        
        agent_capabilities = [cap.value for cap in agent_info.capabilities]
        
        if intent_type == "WEATHER" and "weather_information" in agent_capabilities:
            base_confidence += 0.3
        elif intent_type == "RESEARCH" and "research_intelligence" in agent_capabilities:
            base_confidence += 0.3
        elif intent_type == "DATA_FORMATTING" and "data_formatting" in agent_capabilities:
            base_confidence += 0.3
        
        # Active agent conversation bonus
        active_agent_context = conversation_context.get("active_agent_context", {})
        if active_agent_context.get("agent") == agent_info.agent_type:
            active_confidence = active_agent_context.get("confidence", 0.0)
            base_confidence += (active_confidence * 0.4)  # Up to 40% bonus for active conversations
        
        # Location clarification bonus for weather agent
        if (agent_info.agent_type == "weather_agent" and 
            conversation_context.get("location_clarification_requested", False)):
            base_confidence += 0.2
        
        # Intent confidence factor
        intent_confidence = intent_result.get("confidence", 0.5)
        final_confidence = base_confidence * intent_confidence
        
        return min(0.95, final_confidence)  # Cap at 95%
    
    def suggest_collaboration_targets(self, source_agent: str, context: Dict[str, Any]) -> List[str]:
        """Suggest which agents might be valuable for collaboration based on context"""
        suggestions = []
        
        # Get source agent info
        source_info = self.get_agent_info(source_agent)
        if not source_info:
            return suggestions
        
        # Analyze context for collaboration opportunities
        user_message = context.get("user_message", "").lower()
        
        if source_agent == "weather_agent":
            # Weather agent looking for research collaboration
            if any(trigger in user_message for trigger in [
                "trip", "travel", "vacation", "visit", "flight", "hotel"
            ]):
                suggestions.append("research_agent")
                
            elif any(trigger in user_message for trigger in [
                "hiking", "camping", "skiing", "beach", "festival", "concert", "wedding", "event"
            ]):
                suggestions.append("research_agent")
                
        elif source_agent == "research_agent":
            # Research agent looking for weather collaboration
            research_findings = context.get("research_findings", "").lower()
            locations_found = context.get("locations", [])
            
            if locations_found and any(keyword in research_findings for keyword in [
                "outdoor", "venue", "location", "destination", "travel", "activity"
            ]):
                suggestions.append("weather_agent")
                
            # Check for data formatting needs
            if any(keyword in research_findings for keyword in [
                "comparison", "list", "table", "data", "statistics", "multiple"
            ]) and len(research_findings) > 500:  # Substantial data
                suggestions.append("data_formatting_agent")
        
        return suggestions
    
    def get_collaboration_prompt_context(self, target_agent: str) -> str:
        """Get context for LLM prompts about collaboration with target agent"""
        agent_info = self.get_agent_info(target_agent)
        if not agent_info:
            return ""
        
        return f"""
COLLABORATION TARGET: {agent_info.display_name}
DESCRIPTION: {agent_info.description}
SPECIALTIES: {', '.join(agent_info.specialties)}
BEST USED FOR: {', '.join(agent_info.handoff_triggers)}
"""
    
    def format_collaboration_suggestion(self, target_agent: str, context: str) -> str:
        """Format a user-friendly collaboration suggestion"""
        agent_info = self.get_agent_info(target_agent)
        if not agent_info:
            return ""
        
        return f"Would you like me to hand this off to the {agent_info.display_name} for {context}?"
    
    def get_collaboration_permission(self, agent_type: str) -> Optional[CollaborationPermission]:
        """ROOSEVELT'S PERMISSION LOOKUP: Get collaboration permission for an agent"""
        agent_info = self.get_agent_info(agent_type)
        return agent_info.collaboration_permission if agent_info else None
    
    def can_auto_execute(self, agent_type: str) -> bool:
        """ROOSEVELT'S AUTO-EXECUTION CHECK: Check if agent allows automatic execution"""
        permission = self.get_collaboration_permission(agent_type)
        return permission == CollaborationPermission.AUTO_USE
    
    def requires_user_approval(self, agent_type: str) -> bool:
        """ROOSEVELT'S APPROVAL CHECK: Check if agent requires user approval before execution"""
        permission = self.get_collaboration_permission(agent_type)
        return permission in [CollaborationPermission.SUGGEST_ONLY, CollaborationPermission.HITL_REQUIRED]


# Global instance
_agent_network_instance: Optional[AgentIntelligenceNetwork] = None


def get_agent_network() -> AgentIntelligenceNetwork:
    """Get the global agent intelligence network instance"""
    global _agent_network_instance
    if _agent_network_instance is None:
        _agent_network_instance = AgentIntelligenceNetwork()
    return _agent_network_instance
