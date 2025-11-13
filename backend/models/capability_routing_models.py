"""
Capability-Based Routing Models - Roosevelt's "Dynamic Agent Command"
Pydantic models for structured intent classification and capability-based routing
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class RoutingConfidence(str, Enum):
    """Routing confidence levels"""
    HIGH = "high"
    MEDIUM = "medium" 
    LOW = "low"


class ConversationFlow(str, Enum):
    """Conversation flow analysis"""
    CONTINUATION = "continuation"
    NEW_TOPIC = "new_topic"
    COLLABORATION_RESPONSE = "collaboration_response"
    PERMISSION_RESPONSE = "permission_response"
    
    @classmethod
    def from_string(cls, value: str):
        """Convert string to enum, handling case variations"""
        value_lower = value.lower()
        for item in cls:
            if item.value.lower() == value_lower:
                return item
        return cls.NEW_TOPIC  # Default fallback


class CollaborationState(str, Enum):
    """Collaboration state tracking"""
    PENDING = "pending"
    NONE = "none"
    DECLINED = "declined"
    ACCEPTED = "accepted"
    
    @classmethod
    def from_string(cls, value: str):
        """Convert string to enum, handling case variations"""
        value_lower = value.lower()
        for item in cls:
            if item.value.lower() == value_lower:
                return item
        return cls.NONE  # Default fallback


class ContextRelevance(str, Enum):
    """Context relevance assessment"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    
    @classmethod
    def from_string(cls, value: str):
        """Convert string to enum, handling case variations"""
        value_lower = value.lower()
        for item in cls:
            if item.value.lower() == value_lower:
                return item
        return cls.MEDIUM  # Default fallback


class PermissionRequirement(BaseModel):
    """Permission requirement details"""
    required: bool = Field(description="Whether permission is required")
    permission_type: Optional[str] = Field(default=None, description="Type of permission needed")
    reasoning: Optional[str] = Field(default=None, description="Why permission is needed")
    auto_grant_eligible: bool = Field(default=False, description="Whether auto-grant is possible")


class AgentCapabilityMatch(BaseModel):
    """Agent capability matching result"""
    agent_type: str = Field(description="Agent identifier")
    display_name: str = Field(description="Human-readable agent name")
    capabilities_matched: List[str] = Field(description="Capabilities that match the request")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in this agent match")
    specialties_relevant: List[str] = Field(description="Relevant specialties for this request")
    collaboration_permission: str = Field(description="Collaboration permission level")
    reasoning: str = Field(description="Why this agent is suitable")


class ContextAnalysis(BaseModel):
    """Conversation context analysis"""
    conversation_flow: ConversationFlow = Field(description="Type of conversation flow")
    active_agent: Optional[str] = Field(default=None, description="Currently active agent if any")
    collaboration_state: CollaborationState = Field(description="Current collaboration state")
    context_relevance: ContextRelevance = Field(description="Relevance of conversation context")
    topic_continuity: bool = Field(description="Whether user is continuing previous topic")
    permission_context: Optional[str] = Field(default=None, description="Permission context if relevant")


class RoutingDecision(BaseModel):
    """Complete routing decision with alternatives"""
    primary_agent: str = Field(description="Primary recommended agent")
    primary_confidence: float = Field(ge=0.0, le=1.0, description="Confidence in primary choice")
    alternative_agents: List[AgentCapabilityMatch] = Field(description="Alternative agent options")
    routing_reasoning: str = Field(description="Explanation of routing decision")
    requires_context_preservation: bool = Field(description="Whether to preserve conversation context")
    permission_requirement: PermissionRequirement = Field(description="Permission requirements")


class CapabilityBasedIntentResult(BaseModel):
    """Complete capability-based intent classification result"""
    # Intent Classification
    intent_type: str = Field(description="Classified intent type")
    intent_confidence: float = Field(ge=0.0, le=1.0, description="Confidence in intent classification")
    intent_reasoning: str = Field(description="Reasoning for intent classification")
    
    # Context Analysis
    context_analysis: ContextAnalysis = Field(description="Conversation context analysis")
    
    # Capability Matching
    capable_agents: List[AgentCapabilityMatch] = Field(description="All agents capable of handling this request")
    
    # Routing Decision
    routing_decision: RoutingDecision = Field(description="Final routing recommendation")
    
    # Legacy Compatibility
    routing_recommendation: str = Field(description="Legacy routing field for backward compatibility")
    target_agent: str = Field(description="Legacy target agent field for backward compatibility")


class AgentCapabilityInfo(BaseModel):
    """Agent capability information for dynamic prompting"""
    agent_type: str = Field(description="Agent identifier")
    display_name: str = Field(description="Human-readable name")
    description: str = Field(description="Agent description")
    capabilities: List[str] = Field(description="List of agent capabilities")
    specialties: List[str] = Field(description="Specific areas of expertise")
    tools_available: List[str] = Field(description="Tools this agent can use")
    collaboration_permission: str = Field(description="How this agent can be invoked")
    handoff_triggers: List[str] = Field(description="Scenarios where this agent should be suggested")


class SystemCapabilitySnapshot(BaseModel):
    """Complete system capability snapshot for dynamic prompting"""
    available_agents: List[AgentCapabilityInfo] = Field(description="All available agents with capabilities")
    tool_registry: Dict[str, Dict[str, Any]] = Field(description="Available tools by category")
    collaboration_patterns: Dict[str, List[str]] = Field(description="Known collaboration patterns")
    timestamp: str = Field(description="When this snapshot was generated")
