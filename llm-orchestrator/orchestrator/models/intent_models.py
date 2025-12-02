"""
Simple Intent Classification Models

Ported from backend with 1:1 functionality for consistent routing across services.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, TypedDict


class SimpleIntentResult(BaseModel):
    """
    Simple, focused intent classification result
    
    Provides agent routing, action intent, and permission requirements.
    """
    # Required routing target
    target_agent: str = Field(description="Agent to route to (research_agent, chat_agent, rules_editing_agent, etc.)")

    # Semantic action type for intelligent routing
    action_intent: str = Field(
        default="query",
        description="Semantic action type: observation, generation, modification, query, analysis, management"
    )

    # HITL / permission flag
    permission_required: bool = Field(default=False, description="Whether permission needed for this request")

    # Confidence for UI/logging
    confidence: float = Field(default=0.8, description="Confidence in classification (0.0-1.0). Note: Anthropic API doesn't support min/max constraints on number types.")

    # Optional human-readable reasoning
    reasoning: Optional[str] = Field(default=None, description="Brief explanation of routing decision")
    
    @property
    def routing_decision(self) -> dict:
        """Legacy compatibility property for existing code"""
        return {
            "primary_agent": self.target_agent,
            "primary_confidence": self.confidence,
            "requires_context_preservation": True,
            "permission_requirement": {
                "required": self.permission_required,
                "permission_type": "web_search" if self.permission_required else None,
                "reasoning": self.reasoning if self.permission_required else None,
                "auto_grant_eligible": False
            }
        }
    
    @property
    def context_analysis(self) -> dict:
        """Legacy compatibility property for existing code"""
        return {
            "conversation_flow": "new_topic",
            "active_agent": None,
            "collaboration_state": "none",
            "context_relevance": "medium"
        }
    
    @property
    def capable_agents(self) -> list:
        """Legacy compatibility property for existing code"""
        return [{
            "agent_type": self.target_agent,
            "display_name": f"{self.target_agent.replace('_', ' ').title()}",
            "capabilities_matched": ["general"],
            "confidence_score": self.confidence,
            "specialties_relevant": ["general"],
            "collaboration_permission": "auto_use",
            "reasoning": self.reasoning or "Primary agent match"
        }]


class IntentClassificationState(TypedDict):
    """State for intent classification workflow"""
    user_message: str
    conversation_context: Dict[str, Any]
    
    # Stage 1: Domain detection
    domain: str
    
    # Stage 2: Action intent
    action_intent: str
    
    # Stage 3: Routing
    target_agent: str
    confidence: float
    reasoning: str
    permission_required: bool
    
    # Final result
    result: SimpleIntentResult

