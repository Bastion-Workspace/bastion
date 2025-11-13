"""
Enhanced LangGraph State - Roosevelt's "Clean Cavalry" State Management
Minimal, bulletproof state for Chat + Research HITL system

CLEAN STATE ARCHITECTURE:
- Essential fields only (9 core fields)
- LangGraph shared memory for agent communication
- No legacy operation management
- Simple agent handoff patterns
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from enum import Enum

logger = logging.getLogger(__name__)


class CachedResultType(str, Enum):
    """Types of cached results in conversation intelligence"""
    RESEARCH_FINDINGS = "research_findings"
    CHAT_OUTPUT = "chat_output" 
    AGENT_COLLABORATION = "agent_collaboration"
    WEB_SOURCES = "web_sources"
    LOCAL_DOCUMENTS = "local_documents"


class CachedResult(BaseModel):
    """Individual cached result with metadata"""
    content: str = Field(description="The actual content/findings")
    result_type: CachedResultType = Field(description="Type of cached result")
    source_agent: str = Field(description="Agent that produced this result")
    timestamp: str = Field(description="When this result was created")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in result quality")
    topics: List[str] = Field(default_factory=list, description="Key topics covered")
    citations: List[str] = Field(default_factory=list, description="Source citations if any")


class ConversationIntelligence(BaseModel):
    """Roosevelt's Conversation Intelligence - Built-in context analysis"""
    
    # CACHED RESULTS ARCHIVE
    cached_results: Dict[str, CachedResult] = Field(
        default_factory=dict,
        description="Archive of all agent outputs indexed by content hash"
    )
    
    # TOPIC CONTINUITY TRACKING  
    current_topic: Optional[str] = Field(default=None, description="Current conversation topic")
    topic_history: List[str] = Field(default_factory=list, description="Previous topics discussed")
    topic_transitions: List[Dict[str, Any]] = Field(default_factory=list, description="Topic change events")
    
    # AGENT COLLABORATION HISTORY
    agent_outputs: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Outputs by agent type for quick reference"
    )
    collaboration_suggestions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recent collaboration offers from agents"
    )
    
    # RESEARCH INTELLIGENCE CACHE
    research_cache: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Cached research results by query similarity"
    )
    source_cache: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Cached web source content by URL"
    )
    
    # COVERAGE ANALYSIS (Pre-computed)
    query_coverage_cache: Dict[str, float] = Field(
        default_factory=dict,
        description="Pre-computed coverage scores for common follow-up patterns"
    )
    
    # METADATA
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat())
    intelligence_version: str = Field(default="1.0", description="Version for migration support")


class ConversationState(TypedDict):
    """
    Clean ConversationState optimized for Agent Autonomy Pattern
    Minimal fields for maximum reliability
    """
    # === CORE IDENTITY ===
    user_id: str
    conversation_id: str
    messages: List[BaseMessage]
    
    # === AGENT COORDINATION ===
    current_query: Optional[str]                     # Current user query being processed
    active_agent: Optional[str]                      # Which agent is currently active
    shared_memory: Dict[str, Any]                    # LangGraph shared memory for agent communication
    
    # === AGENT RESULTS ===
    intent_classification: Optional[Dict[str, Any]]  # Intent classification results
    agent_results: Optional[Dict[str, Any]]          # Agent processing results
    latest_response: Optional[str]                   # Latest agent response
    chat_agent_response: Optional[str]               # Chat agent specific response
    
    # === ROOSEVELT'S HITL EDITING OPERATIONS ===
    editor_operations: Optional[List[Dict[str, Any]]]  # Structured editor operations for frontend
    manuscript_edit: Optional[Dict[str, Any]]          # ManuscriptEdit metadata for editing operations
    
    # === FLOW CONTROL ===
    requires_user_input: Optional[bool]              # True when waiting for HITL response
    is_complete: Optional[bool]                      # True when conversation turn complete
    error_state: Optional[str]                       # Error information if any
    
    # === CONTEXT ===
    persona: Optional[Dict[str, Any]]                # User persona for personalization
    conversation_topic: Optional[str]                # Current conversation topic
    
    # === ROOSEVELT'S CONVERSATION INTELLIGENCE ===
    conversation_intelligence: Optional[Dict[str, Any]]  # Built-in conversation context and cache
    
    # === CONVERSATION METADATA (LangGraph-native) ===
    conversation_title: Optional[str]               # Generated conversation title
    conversation_created_at: Optional[str]          # ISO timestamp when conversation started
    conversation_updated_at: Optional[str]          # ISO timestamp of last activity
    conversation_tags: Optional[List[str]]          # User-defined tags
    conversation_description: Optional[str]         # Optional description
    is_pinned: Optional[bool]                       # Whether conversation is pinned
    is_archived: Optional[bool]                     # Whether conversation is archived
    
    # === AGENT AUTONOMY SUPPORT ===
    active_agent_checkpoint: Optional[Dict[str, Any]]  # Active agent HITL checkpoint info
    orchestrator_routing_decision: Optional[Dict[str, Any]]  # Temporary routing decision (will be removed)


def create_initial_state(
    user_id: str,
    conversation_id: str,
    messages: Optional[List[BaseMessage]] = None,
    persona: Optional[Dict[str, Any]] = None
) -> ConversationState:
    """Create clean initial conversation state"""
    
    # Initialize shared memory structure for agent communication
    shared_memory = {
        "search_results": {},           # Results from search operations
        "conversation_context": {},     # Context for ongoing conversation
        "agent_handoffs": [],          # Record of agent handoffs
        "data_sufficiency": {}         # Assessment of available data
    }
    
    return ConversationState(
        # Core identity
        user_id=user_id,
        conversation_id=conversation_id,
        messages=messages or [],
        
        # Agent coordination  
        current_query=None,
        active_agent=None,
        shared_memory=shared_memory,
        
        # Agent results
        intent_classification=None,
        agent_results=None,
        latest_response=None,
        chat_agent_response=None,
        
        # Flow control
        requires_user_input=False,
        is_complete=False,
        error_state=None,
        
        # Context
        persona=persona,
        conversation_topic=None,
        
        # Roosevelt's Conversation Intelligence
        conversation_intelligence=ConversationIntelligence().dict(),
        
        # Conversation metadata (LangGraph-native)
        conversation_title=None,  # Will be generated from first user message
        conversation_created_at=datetime.now().isoformat(),
        conversation_updated_at=datetime.now().isoformat(),
        conversation_tags=[],
        conversation_description=None,
        is_pinned=False,
        is_archived=False,
        
        # Agent autonomy support
        active_agent_checkpoint=None,
        orchestrator_routing_decision=None,
        
        # Agent intelligence
        agent_insights={},
        pending_operations=[],
        pending_permission_requests=[]
    )


class ConversationIntentParser:
    """Clean intent parser for HITL permission responses"""
    
    @staticmethod
    def parse_user_intent(message: str, state: ConversationState) -> Dict[str, Any]:
        """Parse user intent for clean permission handling"""
        
        message_lower = message.lower().strip()
        
        # Permission approval patterns
        if any(word in message_lower for word in ["yes", "approve", "go ahead", "proceed", "ok", "okay"]):
            return {
                "action": "approve",
                "confidence": 0.9,
                "reasoning": "User approved permission"
            }
        
        # Cancellation patterns  
        if any(word in message_lower for word in ["no", "cancel", "stop", "skip", "don't"]):
            return {
                "action": "cancel",
                "confidence": 0.9,
                "reasoning": "User cancelled permission"
            }
        
        # New query (anything else)
        return {
            "action": "new_query",
            "confidence": 0.8,
            "reasoning": "User sent new query"
        }


async def load_user_memory(state: ConversationState, memory_store) -> None:
    """Load user preferences and learned behaviors from memory store"""
    try:
        user_id = state["user_id"]
        
        # Load user preferences
        preferences = await memory_store.get_user_preferences(user_id)
        if preferences:
            state["user_preferences"] = {
                pref["context"]: pref["pattern_data"] 
                for pref in preferences
            }
            logger.info(f"✅ Loaded {len(preferences)} user preferences for {user_id}")
        
        # Load agent behaviors
        behaviors = await memory_store.get_agent_behaviors(user_id)
        if behaviors:
            state["agent_behaviors"] = {
                behavior["context"]: behavior["pattern_data"]
                for behavior in behaviors
            }
            logger.info(f"✅ Loaded {len(behaviors)} agent behaviors for {user_id}")
            
    except Exception as e:
        logger.warning(f"⚠️ Failed to load user memory: {e}")
        # Continue without memory - don't fail the conversation


async def save_user_interaction_patterns(state: ConversationState, memory_store) -> None:
    """Save learned patterns from this interaction"""
    try:
        user_id = state["user_id"]
        
        # Save user preferences based on interaction
        if state.get("agent_results") and state.get("active_agent"):
            active_agent = state["active_agent"]
            agent_results = state["agent_results"]
            
            # Learn from successful interactions
            if agent_results.get("response") and not state.get("error_state"):
                preference_data = {
                    "preferred_agent": active_agent,
                    "response_style": "successful",
                    "tools_used": agent_results.get("tools_used", []),
                    "processing_time": agent_results.get("processing_time", 0),
                    "timestamp": datetime.now().isoformat()
                }
                
                await memory_store.store_user_preference(
                    user_id, 
                    f"agent_preference_{active_agent}", 
                    preference_data
                )
                
                logger.info(f"✅ Saved user preference for {active_agent} agent")
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to save user interaction patterns: {e}")
        # Don't fail the conversation for memory issues