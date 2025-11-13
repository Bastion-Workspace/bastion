"""
Clarity Assessment Service - Assesses query clarity and generates intelligent clarification requests
Provides ambiguity detection and context-aware clarification suggestions
"""

import logging
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ClarityLevel(str, Enum):
    """Levels of query clarity"""
    CLEAR = "clear"
    AMBIGUOUS = "ambiguous"
    VAGUE = "vague"
    UNSPECIFIC = "unspecific"


@dataclass
class Ambiguity:
    """Represents a specific ambiguity in a query"""
    type: str  # pronoun, reference, scope, temporal, etc.
    description: str
    affected_text: str
    suggested_resolution: str


@dataclass
class ClarityAssessment:
    """Assessment of query clarity for research planning"""
    is_clear: bool
    clarity_level: ClarityLevel
    clarity_score: float  # 0-100
    ambiguities: List[Ambiguity]
    clarification_questions: List[str]
    suggested_resolution: str
    confidence: float  # 0-1 confidence in assessment


@dataclass
class ClarificationRequest:
    """Structured clarification request with intelligent options"""
    message: str
    clarification_type: str
    questions: List[str]
    suggested_options: List[str]
    query_id: Optional[str] = None  # Link to pending query
    quick_actions: List[str] = None
    context_hints: List[str] = None
    
    def __post_init__(self):
        if self.quick_actions is None:
            self.quick_actions = []
        if self.context_hints is None:
            self.context_hints = []


class ClarityAssessmentService:
    """Service for assessing query clarity and generating clarification requests"""
    
    def __init__(self, mcp_chat_service=None, pending_query_manager=None):
        self.mcp_chat_service = mcp_chat_service
        self.pending_query_manager = pending_query_manager
        
    async def assess_query_clarity(self, query: str, conversation_context: Dict) -> ClarityAssessment:
        """Assess if query is clear enough for research planning"""
        try:
            logger.info(f"🔍 Assessing query clarity: {query[:50]}...")
            
            if self.mcp_chat_service:
                assessment = await self._assess_with_llm(query, conversation_context)
            else:
                assessment = self._assess_simple(query, conversation_context)
            
            logger.info(f"✅ Clarity assessment: {assessment.clarity_level} (score: {assessment.clarity_score})")
            return assessment
            
        except Exception as e:
            logger.error(f"❌ Failed to assess query clarity: {e}")
            return self._create_default_assessment(query)
    
    async def _assess_with_llm(self, query: str, conversation_context: Dict) -> ClarityAssessment:
        """Assess query clarity using LLM for comprehensive analysis"""
        try:
            context_summary = conversation_context.get('summary', 'No prior context')
            entities = conversation_context.get('entities', [])
            topics = conversation_context.get('topics', [])
            
            assessment_prompt = f"""
            Analyze this research query for clarity and completeness:
            
            Query: "{query}"
            
            Conversation Context: {context_summary}
            Recent Entities: {[e.name for e in entities] if hasattr(entities[0], 'name') else entities}
            Recent Topics: {topics}
            
            Assess the following aspects:
            1. **Pronoun Ambiguity**: Are there unclear pronouns? (he, she, it, they, this, that)
            2. **Reference Ambiguity**: Are there vague references? (the company, the book, the policy)
            3. **Scope Ambiguity**: Is the scope too broad or vague?
            4. **Temporal Ambiguity**: Are there unclear time references? (recent, recent, latest)
            5. **Missing Details**: Are essential details missing?
            
            Return JSON response:
            {{
                "is_clear": boolean,
                "clarity_level": "clear|ambiguous|vague|unspecific",
                "clarity_score": 0-100,
                "ambiguities": [
                    {{
                        "type": "pronoun|reference|scope|temporal|missing",
                        "description": "description of the ambiguity",
                        "affected_text": "specific text that's ambiguous",
                        "suggested_resolution": "how to resolve this"
                    }}
                ],
                "clarification_questions": ["specific questions to ask"],
                "suggested_resolution": "how to resolve with context",
                "confidence": 0.0-1.0
            }}
            
            Be thorough but concise. Focus on research-relevant ambiguities.
            """
            
            conversation_log = [
                {"role": "system", "content": "You are a query clarity assessment specialist. Return only valid JSON."},
                {"role": "user", "content": assessment_prompt}
            ]
            
            response = await self.mcp_chat_service._get_llm_response(conversation_log)
            
            # Parse JSON response
            try:
                assessment_data = json.loads(response)
                
                # Convert ambiguities to Ambiguity objects
                ambiguities = []
                for amb_data in assessment_data.get("ambiguities", []):
                    ambiguity = Ambiguity(
                        type=amb_data["type"],
                        description=amb_data["description"],
                        affected_text=amb_data["affected_text"],
                        suggested_resolution=amb_data["suggested_resolution"]
                    )
                    ambiguities.append(ambiguity)
                
                assessment = ClarityAssessment(
                    is_clear=assessment_data.get("is_clear", True),
                    clarity_level=ClarityLevel(assessment_data.get("clarity_level", "clear")),
                    clarity_score=float(assessment_data.get("clarity_score", 100)),
                    ambiguities=ambiguities,
                    clarification_questions=assessment_data.get("clarification_questions", []),
                    suggested_resolution=assessment_data.get("suggested_resolution", ""),
                    confidence=float(assessment_data.get("confidence", 1.0))
                )
                
                return assessment
                
            except json.JSONDecodeError:
                logger.warning("⚠️ Failed to parse LLM clarity assessment response")
                return self._assess_simple(query, conversation_context)
                
        except Exception as e:
            logger.error(f"❌ LLM clarity assessment failed: {e}")
            return self._assess_simple(query, conversation_context)
    
    def _assess_simple(self, query: str, conversation_context: Dict) -> ClarityAssessment:
        """Simple clarity assessment using pattern matching"""
        ambiguities = []
        clarification_questions = []
        
        # Check for pronoun ambiguity
        pronouns = ["he", "she", "it", "they", "his", "her", "their", "this", "that", "these", "those"]
        for pronoun in pronouns:
            if pronoun in query.lower():
                ambiguities.append(Ambiguity(
                    type="pronoun",
                    description=f"Unclear reference to '{pronoun}'",
                    affected_text=pronoun,
                    suggested_resolution="Specify the subject being referred to"
                ))
                clarification_questions.append(f"Who or what does '{pronoun}' refer to?")
        
        # Check for vague references
        vague_terms = ["the company", "the book", "the policy", "the event", "the person"]
        for term in vague_terms:
            if term in query.lower():
                ambiguities.append(Ambiguity(
                    type="reference",
                    description=f"Vague reference to '{term}'",
                    affected_text=term,
                    suggested_resolution="Specify which company/book/policy/event/person"
                ))
                clarification_questions.append(f"Which specific {term.split()[-1]} are you referring to?")
        
        # Check for broad scope
        broad_indicators = ["everything", "all", "comprehensive", "complete", "extensive"]
        for indicator in broad_indicators:
            if indicator in query.lower():
                ambiguities.append(Ambiguity(
                    type="scope",
                    description=f"Very broad scope with '{indicator}'",
                    affected_text=indicator,
                    suggested_resolution="Narrow down to specific aspects or time periods"
                ))
                clarification_questions.append("What specific aspects or time periods are you most interested in?")
        
        # Check for temporal ambiguity
        temporal_terms = ["recent", "latest", "current", "modern", "contemporary"]
        for term in temporal_terms:
            if term in query.lower():
                ambiguities.append(Ambiguity(
                    type="temporal",
                    description=f"Unclear time reference '{term}'",
                    affected_text=term,
                    suggested_resolution="Specify time period (e.g., last 5 years, 2020s, etc.)"
                ))
                clarification_questions.append(f"What time period do you mean by '{term}'?")
        
        # Calculate clarity score
        clarity_score = max(0, 100 - (len(ambiguities) * 20))
        is_clear = clarity_score >= 70
        clarity_level = ClarityLevel.CLEAR if is_clear else ClarityLevel.AMBIGUOUS
        
        return ClarityAssessment(
            is_clear=is_clear,
            clarity_level=clarity_level,
            clarity_score=clarity_score,
            ambiguities=ambiguities,
            clarification_questions=clarification_questions,
            suggested_resolution="Please provide more specific details about your research request.",
            confidence=0.8 if ambiguities else 0.9
        )
    
    def _create_default_assessment(self, query: str) -> ClarityAssessment:
        """Create default assessment when analysis fails"""
        return ClarityAssessment(
            is_clear=True,
            clarity_level=ClarityLevel.CLEAR,
            clarity_score=80,
            ambiguities=[],
            clarification_questions=[],
            suggested_resolution="",
            confidence=0.5
        )
    
    async def generate_clarification_request(
        self, 
        assessment: ClarityAssessment, 
        conversation_context: Dict,
        query: str = None,
        conversation_id: str = None,
        user_id: str = None
    ) -> ClarificationRequest:
        """Generate intelligent clarification request based on assessment"""
        try:
            if assessment.is_clear:
                return None  # No clarification needed
            
            # Build context-aware message
            message = self._build_clarification_message(assessment, conversation_context)
            
            # Generate context-aware suggestions
            suggested_options = self._generate_suggested_options(assessment, conversation_context)
            
            # Generate quick actions
            quick_actions = self._generate_quick_actions(assessment)
            
            # Generate context hints
            context_hints = self._generate_context_hints(conversation_context)
            
            # Store pending query if we have the manager and required data
            query_id = None
            if (self.pending_query_manager and query and conversation_id and user_id):
                try:
                    query_id = await self.pending_query_manager.store_pending_query(
                        original_query=query,
                        conversation_id=conversation_id,
                        user_id=user_id,
                        ambiguities=[ambiguity.ambiguity_type for ambiguity in assessment.ambiguities],
                        context_snapshot=conversation_context
                    )
                    logger.info(f"✅ Stored pending query: {query_id}")
                except Exception as e:
                    logger.error(f"❌ Failed to store pending query: {e}")
            
            request = ClarificationRequest(
                message=message,
                clarification_type="context_resolution",
                questions=assessment.clarification_questions,
                suggested_options=suggested_options,
                query_id=query_id,
                quick_actions=quick_actions,
                context_hints=context_hints
            )
            
            return request
            
        except Exception as e:
            logger.error(f"❌ Failed to generate clarification request: {e}")
            return self._create_fallback_clarification_request(assessment)
    
    def _build_clarification_message(self, assessment: ClarityAssessment, 
                                   conversation_context: Dict) -> str:
        """Build context-aware clarification message"""
        message = "I'd like to clarify your research request to ensure I provide exactly what you need:"
        
        # Add context awareness
        if conversation_context.get('summary') and conversation_context.get('summary') != "No conversation history available.":
            message += f"\n\nBased on our recent discussion about {conversation_context.get('summary', 'various topics')}, "
        
        # Add specific ambiguity mentions
        if assessment.ambiguities:
            ambiguity_types = set(amb.type for amb in assessment.ambiguities)
            if "pronoun" in ambiguity_types:
                message += "I noticed some unclear references (like 'he', 'she', 'it', 'they'). "
            if "reference" in ambiguity_types:
                message += "I found some vague references (like 'the company', 'the policy'). "
            if "scope" in ambiguity_types:
                message += "The scope seems quite broad. "
            if "temporal" in ambiguity_types:
                message += "The time period is unclear. "
        
        message += "\n\nCould you please clarify:"
        
        return message
    
    def _generate_suggested_options(self, assessment: ClarityAssessment, 
                                  conversation_context: Dict) -> List[str]:
        """Generate context-aware suggested options"""
        options = []
        
        # Add options based on recent entities
        entities = conversation_context.get('entities', [])
        if entities:
            for entity in entities[:3]:  # Limit to 3 entities
                if hasattr(entity, 'name'):
                    options.append(f"Research {entity.name} (based on our recent discussion)")
        
        # Add options based on ambiguity types
        for ambiguity in assessment.ambiguities:
            if ambiguity.type == "pronoun" and ambiguity.suggested_resolution:
                options.append(ambiguity.suggested_resolution)
            elif ambiguity.type == "scope":
                options.append("Focus on specific aspects or time periods")
            elif ambiguity.type == "temporal":
                options.append("Specify the time period you're interested in")
        
        # Add generic helpful options
        if not options:
            options.extend([
                "Expand scope to include related topics",
                "Focus on specific time period",
                "Compare multiple subjects"
            ])
        
        return options[:5]  # Limit to 5 options
    
    def _generate_quick_actions(self, assessment: ClarityAssessment) -> List[str]:
        """Generate quick action suggestions"""
        actions = []
        
        for ambiguity in assessment.ambiguities:
            if ambiguity.type == "scope":
                actions.append("Narrow down to specific aspects")
            elif ambiguity.type == "temporal":
                actions.append("Specify time period")
            elif ambiguity.type == "reference":
                actions.append("Provide specific names/entities")
        
        # Add general actions
        actions.extend([
            "Provide more specific details",
            "Give me examples of what you're looking for"
        ])
        
        return actions[:4]  # Limit to 4 actions
    
    def _generate_context_hints(self, conversation_context: Dict) -> List[str]:
        """Generate helpful context hints"""
        hints = []
        
        # Add entity hints
        entities = conversation_context.get('entities', [])
        if entities:
            entity_names = [e.name for e in entities if hasattr(e, 'name')]
            if entity_names:
                hints.append(f"Recent subjects: {', '.join(entity_names[:3])}")
        
        # Add topic hints
        topics = conversation_context.get('topics', [])
        if topics:
            hints.append(f"Recent topics: {', '.join(topics[:3])}")
        
        # Add summary hint
        summary = conversation_context.get('summary', '')
        if summary and summary != "No conversation history available.":
            hints.append(f"Context: {summary[:100]}...")
        
        return hints
    
    def _create_fallback_clarification_request(self, assessment: ClarityAssessment) -> ClarificationRequest:
        """Create fallback clarification request when generation fails"""
        return ClarificationRequest(
            message="I'd like to clarify your research request to ensure I provide exactly what you need:",
            clarification_type="general",
            questions=assessment.clarification_questions or ["Could you provide more specific details?"],
            suggested_options=["Provide more specific details", "Give examples of what you're looking for"],
            quick_actions=["Be more specific", "Provide examples"],
            context_hints=[]
        )
    
    def should_request_clarification(self, assessment: ClarityAssessment) -> bool:
        """Determine if clarification should be requested"""
        # Request clarification if:
        # 1. Clarity score is low
        # 2. There are specific ambiguities
        # 3. Confidence in assessment is reasonable
        
        return (
            assessment.clarity_score < 70 and
            len(assessment.ambiguities) > 0 and
            assessment.confidence > 0.5
        )
    
    def get_clarification_priority(self, assessment: ClarityAssessment) -> str:
        """Get priority level for clarification request"""
        if assessment.clarity_score < 40:
            return "high"
        elif assessment.clarity_score < 70:
            return "medium"
        else:
            return "low"
