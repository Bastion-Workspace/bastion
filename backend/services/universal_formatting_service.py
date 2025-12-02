"""
Universal Formatting Service - Roosevelt's "One Format to Rule Them All"
Provides intelligent formatting detection and capabilities for all agents

Following LangGraph best practices:
- Centralized formatting logic
- LLM-driven intelligent detection
- Agent-agnostic formatting capabilities
"""

import logging
from typing import Dict, Any, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class FormattingType(str, Enum):
    """Types of formatting that can be applied"""
    TABLE = "table"
    LIST = "list"
    COMPARISON = "comparison"
    SUMMARY = "summary"
    NONE = "none"


class UniversalFormattingService:
    """
    Roosevelt's Universal Formatting Service
    Provides intelligent formatting detection and application for all agents
    """
    
    def __init__(self):
        self.formatting_triggers = {
            "table": ["comparison", "multiple", "data points", "versus", "compare", "list of", "several", "various"],
            "list": ["steps", "points", "items", "recommendations", "suggestions", "factors"],
            "comparison": ["better", "worse", "advantage", "disadvantage", "difference", "similar", "contrast"],
            "summary": ["overview", "summary", "key points", "main findings", "conclusion"]
        }
    
    async def detect_formatting_need(self, 
                                   agent_type: str,
                                   user_query: str, 
                                   agent_response: str,
                                   confidence_threshold: float = 0.7) -> Optional[Dict[str, Any]]:
        """
        ROOSEVELT'S UNIVERSAL FORMATTING DETECTION
        Intelligently detect if agent response would benefit from formatting
        """
        try:
            # Skip formatting for very short responses
            if len(agent_response) < 200:
                return None
            
            # Skip formatting for simple conversational acknowledgments
            user_query_lower = user_query.lower().strip()
            if any(phrase in user_query_lower for phrase in [
                "thanks", "thank you", "thx", "ty", "appreciate", 
                "cool", "nice", "ok", "okay", "alright"
            ]) and len(user_query) < 50:
                return None
            
            # Use LLM to analyze formatting needs
            formatting_analysis = await self._llm_analyze_formatting_need(
                agent_type, user_query, agent_response
            )
            
            if (formatting_analysis and 
                formatting_analysis.get("confidence", 0) >= confidence_threshold and
                formatting_analysis.get("formatting_type") != FormattingType.NONE):
                
                logger.info(f"📊 UNIVERSAL FORMATTING: {agent_type} response needs {formatting_analysis['formatting_type']} formatting (confidence: {formatting_analysis['confidence']:.2f})")
                return formatting_analysis
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Universal formatting detection error: {e}")
            return None
    
    async def _llm_analyze_formatting_need(self, agent_type: str, user_query: str, agent_response: str) -> Optional[Dict[str, Any]]:
        """Use LLM to intelligently analyze formatting needs"""
        try:
            from utils.openrouter_client import get_openrouter_client
            
            client = get_openrouter_client()
            
            # Build context-aware prompt
            prompt = f"""Analyze if this agent response would benefit from structured formatting.

AGENT TYPE: {agent_type}
USER QUERY: "{user_query}"
AGENT RESPONSE: "{agent_response[:1000]}..."

FORMATTING ANALYSIS CRITERIA:

1. **TABLE FORMATTING** - Use when response contains:
   - Multiple data points with comparable attributes
   - Comparison of features/options/items
   - Structured data that would be clearer in rows/columns
   - Examples: product comparisons, feature lists, statistics

2. **LIST FORMATTING** - Use when response contains:
   - Sequential steps or procedures
   - Multiple recommendations or suggestions
   - Enumerated points or factors
   - Examples: instructions, bullet points, ordered items

3. **COMPARISON FORMATTING** - Use when response contains:
   - Side-by-side comparisons
   - Pros and cons analysis
   - Before/after scenarios
   - Examples: option evaluation, A vs B analysis

4. **SUMMARY FORMATTING** - Use when response contains:
   - Key findings from longer text
   - Executive summary style content
   - Condensed conclusions
   - Examples: research summaries, final recommendations

5. **NO FORMATTING** - Use when:
   - Response is narrative/conversational
   - Content flows better as natural text
   - Formatting would reduce readability

RESPONSE FORMAT (JSON only):
{{
    "formatting_type": "table|list|comparison|summary|none",
    "confidence": 0.0-1.0,
    "reasoning": "why this formatting type is recommended",
    "data_structure": "description of how data should be organized"
}}

Return ONLY valid JSON, nothing else."""

            response = await client.chat.completions.create(
                model=settings.FAST_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1  # Low temperature for consistent analysis
            )
            
            # Parse LLM response
            import json
            try:
                formatting_data = json.loads(response.choices[0].message.content)
                return formatting_data
            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse formatting analysis JSON: {e}")
                return None
                
        except Exception as e:
            logger.error(f"❌ LLM formatting analysis failed: {e}")
            return None
    
    def should_auto_format(self, agent_type: str, formatting_analysis: Dict[str, Any]) -> bool:
        """
        Determine if formatting should be applied automatically based on agent type and confidence
        """
        if not formatting_analysis:
            return False
            
        confidence = formatting_analysis.get("confidence", 0)
        formatting_type = formatting_analysis.get("formatting_type", FormattingType.NONE)
        
        # Agent-specific auto-formatting thresholds
        auto_format_thresholds = {
            "research_agent": 0.8,  # High threshold for research - user expects data
            "weather_agent": 0.7,   # Medium threshold - weather data often tabular
            "chat_agent": 0.9,      # Very high threshold - chat should stay conversational
            "data_formatting_agent": 0.0,  # Always format - that's its job
            "rss_agent": 0.8        # High threshold - RSS data is often structured
        }
        
        threshold = auto_format_thresholds.get(agent_type, 0.8)  # Default high threshold
        
        # Never auto-format if confidence is below threshold
        if confidence < threshold:
            return False
            
        # Never auto-format conversational responses for chat agent
        if agent_type == "chat_agent" and formatting_type in [FormattingType.NONE]:
            return False
            
        return True
    
    async def apply_formatting(self, 
                             agent_response: str, 
                             formatting_analysis: Dict[str, Any],
                             state: Dict[str, Any] = None) -> str:
        """
        Apply the recommended formatting to the agent response
        ROOSEVELT'S AUTO-FORMATTING: Actually execute Data Formatting Agent when beneficial
        """
        try:
            formatting_type = formatting_analysis.get("formatting_type", FormattingType.NONE)
            
            if formatting_type == FormattingType.NONE:
                return agent_response
            
            logger.info(f"📊 UNIVERSAL FORMATTING: Applying {formatting_type} formatting")
            
            # ROOSEVELT'S PERMISSION-AWARE AUTO-EXECUTION: Check collaboration permission before execution
            if formatting_type == FormattingType.TABLE and state:
                try:
                    # Check if Data Formatting Agent allows auto-execution
                    auto_execution_allowed = await self._check_auto_execution_permission("data_formatting_agent")
                    
                    if auto_execution_allowed:
                        formatted_result = await self._execute_data_formatting_agent(
                            agent_response, formatting_analysis, state
                        )
                        if formatted_result:
                            logger.info("✅ UNIVERSAL AUTO-FORMATTING: Data Formatting Agent successfully applied")
                            return formatted_result
                    else:
                        logger.info("📋 UNIVERSAL FORMATTING: Data Formatting Agent requires suggestion, not auto-execution")
                except Exception as e:
                    logger.warning(f"⚠️ Data Formatting Agent execution failed, using fallback: {e}")
            
            # Fallback: Add formatting suggestion message
            formatted_response = agent_response
            if formatting_type != FormattingType.NONE:
                formatted_response += f"\n\n*[This data would benefit from {formatting_type} formatting for better readability]*"
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"❌ Formatting application error: {e}")
            return agent_response  # Return original on error
    
    async def _execute_data_formatting_agent(self, 
                                           agent_response: str, 
                                           formatting_analysis: Dict[str, Any],
                                           state: Dict[str, Any]) -> Optional[str]:
        """ROOSEVELT'S AUTO-FORMATTING: Execute Data Formatting Agent for table creation"""
        try:
            from services.langgraph_agents.data_formatting_agent import DataFormattingAgent
            
            # Create a Data Formatting Agent instance
            formatting_agent = DataFormattingAgent()
            
            # Create a formatting request state
            formatting_request = f"Please format this data into a table:\n\n{agent_response}"
            
            # Create a minimal state for the formatting agent
            formatting_state = {
                "messages": state.get("messages", []) + [
                    type('Message', (), {
                        'content': formatting_request,
                        'type': 'human'
                    })()
                ],
                "shared_memory": state.get("shared_memory", {}),
                "user_id": state.get("user_id"),
                "active_agent": "data_formatting_agent"
            }
            
            # Execute the formatting agent
            result_state = await formatting_agent._process_request(formatting_state)
            
            # Extract the formatted result
            formatted_output = result_state.get("latest_response", "")
            if formatted_output and len(formatted_output) > len(agent_response) * 0.8:
                # If formatting produced substantial output, use it
                return formatted_output
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Data Formatting Agent execution failed: {e}")
            return None
    
    async def _check_auto_execution_permission(self, target_agent: str) -> bool:
        """ROOSEVELT'S PERMISSION CHECK: Verify if target agent allows auto-execution"""
        try:
            from services.agent_intelligence_network import get_agent_network, CollaborationPermission
            
            agent_network = get_agent_network()
            agent_info = agent_network.get_agent_info(target_agent)
            
            if not agent_info:
                logger.warning(f"⚠️ Agent not found in intelligence network: {target_agent}")
                return False
            
            permission = agent_info.collaboration_permission
            auto_allowed = permission == CollaborationPermission.AUTO_USE
            
            logger.info(f"🔐 COLLABORATION PERMISSION CHECK: {target_agent} = {permission.value} (auto_allowed: {auto_allowed})")
            return auto_allowed
            
        except Exception as e:
            logger.error(f"❌ Permission check failed: {e}")
            return False  # Fail safe - no auto-execution if permission check fails


# Global instance
_universal_formatting_service: Optional[UniversalFormattingService] = None


def get_universal_formatting_service() -> UniversalFormattingService:
    """Get the global universal formatting service instance"""
    global _universal_formatting_service
    if _universal_formatting_service is None:
        _universal_formatting_service = UniversalFormattingService()
    return _universal_formatting_service
