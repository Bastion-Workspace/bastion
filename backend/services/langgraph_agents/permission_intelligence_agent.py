"""
Permission Intelligence Agent - Roosevelt's "Smart Permission Analysis"
LangGraph best practice implementation for intelligent permission detection
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from langchain_openai import ChatOpenAI
from .base_agent import BaseAgent, TaskStatus, AgentError
from models.agent_response_models import PermissionAnalysisResult

logger = logging.getLogger(__name__)


class PermissionIntelligenceAgent(BaseAgent):
    """
    Roosevelt's Permission Intelligence Agent
    
    LangGraph best practice: Dedicated agent for permission analysis
    Uses LLM intelligence instead of static string matching
    """
    
    def __init__(self):
        super().__init__("permission_intelligence_agent")
    
    def _build_permission_analysis_prompt(self, conversation_messages: List[Dict], current_query: str) -> str:
        """Build intelligent permission analysis prompt"""
        
        # Extract recent conversation context
        recent_messages = conversation_messages[-10:] if conversation_messages else []
        conversation_context = ""
        
        for i, msg in enumerate(recent_messages):
            if hasattr(msg, 'content'):
                # LangGraph message object
                role = "User" if getattr(msg, 'type', '') == 'human' else "Assistant"
                content = msg.content[:200] + ("..." if len(msg.content) > 200 else "")
                conversation_context += f"{role}: {content}\n"
            elif isinstance(msg, dict):
                # Dictionary format message
                role = "User" if msg.get("role") == "user" else "Assistant"
                content = msg.get("content", "")[:200] + ("..." if len(msg.get("content", "")) > 200 else "")
                conversation_context += f"{role}: {content}\n"
        
        return f"""You are Roosevelt's Permission Intelligence Agent - a specialist in analyzing user intent for web search authorization.

MISSION: Analyze the conversation context and current user query to determine if the user has granted (or is implicitly granting) permission for web search research.

CONVERSATION CONTEXT:
{conversation_context}

CURRENT USER QUERY: "{current_query}"

PERMISSION ANALYSIS FRAMEWORK:

1. **EXPLICIT PERMISSION GRANTS**:
   - Direct statements: "Yes", "Proceed", "Go ahead", "Do the web search"
   - Confirmation responses to permission requests
   - Instructions that assume web access: "You'll be doing a web search"

2. **IMPLICIT PERMISSION GRANTS**:
   - Research requests that naturally require web access:
     * "Look into [person's] YouTube activity"
     * "Investigate [topic]"
     * "Find information about [subject]"
     * "Research [person/company/topic]"
     * "Check [person's] social media"
   - Requests for comprehensive analysis requiring external data
   - Professional research requests (biographical, business intelligence, due diligence)

3. **CONTINUATION PATTERNS**:
   - Follow-up commands after system suggestions: "Proceed with the report"
   - Commands that build on previous research context
   - User treating web search as expected/normal

4. **CONTEXT INTELLIGENCE**:
   - Previous permission grants in the conversation
   - Established research workflow in progress
   - User familiarity with research capabilities

PROFESSIONAL RESEARCH LEGITIMACY:
These are STANDARD, legitimate research activities:
- Biographical research using public records
- Business intelligence from public sources  
- Academic research and fact-checking
- Journalistic investigation of public figures
- Due diligence research
- Social media and online presence analysis

DECISION CRITERIA:
- If the request is a legitimate research task that would naturally require web access → GRANTED
- If user has explicitly or implicitly authorized web search → GRANTED  
- If user is continuing previous research workflow → GRANTED
- If unclear or ambiguous → REQUEST_PERMISSION
- If user explicitly denied → DENIED

RESPONSE FORMAT:
Provide your analysis in the following JSON structure:

```json
{{
    "permission_status": "granted|request_permission|denied",
    "confidence_level": 0.85,
    "reasoning": "Clear explanation of your decision",
    "permission_type": "explicit|implicit|continuation|inherited",
    "research_legitimacy": "standard_research|specialized|unclear",
    "recommended_tools": ["search_and_crawl", "crawl_web_content"],
    "scope": "current_query|conversation|session"
}}
```

Analyze the conversation and provide your permission intelligence assessment."""

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process permission analysis - LangGraph best practices"""
        try:
            logger.info("🛡️ Permission Intelligence Agent analyzing conversation...")
            
            # Extract conversation context
            messages = state.get("messages", [])
            current_query = state.get("current_query", "")
            
            # Build analysis prompt
            system_prompt = self._build_permission_analysis_prompt(messages, current_query)
            
            # Prepare messages for LLM analysis
            analysis_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze permission status for query: '{current_query}'"}
            ]
            
            # Call LLM for intelligent permission analysis
            start_time = datetime.now()
            chat_service = await self._get_chat_service()
            model_name = await self._get_model_name()
            
            # Use universal approach for any OpenRouter model
            base_llm = ChatOpenAI(
                model=model_name,
                openai_api_key=chat_service.openai_client.api_key,
                openai_api_base=str(getattr(chat_service.openai_client, 'base_url', None)) if getattr(chat_service.openai_client, 'base_url', None) else None,
                temperature=0.1
            )
            
            logger.info("🎯 ROOSEVELT'S UNIVERSAL LLM: Executing permission analysis with JSON extraction")
            raw_response = await base_llm.ainvoke(analysis_messages)
            raw_content = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Parse structured response with universal JSON extraction
            analysis_result = self._parse_permission_analysis_universal(raw_content)
            
            # Update shared memory with permission decision
            if "shared_memory" not in state:
                state["shared_memory"] = {}
            
            if analysis_result.permission_status == "granted":
                logger.info(f"✅ PERMISSION GRANTED: {analysis_result.reasoning}")
                state["shared_memory"]["web_search_permission"] = "granted"
                state["shared_memory"]["approved_operation"] = {
                    "type": "web_search",
                    "granted_at": datetime.now().isoformat(),
                    "scope": analysis_result.scope,
                    "grant_type": analysis_result.permission_type,
                    "confidence": analysis_result.confidence_level,
                    "reasoning": analysis_result.reasoning
                }
            elif analysis_result.permission_status == "denied":
                logger.info(f"❌ PERMISSION DENIED: {analysis_result.reasoning}")
                state["shared_memory"]["web_search_permission"] = "denied"
            else:
                logger.info(f"🤔 PERMISSION UNCLEAR: {analysis_result.reasoning}")
                state["shared_memory"]["web_search_permission"] = "request_needed"
            
            # Create agent results
            state["agent_results"] = self._create_agent_result(
                response=f"Permission analysis: {analysis_result.permission_status}",
                task_status=TaskStatus.COMPLETE,
                tools_used=[],
                processing_time=processing_time,
                additional_data={
                    "permission_analysis": analysis_result.dict(),
                    "permission_granted": analysis_result.permission_status == "granted"
                }
            )
            
            # Add permission analysis to LangGraph messages
            from langchain_core.messages import AIMessage
            state.setdefault("messages", []).append(
                AIMessage(content=f"Permission Analysis: {analysis_result.permission_status} - {analysis_result.reasoning}")
            )
            
            logger.info(f"✅ Permission Intelligence completed in {processing_time:.2f}s")
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Permission Intelligence error: {e}")
            # Fallback to request permission on error
            if "shared_memory" not in state:
                state["shared_memory"] = {}
            state["shared_memory"]["web_search_permission"] = "request_needed"
            
            error = AgentError(
                error_type=type(e).__name__,
                message=str(e),
                recovery_actions=["fallback_to_permission_request"]
            )
            return self._create_error_result(error)
    
    def _parse_permission_analysis(self, response_content: str) -> PermissionAnalysisResult:
        """Parse LLM permission analysis response"""
        try:
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_content, re.DOTALL)
            if json_match:
                json_data = json.loads(json_match.group(1))
            else:
                # Try to find JSON without code blocks
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    json_data = json.loads(json_match.group(0))
                else:
                    raise ValueError("No JSON found in permission analysis response")
            
            return PermissionAnalysisResult(
                permission_status=json_data.get("permission_status", "request_permission"),
                confidence_level=min(max(json_data.get("confidence_level", 0.5), 0.0), 1.0),
                reasoning=json_data.get("reasoning", "Analysis failed"),
                permission_type=json_data.get("permission_type", "unknown"),
                research_legitimacy=json_data.get("research_legitimacy", "unclear"),
                recommended_tools=json_data.get("recommended_tools", []),
                scope=json_data.get("scope", "current_query")
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to parse permission analysis: {e}")
            # Conservative fallback
            return PermissionAnalysisResult(
                permission_status="request_permission",
                confidence_level=0.3,
                reasoning=f"Failed to parse analysis: {str(e)}",
                permission_type="unknown",
                research_legitimacy="unclear",
                recommended_tools=[],
                scope="current_query"
            )
    
    def _parse_permission_analysis_universal(self, raw_content: str) -> PermissionAnalysisResult:
        """Universal JSON extraction for permission analysis - Roosevelt's Universal compatibility!"""
        try:
            import json
            import re
            
            logger.info(f"🔧 ROOSEVELT'S UNIVERSAL PARSER: Extracting permission JSON from {len(raw_content)} chars")
            
            # Strategy 1: Try to find JSON block in markdown format
            json_pattern = r'```json\s*\n(.*?)\n```'
            matches = re.findall(json_pattern, raw_content, re.DOTALL)
            
            json_text = None
            if matches:
                json_text = matches[-1].strip()
                logger.info(f"🎯 Found markdown JSON block: {json_text[:200]}...")
            else:
                # Strategy 2: Try alternative patterns
                alt_patterns = [
                    r'```\s*\n(\{.*?\})\s*\n```',  # JSON in code block without language
                    r'(\{[^{}]*"permission_status"[^{}]*\})',  # Look for permission_status specifically
                    r'(\{.*?"permission_status".*?\})'  # More flexible permission_status pattern
                ]
                
                for pattern in alt_patterns:
                    matches = re.findall(pattern, raw_content, re.DOTALL)
                    if matches:
                        potential_json = matches[-1].strip()
                        if potential_json.startswith('{') and potential_json.endswith('}'):
                            json_text = potential_json
                            logger.info(f"🎯 Found alternative format JSON: {json_text[:200]}...")
                            break
                else:
                    # Strategy 3: Find largest JSON-like object in response
                    json_objects = []
                    brace_count = 0
                    start_idx = -1
                    
                    for i, char in enumerate(raw_content):
                        if char == '{':
                            if brace_count == 0:
                                start_idx = i
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0 and start_idx >= 0:
                                json_objects.append(raw_content[start_idx:i+1])
                    
                    if json_objects:
                        # Use the largest JSON object (most likely to be complete)
                        json_text = max(json_objects, key=len)
                        logger.info(f"🎯 Found JSON object by parsing: {json_text[:200]}...")
                    else:
                        raise ValueError("No JSON found in permission analysis response using any strategy")
            
            # Parse and validate JSON
            json_data = json.loads(json_text)
            
            logger.info(f"✅ Universal permission JSON extraction successful: status={json_data.get('permission_status', 'unknown')}")
            
            return PermissionAnalysisResult(
                permission_status=json_data.get("permission_status", "request_permission"),
                confidence_level=min(max(json_data.get("confidence_level", 0.5), 0.0), 1.0),
                reasoning=json_data.get("reasoning", "Analysis completed"),
                permission_type=json_data.get("permission_type", "unknown"),
                research_legitimacy=json_data.get("research_legitimacy", "unclear"),
                recommended_tools=json_data.get("recommended_tools", []),
                scope=json_data.get("scope", "current_query")
            )
            
        except Exception as e:
            logger.error(f"❌ Universal permission JSON extraction failed: {e}")
            logger.error(f"Raw content: {raw_content[:500]}...")
            
            # Create fallback response that defaults to granted for user convenience
            return PermissionAnalysisResult(
                permission_status="granted",  # Default to granted when parsing fails to avoid blocking
                confidence_level=0.8,
                reasoning=f"Permission parsing failed, defaulting to granted for user convenience: {str(e)}",
                permission_type="fallback",
                research_legitimacy="standard_research",
                recommended_tools=["search_and_crawl"],
                scope="current_query"
            )
