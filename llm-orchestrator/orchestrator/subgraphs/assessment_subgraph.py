"""
Assessment Subgraph

Reusable subgraph for LLM-based quality assessment with structured output.
Evaluates whether research results are sufficient to answer a query.

Can be used by:
- Full Research Agent (web assessment)
- Knowledge Builder Agent (research quality checks)
- Project agents (deliverable assessment)

Inputs:
- query: The original query/question
- results: Content to assess (truncated to ~4000 chars)
- context: Optional context (e.g., "web search results", "local documents")
- domain: Domain context (e.g., "research", "web_research", "project")

Outputs:
- assessment: ResearchAssessmentResult as dict
- sufficient: bool
- confidence: float
- reasoning: str
- missing_info: List[str]
- has_relevant_info: bool
"""

import logging
import json
import re
from typing import Dict, Any

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from orchestrator.models import ResearchAssessmentResult
from orchestrator.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


# Use Dict[str, Any] for compatibility with any agent state
AssessmentSubgraphState = Dict[str, Any]


async def assessment_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LLM-based quality assessment"""
    try:
        query = state.get("query", "")
        results = state.get("results", "")
        context = state.get("context", "results")
        domain = state.get("domain", "general")
        
        # Handle different result formats
        if isinstance(results, str):
            results_text = results
        elif isinstance(results, dict):
            # Extract content from structured results
            if "content" in results:
                results_text = results["content"]
            elif "search_results" in results:
                results_text = results["search_results"]
            else:
                results_text = str(results)
        else:
            results_text = str(results)
        
        # Truncate results for prompt (limit to ~4000 chars)
        results_text = results_text[:4000] if len(results_text) > 4000 else results_text
        
        logger.info(f"Assessing {context} quality for query: {query[:100]}")
        
        # Build assessment prompt
        assessment_prompt = f"""Assess the quality and sufficiency of these {context} for answering the user's query.

USER QUERY: {query}

{context.upper()}:
{results_text if results_text else "No results provided."}

Evaluate:
1. Do the results contain relevant information?
2. Is there enough detail to answer the query comprehensively?
3. What information is missing (if any)?

STRUCTURED OUTPUT REQUIRED - Respond with ONLY valid JSON matching this exact schema:
{{
    "sufficient": boolean,
    "has_relevant_info": boolean,
    "missing_info": ["list", "of", "specific", "gaps"],
    "confidence": number (0.0-1.0),
    "reasoning": "brief explanation of assessment"
}}"""
        
        # Get LLM with proper context
        base_agent = BaseAgent("assessment_subgraph")
        llm = base_agent._get_llm(temperature=0.7, state=state)
        datetime_context = base_agent._get_datetime_context()
        
        # Build messages including conversation history for context
        assessment_messages = [
            SystemMessage(content="You are a research quality assessor. Always respond with valid JSON."),
            SystemMessage(content=datetime_context)
        ]
        
        # Include conversation history if available (critical for follow-up queries)
        conversation_messages = state.get("messages", [])
        if conversation_messages:
            assessment_messages.extend(conversation_messages)
        
        assessment_messages.append(HumanMessage(content=assessment_prompt))
        
        response = await llm.ainvoke(assessment_messages)
        
        logger.info("Tool used: LLM assessment (quality assessment)")
        
        return {
            "llm_response": response.content,
            "raw_assessment": response.content
        }
        
    except Exception as e:
        logger.error(f"Assessment node error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "llm_response": "",
            "raw_assessment": "",
            "error": str(e)
        }


async def parse_and_validate_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Parse and validate assessment with Pydantic"""
    try:
        raw_assessment = state.get("raw_assessment", "")
        query = state.get("query", "")
        
        if not raw_assessment:
            logger.warning("No assessment response to parse")
            return {
                "assessment": {
                    "sufficient": False,
                    "has_relevant_info": False,
                    "missing_info": [],
                    "confidence": 0.0,
                    "reasoning": "No assessment response received"
                },
                "sufficient": False,
                "confidence": 0.0,
                "reasoning": "No assessment response received",
                "missing_info": [],
                "has_relevant_info": False
            }
        
        # Parse response with Pydantic validation
        try:
            text = raw_assessment.strip()
            
            # Extract JSON from markdown code blocks
            if '```json' in text:
                m = re.search(r'```json\s*\n([\s\S]*?)\n```', text)
                if m:
                    text = m.group(1).strip()
            elif '```' in text:
                m = re.search(r'```\s*\n([\s\S]*?)\n```', text)
                if m:
                    text = m.group(1).strip()
            
            # Extract JSON object
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                text = json_match.group(0)
            
            # Validate with Pydantic
            assessment = ResearchAssessmentResult.parse_raw(text)
            
            logger.info(f"Assessment parsed: sufficient={assessment.sufficient}, confidence={assessment.confidence}, relevant={assessment.has_relevant_info}")
            
            return {
                "assessment": {
                    "sufficient": assessment.sufficient,
                    "has_relevant_info": assessment.has_relevant_info,
                    "missing_info": assessment.missing_info,
                    "confidence": assessment.confidence,
                    "reasoning": assessment.reasoning
                },
                "sufficient": assessment.sufficient,
                "confidence": assessment.confidence,
                "reasoning": assessment.reasoning,
                "missing_info": assessment.missing_info,
                "has_relevant_info": assessment.has_relevant_info
            }
            
        except (json.JSONDecodeError, ValidationError, Exception) as e:
            logger.warning(f"Failed to parse assessment with Pydantic: {e}")
            logger.debug(f"Raw assessment text: {raw_assessment[:500]}")
            
            # Fallback: return default assessment
            return {
                "assessment": {
                    "sufficient": False,
                    "has_relevant_info": True,
                    "missing_info": [],
                    "confidence": 0.5,
                    "reasoning": f"Assessment parsing failed: {str(e)}"
                },
                "sufficient": False,
                "confidence": 0.5,
                "reasoning": f"Assessment parsing failed: {str(e)}",
                "missing_info": [],
                "has_relevant_info": True
            }
        
    except Exception as e:
        logger.error(f"Parse and validate node error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "assessment": {
                "sufficient": False,
                "has_relevant_info": False,
                "missing_info": [],
                "confidence": 0.0,
                "reasoning": f"Assessment error: {str(e)}"
            },
            "sufficient": False,
            "confidence": 0.0,
            "reasoning": f"Assessment error: {str(e)}",
            "missing_info": [],
            "has_relevant_info": False,
            "error": str(e)
        }


def build_assessment_subgraph(checkpointer) -> StateGraph:
    """
    Build assessment subgraph for LLM-based quality assessment
    
    This subgraph evaluates whether research results are sufficient to answer a query.
    
    Expected state inputs:
    - query: str - The original query/question
    - results: str | dict - Content to assess (will be truncated to ~4000 chars)
    - context: str (optional) - Context description (e.g., "web search results", "local documents")
    - domain: str (optional) - Domain context (default: "general")
    - messages: List (optional) - Conversation history for context-aware assessment
    - metadata: Dict[str, Any] (optional) - Metadata for checkpointing
    
    Returns state with:
    - assessment: Dict[str, Any] - ResearchAssessmentResult as dict
    - sufficient: bool - Whether results are sufficient
    - confidence: float - Confidence in assessment (0.0-1.0)
    - reasoning: str - Explanation of assessment
    - missing_info: List[str] - Specific information gaps
    - has_relevant_info: bool - Whether results contain relevant information
    """
    subgraph = StateGraph(Dict[str, Any])
    
    # Add nodes
    subgraph.add_node("assessment", assessment_node)
    subgraph.add_node("parse_and_validate", parse_and_validate_node)
    
    # Set entry point
    subgraph.set_entry_point("assessment")
    
    # Linear flow: assess -> parse
    subgraph.add_edge("assessment", "parse_and_validate")
    subgraph.add_edge("parse_and_validate", END)
    
    return subgraph.compile(checkpointer=checkpointer)







