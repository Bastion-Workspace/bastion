"""
Universal Gap Analysis Subgraph

Reusable subgraph for analyzing information gaps in research results.
Can be used by any agent that needs to identify missing information and determine next steps.

Inputs:
- query: The original query/question
- results: Current results/research findings (can be from any source)
- context: Optional additional context (conversation history, domain, etc.)

Outputs:
- gap_analysis: Structured gap analysis with severity, missing entities, suggested queries
- identified_gaps: List of specific gaps to fill
- needs_web_search: Whether web search would help
- gap_severity: minor | moderate | severe
"""

import logging
import json
import re
from typing import Dict, Any, Optional, List

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage

from orchestrator.models import ResearchGapAnalysis
from orchestrator.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


# Use Dict[str, Any] for compatibility with any agent state
GapAnalysisSubgraphState = Dict[str, Any]


async def analyze_gaps_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Universal gap analysis node
    
    Analyzes what information is missing from current results to fully answer the query.
    Works with any type of results (local documents, web search, mixed, etc.)
    """
    try:
        query = state.get("query", "")
        results = state.get("results", "")  # Flexible: can be search_results, content, findings, etc.
        context = state.get("context", "")  # Optional additional context
        domain = state.get("domain", "general")  # Optional domain context (research, project, etc.)
        
        logger.info(f"🔍 Performing universal gap analysis for query: {query[:100]}")
        
        # Build flexible prompt that works with any result type
        results_text = ""
        if isinstance(results, str):
            results_text = results
        elif isinstance(results, dict):
            # Handle structured results
            if "search_results" in results:
                results_text = results["search_results"]
            elif "content" in results:
                results_text = results["content"]
            elif "findings" in results:
                results_text = str(results["findings"])
            elif "combined_results" in results:
                results_text = results["combined_results"]
            else:
                results_text = str(results)
        elif isinstance(results, list):
            results_text = "\n".join(str(r) for r in results)
        else:
            results_text = str(results)
        
        # Limit results size for prompt
        results_text = results_text[:8000] if len(results_text) > 8000 else results_text
        
        # Build context section if provided
        context_section = ""
        if context:
            context_section = f"\n\nADDITIONAL CONTEXT:\n{context[:2000]}"
        
        gap_prompt = f"""Analyze what information is missing from the current results to fully answer the query.

DOMAIN: {domain}

USER QUERY: {query}

CURRENT RESULTS:
{results_text if results_text else "No results found yet."}{context_section}

Identify:
1. Specific missing entities, people, facts, concepts, or data points
2. Targeted search queries that could fill those specific gaps
3. Whether web search, local search, or other sources would be beneficial
4. How severe the information gaps are (minor = nice to have, moderate = important but not critical, severe = critical missing information)

STRUCTURED OUTPUT REQUIRED - Respond with ONLY valid JSON matching this exact schema:
{{
    "missing_entities": ["specific", "missing", "entities", "facts", "or", "concepts"],
    "suggested_queries": ["targeted search query 1", "targeted search query 2"],
    "needs_web_search": boolean,
    "needs_local_search": boolean,
    "gap_severity": "minor" | "moderate" | "severe",
    "reasoning": "explanation of gaps and how to fill them",
    "confidence": number (0.0-1.0)
}}"""
        
        # Get LLM with proper context
        base_agent = BaseAgent("gap_analysis_subgraph")
        llm = base_agent._get_llm(temperature=0.7, state=state)
        datetime_context = base_agent._get_datetime_context()
        
        gap_messages = [
            SystemMessage(content="You are a universal information gap analyst. Analyze what's missing from any type of research results. Always respond with valid JSON."),
            SystemMessage(content=datetime_context)
        ]
        
        # Include conversation history if available
        conversation_messages = state.get("messages", [])
        if conversation_messages:
            gap_messages.extend(conversation_messages)
        
        gap_messages.append(HumanMessage(content=gap_prompt))
        
        response = await llm.ainvoke(gap_messages)
        
        # Parse response
        identified_gaps = []
        try:
            text = response.content.strip()
            if '```json' in text:
                m = re.search(r'```json\s*\n([\s\S]*?)\n```', text)
                if m:
                    text = m.group(1).strip()
            elif '```' in text:
                m = re.search(r'```\s*\n([\s\S]*?)\n```', text)
                if m:
                    text = m.group(1).strip()
            
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                text = json_match.group(0)
            
            gap_analysis = ResearchGapAnalysis.parse_raw(text)
            
            # Extract identified gaps (prefer suggested queries, fallback to missing entities)
            if gap_analysis.suggested_queries:
                identified_gaps = gap_analysis.suggested_queries
            elif gap_analysis.missing_entities:
                identified_gaps = gap_analysis.missing_entities
            else:
                identified_gaps = [query]  # Fallback to original query
            
            logger.info(f"✅ Gap analysis complete: severity={gap_analysis.gap_severity}, web_search={gap_analysis.needs_web_search}, gaps={len(identified_gaps)}")
            
            return {
                "gap_analysis": {
                    "analysis": response.content,
                    "has_gaps": True,
                    "needs_web_search": gap_analysis.needs_web_search,
                    "needs_local_search": gap_analysis.needs_local_search if gap_analysis.needs_local_search is not None else False,
                    "gap_severity": gap_analysis.gap_severity,
                    "confidence": gap_analysis.confidence if gap_analysis.confidence is not None else 0.7,
                    "missing_entities": gap_analysis.missing_entities,
                    "suggested_queries": gap_analysis.suggested_queries,
                    "reasoning": gap_analysis.reasoning
                },
                "identified_gaps": identified_gaps,
                # CRITICAL: Preserve metadata for user model selection
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", [])
            }
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"⚠️ Failed to parse gap analysis: {e}")
            identified_gaps = [query]
            return {
                "gap_analysis": {
                    "analysis": response.content if 'response' in locals() else "",
                    "has_gaps": True,
                    "needs_web_search": True,
                    "needs_local_search": True,
                    "gap_severity": "severe",
                    "confidence": 0.3,
                    "missing_entities": [],
                    "suggested_queries": identified_gaps,
                    "reasoning": f"Gap analysis parsing failed: {str(e)}"
                },
                "identified_gaps": identified_gaps,
                # CRITICAL: Preserve metadata for user model selection
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", [])
            }
        
    except Exception as e:
        logger.error(f"❌ Gap analysis error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "gap_analysis": {
                "has_gaps": False,
                "needs_web_search": False,
                "needs_local_search": False,
                "gap_severity": "minor",
                "confidence": 0.0,
                "reasoning": f"Gap analysis failed: {str(e)}"
            },
            "identified_gaps": [],
            # CRITICAL: Preserve metadata for user model selection
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", [])
        }


def build_gap_analysis_subgraph(checkpointer) -> StateGraph:
    """
    Build universal gap analysis subgraph
    
    This subgraph can be used by any agent to analyze information gaps.
    
    Expected state inputs:
    - query: str - The original query/question
    - results: str | dict | list - Current results/research findings
    - context: str (optional) - Additional context
    - domain: str (optional) - Domain context (default: "general")
    - messages: List (optional) - Conversation history
    
    Returns state with:
    - gap_analysis: dict - Structured gap analysis
    - identified_gaps: List[str] - Specific gaps to fill
    """
    subgraph = StateGraph(Dict[str, Any])
    
    # Add gap analysis node
    subgraph.add_node("analyze_gaps", analyze_gaps_node)
    
    # Set entry point
    subgraph.set_entry_point("analyze_gaps")
    
    # Gap analysis always completes (no conditional routing within subgraph)
    subgraph.add_edge("analyze_gaps", END)
    
    return subgraph.compile(checkpointer=checkpointer)

