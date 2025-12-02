"""
Formatting Detection Utilities

Intelligent LLM-based detection of when research data would benefit from structured formatting.
Analyzes user intent and data characteristics to recommend routing to data formatting agent.
"""

import logging
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config.settings import settings

logger = logging.getLogger(__name__)


async def detect_formatting_need(user_query: str, research_response: str) -> Optional[str]:
    """
    Detect if research results would benefit from structured formatting
    
    Uses LLM to intelligently analyze:
    1. User preferences (explicit format requests, conversational context)
    2. Data characteristics (comparative, statistical, hierarchical)
    3. Value assessment (would formatting significantly improve comprehension?)
    
    Args:
        user_query: Original user query
        research_response: Research findings/response
        
    Returns:
        "data_formatting" if formatting recommended, None otherwise
    """
    try:
        # Skip if no substantial research data
        if not research_response or len(research_response) < 100:
            logger.debug("Skipping formatting detection: insufficient data")
            return None
            
        # Use LLM to intelligently detect formatting needs
        formatting_decision = await _llm_analyze_formatting_need(user_query, research_response)
        
        if formatting_decision:
            logger.info(f"📊 Formatting recommended: {formatting_decision}")
            return "data_formatting"
        
        return None
        
    except Exception as e:
        logger.error(f"Format detection failed: {e}")
        return None


async def _llm_analyze_formatting_need(user_query: str, research_findings: str) -> Optional[str]:
    """Use LLM to analyze if research data would benefit from structured formatting"""
    try:
        # Enhanced user-preference-aware formatting analysis prompt
        analysis_prompt = f"""You are an Intelligent Formatting Specialist. Balance user preferences with data readability.

USER QUERY: "{user_query}"
RESEARCH FINDINGS: {research_findings[:1000]}...

DECISION FRAMEWORK:

1. USER PREFERENCE ANALYSIS:
- Did user explicitly request "no table/formatting/chart"?
- Did user ask for specific format (paragraph, explanation, summary)?
- Is this a conversational query suggesting narrative preference?
- Context: Follow-up question where formatting might be jarring?

2. DATA VALUE ASSESSMENT:
- Would structured format SIGNIFICANTLY improve comprehension?
- Is this genuinely comparative/statistical data with multiple dimensions?
- Does tabular format provide clear advantage over prose?
- Is there sufficient structured data to warrant formatting?

3. APPROPRIATENESS ANALYSIS:
- Table: Multi-dimensional comparisons, statistics, rankings
- Chart: Trends, relationships, numerical patterns
- Organize: Complex hierarchical or categorized data
- Text: Simple answers, conversational responses, narratives

DECISION RULES:
- EXPLICIT USER PREFERENCE OVERRIDES data characteristics
- Only volunteer formatting when it provides SIGNIFICANT value
- Respect conversational context and user intent
- Prioritize user experience over data perfectionism
- **CONTENT ANALYSIS QUERIES**: Queries asking "what does X say" or "key insights from Y" prefer narrative analysis over data formatting
- **COMPARATIVE QUERIES**: Queries asking "compare A vs B" or "which has more X" benefit from structured formatting

EXAMPLES:
- "Compare debt by country" + comparative data → TABLE
- "Create a timeline for the history" + historical data → TIMELINE
- "Tell me about debt, no tables please" + any data → NO
- "Who has the most debt?" + simple data → NO (overkill)
- "Explain the situation" + any data → NO (narrative preferred)
- "What does [book/document] say?" + content analysis → NO (content analysis, not data formatting)
- "What are the key insights from [source]?" + analysis → NO (narrative analysis preferred)
- "Summarize the content of [document]" + content → NO (narrative summary preferred)

RESPOND WITH ONLY:
- "TABLE" if data significantly benefits from table AND user hasn't declined
- "CHART" if data benefits from visualization AND user hasn't declined  
- "TIMELINE" if historical/chronological data would benefit from timeline format
- "ORGANIZE" if data needs structure AND user hasn't declined
- "NO" if user prefers text OR formatting provides minimal value

Response:"""

        # Use fast model for quick decision
        llm = ChatOpenAI(
            model=settings.FAST_MODEL,
            openai_api_key=settings.OPENROUTER_API_KEY,
            openai_api_base=settings.OPENROUTER_BASE_URL,
            temperature=0.1,
            max_tokens=10
        )
        
        response = await llm.ainvoke([
            SystemMessage(content="You are a data formatting specialist. Respond with only one word: TABLE, CHART, TIMELINE, ORGANIZE, or NO."),
            HumanMessage(content=analysis_prompt)
        ])
        
        decision = response.content.strip().upper()
        
        # Map LLM decision to routing
        if decision in ["TABLE", "CHART", "TIMELINE", "ORGANIZE"]:
            logger.info(f"🧠 LLM formatting decision: {decision}")
            return decision.lower()
        
        logger.info("🧠 LLM formatting decision: NO formatting needed")
        return None
        
    except Exception as e:
        logger.error(f"LLM formatting analysis failed: {e}")
        return None





