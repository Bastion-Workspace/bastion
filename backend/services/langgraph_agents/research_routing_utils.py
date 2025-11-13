"""
Research Routing Utils - Roosevelt's Smart Routing Utilities
Extracted from clean_research_agent to maintain 500-line limit
"""

import logging
from typing import Dict, Any, Optional
from models.agent_response_models import ResearchTaskResult, TaskStatus

logger = logging.getLogger(__name__)


async def detect_formatting_request(state: Dict[str, Any], structured_result: ResearchTaskResult, get_latest_user_message_func) -> Optional[str]:
    """ROOSEVELT'S LLM-BASED FORMATTING DETECTION: Let AI decide if data should be formatted"""
    try:
        # Only suggest formatting for completed research with data
        if structured_result.task_status != TaskStatus.COMPLETE:
            return None
            
        # Get original user query and research findings
        user_query = get_latest_user_message_func(state)
        research_findings = structured_result.findings if structured_result.findings else ""
        
        # Skip if no substantial research data
        if len(research_findings) < 100:  # Too little data to format
            return None
            
        # Use LLM to intelligently detect formatting needs
        formatting_decision = await _llm_analyze_formatting_need(user_query, research_findings)
        
        if formatting_decision:
            logger.info(f"📊 LLM FORMATTING DETECTED: {formatting_decision}")
            return "data_formatting"
        
        return None
    except Exception as e:
        logger.error(f"❌ Format detection failed: {e}")
        return None


async def _llm_analyze_formatting_need(user_query: str, research_findings: str) -> Optional[str]:
    """Use LLM to analyze if research data would benefit from structured formatting"""
    try:
        # Import here to avoid circular dependencies
        from services.chat_service import ChatService
        from config import settings
        
        chat_service = ChatService()
        if not chat_service.openai_client:
            await chat_service.initialize()
        
        # Enhanced user-preference-aware formatting analysis prompt
        analysis_prompt = f"""You are Roosevelt's Intelligent Formatting Specialist. Balance user preferences with data readability.

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
        fast_model = settings.FAST_MODEL or "anthropic/claude-3.5-haiku"
        
        response = await chat_service.openai_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a data formatting specialist. Respond with only one word: TABLE, CHART, TIMELINE, ORGANIZE, or NO."},
                {"role": "user", "content": analysis_prompt}
            ],
            model=fast_model,
            temperature=0.1,
            max_tokens=10
        )
        
        decision = response.choices[0].message.content.strip().upper()
        
        # Map LLM decision to routing
        if decision in ["TABLE", "CHART", "TIMELINE", "ORGANIZE"]:
            logger.info(f"🧠 LLM DECISION: {decision} formatting recommended")
            return decision.lower()
        
        logger.info(f"🧠 LLM DECISION: NO formatting needed")
        return None
        
    except Exception as e:
        logger.error(f"❌ LLM formatting analysis failed: {e}")
        return None
