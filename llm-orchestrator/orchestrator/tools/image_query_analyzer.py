"""
Image Query Analyzer - LLM-based intelligent parsing of image search queries
Uses LLM to extract structured search parameters from natural language queries
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
import re

from orchestrator.agents.base_agent import BaseAgent
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)


async def analyze_image_query(
    query: str,
    user_id: str = "system",
    metadata: Optional[Dict[str, Any]] = None,
    shared_memory: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Use LLM to intelligently parse an image search query and extract structured parameters.
    
    Args:
        query: Natural language query (e.g., "Can you show me Dilbert from April 16th, 1989?")
        user_id: User ID for LLM model selection
        metadata: Optional metadata dict for model preferences
        shared_memory: Optional shared memory dict (may contain user_chat_model)
        
    Returns:
        Dict with:
        - query: Cleaned search query for vector search
        - series: Comic/series name if detected (e.g., "Dilbert")
        - author: Actual author name if mentioned (e.g., "Scott Adams")
        - date: Parsed date in YYYY-MM-DD format if found
        - image_type: Detected image type (comic, artwork, meme, etc.)
    """
    try:
        logger.info(f"Analyzing image query with LLM: {query[:100]}")
        
        # Use BaseAgent for LLM access
        base_agent = BaseAgent("image_query_analyzer")
        
        # Merge user_chat_model from shared_memory into metadata if not already in metadata
        # This ensures _get_llm can find it in the standard location
        merged_metadata = (metadata or {}).copy()
        if shared_memory and "user_chat_model" in shared_memory and "user_chat_model" not in merged_metadata:
            merged_metadata["user_chat_model"] = shared_memory.get("user_chat_model")
        if shared_memory and "user_fast_model" in shared_memory and "user_fast_model" not in merged_metadata:
            merged_metadata["user_fast_model"] = shared_memory.get("user_fast_model")
        
        minimal_state = {
            "metadata": merged_metadata,  # Use merged metadata with user_chat_model
            "user_id": user_id,
            "shared_memory": shared_memory or {}  # Include shared_memory for other context
        }
        llm = base_agent._get_llm(temperature=0.1, state=minimal_state)
        
        system_prompt = """You are an image search query analyzer. Your job is to extract structured search parameters from natural language queries about images, comics, artwork, etc.

CRITICAL DISTINCTIONS:
- **Series** = Comic strip name, book series, TV show, etc. (e.g., "Dilbert", "Calvin and Hobbes", "Garfield")
- **Author** = Creator/artist name (e.g., "Scott Adams", "Bill Watterson", "Jim Davis")
- **Date** = Publication or creation date (extract and normalize to YYYY-MM-DD format)

EXAMPLES:
Query: "Can you show me Dilbert from April 16th, 1989?"
→ series: "Dilbert", date: "1989-04-16", query: "Dilbert April 16 1989"

Query: "Find a Garfield comic"
→ series: "Garfield", query: "Garfield comic"

Query: "Show me artwork by Van Gogh"
→ author: "Van Gogh", image_type: "artwork", query: "artwork Van Gogh"

Query: "Display a meme about cats"
→ image_type: "meme", query: "meme cats"

Query: "Show me a screenshot from yesterday"
→ image_type: "screenshot", query: "screenshot" (date would be calculated separately)

Return ONLY valid JSON matching this schema:
{
  "query": "cleaned search query for vector search",
  "series": "series name or null",
  "author": "author/creator name or null",
  "date": "YYYY-MM-DD format or null",
  "image_type": "comic|artwork|meme|screenshot|medical|documentation|maps|other|null"
}"""

        prompt = f"Analyze this image search query and extract structured parameters:\n\n**QUERY**: {query}\n\nReturn ONLY valid JSON with the extracted parameters."

        try:
            # LangChain requires "title" and "description" at top level for structured output
            schema = {
                "title": "ImageQueryAnalysis",
                "description": "Structured analysis of an image search query",
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Cleaned search query for vector search"},
                    "series": {"type": ["string", "null"], "description": "Comic/series name if detected"},
                    "author": {"type": ["string", "null"], "description": "Author/creator name if mentioned"},
                    "date": {"type": ["string", "null"], "description": "Date in YYYY-MM-DD format if found"},
                    "image_type": {"type": ["string", "null"], "description": "Image type: comic, artwork, meme, screenshot, medical, documentation, maps, other"}
                },
                "required": ["query"]
            }
            structured_llm = llm.with_structured_output(schema)
            result = await structured_llm.ainvoke([{"role": "user", "content": prompt}])
            result_dict = result if isinstance(result, dict) else result.dict() if hasattr(result, 'dict') else result.model_dump()
        except Exception as e:
            logger.warning(f"Structured output failed, trying fallback: {e}")
            response = await llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ])
            content = response.content if hasattr(response, 'content') else str(response)
            result_dict = _parse_json_response(content)
        
        # Validate and normalize date format
        if result_dict.get("date"):
            date_str = result_dict["date"]
            # Try to parse and normalize to YYYY-MM-DD
            try:
                # Handle various date formats
                parsed_date = _parse_flexible_date(date_str)
                if parsed_date:
                    result_dict["date"] = parsed_date.strftime("%Y-%m-%d")
                else:
                    # If LLM returned a date but we can't parse it, set to None
                    logger.warning(f"Could not parse date from LLM response: {date_str}")
                    result_dict["date"] = None
            except Exception as e:
                logger.warning(f"Date parsing failed: {e}")
                result_dict["date"] = None
        
        # Ensure query is not empty
        if not result_dict.get("query") or len(result_dict.get("query", "").strip()) < 3:
            result_dict["query"] = query  # Fallback to original query
        
        logger.info(f"Query analyzed: series={result_dict.get('series')}, author={result_dict.get('author')}, date={result_dict.get('date')}, type={result_dict.get('image_type')}")
        
        return result_dict
        
    except Exception as e:
        logger.error(f"Image query analysis failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Return safe defaults
        return {
            "query": query,
            "series": None,
            "author": None,
            "date": None,
            "image_type": None
        }


def _parse_json_response(content: str) -> Dict[str, Any]:
    """Parse JSON response from LLM, handling markdown code blocks"""
    import json
    
    json_text = content.strip()
    
    # Remove markdown code blocks if present
    if '```json' in json_text:
        match = re.search(r'```json\s*\n(.*?)\n```', json_text, re.DOTALL)
        if match:
            json_text = match.group(1).strip()
    elif '```' in json_text:
        match = re.search(r'```\s*\n(.*?)\n```', json_text, re.DOTALL)
        if match:
            json_text = match.group(1).strip()
    
    # Extract JSON object
    json_match = re.search(r'\{.*\}', json_text, re.DOTALL)
    if json_match:
        json_text = json_match.group(0)
    
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON response: {e}")
        return {"query": content, "series": None, "author": None, "date": None, "image_type": None}


def _parse_flexible_date(date_str: str) -> Optional[datetime]:
    """Parse date from various formats and return datetime object"""
    if not date_str:
        return None
    
    date_str = date_str.strip()
    
    # Try standard formats first
    formats = [
        "%Y-%m-%d",
        "%m-%d-%Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%d/%m/%Y",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    # Try text date parsing (e.g., "April 16th, 1989")
    text_date_patterns = [
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b',
        r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b',
    ]
    month_map = {
        'january': '01', 'february': '02', 'march': '03', 'april': '04',
        'may': '05', 'june': '06', 'july': '07', 'august': '08',
        'september': '09', 'october': '10', 'november': '11', 'december': '12',
        'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
        'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
        'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
    }
    
    for pattern in text_date_patterns:
        match = re.search(pattern, date_str, re.IGNORECASE)
        if match:
            month_name = match.group(1).lower()
            day = match.group(2)
            year = match.group(3)
            if month_name in month_map:
                try:
                    normalized = f"{year}-{month_map[month_name]}-{day.zfill(2)}"
                    return datetime.strptime(normalized, "%Y-%m-%d")
                except ValueError:
                    continue
    
    return None
