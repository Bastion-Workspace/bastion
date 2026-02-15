"""
Data Formatting Subgraph

Reusable subgraph for transforming research data into structured formats (tables, charts, organized data).
Can be used by:
- Full Research Agent (formatting research results)
- Any agent needing structured data formatting

Inputs:
- query: The formatting request or original query
- messages: Conversation history for context
- metadata: Optional metadata (user_id, etc.)

Outputs:
- formatted_output: The formatted content (markdown tables, lists, etc.)
- format_type: Type of formatting applied
- task_status: Completion status
- confidence_level: Confidence in formatting quality
"""

import logging
import json
import re
from typing import Dict, Any
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from orchestrator.agents.base_agent import BaseAgent, TaskStatus

logger = logging.getLogger(__name__)


# Use Dict[str, Any] for compatibility with any agent state
DataFormattingSubgraphState = Dict[str, Any]


def _build_formatting_prompt(user_request: str, conversation_context: str) -> str:
    """Build data formatting prompt with conversation context"""
    
    return f"""You are a Data Formatting Specialist - an expert at transforming research data into organized, readable formats.

**MISSION**: Transform existing research data into the requested format (tables, lists, structured data) using conversation context.

**USER REQUEST**: {user_request}

**CONVERSATION CONTEXT**:
{conversation_context}

**STRUCTURED OUTPUT REQUIREMENT**:

You MUST respond with valid JSON matching this schema:
{{
    "task_status": "complete|incomplete|error",
    "formatted_output": "Your formatted content (markdown tables, lists, etc.)",
    "format_type": "markdown_table|markdown_list|structured_text|comparative_analysis|timeline|chronological_timeline",
    "confidence_level": 0.9,
    "data_sources_used": ["source1", "source2"],
    "formatting_notes": "Optional notes about the formatting process"
}}

**FORMATTING EXPERTISE**:
- **MARKDOWN TABLES**: Perfect table formatting with proper alignment
- **TIMELINE FORMATTING**: Chronological organization of historical events
- **DATA ORGANIZATION**: Logical grouping and categorization
- **REGIONAL BREAKDOWN**: Organize by geographic regions when relevant
- **COMPARATIVE ANALYSIS**: Clear side-by-side comparisons
- **STRUCTURED LISTS**: Bullet points, numbered lists, nested organization

**FORMATTING REQUIREMENTS**:
- **USE PROPER MARKDOWN**: Clean table syntax with | separators
- **CLEAR HEADERS**: Descriptive column/row headers
- **LOGICAL GROUPING**: Group related data together
- **READABLE FORMAT**: Proper spacing and alignment
- **COMPREHENSIVE**: Include ALL relevant data from context
- **ACCURATE**: Only use data that was actually provided in the research

**TABLE FORMATTING EXAMPLE**:
```markdown
| Nation/Region | State-Sponsored Messaging | General Public Sentiment | Key Differences |
|---------------|---------------------------|--------------------------|-----------------|
| Russia        | Pro-invasion, justification | Mixed, some opposition | State control vs. diverse views |
| EU Countries  | Support for Ukraine | Largely supportive | Aligned messaging |
```

**CRITICAL TABLE FORMATTING RULES**:
- Separator row MUST use exactly 3+ dashes per column: `|----------|----------|`
- NEVER use long dashes that span the entire column width
- Each column separator should be: `|` followed by 3+ dashes, ending with `|`
- Example CORRECT: `|------|------|` or `|-------|-------|`
- Example WRONG: `|-------------------------------------------------------------|--------------------------------|`

**TIMELINE FORMATTING EXAMPLE**:
```markdown
# Timeline: History of Tequila

## Pre-Columbian Era (Before 1500s)
- **Ancient Times**: Indigenous peoples ferment agave to create pulque
- **Aztec Empire**: Pulque becomes sacred ceremonial drink

## Spanish Colonial Period (1500s-1800s)
- **1521**: Spanish conquistadors arrive in Mexico
- **1600s**: Spanish introduce distillation techniques
- **1700s**: First proto-tequila distilleries in Jalisco

## Modern Era (1800s-Present)
- **1873**: Don Cenobio Sauza exports first tequila to US
- **1994**: NAFTA increases international trade
```

**JSON RESPONSE EXAMPLE**:
```json
{{
    "task_status": "complete",
    "formatted_output": "| Location | Type | Description |\\n|----------|------|-------------|\\n| Sonnenberg Gardens | Garden | Historic estate |\\n| CMAC | Venue | Concert amphitheater |",
    "format_type": "markdown_table",
    "confidence_level": 0.95,
    "data_sources_used": ["Research findings"],
    "formatting_notes": "Organized by location and type"
}}
```

**CRITICAL**:
1. **STRUCTURED JSON ONLY** - No plain text responses!
2. **Use actual data** - Only data provided in conversation context
3. **Proper escaping** - Escape newlines as \\\\n in JSON strings
4. **Valid JSON structure always!**
"""


async def prepare_formatting_context_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare conversation context for formatting"""
    try:
        logger.info("Preparing conversation context for data formatting...")
        
        messages = state.get("messages", [])
        
        # Extract conversation context
        if not messages:
            conversation_context = "No previous conversation context available."
        else:
            # Get last few messages for context
            recent_messages = messages[-5:] if len(messages) > 5 else messages
            
            context_parts = []
            for i, msg in enumerate(recent_messages):
                if hasattr(msg, 'content'):
                    role = "ASSISTANT" if hasattr(msg, 'type') and msg.type == "ai" else "USER"
                    content = msg.content
                    context_parts.append(f"{i+1}. {role}: {content}")
            
            conversation_context = "\n".join(context_parts)
        
        # Build formatting prompt
        query = state.get("query", "")
        system_prompt = _build_formatting_prompt(query, conversation_context)
        
        return {
            "conversation_context": conversation_context,
            "system_prompt": system_prompt,
            # ✅ CRITICAL: Preserve all state
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }
        
    except Exception as e:
        logger.error(f"Failed to prepare context: {e}")
        return {
            "conversation_context": "Error extracting conversation context.",
            "system_prompt": "",
            "error": str(e),
            # ✅ CRITICAL: Preserve even on error
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }


async def format_data_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Format data using LLM with structured output"""
    try:
        logger.info("Formatting data with structured output...")
        
        query = state.get("query", "")
        system_prompt = state.get("system_prompt", "")
        
        if not system_prompt:
            raise ValueError("System prompt not prepared")
        
        # Get LLM with low temperature for consistent formatting
        base_agent = BaseAgent("data_formatting_subgraph")
        llm = base_agent._get_llm(temperature=0.1, state=state)
        datetime_context = base_agent._get_datetime_context()
        
        # Build messages including conversation history
        messages_list = state.get("messages", [])
        format_messages = [
            SystemMessage(content=system_prompt),
            SystemMessage(content=datetime_context)
        ]
        
        # Include conversation history if available
        if messages_list:
            format_messages.extend(messages_list)
        
        format_messages.append(HumanMessage(content=query))
        
        # Call LLM
        start_time = datetime.now()
        response = await llm.ainvoke(format_messages)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Parse structured response
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        # Extract JSON from response
        text = response_content.strip()
        
        # Extract JSON from markdown code blocks
        if '```json' in text:
            match = re.search(r'```json\s*\n([\s\S]*?)\n```', text)
            if match:
                text = match.group(1).strip()
        elif '```' in text:
            match = re.search(r'```\s*\n([\s\S]*?)\n```', text)
            if match:
                text = match.group(1).strip()
        
        # Extract JSON object
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            text = json_match.group(0)
        
        # Parse JSON
        try:
            structured_result = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            structured_result = {
                "task_status": "error",
                "formatted_output": "",
                "format_type": "error",
                "confidence_level": 0.0,
                "data_sources_used": [],
                "formatting_notes": f"JSON parsing failed: {str(e)}"
            }
        
        # Unescape JSON newlines for proper markdown display
        formatted_output = structured_result.get("formatted_output", "")
        if formatted_output:
            formatted_output = formatted_output.replace("\\n", "\n")
        
        # Build result
        format_type = structured_result.get("format_type", "structured_text")
        task_status = structured_result.get("task_status", "complete")
        
        logger.info(f"Data formatting completed in {processing_time:.2f}s: {format_type}")
        
        return {
            "formatted_output": formatted_output,
            "format_type": format_type,
            "task_status": task_status,
            "confidence_level": structured_result.get("confidence_level", 0.9),
            "data_sources_used": structured_result.get("data_sources_used", []),
            "formatting_notes": structured_result.get("formatting_notes", ""),
            "processing_time": processing_time,
            # ✅ CRITICAL: Preserve all state
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }
        
    except Exception as e:
        logger.error(f"Data formatting failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "formatted_output": "",
            "format_type": "error",
            "task_status": "error",
            "confidence_level": 0.0,
            "data_sources_used": [],
            "formatting_notes": f"Formatting error: {str(e)}",
            "error": str(e),
            # ✅ CRITICAL: Preserve even on error
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }


def build_data_formatting_subgraph(checkpointer) -> StateGraph:
    """
    Build data formatting subgraph for transforming research data into structured formats
    
    This subgraph formats data into tables, lists, timelines, and other organized structures.
    
    Expected state inputs:
    - query: str - The formatting request or original query
    - messages: List (optional) - Conversation history for context
    - metadata: Dict[str, Any] (optional) - Metadata for checkpointing and user model selection
    
    Returns state with:
    - formatted_output: str - The formatted content (markdown tables, lists, etc.)
    - format_type: str - Type of formatting applied (markdown_table, timeline, etc.)
    - task_status: str - Completion status
    - confidence_level: float - Confidence in formatting quality (0.0-1.0)
    - data_sources_used: List[str] - Sources of data used
    - formatting_notes: str - Optional notes about the formatting process
    """
    subgraph = StateGraph(dict)
    
    # Add nodes
    subgraph.add_node("prepare_context", prepare_formatting_context_node)
    subgraph.add_node("format_data", format_data_node)
    
    # Set entry point
    subgraph.set_entry_point("prepare_context")
    
    # Linear flow: prepare_context -> format_data -> END
    subgraph.add_edge("prepare_context", "format_data")
    subgraph.add_edge("format_data", END)
    
    return subgraph.compile(checkpointer=checkpointer)

