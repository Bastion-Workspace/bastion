"""
Data Formatting Agent - Roosevelt's Table and Structure Specialist
Transforms research data into tables, charts, and organized formats
"""

import logging
import json
from typing import Dict, Any, List, Optional
from enum import Enum
from pydantic import ValidationError

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from .base_agent import BaseAgent
from models.agent_response_models import AgentType, DataFormattingResult, TaskStatus

logger = logging.getLogger(__name__)


class DataFormattingAgent(BaseAgent):
    """
    ROOSEVELT'S DATA FORMATTING SPECIALIST
    Transforms research findings into structured formats like tables, charts, and organized data
    """

    def __init__(self):
        super().__init__("data_formatting_agent")  # Use string, not enum
        logger.info("🔢 BULLY! Data Formatting Agent assembled and ready to organize!")

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process data formatting request using conversation context"""
        try:
            logger.info("🔢 Data Formatting Agent charging forward with table organization...")
            
            # Get the user's formatting request
            messages = state.get("messages", [])
            if not messages:
                return self._create_error_response("No messages found to process")
                
            user_request = messages[-1].content
            logger.info(f"📊 FORMATTING REQUEST: {user_request[:100]}...")

            # Get conversation context and previous research
            conversation_context = self._extract_conversation_context(state)
            
            # Execute data formatting
            formatting_result = await self._execute_data_formatting(state, user_request, conversation_context)
            
            # Update state with results
            state["agent_results"] = {
                "agent_type": "data_formatting",
                "structured_response": formatting_result,
                "task_status": formatting_result.get("task_status", "complete"),
                "confidence": formatting_result.get("confidence_level", 0.9)
            }
            # ROOSEVELT'S MARKDOWN UNESCAPING: Convert JSON escape sequences to proper newlines
            formatted_output = formatting_result.get("formatted_output", "")
            if formatted_output:
                # Unescape JSON newlines for proper markdown display
                formatted_output = formatted_output.replace("\\n", "\n")
            state["latest_response"] = formatted_output
            
            # ROOSEVELT'S PURE LANGGRAPH: Add formatting response to LangGraph state messages
            if formatted_output and formatted_output.strip():
                from langchain_core.messages import AIMessage
                state.setdefault("messages", []).append(AIMessage(content=formatted_output))
                logger.info(f"✅ FORMATTING AGENT: Added formatted response to LangGraph messages")
            
            state["is_complete"] = True
            
            logger.info("✅ Data formatting completed successfully!")
            return state
            
        except Exception as e:
            logger.error(f"❌ Data formatting failed: {e}")
            return self._create_error_response(f"Data formatting failed: {str(e)}")

    def _extract_conversation_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant data from conversation context and previous research"""
        try:
            # Get previous messages for context
            messages = state.get("messages", [])
            conversation_history = []
            
            for message in messages[:-1]:  # Exclude current request
                if hasattr(message, 'content'):
                    conversation_history.append({
                        "role": "assistant" if hasattr(message, 'type') and message.type == "ai" else "user",
                        "content": message.content  # Keep full content for proper data formatting
                    })
            
            # Get previous research results from shared memory
            shared_memory = state.get("shared_memory", {})
            previous_research = shared_memory.get("search_results", {})
            
            # ROOSEVELT'S COMPLETE DATA ACCESS: Get research findings from ALL possible locations
            # Check both shared_memory locations AND agent_results
            previous_agent_results = state.get("agent_results", {})
            
            # Extract complete research findings from shared memory (primary source)
            complete_research_findings = {}
            if "search_results" in shared_memory:
                search_results = shared_memory["search_results"]
                if "local" in search_results and "findings" in search_results["local"]:
                    complete_research_findings["local_findings"] = search_results["local"]["findings"]
            
            # Also check research_findings in shared memory
            if "research_findings" in shared_memory:
                complete_research_findings["research_findings"] = shared_memory["research_findings"]
            
            return {
                "conversation_history": conversation_history[-5:],  # Last 5 exchanges
                "previous_research": previous_research,
                "previous_agent_results": previous_agent_results,
                "shared_memory_keys": list(shared_memory.keys())
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to extract conversation context: {e}")
            return {}

    async def _execute_data_formatting(self, state: Dict[str, Any], user_request: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data formatting using LLM with conversation context"""
        try:
            logger.info("📊 Executing data formatting with conversation context...")
            
            # Build the formatting prompt
            system_prompt = self._build_data_formatting_prompt(user_request, context)
            
            # Get OpenAI-compatible tools (none needed for basic formatting)
            # Future: This is where we'd add chart/graph generation tools
            
            # Execute formatting with LLM
            logger.info("🤖 Calling OpenAI client for data formatting...")
            
            # Get configuration from BaseAgent (like other agents)
            chat_service = await self._get_chat_service()
            model_name = await self._get_model_name()
            
            client = ChatOpenAI(
                model=model_name,
                temperature=0.1,  # Low temperature for consistent formatting
                openai_api_key=chat_service.openai_client.api_key,
                openai_api_base=str(chat_service.openai_client.base_url)  # Convert URL to string
            )
            
            response = await client.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_request)
            ])
            
            # Parse the response content using Roosevelt's Type Safety
            response_content = response.content if hasattr(response, 'content') else str(response)
            logger.info(f"🎯 ROOSEVELT'S STRUCTURED PARSER: Processing response length: {len(response_content)}")
            
            # First, try to parse as structured JSON with validation
            try:
                # Clean the response content (remove markdown code blocks if present)
                cleaned_content = self._clean_json_response(response_content)
                structured_result = DataFormattingResult.parse_raw(cleaned_content)
                logger.info("✅ ROOSEVELT'S TYPE SAFETY: Parsed with Pydantic validation!")
                return structured_result.dict()
                
            except (json.JSONDecodeError, ValidationError) as e:
                logger.info(f"⚠️ ROOSEVELT'S FALLBACK: JSON parsing failed ({e}), wrapping direct response")
                
                # Fallback: Create structured response from direct content
                fallback_result = DataFormattingResult(
                    task_status=TaskStatus.COMPLETE,
                    formatted_output=response_content,
                    format_type="structured_text",
                    confidence_level=0.8,
                    data_sources_used=self._extract_data_sources(context),
                    formatting_notes="Direct LLM response (non-JSON format)"
                )
                logger.info("✅ ROOSEVELT'S STRUCTURED FALLBACK: Created valid structure!")
                formatted_result = fallback_result.dict()
            
            return formatted_result
            
        except Exception as e:
            logger.error(f"❌ Data formatting execution failed: {e}")
            # Return structured error response
            error_result = DataFormattingResult(
                task_status=TaskStatus.ERROR,
                formatted_output=f"Error during formatting: {str(e)}",
                format_type="error",
                confidence_level=0.0,
                formatting_notes=f"Exception: {type(e).__name__}"
            )
            return error_result.dict()

    def _build_data_formatting_prompt(self, user_request: str, context: Dict[str, Any]) -> str:
        """Build data formatting prompt with conversation context"""
        
        # Extract context information
        conversation_summary = self._summarize_conversation_context(context)
        
        return f"""You are Roosevelt's Data Formatting Specialist - an expert at transforming research data into organized, readable formats.

**BULLY!** Follow Roosevelt's "Type Safety" doctrine and respond with STRUCTURED JSON!

**MISSION**: Transform existing research data into the requested format (tables, lists, structured data) using conversation context.

**USER REQUEST**: {user_request}

**CONVERSATION CONTEXT**:
{conversation_summary}

**ROOSEVELT'S STRUCTURED OUTPUT REQUIREMENT**:

You MUST respond with valid JSON matching this schema:
{{
    "task_status": "complete|incomplete|error",
    "formatted_output": "Your formatted content (markdown tables, lists, etc.)",
    "format_type": "markdown_table|markdown_list|structured_text|comparative_analysis|timeline|chronological_timeline|visual_timeline",
    "confidence_level": 0.9,
    "data_sources_used": ["source1", "source2"],
    "formatting_notes": "Optional notes about the formatting process"
}}

**FORMATTING EXPERTISE**:
- **MARKDOWN TABLES**: Perfect table formatting with proper alignment
- **TIMELINE FORMATTING**: Chronological organization of historical events
- **VISUAL TIMELINES**: ASCII-art style timelines with clear progression
- **DATA ORGANIZATION**: Logical grouping and categorization  
- **REGIONAL BREAKDOWN**: Organize by geographic regions when relevant
- **COMPARATIVE ANALYSIS**: Clear side-by-side comparisons
- **SENTIMENT ANALYSIS**: Organize by sentiment, stance, or opinion
- **STRUCTURED LISTS**: Bullet points, numbered lists, nested organization

**FORMATTING REQUIREMENTS**:
- **USE PROPER MARKDOWN**: Clean table syntax with | separators
- **CLEAR HEADERS**: Descriptive column/row headers
- **LOGICAL GROUPING**: Group related data together
- **READABLE FORMAT**: Proper spacing and alignment
- **COMPREHENSIVE**: Include ALL relevant data from context
- **ACCURATE**: Only use data that was actually provided in the research

**TIMELINE FORMATTING REQUIREMENTS**:
- **CHRONOLOGICAL ORDER**: Always organize events from earliest to latest
- **CLEAR DATE MARKERS**: Use specific dates when available, approximate periods when not
- **ERA ORGANIZATION**: Group events into logical historical periods
- **VISUAL HIERARCHY**: Use headers (##) for eras, bold (**) for dates
- **CONTEXT INCLUSION**: Provide brief context for each event's significance
- **PROGRESSION CLARITY**: Show how events lead to and influence each other

**TIMELINE FORMATTING EXAMPLES**:

**Option 1 - Chronological Timeline**:
```markdown
# Timeline: History of Tequila

## Pre-Columbian Era (Before 1500s)
- **Ancient Times**: Indigenous peoples ferment agave to create pulque
- **Aztec Empire**: Pulque becomes sacred ceremonial drink

## Spanish Colonial Period (1500s-1800s)  
- **1521**: Spanish conquistadors arrive in Mexico
- **1600s**: Spanish introduce distillation techniques to agave production
- **1700s**: First proto-tequila distilleries established in Jalisco region

## Modern Era (1800s-Present)
- **1873**: Don Cenobio Sauza exports first tequila to United States
- **1974**: Mexican government establishes Denomination of Origin protection
- **1994**: NAFTA increases international tequila trade
```

**Option 2 - Visual Timeline**:
```markdown
    1500s           1600s           1700s           1800s           1900s
      |               |               |               |               |
  Conquistadors  → Distillation  → First Distill. → US Export → Global Trade
    Arrive        Introduced      in Jalisco       (1873)      (NAFTA 1994)
      |               |               |               |               |
   [Pulque Era] → [Spanish Tech] → [Proto-Tequila] → [Commercial] → [Modern Era]
```

**TABLE FORMATTING EXAMPLES**:
```markdown
| Nation/Region | State-Sponsored Messaging | General Public Sentiment | Key Differences |
|---------------|---------------------------|--------------------------|-----------------|
| Russia        | Pro-invasion, justification | Mixed, some opposition | State control vs. diverse views |
| EU Countries  | Support for Ukraine | Largely supportive | Aligned messaging |
```

**DUAL OUTPUT CAPABILITY**:
You can provide BOTH text explanation AND formatted data when beneficial:

1. **PURE FORMATTING**: If user explicitly wants just a table/chart, provide only the formatted output
2. **ENHANCED RESPONSE**: If beneficial, provide brief context + formatted data
3. **INTELLIGENT DECISION**: Choose the best presentation for maximum clarity

**RESPONSE FORMATS**:

**Option 1 - Pure Table**: 
```markdown
| Column 1 | Column 2 |
|----------|----------|
| Data     | Data     |
```

**Option 2 - Enhanced Response**:
```markdown
Based on the research, here are the key findings:

| Country | Debt-to-GDP | Trend |
|---------|-------------|-------|
| US      | 122%        | ↑     |
| Japan   | 260%        | ↑     |

The data shows significant variations in debt levels across nations.
```

**JSON EXAMPLES**:

Simple table example:
```json
{{
    "task_status": "complete",
    "formatted_output": "| Location | Type | Description |\\n|----------|------|-------------|\\n| Sonnenberg Gardens | Garden | Historic estate with formal gardens |\\n| CMAC | Venue | Outdoor concert amphitheater |",
    "format_type": "markdown_table",
    "confidence_level": 0.95,
    "data_sources_used": ["Research findings"],
    "formatting_notes": "Organized by location and type"
}}
```

Timeline formatting example:
```json
{{
    "task_status": "complete",
    "formatted_output": "# Timeline: History of Tequila\\n\\n## Pre-Columbian Era (Before 1500s)\\n- **Ancient Times**: Indigenous peoples ferment agave to create pulque\\n- **Aztec Empire**: Pulque becomes sacred ceremonial drink\\n\\n## Spanish Colonial Period (1500s-1800s)\\n- **1521**: Spanish conquistadors arrive in Mexico\\n- **1600s**: Spanish introduce distillation techniques\\n- **1700s**: First proto-tequila distilleries in Jalisco\\n\\n## Modern Era (1800s-Present)\\n- **1873**: Don Cenobio Sauza exports first tequila to US\\n- **1974**: Mexican government establishes protection\\n- **1994**: NAFTA increases international trade",
    "format_type": "chronological_timeline",
    "confidence_level": 0.95,
    "data_sources_used": ["Research findings", "Historical sources"],
    "formatting_notes": "Organized chronologically with clear era divisions"
}}
```

Enhanced response example:
```json
{{
    "task_status": "complete",
    "formatted_output": "Based on the research, here are places to visit near Canandaigua, NY:\\n\\n| Location | Type | Season | Description |\\n|----------|------|--------|-------------|\\n| Sonnenberg Gardens | Garden | Spring-Fall | Historic estate with formal gardens |\\n| CMAC | Venue | Summer | Outdoor concert amphitheater |\\n\\nThese locations offer diverse recreational opportunities in the Finger Lakes region.",
    "format_type": "comparative_analysis",
    "confidence_level": 0.9,
    "data_sources_used": ["TripAdvisor", "Local guides"],
    "formatting_notes": "Added seasonal information and context"
}}
```

**CRITICAL**: 
1. **STRUCTURED JSON ONLY** - No plain text responses!
2. **Use actual data** - Only data provided in conversation context
3. **Proper escaping** - Escape newlines as \\\\n in JSON strings
4. **Roosevelt's Type Safety** - Valid JSON structure always!

**By George!** Make that JSON structure as solid as a cavalry charge!
"""

    def _summarize_conversation_context(self, context: Dict[str, Any]) -> str:
        """Summarize conversation context for the formatting prompt"""
        try:
            summary_parts = []
            
            # Add conversation history summary
            conversation_history = context.get("conversation_history", [])
            if conversation_history:
                summary_parts.append("**RECENT CONVERSATION**:")
                for i, msg in enumerate(conversation_history[-3:]):  # Last 3 messages
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")  # Keep full content
                    summary_parts.append(f"{i+1}. {role.upper()}: {content}")
            
            # Add previous research summary
            previous_research = context.get("previous_research", {})
            if previous_research:
                summary_parts.append("\n**PREVIOUS RESEARCH DATA**:")
                for source, data in previous_research.items():
                    if isinstance(data, str):
                        summary_parts.append(f"- {source.upper()}: {data}")
                    elif isinstance(data, dict):
                        summary_parts.append(f"- {source.upper()}: {str(data)}")
            
            # Add agent results summary
            previous_agent_results = context.get("previous_agent_results", {})
            if previous_agent_results and isinstance(previous_agent_results, dict):
                structured_response = previous_agent_results.get("structured_response", {})
                if structured_response:
                    findings = structured_response.get("findings", "")
                    if findings:
                        summary_parts.append(f"\n**RESEARCH FINDINGS**: {findings}")
            
            return "\n".join(summary_parts) if summary_parts else "No previous research context available."
            
        except Exception as e:
            logger.error(f"❌ Failed to summarize context: {e}")
            return "Error extracting conversation context."

    def _clean_json_response(self, response_content: str) -> str:
        """Clean LLM response to extract JSON content"""
        try:
            # Remove markdown code blocks if present
            if "```json" in response_content:
                start = response_content.find("```json") + 7
                end = response_content.find("```", start)
                if end != -1:
                    return response_content[start:end].strip()
            
            # Remove markdown code blocks without language
            if "```" in response_content:
                start = response_content.find("```") + 3
                end = response_content.find("```", start)
                if end != -1:
                    potential_json = response_content[start:end].strip()
                    # Check if this looks like JSON
                    if potential_json.startswith("{") and potential_json.endswith("}"):
                        return potential_json
            
            # Return as-is if no code blocks found
            return response_content.strip()
            
        except Exception as e:
            logger.error(f"❌ Failed to clean JSON response: {e}")
            return response_content

    def _extract_data_sources(self, context: Dict[str, Any]) -> List[str]:
        """Extract data sources from context for structured response"""
        try:
            sources = []
            
            # Get sources from previous research
            previous_research = context.get("previous_research", {})
            if previous_research:
                sources.extend(list(previous_research.keys()))
            
            # Get sources from conversation context
            conversation_history = context.get("conversation_history", [])
            if conversation_history:
                sources.append("Conversation context")
            
            # Get sources from agent results
            previous_agent_results = context.get("previous_agent_results", {})
            if previous_agent_results:
                sources.append("Research findings")
            
            return sources[:5] if sources else ["Direct input"]
            
        except Exception as e:
            logger.error(f"❌ Failed to extract data sources: {e}")
            return ["Unknown sources"]

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        error_result = DataFormattingResult(
            task_status=TaskStatus.ERROR,
            formatted_output=f"❌ Data formatting error: {error_message}",
            format_type="error",
            confidence_level=0.0,
            formatting_notes=f"Error: {error_message}"
        )
        
        return {
            "agent_results": {
                "agent_type": "data_formatting",
                "structured_response": error_result.dict(),
                "task_status": "error",
                "error_message": error_message,
                "confidence": 0.0
            },
            "latest_response": error_result.formatted_output,
            "is_complete": True
        }
