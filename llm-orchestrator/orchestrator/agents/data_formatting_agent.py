"""
Data Formatting Agent - Table and Structure Specialist
Transforms research data into tables, charts, and organized formats
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base_agent import BaseAgent, TaskStatus

logger = logging.getLogger(__name__)


class DataFormattingAgent(BaseAgent):
    """
    Data Formatting Specialist
    Transforms research findings into structured formats like tables, charts, and organized data
    """
    
    def __init__(self):
        super().__init__("data_formatting_agent")
        logger.info("🔢 Data Formatting Agent assembled and ready!")
    
    def _build_formatting_prompt(self, user_request: str, conversation_context: str) -> str:
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
    
    def _extract_conversation_context(self, messages: List[Any]) -> str:
        """Extract conversation context for formatting"""
        try:
            if not messages:
                return "No previous conversation context available."
            
            # Get last few messages for context
            recent_messages = messages[-5:] if len(messages) > 5 else messages
            
            context_parts = []
            for i, msg in enumerate(recent_messages):
                if hasattr(msg, 'content'):
                    role = "ASSISTANT" if hasattr(msg, 'type') and msg.type == "ai" else "USER"
                    content = msg.content
                    context_parts.append(f"{i+1}. {role}: {content}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"❌ Failed to extract conversation context: {e}")
            return "Error extracting conversation context."
    
    async def process(self, query: str, metadata: Dict[str, Any] = None, messages: List[Any] = None) -> Dict[str, Any]:
        """Process data formatting request"""
        try:
            logger.info(f"🔢 Data Formatting Agent processing: {query[:100]}...")
            
            # Extract conversation context
            conversation_context = self._extract_conversation_context(messages)
            
            # Build formatting prompt
            system_prompt = self._build_formatting_prompt(query, conversation_context)
            
            # Call LLM with low temperature for consistent formatting
            start_time = datetime.now()
            llm = self._get_llm(temperature=0.1)
            response = await llm.ainvoke(self._build_messages(system_prompt, query))
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Parse structured response
            response_content = response.content if hasattr(response, 'content') else str(response)
            structured_result = self._parse_json_response(response_content)
            
            # Unescape JSON newlines for proper markdown display
            formatted_output = structured_result.get("formatted_output", "")
            if formatted_output:
                formatted_output = formatted_output.replace("\\n", "\n")
            
            # Build result
            result = {
                "response": formatted_output,
                "task_status": structured_result.get("task_status", "complete"),
                "format_type": structured_result.get("format_type", "structured_text"),
                "confidence": structured_result.get("confidence_level", 0.9),
                "agent_type": "data_formatting",
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                "formatting_notes": structured_result.get("formatting_notes", "")
            }
            
            logger.info(f"✅ Data formatting completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Data formatting failed: {e}")
            return self._create_error_response(str(e))

