"""
Data Formatting Agent - Table and Structure Specialist
Transforms research data into tables, charts, and organized formats
"""

import logging
from typing import Dict, Any, List, Optional, TypedDict
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from .base_agent import BaseAgent, TaskStatus

logger = logging.getLogger(__name__)


class DataFormattingState(TypedDict):
    """State for data formatting agent LangGraph workflow"""
    query: str
    user_id: str
    metadata: Dict[str, Any]
    messages: List[Any]
    conversation_context: str
    system_prompt: str
    formatted_output: str
    format_type: str
    response: Dict[str, Any]
    task_status: str
    error: str


class DataFormattingAgent(BaseAgent):
    """
    Data Formatting Specialist
    Transforms research findings into structured formats like tables, charts, and organized data
    Uses LangGraph workflow for explicit state management
    """
    
    def __init__(self):
        super().__init__("data_formatting_agent")
        logger.info("🔢 Data Formatting Agent assembled and ready!")
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """Build LangGraph workflow for data formatting agent"""
        workflow = StateGraph(DataFormattingState)
        
        # Add nodes
        workflow.add_node("prepare_context", self._prepare_context_node)
        workflow.add_node("format_data", self._format_data_node)
        
        # Entry point
        workflow.set_entry_point("prepare_context")
        
        # Linear flow: prepare_context -> format_data -> END
        workflow.add_edge("prepare_context", "format_data")
        workflow.add_edge("format_data", END)
        
        return workflow.compile(checkpointer=checkpointer)
    
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
    
    async def _prepare_context_node(self, state: DataFormattingState) -> Dict[str, Any]:
        """Prepare conversation context for formatting"""
        try:
            logger.info("📋 Preparing conversation context for data formatting...")
            
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
            system_prompt = self._build_formatting_prompt(state.get("query", ""), conversation_context)
            
            return {
                "conversation_context": conversation_context,
                "system_prompt": system_prompt
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to prepare context: {e}")
            return {
                "conversation_context": "Error extracting conversation context.",
                "system_prompt": "",
                "error": str(e)
            }
    
    async def _format_data_node(self, state: DataFormattingState) -> Dict[str, Any]:
        """Format data using LLM with structured output"""
        try:
            logger.info("🔢 Formatting data with structured output...")
            
            query = state.get("query", "")
            system_prompt = state.get("system_prompt", "")
            
            if not system_prompt:
                raise ValueError("System prompt not prepared")
            
            # Extract conversation history for proper message structure
            # Call LLM with low temperature for consistent formatting - pass state to access user's model selection
            start_time = datetime.now()
            llm = self._get_llm(temperature=0.1, state=state)
            messages_list = state.get("messages", [])
            llm_messages = self._build_conversational_agent_messages(
                system_prompt=system_prompt,
                user_prompt=query,
                messages_list=messages_list,
                look_back_limit=5
            )
            response = await llm.ainvoke(llm_messages)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Parse structured response
            response_content = response.content if hasattr(response, 'content') else str(response)
            structured_result = self._parse_json_response(response_content)
            
            # Unescape JSON newlines for proper markdown display
            formatted_output = structured_result.get("formatted_output", "")
            if formatted_output:
                formatted_output = formatted_output.replace("\\n", "\n")
            
            # Add assistant response to messages for checkpoint persistence
            state = self._add_assistant_response_to_messages(state, formatted_output)
            
            # Build result
            result = {
                "formatted_output": formatted_output,
                "format_type": structured_result.get("format_type", "structured_text"),
                "response": {
                "response": formatted_output,
                "task_status": structured_result.get("task_status", "complete"),
                "format_type": structured_result.get("format_type", "structured_text"),
                "confidence": structured_result.get("confidence_level", 0.9),
                "agent_type": "data_formatting",
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                "formatting_notes": structured_result.get("formatting_notes", "")
                },
                "task_status": structured_result.get("task_status", "complete"),
                "messages": state.get("messages", [])
            }
            
            logger.info(f"✅ Data formatting completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Data formatting failed: {e}")
            error_response = self._create_error_response(str(e))
            return {
                "formatted_output": "",
                "format_type": "error",
                "response": error_response,
                "task_status": "error",
                "error": str(e)
            }
    
    async def process(self, query: str, metadata: Dict[str, Any] = None, messages: List[Any] = None) -> Dict[str, Any]:
        """
        Process data formatting request using LangGraph workflow
        
        Args:
            query: User query or formatting request
            metadata: Optional metadata (user_id, persona, etc.)
            messages: Optional conversation history
            
        Returns:
            Dictionary with formatted output and metadata
        """
        try:
            logger.info(f"🔢 Data Formatting Agent processing: {query[:100]}...")
            
            metadata = metadata or {}
            user_id = metadata.get("user_id", "system")
            
            # Get workflow (lazy initialization with checkpointer)
            workflow = await self._get_workflow()
            
            # Get checkpoint config (handles thread_id from conversation_id/user_id)
            config = self._get_checkpoint_config(metadata)
            
            # Prepare new messages (current query)
            new_messages = self._prepare_messages_with_query(messages, query)
            
            # Load and merge checkpointed messages to preserve conversation history
            conversation_messages = await self._load_and_merge_checkpoint_messages(
                workflow, config, new_messages
            )
            
            # Load shared_memory from checkpoint if available
            checkpoint_state = await workflow.aget_state(config)
            existing_shared_memory = {}
            if checkpoint_state and checkpoint_state.values:
                existing_shared_memory = checkpoint_state.values.get("shared_memory", {})
            
            # Merge with any shared_memory from metadata
            shared_memory = metadata.get("shared_memory", {}) or {}
            shared_memory.update(existing_shared_memory)
            metadata["shared_memory"] = shared_memory
            
            # Build initial state
            initial_state: DataFormattingState = {
                "query": query,
                "user_id": user_id,
                "metadata": metadata,
                "messages": conversation_messages,
                "conversation_context": "",
                "system_prompt": "",
                "formatted_output": "",
                "format_type": "",
                "response": {},
                "task_status": "",
                "error": ""
            }
            
            # Run workflow with checkpointing
            final_state = await workflow.ainvoke(initial_state, config=config)
            
            # Return response from final state
            return final_state.get("response", {
                "response": final_state.get("formatted_output", ""),
                "task_status": final_state.get("task_status", "complete"),
                "format_type": final_state.get("format_type", "structured_text"),
                "agent_type": "data_formatting"
            })
            
        except Exception as e:
            logger.error(f"❌ Data formatting workflow failed: {e}")
            return self._create_error_response(str(e))

