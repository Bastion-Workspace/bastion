"""
Chat Agent Implementation for LLM Orchestrator
Handles general conversation and knowledge queries
"""

import logging
from typing import Dict, Any, List, Optional, TypedDict
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from .base_agent import BaseAgent, TaskStatus

logger = logging.getLogger(__name__)


class ChatState(TypedDict):
    """State for chat agent LangGraph workflow"""
    query: str
    user_id: str
    metadata: Dict[str, Any]
    messages: List[Any]
    persona: Optional[Dict[str, Any]]
    system_prompt: str
    llm_messages: List[Any]
    needs_calculations: bool
    calculation_result: Optional[Dict[str, Any]]
    local_data_results: Optional[str]  # Vector search results for local data queries
    response: Dict[str, Any]
    task_status: str
    error: str
    shared_memory: Dict[str, Any]  # For storing primary_agent_selected and continuity data


class ChatAgent(BaseAgent):
    """Chat agent for general conversation and knowledge queries"""
    
    def __init__(self):
        super().__init__("chat_agent")
        logger.info("💬 Chat Agent ready for conversation!")
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """Build LangGraph workflow for chat agent"""
        workflow = StateGraph(ChatState)
        
        # Add nodes
        workflow.add_node("prepare_context", self._prepare_context_node)
        workflow.add_node("check_local_data", self._check_local_data_node)
        workflow.add_node("detect_calculations", self._detect_calculations_node)
        workflow.add_node("perform_calculations", self._perform_calculations_node)
        workflow.add_node("generate_response", self._generate_response_node)
        
        # Entry point
        workflow.set_entry_point("prepare_context")
        
        # Flow: prepare context -> check local data -> detect calculations -> (conditional) -> generate response
        workflow.add_edge("prepare_context", "check_local_data")
        workflow.add_edge("check_local_data", "detect_calculations")
        
        # Route based on whether calculations are needed
        workflow.add_conditional_edges(
            "detect_calculations",
            self._route_from_calculation_detection,
            {
                "calculate": "perform_calculations",
                "respond": "generate_response"
            }
        )
        
        # After calculations, go to response generation
        workflow.add_edge("perform_calculations", "generate_response")
        workflow.add_edge("generate_response", END)
        
        # Compile with checkpointer for state persistence
        return workflow.compile(checkpointer=checkpointer)
    
    def _route_from_calculation_detection(self, state: ChatState) -> str:
        """Route based on whether calculations are needed"""
        if state.get("needs_calculations", False):
            return "calculate"
        return "respond"
    
    def _build_chat_prompt(self, persona: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Build system prompt for chat agent"""
        ai_name = persona.get("ai_name", "Alex") if persona else "Alex"
        persona_style = persona.get("persona_style", "professional") if persona else "professional"
        
        # Extract user context from metadata
        preferred_name = metadata.get("user_preferred_name", "") if metadata else ""
        user_context = metadata.get("user_ai_context", "") if metadata else ""
        
        # Build style instruction based on persona_style
        style_instruction = self._get_style_instruction(persona_style)
        
        # Build user context sections
        user_context_sections = []
        if preferred_name and preferred_name.strip():
            user_context_sections.append(f"USER PREFERENCE:\nThe user prefers to be addressed as: {preferred_name.strip()}")
        if user_context and user_context.strip():
            user_context_sections.append(f"USER CONTEXT:\n{user_context.strip()}")
        
        user_context_text = "\n\n".join(user_context_sections) + "\n\n" if user_context_sections else ""
        
        base_prompt = f"""You are {ai_name}, a conversational AI assistant. Your role is to have natural conversations while providing accurate, useful information.

{style_instruction}

{user_context_text}

CONVERSATION GUIDELINES:
1. **BE APPROPRIATELY RESPONSIVE**: Match your response length to the user's input - brief acknowledgments get brief responses
2. **MAINTAIN CONTEXT**: Use conversation history to understand follow-up questions and maintain flow
3. **ASK FOR CLARIFICATION**: If a question is unclear, ask for more details
4. **BE CONCISE AND NATURAL**: Provide appropriate conversational responses
5. **STAY CONVERSATIONAL**: Focus on dialogue and helpful information
6. **USE MARKDOWN FORMATTING**: Format your responses using Markdown for better readability:
   - Use **bold** for emphasis and key terms
   - Use *italics* for subtle emphasis
   - Use bullet points (-) or numbered lists (1.) for lists
   - Use `code blocks` for technical terms, code, or specific values
   - Use ## headings for longer responses with multiple sections
   - Use tables (| column | column |) when presenting structured data

RESPONSE LENGTH GUIDELINES:
- **Simple acknowledgments** ("thanks", "thank you"): Brief friendly response (1-2 sentences)
- **Questions or requests**: Helpful detailed responses  
- **Complex topics**: Thorough explanations with context
- **Casual conversation**: Natural, proportionate responses

WHAT YOU HANDLE:
- Greetings and casual conversation
- Creative brainstorming and idea generation
- General knowledge synthesis and explanations
- Opinion requests and strategic advice
- Hypothetical scenarios and "what if" questions
- Follow-up questions and clarifications
- Technical discussions using your training knowledge
- Mathematical calculations (the system will automatically calculate for you)
- Questions about local documents and data (the system will search your documents automatically)

PROJECT GUIDANCE:
- If user asks about electronics/circuits/components without an electronics project open:
  * Suggest: "To work on electronics projects, create one first: Right-click a folder → 'New Project' → select 'Electronics'."
  * Then provide general information if helpful
- If user asks about project-specific work (e.g., "add a component to our system") without a project open:
  * Guide them to create a project first using the same instructions

STRUCTURED OUTPUT REQUIREMENT:
You MUST respond with valid JSON matching this schema:
{{
    "message": "Your conversational response",
    "task_status": "complete"
}}

EXAMPLES:

Simple acknowledgment:
{{
    "message": "You're welcome! Let me know if you need anything else.",
    "task_status": "complete"
}}

Detailed response:
{{
    "message": "Here's what I think about that topic...",
    "task_status": "complete"
}}

CONVERSATION CONTEXT:
You have access to conversation history for context. Use this to understand follow-up questions and maintain conversational flow.

LOCAL DATA SEARCH:
If the system provides "RELEVANT LOCAL INFORMATION" in the context, use that information to answer the user's question accurately. The local information takes precedence over general knowledge when answering questions about specific documents, files, or data in the user's knowledge base."""

        return base_prompt
    
    async def _check_local_data_node(self, state: ChatState) -> Dict[str, Any]:
        """Check local documents for relevant information using intelligent retrieval subgraph"""
        try:
            query = state.get("query", "")
            
            # Extract user_id from metadata, shared_memory, or state
            metadata = state.get("metadata", {})
            shared_memory = state.get("shared_memory", {})
            user_id = metadata.get("user_id") or shared_memory.get("user_id") or state.get("user_id", "system")
            
            # Use intelligent document retrieval subgraph
            from orchestrator.subgraphs.intelligent_document_retrieval_subgraph import retrieve_documents_intelligently
            
            result = await retrieve_documents_intelligently(
                query=query,
                user_id=user_id,
                mode="fast",  # Quick retrieval for chat agent
                max_results=3,
                small_doc_threshold=10000  # Increased to handle medium-sized docs
            )
            
            if result.get("success") and result.get("formatted_context"):
                logger.info(f"💬 Found relevant local documents via intelligent retrieval")
                return {
                    "local_data_results": result.get("formatted_context"),
                    # ✅ CRITICAL: Preserve state for subsequent nodes
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", "")
                }
            
            return {
                "local_data_results": None,
                # ✅ CRITICAL: Preserve state for subsequent nodes
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
            
        except Exception as e:
            logger.warning(f"💬 Local data check failed: {e} - continuing without local data")
            return {
                "local_data_results": None,
                # ✅ CRITICAL: Preserve state even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    async def _prepare_context_node(self, state: ChatState) -> Dict[str, Any]:
        """Prepare context: extract persona, build prompt, extract conversation history"""
        try:
            logger.info(f"💬 Preparing context for chat query: {state['query'][:100]}...")
            
            # Extract metadata and persona
            metadata = state.get("metadata", {})
            persona = metadata.get("persona")
            
            # Build system prompt (pass metadata for user context)
            system_prompt = self._build_chat_prompt(persona, metadata)
            
            # Build messages for LLM using standardized helper
            messages_list = state.get("messages", [])
            llm_messages = self._build_conversational_agent_messages(
                system_prompt=system_prompt,
                user_prompt=state["query"],
                messages_list=messages_list,
                look_back_limit=10
            )
            
            return {
                "persona": persona,
                "system_prompt": system_prompt,
                "llm_messages": llm_messages,
                # ✅ CRITICAL: Preserve state for subsequent nodes
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
            
        except Exception as e:
            logger.error(f"❌ Context preparation failed: {e}")
            return {
                "error": str(e),
                "task_status": "error",
                # ✅ CRITICAL: Preserve state even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    async def _detect_calculations_node(self, state: ChatState) -> Dict[str, Any]:
        """Detect if query requires mathematical calculations"""
        try:
            query = state.get("query", "")
            query_lower = query.lower().strip()
            
            # Simple detection: look for math patterns
            # Arithmetic expressions: "84+92", "100 - 50", "5 * 3", "10 / 2"
            # Calculation keywords: "calculate", "compute", "what is X + Y", "how much is"
            # Math symbols: +, -, *, /, =, ^
            
            has_math_symbols = any(symbol in query for symbol in ['+', '-', '*', '/', '=', '^', '×', '÷'])
            has_calculation_keywords = any(kw in query_lower for kw in [
                "calculate", "compute", "what is", "how much is", "what's", "equals",
                "plus", "minus", "times", "divided by", "multiply", "add", "subtract"
            ])
            
            # Check for simple arithmetic patterns (e.g., "84+92", "100 - 50")
            import re
            arithmetic_pattern = re.search(r'\d+\s*[+\-*/×÷]\s*\d+', query)
            
            needs_calculations = has_math_symbols or has_calculation_keywords or bool(arithmetic_pattern)
            
            logger.info(f"💬 Calculation detection: {needs_calculations} (symbols: {has_math_symbols}, keywords: {has_calculation_keywords}, pattern: {bool(arithmetic_pattern)})")
            
            return {
                "needs_calculations": needs_calculations,
                # ✅ CRITICAL: Preserve state for subsequent nodes
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
            
        except Exception as e:
            logger.error(f"❌ Calculation detection failed: {e}")
            return {
                "needs_calculations": False,
                # ✅ CRITICAL: Preserve state even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    async def _perform_calculations_node(self, state: ChatState) -> Dict[str, Any]:
        """Perform calculations using math tool"""
        try:
            query = state.get("query", "")
            
            # Extract mathematical expression from query
            import re
            
            # Try to find arithmetic expression
            arithmetic_match = re.search(r'(\d+(?:\.\d+)?)\s*([+\-*/×÷])\s*(\d+(?:\.\d+)?)', query)
            
            if arithmetic_match:
                # Simple arithmetic found
                num1 = float(arithmetic_match.group(1))
                operator = arithmetic_match.group(2)
                num2 = float(arithmetic_match.group(3))
                
                # Map operators
                operator_map = {
                    '+': '+',
                    '-': '-',
                    '*': '*',
                    '/': '/',
                    '×': '*',
                    '÷': '/'
                }
                
                op_symbol = operator_map.get(operator, operator)
                expression = f"{num1} {op_symbol} {num2}"
                
                logger.info(f"💬 Performing calculation: {expression}")
                
                from orchestrator.tools.math_tools import calculate_expression_tool
                result = await calculate_expression_tool(expression)
                
                if result.get("success"):
                    calculation_result = result.get("result")
                    logger.info(f"✅ Calculation result: {calculation_result}")
                    
                    return {
                        "calculation_result": {
                            "expression": expression,
                            "result": calculation_result,
                            "steps": result.get("steps", [])
                        },
                        "needs_calculations": False,
                        # ✅ CRITICAL: Preserve state for subsequent nodes
                        "metadata": state.get("metadata", {}),
                        "user_id": state.get("user_id", "system"),
                        "shared_memory": state.get("shared_memory", {}),
                        "messages": state.get("messages", []),
                        "query": state.get("query", "")
                    }
                else:
                    logger.warning(f"⚠️ Calculation failed: {result.get('error')}")
                    return {
                        "calculation_result": None,
                        "needs_calculations": False,
                        # ✅ CRITICAL: Preserve state for subsequent nodes
                        "metadata": state.get("metadata", {}),
                        "user_id": state.get("user_id", "system"),
                        "shared_memory": state.get("shared_memory", {}),
                        "messages": state.get("messages", []),
                        "query": state.get("query", "")
                    }
            else:
                # Try to extract expression using LLM
                fast_model = self._get_fast_model(state)
                llm = self._get_llm(temperature=0.1, model=fast_model, state=state)
                
                prompt = f"""Extract the mathematical expression or calculation from this query:

**QUERY**: {query}

**TASK**: Extract a mathematical expression that can be evaluated.

**EXAMPLES**:
- "What is 84+92?" → "84+92"
- "Calculate 100 times 5" → "100*5"
- "How much is 50 divided by 2?" → "50/2"
- "What's 10 minus 3?" → "10-3"

Return ONLY the mathematical expression as a string, or "null" if no clear expression can be extracted.

Return ONLY valid JSON:
{{
  "expression": "84+92" or null
}}"""
                
                try:
                    schema = {
                        "type": "object",
                        "properties": {
                            "expression": {"type": ["string", "null"]}
                        },
                        "required": ["expression"]
                    }
                    structured_llm = llm.with_structured_output(schema)
                    result = await structured_llm.ainvoke([{"role": "user", "content": prompt}])
                    result_dict = result if isinstance(result, dict) else result.dict() if hasattr(result, 'dict') else result.model_dump()
                    expression = result_dict.get("expression")
                except Exception:
                    response = await llm.ainvoke([{"role": "user", "content": prompt}])
                    content = response.content if hasattr(response, 'content') else str(response)
                    result_dict = self._parse_json_response(content) or {}
                    expression = result_dict.get("expression")
                
                if expression:
                    from orchestrator.tools.math_tools import calculate_expression_tool
                    calc_result = await calculate_expression_tool(expression)
                    
                    if calc_result.get("success"):
                        return {
                            "calculation_result": {
                                "expression": expression,
                                "result": calc_result.get("result"),
                                "steps": calc_result.get("steps", [])
                            },
                            "needs_calculations": False,
                            # ✅ CRITICAL: Preserve state for subsequent nodes
                            "metadata": state.get("metadata", {}),
                            "user_id": state.get("user_id", "system"),
                            "shared_memory": state.get("shared_memory", {}),
                            "messages": state.get("messages", []),
                            "query": state.get("query", "")
                        }
            
            # No calculation could be extracted
            return {
                "calculation_result": None,
                "needs_calculations": False,
                # ✅ CRITICAL: Preserve state for subsequent nodes
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
            
        except Exception as e:
            logger.error(f"❌ Calculation failed: {e}")
            return {
                "calculation_result": None,
                "needs_calculations": False,
                "error": str(e),
                # ✅ CRITICAL: Preserve state even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    async def _generate_response_node(self, state: ChatState) -> Dict[str, Any]:
        """Generate response: call LLM and parse structured output"""
        try:
            logger.info("💬 Generating chat response...")
            
            llm_messages = state.get("llm_messages", [])
            if not llm_messages:
                return {
                    "error": "No LLM messages prepared",
                    "task_status": "error",
                    "response": {},
                    # ✅ CRITICAL: Preserve state even on error
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", "")
                }
            
            # Include local data results in the prompt if available
            local_data_results = state.get("local_data_results")
            if local_data_results:
                # Add local data context to the last user message
                # Find the last HumanMessage and append the local data context
                if llm_messages and len(llm_messages) > 0:
                    last_message = llm_messages[-1]
                    if isinstance(last_message, HumanMessage):
                        # Append local data to the query
                        enhanced_content = f"{last_message.content}\n\n{local_data_results}"
                        llm_messages[-1] = HumanMessage(content=enhanced_content)
                        logger.info("💬 Enhanced prompt with local data context")
            
            # Call LLM - pass state to access user's model selection from metadata
            start_time = datetime.now()
            llm = self._get_llm(temperature=0.7, state=state)
            response = await llm.ainvoke(llm_messages)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Check if we have calculation results to include
            calculation_result = state.get("calculation_result")
            calc_value = None
            calc_expression = None
            
            if calculation_result:
                calc_value = calculation_result.get("result")
                calc_expression = calculation_result.get("expression", "")
                logger.info(f"💬 Including calculation result in response: {calc_expression} = {calc_value}")
            
            # Call LLM
            response = await llm.ainvoke(llm_messages)
            
            # Parse structured response
            response_content = response.content if hasattr(response, 'content') else str(response)
            structured_response = self._parse_json_response(response_content)
            
            # Extract message
            final_message = structured_response.get("message", response_content)
            
            # If calculation was performed, prepend the result
            if calculation_result and calc_value is not None:
                # Prepend calculation result for clarity
                final_message = f"{calc_expression} = {calc_value}\n\n{final_message}"
            
            # Add assistant response to messages for checkpoint persistence
            state = self._add_assistant_response_to_messages(state, final_message)
            
            # Store primary_agent_selected in shared_memory for conversation continuity
            shared_memory = state.get("shared_memory", {})
            shared_memory["primary_agent_selected"] = "chat_agent"
            shared_memory["last_agent"] = "chat_agent"
            state["shared_memory"] = shared_memory
            
            # Clear request-scoped data (active_editor) before checkpoint save
            # This ensures it's available during the request (for subgraphs) but doesn't persist
            state = self._clear_request_scoped_data(state)
            shared_memory = state.get("shared_memory", {})
            
            # Build result
            result = {
                "response": {
                "response": final_message,
                "task_status": structured_response.get("task_status", "complete"),
                "agent_type": "chat_agent",
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
                },
                "task_status": structured_response.get("task_status", "complete"),
                "messages": state.get("messages", []),
                "shared_memory": shared_memory
            }
            
            logger.info(f"✅ Chat response generated in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Response generation failed: {e}")
            return {
                "error": str(e),
                "task_status": "error",
                "response": self._create_error_response(str(e))
            }
    
    async def process(self, query: str, metadata: Dict[str, Any] = None, messages: List[Any] = None) -> Dict[str, Any]:
        """Process chat query using LangGraph workflow"""
        try:
            logger.info(f"💬 Chat agent processing: {query[:100]}...")
            
            # Get workflow (lazy initialization with checkpointer)
            workflow = await self._get_workflow()
            
            # Extract user_id from metadata
            metadata = metadata or {}
            user_id = metadata.get("user_id", "system")
            
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
            
            # Initialize state for LangGraph workflow
            initial_state: ChatState = {
                "query": query,
                "user_id": user_id,
                "metadata": metadata,
                "messages": conversation_messages,
                "persona": None,
                "system_prompt": "",
                "llm_messages": [],
                "needs_calculations": False,
                "calculation_result": None,
                "local_data_results": None,
                "response": {},
                "task_status": "",
                "error": "",
                "shared_memory": shared_memory
            }
            
            # Run LangGraph workflow with checkpointing
            result_state = await workflow.ainvoke(initial_state, config=config)
            
            # Extract final response
            response = result_state.get("response", {})
            task_status = result_state.get("task_status", "complete")
            
            if task_status == "error":
                error_msg = result_state.get("error", "Unknown error")
                logger.error(f"❌ Chat agent failed: {error_msg}")
                return self._create_error_response(error_msg)
            
            logger.info(f"✅ Chat agent completed: {task_status}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Chat agent failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._create_error_response(str(e))

