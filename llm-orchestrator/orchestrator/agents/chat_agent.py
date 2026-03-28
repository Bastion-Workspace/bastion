"""
Chat Agent Implementation for LLM Orchestrator
Handles general conversation and knowledge queries
"""

import logging
import re
from typing import Dict, Any, List, Optional, TypedDict
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from .base_agent import BaseAgent, TaskStatus
from orchestrator.models.agent_response_contract import AgentResponse
from orchestrator.middleware.message_preprocessor import MessagePreprocessor

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
    local_data_results: Optional[str]
    image_search_results: Optional[str]
    image_result_count: int
    extracted_images: List[str]
    attached_image_analysis: Optional[Dict[str, Any]]
    response: Dict[str, Any]
    task_status: str
    error: str
    shared_memory: Dict[str, Any]


class ChatAgent(BaseAgent):
    """Chat agent for general conversation and knowledge queries"""
    
    def __init__(self):
        super().__init__("chat_agent")
        logger.debug("💬 Chat Agent ready for conversation!")
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """Build LangGraph workflow for chat agent"""
        workflow = StateGraph(ChatState)
        
        workflow.add_node("prepare_context", self._prepare_context_node)
        workflow.add_node("process_attached_images", self._process_attached_images_node)
        workflow.add_node("fast_time_response", self._fast_time_response_node)
        workflow.add_node("check_local_data", self._check_local_data_node)
        workflow.add_node("detect_calculations", self._detect_calculations_node)
        workflow.add_node("perform_calculations", self._perform_calculations_node)
        workflow.add_node("generate_response", self._generate_response_node)
        
        workflow.set_entry_point("prepare_context")
        
        workflow.add_conditional_edges(
            "prepare_context",
            self._route_from_prepare_context,
            {
                "process_images": "process_attached_images",
                "fast_time": "fast_time_response",
                "normal": "check_local_data",
            }
        )
        
        workflow.add_conditional_edges(
            "process_attached_images",
            self._route_from_image_processing,
            {
                "image_search": "check_local_data",
                "image_generation": "generate_response",
                "normal": "check_local_data",
            }
        )
        
        workflow.add_edge("fast_time_response", END)
        workflow.add_edge("check_local_data", "detect_calculations")
        
        workflow.add_conditional_edges(
            "detect_calculations",
            self._route_from_calculation_detection,
            {
                "calculate": "perform_calculations",
                "respond": "generate_response",
            }
        )
        
        workflow.add_edge("perform_calculations", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow.compile(checkpointer=checkpointer)
    
    def _route_from_prepare_context(self, state: ChatState) -> str:
        """Route based on whether there are attached images or if this is a simple time/date query"""
        # Check for attached images in shared_memory
        shared_memory = state.get("shared_memory", {})
        attached_images = shared_memory.get("attached_images", [])
        
        if attached_images:
            logger.info(f"📎 Found {len(attached_images)} attached image(s) - processing for analysis")
            return "process_images"
        
        query = state.get("query", "").lower().strip()
        
        # Simple time/date queries that don't need document search
        simple_time_queries = [
            "what time is it", "what's the time", "current time", "time now",
            "what date is it", "what's the date", "current date", "date today",
            "what day is it", "what day is today", "day of week"
        ]
        
        is_simple_time_query = any(time_query in query for time_query in simple_time_queries)
        
        if is_simple_time_query:
            logger.info(f"🕐 FAST PATH: Simple time/date query detected, skipping document search")
            return "fast_time"
        
        return "normal"
    
    def _route_from_image_processing(self, state: ChatState) -> str:
        """Route based on image analysis results and user query intent"""
        query = state.get("query", "").lower()
        analysis = state.get("attached_image_analysis", {})
        
        # Check if user wants to modify/edit the image
        modification_keywords = ["modify", "edit", "change", "add", "remove", "make", "transform", "convert"]
        if any(kw in query for kw in modification_keywords):
            logger.info("🎨 Image modification requested - routing to image generation")
            return "image_generation"
        
        # Check if user wants to identify/search for similar
        identification_keywords = ["who is", "what is", "identify", "find similar", "search for", "who", "what"]
        if any(kw in query for kw in identification_keywords) or analysis.get("detected_identities"):
            logger.info("🔍 Image identification/search requested - routing to image search")
            return "image_search"
        
        # Otherwise continue with normal flow
        return "normal"
    
    def _route_from_calculation_detection(self, state: ChatState) -> str:
        """Route based on whether calculations are needed"""
        if state.get("needs_calculations", False):
            return "calculate"
        return "respond"
    
    async def _process_attached_images_node(self, state: ChatState) -> Dict[str, Any]:
        """Process attached images for face detection and identification"""
        try:
            shared_memory = state.get("shared_memory", {})
            attached_images = shared_memory.get("attached_images", [])
            user_id = state.get("user_id", "system")
            metadata = state.get("metadata", {})
            
            if not attached_images:
                return {
                    "attached_image_analysis": None,
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", ""),
                }
            
            logger.info(f"📎 Processing {len(attached_images)} attached image(s) for analysis")
            
            # Process first image (for now, we'll process one at a time)
            first_image = attached_images[0]
            attachment_path = first_image.get("storage_path") or first_image.get("file_path")
            
            if not attachment_path:
                logger.warning("⚠️ Attached image missing storage_path")
                return {
                    "attached_image_analysis": {"error": "Image path not found"},
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", ""),
                }
            
            # Import and use attachment processor
            import sys
            import os
            # Add backend to path if needed
            backend_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "backend")
            if backend_path not in sys.path:
                sys.path.insert(0, backend_path)
            
            from services.attachment_processor_service import attachment_processor_service
            
            # Process image
            analysis_result = await attachment_processor_service.process_image_for_search(
                attachment_path=attachment_path,
                user_id=user_id
            )
            
            logger.info(f"✅ Image analysis complete: {analysis_result.get('face_count', 0)} faces, {len(analysis_result.get('detected_identities', []))} identified")
            
            return {
                "attached_image_analysis": analysis_result,
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to process attached images: {e}")
            return {
                "attached_image_analysis": {"error": str(e)},
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
            }
    
    def _is_simple_conversational_query(self, query: str, query_lower: str, messages: List[Any] = None) -> bool:
        """Detect if query is a simple conversational response that doesn't need document search"""
        messages = messages or []
        
        # Check if this appears to be a direct response to the agent's previous message
        # If the last message was from the assistant and this is a short response, likely conversational
        is_follow_up_response = False
        if messages and len(messages) >= 2:
            last_message = messages[-1]
            # Check if last message was from assistant (AIMessage or has role='assistant')
            from langchain_core.messages import AIMessage
            if isinstance(last_message, AIMessage) or getattr(last_message, 'type', '') == 'ai' or getattr(last_message, 'role', '') == 'assistant':
                is_follow_up_response = True
        
        # Very short queries (3 words or less) are likely conversational
        word_count = len(query.split())
        if word_count <= 3:
            # Check if it contains question words or search intent - if so, might need search
            question_words = ["what", "where", "when", "how", "why", "who", "which", "whose"]
            search_keywords = ["find", "search", "look", "show", "tell", "explain", "describe", "list"]
            
            has_question_word = any(qw in query_lower for qw in question_words)
            has_search_keyword = any(sk in query_lower for sk in search_keywords)
            
            # If it has question words or search keywords, might need search
            if has_question_word or has_search_keyword:
                return False
            
            # Very short queries without question/search words are likely conversational
            # Especially if they're responding to the agent's previous message
            if is_follow_up_response:
                logger.info(f"💬 Detected follow-up response to agent message - skipping document search")
            return True
        
        # For longer queries, check if it's just an acknowledgment or confirmation
        # Common patterns: "yes", "no", "thanks", "ok", "got it", "that's it", "exactly", etc.
        import re
        acknowledgment_patterns = [
            r"^(yes|no|ok|okay|sure|yep|nope|thanks|thank you|got it|that's it|exactly|right|correct|perfect|good|great|awesome|nice|cool)$",
            r"^(that's|that is) (it|the one|right|correct|exactly|perfect)$",
            r"^(you're|you are) (right|correct)$",
            r"^(i|I) (think|thought) (so|that|it)$",
            r"^(sounds|looks|seems) (good|great|right|correct|perfect)$"
        ]
        
        for pattern in acknowledgment_patterns:
            if re.match(pattern, query_lower):
                return True
        
        # If this is a follow-up response and the query is short-medium length (4-8 words)
        # and doesn't have question/search keywords, it's likely conversational
        if is_follow_up_response and 4 <= word_count <= 8:
            question_words = ["what", "where", "when", "how", "why", "who", "which", "whose"]
            search_keywords = ["find", "search", "look", "show", "tell", "explain", "describe", "list", "about"]
            
            has_question_word = any(qw in query_lower for qw in question_words)
            has_search_keyword = any(sk in query_lower for sk in search_keywords)
            
            # If no question/search intent, likely conversational
            if not has_question_word and not has_search_keyword:
                logger.info(f"💬 Detected conversational follow-up ({word_count} words) - skipping document search")
                return True
        
        # Check for simple confirmations like "ARBITRARY IS IT!" or "THAT'S THE WORD!"
        # Only match all-caps if it's short (3 words or less) to avoid false positives
        if word_count <= 3:
            # Check for all-caps confirmations (use original query, not lowercased)
            if re.match(r"^[A-Z\s!]+$", query):
                return True
            
            # Check for lowercase confirmations
            confirmation_patterns = [
                r"^(that|this|it)('s| is) (the|a|an) \w+!?$",  # "that's the word!", "it is arbitrary!"
            ]
            
            for pattern in confirmation_patterns:
                if re.match(pattern, query_lower):
                    return True
        
        return False
    
    def _build_chat_prompt(self, persona: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Build system prompt for chat agent"""
        team_context_prefix = ""
        if metadata and metadata.get("team_chat_context"):
            team_context_prefix = (
                "TEAM CONTEXT (the user asked about this team; use this to answer):\n\n"
                + (metadata.get("team_chat_context") or "").strip()
                + "\n\n---\n\n"
            )
        ai_name = persona.get("ai_name", "Alex") if persona else "Alex"
        persona_style = persona.get("persona_style", "professional") if persona else "professional"
        
        # Extract user context from metadata
        preferred_name = metadata.get("user_preferred_name", "") if metadata else ""
        user_context = metadata.get("user_ai_context", "") if metadata else ""
        user_memory = metadata.get("user_memory", "") if metadata else ""
        user_facts = metadata.get("user_facts", "") if metadata else ""
        user_episodes = metadata.get("user_episodes", "") if metadata else ""
        
        # Build style instruction: use custom style_instruction from DB persona when present
        custom_prefs = (persona or {}).get("custom_preferences") or {}
        style_instruction = custom_prefs.get("style_instruction") or (persona or {}).get("style_instruction")
        if not style_instruction:
            style_instruction = self._get_style_instruction(persona_style)
        
        # Build user context sections
        user_context_sections = []
        if preferred_name and preferred_name.strip():
            user_context_sections.append(f"USER PREFERENCE:\nThe user prefers to be addressed as: {preferred_name.strip()}")
        if user_context and user_context.strip():
            user_context_sections.append(f"USER CONTEXT:\n{user_context.strip()}")
        if user_memory and user_memory.strip():
            user_context_sections.append(user_memory.strip())
        else:
            if user_facts and user_facts.strip():
                user_context_sections.append(f"USER FACTS:\n{user_facts.strip()}")
            if user_episodes and user_episodes.strip():
                user_context_sections.append(user_episodes.strip())
        
        user_context_text = "\n\n".join(user_context_sections) + "\n\n" if user_context_sections else ""
        
        base_prompt = f"""{team_context_prefix}You are {ai_name}, a conversational AI assistant. Your role is to have natural conversations while providing accurate, useful information.

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
- General knowledge synthesis and explanations using your training data
- Opinion requests and strategic advice
- Hypothetical scenarios and "what if" questions
- Follow-up questions and clarifications
- Technical discussions using your training knowledge
- Mathematical calculations (the system will automatically calculate for you)

WHAT YOU DON'T HANDLE (signal cannot_answer instead of suggesting):
- Questions requiring search of the user's local documents or files
- Questions needing external web research, citations, or factual verification you can't answer from context
For these, set cannot_answer to true. The system will automatically route to the right specialist. Do NOT suggest the user "use the research agent" — signal cannot_answer and let the system handle it.

VISUALIZATION TOOL:
You have access to a chart generation tool that can create visual representations of data.
Use this tool when:
- Comparing multiple values or categories
- Showing trends over time
- Displaying distributions or proportions
- Data would be clearer as a chart than as text
- User explicitly requests a chart, graph, or visualization

Available chart types: bar, line, pie, scatter, area, heatmap, box_plot, histogram

To use, provide structured data matching the chart type format. The tool will generate an interactive chart that can be embedded in your response.

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
    "task_status": "complete",
    "cannot_answer": false
}}

Set cannot_answer to true ONLY when: (1) the user asks for external research, web citations, or factual information you cannot answer from your training data or provided context, OR (2) the user explicitly asks to search their documents (e.g. "see if we have X", "find any X in my documents").
Set cannot_answer to false for: greetings, comments, thanks, acknowledgments, casual conversation, general knowledge, creative brainstorming, opinions, or anything you CAN answer.
When cannot_answer is true, still provide a brief message if possible (e.g. "Let me look that up for you.").

EXAMPLES:

Simple acknowledgment:
{{
    "message": "You're welcome! Let me know if you need anything else.",
    "task_status": "complete",
    "cannot_answer": false
}}

Detailed response:
{{
    "message": "Here's what I think about that topic...",
    "task_status": "complete",
    "cannot_answer": false
}}

Needs specialist help:
{{
    "message": "Let me look into that for you.",
    "task_status": "complete",
    "cannot_answer": true
}}

CONVERSATION CONTEXT:
You have access to conversation history for context. Use this to understand follow-up questions and maintain conversational flow.

IMAGES:
**CRITICAL RULE**: Do NOT create markdown image links (like ![alt](filename.gif) or ![alt](url)) in your response. The system automatically includes relevant images after your text response. 

**IMPORTANT**: When images are found, you will see a note saying "X images found and will be included". ONLY describe or mention those EXACT X images in your response text. If 2 images are included, mention only 2 items. If 3 images are included, mention all 3. Never mention more images than will be displayed. Just write your conversational reply - images will appear automatically below your message."""

        return base_prompt
    
    async def _check_local_data_node(self, state: ChatState) -> Dict[str, Any]:
        """Check local documents for relevant information using intelligent retrieval subgraph"""
        try:
            query = state.get("query", "")
            query_lower = query.lower().strip()
            messages = state.get("messages", [])
            
            # Skip document retrieval for simple conversational responses
            # These are typically short acknowledgments, confirmations, or brief responses
            # that don't need document search
            is_simple_conversational = self._is_simple_conversational_query(query, query_lower, messages)
            
            if is_simple_conversational:
                logger.info(f"💬 SKIPPING document retrieval: Simple conversational response detected")
                return {
                    "local_data_results": None,
                    # ✅ CRITICAL: Preserve state for subsequent nodes
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", ""),
                    "image_search_results": state.get("image_search_results"),
                    "image_result_count": state.get("image_result_count", 0)
                }
            
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
                small_doc_threshold=10000,  # Increased to handle medium-sized docs
                metadata=metadata,  # Pass metadata for model selection
                messages=state.get("messages", []),  # Pass conversation messages for context-aware detection
                shared_memory=shared_memory  # Pass shared memory for conversation context
            )
            
            logger.info(f"💬 Document retrieval result keys: {list(result.keys()) if result else 'None'}")
            logger.info(f"💬 result.get('success'): {result.get('success') if result else 'N/A'}")
            logger.info(f"💬 result.get('formatted_context') length: {len(result.get('formatted_context', '')) if result and result.get('formatted_context') else 0}")
            
            if result.get("success") and result.get("formatted_context"):
                logger.info(f"💬 Found relevant local documents via intelligent retrieval")
                formatted_context = result.get("formatted_context")
                image_search_results = result.get("image_search_results")  # Base64 images markdown
                retrieval_metadata = result.get("metadata", {})  # Contains total_results count
                total_image_results = retrieval_metadata.get("total_results", 0)
                logger.info(f"💬 formatted_context length: {len(formatted_context)} chars")
                logger.info(f"💬 formatted_context preview (first 200 chars): {formatted_context[:200]}")
                if image_search_results:
                    logger.info(f"💬 image_search_results present: {len(image_search_results)} chars (base64 images)")
                    logger.info(f"💬 Total image results from metadata: {total_image_results}")
                return {
                    "local_data_results": formatted_context,
                    "image_search_results": image_search_results,  # Store base64 images for later appending
                    "image_result_count": total_image_results,  # Store count to ensure we only show matching images
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
                "query": state.get("query", ""),
                "image_search_results": state.get("image_search_results"),  # ✅ Preserve base64 images
                "image_result_count": state.get("image_result_count", 0)  # ✅ Preserve count
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
                "query": state.get("query", ""),
                "image_search_results": state.get("image_search_results"),  # ✅ Preserve base64 images
                "image_result_count": state.get("image_result_count", 0)  # ✅ Preserve count
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
            context_window = int(metadata.get("context_window_size", 20))
            llm_messages = MessagePreprocessor.build_conversational_messages(
                system_prompt=system_prompt,
                user_prompt=state["query"],
                messages_list=messages_list,
                look_back_limit=context_window,
                datetime_context=self._get_datetime_context(state),
                sanitize_ai_responses=False,
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
                "query": state.get("query", ""),
                "local_data_results": state.get("local_data_results"),  # ✅ Preserve for LLM context
                "image_search_results": state.get("image_search_results"),  # ✅ Preserve base64 images
                "image_result_count": state.get("image_result_count", 0)  # ✅ Preserve count to limit images
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
                "query": state.get("query", ""),
                "local_data_results": state.get("local_data_results")  # ✅ Preserve for LLM context
            }
    
    async def _fast_time_response_node(self, state: ChatState) -> Dict[str, Any]:
        """Fast path for simple time/date queries - skip document search, use datetime context directly"""
        try:
            query = state.get("query", "")
            logger.info(f"🕐 FAST TIME RESPONSE: Answering time/date query directly")
            
            # Get datetime context with user's timezone
            datetime_context = self._get_datetime_context(state)
            
            # Build a focused prompt for time/date queries
            metadata = state.get("metadata", {})
            persona = metadata.get("persona")
            ai_name = persona.get("ai_name", "Alex") if persona else "Alex"
            
            system_prompt = f"""You are {ai_name}, a helpful assistant. Answer the user's time/date question directly and concisely.

{datetime_context}

**CRITICAL**: The time shown above is in YOUR LOCAL TIMEZONE (not UTC). When answering, use this local time directly. Do NOT convert to UTC or mention UTC unless the user specifically asks for UTC time.

Answer the user's question about time or date using the information provided above."""
            
            user_prompt = query
            
            # Use LLM to generate response
            llm = self._get_llm(temperature=0.3, state=state)
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await llm.ainvoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse structured output if present, otherwise use raw response
            try:
                import json
                if "```json" in response_text:
                    match = re.search(r'```json\s*\n(.*?)\n```', response_text, re.DOTALL)
                    if match:
                        response_text = match.group(1).strip()
                parsed = json.loads(response_text)
                final_message = parsed.get("message", response_text)
            except:
                final_message = response_text.strip()
            
            # Add assistant response to messages for checkpoint persistence
            state = self._add_assistant_response_to_messages(state, final_message)
            
            # Store primary_agent_selected in shared_memory for conversation continuity
            shared_memory = state.get("shared_memory", {})
            shared_memory["primary_agent_selected"] = "chat_agent"
            shared_memory["last_agent"] = "chat_agent"
            state["shared_memory"] = shared_memory
            
            # Build result matching _generate_response_node format
            return {
                "response": {
                    "response": final_message,  # Use "response" key to match _generate_response_node format
                    "task_status": "complete",
                    "agent_type": "chat_agent"
                },
                "task_status": "complete",
                "messages": state.get("messages", []),
                "shared_memory": shared_memory,
                # ✅ CRITICAL: Preserve state
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "query": state.get("query", "")
            }
            
        except Exception as e:
            logger.error(f"❌ Fast time response failed: {e}")
            return {
                "response": {
                    "response": f"I apologize, but I encountered an error while getting the time.",
                    "task_status": "error",
                    "agent_type": "chat_agent"
                },
                "task_status": "error",
                "error": str(e),
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
                "query": state.get("query", ""),
                "local_data_results": state.get("local_data_results"),  # ✅ Preserve for LLM context
                "image_search_results": state.get("image_search_results"),  # ✅ Preserve base64 images
                "image_result_count": state.get("image_result_count", 0)  # ✅ Preserve count to limit images
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
                "query": state.get("query", ""),
                "local_data_results": state.get("local_data_results"),  # ✅ Preserve for LLM context
                "image_search_results": state.get("image_search_results"),  # ✅ Preserve base64 images
                "image_result_count": state.get("image_result_count", 0)  # ✅ Preserve count to limit images
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
                        "query": state.get("query", ""),
                        "local_data_results": state.get("local_data_results"),  # ✅ Preserve for LLM context
                        "image_search_results": state.get("image_search_results"),  # ✅ Preserve base64 images
                        "image_result_count": state.get("image_result_count", 0)  # ✅ Preserve count to limit images
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
                        "query": state.get("query", ""),
                        "local_data_results": state.get("local_data_results"),  # ✅ Preserve for LLM context
                        "image_search_results": state.get("image_search_results"),  # ✅ Preserve base64 images
                        "image_result_count": state.get("image_result_count", 0)  # ✅ Preserve count to limit images
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
                            "query": state.get("query", ""),
                            "local_data_results": state.get("local_data_results"),  # ✅ Preserve for LLM context
                            "image_search_results": state.get("image_search_results"),  # ✅ Preserve base64 images
                            "image_result_count": state.get("image_result_count", 0)  # ✅ Preserve count to limit images
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
                "query": state.get("query", ""),
                "local_data_results": state.get("local_data_results"),  # ✅ Preserve for LLM context
                "image_search_results": state.get("image_search_results"),  # ✅ Preserve base64 images
                "image_result_count": state.get("image_result_count", 0)  # ✅ Preserve count to limit images
            }
            
        except Exception as e:
            logger.error(f"❌ Calculation failed: {e}")
            return {
                "calculation_result": None,
                "needs_calculations": False,
                "error": str(e),
                # ✅ CRITICAL: Preserve state even on error
                "metadata": state.get("metadata", {}),
                "image_search_results": state.get("image_search_results"),  # ✅ Preserve base64 images
                "image_result_count": state.get("image_result_count", 0),  # ✅ Preserve count to limit images
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "local_data_results": state.get("local_data_results")  # ✅ Preserve for LLM context
            }
    
    async def _generate_response_node(self, state: ChatState) -> Dict[str, Any]:
        """Generate response: call LLM and parse structured output"""
        try:
            logger.info("💬 Generating chat response...")
            
            llm_messages = state.get("llm_messages", [])
            if not llm_messages:
                logger.error(f"❌ CHAT GENERATE: No LLM messages prepared")
                error_response = AgentResponse(
                    response="No LLM messages prepared for chat response",
                    task_status="error",
                    agent_type="chat_agent",
                    timestamp=datetime.now().isoformat(),
                    error="No LLM messages prepared"
                )
                return {
                    "response": error_response.dict(exclude_none=True),
                    "task_status": "error",
                    # ✅ CRITICAL: Preserve state even on error
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", "")
                }
            
            # Chat agent answers from its knowledge - no document retrieval
            # If user needs document search, they should use research agent
            
            # Call LLM - pass state to access user's model selection from metadata
            start_time = datetime.now()
            llm = self._get_llm(temperature=0.7, state=state)
            
            # Check if we have calculation results to include
            calculation_result = state.get("calculation_result")
            if calculation_result:
                calc_value = calculation_result.get("result")
                calc_expression = calculation_result.get("expression", "")
                logger.info(f"💬 Including calculation result in response: {calc_expression} = {calc_value}")
            
            # Call LLM (single call)
            response = await llm.ainvoke(llm_messages)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            response_content = response.content if hasattr(response, 'content') else str(response)
            structured_response = self._parse_json_response(response_content) or {}
            final_message = structured_response.get("message", response_content)
            cannot_answer = bool(structured_response.get("cannot_answer", False))

            query_lower = (state.get("query") or "").lower()
            is_trivial = self._is_simple_conversational_query(
                state.get("query", ""), query_lower, state.get("messages", [])
            )
            has_local_images = bool(state.get("image_search_results") and str(state.get("image_search_results", "")).strip())
            find_my_photos = any(p in query_lower for p in ["find me photos", "find me some photos", "show me photos", "my photos", "my pictures", "my images"])

            if is_trivial:
                cannot_answer = False
            elif has_local_images or (find_my_photos and not any(kw in query_lower for kw in ["research", "search for", "look up", "find out", "investigate"])):
                cannot_answer = False
            elif cannot_answer:
                logger.info("💬 REJECT: Chat cannot answer — returning rejected for orchestrator re-routing")

            if cannot_answer:
                rejected_response = AgentResponse(
                    response=final_message or "Let me find the right specialist for that.",
                    task_status="rejected",
                    agent_type="chat_agent",
                    timestamp=datetime.now().isoformat(),
                )
                return {
                    "response": rejected_response.dict(exclude_none=True),
                    "task_status": "rejected",
                    "shared_memory": state.get("shared_memory", {}),
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "messages": state.get("messages", []),
                    "query": state.get("query", ""),
                    "local_data_results": state.get("local_data_results"),
                    "image_search_results": state.get("image_search_results"),
                    "image_result_count": state.get("image_result_count", 0),
                    "calculation_result": state.get("calculation_result"),
                    "persona": state.get("persona"),
                    "system_prompt": state.get("system_prompt"),
                    "llm_messages": state.get("llm_messages"),
                }
            
            # If calculation was performed, prepend the result
            if calculation_result and calc_value is not None:
                # Prepend calculation result for clarity
                final_message = f"{calc_expression} = {calc_value}\n\n{final_message}"
            
            # Append extracted base64 images to final message (these were excluded from LLM prompt)
            extracted_images = state.get("extracted_images", [])
            logger.info(f"💬 DEBUG: extracted_images in state: {len(extracted_images) if extracted_images else 0}")
            logger.info(f"💬 DEBUG: extracted_images type: {type(extracted_images)}")
            if extracted_images:
                logger.info(f"💬 DEBUG: First image preview (first 100 chars): {extracted_images[0][:100] if extracted_images else 'empty'}")
                # Append base64 images directly (each image already has trailing newline)
                images_text = "\n".join(extracted_images)
                final_message = final_message + images_text
                logger.info(f"💬 Appended {len(extracted_images)} base64 image(s) to response (excluded from LLM prompt)")
            else:
                logger.warning(f"💬 WARNING: No extracted_images found in state to append!")
            
            # Chat agent doesn't do document search, so no images from local_data_results
            
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
            
            # Build standard response using AgentResponse contract
            task_status = structured_response.get("task_status", "complete")
            # Normalize task_status to valid enum value
            if task_status not in ["complete", "incomplete", "permission_required", "error"]:
                logger.warning(f"⚠️ CHAT GENERATE: Invalid task_status '{task_status}', normalizing to 'complete'")
                task_status = "complete"
            
            logger.info(f"📊 CHAT GENERATE: Creating AgentResponse with task_status='{task_status}'")
            standard_response = AgentResponse(
                response=final_message,
                task_status=task_status,
                agent_type="chat_agent",
                timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"✅ Chat response generated in {processing_time:.2f}s")
            logger.info(f"📊 CHAT GENERATE: Response text length: {len(final_message)} chars")
            logger.info(f"📊 CHAT GENERATE: Extracted images: {len(extracted_images) if extracted_images else 0} image(s)")
            
            return {
                "response": standard_response.dict(exclude_none=True),
                "task_status": task_status,
                "messages": state.get("messages", []),
                "shared_memory": shared_memory,
                # ✅ CRITICAL: Preserve critical state keys
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "query": state.get("query", "")
            }
            
        except Exception as e:
            logger.error(f"❌ CHAT GENERATE: Response generation failed: {e}")
            import traceback
            logger.error(f"❌ CHAT GENERATE: Traceback: {traceback.format_exc()}")
            # Return standard error response
            error_response = AgentResponse(
                response=f"Chat response generation failed: {str(e)}",
                task_status="error",
                agent_type="chat_agent",
                timestamp=datetime.now().isoformat(),
                error=str(e)
            )
            return {
                "response": error_response.dict(exclude_none=True),
                "task_status": "error",
                # ✅ CRITICAL: Preserve critical state keys even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    async def process(self, query: str, metadata: Dict[str, Any] = None, messages: List[Any] = None) -> Dict[str, Any]:
        """Process chat query using LangGraph workflow"""
        try:
            metadata = metadata or {}
            messages = messages or []
            
            query_preview = query[:100] + "..." if len(query) > 100 else query
            logger.info(f"📥 CHAT PROCESS: Starting chat agent - query: {query_preview}")
            
            # Get workflow (lazy initialization with checkpointer)
            workflow = await self._get_workflow()
            
            # Extract user_id from metadata
            user_id = metadata.get("user_id", "system")
            
            # Get checkpoint config (handles thread_id from conversation_id/user_id)
            config = self._get_checkpoint_config(metadata)
            
            # Prepare new messages (current query)
            new_messages = self._prepare_messages_with_query(messages, query)
            
            # Load and merge checkpointed messages to preserve conversation history
            context_window = int(metadata.get("context_window_size", 20))
            conversation_messages = await self._load_and_merge_checkpoint_messages(
                workflow, config, new_messages, look_back_limit=context_window
            )
            
            # Load shared_memory from checkpoint if available
            checkpoint_state = await workflow.aget_state(config)
            existing_shared_memory = {}
            if checkpoint_state and checkpoint_state.values:
                existing_shared_memory = checkpoint_state.values.get("shared_memory", {})
            
            # Merge shared_memory: NEW data from metadata overwrites OLD checkpoint data
            # This ensures fresh active_editor content overwrites stale cached content
            shared_memory = metadata.get("shared_memory", {}) or {}
            shared_memory_merged = existing_shared_memory.copy()
            shared_memory_merged.update(shared_memory)
            
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
                "task_status": "complete",
                "error": "",
                "shared_memory": shared_memory_merged,
            }
            
            # Run LangGraph workflow with checkpointing
            logger.info(f"🔄 CHAT PROCESS: Invoking LangGraph workflow")
            result_state = await workflow.ainvoke(initial_state, config=config)
            logger.info(f"✅ CHAT PROCESS: Workflow completed - extracting response from state")
            
            # Extract final response
            response = result_state.get("response", {})
            task_status = result_state.get("task_status", "complete")
            result_shared_memory = result_state.get("shared_memory", {})
            
            if task_status == "rejected":
                logger.info("📤 CHAT PROCESS: Returning rejection — orchestrator will re-route")
                return response if isinstance(response, dict) else {"task_status": "rejected", "response": str(response)}

            if task_status not in ["complete", "incomplete", "permission_required", "error"]:
                logger.warning(f"⚠️ CHAT PROCESS: Invalid task_status '{task_status}', normalizing to 'complete'")
                task_status = "complete"
            
            logger.info(f"📊 CHAT PROCESS: Extracted response from state - type: {type(response)}, is_dict: {isinstance(response, dict)}")
            
            # Check if response is already in standard AgentResponse format
            # Standard format has: response, task_status, agent_type, timestamp
            if isinstance(response, dict) and all(key in response for key in ["response", "task_status", "agent_type", "timestamp"]):
                logger.info(f"📊 CHAT PROCESS: Response is already in standard format")
                logger.info(f"📊 CHAT PROCESS: Response dict keys: {list(response.keys())}")
                logger.info(f"📊 CHAT PROCESS: Response dict has 'response' key: {'response' in response}")
                logger.info(f"📤 CHAT PROCESS: Returning full AgentResponse dict (standard format)")
                return response
            
            # Legacy format - extract and reconstruct
            logger.warning(f"⚠️ CHAT PROCESS: Response is in legacy format, extracting...")
            if isinstance(response, dict):
                # Handle nested response structure
                nested_response = response.get("response", {})
                if isinstance(nested_response, dict):
                    response_text = nested_response.get("response", "No response generated")
                else:
                    response_text = str(nested_response) if nested_response else "No response generated"
            else:
                response_text = str(response) if response else "No response generated"
            
            logger.info(f"📊 CHAT PROCESS: Extracted response text length: {len(response_text)} chars")
            
            # Build standard response
            standard_response = AgentResponse(
                response=response_text,
                task_status=task_status,
                agent_type="chat_agent",
                timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"📤 CHAT PROCESS: Returning standard AgentResponse dict (reconstructed from legacy)")
            return standard_response.dict(exclude_none=True)
            
        except Exception as e:
            logger.error(f"❌ CHAT PROCESS: Exception in process method: {e}")
            import traceback
            logger.error(f"❌ CHAT PROCESS: Traceback: {traceback.format_exc()}")
            # Return standard error response
            error_response = AgentResponse(
                response=f"Chat processing failed: {str(e)}",
                task_status="error",
                agent_type="chat_agent",
                timestamp=datetime.now().isoformat(),
                error=str(e)
            )
            logger.info(f"📤 CHAT PROCESS: Returning error response (standard format) after exception")
            return error_response.dict(exclude_none=True)

