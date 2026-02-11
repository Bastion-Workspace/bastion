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
from pydantic import BaseModel, Field
from .base_agent import BaseAgent, TaskStatus
from orchestrator.models.agent_response_contract import AgentResponse

logger = logging.getLogger(__name__)


class HandoffDecision(BaseModel):
    """Structured output for LLM-based handoff-to-research decision"""
    should_handoff_to_research: bool = Field(
        description="True only if this is a new factual/research question that requires external research and cannot be answered from local context. False for comments, thanks, acknowledgments, or follow-ups that do not ask for new external information."
    )
    reason: Optional[str] = Field(default=None, description="Brief reason for the decision")


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
    image_search_results: Optional[str]  # Base64-encoded images from image search
    image_result_count: int  # Count of images returned
    extracted_images: List[str]  # List of base64 image markdown strings to append
    attached_image_analysis: Optional[Dict[str, Any]]  # Analysis results for attached images
    response: Dict[str, Any]
    task_status: str
    error: str
    shared_memory: Dict[str, Any]  # For storing primary_agent_selected and continuity data
    should_handoff_to_research: Optional[bool]  # Set by LLM handoff decision node
    handoff_response: Optional[Dict[str, Any]]  # Research result when handoff was taken


class ChatAgent(BaseAgent):
    """Chat agent for general conversation and knowledge queries"""
    
    def __init__(self):
        super().__init__("chat_agent")
        logger.debug("💬 Chat Agent ready for conversation!")
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """Build LangGraph workflow for chat agent"""
        workflow = StateGraph(ChatState)
        
        # Add nodes
        workflow.add_node("prepare_context", self._prepare_context_node)
        workflow.add_node("process_attached_images", self._process_attached_images_node)
        workflow.add_node("fast_time_response", self._fast_time_response_node)
        workflow.add_node("check_local_data", self._check_local_data_node)
        workflow.add_node("detect_calculations", self._detect_calculations_node)
        workflow.add_node("perform_calculations", self._perform_calculations_node)
        workflow.add_node("call_research_agent", self._call_research_agent_node)
        workflow.add_node("generate_response", self._generate_response_node)
        
        # Entry point
        workflow.set_entry_point("prepare_context")
        
        # Route from prepare_context: check for attached images, then fast path or normal flow
        workflow.add_conditional_edges(
            "prepare_context",
            self._route_from_prepare_context,
            {
                "process_images": "process_attached_images",
                "fast_time": "fast_time_response",
                "normal": "check_local_data"  # Run local retrieval (incl. image/object search) before handoff decision
            }
        )
        
        # After processing images, route based on intent
        workflow.add_conditional_edges(
            "process_attached_images",
            self._route_from_image_processing,
            {
                "image_search": "check_local_data",  # Check local data before generating response
                "image_generation": "generate_response",  # Will trigger image generation
                "normal": "check_local_data"
            }
        )
        
        # Fast time response goes directly to END
        workflow.add_edge("fast_time_response", END)
        
        # After checking local data, always go to detect_calculations (handoff decided by LLM later)
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
        
        # After calculations, go to generate_response (handoff decided inside that node)
        workflow.add_edge("perform_calculations", "generate_response")
        
        # After generate_response: if it requested handoff, call Research; else END
        workflow.add_conditional_edges(
            "generate_response",
            self._route_from_generate_response,
            {
                "call_research": "call_research_agent",
                "end": END
            }
        )
        
        workflow.add_edge("call_research_agent", END)
        
        # Compile with checkpointer for state persistence
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
- General knowledge synthesis and explanations using your training data
- Opinion requests and strategic advice
- Hypothetical scenarios and "what if" questions
- Follow-up questions and clarifications
- Technical discussions using your training knowledge
- Mathematical calculations (the system will automatically calculate for you)

WHAT YOU DON'T HANDLE (hand off instead of suggesting):
- Questions requiring search of the user's local documents or files: do NOT suggest they "use the research agent". Set handoff_to_research to TRUE so we call the Research agent for them (Research does local + web). Only answer from context when we already have strong local results and the user asked a simple follow-up.

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

HANDOFF TO RESEARCH:
- Set handoff_to_research to TRUE when: (1) the user asks for EXTERNAL research (web, citations) you cannot answer from context, OR (2) the user explicitly asks to research or search their documents (e.g. "research and see if we have X", "see if we have any X", "find any X in my documents"). We will then call the Research agent (it does local + web). Do NOT reply with "use the research agent" — hand off instead.
- Set handoff_to_research to FALSE for: comments, thanks, acknowledgments, casual conversation, or when you already have strong local results and are giving a direct answer. Do NOT hand off for "find my photos" / "show me my pictures" (local image browse only).

STRUCTURED OUTPUT REQUIREMENT:
You MUST respond with valid JSON matching this schema:
{{
    "message": "Your conversational response (empty or brief if handoff_to_research is true)",
    "task_status": "complete",
    "handoff_to_research": false
}}

When handoff_to_research is true, leave message empty or a brief placeholder; we will call Research and replace it.

EXAMPLES:

Simple acknowledgment (no handoff):
{{
    "message": "You're welcome! Let me know if you need anything else.",
    "task_status": "complete",
    "handoff_to_research": false
}}

Detailed response (no handoff):
{{
    "message": "Here's what I think about that topic...",
    "task_status": "complete",
    "handoff_to_research": false
}}

Needs external research (hand off):
{{
    "message": "",
    "task_status": "complete",
    "handoff_to_research": true
}}

CONVERSATION CONTEXT:
You have access to conversation history for context. Use this to understand follow-up questions and maintain conversational flow.

IMAGES:
**CRITICAL RULE**: Do NOT create markdown image links (like ![alt](filename.gif) or ![alt](url)) in your response. The system automatically includes relevant images after your text response. 

**IMPORTANT**: When images are found, you will see a note saying "X images found and will be included". ONLY describe or mention those EXACT X images in your response text. If 2 images are included, mention only 2 items. If 3 images are included, mention all 3. Never mention more images than will be displayed. Just write your conversational reply - images will appear automatically below your message."""

        return base_prompt
    
    def _route_from_generate_response(self, state: ChatState) -> str:
        """Route based on main response LLM handoff decision: call Research or END"""
        if state.get("should_handoff_to_research") is True:
            logger.info("💬 HANDOFF: Main response LLM requested handoff to research_agent")
            return "call_research"
        return "end"
    
    async def _assess_handoff_need_node(self, state: ChatState) -> Dict[str, Any]:
        """LLM-based assessment: should Chat hand off to Research for this query?"""
        try:
            query = state.get("query", "")
            local_data_results = state.get("local_data_results") or ""
            messages = state.get("messages", [])
            shared_memory = state.get("shared_memory", {})
            
            # Build context for LLM: query, local data summary, last turn
            local_summary = "none"
            if local_data_results and str(local_data_results).strip():
                local_summary = str(local_data_results)[:500] + ("..." if len(str(local_data_results)) > 500 else "")
            
            last_turn = ""
            for m in reversed(messages[-4:]):
                if hasattr(m, "content") and getattr(m, "type", None) != "human":
                    role = getattr(m, "type", "") or (getattr(m, "name", "") and "human" or "assistant")
                    if role == "human":
                        last_turn = f"User: {m.content[:200]}" + ("..." if len(str(m.content)) > 200 else "") + "\n" + last_turn
                    else:
                        last_turn = f"Assistant: {str(m.content)[:200]}..." + "\n" + last_turn
                elif hasattr(m, "content"):
                    last_turn = str(m.content)[:300] + "\n" + last_turn
            if not last_turn:
                last_turn = "(no prior messages)"
            
            # Do NOT hand off when user is asking for their own photos/images (local search, not web research)
            # BUT: Allow handoff if user explicitly requests research/search
            image_search_results = state.get("image_search_results") or ""
            has_local_images = bool(image_search_results and str(image_search_results).strip())
            
            # Detect explicit research requests (these override local-only detection)
            query_lower_handoff = query.lower()
            research_keywords = ["research", "search for", "look up", "find out", "investigate"]
            explicit_research_request = any(kw in query_lower_handoff for kw in research_keywords)
            
            # Detect local-only photo queries (only if NOT an explicit research request)
            find_my_photos_patterns = [
                "find me photos", "find me some photos", "show me photos", "show me pictures", "get me photos",
                "my photos", "my pictures", "my images"
            ]
            is_find_my_photos = (
                not explicit_research_request and
                any(p in query_lower_handoff for p in find_my_photos_patterns)
            )

            prompt = f"""You are deciding whether the Chat agent should hand off to the Research agent (external web search and citations).

CURRENT USER QUERY: {query}

LOCAL DATA AVAILABLE (from user's documents): {local_summary}

RECENT CONVERSATION (last turn): {last_turn}

RULES:
- Set should_handoff_to_research to TRUE only if: the user is asking a NEW factual or research question that requires EXTERNAL information (web, citations) and you cannot answer it from the local data above.
- Set should_handoff_to_research to FALSE for: comments, thanks, acknowledgments ("ok", "thanks", "that's helpful"), or follow-ups that do NOT ask for new external research (e.g. "what about X?" when X is a clarification, not a new research question).
- Set should_handoff_to_research to FALSE for: queries asking to find the user's OWN photos or images (e.g. "find me photos with X", "photos with X in them", "my photos of X", "show me pictures with Y"). Those are LOCAL photo search; the system already checked local data — do NOT hand off to Research for web search.
- Do NOT hand off for casual conversation or when local data already answers the question.

Return valid JSON only: {{ "should_handoff_to_research": true or false, "reason": "brief reason or null" }}"""
            
            llm = self._get_llm(temperature=0.0, state=state)
            try:
                structured_llm = llm.with_structured_output(HandoffDecision)
                decision = await structured_llm.ainvoke([HumanMessage(content=prompt)])
            except Exception:
                response = await llm.ainvoke([HumanMessage(content=prompt)])
                content = response.content if hasattr(response, "content") else str(response)
                parsed = self._parse_json_response(content) or {}
                decision = HandoffDecision(
                    should_handoff_to_research=bool(parsed.get("should_handoff_to_research", False)),
                    reason=parsed.get("reason")
                )
            
            should_handoff = decision.should_handoff_to_research
            # Override: never hand off for "find my photos" queries (local search) or when we already have local image results
            # BUT: Explicit research requests always allow handoff
            if explicit_research_request and should_handoff:
                logger.info("💬 HANDOFF ALLOWED: Explicit research request detected — allowing handoff to Research")
            elif has_local_images:
                should_handoff = False
                logger.info("💬 HANDOFF OVERRIDE: Local image results present — not handing off to Research")
            elif is_find_my_photos:
                should_handoff = False
                logger.info("💬 HANDOFF OVERRIDE: Query is find-my-photos (local search) — not handing off to Research")
            else:
                logger.info(f"💬 HANDOFF DECISION: should_handoff_to_research={should_handoff}, reason={decision.reason}")

            return {
                "should_handoff_to_research": should_handoff,
                "shared_memory": shared_memory,
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "messages": messages,
                "query": query,
                "local_data_results": state.get("local_data_results"),
                "image_search_results": state.get("image_search_results"),
                "image_result_count": state.get("image_result_count", 0),
                "calculation_result": state.get("calculation_result"),
                "persona": state.get("persona"),
                "system_prompt": state.get("system_prompt"),
                "llm_messages": state.get("llm_messages"),
            }
            
        except Exception as e:
            logger.error(f"❌ Handoff assessment failed: {e}")
            return {
                "should_handoff_to_research": False,
                "shared_memory": state.get("shared_memory", {}),
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "local_data_results": state.get("local_data_results"),
                "image_search_results": state.get("image_search_results"),
                "image_result_count": state.get("image_result_count", 0),
            }
    
    async def _call_research_agent_node(self, state: ChatState) -> Dict[str, Any]:
        """Call Research agent and return its response. Same as being routed to Research: Research runs (local + web if permitted)."""
        try:
            shared_memory = state.get("shared_memory", {})
            from orchestrator.agents import get_full_research_agent

            research_agent = get_full_research_agent()
            query = state.get("query", "")
            metadata = state.get("metadata", {})
            messages = state.get("messages", [])

            # Pass full shared_memory so Research sees web_search_permission and behaves like direct routing
            handoff_shared_memory = {
                "user_chat_model": metadata.get("user_chat_model"),
                "handoff_context": {
                    "source_agent": "chat_agent",
                    "handoff_type": "quick_lookup",
                    "conversation_context": "User asked a question Chat could not answer from local context; Research is providing cited answer.",
                },
            }
            research_shared_memory = {**shared_memory, **handoff_shared_memory}
            research_metadata = {
                "user_id": state.get("user_id"),
                "conversation_id": metadata.get("conversation_id"),
                "persona": metadata.get("persona"),
                "user_chat_model": metadata.get("user_chat_model"),
                "shared_memory": research_shared_memory,
            }

            logger.info(f"💬 HANDOFF: Calling research_agent for: {query[:80]}... (same as direct Research routing)")
            research_result = await research_agent.process(
                query=query,
                metadata=research_metadata,
                messages=messages,
            )
            
            shared_memory["primary_agent_selected"] = "research_agent"
            shared_memory["last_agent"] = "research_agent"
            
            return {
                "response": research_result if isinstance(research_result, dict) else research_result.dict(exclude_none=True) if hasattr(research_result, "dict") else {"response": str(research_result), "agent_type": "research_agent", "task_status": "complete", "timestamp": datetime.now().isoformat()},
                "handoff_response": research_result if isinstance(research_result, dict) else research_result.dict(exclude_none=True) if hasattr(research_result, "dict") else None,
                "shared_memory": shared_memory,
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
            }
            
        except Exception as e:
            logger.error(f"❌ Research handoff failed: {e}")
            shared_memory = state.get("shared_memory", {})
            error_response = AgentResponse(
                response=f"Research handoff failed: {str(e)}. You can try asking: \"Do some research and [your query]\".",
                task_status="error",
                agent_type="chat_agent",
                timestamp=datetime.now().isoformat(),
                error=str(e),
            )
            return {
                "response": error_response.dict(exclude_none=True),
                "handoff_response": None,
                "shared_memory": shared_memory,
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
            }
    
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
            
            # Parse structured response (message + optional handoff_to_research)
            response_content = response.content if hasattr(response, 'content') else str(response)
            structured_response = self._parse_json_response(response_content) or {}
            final_message = structured_response.get("message", response_content)
            handoff_to_research = bool(structured_response.get("handoff_to_research", False))

            # Overrides: never hand off for local-only cases; force handoff for explicit research/document-lookup phrasing
            query_lower = (state.get("query") or "").lower()
            has_local_images = bool(state.get("image_search_results") and str(state.get("image_search_results", "")).strip())
            find_my_photos = any(p in query_lower for p in ["find me photos", "find me some photos", "show me photos", "my photos", "my pictures", "my images"])
            explicit_research = any(kw in query_lower for kw in ["research", "search for", "look up", "find out", "investigate"])
            document_lookup_phrases = ["research and ", "see if we have", "check if we have", "do we have any", "find any "]
            explicit_document_lookup = any(p in query_lower for p in document_lookup_phrases)
            if explicit_document_lookup and not find_my_photos:
                handoff_to_research = True
                logger.info("💬 HANDOFF: Explicit document-lookup phrasing — forcing handoff to Research (do not suggest user switch)")
            elif explicit_research and handoff_to_research:
                logger.info("💬 HANDOFF: Explicit research request — allowing handoff to Research")
            elif has_local_images:
                handoff_to_research = False
                logger.info("💬 HANDOFF OVERRIDE: Local image results present — not handing off")
            elif find_my_photos and not explicit_research:
                handoff_to_research = False
                logger.info("💬 HANDOFF OVERRIDE: Find-my-photos (local search) — not handing off")
            elif handoff_to_research:
                logger.info("💬 HANDOFF: Main response LLM requested handoff to Research")

            if handoff_to_research:
                shared_memory = state.get("shared_memory", {})
                return {
                    "should_handoff_to_research": True,
                    "shared_memory": shared_memory,
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
            conversation_messages = await self._load_and_merge_checkpoint_messages(
                workflow, config, new_messages
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
                "task_status": "complete",  # Initialize with valid enum value
                "error": "",
                "shared_memory": shared_memory_merged,
                "should_handoff_to_research": None,
                "handoff_response": None,
            }
            
            # Run LangGraph workflow with checkpointing
            logger.info(f"🔄 CHAT PROCESS: Invoking LangGraph workflow")
            result_state = await workflow.ainvoke(initial_state, config=config)
            logger.info(f"✅ CHAT PROCESS: Workflow completed - extracting response from state")
            
            # Extract final response
            response = result_state.get("response", {})
            task_status = result_state.get("task_status", "complete")
            result_shared_memory = result_state.get("shared_memory", {})
            
            # Handoff path: Research was called; return Research response with continuity so last_agent = research_agent
            is_handoff = (
                isinstance(response, dict)
                and response.get("agent_type") == "research_agent"
            ) or result_state.get("handoff_response") is not None
            if is_handoff and isinstance(response, dict) and response.get("agent_type") == "research_agent":
                logger.info(f"📤 CHAT PROCESS: Returning Research handoff response (continuity: research_agent)")
                return {
                    **response,
                    "shared_memory": {
                        **result_shared_memory,
                        "primary_agent_selected": "research_agent",
                        "last_agent": "research_agent",
                    },
                }
            
            # Normalize task_status to valid enum value
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

