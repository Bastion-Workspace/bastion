"""
Chat Agent Implementation
Handles general conversation and local knowledge queries
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from .base_agent import BaseAgent
from models.agent_response_models import ChatResponse, TaskStatus

logger = logging.getLogger(__name__)


class ChatAgent(BaseAgent):
    """Chat agent for general conversation and local knowledge queries"""
    
    def __init__(self):
        super().__init__("chat_agent")
    
    def _build_chat_prompt(self, persona: Optional[Dict[str, Any]] = None) -> str:
        """Build system prompt for chat agent with collaboration awareness"""
        ai_name = persona.get("ai_name", "Codex") if persona else "Codex"
        base_prompt = f"""You are {ai_name}, a helpful and engaging conversational AI assistant with concierge capabilities. Your role is to have natural conversations while suggesting specialized help when appropriate.

CONVERSATION GUIDELINES:
1. **BE APPROPRIATELY RESPONSIVE**: Match your response length to the user's input - brief acknowledgments get brief responses
2. **MAINTAIN CONTEXT**: Use conversation history to understand follow-up questions and maintain flow
3. **ASK FOR CLARIFICATION**: If a question is unclear, ask for more details
4. **BE CONCISE AND NATURAL**: Provide appropriate conversational responses - simple thanks gets simple acknowledgment
5. **STAY CONVERSATIONAL**: Focus on dialogue, not information retrieval
6. **ENHANCEMENT SUGGESTIONS**: Only suggest specialized agents to ENHANCE your complete response, never to replace it

RESPONSE LENGTH GUIDELINES:
- **Simple acknowledgments** ("thanks", "thank you"): Brief friendly response (1-2 sentences)
- **Questions or requests**: Helpful detailed responses
- **Complex topics**: Thorough explanations with context
- **Casual conversation**: Natural, proportionate responses

WHAT YOU HANDLE CONFIDENTLY:
- Greetings and casual conversation (including brief acknowledgments)
- Creative brainstorming and idea generation  
- General knowledge synthesis and explanations
- Opinion requests and strategic advice
- Hypothetical scenarios and "what if" questions
- Follow-up questions and clarifications
- Emotional support and encouragement
- General advice and recommendations
- Technical discussions using your training knowledge

ENHANCEMENT OPPORTUNITIES (suggest only to ADD VALUE to your complete response):
- **RESEARCH**: After providing your answer, offer deeper research if it could add specific facts, current data, or document verification
- **WEATHER**: After discussing travel/locations, offer specific forecasts if helpful
- **DATA FORMATTING**: After providing information, offer table formatting if data is complex

CONFIDENCE PRINCIPLE:
- ALWAYS provide a complete, helpful response first
- ONLY suggest collaboration as optional enhancement: "I've given you X, but I could also..."  
- NEVER defer or handoff - you are fully capable of handling conversational tasks

STRUCTURED OUTPUT REQUIREMENT:
You MUST respond with valid JSON matching this schema:
{{
    "message": "Your conversational response",
    # ROOSEVELT'S NATURAL COLLABORATION: Removed collaboration suggestion field
    # ROOSEVELT'S NATURAL COLLABORATION: Removed collaboration fields
}}

ENHANCEMENT EXAMPLES:
✅ APPROPRIATE ENHANCEMENTS (after giving complete response):
- Creative brainstorming → "I've given you some ideas from my knowledge. Would you like me to research specific domains for even more examples?"
- Travel discussion → "I've covered the basics. Would you like me to check specific weather forecasts for your dates?"
- Complex data in response → "I've provided the information. Would you like me to format this as a comparison table?"

❌ NEVER DEFER OR HANDOFF:
- Don't say "Research Agent should handle this" - YOU handle it first
- Don't suggest collaboration instead of answering - answer THEN optionally enhance
- Don't lack confidence in your conversational abilities
- Don't treat collaboration as routing correction - that's Intent Classifier's job

CONVERSATION CONTEXT:
You have access to conversation history for context. Use this to understand follow-up questions and maintain conversational flow."""

        # Add persona if available
        persona_prompt = self._build_persona_prompt(persona)
        return base_prompt + persona_prompt
    
    def _build_chat_prompt_with_reminders(self, persona: Optional[Dict[str, Any]] = None, pending_operations: List[Dict[str, Any]] = None) -> str:
        """Build chat prompt with editor context and pending operations awareness"""
        # Start with base chat prompt
        base_prompt = self._build_chat_prompt(persona)
        
        # Add editor context if available and allowed
        editor_context = ""
        if hasattr(self, '_active_editor') and self._active_editor:
            editor_content = self._active_editor.get("content", "")
            editor_filename = self._active_editor.get("filename", "document")
            editor_type = self._active_editor.get("frontmatter", {}).get("type", "document")
            
            if editor_content:
                # Truncate content for prompt (keep it reasonable)
                content_preview = editor_content[:2000] + "..." if len(editor_content) > 2000 else editor_content
                editor_context = f"""

**EDITOR CONTEXT** (You have access to the current document):
- **File**: {editor_filename}
- **Type**: {editor_type}
- **Content Preview**: {content_preview}

Use this context to provide relevant, document-aware responses when appropriate."""
        
        # Add pending operations awareness
        pending_context = ""
        if pending_operations:
            pending_context = f"""

**PENDING OPERATIONS**: You have {len(pending_operations)} pending operations that may need attention.
Consider mentioning these if relevant to the conversation."""
        
        return base_prompt + editor_context + pending_context
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process chat query with smart pending operation awareness"""
        try:
            logger.info("💬 Chat agent processing...")
            
            # Check for pending operations in shared memory for smart reminders
            shared_memory = state.get("shared_memory", {})
            pending_operations = shared_memory.get("pending_operations", [])
            
            # BULLY! Check editor preference - only access editor context if allowed
            editor_preference = shared_memory.get("editor_preference", "").lower()
            active_editor = None
            if editor_preference != "ignore":
                active_editor = shared_memory.get("active_editor", {})
                if active_editor and not active_editor.get("is_editable"):
                    active_editor = None  # Don't use non-editable editor context
            
            # Store shared memory for prompt building and conversation context
            self._current_shared_memory = shared_memory
            self._active_editor = active_editor  # Store editor context for prompt building
            
            # ROOSEVELT'S CONVERSATION INTELLIGENCE: Check built-in state context first
            query = self._extract_current_user_query(state)
            # Store user query for smart fallback parsing
            self._last_user_query = query
            cached_context = await self._get_relevant_context(query, state)
            
            if cached_context and self._should_use_cached_context(query, state):
                logger.info("🏆 CHAT CACHE: Using conversation intelligence for follow-up work")
                # Use built-in conversation intelligence instead of extracting manually
                self._conversation_cache = {"has_relevant_context": True, "cached_content": cached_context}
            else:
                # Fallback to manual extraction for backward compatibility
                conversation_cache = self._extract_conversation_cache(state)
                self._conversation_cache = conversation_cache
            
            # Build system prompt for chat with pending operations and research context awareness
            system_prompt = self._build_chat_prompt_with_reminders(state.get("persona"), pending_operations)
            
            # ROOSEVELT'S CONVERSATION HISTORY: Provide full conversation history to LLM
            # Use full context to maintain conversation continuity
            messages = await self._prepare_messages_with_full_context(state, system_prompt)
            
            # Call LLM with structured output for collaboration awareness
            start_time = datetime.now()
            chat_service = await self._get_chat_service()
            model_name = await self._get_model_name()
            response = await chat_service.openai_client.chat.completions.create(
                messages=messages,
                model=model_name,
                temperature=0.7
                # NO tools - structured conversation with collaboration awareness
            )
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Parse structured response
            response_content = response.choices[0].message.content
            structured_response = self._parse_structured_chat_response(response_content)
            
            # Extract components
            final_answer = structured_response.message
            # ROOSEVELT'S NATURAL COLLABORATION: Removed collaboration suggestion fields
            # Let the LLM handle collaboration decisions with full conversation context
            
            # Add smart reminder if appropriate
            final_answer = self._add_smart_reminder(final_answer, pending_operations, state)
            
            # ROOSEVELT'S UNIVERSAL FORMATTING: Apply intelligent formatting if beneficial
            messages = state.get("messages", [])
            user_query = messages[-1].content if messages else ""
            final_answer = await self._apply_universal_formatting(user_query, final_answer)
            
            # Update state with conversational response and collaboration data
            state["agent_results"] = {
                "agent_type": "chat_agent_enhanced",
                "response": final_answer,
                "structured_response": structured_response.dict(),
                "tools_used": [],  # No tools used
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                "conversation_only": True,  # Mark as pure conversation
                "pending_operations_count": len(pending_operations),
                # ROOSEVELT'S NATURAL COLLABORATION: Removed collaboration fields
                # Let the LLM handle collaboration decisions with full conversation context
            }
            
            # ROOSEVELT'S NATURAL COLLABORATION: Let the LLM handle collaboration decisions with full context
            shared_memory = state.get("shared_memory", {})
            shared_memory["chat_agent_context"] = {
                "latest_response": final_answer[:1000],  # Store recent work for context
                "timestamp": datetime.now().isoformat(),
                "response_type": "conversational"
            }
            
            # ROOSEVELT'S CONVERSATION INTELLIGENCE: Let the LLM handle context naturally
            # No brittle pattern matching - trust the LLM's semantic understanding
            
            state["shared_memory"] = shared_memory
            logger.info(f"💬 CONTEXT: Stored chat agent work for natural conversation flow")
            
            # Store minimal insights - no search results since this is pure conversation
            if "agent_insights" not in state:
                state["agent_insights"] = {}
            
            state["agent_insights"]["chat_agent"] = {
                "local_results_found": 0,  # No searching performed
                "tools_used": [],
                "response_mentions_web_search": False,  # Pure conversation
                "escalated": False,
                "confidence_level": 1.0,  # High confidence for conversation
                "conversation_only": True,
                "pending_operations_awareness": len(pending_operations) > 0
            }
            
            # ROOSEVELT'S PURE LANGGRAPH: Add chat response to LangGraph state messages
            if final_answer and final_answer.strip():
                from langchain_core.messages import AIMessage
                state.setdefault("messages", []).append(AIMessage(content=final_answer))
                logger.info(f"✅ CHAT AGENT: Added chat response to LangGraph messages")
            
            state["is_complete"] = True
            logger.info(f"✅ Chat agent completed enhanced conversational response in {processing_time:.2f}s")
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Chat agent error: {e}")
            state["error_message"] = str(e)
            state["is_complete"] = True
            return state
    

    def _extract_conversation_cache(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """ROOSEVELT'S CONVERSATION CACHE: Extract relevant context from conversation history"""
        try:
            messages = state.get("messages", [])
            shared_memory = state.get("shared_memory", {})
            
            # Look for previous chat agent work in conversation
            chat_context = []
            questions_generated = []
            
            for msg in messages[-10:]:  # Last 10 messages
                if hasattr(msg, 'type') and msg.type == "ai" and hasattr(msg, 'content'):
                    content = msg.content
                    
                    # Look for generated questions or brainstorming content
                    if any(indicator in content.lower() for indicator in [
                        "here are some", "questions:", "examples:", "ideas:",
                        "interview questions", "you could ask", "consider these"
                    ]):
                        chat_context.append({
                            "type": "generated_content",
                            "content": content[:500],  # First 500 chars
                            "timestamp": "recent"
                        })
                        
                        # Extract questions if they exist
                        lines = content.split('\n')
                        for line in lines:
                            if '?' in line and len(line.strip()) > 10:
                                questions_generated.append(line.strip())
            
            return {
                "has_relevant_context": len(chat_context) > 0,
                "chat_context": chat_context,
                "questions_generated": questions_generated[-10:],  # Last 10 questions
                "context_summary": f"Found {len(chat_context)} previous chat outputs, {len(questions_generated)} questions"
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to extract conversation cache: {e}")
            return {"has_relevant_context": False}
    
    # ROOSEVELT'S PRINCIPLE: No brittle pattern matching - trust LLM intelligence
    # Removed _extract_destination_context method - let conversation flow naturally
    
    def _clean_control_characters(self, text: str) -> str:
        """ROOSEVELT'S JSON SANITIZER: Remove control characters that break JSON parsing"""
        try:
            import re
            
            # Remove control characters except newlines and tabs
            # Keep \n (newline) and \t (tab) but remove other control chars
            cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
            
            # Also clean up any escaped control characters in the JSON
            cleaned = cleaned.replace('\\x', '\\\\x')  # Escape literal \x sequences
            
            return cleaned
            
        except Exception as e:
            logger.error(f"❌ Control character cleaning failed: {e}")
            return text  # Return original if cleaning fails
    
    def _build_chat_prompt_with_reminders(self, persona: Optional[Dict[str, Any]], pending_operations: list) -> str:
        """Build chat prompt with pending operations and shared context awareness"""
        base_prompt = self._build_chat_prompt(persona)
        
        # Add shared memory context - ROOSEVELT'S AGENT INTELLIGENCE SHARING
        shared_memory = getattr(self, '_current_shared_memory', {})
        
        # ROOSEVELT'S CONVERSATION INTELLIGENCE: Include built-in conversation context
        conversation_cache = getattr(self, '_conversation_cache', {})
        if conversation_cache.get("has_relevant_context", False):
            # Use enhanced conversation intelligence if available
            cached_content = conversation_cache.get("cached_content", "")
            if cached_content:
                cache_context = f"""

CONVERSATION INTELLIGENCE CONTEXT:
{cached_content}

**IMPORTANT**: Use this conversation context to build upon previous work naturally.
For follow-up requests like "generate model answers", use the questions and content you previously created in this conversation.
"""
                base_prompt += cache_context
            else:
                # Fallback to manual extraction
                cache_context = f"""

CONVERSATION CONTEXT:
You have access to previous work from this conversation:
{conversation_cache.get('context_summary', '')}

Previous questions you generated:
"""
                for i, question in enumerate(conversation_cache.get('questions_generated', [])[:5], 1):
                    cache_context += f"{i}. {question}\n"
                
                cache_context += """
You can reference and build upon this previous work naturally in your response.
When generating model answers, use the questions you previously created in this conversation.
"""
                base_prompt += cache_context
        
        # ROOSEVELT'S CONVERSATION INTELLIGENCE: Trust the LLM's natural context understanding
        # No brittle destination extraction - let conversation flow naturally
        
        # Research findings context
        research_findings = shared_memory.get("research_findings", {})
        if research_findings:
            context_addition = f"""

PREVIOUS RESEARCH CONTEXT:
Your conversation partner has access to recent research findings from our research agent:
"""
            for key, research_data in list(research_findings.items())[-3:]:  # Last 3 research items
                if isinstance(research_data, dict):
                    # ROOSEVELT'S FIX: No data wrapper required - use direct fields
                    context_addition += f"- Research on '{key}': {research_data.get('findings', 'N/A')[:200]}...\n"
            
            context_addition += """
You can reference these findings naturally in conversation if relevant to the user's questions.
"""
            base_prompt += context_addition
        
        if pending_operations:
            reminder_context = f"""

PENDING OPERATIONS AWARENESS:
The user currently has {len(pending_operations)} pending operations that require their attention:
"""
            for op in pending_operations:
                reminder_context += f"- #{op.get('id', 'unknown')}: {op.get('summary', 'Unknown operation')}\n"
            
            reminder_context += """
REMINDER GUIDELINES:
- If the conversation naturally allows, you MAY include a gentle reminder about pending operations
- Don't force reminders into every response - be contextually appropriate
- Use natural language like "By the way, you still have that research request pending..."
- Only remind if it's been several conversation turns since the last reminder
- Be helpful, not pushy"""
            
            base_prompt += reminder_context
        
        return base_prompt
    
    def _add_smart_reminder(self, response: str, pending_operations: list, state: Dict[str, Any]) -> str:
        """Add smart reminder about pending operations if appropriate"""
        if not pending_operations:
            return response
        
        # Check if we should remind based on conversation flow
        if not self._should_add_reminder(state, pending_operations):
            return response
        
        # Generate appropriate reminder
        if len(pending_operations) == 1:
            op = pending_operations[0]
            reminder = f"\n\n*By the way, you still have a pending {op.get('type', 'operation')}: '{op.get('summary', 'unknown')}'. Say 'yes' to proceed or 'cancel' to forget it.*"
        elif len(pending_operations) <= 3:
            summaries = [f"#{op.get('id', '?')}: {op.get('summary', 'unknown')}" for op in pending_operations]
            reminder = f"\n\n*You have {len(pending_operations)} pending operations:\n" + "\n".join(summaries) + "\nSay 'yes #<number>' to proceed with a specific one, or 'cancel all' to clear them.*"
        else:
            reminder = f"\n\n*You have {len(pending_operations)} pending operations. Say 'list pending' to see them all.*"
        
        return response + reminder
    
    def _parse_structured_chat_response(self, content: str) -> ChatResponse:
        """Parse structured chat response with robust error handling"""
        try:
            import json
            import re
            
            # Clean JSON from markdown blocks
            json_text = content.strip()
            if '```json' in json_text:
                match = re.search(r'```json\s*\n(.*?)\n```', json_text, re.DOTALL)
                if match:
                    json_text = match.group(1).strip()
            elif '```' in json_text:
                match = re.search(r'```\s*\n(.*?)\n```', json_text, re.DOTALL)
                if match:
                    json_text = match.group(1).strip()
            
            # Find JSON object if mixed with other text
            if not json_text.startswith('{'):
                match = re.search(r'\{.*\}', json_text, re.DOTALL)
                if match:
                    json_text = match.group(0)
            
            # ROOSEVELT'S JSON CLEANING: Remove control characters that break JSON parsing
            json_text = self._clean_control_characters(json_text)
            
            # Parse and validate
            data = json.loads(json_text)
            return ChatResponse(**data)
            
        except Exception as e:
            logger.warning(f"⚠️ Chat response parsing failed: {e}")
            # ROOSEVELT'S SMART FALLBACK: For simple inputs, create appropriate brief responses
            original_content = content.strip()
            
            # Extract the last user message to check for simple acknowledgments
            user_query = getattr(self, '_last_user_query', '').lower().strip()
            
            # If it's a simple acknowledgment, create a brief response
            if any(phrase in user_query for phrase in [
                "thanks", "thank you", "thx", "ty", "appreciate",
                "cool", "nice", "good", "great"
            ]) and len(user_query) < 50:
                brief_message = "You're welcome! Let me know if you need anything else."
                return ChatResponse(message=brief_message)
            
            # For other parsing failures, truncate verbose responses
            if len(original_content) > 300:
                # Try to find the first complete sentence
                sentences = original_content.split('. ')
                if len(sentences) > 1:
                    brief_message = sentences[0] + '.'
                    return ChatResponse(message=brief_message)
            
            # Fallback: treat entire content as message (but warn about length)
            if len(original_content) > 500:
                logger.warning(f"⚠️ Using verbose fallback response ({len(original_content)} chars) - consider improving prompt")
            
            return ChatResponse(message=original_content)
    
    def _should_add_reminder(self, state: Dict[str, Any], pending_operations: list) -> bool:
        """Determine if we should add a reminder to this response"""
        # Simple heuristic: remind if we have pending operations and it's been a few turns
        if not pending_operations:
            return False
        
        # Always remind if we have high-priority operations waiting
        for op in pending_operations:
            if op.get("permission_required", False):
                # Count messages since this operation was created
                # For now, use a simple approach - remind periodically
                return True
        
        return False
    
    async def _apply_universal_formatting(self, user_query: str, chat_response: str) -> str:
        """ROOSEVELT'S UNIVERSAL FORMATTING: Apply intelligent formatting to chat responses"""
        try:
            from services.universal_formatting_service import get_universal_formatting_service
            
            formatting_service = get_universal_formatting_service()
            
            # Detect if chat response would benefit from formatting
            formatting_analysis = await formatting_service.detect_formatting_need(
                agent_type="chat_agent",
                user_query=user_query,
                agent_response=chat_response,
                confidence_threshold=0.9  # Very high threshold for chat - keep conversational
            )
            
            # Apply formatting if confidence is very high for chat (raised threshold for conversational appropriateness)
            if (formatting_analysis and 
                formatting_analysis.get("confidence", 0.0) > 0.95):  # Raised to 0.95 for chat - be very conservative
                
                logger.info(f"📊 CHAT FORMATTING: Applying {formatting_analysis['formatting_type']} formatting")
                return await formatting_service.apply_formatting(chat_response, formatting_analysis, None)
            
            return chat_response
            
        except Exception as e:
            logger.error(f"❌ Universal formatting failed: {e}")
            return chat_response  # Return original on error

