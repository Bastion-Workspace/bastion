"""
Dictionary Agent - Specialized agent for word definitions, synonyms, antonyms, and etymology
Activated by "define:" prefix for instant routing without intent classification
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from orchestrator.agents.base_agent import BaseAgent, TaskStatus

logger = logging.getLogger(__name__)


class DictionaryState(TypedDict):
    """State for Dictionary Agent LangGraph workflow"""
    query: str
    word: str  # The word to define (extracted from query)
    user_id: str
    metadata: Dict[str, Any]
    messages: List[Any]
    shared_memory: Dict[str, Any]
    
    # Dictionary data
    definition_data: Dict[str, Any]
    
    # Response
    response: Dict[str, Any]
    task_status: str
    error: str


class DictionaryAgent(BaseAgent):
    """
    Dictionary Agent for word definitions, thesaurus, and etymology
    
    Handles queries starting with "define:" for instant routing.
    Provides comprehensive lexicographic information with conversation context.
    """
    
    def __init__(self):
        super().__init__("dictionary_agent")
        logger.info("📖 Dictionary Agent ready for lexicographic assistance!")
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """Build LangGraph workflow for dictionary agent"""
        workflow = StateGraph(DictionaryState)
        
        # Add nodes
        workflow.add_node("extract_word", self._extract_word_node)
        workflow.add_node("lookup_definition", self._lookup_definition_node)
        workflow.add_node("generate_response", self._generate_response_node)
        
        # Entry point
        workflow.set_entry_point("extract_word")
        
        # Linear flow: extract -> lookup -> generate -> END
        workflow.add_edge("extract_word", "lookup_definition")
        workflow.add_edge("lookup_definition", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow.compile(checkpointer=checkpointer)
    
    async def _extract_word_node(self, state: DictionaryState) -> Dict[str, Any]:
        """Extract the word to define from the query"""
        try:
            query = state.get("query", "")
            
            # Remove "define:" prefix if present
            word = query.lower().replace("define:", "").strip()
            
            # Handle multi-word queries - take the main term
            # Examples: "define: run fast" -> "run", "define: quantum mechanics" -> keep both
            if not word:
                return {
                    "word": "",
                    "error": "No word provided to define",
                    "task_status": "error"
                }
            
            logger.info(f"📖 Extracting word to define: '{word}'")
            
            return {
                "word": word,
                "task_status": "processing"
            }
            
        except Exception as e:
            logger.error(f"Error extracting word: {e}")
            return {
                "error": str(e),
                "task_status": "error"
            }
    
    async def _lookup_definition_node(self, state: DictionaryState) -> Dict[str, Any]:
        """Look up word definition using Free Dictionary API"""
        try:
            word = state.get("word", "")
            if not word:
                return {
                    "definition_data": {},
                    "error": "No word to look up",
                    "task_status": "error"
                }
            
            import aiohttp
            
            # Use Free Dictionary API (no auth required)
            api_url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
            
            logger.info(f"📖 Looking up definition for: {word}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✅ Found definition data for '{word}'")
                        return {
                            "definition_data": {
                                "success": True,
                                "api_response": data,
                                "word": word
                            },
                            "task_status": "processing"
                        }
                    elif response.status == 404:
                        logger.warning(f"⚠️ Word '{word}' not found in dictionary")
                        return {
                            "definition_data": {
                                "success": False,
                                "error": "word_not_found",
                                "word": word
                            },
                            "task_status": "processing"
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Dictionary API error {response.status}: {error_text}")
                        return {
                            "definition_data": {
                                "success": False,
                                "error": "api_error",
                                "word": word
                            },
                            "task_status": "processing"
                        }
                        
        except Exception as e:
            logger.error(f"Error looking up definition: {e}")
            return {
                "definition_data": {
                    "success": False,
                    "error": "lookup_failed",
                    "word": word,
                    "exception": str(e)
                },
                "task_status": "processing"
            }
    
    async def _generate_response_node(self, state: DictionaryState) -> Dict[str, Any]:
        """Generate formatted response with definition data"""
        try:
            word = state.get("word", "")
            definition_data = state.get("definition_data", {})
            metadata = state.get("metadata", {})
            messages = state.get("messages", [])
            
            # Get persona for style
            persona = metadata.get("persona", {})
            persona_style = persona.get("persona_style", "professional") if persona else "professional"
            style_instruction = self._get_style_instruction(persona_style)
            
            if not definition_data.get("success"):
                # Handle lookup failures
                error_type = definition_data.get("error", "unknown")
                
                if error_type == "word_not_found":
                    response_text = f"I couldn't find a definition for '{word}' in the dictionary. Could you check the spelling, or would you like to ask about a different word?"
                else:
                    response_text = f"I encountered an issue looking up '{word}'. The dictionary service may be temporarily unavailable. Please try again in a moment."
                
                response = {
                    "message": response_text,
                    "task_status": "complete",
                    "word": word,
                    "found": False
                }
                
                # Add assistant message to conversation history
                updated_state = self._add_assistant_response_to_messages(state, response_text)
                
                return {
                    "response": response,
                    "task_status": "complete",
                    "messages": updated_state.get("messages", [])
                }
            
            # Parse API response
            api_response = definition_data.get("api_response", [])
            if not api_response or not isinstance(api_response, list):
                response_text = f"I received an unexpected response format for '{word}'. Please try again."
                response = {
                    "message": response_text,
                    "task_status": "complete",
                    "word": word,
                    "found": False
                }
                
                updated_state = self._add_assistant_response_to_messages(state, response_text)
                
                return {
                    "response": response,
                    "task_status": "complete",
                    "messages": updated_state.get("messages", [])
                }
            
            # Extract first entry (most common)
            entry = api_response[0]
            
            # Get conversation history for context-aware responses
            conversation_history = self._extract_conversation_history(messages, limit=5)
            
            # Build system prompt for LLM to format the response
            system_prompt = f"""You are a helpful dictionary assistant. Your role is to present word definitions in a clear, engaging way.

{style_instruction}

AVAILABLE DICTIONARY DATA:
Word: {entry.get('word', word)}
Phonetic: {entry.get('phonetic', 'N/A')}
Phonetics: {entry.get('phonetics', [])}
Meanings: {entry.get('meanings', [])}
Origin: {entry.get('origin', 'N/A')}
Source URLs: {entry.get('sourceUrls', [])}

YOUR TASK:
1. Present the definition(s) in a clear, organized format
2. Include pronunciation if available
3. Show multiple meanings/parts of speech
4. Include etymology/origin if available
5. Provide synonyms/antonyms if available
6. Be conversational and match the requested style
7. If the user has asked follow-up questions in the conversation, address them

RESPONSE FORMAT:
Present the information in a natural, readable way. Use markdown formatting for clarity:
- Bold for the word and parts of speech
- Bullet points for multiple definitions
- Italics for examples
- Keep it conversational but informative

IMPORTANT: If the user is asking follow-up questions about the word (like "What's the origin?" or "Give me synonyms"), focus your response on those specific aspects while still providing the core definition."""

            # Build messages for LLM
            llm_messages = self._build_messages(
                system_prompt=system_prompt,
                user_query=state.get("query", f"define: {word}"),
                conversation_history=conversation_history
            )
            
            # Get LLM response
            llm = self._get_llm(temperature=0.7, state=state)
            llm_response = await llm.ainvoke(llm_messages)
            
            response_text = llm_response.content
            
            # Build structured response
            response = {
                "message": response_text,
                "task_status": "complete",
                "word": word,
                "found": True,
                "raw_data": entry  # Include raw data for potential follow-up queries
            }
            
            # Add assistant message to conversation history
            updated_state = self._add_assistant_response_to_messages(state, response_text)
            
            logger.info(f"✅ Generated response for '{word}'")
            
            return {
                "response": response,
                "task_status": "complete",
                "messages": updated_state.get("messages", [])
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            error_response = {
                "message": f"I encountered an error generating the response: {str(e)}",
                "task_status": "error",
                "word": word,
                "found": False
            }
            
            return {
                "response": error_response,
                "task_status": "error",
                "error": str(e)
            }
    
    async def process(
        self,
        query: str,
        metadata: Dict[str, Any] = None,
        messages: List[Any] = None
    ) -> Dict[str, Any]:
        """
        Process dictionary lookup request using LangGraph workflow
        
        Args:
            query: User query (typically "define: word")
            metadata: Optional metadata dictionary (persona, editor context, etc.)
            messages: Optional conversation history
            
        Returns:
            Dict with structured response and task status
        """
        try:
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
            initial_state: DictionaryState = {
                "query": query,
                "word": "",
                "user_id": user_id,
                "metadata": metadata,
                "messages": conversation_messages,
                "shared_memory": shared_memory,
                "definition_data": {},
                "response": {},
                "task_status": "",
                "error": ""
            }
            
            # Run LangGraph workflow with checkpointing
            result_state = await workflow.ainvoke(initial_state, config=config)
            
            # Extract final response
            response = result_state.get("response", {})
            task_status = result_state.get("task_status", "complete")
            
            if task_status == "error":
                error_msg = result_state.get("error", "Unknown error")
                logger.error(f"❌ Dictionary Agent failed: {error_msg}")
                return self._create_error_response(error_msg)
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Dictionary Agent failed: {e}")
            return self._create_error_response(str(e))





