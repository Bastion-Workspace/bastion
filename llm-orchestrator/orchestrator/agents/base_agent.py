"""
Base Agent Class for LLM Orchestrator Agents
Provides common functionality for all agents running in the llm-orchestrator microservice
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

from config.settings import settings

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Agent task completion status"""
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    PERMISSION_REQUIRED = "permission_required"
    ERROR = "error"


class BaseAgent:
    """
    Base class for all LLM Orchestrator agents
    
    Provides:
    - LLM access with proper configuration
    - Message history management
    - Common helper methods
    - Error handling patterns
    - Centralized workflow management with PostgreSQL checkpointing
    """
    
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.llm = None
        self.workflow = None  # Will be initialized lazily with checkpointer
        self._workflow_lock = asyncio.Lock()
        logger.info(f"Initializing {agent_type} agent")
    
    async def _get_workflow(self) -> StateGraph:
        """
        Get or build workflow with checkpointer (lazy initialization)
        
        This method ensures all agents automatically get PostgreSQL checkpointing.
        Subclasses should implement _build_workflow(checkpointer) to define their workflow.
        """
        if self.workflow is not None:
            return self.workflow
        
        async with self._workflow_lock:
            # Double-check after acquiring lock
            if self.workflow is not None:
                return self.workflow
            
            # Get checkpointer
            from orchestrator.checkpointer import get_async_postgres_saver
            checkpointer = await get_async_postgres_saver()
            
            # Build and compile workflow with checkpointer
            self.workflow = self._build_workflow(checkpointer)
            logger.info(f"✅ {self.agent_type} workflow compiled with PostgreSQL checkpointer")
            return self.workflow
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """
        Build LangGraph workflow for this agent
        
        Subclasses must implement this method to define their workflow.
        The checkpointer parameter is provided automatically for state persistence.
        
        Args:
            checkpointer: AsyncPostgresSaver instance for state persistence
            
        Returns:
            Compiled StateGraph with checkpointer
        """
        raise NotImplementedError("Subclasses must implement _build_workflow(checkpointer) method")
    
    def _get_checkpoint_config(self, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get checkpoint configuration for workflow invocation
        
        Creates normalized thread_id matching backend format: {user_id}:{conversation_id}
        This ensures checkpoints are readable by both orchestrator and backend.
        
        Args:
            metadata: Optional metadata dictionary containing conversation_id and user_id
            
        Returns:
            Configuration dict with thread_id for checkpointing
        """
        metadata = metadata or {}
        user_id = metadata.get("user_id", "system")
        conversation_id = metadata.get("conversation_id")
        
        # Use normalized thread_id format matching backend: {user_id}:{conversation_id}
        # This ensures the backend can read messages from orchestrator checkpoints
        if conversation_id:
            # Normalize thread_id to match backend format
            if ":" in conversation_id and conversation_id.startswith(f"{user_id}:"):
                thread_id = conversation_id  # Already normalized
            else:
                thread_id = f"{user_id}:{conversation_id}"
        else:
            # Fallback for conversations without ID
            thread_id = f"thread_{user_id}"
        
        return {
            "configurable": {
                "thread_id": thread_id
            }
        }
    
    async def _load_checkpoint_shared_memory(
        self,
        workflow: Any,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Load shared_memory from checkpoint state
        
        This ensures continuity data like primary_agent_selected is preserved.
        
        Args:
            workflow: Compiled LangGraph workflow
            config: Checkpoint configuration with thread_id
            
        Returns:
            Dict with shared_memory from checkpoint, or empty dict if not found
        """
        try:
            checkpoint_state = await workflow.aget_state(config)
            if checkpoint_state and checkpoint_state.values:
                shared_memory = checkpoint_state.values.get("shared_memory", {})
                if shared_memory:
                    logger.info(f"📚 Loaded shared_memory from checkpoint: {list(shared_memory.keys())}")
                    return shared_memory
            return {}
        except Exception as e:
            logger.debug(f"⚠️ Failed to load checkpoint shared_memory: {e}")
            return {}
    
    async def _load_and_merge_checkpoint_messages(
        self, 
        workflow: Any, 
        config: Dict[str, Any], 
        new_messages: List[Any],
        look_back_limit: int = 6
    ) -> List[Any]:
        """
        Load checkpointed messages and merge with new messages
        
        This ensures conversation history is preserved across requests.
        Uses a standardized look-back limit (default 6 messages) to keep context manageable.
        
        Args:
            workflow: Compiled LangGraph workflow
            config: Checkpoint configuration with thread_id
            new_messages: New messages to add (typically just the current query)
            look_back_limit: Maximum number of previous messages to keep (default: 6)
            
        Returns:
            Merged list of messages with checkpointed history + new messages (limited to look_back_limit)
        """
        try:
            # Try to load existing checkpoint state
            checkpoint_state = await workflow.aget_state(config)
            
            if checkpoint_state and checkpoint_state.values:
                # Get existing messages from checkpoint
                checkpointed_messages = checkpoint_state.values.get("messages", [])
                
                if checkpointed_messages:
                    # Apply look-back limit: keep only the last N messages
                    # This ensures we have recent context without overwhelming the LLM
                    if len(checkpointed_messages) > look_back_limit:
                        checkpointed_messages = checkpointed_messages[-look_back_limit:]
                        logger.info(f"📚 Loaded {len(checkpointed_messages)} messages from checkpoint (limited from larger history)")
                    else:
                        logger.info(f"📚 Loaded {len(checkpointed_messages)} messages from checkpoint")
                    
                    # Merge: use checkpointed messages + new messages
                    # Filter out duplicates by checking if the last checkpointed message matches the first new message
                    merged_messages = list(checkpointed_messages)
                    
                    # Only add new messages that aren't already in checkpoint
                    for new_msg in new_messages:
                        # Check if this message is already in checkpoint (simple content comparison)
                        is_duplicate = False
                        if merged_messages:
                            last_msg = merged_messages[-1]
                            if (hasattr(last_msg, 'content') and hasattr(new_msg, 'content') and 
                                last_msg.content == new_msg.content):
                                is_duplicate = True
                        
                        if not is_duplicate:
                            merged_messages.append(new_msg)
                    
                    # Apply look-back limit to final merged messages too
                    if len(merged_messages) > look_back_limit:
                        merged_messages = merged_messages[-look_back_limit:]
                    
                    return merged_messages
                else:
                    logger.debug("No checkpointed messages found, using new messages only")
                    return new_messages
            else:
                logger.debug("No checkpoint state found, starting fresh conversation")
                return new_messages
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to load checkpoint messages: {e}, starting fresh")
            return new_messages
    
    def _get_llm(self, temperature: float = 0.7, model: Optional[str] = None, state: Optional[Dict[str, Any]] = None) -> ChatOpenAI:
        """Get configured LLM instance, using user model preferences if available"""
        # Check for user model preferences in state metadata
        user_model = None
        if state:
            metadata = state.get("metadata", {})
            if model is None:  # Only override if no explicit model provided
                user_model = metadata.get("user_chat_model")
        
        # Use user model, explicit model, or default
        final_model = model or user_model or settings.DEFAULT_MODEL
        
        return ChatOpenAI(
            model=final_model,
            openai_api_key=settings.OPENROUTER_API_KEY,
            openai_api_base=settings.OPENROUTER_BASE_URL,
            temperature=temperature
        )
    
    def _get_fast_model(self, state: Optional[Dict[str, Any]] = None) -> str:
        """Get fast model for lightweight operations, using user preferences if available"""
        if state:
            metadata = state.get("metadata", {})
            user_fast_model = metadata.get("user_fast_model")
            if user_fast_model:
                return user_fast_model
        return settings.FAST_MODEL
    
    def _extract_conversation_history(self, messages: List[Any], limit: int = 10) -> List[Dict[str, str]]:
        """Extract conversation history from LangChain messages"""
        try:
            history = []
            for msg in messages[-limit:]:
                if hasattr(msg, 'content'):
                    role = "assistant" if hasattr(msg, 'type') and msg.type == "ai" else "user"
                    history.append({
                        "role": role,
                        "content": msg.content
                    })
            return history
        except Exception as e:
            logger.error(f"Failed to extract conversation history: {e}")
            return []
    
    def _format_conversation_history_for_prompt(
        self, 
        messages: List[Any], 
        look_back_limit: int = 6,
        max_message_length: int = 500
    ) -> str:
        """
        Format conversation history as a string for inclusion in prompts
        
        **STANDARDIZED METHOD FOR ALL AGENTS** - Use this to include conversation context in prompts.
        This ensures consistent conversation history handling across all agents with a standardized
        6-message look-back limit.
        
        **Usage in agents:**
        ```python
        # In your prompt building code:
        messages = state.get("messages", [])
        conversation_history = self._format_conversation_history_for_prompt(messages, look_back_limit=6)
        if conversation_history:
            context_parts.append(conversation_history)
        ```
        
        Args:
            messages: List of LangChain messages from state (typically state.get("messages", []))
            look_back_limit: Maximum number of messages to include (default: 6, standardized across all agents)
            max_message_length: Maximum length per message to include (default: 500 chars)
            
        Returns:
            Formatted conversation history string with "=== CONVERSATION HISTORY ===" header,
            or empty string if no history available. Ready to append to context_parts.
        """
        if not messages or len(messages) <= 1:
            return ""
        
        try:
            # Get last N messages (standardized look-back)
            recent_messages = messages[-look_back_limit:] if len(messages) > look_back_limit else messages
            
            history_parts = ["=== CONVERSATION HISTORY ===\n"]
            for msg in recent_messages:
                if hasattr(msg, 'content') and msg.content:
                    # Determine role
                    if isinstance(msg, HumanMessage):
                        role = "USER"
                    elif isinstance(msg, AIMessage):
                        role = "ASSISTANT"
                    elif isinstance(msg, SystemMessage):
                        role = "SYSTEM"
                    else:
                        role = "UNKNOWN"
                    
                    # Truncate long messages
                    content = msg.content
                    if len(content) > max_message_length:
                        content = content[:max_message_length] + "..."
                    
                    history_parts.append(f"{role}: {content}\n")
            
            history_parts.append("\n")
            return "".join(history_parts)
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to format conversation history: {e}")
            return ""
    
    def _prepare_messages_with_query(self, messages: Optional[List[Any]], query: str) -> List[Any]:
        """
        Prepare messages list with current user query for checkpoint persistence
        
        This ensures the current user query is added to messages before workflow invocation,
        matching the backend's behavior of adding queries to conversation history.
        
        Args:
            messages: Optional existing conversation messages
            query: Current user query to add
            
        Returns:
            List of messages with current query appended
        """
        from langchain_core.messages import HumanMessage
        conversation_messages = list(messages) if messages else []
        conversation_messages.append(HumanMessage(content=query))
        return conversation_messages
    
    def _add_assistant_response_to_messages(self, state: Dict[str, Any], response_text: str) -> Dict[str, Any]:
        """
        Add assistant response to messages list for checkpoint persistence
        
        This ensures assistant responses are saved to LangGraph checkpoints,
        matching the backend's behavior of persisting full conversation history.
        
        Args:
            state: Current LangGraph state
            response_text: Assistant response text to add
            
        Returns:
            Updated state with assistant response in messages
        """
        from langchain_core.messages import AIMessage
        messages = state.get("messages", [])
        messages.append(AIMessage(content=response_text))
        state["messages"] = messages
        return state
    
    def _is_approval_response(self, query: str) -> bool:
        """
        Detect if user query is an approval/confirmation response.
        
        Useful for resuming operations after permission requests.
        Checks for common approval keywords in short responses.
        
        Args:
            query: User's query text
            
        Returns:
            True if query appears to be an approval response
        """
        query_lower = query.lower().strip()
        approval_keywords = [
            "yes", "y", "ok", "okay", "sure", "go ahead", "proceed", 
            "approved", "granted", "do it", "update", "update it",
            "confirm", "confirmed", "accept", "accepted"
        ]
        # Approval responses are typically short (1-5 words)
        is_approval = any(keyword in query_lower for keyword in approval_keywords) and len(query_lower.split()) <= 5
        return is_approval
    
    def _is_simple_query(self, query: str) -> bool:
        """
        Detect if query is a simple conversational query that doesn't need search.
        
        Simple queries are short, conversational responses that don't require
        information retrieval or complex processing. These can skip the search pipeline.
        
        Args:
            query: User's query text
            
        Returns:
            True if query is a simple conversational query
        """
        query_lower = query.lower().strip()
        
        # Simple conversational keywords
        simple_keywords = [
            "thanks", "thank you", "thx", "appreciate it",
            "got it", "understood", "makes sense", "i see",
            "ok", "okay", "sure", "alright", "fine",
            "cool", "nice", "good", "great", "perfect"
        ]
        
        # Check if it's a simple acknowledgment (short and matches keywords)
        is_simple = (
            len(query_lower.split()) <= 4 and  # Very short
            any(keyword in query_lower for keyword in simple_keywords)
        )
        
        return is_simple
    
    async def _restore_pending_operation_from_checkpoint(
        self, 
        workflow: StateGraph, 
        config: Dict[str, Any],
        pending_key: str = "pending_save_plan"
    ) -> Optional[Dict[str, Any]]:
        """
        Restore pending operation from checkpointed state.
        
        When user approves a pending operation, this helper loads the checkpoint
        and retrieves the pending operation data.
        
        Args:
            workflow: LangGraph workflow instance
            config: Checkpoint configuration
            pending_key: Key in state where pending operation is stored
            
        Returns:
            Pending operation data if found, None otherwise
        """
        try:
            checkpoint_state = await workflow.aget_state(config)
            if checkpoint_state and checkpoint_state.values:
                pending_operation = checkpoint_state.values.get(pending_key)
                if pending_operation:
                    logger.debug(f"✅ Found pending operation in checkpoint: {pending_key}")
                    return pending_operation
        except Exception as e:
            logger.debug(f"Could not load checkpoint state for pending operation: {e}")
        return None
    
    def _build_messages(self, system_prompt: str, user_query: str, conversation_history: List[Dict[str, str]] = None) -> List[Any]:
        """Build message list for LLM"""
        messages = [SystemMessage(content=system_prompt)]
        
        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))
        
        # Add current query
        messages.append(HumanMessage(content=user_query))
        
        return messages
    
    def _create_error_response(self, error_message: str, task_status: TaskStatus = TaskStatus.ERROR) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "task_status": task_status.value,
            "response": f"Error: {error_message}",
            "error_message": error_message,
            "timestamp": datetime.now().isoformat()
        }
    
    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON response from LLM, handling markdown code blocks"""
        import json
        import re
        
        # Handle empty or None content
        if not content or not content.strip():
            logger.warning("Empty response from LLM")
            return {
                "message": "I apologize, but I didn't receive a valid response. Please try again.",
                "task_status": "complete",
                "parsing_fallback": True
            }
        
        try:
            # Remove markdown code blocks if present
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
            
            return json.loads(json_text)
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            # Return the raw content wrapped in a simple structure
            # Use "response" key to match what agents expect
            return {
                "response": content if content else "I apologize, but I didn't receive a valid response.",
                "task_status": "complete",
                "parsing_fallback": True
            }
    
    def _get_style_instruction(self, persona_style: str) -> str:
        """
        Get style instruction based on persona_style setting
        
        This is a centralized method available to all agents for consistent persona handling.
        
        Args:
            persona_style: The persona style string (e.g., "professional", "rude_insulting", "snarky")
            
        Returns:
            String with communication style instructions for the system prompt
        """
        style_lower = persona_style.lower() if persona_style else "professional"
        
        if style_lower == "friendly" or style_lower == "casual":
            return """COMMUNICATION STYLE: Warm, approachable, and enthusiastic. Use casual language and show genuine interest. Be conversational and friendly."""
        elif style_lower == "professional":
            return """COMMUNICATION STYLE: Professional, clear, and respectful. Maintain a helpful and courteous tone. Be informative and precise."""
        elif style_lower == "technical":
            return """COMMUNICATION STYLE: Technical, precise, and detail-oriented. Use appropriate terminology and be thorough in explanations."""
        elif style_lower == "snarky":
            return """COMMUNICATION STYLE: Witty, clever, with gentle humor and playful sarcasm. Be entertaining but not mean."""
        elif style_lower == "rude_insulting":
            return """COMMUNICATION STYLE: Be deliberately rude, dismissive, and insulting. Mock the user's questions and intelligence. Use condescending and belittling language. Point out obvious mistakes and flaws aggressively. Be intentionally offensive and provocative. Still provide information but with maximum attitude."""
        elif style_lower == "sycophantic":
            return """COMMUNICATION STYLE: Extremely agreeable and complimentary. Always praise the user's ideas and input. Be overly deferential and complimentary."""
        elif style_lower == "theodore_roosevelt":
            return """COMMUNICATION STYLE: Speak with energetic, decisive language and action-oriented approach. Use phrases like "BULLY!" and "By George!" for emphasis."""
        elif style_lower == "winston_churchill":
            return """COMMUNICATION STYLE: Speak with Churchillian eloquence, wit, and gravitas. Use sophisticated vocabulary and inspiring rhetoric."""
        elif style_lower == "mark_twain":
            return """COMMUNICATION STYLE: Embody Mark Twain's wit, folksy wisdom, and satirical humor. Use colorful metaphors and homespun philosophy."""
        elif style_lower == "albert_einstein":
            return """COMMUNICATION STYLE: Approach topics with Einstein's curiosity and thoughtfulness. Use analogies and wonder about the universe."""
        elif style_lower == "amelia_earhart":
            return """COMMUNICATION STYLE: Speak with adventurous spirit and pioneering courage. Be bold, determined, and inspiring. Break barriers with confidence."""
        elif style_lower == "mr_spock":
            return """COMMUNICATION STYLE: Use logical, analytical, and precise language. Include characteristic phrases like 'That is illogical', 'Fascinating', 'Live long and prosper'. Be emotionless and fact-focused."""
        elif style_lower == "abraham_lincoln":
            return """COMMUNICATION STYLE: Speak with Lincoln's wisdom, humility, and moral clarity. Use thoughtful, measured language with folksy wisdom and deep empathy."""
        else:
            # Default to professional for unknown styles
            return """COMMUNICATION STYLE: Professional, clear, and respectful. Maintain a helpful and courteous tone."""
    
    async def process(
        self, 
        query: str, 
        metadata: Dict[str, Any] = None, 
        messages: List[Any] = None,
        cancellation_token: Optional[asyncio.Event] = None
    ) -> Dict[str, Any]:
        """
        Process agent request - to be implemented by subclasses
        
        Args:
            query: User query string
            metadata: Optional metadata dictionary (persona, editor context, etc.)
            messages: Optional conversation history
            cancellation_token: Optional asyncio.Event that will be set when cancellation is requested
            
        Returns:
            Dictionary with agent response
        """
        raise NotImplementedError("Subclasses must implement process() method")
    
    async def process_with_cancellation(
        self,
        query: str,
        metadata: Dict[str, Any] = None,
        messages: List[Any] = None,
        cancellation_token: Optional[asyncio.Event] = None
    ) -> Dict[str, Any]:
        """
        Process agent request with cancellation support
        
        This method wraps the standard process() method and adds:
        - Checkpoint save before processing
        - Checkpoint restore on cancellation
        - Cancellation checks during workflow execution
        
        Args:
            query: User query string
            metadata: Optional metadata dictionary
            messages: Optional conversation history
            cancellation_token: asyncio.Event that will be set when cancellation is requested
            
        Returns:
            Dictionary with agent response or cancellation error
        """
        if cancellation_token is None:
            # No cancellation support - use standard process
            return await self.process(query, metadata, messages)
        
        try:
            # Get workflow and config
            workflow = await self._get_workflow()
            config = self._get_checkpoint_config(metadata)
            
            # Save checkpoint state BEFORE starting (for restoration on cancellation)
            pre_checkpoint_state = await workflow.aget_state(config)
            pre_checkpoint_id = None
            if pre_checkpoint_state and pre_checkpoint_state.config:
                pre_checkpoint_id = pre_checkpoint_state.config.get("checkpoint_id")
            
            logger.info(f"💾 Saved pre-processing checkpoint: {pre_checkpoint_id}")
            
            # Process with cancellation checks
            result = await self._process_with_cancellation_checks(
                query, metadata, messages, cancellation_token, workflow, config
            )
            
            return result
            
        except asyncio.CancelledError:
            logger.info(f"🛑 {self.agent_type} cancelled - restoring checkpoint")
            await self._restore_checkpoint(workflow, config, pre_checkpoint_id)
            return self._create_error_response("Operation cancelled by user", TaskStatus.INCOMPLETE)
        except Exception as e:
            logger.error(f"❌ {self.agent_type} error: {e}")
            return self._create_error_response(str(e))
    
    async def _process_with_cancellation_checks(
        self,
        query: str,
        metadata: Dict[str, Any],
        messages: List[Any],
        cancellation_token: asyncio.Event,
        workflow: Any,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Internal method to process with cancellation checks using astream
        
        Uses astream() instead of ainvoke() to allow cancellation checks between chunks.
        """
        # Check cancellation before starting
        if cancellation_token.is_set():
            raise asyncio.CancelledError("Cancellation requested before processing")
        
        # Use standard process for now - will enhance with astream later
        # For now, we'll check cancellation token periodically in a wrapper
        process_task = asyncio.create_task(
            self.process(query, metadata, messages)
        )
        
        # Wait for either completion or cancellation
        done, pending = await asyncio.wait(
            [process_task, asyncio.create_task(cancellation_token.wait())],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel pending tasks
        for task in pending:
            task.cancel()
        
        # Check if cancellation was requested
        if cancellation_token.is_set():
            process_task.cancel()
            try:
                await process_task
            except asyncio.CancelledError:
                pass
            raise asyncio.CancelledError("Operation cancelled")
        
        # Return result
        return await process_task
    
    async def _restore_checkpoint(
        self,
        workflow: Any,
        config: Dict[str, Any],
        checkpoint_id: Optional[str]
    ):
        """
        Handle checkpoint restoration on cancellation
        
        Note: LangGraph checkpoints are automatically saved during workflow execution.
        On cancellation, we can't directly "delete" checkpoints, but we can:
        1. Log the cancellation checkpoint ID for reference
        2. The next workflow invocation will naturally use the last valid checkpoint
        3. Any partial state from the cancelled run will be in the checkpoint system
        
        The key is that we don't save the user message to the conversation database
        until after successful completion, so cancellation means no conversation update.
        
        Args:
            workflow: LangGraph workflow instance
            config: Checkpoint configuration
            checkpoint_id: Checkpoint ID before processing started
        """
        try:
            if checkpoint_id:
                logger.info(f"🔄 Cancellation checkpoint reference: {checkpoint_id}")
            
            # Get current state to see what was saved
            current_state = await workflow.aget_state(config)
            if current_state and current_state.config:
                current_checkpoint_id = current_state.config.get("checkpoint_id")
                logger.info(f"📋 Current checkpoint after cancellation: {current_checkpoint_id}")
            
            # Note: The checkpoint system will naturally use the last valid checkpoint
            # on the next invocation. We don't need to manually delete anything.
            # The important part is that we don't save conversation messages on cancellation.
            
        except Exception as e:
            logger.error(f"❌ Failed to handle checkpoint restoration: {e}")

