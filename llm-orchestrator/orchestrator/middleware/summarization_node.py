"""
Summarization Middleware Node for LangGraph
Adapts LangChain's SummarizationMiddleware concept as a LangGraph node

This node can be inserted into any StateGraph workflow to automatically
summarize conversation history when token limits are approached.
"""

import logging
from typing import Dict, Any, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from config.settings import settings

logger = logging.getLogger(__name__)


class SummarizationNode:
    """
    LangGraph node that summarizes conversation history when token limits approach
    
    This adapts LangChain's SummarizationMiddleware concept to work within
    LangGraph StateGraph workflows. Can be added as a node before LLM calls.
    
    Usage:
        # In agent's _build_workflow():
        summarization_node = SummarizationNode(
            trigger_tokens=4000,
            keep_messages=20
        )
        workflow.add_node("summarize_history", summarization_node)
        workflow.add_edge("prepare_context", "summarize_history")
        workflow.add_edge("summarize_history", "generate_response")
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        trigger_tokens: int = 4000,
        keep_messages: int = 20,
        summary_prefix: str = "[Previous conversation summarized]"
    ):
        """
        Initialize summarization node
        
        Args:
            model: Model to use for summarization (defaults to FAST_MODEL)
            trigger_tokens: Token count threshold to trigger summarization
            keep_messages: Number of recent messages to keep after summarization
            summary_prefix: Prefix for summary message
        """
        self.trigger_tokens = trigger_tokens
        self.keep_messages = keep_messages
        self.summary_prefix = summary_prefix
        
        # Use fast model for summarization to save costs
        summary_model = model or settings.FAST_MODEL
        self.summary_llm = ChatOpenAI(
            model=summary_model,
            openai_api_key=settings.OPENROUTER_API_KEY,
            openai_api_base=settings.OPENROUTER_BASE_URL,
            temperature=0.3  # Lower temperature for more consistent summaries
        )
    
    def _count_tokens(self, messages: List[BaseMessage]) -> int:
        """
        Estimate token count from messages
        Uses character-based estimation (rough: 4 chars per token)
        """
        total_chars = sum(len(str(msg.content)) for msg in messages if hasattr(msg, 'content'))
        return total_chars // 4
    
    def _should_summarize(self, messages: List[BaseMessage]) -> bool:
        """Check if summarization is needed based on token count"""
        if not messages or len(messages) <= self.keep_messages:
            return False
        
        token_count = self._count_tokens(messages)
        return token_count >= self.trigger_tokens
    
    async def _generate_summary(self, messages_to_summarize: List[BaseMessage]) -> str:
        """Generate summary of older messages using LLM"""
        try:
            # Build summary prompt
            conversation_text = "\n\n".join([
                f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
                for msg in messages_to_summarize
                if hasattr(msg, 'content')
            ])
            
            summary_prompt = f"""Summarize the following conversation history, preserving key information, decisions, and context that would be useful for continuing the conversation:

{conversation_text}

Provide a concise summary that captures the essential information:"""
            
            response = await self.summary_llm.ainvoke([
                HumanMessage(content=summary_prompt)
            ])
            
            summary = response.content if hasattr(response, 'content') else str(response)
            logger.info(f"Generated summary ({len(summary)} chars) from {len(messages_to_summarize)} messages")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            # Fallback: return a simple note
            return f"Previous conversation with {len(messages_to_summarize)} messages"
    
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        LangGraph node function - summarizes messages if needed
        
        This is called by LangGraph when the node executes.
        Checks token count and summarizes if threshold exceeded.
        
        Args:
            state: Current LangGraph state
            
        Returns:
            Updated state with summarized messages if needed
        """
        try:
            # Extract messages from state
            # Different agents store messages in different keys
            messages = state.get("messages") or state.get("llm_messages") or []
            
            if not messages:
                return state
            
            # Check if summarization needed
            if not self._should_summarize(messages):
                logger.debug(f"No summarization needed ({self._count_tokens(messages)} tokens)")
                return state
            
            logger.info(f"Summarizing conversation history ({self._count_tokens(messages)} tokens)")
            
            # Split messages: keep recent, summarize older
            messages_to_keep = messages[-self.keep_messages:]
            messages_to_summarize = messages[:-self.keep_messages]
            
            # Generate summary
            summary_text = await self._generate_summary(messages_to_summarize)
            
            # Create summary message
            summary_message = AIMessage(
                content=f"{self.summary_prefix}: {summary_text}"
            )
            
            # Reconstruct message list with summary + recent messages
            summarized_messages = [summary_message] + messages_to_keep
            
            # Update state with summarized messages
            updated_state = state.copy()
            if "messages" in state:
                updated_state["messages"] = summarized_messages
            if "llm_messages" in state:
                updated_state["llm_messages"] = summarized_messages
            
            logger.info(f"Summarization complete: {len(messages_to_summarize)} messages → 1 summary + {len(messages_to_keep)} recent")
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Summarization node error: {e}")
            # Return state unchanged on error
            return state

