"""
Call Limit Middleware Nodes for LangGraph
Adapts LangChain's ModelCallLimitMiddleware and ToolCallLimitMiddleware concepts

These nodes track and enforce limits on model and tool calls to prevent
runaway costs and infinite loops.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ModelCallLimitNode:
    """
    LangGraph node that enforces model call limits
    
    Tracks model call count in state and raises error if limit exceeded.
    Can be added before LLM invocation nodes.
    
    Usage:
        # In agent's _build_workflow():
        call_limit_node = ModelCallLimitNode(max_calls=50)
        workflow.add_node("check_call_limit", call_limit_node)
        workflow.add_edge("prepare_context", "check_call_limit")
        workflow.add_edge("check_call_limit", "generate_response")
    """
    
    def __init__(self, max_calls: int = 50, limit_key: str = "model_call_count"):
        """
        Initialize model call limit node
        
        Args:
            max_calls: Maximum number of model calls allowed
            limit_key: State key to store call count
        """
        self.max_calls = max_calls
        self.limit_key = limit_key
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check model call limit and increment counter
        
        Args:
            state: Current LangGraph state
            
        Returns:
            Updated state with incremented call count
            
        Raises:
            ValueError if call limit exceeded
        """
        # Get current call count
        call_count = state.get(self.limit_key, 0)
        
        # Check limit
        if call_count >= self.max_calls:
            error_msg = f"Model call limit ({self.max_calls}) exceeded"
            logger.error(error_msg)
            return {
                **state,
                "error": error_msg,
                "task_status": "error"
            }
        
        # Increment and update state
        updated_state = state.copy()
        updated_state[self.limit_key] = call_count + 1
        
        logger.debug(f"Model call count: {call_count + 1}/{self.max_calls}")
        
        return updated_state


class ToolCallLimitNode:
    """
    LangGraph node that enforces tool call iteration limits
    
    Tracks tool call iterations in state and raises error if limit exceeded.
    Can be added before tool execution nodes.
    
    Usage:
        # In agent's _build_workflow():
        tool_limit_node = ToolCallLimitNode(max_iterations=10)
        workflow.add_node("check_tool_limit", tool_limit_node)
        workflow.add_edge("prepare_context", "check_tool_limit")
        workflow.add_edge("check_tool_limit", "execute_tools")
    """
    
    def __init__(self, max_iterations: int = 10, limit_key: str = "tool_iteration_count"):
        """
        Initialize tool call limit node
        
        Args:
            max_iterations: Maximum number of tool call iterations allowed
            limit_key: State key to store iteration count
        """
        self.max_iterations = max_iterations
        self.limit_key = limit_key
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check tool call limit and increment counter
        
        Args:
            state: Current LangGraph state
            
        Returns:
            Updated state with incremented iteration count
            
        Raises:
            ValueError if iteration limit exceeded
        """
        # Get current iteration count
        iteration_count = state.get(self.limit_key, 0)
        
        # Check limit
        if iteration_count >= self.max_iterations:
            error_msg = f"Tool call iteration limit ({self.max_iterations}) exceeded"
            logger.error(error_msg)
            return {
                **state,
                "error": error_msg,
                "task_status": "error"
            }
        
        # Increment and update state
        updated_state = state.copy()
        updated_state[self.limit_key] = iteration_count + 1
        
        logger.debug(f"Tool iteration count: {iteration_count + 1}/{self.max_iterations}")
        
        return updated_state

