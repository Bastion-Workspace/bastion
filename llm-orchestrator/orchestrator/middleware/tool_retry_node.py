"""
Tool Retry Middleware Node for LangGraph
Adapts LangChain's ToolRetryMiddleware concept as a LangGraph node wrapper

This provides retry logic with exponential backoff for tool calls.
Can be used to wrap tool execution within agent nodes.
"""

import logging
import asyncio
from typing import Dict, Any, Callable, Optional
from functools import wraps

logger = logging.getLogger(__name__)


class ToolRetryNode:
    """
    Wrapper for tool calls with automatic retry and exponential backoff
    
    This adapts LangChain's ToolRetryMiddleware concept to work within
    LangGraph nodes. Use it to wrap tool calls in agent nodes.
    
    Usage:
        # In agent node:
        retry_wrapper = ToolRetryNode(max_retries=3)
        
        async def _execute_tool_node(self, state):
            tool_result = await retry_wrapper.execute_with_retry(
                lambda: self._call_tool(state["tool_name"], state["tool_args"])
            )
            return {"tool_result": tool_result}
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 10.0,
        retryable_exceptions: Optional[tuple] = None
    ):
        """
        Initialize tool retry wrapper
        
        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds before first retry
            max_delay: Maximum delay in seconds (caps exponential backoff)
            retryable_exceptions: Tuple of exception types to retry (default: all)
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.retryable_exceptions = retryable_exceptions or (Exception,)
    
    def _is_retryable(self, exception: Exception) -> bool:
        """Check if exception is retryable"""
        return isinstance(exception, self.retryable_exceptions)
    
    async def execute_with_retry(
        self,
        tool_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute tool function with retry logic
        
        Args:
            tool_func: Async function to execute
            *args: Positional arguments for tool function
            **kwargs: Keyword arguments for tool function
            
        Returns:
            Result from tool function
            
        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                # Execute tool
                if asyncio.iscoroutinefunction(tool_func):
                    result = await tool_func(*args, **kwargs)
                else:
                    result = tool_func(*args, **kwargs)
                
                # Success - log if retried
                if attempt > 0:
                    logger.info(f"Tool succeeded on retry attempt {attempt + 1}")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if exception is retryable
                if not self._is_retryable(e):
                    logger.warning(f"Non-retryable exception: {e}")
                    raise
                
                # Last attempt - raise exception
                if attempt == self.max_retries - 1:
                    logger.error(f"Tool failed after {self.max_retries} attempts: {e}")
                    raise
                
                # Calculate exponential backoff delay
                delay = min(self.initial_delay * (2 ** attempt), self.max_delay)
                
                logger.warning(
                    f"Tool failed (attempt {attempt + 1}/{self.max_retries}), "
                    f"retrying in {delay:.1f}s: {e}"
                )
                
                await asyncio.sleep(delay)
        
        # Should never reach here, but just in case
        if last_exception:
            raise last_exception
    
    def wrap_tool(self, tool_func: Callable) -> Callable:
        """
        Decorator-style wrapper for tool functions
        
        Usage:
            @retry_wrapper.wrap_tool
            async def my_tool(arg1, arg2):
                # tool logic
                return result
        """
        @wraps(tool_func)
        async def wrapped(*args, **kwargs):
            return await self.execute_with_retry(tool_func, *args, **kwargs)
        
        return wrapped


def create_retry_wrapper(max_retries: int = 3) -> ToolRetryNode:
    """Convenience function to create retry wrapper"""
    return ToolRetryNode(max_retries=max_retries)

