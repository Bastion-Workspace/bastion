"""
Example: Integrating Middleware Nodes into LangGraph Agents

This file shows concrete examples of how to use middleware nodes
in existing agent workflows.
"""

from langgraph.graph import StateGraph, END
from typing import Dict, Any, TypedDict
from orchestrator.middleware import (
    SummarizationNode,
    ToolRetryNode,
    ModelCallLimitNode,
    ToolCallLimitNode
)


# ============================================================================
# Example 1: Chat Agent with Summarization
# ============================================================================

class ChatStateWithSummarization(TypedDict):
    query: str
    messages: list
    llm_messages: list
    response: Dict[str, Any]
    task_status: str


class ChatAgentWithSummarization:
    """Example: Chat agent with automatic summarization"""
    
    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(ChatStateWithSummarization)
        
        # Create summarization node
        summarization_node = SummarizationNode(
            trigger_tokens=4000,
            keep_messages=20
        )
        
        # Add nodes
        workflow.add_node("prepare_context", self._prepare_context_node)
        workflow.add_node("summarize_history", summarization_node)  # ← Middleware node
        workflow.add_node("generate_response", self._generate_response_node)
        
        # Flow: prepare → summarize (if needed) → generate
        workflow.set_entry_point("prepare_context")
        workflow.add_edge("prepare_context", "summarize_history")
        workflow.add_edge("summarize_history", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow.compile()
    
    async def _prepare_context_node(self, state: ChatStateWithSummarization) -> Dict[str, Any]:
        # Your existing context preparation logic
        return state
    
    async def _generate_response_node(self, state: ChatStateWithSummarization) -> Dict[str, Any]:
        # Your existing response generation logic
        return state


# ============================================================================
# Example 2: Research Agent with Call Limits and Tool Retry
# ============================================================================

class ResearchStateWithLimits(TypedDict):
    query: str
    messages: list
    model_call_count: int
    tool_iteration_count: int
    tool_results: list
    response: Dict[str, Any]
    task_status: str


class ResearchAgentWithLimits:
    """Example: Research agent with call limits and tool retry"""
    
    def __init__(self):
        # Create middleware nodes
        self.model_limit_node = ModelCallLimitNode(max_calls=50)
        self.tool_limit_node = ToolCallLimitNode(max_iterations=10)
        self.tool_retry = ToolRetryNode(max_retries=3)
    
    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(ResearchStateWithLimits)
        
        # Add nodes including middleware
        workflow.add_node("prepare_context", self._prepare_context_node)
        workflow.add_node("check_model_limit", self.model_limit_node)  # ← Middleware node
        workflow.add_node("execute_tools", self._execute_tools_node)
        workflow.add_node("check_tool_limit", self.tool_limit_node)  # ← Middleware node
        workflow.add_node("generate_response", self._generate_response_node)
        
        # Flow with limits
        workflow.set_entry_point("prepare_context")
        workflow.add_edge("prepare_context", "check_model_limit")
        workflow.add_edge("check_model_limit", "execute_tools")
        workflow.add_edge("execute_tools", "check_tool_limit")
        workflow.add_edge("check_tool_limit", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow.compile()
    
    async def _prepare_context_node(self, state: ResearchStateWithLimits) -> Dict[str, Any]:
        # Initialize counters if not present
        if "model_call_count" not in state:
            state["model_call_count"] = 0
        if "tool_iteration_count" not in state:
            state["tool_iteration_count"] = 0
        return state
    
    async def _execute_tools_node(self, state: ResearchStateWithLimits) -> Dict[str, Any]:
        """Execute tools with retry wrapper"""
        tool_name = state.get("tool_name")
        tool_args = state.get("tool_args", {})
        
        # Use retry wrapper for tool execution
        try:
            tool_result = await self.tool_retry.execute_with_retry(
                lambda: self._call_tool(tool_name, tool_args)
            )
            
            # Update state with result
            tool_results = state.get("tool_results", [])
            tool_results.append(tool_result)
            
            return {
                **state,
                "tool_results": tool_results
            }
        except Exception as e:
            return {
                **state,
                "error": str(e),
                "task_status": "error"
            }
    
    async def _call_tool(self, tool_name: str, tool_args: Dict[str, Any]):
        """Actual tool call - wrapped by retry logic"""
        # Your tool execution logic here
        pass
    
    async def _generate_response_node(self, state: ResearchStateWithLimits) -> Dict[str, Any]:
        # Your response generation logic
        return state


# ============================================================================
# Example 3: Using Retry as Decorator
# ============================================================================

class AgentWithRetryDecorator:
    """Example: Using retry wrapper as decorator"""
    
    def __init__(self):
        self.tool_retry = ToolRetryNode(max_retries=3)
    
    @property
    def search_tool(self):
        """Tool wrapped with retry logic"""
        @self.tool_retry.wrap_tool
        async def _search_tool(query: str):
            # Your search tool logic
            return {"results": []}
        
        return _search_tool
    
    async def process(self, query: str):
        """Use retry-wrapped tool"""
        result = await self.search_tool(query)
        return result


# ============================================================================
# Example 4: Conditional Summarization (Only for Long Conversations)
# ============================================================================

class ConditionalSummarizationNode:
    """
    Example: Only summarize if conversation is actually long
    Can be used as a conditional edge instead of always running
    """
    
    def __init__(self, summarization_node: SummarizationNode):
        self.summarization_node = summarization_node
    
    def should_summarize(self, state: Dict[str, Any]) -> bool:
        """Check if summarization is needed"""
        messages = state.get("messages", [])
        if not messages:
            return False
        
        # Use summarization node's internal check
        return self.summarization_node._should_summarize(messages)
    
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Only summarize if needed"""
        if self.should_summarize(state):
            return await self.summarization_node(state)
        return state


# Usage in workflow:
"""
workflow = StateGraph(State)

# Add conditional summarization
summarization_node = SummarizationNode()
conditional_summarize = ConditionalSummarizationNode(summarization_node)

workflow.add_node("prepare_context", prepare_context)
workflow.add_node("summarize", conditional_summarize)
workflow.add_node("generate_response", generate_response)

# Conditional edge: only summarize if needed
workflow.add_conditional_edges(
    "prepare_context",
    lambda state: "summarize" if conditional_summarize.should_summarize(state) else "generate_response",
    {
        "summarize": "summarize",
        "generate_response": "generate_response"
    }
)

workflow.add_edge("summarize", "generate_response")
workflow.add_edge("generate_response", END)
"""

