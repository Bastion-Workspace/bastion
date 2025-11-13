"""
Query Expansion Tool - MCP Tool for Generating Alternative Search Queries
Allows LLM to generate alternative search queries when initial results are insufficient
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional

from mcp.schemas.tool_schemas import ToolResponse, ToolError
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class QueryExpansionInput(BaseModel):
    """Input for query expansion"""
    original_query: str = Field(..., description="Original search query that needs expansion")
    expansion_type: str = Field("semantic", description="Type of expansion: 'semantic', 'synonym', 'related', 'broader', 'narrower'")
    num_expansions: int = Field(3, ge=1, le=10, description="Number of alternative queries to generate")
    context: Optional[str] = Field(None, description="Additional context to guide expansion")


class QueryExpansionResult(BaseModel):
    """Result from query expansion"""
    expanded_query: str = Field(..., description="Alternative search query")
    expansion_type: str = Field(..., description="Type of expansion used")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this expansion")
    reasoning: str = Field(..., description="Explanation of why this expansion was generated")


class QueryExpansionOutput(BaseModel):
    """Output from query expansion"""
    original_query: str = Field(..., description="Original query")
    expansions: List[QueryExpansionResult] = Field(..., description="Generated alternative queries")
    total_generated: int = Field(..., description="Total number of expansions generated")
    expansion_time: float = Field(..., description="Time taken to generate expansions")


class QueryExpansionTool:
    """MCP tool for generating alternative search queries"""
    
    def __init__(self, embedding_manager=None, chat_service=None):
        """Initialize with required services"""
        self.embedding_manager = embedding_manager
        self.chat_service = chat_service
        self.name = "expand_query"
        self.description = "Generate alternative search queries when initial results are insufficient"
        
    async def initialize(self):
        """Initialize the query expansion tool"""
        if not self.embedding_manager:
            raise ValueError("EmbeddingManager is required")
        
        logger.info("🔍 QueryExpansionTool initialized")
    
    async def execute(self, input_data: QueryExpansionInput) -> ToolResponse:
        """Execute query expansion with the given parameters"""
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Expanding query: '{input_data.original_query}' (type: {input_data.expansion_type})")
            
            # Get the currently selected model from the chat service
            selected_model = None
            if self.chat_service and hasattr(self.chat_service, 'current_model') and self.chat_service.current_model:
                selected_model = self.chat_service.current_model
            else:
                # Fallback to default model
                from services.settings_service import settings_service
                selected_model = await settings_service.get_setting("llm_model")
            
            # Generate expansions using the embedding manager
            expanded_queries = await self.embedding_manager._generate_query_expansions(
                input_data.original_query, 
                selected_model
            )
            
            # Format results
            expansion_results = []
            for i, query in enumerate(expanded_queries[:input_data.num_expansions]):
                expansion_results.append(QueryExpansionResult(
                    expanded_query=query,
                    expansion_type=input_data.expansion_type,
                    confidence=0.8 - (i * 0.1),  # Decreasing confidence for each expansion
                    reasoning=f"Generated alternative query using {input_data.expansion_type} expansion"
                ))
            
            # Create output
            output = QueryExpansionOutput(
                original_query=input_data.original_query,
                expansions=expansion_results,
                total_generated=len(expansion_results),
                expansion_time=time.time() - start_time
            )
            
            logger.info(f"✅ Query expansion completed: {len(expansion_results)} alternatives in {output.expansion_time:.2f}s")
            
            return ToolResponse(
                success=True,
                data=output,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"❌ Query expansion failed: {e}")
            return ToolResponse(
                success=False,
                error=ToolError(
                    error_code="EXPANSION_FAILED",
                    error_message=str(e),
                    details={"original_query": input_data.original_query, "expansion_type": input_data.expansion_type}
                ),
                execution_time=time.time() - start_time
            )
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get MCP tool definition for registration"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": QueryExpansionInput.schema(),
            "outputSchema": QueryExpansionOutput.schema()
        } 