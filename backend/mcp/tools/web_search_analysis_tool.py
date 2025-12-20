"""
Web Search Analysis Tool - MCP Tool for Searching and Analyzing Web Content
Allows LLM to search the web, analyze results, and select which sources to ingest
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import httpx
from datetime import datetime

from mcp.schemas.tool_schemas import ToolResponse, ToolError
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from mcp.tools.web_search_tool import WebSearchResult as WebSearchResultType

logger = logging.getLogger(__name__)


class WebSearchAnalysisInput(BaseModel):
    """Input for web search analysis"""
    query: str = Field(..., description="Search query")
    num_results: int = Field(20, ge=1, le=20, description="Number of results to return for analysis")
    search_type: str = Field("web", description="Search type: 'web', 'news', 'images'")
    language: str = Field("en", description="Search language (ISO code)")
    region: str = Field("us", description="Search region")
    include_snippets: bool = Field(True, description="Whether to include result snippets")
    include_metadata: bool = Field(True, description="Whether to include basic metadata")


class WebSearchResult(BaseModel):
    """Individual web search result"""
    title: str = Field(..., description="Result title")
    url: str = Field(..., description="Result URL")
    snippet: str = Field(..., description="Result snippet/description")
    source: str = Field(..., description="Source domain")
    result_index: int = Field(..., description="Index of this result in search results")
    estimated_relevance: float = Field(0.0, description="Estimated relevance score (0-1)")
    content_preview: str = Field("", description="Brief content preview if available")


class WebSearchAnalysisOutput(BaseModel):
    """Output from web search analysis"""
    query: str = Field(..., description="Original search query")
    results: List[WebSearchResult] = Field(..., description="Search results for analysis")
    total_results: int = Field(..., description="Total number of results found")
    search_time: float = Field(..., description="Time taken for search")
    analysis_guidance: str = Field(..., description="Guidance for LLM analysis")
    next_steps: List[str] = Field(..., description="Suggested next steps for LLM")


class WebSearchAnalysisTool:
    """MCP tool for web search with LLM analysis"""
    
    def __init__(self, web_search_tool=None):
        """Initialize with required services"""
        self.web_search_tool = web_search_tool
        self.name = "web_search_and_analyze"
        self.description = "Search the web and return results for LLM analysis and selection"
        
    async def initialize(self):
        """Initialize the web search analysis tool"""
        if not self.web_search_tool:
            raise ValueError("WebSearchTool is required")
        
        logger.info("🔍 WebSearchAnalysisTool initialized")
    
    async def execute(self, input_data: WebSearchAnalysisInput) -> ToolResponse:
        """Execute web search for analysis"""
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Executing web search for analysis: '{input_data.query}'")
            
            # Perform web search
            search_results = await self._perform_web_search(input_data)
            
            if not search_results:
                return ToolResponse(
                    success=False,
                    error=ToolError(
                        error_code="SEARCH_FAILED",
                        error_message="No search results found",
                        details={"query": input_data.query}
                    ),
                    execution_time=time.time() - start_time
                )
            
            # Process results for analysis
            processed_results = []
            for i, result in enumerate(search_results[:input_data.num_results]):
                processed_result = WebSearchResult(
                    title=result.title,
                    url=result.url,
                    snippet=result.snippet if input_data.include_snippets else "",
                    source=result.source,
                    result_index=i + 1,
                    estimated_relevance=self._estimate_relevance(result, input_data.query),
                    content_preview=self._generate_content_preview(result)
                )
                processed_results.append(processed_result)
            
            # Generate analysis guidance
            analysis_guidance = self._generate_analysis_guidance(input_data.query, processed_results)
            next_steps = self._generate_next_steps(processed_results)
            
            # Create output
            output = WebSearchAnalysisOutput(
                query=input_data.query,
                results=processed_results,
                total_results=len(search_results),
                search_time=time.time() - start_time,
                analysis_guidance=analysis_guidance,
                next_steps=next_steps
            )
            
            logger.info(f"✅ Web search analysis completed: {len(processed_results)} results in {output.search_time:.2f}s")
            
            return ToolResponse(
                success=True,
                data=output,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"❌ Web search analysis failed: {e}")
            return ToolResponse(
                success=False,
                error=ToolError(
                    error_code="ANALYSIS_FAILED",
                    error_message=str(e),
                    details={"query": input_data.query}
                ),
                execution_time=time.time() - start_time
            )
    
    async def _perform_web_search(self, input_data: WebSearchAnalysisInput) -> List[Dict[str, Any]]:
        """Perform web search using the web search tool"""
        try:
            # Create input for web search tool
            from mcp.tools.web_search_tool import WebSearchInput
            
            search_input = WebSearchInput(
                query=input_data.query,
                num_results=input_data.num_results,
                search_type=input_data.search_type,
                language=input_data.language,
                region=input_data.region
            )
            
            # Execute web search
            search_response = await self.web_search_tool.execute(search_input)
            
            if not search_response.success:
                logger.error(f"❌ Web search failed: {search_response.error}")
                return []
            
            # Extract results
            search_data = search_response.data
            return search_data.results
            
        except Exception as e:
            logger.error(f"❌ Web search execution failed: {e}")
            return []
    
    def _estimate_relevance(self, result: "WebSearchResultType", query: str) -> float:
        """Estimate relevance score for a search result"""
        try:
            title = result.title.lower()
            snippet = result.snippet.lower()
            url = result.url.lower()
            query_terms = query.lower().split()
            
            # Simple relevance scoring
            score = 0.0
            
            # Title relevance (highest weight)
            title_matches = sum(1 for term in query_terms if term in title)
            score += (title_matches / len(query_terms)) * 0.6
            
            # Snippet relevance
            snippet_matches = sum(1 for term in query_terms if term in snippet)
            score += (snippet_matches / len(query_terms)) * 0.3
            
            # URL relevance (lowest weight)
            url_matches = sum(1 for term in query_terms if term in url)
            score += (url_matches / len(query_terms)) * 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to estimate relevance: {e}")
            return 0.5
    
    def _generate_content_preview(self, result: "WebSearchResultType") -> str:
        """Generate a brief content preview"""
        try:
            snippet = result.snippet
            if len(snippet) > 150:
                return snippet[:150] + "..."
            return snippet
        except Exception:
            return ""
    
    def _generate_analysis_guidance(self, query: str, results: List[WebSearchResult]) -> str:
        """Generate guidance for LLM analysis"""
        guidance = f"Analyze these {len(results)} search results for '{query}'. "
        guidance += "Consider:\n"
        guidance += "• Relevance to the search query\n"
        guidance += "• Source credibility and authority\n"
        guidance += "• Content recency and timeliness\n"
        guidance += "• Information depth and comprehensiveness\n"
        guidance += "• Potential bias or perspective\n\n"
        guidance += "Select 1-3 of the most valuable sources for ingestion into the knowledge base. "
        guidance += "Focus on high-quality, relevant content that adds unique value. "
        guidance += "CRITICAL: You must follow this analysis with web_ingest_selected_results to actually store the chosen content."
        
        return guidance
    
    def _generate_next_steps(self, results: List[WebSearchResult]) -> List[str]:
        """Generate suggested next steps for LLM"""
        steps = [
            "Review each result for relevance and quality",
            "Select the most valuable sources for ingestion (typically 1-3 best sources)",
            "IMMEDIATELY use 'web_ingest_selected_results' to ingest chosen sources",
            "Use the ingested content in your final response with proper citations",
            "Consider searching for additional perspectives if critical information is still missing"
        ]
        return steps
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get MCP tool definition for registration"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": WebSearchAnalysisInput.schema(),
            "outputSchema": WebSearchAnalysisOutput.schema()
        } 