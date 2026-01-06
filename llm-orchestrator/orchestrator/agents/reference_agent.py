"""
Reference Agent for LLM Orchestrator
Analyzes and interacts with personal reference documents (journals, logs, records, etc.)
Read-only agent focused on analysis and understanding
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional, TypedDict
from datetime import datetime

from langgraph.graph import StateGraph, END
from .base_agent import BaseAgent, TaskStatus
from orchestrator.tools.reference_file_loader import load_referenced_files
from orchestrator.tools.document_tools import get_document_content_tool
from orchestrator.models.reference_models import (
    QueryComplexityAnalysis,
    PatternAnalysisResult,
    InsightResult,
    ReferenceResponse
)

logger = logging.getLogger(__name__)


class ReferenceAgentState(TypedDict):
    """State for reference agent LangGraph workflow"""
    query: str
    user_id: str
    metadata: Dict[str, Any]
    messages: List[Any]
    shared_memory: Dict[str, Any]
    
    # Reference document context
    reference_content: str  # Main active editor content
    referenced_files: Dict[str, Any]  # Additional referenced documents
    document_type: str  # Type of reference (journal, food_log, etc.)
    
    # Analysis tracking
    query_complexity: str  # "simple_qa", "pattern_analysis", "insights", "unrelated_query"
    query_relevance: float  # 0.0-1.0 score of how relevant query is to document
    is_unrelated: bool  # True if query is unrelated to document content
    analysis_results: Dict[str, Any]
    patterns_found: List[Dict[str, Any]]
    
    # Research integration
    needs_external_info: bool
    research_query: Optional[str]
    research_results: Optional[Dict[str, Any]]
    
    # Weather tool integration
    needs_weather: bool
    weather_request: Optional[Dict[str, Any]]  # Structured weather request
    weather_results: Optional[Dict[str, Any]]  # Weather data results
    
    # Math tool integration
    needs_calculations: bool
    calculation_type: Optional[str]  # "expression", "formula", "conversion"
    calculation_request: Optional[Dict[str, Any]]  # Structured calculation request
    calculation_results: Optional[Dict[str, Any]]  # Math tool results
    
    # Visualization integration
    needs_visualization: bool
    visualization_request: Optional[Dict[str, Any]]  # Structured visualization request
    visualization_results: Optional[Dict[str, Any]]  # Chart generation results
    
    # Response
    response: Dict[str, Any]
    task_status: str
    error: str


class ReferenceAgent(BaseAgent):
    """
    Reference Analysis Agent
    Provides analysis and interaction with personal reference documents
    Read-only: No editing capabilities, pure analysis and understanding
    """
    
    def __init__(self):
        super().__init__("reference_agent")
        logger.info("📚 Reference Agent ready for document analysis!")
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """Build LangGraph workflow for reference agent"""
        workflow = StateGraph(ReferenceAgentState)
        
        # Add nodes
        workflow.add_node("load_reference_context", self._load_reference_context_node)
        workflow.add_node("analyze_query_complexity", self._analyze_query_complexity_node)
        workflow.add_node("handle_unrelated_query", self._handle_unrelated_query_node)
        workflow.add_node("process_simple_qa", self._process_simple_qa_node)
        workflow.add_node("process_pattern_analysis", self._process_pattern_analysis_node)
        workflow.add_node("process_insights", self._process_insights_node)
        workflow.add_node("perform_calculations", self._perform_calculations_node)
        workflow.add_node("perform_visualization", self._perform_visualization_node)
        workflow.add_node("get_weather_data", self._get_weather_data_node)
        workflow.add_node("call_research_subgraph", self._call_research_subgraph_node)
        workflow.add_node("synthesize_response", self._synthesize_response_node)
        workflow.add_node("format_response", self._format_response_node)
        
        # Entry point
        workflow.set_entry_point("load_reference_context")
        
        # Edges
        workflow.add_edge("load_reference_context", "analyze_query_complexity")
        
        # Route based on query complexity and relevance
        workflow.add_conditional_edges(
            "analyze_query_complexity",
            self._route_from_complexity,
            {
                "unrelated_query": "handle_unrelated_query",
                "simple_qa": "process_simple_qa",
                "pattern_analysis": "process_pattern_analysis",
                "insights": "process_insights"
            }
        )
        
        # Unrelated query goes directly to formatting
        workflow.add_edge("handle_unrelated_query", "format_response")
        
        # All analysis nodes check if calculations, visualization, weather, or research are needed
        workflow.add_conditional_edges(
            "process_simple_qa",
            self._route_from_analysis,
            {
                "calculations": "perform_calculations",
                "visualization": "perform_visualization",
                "weather": "get_weather_data",
                "research": "call_research_subgraph",
                "synthesize": "synthesize_response"
            }
        )
        
        workflow.add_conditional_edges(
            "process_pattern_analysis",
            self._route_from_analysis,
            {
                "calculations": "perform_calculations",
                "visualization": "perform_visualization",
                "weather": "get_weather_data",
                "research": "call_research_subgraph",
                "synthesize": "synthesize_response"
            }
        )
        
        workflow.add_conditional_edges(
            "process_insights",
            self._route_from_analysis,
            {
                "calculations": "perform_calculations",
                "visualization": "perform_visualization",
                "weather": "get_weather_data",
                "research": "call_research_subgraph",
                "synthesize": "synthesize_response"
            }
        )
        
        # After calculations, check if visualization, weather, or research is still needed
        workflow.add_conditional_edges(
            "perform_calculations",
            self._route_after_calculations,
            {
                "visualization": "perform_visualization",
                "weather": "get_weather_data",
                "research": "call_research_subgraph",
                "synthesize": "synthesize_response"
            }
        )
        
        # After visualization, check if weather or research is still needed
        workflow.add_conditional_edges(
            "perform_visualization",
            self._route_after_visualization,
            {
                "weather": "get_weather_data",
                "research": "call_research_subgraph",
                "synthesize": "synthesize_response"
            }
        )
        
        # After weather, check if research is still needed
        workflow.add_conditional_edges(
            "get_weather_data",
            self._route_after_weather,
            {
                "research": "call_research_subgraph",
                "synthesize": "synthesize_response"
            }
        )
        
        # After research, check if calculations are still needed (may have been missed or enabled by research)
        workflow.add_conditional_edges(
            "call_research_subgraph",
            self._route_after_research,
            {
                "calculations": "perform_calculations",
                "synthesize": "synthesize_response"
            }
        )
        
        # Synthesis routes to formatting
        workflow.add_edge("synthesize_response", "format_response")
        
        # Formatting is the end
        workflow.add_edge("format_response", END)
        
        return workflow.compile(checkpointer=checkpointer)
    
    def _route_from_complexity(self, state: ReferenceAgentState) -> str:
        """Route based on query complexity and relevance"""
        # Check if query is unrelated first
        if state.get("is_unrelated", False):
            return "unrelated_query"
        complexity = state.get("query_complexity", "simple_qa")
        return complexity
    
    def _route_from_analysis(self, state: ReferenceAgentState) -> str:
        """Route based on whether calculations, visualization, weather, or research are needed"""
        # Prioritize calculations first, then visualization, then weather, then research
        if state.get("needs_calculations", False):
            return "calculations"
        if state.get("needs_visualization", False):
            return "visualization"
        if state.get("needs_weather", False):
            return "weather"
        if state.get("needs_external_info", False):
            return "research"
        return "synthesize"
    
    def _route_after_calculations(self, state: ReferenceAgentState) -> str:
        """Route after calculations - check if visualization, weather, or research is still needed"""
        if state.get("needs_visualization", False):
            return "visualization"
        if state.get("needs_weather", False):
            return "weather"
        if state.get("needs_external_info", False):
            return "research"
        return "synthesize"
    
    def _route_after_visualization(self, state: ReferenceAgentState) -> str:
        """Route after visualization - check if weather or research is still needed"""
        if state.get("needs_weather", False):
            return "weather"
        if state.get("needs_external_info", False):
            return "research"
        return "synthesize"
    
    def _route_after_weather(self, state: ReferenceAgentState) -> str:
        """Route after weather - check if research is still needed"""
        if state.get("needs_external_info", False):
            return "research"
        return "synthesize"
    
    def _route_after_research(self, state: ReferenceAgentState) -> str:
        """Route after research - check if calculations are still needed"""
        # Check if calculations were needed but not yet performed
        # This handles cases where:
        # 1. Complexity analysis missed calculation detection
        # 2. Research provided data that enables calculations
        # 3. Query requires calculations that should happen after research
        if state.get("needs_calculations", False):
            logger.info("📊 Calculations still needed after research - routing to calculations")
            return "calculations"
        return "synthesize"
    
    async def _load_reference_context_node(self, state: ReferenceAgentState) -> Dict[str, Any]:
        """Load active editor content and referenced files"""
        try:
            user_id = state.get("user_id", "")
            metadata = state.get("metadata", {})
            shared_memory = metadata.get("shared_memory", {})
            
            # Get active editor
            active_editor = metadata.get("active_editor") or shared_memory.get("active_editor", {})
            
            if not active_editor:
                logger.warning("⚠️ No active editor found")
                return {
                    "reference_content": "",
                    "referenced_files": {},
                    "document_type": "unknown",
                    "error": "No active editor found. Please open a reference document with type: reference in frontmatter.",
                    # ✅ CRITICAL: Preserve state even on error
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", "")
                }
            
            # Check if document type is "reference"
            frontmatter = active_editor.get("frontmatter", {})
            doc_type = frontmatter.get("type", "").lower()
            
            if doc_type != "reference":
                logger.warning(f"⚠️ Document type is '{doc_type}', not 'reference'")
                return {
                    "reference_content": "",
                    "referenced_files": {},
                    "document_type": doc_type,
                    "error": f"Document type is '{doc_type}', not 'reference'. Please open a document with type: reference in frontmatter.",
                    # ✅ CRITICAL: Preserve state even on error
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", "")
                }
            
            # Get main reference content
            reference_content = active_editor.get("content", "")
            document_type = frontmatter.get("document_type", "reference")  # e.g., "journal", "food_log", "weight_log"
            
            logger.info(f"📚 Loaded main reference document: {len(reference_content)} chars, type: {document_type}")
            
            # Load referenced files from frontmatter
            reference_config = {
                "references": ["references", "reference", "related", "files"],
                "other": ["other", "additional", "supplementary"]
            }
            
            result = await load_referenced_files(
                active_editor=active_editor,
                user_id=user_id,
                reference_config=reference_config,
                doc_type_filter="reference"
            )
            
            referenced_files = result.get("loaded_files", {})
            total_referenced = sum(len(docs) for docs in referenced_files.values() if isinstance(docs, list))
            
            logger.info(f"📚 Loaded {total_referenced} referenced file(s)")
            
            return {
                "reference_content": reference_content,
                "referenced_files": referenced_files,
                "document_type": document_type,
                # ✅ CRITICAL: Preserve state for subsequent nodes
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to load reference context: {e}")
            return {
                "reference_content": "",
                "referenced_files": {},
                "document_type": "unknown",
                "error": str(e),
                # ✅ CRITICAL: Preserve state even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    async def _analyze_query_complexity_node(self, state: ReferenceAgentState) -> Dict[str, Any]:
        """Analyze query complexity and determine analysis level needed"""
        try:
            query = state.get("query", "")
            reference_content = state.get("reference_content", "")
            document_type = state.get("document_type", "reference")
            
            if not reference_content:
                return {
                    "query_complexity": "simple_qa",
                    "needs_external_info": False,
                    "error": "No reference content available",
                    # ✅ CRITICAL: Preserve state even on error
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", "")
                }
            
            # Use fast model for complexity analysis
            fast_model = self._get_fast_model(state)
            llm = self._get_llm(temperature=0.1, model=fast_model, state=state)
            
            # Build prompt for complexity analysis with relevance checking
            # Sample document content for relevance assessment (first 1000 chars)
            content_sample = reference_content[:1000] if reference_content else ""
            
            prompt = f"""Analyze this query about a {document_type} reference document and determine the complexity level needed.

**QUERY**: {query}

**DOCUMENT TYPE**: {document_type}

**DOCUMENT CONTENT SAMPLE** (for relevance assessment):
{content_sample}

**ANALYSIS NEEDED**:
1. **query_relevance**: Score from 0.0 to 1.0 indicating how relevant the query is to the document content. 
   - 0.9-1.0: Highly relevant (query directly asks about document content)
   - 0.5-0.89: Moderately relevant (query might relate to document themes/topics)
   - 0.0-0.49: Low relevance (query appears unrelated to document content)
2. **is_unrelated**: true if query_relevance < 0.3 (query is clearly unrelated to document)
3. **complexity_level**: "simple_qa" (factual questions like "What did I write on X date?"), "pattern_analysis" (trends/frequencies like "Show me days I ate pizza"), or "insights" (deep analysis like "What patterns do you notice?")
4. **needs_calculations**: Does this query require mathematical calculations? **ALWAYS set to true if query contains:**
   - "calculate", "compute", "figure out", "determine" + any technical term (BTU, heat loss, voltage, etc.)
   - "heat loss", "actual heat losses", "Manual J", "BTU requirements", "HVAC sizing"
   - "how much", "what is the", "total" + technical calculations
   - Any formula-related terms (BTU, electrical, conversion, etc.)
   - Examples: "calculate actual heat losses", "what BTU do I need", "compute the heat loss", "figure out the BTU requirements"
5. **calculation_type**: If calculations needed, what type? "formula" (BTU, heat loss, electrical formulas), "expression" (simple math), or "conversion" (unit conversion)
6. **needs_external_info**: Does this query require external information (e.g., nutritional data, definitions, research)?
7. **research_query**: If external info is needed, what should be researched? (e.g., "nutritional content of pizza")
8. **needs_weather**: Does this query ask about weather conditions (current, forecast, or historical)? Examples: "What's the weather?", "Weather forecast", "What was the weather on [date]?"
9. **weather_request**: If weather is needed, structure as: {{"location": "city name or location", "request_type": "current|forecast|history", "date_str": "YYYY-MM-DD or YYYY-MM" (for history)}}
10. **needs_visualization**: Would this query benefit from a chart or graph? **CRITICAL**: Only recommend visualization if:
   - The user EXPLICITLY requests a chart/graph/plot/visualization, OR
   - There is SUBSTANTIAL structured data (multiple data points, clear patterns, meaningful comparisons)
   - The visualization would provide SIGNIFICANT value beyond what text can convey
   - There are at least 3-5 distinct data points to visualize (not just 1-2 values)
   - The data has clear patterns, trends, or relationships worth visualizing
   
   **DO NOT recommend visualization for:**
   - Simple single-value questions ("What's my weight today?")
   - Questions with insufficient data points (< 3 data points)
   - Questions where text would be clearer than a chart
   - Exploratory queries without clear visualization intent
   
   Consider:
   - Queries explicitly asking to "show", "graph", "chart", "plot", or "visualize" data → needs_visualization=true
   - Queries about trends over time with multiple data points (weight, mood, food intake over weeks/months) → needs_visualization=true
   - Queries comparing values across categories or time periods with sufficient data → needs_visualization=true
   - Queries about distributions or frequency patterns with multiple categories → needs_visualization=true
   - Simple factual questions or single data point queries → needs_visualization=false
11. **visualization_type**: If visualization needed, what type? "bar" (comparisons), "line" (trends over time), "pie" (proportions), "scatter" (correlations), "area" (cumulative trends), "heatmap" (2D patterns), "box_plot" (distributions), "histogram" (frequency distributions)

**RELEVANCE EXAMPLES**:
- "What did I write in my journal on December 5th?" → relevance: 0.95 (directly about document)
- "What's the weather today?" → relevance: 0.1 (completely unrelated to journal)
- "Tell me about quantum physics" → relevance: 0.05 (unrelated to personal reference document)
- "Show me all days I mentioned feeling anxious" → relevance: 0.9 (directly about document patterns)

**COMPLEXITY EXAMPLES**:
- "What did I write in my journal on December 5th?" → simple_qa, no calculations, no external info, no weather, no visualization
- "Show me all days I mentioned feeling anxious" → pattern_analysis, no calculations, no external info, no weather, no visualization (text list is sufficient)
- "Graph my weight over time" → pattern_analysis, no calculations, no external info, no weather, needs visualization (line chart - explicit request)
- "Chart the frequency of different foods I ate" → pattern_analysis, no calculations, no external info, no weather, needs visualization (bar or pie chart - explicit request)
- "What's my weight today?" → simple_qa, no calculations, no external info, no weather, no visualization (single value, no chart needed)
- "Show me my weight trends over the last 6 months" → pattern_analysis, no calculations, no external info, no weather, needs visualization (line chart - multiple data points, clear trend)
- "How many times did I eat pizza?" → simple_qa, no calculations, no external info, no weather, no visualization (single count, text is sufficient)
- "What BTU requirements do I need for these rooms?" → simple_qa, needs calculations (formula: btu_hvac), no external info, no weather, no visualization
- "Calculate actual heat losses" → simple_qa, needs calculations (formula: manual_j_heat_loss), no external info, no weather, no visualization
- "What's the heat loss for this building?" → simple_qa, needs calculations (formula: manual_j_heat_loss), no external info, no weather, no visualization
- "What's the calorie count of the foods I logged yesterday?" → simple_qa, no calculations, needs external info (nutritional data), no weather, no visualization
- "What patterns do you notice in my logs?" → insights, no calculations, no external info, no weather, no visualization
- "What's the weather today?" → simple_qa, no calculations, no external info, needs weather (current), no visualization
- "What was the weather on December 5th?" → simple_qa, no calculations, no external info, needs weather (history, date_str: "2024-12-05"), no visualization

Return ONLY valid JSON:
{{
  "query_relevance": 0.95,
  "is_unrelated": false,
  "complexity_level": "simple_qa",
  "needs_calculations": false,
  "calculation_type": null,
  "needs_external_info": false,
  "research_query": null,
  "needs_visualization": false,
  "visualization_type": null,
  "reasoning": "Brief explanation"
}}"""
            
            # Build messages with conversation history using standardized helper
            messages_list = state.get("messages", [])
            system_prompt = "You are a query analysis assistant. Analyze queries to determine their relevance to reference documents and complexity level."
            llm_messages = self._build_conversational_agent_messages(
                system_prompt=system_prompt,
                user_prompt=prompt,
                messages_list=messages_list,
                look_back_limit=4
            )
            
            try:
                # Try structured output
                schema = {
                    "type": "object",
                    "properties": {
                        "query_relevance": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "is_unrelated": {"type": "boolean"},
                        "complexity_level": {"type": "string"},
                        "needs_calculations": {"type": "boolean"},
                        "calculation_type": {"type": ["string", "null"]},
                        "needs_external_info": {"type": "boolean"},
                        "research_query": {"type": ["string", "null"]},
                        "needs_weather": {"type": "boolean"},
                        "weather_request": {"type": ["object", "null"]},
                        "needs_visualization": {"type": "boolean"},
                        "visualization_type": {"type": ["string", "null"]},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["query_relevance", "is_unrelated", "complexity_level", "needs_calculations", "needs_external_info", "needs_weather", "needs_visualization"]
                }
                structured_llm = llm.with_structured_output(schema)
                result = await structured_llm.ainvoke(llm_messages)
                result_dict = result if isinstance(result, dict) else result.dict() if hasattr(result, 'dict') else result.model_dump()
            except Exception:
                # Fallback to manual parsing
                response = await llm.ainvoke(llm_messages)
                content = response.content if hasattr(response, 'content') else str(response)
                result_dict = self._parse_json_response(content) or {}
            
            query_relevance = result_dict.get("query_relevance", 1.0)
            is_unrelated = result_dict.get("is_unrelated", False)
            # Override is_unrelated if relevance is below threshold
            if query_relevance < 0.3:
                is_unrelated = True
            
            complexity = result_dict.get("complexity_level", "simple_qa")
            needs_calculations = result_dict.get("needs_calculations", False)
            calculation_type = result_dict.get("calculation_type")
            needs_external = result_dict.get("needs_external_info", False)
            research_query = result_dict.get("research_query")
            needs_weather = result_dict.get("needs_weather", False)
            weather_request = result_dict.get("weather_request")
            needs_visualization = result_dict.get("needs_visualization", False)
            visualization_type = result_dict.get("visualization_type")
            
            # Log calculation detection with emphasis
            if needs_calculations:
                logger.info(f"🔢 CALCULATION DETECTED: needs_calculations=True, type={calculation_type}, query='{query[:100]}'")
            else:
                # Check if query contains calculation keywords but wasn't detected
                calc_keywords = ["calculate", "compute", "heat loss", "btu", "manual j", "figure out", "determine"]
                if any(kw in query.lower() for kw in calc_keywords):
                    logger.warning(f"⚠️ CALCULATION KEYWORDS FOUND but needs_calculations=False: query='{query[:100]}'")
            
            logger.info(f"📚 Query relevance: {query_relevance:.2f}, is_unrelated: {is_unrelated}, complexity: {complexity}, needs calculations: {needs_calculations}, needs external info: {needs_external}, needs weather: {needs_weather}, needs visualization: {needs_visualization}")
            
            return {
                "query_relevance": query_relevance,
                "is_unrelated": is_unrelated,
                "query_complexity": complexity,
                "needs_calculations": needs_calculations,
                "calculation_type": calculation_type,
                "needs_external_info": needs_external,
                "research_query": research_query,
                "needs_weather": needs_weather,
                "weather_request": weather_request,
                "needs_visualization": needs_visualization,
                "visualization_request": {"chart_type": visualization_type} if visualization_type else None,
                # ✅ CRITICAL: Preserve state for subsequent nodes
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
            
        except Exception as e:
            logger.error(f"❌ Complexity analysis failed: {e}")
            return {
                "query_complexity": "simple_qa",
                "query_relevance": 1.0,
                "is_unrelated": False,
                "needs_external_info": False,
                "needs_weather": False,
                "needs_visualization": False,
                # ✅ CRITICAL: Preserve state even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    async def _handle_unrelated_query_node(self, state: ReferenceAgentState) -> Dict[str, Any]:
        """Handle queries that are unrelated to the reference document"""
        try:
            query = state.get("query", "")
            document_type = state.get("document_type", "reference")
            query_relevance = state.get("query_relevance", 0.0)
            
            # Build helpful response explaining the situation
            response_text = f"""I notice your question doesn't seem related to the {document_type} document you have open.

**Your question**: "{query}"

**Why this happened**: The question appears to be about a general topic rather than the content of your reference document (relevance score: {query_relevance:.1%}).

**What you can do**:
1. **Ask about the document**: Try questions like:
   - "What did I write about [topic]?"
   - "Show me patterns in my logs"
   - "What trends do you notice?"
   - "When did I mention [keyword]?"

2. **Ask as a general question**: To ask this as a general question:
   - Close the reference document, or
   - Uncheck "Prefer Editor" in the chat sidebar

3. **If you meant to ask about the document**: Try rephrasing to reference specific content, dates, or patterns from your document.

Would you like to rephrase your question about the document, or would you prefer to ask this as a general question instead?"""
            
            logger.info(f"📚 Detected unrelated query (relevance: {query_relevance:.2f}) - providing helpful response")
            
            return {
                "analysis_results": {
                    "response": response_text,
                    "complexity_level": "unrelated_query",
                    "query_relevance": query_relevance
                },
                "response": {
                    "response": response_text,
                    "task_status": "complete",
                    "complexity_level": "unrelated_query",
                    "query_relevance": query_relevance,
                    "is_unrelated": True
                },
                # ✅ CRITICAL: Preserve state for subsequent nodes
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
            
        except Exception as e:
            logger.error(f"❌ Unrelated query handling failed: {e}")
            return {
                "analysis_results": {
                    "response": f"I couldn't process your query. Error: {str(e)}",
                    "complexity_level": "unrelated_query"
                },
                "error": str(e),
                # ✅ CRITICAL: Preserve state even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    async def _process_simple_qa_node(self, state: ReferenceAgentState) -> Dict[str, Any]:
        """Process simple Q&A queries"""
        try:
            query = state.get("query", "")
            reference_content = state.get("reference_content", "")
            referenced_files = state.get("referenced_files", {})
            document_type = state.get("document_type", "reference")
            messages = state.get("messages", [])
            
            # Build context from main document and referenced files
            context_parts = [f"### MAIN REFERENCE DOCUMENT ({document_type}):\n{reference_content}\n"]
            
            # Add referenced files
            for category, docs in referenced_files.items():
                if isinstance(docs, list):
                    for doc in docs:
                        title = doc.get("title", "Referenced File")
                        content = doc.get("content", "")
                        context_parts.append(f"### REFERENCED FILE ({title}):\n{content}\n")
            
            context = "\n".join(context_parts)
            
            # Build prompt
            system_prompt = self._build_reference_prompt()
            
            prompt = f"""Answer this factual question about the reference document:

**QUERY**: {query}

**REFERENCE DOCUMENT CONTENT**:
{context}

**INSTRUCTIONS**:
- Answer based ONLY on the document content provided
- Be factual and precise
- If the information is not in the document, say so clearly
- Keep your response concise and direct
- Use conversation history to understand follow-up questions and maintain context

Return your response as natural language text (not JSON)."""
            
            llm = self._get_llm(temperature=0.3, state=state)
            llm_messages = self._build_conversational_agent_messages(
                system_prompt=system_prompt,
                user_prompt=prompt,
                messages_list=messages,
                look_back_limit=6
            )
            response = await llm.ainvoke(llm_messages)
            
            content = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "analysis_results": {
                    "response": content,
                    "complexity_level": "simple_qa"
                },
                # ✅ CRITICAL: Preserve state for subsequent nodes
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
            
        except Exception as e:
            logger.error(f"❌ Simple Q&A processing failed: {e}")
            return {
                "analysis_results": {
                    "response": f"Error processing query: {str(e)}",
                    "complexity_level": "simple_qa"
                },
                "error": str(e),
                # ✅ CRITICAL: Preserve state even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    async def _process_pattern_analysis_node(self, state: ReferenceAgentState) -> Dict[str, Any]:
        """Process pattern analysis queries"""
        try:
            query = state.get("query", "")
            reference_content = state.get("reference_content", "")
            referenced_files = state.get("referenced_files", {})
            document_type = state.get("document_type", "reference")
            messages = state.get("messages", [])
            
            # Build context
            context_parts = [f"### MAIN REFERENCE DOCUMENT ({document_type}):\n{reference_content}\n"]
            
            for category, docs in referenced_files.items():
                if isinstance(docs, list):
                    for doc in docs:
                        title = doc.get("title", "Referenced File")
                        content = doc.get("content", "")
                        context_parts.append(f"### REFERENCED FILE ({title}):\n{content}\n")
            
            context = "\n".join(context_parts)
            
            # Build prompt for pattern analysis
            system_prompt = self._build_reference_prompt()
            
            prompt = f"""Analyze this reference document for patterns, trends, and frequencies:

**QUERY**: {query}

**REFERENCE DOCUMENT CONTENT**:
{context}

**ANALYSIS TASK**:
- Find patterns, trends, and frequencies in the document
- Identify temporal patterns (daily, weekly, monthly)
- Find correlations and recurring items
- Extract specific examples with dates/context
- Use conversation history to understand follow-up questions and build on previous analysis

**OUTPUT FORMAT**: Return valid JSON matching this schema:
{{
  "patterns": [
    {{
      "pattern_type": "frequency|temporal|correlation|anomaly|trend",
      "description": "Description of the pattern",
      "occurrences": 5,
      "examples": ["example1", "example2"],
      "metadata": {{"dates": ["2024-01-01"], "values": [100]}}
    }}
  ],
  "temporal_trends": ["trend1", "trend2"],
  "frequencies": {{"item1": 5, "item2": 12}},
  "correlations": ["correlation1"],
  "summary": "Summary of all patterns found"
}}

Return ONLY the JSON object, no markdown, no code blocks."""
            
            llm = self._get_llm(temperature=0.3, state=state)
            llm_messages = self._build_conversational_agent_messages(
                system_prompt=system_prompt,
                user_prompt=prompt,
                messages_list=messages,
                look_back_limit=6
            )
            response = await llm.ainvoke(llm_messages)
            
            content = response.content if hasattr(response, 'content') else str(response)
            parsed = self._parse_json_response(content) or {}
            
            # Validate with Pydantic if possible
            try:
                pattern_result = PatternAnalysisResult(**parsed)
                patterns_found = pattern_result.dict() if hasattr(pattern_result, 'dict') else pattern_result.model_dump()
            except Exception:
                patterns_found = parsed
            
            return {
                "analysis_results": {
                    "response": patterns_found.get("summary", "Pattern analysis complete"),
                    "patterns": patterns_found.get("patterns", []),
                    "temporal_trends": patterns_found.get("temporal_trends", []),
                    "frequencies": patterns_found.get("frequencies", {}),
                    "correlations": patterns_found.get("correlations", []),
                    "complexity_level": "pattern_analysis"
                },
                "patterns_found": patterns_found.get("patterns", []),
                # ✅ CRITICAL: Preserve state for subsequent nodes
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
            
        except Exception as e:
            logger.error(f"❌ Pattern analysis failed: {e}")
            return {
                "analysis_results": {
                    "response": f"Error analyzing patterns: {str(e)}",
                    "complexity_level": "pattern_analysis"
                },
                "error": str(e),
                # ✅ CRITICAL: Preserve state even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    async def _process_insights_node(self, state: ReferenceAgentState) -> Dict[str, Any]:
        """Process insights and deep analysis queries"""
        try:
            query = state.get("query", "")
            reference_content = state.get("reference_content", "")
            referenced_files = state.get("referenced_files", {})
            document_type = state.get("document_type", "reference")
            messages = state.get("messages", [])
            
            # Build context
            context_parts = [f"### MAIN REFERENCE DOCUMENT ({document_type}):\n{reference_content}\n"]
            
            for category, docs in referenced_files.items():
                if isinstance(docs, list):
                    for doc in docs:
                        title = doc.get("title", "Referenced File")
                        content = doc.get("content", "")
                        context_parts.append(f"### REFERENCED FILE ({title}):\n{content}\n")
            
            context = "\n".join(context_parts)
            
            # Build prompt for insights
            system_prompt = self._build_reference_prompt()
            
            prompt = f"""Provide deep insights and analysis about this reference document:

**QUERY**: {query}

**REFERENCE DOCUMENT CONTENT**:
{context}

**ANALYSIS TASK**:
- Identify key trends and patterns
- Find anomalies or outliers
- Provide observations and insights
- Generate actionable recommendations (if appropriate)
- Support insights with specific evidence from the document
- Use conversation history to understand follow-up questions and build on previous insights

**OUTPUT FORMAT**: Return valid JSON matching this schema:
{{
  "insights": [
    {{
      "insight_type": "trend|anomaly|recommendation|observation|correlation",
      "title": "Short title",
      "description": "Detailed description",
      "confidence": 0.85,
      "supporting_evidence": ["evidence1", "evidence2"]
    }}
  ],
  "key_trends": ["trend1", "trend2"],
  "anomalies": ["anomaly1"],
  "recommendations": ["recommendation1"],
  "summary": "Overall summary of insights"
}}

Return ONLY the JSON object, no markdown, no code blocks."""
            
            llm = self._get_llm(temperature=0.4, state=state)  # Slightly higher temp for creative insights
            llm_messages = self._build_conversational_agent_messages(
                system_prompt=system_prompt,
                user_prompt=prompt,
                messages_list=messages,
                look_back_limit=6
            )
            response = await llm.ainvoke(llm_messages)
            
            content = response.content if hasattr(response, 'content') else str(response)
            parsed = self._parse_json_response(content) or {}
            
            # Validate with Pydantic if possible
            try:
                insight_result = InsightResult(**parsed)
                insights = insight_result.dict() if hasattr(insight_result, 'dict') else insight_result.model_dump()
            except Exception:
                insights = parsed
            
            return {
                "analysis_results": {
                    "response": insights.get("summary", "Insights analysis complete"),
                    "insights": insights.get("insights", []),
                    "key_trends": insights.get("key_trends", []),
                    "anomalies": insights.get("anomalies", []),
                    "recommendations": insights.get("recommendations", []),
                    "complexity_level": "insights"
                },
                # ✅ CRITICAL: Preserve state for subsequent nodes
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
            
        except Exception as e:
            logger.error(f"❌ Insights processing failed: {e}")
            return {
                "analysis_results": {
                    "response": f"Error generating insights: {str(e)}",
                    "complexity_level": "insights"
                },
                "error": str(e),
                # ✅ CRITICAL: Preserve state even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    async def _perform_calculations_node(self, state: ReferenceAgentState) -> Dict[str, Any]:
        """Perform calculations using math tool"""
        try:
            query = state.get("query", "")
            reference_content = state.get("reference_content", "")
            calculation_type = state.get("calculation_type", "formula")
            messages = state.get("messages", [])
            
            # Use LLM to extract calculation request from query and document
            fast_model = self._get_fast_model(state)
            llm = self._get_llm(temperature=0.1, model=fast_model, state=state)
            
            # Build prompt to extract calculation request
            prompt = f"""Extract calculation request from this query and reference document:

**QUERY**: {query}

**REFERENCE DOCUMENT CONTENT**:
{reference_content[:2000]}  # Limit content for prompt

**TASK**: Extract numerical data and structure a calculation request.

**CALCULATION TYPE**: {calculation_type}

**INSTRUCTIONS**:
1. Extract all numerical values from the document (measurements, dimensions, etc.)
2. Use conversation history to understand follow-up questions and previous calculations
3. Identify the calculation type:
   - If heat loss calculation (Manual J, actual heat losses, building heat loss): use formula "manual_j_heat_loss"
   - If BTU/HVAC sizing (room sizing, HVAC requirements): use formula "btu_hvac"
   - If electrical (voltage, current, resistance): use appropriate "ohms_law_*" formula
   - If simple math: use "expression" type
   - If unit conversion: use "conversion" type
4. Structure the calculation request as JSON

**IMPORTANT - Heat Loss vs BTU Sizing**:
- "heat loss", "actual heat losses", "calculate heat loss", "Manual J" → use "manual_j_heat_loss" formula
- "BTU requirements", "HVAC sizing", "room sizing" → use "btu_hvac" formula

**OUTPUT FORMAT**: Return ONLY valid JSON:
{{
  "tool": "evaluate_formula_tool" or "calculate_expression_tool" or "convert_units_tool",
  "formula_name": "btu_hvac" (if using formula),
  "expression": "300 * 25 * 1.2" (if using expression),
  "inputs": {{
    "square_feet": 300,
    "climate_factor": 1.2
  }},
  "from_unit": "sq_ft" (if conversion),
  "to_unit": "sq_m" (if conversion),
  "value": 300 (if conversion)
}}

Return ONLY the JSON object, no markdown, no code blocks."""
            
            # Build messages with conversation history using standardized helper
            system_prompt = "You are a calculation extraction assistant. Extract numerical data and structure calculation requests from queries and documents."
            llm_messages = self._build_conversational_agent_messages(
                system_prompt=system_prompt,
                user_prompt=prompt,
                messages_list=messages,
                look_back_limit=4
            )
            
            try:
                schema = {
                    "type": "object",
                    "properties": {
                        "tool": {"type": "string"},
                        "formula_name": {"type": ["string", "null"]},
                        "expression": {"type": ["string", "null"]},
                        "inputs": {"type": "object"},
                        "from_unit": {"type": ["string", "null"]},
                        "to_unit": {"type": ["string", "null"]},
                        "value": {"type": ["number", "null"]}
                    },
                    "required": ["tool"]
                }
                structured_llm = llm.with_structured_output(schema)
                result = await structured_llm.ainvoke(llm_messages)
                calc_request = result if isinstance(result, dict) else result.dict() if hasattr(result, 'dict') else result.model_dump()
            except Exception:
                # Fallback to manual parsing
                response = await llm.ainvoke(llm_messages)
                content = response.content if hasattr(response, 'content') else str(response)
                calc_request = self._parse_json_response(content) or {}
            
            if not calc_request or "tool" not in calc_request:
                return {
                    "calculation_results": None,
                    "needs_calculations": False,
                    "error": "Could not extract calculation request from query",
                    # ✅ CRITICAL: Preserve state even on error
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", "")
                }
            
            # Call appropriate math tool
            tool_name = calc_request.get("tool")
            calculation_result = None
            
            if tool_name == "evaluate_formula_tool":
                from orchestrator.tools.math_formulas import evaluate_formula_tool
                formula_name = calc_request.get("formula_name")
                inputs = calc_request.get("inputs", {})
                calculation_result = await evaluate_formula_tool(formula_name, inputs)
                
            elif tool_name == "calculate_expression_tool":
                from orchestrator.tools.math_tools import calculate_expression_tool
                expression = calc_request.get("expression")
                variables = calc_request.get("inputs", {})
                calculation_result = await calculate_expression_tool(expression, variables)
                
            elif tool_name == "convert_units_tool":
                from orchestrator.tools.unit_conversions import convert_units_tool
                value = calc_request.get("value")
                from_unit = calc_request.get("from_unit")
                to_unit = calc_request.get("to_unit")
                calculation_result = await convert_units_tool(value, from_unit, to_unit)
                
            else:
                return {
                    "calculation_results": None,
                    "needs_calculations": False,
                    "error": f"Unknown math tool: {tool_name}",
                    # ✅ CRITICAL: Preserve state even on error
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", "")
                }
            
            if calculation_result and calculation_result.get("success"):
                logger.info(f"✅ Calculation completed: {calculation_result.get('result')}")
            else:
                logger.warning(f"⚠️ Calculation failed: {calculation_result.get('error') if calculation_result else 'Unknown error'}")
            
            return {
                "calculation_results": calculation_result,
                "calculation_request": calc_request,
                "needs_calculations": False,  # Calculation completed
                # ✅ CRITICAL: Preserve state for subsequent nodes
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
            
        except Exception as e:
            logger.error(f"❌ Calculation failed: {e}")
            return {
                "calculation_results": None,
                "needs_calculations": False,
                "error": str(e),
                # ✅ CRITICAL: Preserve state even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    def _validate_chart_data_quality(self, chart_type: str, chart_data: Dict[str, Any], query: str) -> bool:
        """
        Validate that chart data has sufficient value for visualization.
        
        Returns True only if:
        - Data has sufficient data points (minimum 3-5 depending on chart type)
        - Data shows meaningful variation or patterns
        - Chart would provide value beyond what text can convey
        - User explicitly requested visualization (overrides strict checks)
        """
        if not chart_data:
            return False
        
        # Check for explicit visualization request in query (overrides strict validation)
        query_lower = query.lower()
        explicit_viz_keywords = ["graph", "chart", "plot", "visualize", "visualization", "show me a", "create a chart", "make a graph"]
        is_explicit_request = any(keyword in query_lower for keyword in explicit_viz_keywords)
        
        # For explicit requests, be more lenient but still check for minimum data
        min_data_points = 2 if is_explicit_request else 3
        
        # Validate based on chart type
        if chart_type in ["bar", "pie"]:
            labels = chart_data.get("labels", [])
            values = chart_data.get("values", [])
            
            if not labels or not values:
                return False
            
            if len(labels) < min_data_points or len(values) < min_data_points:
                logger.debug(f"Bar/pie chart: insufficient data points ({len(labels)} labels, {len(values)} values)")
                return False
            
            # Check for meaningful variation (not all same values)
            if len(set(values)) == 1 and not is_explicit_request:
                logger.debug(f"Bar/pie chart: all values are the same ({values[0]}) - no visualization value")
                return False
            
            return True
        
        elif chart_type in ["line", "scatter", "area"]:
            x = chart_data.get("x", [])
            y = chart_data.get("y", [])
            series = chart_data.get("series", [])
            
            # Check single series format
            if x and y:
                if len(x) < min_data_points or len(y) < min_data_points:
                    logger.debug(f"Line/scatter/area chart: insufficient data points ({len(x)} x, {len(y)} y)")
                    return False
                
                if len(x) != len(y):
                    logger.debug(f"Line/scatter/area chart: mismatched array lengths")
                    return False
                
                # Check for meaningful variation
                if len(set(y)) == 1 and not is_explicit_request:
                    logger.debug(f"Line/scatter/area chart: all y values are the same - no visualization value")
                    return False
                
                return True
            
            # Check multi-series format
            if series and isinstance(series, list):
                if len(series) == 0:
                    return False
                
                # Check if any series has sufficient data
                has_valid_series = False
                for s in series:
                    if isinstance(s, dict):
                        s_x = s.get("x", [])
                        s_y = s.get("y", [])
                        if len(s_x) >= min_data_points and len(s_y) >= min_data_points and len(s_x) == len(s_y):
                            has_valid_series = True
                            break
                
                return has_valid_series
            
            return False
        
        elif chart_type == "heatmap":
            z = chart_data.get("z", [])
            if not z or not isinstance(z, list):
                return False
            
            # Heatmap needs at least 2x2 grid
            if len(z) < 2:
                return False
            
            row_lengths = [len(row) for row in z if isinstance(row, list)]
            if not row_lengths or min(row_lengths) < 2:
                return False
            
            return True
        
        elif chart_type in ["histogram", "box_plot"]:
            values = chart_data.get("values", [])
            if not values or len(values) < min_data_points:
                logger.debug(f"Histogram/box_plot: insufficient data points ({len(values) if values else 0})")
                return False
            
            return True
        
        # Unknown chart type - be conservative
        logger.warning(f"Unknown chart type for validation: {chart_type}")
        return False
    
    async def _perform_visualization_node(self, state: ReferenceAgentState) -> Dict[str, Any]:
        """Perform visualization using chart generation tool"""
        try:
            query = state.get("query", "")
            reference_content = state.get("reference_content", "")
            analysis_results = state.get("analysis_results", {})
            visualization_request = state.get("visualization_request", {})
            chart_type = visualization_request.get("chart_type") if visualization_request else None
            messages = state.get("messages", [])
            
            # Use LLM to extract visualization data from analysis results and document
            fast_model = self._get_fast_model(state)
            llm = self._get_llm(temperature=0.1, model=fast_model, state=state)
            
            # Build prompt to extract visualization data
            prompt = f"""Extract data for visualization from this query, reference document, and analysis results:

**QUERY**: {query}

**CHART TYPE REQUESTED**: {chart_type or "auto-detect"}

**ANALYSIS RESULTS**:
{json.dumps(analysis_results, indent=2)[:2000]}

**REFERENCE DOCUMENT CONTENT** (sample):
{reference_content[:1500]}

**TASK**: Extract data points and structure a visualization request.

**INSTRUCTIONS**:
1. Determine the best chart type if not specified: bar (comparisons), line (trends over time), pie (proportions), scatter (correlations), area (cumulative), heatmap (2D patterns), box_plot (distributions), histogram (frequencies)
2. Extract data from analysis_results:
   - For frequencies: use frequencies dict (labels=keys, values=values)
   - For temporal trends: extract dates/values from patterns metadata
   - For comparisons: use patterns with occurrences
   - For time series: extract dates and corresponding values
3. Structure the visualization request as JSON matching create_chart_tool format

**OUTPUT FORMAT**: Return ONLY valid JSON:
{{
  "chart_type": "bar|line|pie|scatter|area|heatmap|box_plot|histogram",
  "data": {{
    "labels": ["label1", "label2"] (for bar/pie),
    "values": [10, 20] (for bar/pie/histogram),
    "x": [1, 2, 3] (for line/scatter/area),
    "y": [10, 20, 30] (for line/scatter/area),
    "series": [{{"x": [1,2], "y": [10,20], "name": "Series1"}}] (for multi-series line),
    "z": [[1,2],[3,4]] (for heatmap - 2D array)
  }},
  "title": "Chart title",
  "x_label": "X axis label",
  "y_label": "Y axis label",
  "interactive": true
}}

Return ONLY the JSON object, no markdown, no code blocks."""
            
            # Build messages with conversation history using standardized helper
            system_prompt = "You are a data visualization assistant. Extract data from analysis results and structure visualization requests for chart generation."
            llm_messages = self._build_conversational_agent_messages(
                system_prompt=system_prompt,
                user_prompt=prompt,
                messages_list=messages,
                look_back_limit=4
            )
            
            try:
                schema = {
                    "type": "object",
                    "properties": {
                        "chart_type": {"type": "string"},
                        "data": {"type": "object"},
                        "title": {"type": "string"},
                        "x_label": {"type": "string"},
                        "y_label": {"type": "string"},
                        "interactive": {"type": "boolean"}
                    },
                    "required": ["chart_type", "data"]
                }
                structured_llm = llm.with_structured_output(schema)
                result = await structured_llm.ainvoke(llm_messages)
                viz_request = result if isinstance(result, dict) else result.dict() if hasattr(result, 'dict') else result.model_dump()
            except Exception:
                # Fallback to manual parsing
                response = await llm.ainvoke(llm_messages)
                content = response.content if hasattr(response, 'content') else str(response)
                viz_request = self._parse_json_response(content) or {}
            
            if not viz_request or "chart_type" not in viz_request or "data" not in viz_request:
                return {
                    "visualization_results": None,
                    "needs_visualization": False,
                    "error": "Could not extract visualization request from query and analysis",
                    # ✅ CRITICAL: Preserve state even on error
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", "")
                }
            
            # Validate data quality before creating chart
            chart_type = viz_request.get("chart_type")
            chart_data = viz_request.get("data", {})
            
            # Check if data has sufficient value for visualization
            data_is_meaningful = self._validate_chart_data_quality(chart_type, chart_data, query)
            
            if not data_is_meaningful:
                logger.info(f"⚠️ Skipping chart creation: insufficient or low-value data for {chart_type} chart")
                return {
                    "visualization_results": {
                        "success": False,
                        "error": "Insufficient data points or low visualization value. Chart would not provide meaningful insight."
                    },
                    "needs_visualization": False,
                    "error": "Data quality validation failed - chart would not provide meaningful value",
                    # ✅ CRITICAL: Preserve state even on validation failure
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", "")
                }
            
            # Call visualization tool
            from orchestrator.tools.visualization_tools import create_chart_tool
            
            chart_title = viz_request.get("title", "")
            x_label = viz_request.get("x_label", "")
            y_label = viz_request.get("y_label", "")
            interactive = viz_request.get("interactive", True)
            
            visualization_result = await create_chart_tool(
                chart_type=chart_type,
                data=chart_data,
                title=chart_title,
                x_label=x_label,
                y_label=y_label,
                interactive=interactive,
                include_static=True
            )
            
            if visualization_result and visualization_result.get("success"):
                logger.info(f"✅ Visualization created successfully: {chart_type}, format: {visualization_result.get('output_format')}")
            else:
                logger.warning(f"⚠️ Visualization failed: {visualization_result.get('error') if visualization_result else 'Unknown error'}")
            
            return {
                "visualization_results": visualization_result,
                "visualization_request": viz_request,
                "static_visualization_data": visualization_result.get("static_svg") if visualization_result else None,
                "static_format": visualization_result.get("static_format") if visualization_result else None,
                "needs_visualization": False,  # Visualization completed
                # ✅ CRITICAL: Preserve state for subsequent nodes
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
            
        except Exception as e:
            logger.error(f"❌ Visualization failed: {e}")
            return {
                "visualization_results": None,
                "needs_visualization": False,
                "error": str(e),
                # ✅ CRITICAL: Preserve state even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    async def _get_weather_data_node(self, state: ReferenceAgentState) -> Dict[str, Any]:
        """Get weather data using Tools Service via gRPC"""
        try:
            weather_request = state.get("weather_request")
            user_id = state.get("user_id", "system")
            
            if not weather_request:
                logger.warning("⚠️ Weather request not provided")
                return {
                    "weather_results": None,
                    "needs_weather": False,
                    # ✅ CRITICAL: Preserve state even on early return
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", "")
                }
            
            location = weather_request.get("location", "")
            request_type = weather_request.get("request_type", "current")
            date_str = weather_request.get("date_str")
            
            if not location:
                # Try to extract location from query or document context
                query = state.get("query", "")
                # Simple location extraction - could be enhanced with LLM
                # For now, default to a common location or ask user
                logger.warning("⚠️ No location specified in weather request")
                return {
                    "weather_results": {
                        "success": False,
                        "error": "Location is required for weather queries. Please specify a location (e.g., 'What's the weather in Los Angeles?')."
                    },
                    "needs_weather": False,
                    # ✅ CRITICAL: Preserve state even on error
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", "")
                }
            
            logger.info(f"🌤️ Getting weather data: location={location}, type={request_type}, date_str={date_str}")
            
            # Get gRPC client from BaseAgent
            grpc_client = await self._get_grpc_client()
            
            # Determine data types based on request type
            data_types = []
            if request_type == "current":
                data_types = ["current"]
            elif request_type == "forecast":
                data_types = ["forecast"]
            elif request_type == "history":
                data_types = ["history"]
            else:
                # Default to current if unknown
                data_types = ["current"]
            
            # Call weather tool via gRPC
            weather_result = await grpc_client.get_weather(
                location=location,
                user_id=user_id,
                data_types=data_types,
                date_str=date_str
            )
            
            if weather_result and weather_result.get("success"):
                logger.info(f"✅ Weather data retrieved successfully for {location}")
            else:
                error_msg = weather_result.get("error", "Unknown error") if weather_result else "Weather service unavailable"
                logger.warning(f"⚠️ Weather fetch failed: {error_msg}")
            
            return {
                "weather_results": weather_result,
                "needs_weather": False,  # Weather completed
                # ✅ CRITICAL: Preserve state for subsequent nodes
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
            
        except Exception as e:
            logger.error(f"❌ Weather data fetch failed: {e}")
            return {
                "weather_results": {
                    "success": False,
                    "error": f"Weather service error: {str(e)}"
                },
                "needs_weather": False,
                # ✅ CRITICAL: Preserve state even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    async def _call_research_subgraph_node(self, state: ReferenceAgentState) -> Dict[str, Any]:
        """Call research agent as subgraph for external information"""
        try:
            from orchestrator.agents import get_full_research_agent
            
            research_agent = get_full_research_agent()
            research_query = state.get("research_query")
            
            if not research_query:
                logger.warning("⚠️ Research query not provided")
                return {
                    "research_results": None,
                    "needs_external_info": False,
                    # ✅ CRITICAL: Preserve state even on early return
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", "")
                }
            
            logger.info(f"📚 Calling research agent for: {research_query}")
            
            # Prepare metadata for research agent
            metadata = state.get("metadata", {})
            
            # Extract reference document context to provide to research agent
            reference_content = state.get("reference_content", "")
            referenced_files_content = state.get("referenced_files_content", {})
            
            # Build handoff context using shared_memory (clean inter-agent data passing)
            handoff_shared_memory = {
                "context_note": "Research for reference document analysis",
                "user_chat_model": metadata.get("user_chat_model"),
                "handoff_context": {
                    "source_agent": "reference_agent",
                    "handoff_type": "research_delegation",
                    "reference_document": {
                        "content": reference_content,
                        "type": state.get("reference_type", "reference"),
                        "filename": state.get("shared_memory", {}).get("active_editor", {}).get("filename", "unknown"),
                        "has_content": bool(reference_content)
                    },
                    "referenced_files": referenced_files_content,
                    "analysis_context": {
                        "needs_calculations": state.get("needs_calculations", False),
                        "calculation_results": state.get("calculation_results"),
                        "complexity": state.get("complexity", "unknown")
                    }
                }
            }
            
            research_metadata = {
                "user_id": state.get("user_id"),
                "conversation_id": metadata.get("conversation_id"),
                "persona": metadata.get("persona"),
                "user_chat_model": metadata.get("user_chat_model"),
                "shared_memory": handoff_shared_memory
            }
            
            # Augment query to inform research agent about available context
            augmented_query = research_query
            if reference_content:
                # Brief note in query - full data is in shared_memory
                augmented_query = f"""{research_query}

[Context: User has reference document available in shared_memory with relevant data. You can reference this document when providing supplemental information.]"""
            
            # Call research agent's process method
            research_result = await research_agent.process(
                query=augmented_query,
                metadata=research_metadata,
                messages=state.get("messages", [])
            )
            
            logger.info(f"✅ Research agent completed")
            
            return {
                "research_results": research_result,
                "needs_external_info": False,  # Research completed
                # ✅ CRITICAL: Preserve calculation needs - calculations may still be needed after research
                "needs_calculations": state.get("needs_calculations", False),
                "calculation_type": state.get("calculation_type"),
                "calculation_request": state.get("calculation_request"),
                # ✅ CRITICAL: Preserve state for subsequent nodes
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
            
        except Exception as e:
            logger.error(f"❌ Research subgraph failed: {e}")
            return {
                "research_results": None,
                "needs_external_info": False,
                "error": f"Research failed: {str(e)}",
                # ✅ CRITICAL: Preserve state even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    async def _synthesize_response_node(self, state: ReferenceAgentState) -> Dict[str, Any]:
        """Synthesize analysis results with calculations, visualization, weather, and research (if any)"""
        try:
            analysis_results = state.get("analysis_results", {})
            calculation_results = state.get("calculation_results")
            visualization_results = state.get("visualization_results")
            weather_results = state.get("weather_results")
            research_results = state.get("research_results")
            query = state.get("query", "")
            query_complexity = state.get("query_complexity", "simple_qa")
            
            # Start with analysis response
            response_text = analysis_results.get("response", "")
            
            # If calculations were performed, integrate them
            if calculation_results and calculation_results.get("success"):
                calc_result = calculation_results.get("result")
                calc_steps = calculation_results.get("steps", [])
                calc_unit = calculation_results.get("unit", "")
                
                # Format calculation results
                calc_section = f"\n\n**Calculation Results**:\n"
                if calc_steps:
                    calc_section += "\n".join([f"- {step}" for step in calc_steps])
                else:
                    calc_section += f"Result: {calc_result}"
                    if calc_unit:
                        calc_section += f" {calc_unit}"
                
                response_text = f"""{response_text}{calc_section}"""
                
                logger.info("📚 Integrated calculation results into response")
            
            # If visualization was generated, integrate it
            if visualization_results and visualization_results.get("success"):
                chart_data = visualization_results.get("chart_data", "")
                output_format = visualization_results.get("output_format", "")
                chart_type = visualization_results.get("chart_type", "")
                
                # Add visualization to response
                viz_section = f"\n\n**Visualization**:\n"
                
                if output_format == "html":
                    # Interactive HTML chart - use special markdown code block format
                    # Frontend will detect language "html:chart" and render as HTML
                    viz_section += f"\n```html:chart\n{chart_data}\n```\n"
                elif output_format == "base64_png":
                    # Base64 image - embed as image
                    viz_section += f"\n![Chart: {chart_type}]({chart_data})\n"
                else:
                    viz_section += f"Chart generated: {chart_type} ({output_format})\n"
                
                response_text = f"""{response_text}{viz_section}"""
                
                logger.info(f"📚 Integrated visualization into response: {chart_type}, format: {output_format}")
            
            # If weather data was retrieved, integrate it
            if weather_results and weather_results.get("success"):
                weather_data = weather_results.get("current") or weather_results.get("forecast") or weather_results.get("historical")
                location_name = weather_results.get("location", {}).get("name", "Unknown")
                
                if weather_results.get("current"):
                    # Current weather
                    current = weather_results["current"]
                    temp = current.get("temperature", "N/A")
                    condition = current.get("condition", "N/A")
                    humidity = current.get("humidity", "N/A")
                    wind_speed = current.get("wind_speed", "N/A")
                    
                    weather_section = f"\n\n**Current Weather for {location_name}**:\n"
                    weather_section += f"- Temperature: {temp}\n"
                    weather_section += f"- Condition: {condition}\n"
                    weather_section += f"- Humidity: {humidity}\n"
                    weather_section += f"- Wind Speed: {wind_speed}\n"
                    
                    response_text = f"""{response_text}{weather_section}"""
                    logger.info("📚 Integrated current weather into response")
                    
                elif weather_results.get("forecast"):
                    # Forecast weather
                    forecast = weather_results["forecast"]
                    forecast_days = forecast.get("days", [])
                    
                    weather_section = f"\n\n**Weather Forecast for {location_name}**:\n"
                    for day in forecast_days[:5]:  # Show next 5 days
                        date = day.get("date", "Unknown")
                        temp = day.get("temperature", "N/A")
                        condition = day.get("condition", "N/A")
                        weather_section += f"- {date}: {temp}, {condition}\n"
                    
                    response_text = f"""{response_text}{weather_section}"""
                    logger.info("📚 Integrated weather forecast into response")
                    
                elif weather_results.get("historical"):
                    # Historical weather
                    historical = weather_results["historical"]
                    date_str = weather_results.get("period", {}).get("date_str", "Unknown date")
                    temp = historical.get("temperature", "N/A")
                    condition = historical.get("condition", "N/A")
                    
                    weather_section = f"\n\n**Historical Weather for {location_name} on {date_str}**:\n"
                    weather_section += f"- Temperature: {temp}\n"
                    weather_section += f"- Condition: {condition}\n"
                    
                    response_text = f"""{response_text}{weather_section}"""
                    logger.info(f"📚 Integrated historical weather into response for {date_str}")
                else:
                    # Generic weather data
                    weather_section = f"\n\n**Weather Information for {location_name}**:\n"
                    weather_section += f"{json.dumps(weather_data, indent=2)}\n"
                    response_text = f"""{response_text}{weather_section}"""
                    logger.info("📚 Integrated weather data into response")
            
            elif weather_results and not weather_results.get("success"):
                # Weather request failed
                error_msg = weather_results.get("error", "Unknown error")
                weather_section = f"\n\n**Weather Information**: Unable to retrieve weather data: {error_msg}"
                response_text = f"""{response_text}{weather_section}"""
                logger.warning(f"⚠️ Weather request failed: {error_msg}")
            
            # If research was used, integrate it
            if research_results:
                research_response = research_results.get("response", "")
                if isinstance(research_response, dict):
                    research_response = research_response.get("response", str(research_response))
                
                # Combine analysis with research
                response_text = f"""{response_text}

**Additional Information from Research**:
{research_response}"""
                
                logger.info("📚 Integrated research results into response")
            
            # Build structured response
            response = {
                "response": response_text,
                "task_status": "complete",
                "complexity_level": query_complexity,
                "patterns_found": state.get("patterns_found"),
                "insights": analysis_results.get("insights"),
                "calculations_used": calculation_results is not None and calculation_results.get("success", False),
                "calculation_result": calculation_results.get("result") if calculation_results else None,
                "visualization_used": visualization_results is not None and visualization_results.get("success", False),
                "visualization_data": visualization_results if visualization_results and visualization_results.get("success") else None,
                "static_visualization_data": visualization_results.get("static_svg") if visualization_results and visualization_results.get("success") else None,
                "static_format": visualization_results.get("static_format") if visualization_results and visualization_results.get("success") else None,
                "weather_used": weather_results is not None and weather_results.get("success", False),
                "weather_data": weather_results if weather_results and weather_results.get("success") else None,
                "research_used": research_results is not None,
                "research_citations": research_results.get("citations", []) if research_results else None,
                "confidence": 0.85,
                "sources": ["reference_document"]
            }
            
            return {
                "response": response,
                "static_visualization_data": visualization_results.get("static_svg") if visualization_results and visualization_results.get("success") else None,
                "static_format": visualization_results.get("static_format") if visualization_results and visualization_results.get("success") else None,
                # ✅ CRITICAL: Preserve state for subsequent nodes
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
            
        except Exception as e:
            logger.error(f"❌ Response synthesis failed: {e}")
            return {
                "response": {
                    "response": f"Error synthesizing response: {str(e)}",
                    "task_status": "error"
                },
                "error": str(e),
                # ✅ CRITICAL: Preserve state even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    async def _format_response_node(self, state: ReferenceAgentState) -> Dict[str, Any]:
        """Format final response and add to conversation history"""
        try:
            response = state.get("response", {})
            
            # Add assistant response to messages for checkpoint persistence
            response_text = response.get("response", "")
            if response_text:
                state = self._add_assistant_response_to_messages(state, str(response_text))
            
            # Clear request-scoped data (active_editor) before checkpoint save
            # This ensures it's available during the request (for subgraphs) but doesn't persist
            state = self._clear_request_scoped_data(state)
            
            return {
                "response": response,
                "task_status": response.get("task_status", "complete"),
                "static_visualization_data": response.get("static_visualization_data"),
                "static_format": response.get("static_format"),
                "messages": state.get("messages", []),
                "shared_memory": state.get("shared_memory", {}),
                # ✅ CRITICAL: Preserve state (final node, but good practice)
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "query": state.get("query", "")
            }
            
        except Exception as e:
            logger.error(f"❌ Response formatting failed: {e}")
            return {
                "response": {
                    "response": f"Error formatting response: {str(e)}",
                    "task_status": "error"
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
    
    def _build_reference_prompt(self) -> str:
        """Build system prompt for reference analysis"""
        return """You are a Reference Analysis Agent - an expert at analyzing personal records and logs.

**MISSION**: Help users understand and interact with their personal reference documents:
- Journals and daily logs
- Weight loss and food records
- Meeting notes and research logs
- Any personal tracking documents

**YOUR CAPABILITIES**:
1. **Simple Q&A**: Answer factual questions about document content
2. **Pattern Analysis**: Find trends, frequencies, correlations
3. **Insights**: Provide observations and patterns (when requested)

**DOCUMENT CONTEXT**:
- You have access to the main reference document PLUS any referenced files
- Documents marked "(MAIN REFERENCE)" are the primary document
- Documents marked "(REFERENCED FILE)" provide additional context
- Understand how these documents relate to each other

**IMPORTANT RULES**:
- READ-ONLY: You cannot edit or update reference documents
- NON-JUDGMENTAL: Be supportive and analytical, not prescriptive
- FACTUAL: Base responses on actual document content
- MATH TOOL INTEGRATION: When calculations are needed (BTU, electrical, etc.), the system will automatically use the math tool for accurate results
- WEATHER TOOL INTEGRATION: You have access to weather tools via the Tools Service. You can retrieve current weather conditions, forecasts, and historical weather data for any location. The system will automatically detect weather-related queries and fetch the data. Use weather tools when:
  * User asks about current weather conditions
  * User requests weather forecasts
  * User asks about historical weather (specific dates or months)
  * Weather information would enhance analysis of reference documents (e.g., correlating journal entries with weather conditions)
- RESEARCH INTEGRATION: When you need external information, the system will automatically call the research agent
- VISUALIZATION TOOL: You have access to a chart generation tool for visualizing data patterns, trends, and distributions. The system will automatically detect when visualizations would be helpful and generate them. Use visualizations when:
  * Showing trends over time (weight, mood, food intake, etc.)
  * Comparing values across categories or time periods
  * Displaying distributions or frequency patterns
  * User explicitly requests a chart, graph, or visualization
  Available chart types: bar (comparisons), line (trends over time), pie (proportions), scatter (correlations), area (cumulative trends), heatmap (2D patterns), box_plot (distributions), histogram (frequency distributions)
- CLEAR COMMUNICATION: Be clear about what information comes from the document vs. calculations vs. weather data vs. visualizations vs. external research

**RESPONSE STYLE**:
- Supportive and understanding tone
- Analytical and objective
- Focus on what the data shows, not what the user "should" do
- Provide insights when requested, but don't be prescriptive
- Use specific examples from the document to support your observations

**STRUCTURED OUTPUT**:
When providing pattern analysis or insights, use structured JSON with:
- Patterns: type, description, occurrences, examples
- Trends: temporal patterns and frequencies
- Insights: observations with confidence levels and evidence
- Recommendations: only when explicitly requested or clearly appropriate"""
    
    async def process(self, query: str, metadata: Dict[str, Any] = None, messages: List[Any] = None) -> Dict[str, Any]:
        """Process reference query using LangGraph workflow"""
        try:
            logger.info(f"📚 Reference agent processing: {query[:100]}...")
            
            # Extract user_id from metadata
            metadata = metadata or {}
            user_id = metadata.get("user_id", "system")
            
            # Get workflow (lazy initialization with checkpointer)
            workflow = await self._get_workflow()
            
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
            initial_state: ReferenceAgentState = {
                "query": query,
                "user_id": user_id,
                "metadata": metadata,
                "messages": conversation_messages,
                "shared_memory": shared_memory,
                "reference_content": "",
                "referenced_files": {},
                "document_type": "reference",
                "query_complexity": "simple_qa",
                "analysis_results": {},
                "patterns_found": [],
                "needs_external_info": False,
                "research_query": None,
                "research_results": None,
                "needs_weather": False,
                "weather_request": None,
                "weather_results": None,
                "needs_calculations": False,
                "calculation_type": None,
                "calculation_request": None,
                "calculation_results": None,
                "needs_visualization": False,
                "visualization_request": None,
                "visualization_results": None,
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
                logger.error(f"❌ Reference agent failed: {error_msg}")
                return self._create_error_response(error_msg)
            
            logger.info(f"✅ Reference agent completed: {task_status}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Reference agent failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._create_error_response(str(e))


# Factory function for lazy loading
_reference_agent_instance = None

def get_reference_agent() -> ReferenceAgent:
    """Get or create reference agent instance"""
    global _reference_agent_instance
    if _reference_agent_instance is None:
        _reference_agent_instance = ReferenceAgent()
    return _reference_agent_instance

