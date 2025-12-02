"""
Report Formatting Agent - LangGraph Implementation
Converts research findings into structured reports.
Works with research agent outputs from shared_memory.
"""

import logging
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from .base_agent import BaseAgent, TaskStatus

logger = logging.getLogger(__name__)


# ============================================
# State Definition
# ============================================

class ReportFormattingState(TypedDict):
    """State for report formatting agent LangGraph workflow"""
    query: str
    user_id: str
    metadata: Dict[str, Any]
    messages: List[Any]
    shared_memory: Dict[str, Any]
    research_findings: Dict[str, Any]
    web_search_results: Dict[str, Any]
    document_analyses: Dict[str, Any]
    citations: List[str]
    formatted_report: str
    response: Dict[str, Any]
    task_status: str
    error: str


# ============================================
# Report Formatting Agent
# ============================================

class ReportFormattingAgent(BaseAgent):
    """
    Agent for formatting research findings into structured reports.
    
    Handles:
    - Extracting research findings from shared_memory
    - Formatting into structured reports
    - Citation management
    - Professional report structure
    Uses LangGraph workflow for explicit state management
    """
    
    def __init__(self):
        super().__init__("report_formatting_agent")
        logger.info("Report Formatting Agent initialized")
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """Build LangGraph workflow for report formatting agent"""
        workflow = StateGraph(ReportFormattingState)
        
        # Add nodes
        workflow.add_node("extract_research_data", self._extract_research_data_node)
        workflow.add_node("format_report", self._format_report_node)
        workflow.add_node("format_response", self._format_response_node)
        
        # Entry point
        workflow.set_entry_point("extract_research_data")
        
        # Linear flow: extract_research_data -> format_report -> format_response -> END
        workflow.add_edge("extract_research_data", "format_report")
        workflow.add_edge("format_report", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow.compile(checkpointer=checkpointer)
    
    async def _extract_research_data_node(self, state: ReportFormattingState) -> Dict[str, Any]:
        """Extract research findings from shared_memory"""
        try:
            logger.info("Extracting research data from shared_memory...")
            
            shared_memory = state.get("shared_memory", {}) or {}
            
            # Extract research findings
            research_findings = shared_memory.get("research_findings", {})
            web_search_results = shared_memory.get("web_search_results", {})
            document_analyses = shared_memory.get("document_analyses", {})
            
            # Extract citations from various sources
            citations = []
            
            # From agent_results
            agent_results = state.get("metadata", {}).get("agent_results", {})
            if isinstance(agent_results, dict) and "citations" in agent_results:
                citations.extend(agent_results["citations"])
            
            # From shared_memory research findings
            for category in ["research_findings", "web_search_results", "document_analyses"]:
                category_data = shared_memory.get(category, {})
                if isinstance(category_data, dict):
                    for item_key, item_data in category_data.items():
                        if isinstance(item_data, dict) and "citations" in item_data:
                            if isinstance(item_data["citations"], list):
                                citations.extend(item_data["citations"])
            
            # Remove duplicates while preserving order
            unique_citations = []
            seen = set()
            for citation in citations:
                if citation not in seen:
                    unique_citations.append(citation)
                    seen.add(citation)
            
            logger.info(f"Extracted {len(research_findings)} research findings, {len(web_search_results)} web results, {len(document_analyses)} document analyses, {len(unique_citations)} citations")
            
            return {
                "research_findings": research_findings,
                "web_search_results": web_search_results,
                "document_analyses": document_analyses,
                "citations": unique_citations
            }
            
        except Exception as e:
            logger.error(f"Error extracting research data: {e}")
            return {
                "research_findings": {},
                "web_search_results": {},
                "document_analyses": {},
                "citations": [],
                "error": str(e)
            }
    
    async def _format_report_node(self, state: ReportFormattingState) -> Dict[str, Any]:
        """Format research data into structured report"""
        try:
            logger.info("Formatting research data into structured report...")
            
            research_findings = state.get("research_findings", {})
            web_search_results = state.get("web_search_results", {})
            document_analyses = state.get("document_analyses", {})
            citations = state.get("citations", [])
            query = state.get("query", "Research Results")
            
            # Build comprehensive data context
            all_research_data = {
                "local_findings": research_findings,
                "web_findings": web_search_results,
                "document_findings": document_analyses,
                "citations": citations,
                "metadata": {
                    "query": query,
                    "timestamp": datetime.now().isoformat(),
                    "data_sources": self._identify_data_sources(research_findings, web_search_results, document_analyses)
                }
            }
            
            # Build formatting prompt
            system_prompt = (
                "You are a professional intelligence analyst formatting a structured report. "
                "Convert research findings into a well-organized, comprehensive report with clear sections, "
                "proper formatting, and professional presentation."
            )
            
            user_prompt = self._build_formatting_prompt(all_research_data, query)
            
            # Call LLM using centralized mechanism
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            llm = self._get_llm(temperature=0.3, state=state)
            llm_response = await llm.ainvoke(messages)
            formatted_report = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            
            # Ensure report has proper structure
            if not formatted_report.strip().startswith('#'):
                # Add header if missing
                formatted_report = f"# Research Report\n\n{formatted_report}"
            
            logger.info(f"Formatted report: {len(formatted_report)} characters")
            
            return {
                "formatted_report": formatted_report.strip()
            }
            
        except Exception as e:
            logger.error(f"Error formatting report: {e}")
            # Fallback to simple formatting
            return {
                "formatted_report": self._format_general_report(state)
            }
    
    async def _format_response_node(self, state: ReportFormattingState) -> Dict[str, Any]:
        """Format final response"""
        try:
            logger.info("Formatting report formatting response...")
            
            formatted_report = state.get("formatted_report", "")
            query = state.get("query", "Research Results")
            
            if not formatted_report:
                formatted_report = f"# Research Results\n\nQuery: {query}\n\nNo research data available to format."
            
            # Update shared_memory with formatted report
            shared_memory = state.get("shared_memory", {}) or {}
            shared_memory.setdefault("formatted_reports", {})["general"] = {
                "report_content": formatted_report,
                "template_used": "general_format",
                "agent": "report_formatting_agent",
                "timestamp": datetime.now().isoformat(),
                "sections_completed": formatted_report.count("##"),
                "source_data": {
                    "research_findings_count": len(state.get("research_findings", {})),
                    "web_results_count": len(state.get("web_search_results", {})),
                    "document_analyses_count": len(state.get("document_analyses", {}))
                }
            }
            
            return {
                "response": {
                    "task_status": TaskStatus.COMPLETE.value,
                    "response": formatted_report,
                    "formatted_report": formatted_report,
                    "citations": state.get("citations", [])
                },
                "shared_memory": shared_memory,
                "task_status": "complete"
            }
            
        except Exception as e:
            logger.error(f"Error formatting response: {e}")
            return {
                "response": {
                    "task_status": TaskStatus.ERROR.value,
                    "response": f"Error formatting response: {str(e)}",
                    "error": str(e)
                },
                "task_status": "error",
                "error": str(e)
            }
    
    def _build_formatting_prompt(self, research_data: Dict[str, Any], query: str) -> str:
        """Build prompt for formatting research data into report"""
        
        prompt = f"""Format the following research data into a comprehensive, well-structured report.

**USER QUERY**: {query}

**RESEARCH DATA**:
{json.dumps(research_data, indent=2, default=str)}

**REPORT FORMATTING REQUIREMENTS**:

1. **STRUCTURE**: Use clear section headers (## for main sections, ### for subsections)
2. **ORGANIZATION**: Organize information logically (Summary, Findings, Analysis, Conclusions, Sources)
3. **CLARITY**: Use professional, clear language
4. **COMPLETENESS**: Include all relevant information from the research data
5. **CITATIONS**: Include citations section at the end with all sources
6. **FORMATTING**: Use proper markdown formatting (headers, lists, emphasis)
7. **ACCURACY**: Only include information that was actually provided in the research data
8. **OBJECTIVITY**: Maintain objective, analytical tone

**REPORT SECTIONS** (include as appropriate):
- Executive Summary
- Key Findings
- Detailed Analysis
- Supporting Evidence
- Conclusions
- Sources and Citations

Generate a comprehensive report following these guidelines."""
        
        return prompt
    
    def _format_general_report(self, state: ReportFormattingState) -> str:
        """Fallback general report formatting"""
        try:
            research_findings = state.get("research_findings", {})
            query = state.get("query", "Research Results")
            citations = state.get("citations", [])
            
            report_sections = []
            
            # Summary section
            report_sections.append("# Research Report")
            report_sections.append(f"**Query**: {query}")
            report_sections.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_sections.append("")
            
            # Findings sections
            if research_findings:
                report_sections.append("## Research Findings")
                report_sections.append("")
                
                for key, findings in research_findings.items():
                    if isinstance(findings, dict) and findings.get("data"):
                        report_sections.append(f"### {key.replace('_', ' ').title()}")
                        report_sections.append(str(findings["data"]))
                        report_sections.append("")
            
            # Citations if available
            if citations:
                report_sections.append("## Sources and Citations")
                report_sections.append("")
                for i, citation in enumerate(citations, 1):
                    report_sections.append(f"{i}. {citation}")
                report_sections.append("")
            
            return "\n".join(report_sections)
            
        except Exception as e:
            logger.error(f"Error in general report formatting: {e}")
            return f"# Research Results\n\nQuery: {state.get('query', 'Unknown')}\n\nError formatting results: {str(e)}"
    
    def _identify_data_sources(self, research_findings: Dict, web_results: Dict, document_results: Dict) -> List[str]:
        """Identify data sources used in research"""
        sources = []
        
        if research_findings:
            sources.append("Local Knowledge Base")
        if web_results:
            sources.append("Web Search")
        if document_results:
            sources.append("Document Analysis")
        
        return sources


# ============================================
# Factory Functions
# ============================================

_report_formatting_agent_instance: Optional[ReportFormattingAgent] = None


def get_report_formatting_agent() -> ReportFormattingAgent:
    """Get or create singleton report formatting agent instance"""
    global _report_formatting_agent_instance
    if _report_formatting_agent_instance is None:
        _report_formatting_agent_instance = ReportFormattingAgent()
    return _report_formatting_agent_instance

