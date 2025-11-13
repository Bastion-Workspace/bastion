"""
Report Formatting Agent - Roosevelt's "Intelligence Synthesis Specialist"

This agent specializes in converting raw research findings into structured reports
using templates while following LangGraph best practices and shared memory architecture.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

from services.langgraph_agents.base_agent import BaseAgent
from models.shared_memory_models import SharedMemory, validate_shared_memory
from models.report_template_models import ReportTemplate, ReportTemplateSection
from services.template_service import template_service
from utils.system_prompt_utils import get_current_datetime_context

logger = logging.getLogger(__name__)


class ReportFormattingAgent(BaseAgent):
    """
    Specialized agent for converting research data into structured reports
    
    Follows Roosevelt's Intelligence Sharing Architecture:
    - Receives complete shared_memory from research agents
    - Builds upon research findings without repeating work
    - Stores formatted reports in structured shared memory
    - Respects user preferences and conversation context
    """
    
    def __init__(self):
        super().__init__("report_formatting_agent")
        logger.info("📋 Report Formatting Agent initialized - template synthesis specialist")
    
    async def process(self, state: Dict[str, Any]) -> str:
        """
        Process research findings into structured template reports
        
        Following AGENT_INTEGRATION_GUIDE.md:
        - Extract shared memory for full context awareness
        - Access previous agent results (research findings)
        - Update shared memory with formatted reports
        - Respect user preferences and conversation context
        """
        try:
            logger.info("📋 Report Formatting Agent processing...")
            
            # Acknowledge handoff from research agent
            handoff_message = state.get("research_complete_message")
            if handoff_message:
                logger.info(f"🤝 Received from Research Agent: {handoff_message}")
            
            # Extract shared memory - ROOSEVELT'S INTELLIGENCE SHARING
            shared_memory = state.get("shared_memory", {})
            
            # Access previous agent results
            research_findings = shared_memory.get("research_findings", {})
            web_search_results = shared_memory.get("web_search_results", {})
            document_analyses = shared_memory.get("document_analyses", {})
            user_preferences = shared_memory.get("user_preferences", {})
            conversation_context = shared_memory.get("conversation_context", {})
            
            logger.info(f"📊 Found {len(research_findings)} research findings to process")
            
            # Get template information
            template_id = state.get("template_id")
            template_modifications = state.get("template_modifications")
            
            if not template_id:
                logger.warning("⚠️ No template ID provided, falling back to general formatting")
                return await self._format_general_report(state, shared_memory)
            
            # Load template
            template = await template_service.get_template(template_id)
            if not template:
                logger.error(f"❌ Template not found: {template_id}")
                return await self._format_general_report(state, shared_memory)
            
            logger.info(f"📋 Using template: {template.template_name}")
            
            # Notify user of formatting process
            formatting_message = f"📋 **Report Formatting Agent** is now structuring your research data using the **{template.template_name}** template..."
            logger.info(formatting_message)
            
            # Store progress message for user visibility
            state["formatting_progress"] = formatting_message
            
            # Format research data into template structure
            formatted_report = await self._format_with_template(
                research_findings=research_findings,
                web_results=web_search_results,
                document_results=document_analyses,
                template=template,
                user_preferences=user_preferences,
                modifications=template_modifications,
                state=state
            )
            
            # Update shared memory with formatted report - ROOSEVELT'S NEW APPROACH
            shared_memory.setdefault("formatted_reports", {})[template_id] = {
                "report_content": formatted_report,
                "template_used": template.template_name,
                "agent": self.agent_type,
                "timestamp": datetime.now().isoformat(),
                "sections_completed": await self._count_completed_sections(formatted_report),
                "source_data": {
                    "research_findings_count": len(research_findings),
                    "web_results_count": len(web_search_results),
                    "document_analyses_count": len(document_analyses)
                }
            }
            
            # Update conversation context
            conversation_context.setdefault("reports_generated", []).append({
                "template_id": template_id,
                "template_name": template.template_name,
                "timestamp": datetime.now().isoformat()
            })
            
            # Track agent handoff
            shared_memory.setdefault("agent_handoffs", []).append({
                "from_agent": "research_agent",
                "to_agent": self.agent_type,
                "purpose": f"Format research into {template.template_name}",
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"✅ Successfully formatted report using {template.template_name}")
            
            return formatted_report
            
        except Exception as e:
            logger.error(f"❌ Report formatting failed: {e}")
            
            # Log failed operation to shared memory - BEST PRACTICE
            shared_memory.setdefault("failed_operations", []).append({
                "agent": self.agent_type,
                "operation": "template_formatting",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "template_id": state.get("template_id", "unknown")
            })
            
            # Graceful fallback to general formatting
            return await self._format_general_report(state, shared_memory)
    
    async def _format_with_template(
        self,
        research_findings: Dict[str, Any],
        web_results: Dict[str, Any],
        document_results: Dict[str, Any],
        template: ReportTemplate,
        user_preferences: Dict[str, Any],
        modifications: Optional[str],
        state: Dict[str, Any]
    ) -> str:
        """Format research data according to template structure"""
        try:
            # Build comprehensive data context
            all_research_data = {
                "local_findings": research_findings,
                "web_findings": web_results,
                "document_findings": document_results,
                "citations": self._extract_all_citations(state),
                "metadata": {
                    "query": state.get("current_query", ""),
                    "timestamp": datetime.now().isoformat(),
                    "data_sources": self._identify_data_sources(research_findings, web_results, document_results)
                }
            }
            
            # Create template formatting prompt
            system_prompt = await self._build_template_formatting_prompt(
                template, all_research_data, user_preferences, modifications
            )
            
            # Get system context with current date/time
            system_context = get_current_datetime_context()
            
            # Call LLM for template formatting
            from services.chat_service import ChatService
            chat_service = ChatService()
            
            formatting_prompt = f"""
{system_context}

{system_prompt}

RESEARCH DATA TO FORMAT:
{json.dumps(all_research_data, indent=2, default=str)}

USER QUERY: {state.get('current_query', '')}

Generate a comprehensive, well-structured report following the template exactly.
Fill in all sections where data is available. Mark sections as "Data not available" where research didn't yield results.
Ensure proper formatting, clear section headers, and professional presentation.
"""
            
            # Generate formatted report
            response = await chat_service.get_llm_response(
                user_message=formatting_prompt,
                system_prompt="You are a professional report formatting specialist.",
                model_override=None
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"❌ Template formatting error: {e}")
            return await self._format_general_report(state, {"research_findings": research_findings})
    
    async def _format_general_report(self, state: Dict[str, Any], shared_memory: Dict[str, Any]) -> str:
        """Fallback general report formatting when templates aren't available"""
        try:
            logger.info("📝 Using general report formatting")
            
            research_findings = shared_memory.get("research_findings", {})
            query = state.get("current_query", "Research Results")
            
            # Simple structured formatting
            report_sections = []
            
            # Summary section
            if research_findings:
                report_sections.append("## Research Summary")
                report_sections.append(f"Research conducted for: {query}")
                report_sections.append("")
            
            # Findings sections
            for key, findings in research_findings.items():
                if isinstance(findings, dict) and findings.get("data"):
                    report_sections.append(f"## {key.replace('_', ' ').title()}")
                    report_sections.append(str(findings["data"]))
                    report_sections.append("")
            
            # Citations if available
            citations = self._extract_all_citations(state)
            if citations:
                report_sections.append("## Sources")
                for i, citation in enumerate(citations, 1):
                    report_sections.append(f"{i}. {citation}")
                report_sections.append("")
            
            formatted_report = "\n".join(report_sections)
            
            # Update shared memory for general report
            shared_memory.setdefault("formatted_reports", {})["general"] = {
                "report_content": formatted_report,
                "template_used": "general_format",
                "agent": self.agent_type,
                "timestamp": datetime.now().isoformat()
            }
            
            return formatted_report
            
        except Exception as e:
            logger.error(f"❌ General report formatting failed: {e}")
            return f"# Research Results\n\nQuery: {state.get('current_query', 'Unknown')}\n\nError formatting results: {str(e)}"
    
    async def _build_template_formatting_prompt(
        self,
        template: ReportTemplate,
        research_data: Dict[str, Any],
        user_preferences: Dict[str, Any],
        modifications: Optional[str]
    ) -> str:
        """Build LLM prompt for template-based formatting"""
        
        # Extract template structure
        template_structure = []
        for section in template.sections:
            section_info = f"## {section.section_name}"
            if section.description:
                section_info += f"\n{section.description}"
            
            if section.fields:
                section_info += "\nFields to include:"
                for field in section.fields:
                    field_marker = "**REQUIRED**" if field.required else "Optional"
                    section_info += f"\n- {field.field_name} ({field.field_type}) - {field_marker}"
                    if field.description:
                        section_info += f": {field.description}"
            
            template_structure.append(section_info)
        
        # Build comprehensive prompt
        prompt = f"""
You are a professional intelligence analyst formatting a structured report.

TEMPLATE: {template.template_name}
Description: {template.description}

TEMPLATE STRUCTURE:
{chr(10).join(template_structure)}

FORMATTING REQUIREMENTS:
1. Follow the template structure exactly
2. Fill sections based on available research data
3. Use professional, clear language
4. Include specific details and facts from the research
5. Maintain objectivity and accuracy
6. Cite sources where appropriate
"""
        
        # Add user preferences
        if user_preferences.get("communication_style"):
            prompt += f"\n7. Communication style: {user_preferences['communication_style']}"
        
        if user_preferences.get("preferred_detail_level"):
            prompt += f"\n8. Detail level: {user_preferences['preferred_detail_level']}"
        
        # Add modifications if specified
        if modifications:
            prompt += f"\n\nUSER MODIFICATIONS REQUESTED:\n{modifications}"
        
        return prompt
    
    def _extract_all_citations(self, state: Dict[str, Any]) -> List[str]:
        """Extract citations from all agent results"""
        citations = []
        
        # Get citations from agent results
        agent_results = state.get("agent_results", {})
        if "citations" in agent_results:
            citations.extend(agent_results["citations"])
        
        # Get citations from shared memory
        shared_memory = state.get("shared_memory", {})
        for category in ["research_findings", "web_search_results", "document_analyses"]:
            category_data = shared_memory.get(category, {})
            for item_key, item_data in category_data.items():
                if isinstance(item_data, dict) and "citations" in item_data:
                    citations.extend(item_data["citations"])
        
        # Remove duplicates while preserving order
        unique_citations = []
        seen = set()
        for citation in citations:
            if citation not in seen:
                unique_citations.append(citation)
                seen.add(citation)
        
        return unique_citations
    
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
    
    async def _count_completed_sections(self, formatted_report: str) -> int:
        """Count how many sections were successfully completed"""
        try:
            # Simple heuristic: count section headers
            section_count = formatted_report.count("##")
            return max(0, section_count)
        except:
            return 0
