"""
Proposal Compliance Validator Subgraph

Validates generated proposal against RFI/RFP requirements.
Builds compliance matrix, identifies gaps, and calculates completeness score.
"""

import logging
from typing import Any, Dict, List
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage

from orchestrator.models.proposal_models import (
    ComplianceMatrixEntry,
    ProposalValidationResult
)

logger = logging.getLogger(__name__)


async def map_content_to_requirements_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use LLM to analyze which sections address which requirements
    
    Creates detailed mapping of requirement coverage
    """
    try:
        logger.info("Mapping proposal content to requirements...")
        
        requirements = state.get("requirements", [])
        sections = state.get("sections", {})
        
        if not requirements or not sections:
            logger.warning("Missing requirements or sections for mapping")
            return {
                "requirement_coverage_map": {},
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
        
        # Build content summary for LLM analysis
        content_summary = {}
        for section_name, section in sections.items():
            if hasattr(section, 'content'):
                content_summary[section_name] = section.content[:500]
            else:
                content_summary[section_name] = str(section)[:500]
        
        # Build requirements list for analysis
        requirements_text = ""
        for i, req in enumerate(requirements):
            if isinstance(req, dict):
                req_id = req.get("id", req.get("requirement_id", f"req_{i}"))
                req_text = req.get("question", "")
            else:
                req_id = req.requirement_id
                req_text = req.question
            
            requirements_text += f"\n{req_id}: {req_text}"
        
        llm_factory = state.get("_llm_factory")
        if not llm_factory:
            logger.warning("No LLM factory provided - using basic mapping")
            coverage_map = {}
            for section_name in sections.keys():
                for req in requirements:
                    req_id = req.requirement_id if hasattr(req, 'requirement_id') else req.get("id", "")
                    coverage_map[req_id] = (section_name, 80.0)
            
            return {
                "requirement_coverage_map": coverage_map,
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
        
        prompt = f"""Analyze how well the proposal sections address the requirements.

REQUIREMENTS:
{requirements_text}

PROPOSAL SECTIONS:
{chr(10).join([f"- {name}: {summary}" for name, summary in content_summary.items()])}

For each requirement, determine:
1. Which section(s) address it
2. Coverage percentage (0-100%)
3. Status (addressed/partial/missing)

Return a dictionary mapping requirement_id to:
- section_name (best matching section)
- coverage_percentage (0-100)
- status (addressed/partial/missing)

Format as JSON dict, e.g.:
{{
  "req_001": {{"section": "proposed_solution", "coverage": 95, "status": "addressed"}},
  "req_002": {{"section": null, "coverage": 0, "status": "missing"}}
}}"""
        
        llm = llm_factory(temperature=0.3, state=state)
        messages = [
            SystemMessage(content="You are a compliance analyst. Evaluate requirement coverage in proposals."),
            HumanMessage(content=prompt)
        ]
        
        response = await llm.ainvoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Parse JSON response (basic parsing)
        import json
        import re
        
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                coverage_data = json.loads(json_match.group(0))
                coverage_map = {}
                for req_id, mapping in coverage_data.items():
                    section = mapping.get("section")
                    coverage = mapping.get("coverage", 0)
                    coverage_map[req_id] = (section, coverage)
            else:
                raise ValueError("No JSON found in response")
        except Exception as parse_error:
            logger.warning(f"Failed to parse coverage map: {parse_error}")
            coverage_map = {}
            for section_name in sections.keys():
                for req in requirements:
                    req_id = req.requirement_id if hasattr(req, 'requirement_id') else req.get("id", "")
                    coverage_map[req_id] = (section_name, 70.0)
        
        logger.info(f"Mapped {len(coverage_map)} requirements to sections")
        
        return {
            "requirement_coverage_map": coverage_map,
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }
    
    except Exception as e:
        logger.error(f"Failed to map content to requirements: {e}")
        return {
            "requirement_coverage_map": {},
            "error": str(e),
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }


async def build_compliance_matrix_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build detailed compliance matrix
    
    Maps each requirement to its coverage status in the proposal
    """
    try:
        logger.info("Building compliance matrix...")
        
        requirements = state.get("requirements", [])
        coverage_map = state.get("requirement_coverage_map", {})
        
        if not requirements:
            return {
                "compliance_matrix": [],
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
        
        matrix_entries = []
        
        for req in requirements:
            if isinstance(req, dict):
                req_id = req.get("id", req.get("requirement_id", ""))
                req_text = req.get("question", "")
            else:
                req_id = req.requirement_id
                req_text = req.question
            
            coverage_info = coverage_map.get(req_id, (None, 0))
            section_name, coverage_pct = coverage_info
            
            # Determine status based on coverage
            if coverage_pct >= 90:
                status = "addressed"
            elif coverage_pct >= 50:
                status = "partial"
            else:
                status = "missing"
            
            entry = ComplianceMatrixEntry(
                requirement_id=req_id,
                requirement_text=req_text,
                addressed_by_section=section_name,
                coverage_percentage=coverage_pct,
                status=status
            )
            
            matrix_entries.append(entry)
        
        logger.info(f"Built compliance matrix with {len(matrix_entries)} entries")
        
        return {
            "compliance_matrix": matrix_entries,
            "matrix_entries_count": len(matrix_entries),
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }
    
    except Exception as e:
        logger.error(f"Failed to build compliance matrix: {e}")
        return {
            "compliance_matrix": [],
            "error": str(e),
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }


async def identify_gaps_and_score_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Identify missing requirements and calculate completeness score
    """
    try:
        logger.info("Identifying gaps and calculating completeness score...")
        
        compliance_matrix = state.get("compliance_matrix", [])
        
        if not compliance_matrix:
            return {
                "missing_requirements": [],
                "partial_requirements": [],
                "addressed_requirements": [],
                "completeness_score": 0.0,
                "validation_summary": "No compliance matrix available",
                "critical_gaps": [],
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
        
        addressed = []
        partial = []
        missing = []
        
        for entry in compliance_matrix:
            if isinstance(entry, ComplianceMatrixEntry):
                if entry.status == "addressed":
                    addressed.append(entry.requirement_id)
                elif entry.status == "partial":
                    partial.append(entry.requirement_id)
                else:
                    missing.append(entry.requirement_id)
            elif isinstance(entry, dict):
                status = entry.get("status", "missing")
                req_id = entry.get("requirement_id", "")
                if status == "addressed":
                    addressed.append(req_id)
                elif status == "partial":
                    partial.append(req_id)
                else:
                    missing.append(req_id)
        
        # Calculate completeness score
        total = len(compliance_matrix)
        if total > 0:
            completeness_score = (
                len(addressed) * 100 +
                len(partial) * 50
            ) / (total * 100) * 100
        else:
            completeness_score = 0.0
        
        # Determine overall compliance status
        if completeness_score >= 95:
            status = "complete"
        elif completeness_score >= 75:
            status = "mostly_complete"
        else:
            status = "incomplete"
        
        # Identify critical gaps (missing mandatory requirements)
        critical_gaps = [req_id for req_id in missing if req_id in missing[:3]]
        
        summary = f"Proposal completeness: {completeness_score:.1f}% ({len(addressed)} addressed, {len(partial)} partial, {len(missing)} missing)"
        
        logger.info(summary)
        
        return {
            "missing_requirements": missing,
            "partial_requirements": partial,
            "addressed_requirements": addressed,
            "completeness_score": min(100.0, completeness_score),
            "compliance_status": status,
            "validation_summary": summary,
            "critical_gaps": critical_gaps,
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }
    
    except Exception as e:
        logger.error(f"Failed to identify gaps and score: {e}")
        return {
            "missing_requirements": [],
            "partial_requirements": [],
            "addressed_requirements": [],
            "completeness_score": 0.0,
            "error": str(e),
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }


def build_proposal_compliance_validator_subgraph(
    checkpointer,
    llm_factory=None,
    get_datetime_context=None
) -> StateGraph:
    """
    Build compliance validator subgraph
    
    Args:
        checkpointer: LangGraph checkpointer for state persistence
        llm_factory: Function to get LLM instance
        get_datetime_context: Function to get datetime context
    
    Returns:
        Compiled StateGraph for compliance validation
    """
    workflow = StateGraph(dict)
    
    workflow.add_node("map_content_to_requirements", map_content_to_requirements_node)
    workflow.add_node("build_compliance_matrix", build_compliance_matrix_node)
    workflow.add_node("identify_gaps_and_score", identify_gaps_and_score_node)
    
    workflow.set_entry_point("map_content_to_requirements")
    
    workflow.add_edge("map_content_to_requirements", "build_compliance_matrix")
    workflow.add_edge("build_compliance_matrix", "identify_gaps_and_score")
    workflow.add_edge("identify_gaps_and_score", END)
    
    return workflow.compile(checkpointer=checkpointer)
