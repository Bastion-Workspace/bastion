"""
Proposal Requirement Analyzer Subgraph

Parses RFI/RFP documents and extracts structured requirements for proposal generation.
Identifies requirements, categorizes them, and builds an index mapping requirements
to proposed proposal sections.
"""

import logging
import json
import re
from typing import Any, Dict, List, Optional
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage

from orchestrator.models.proposal_models import (
    ProposalRequirement,
    RequirementAnalysisResult
)

logger = logging.getLogger(__name__)


# ============================================
# Node Functions
# ============================================

async def parse_rfp_structure_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse RFI/RFP document to identify structure (questions, sections, subsections)
    
    Uses regex and text analysis to extract structured questions from RFI/RFP markdown.
    """
    try:
        logger.info("Parsing RFI/RFP structure...")
        
        pr_req_content = state.get("pr_req_content", "")
        if not pr_req_content or not pr_req_content.strip():
            logger.warning("No RFI/RFP content provided")
            return {
                "error": "No RFI/RFP document content",
                "task_status": "error",
                # Preserve critical state
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
        
        # Extract questions from markdown
        # Patterns: "1. Question text", "Q1: Question", "- Question", "## Section\nQuestion"
        questions = []
        lines = pr_req_content.split("\n")
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Skip empty lines and headers
            if not stripped or stripped.startswith("#"):
                continue
            
            # Match numbered questions (1., 1), lettered (a., A), or bullet points
            if re.match(r'^[\d\w]+[\.\):\-]\s+', stripped):
                question_text = re.sub(r'^[\d\w]+[\.\):\-]\s*', '', stripped).strip()
                if question_text:
                    questions.append({
                        "raw_text": question_text,
                        "line_number": i
                    })
        
        logger.info(f"Extracted {len(questions)} potential questions from RFI/RFP")
        
        return {
            "parsed_questions": questions,
            "rfp_line_count": len(lines),
            # Preserve critical state
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }
    
    except Exception as e:
        logger.error(f"Failed to parse RFI/RFP structure: {e}")
        return {
            "error": str(e),
            "task_status": "error",
            # Preserve critical state
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }


async def extract_requirements_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use LLM to analyze parsed questions and extract structured requirements
    
    LLM identifies:
    - Core requirement text
    - Category (technical, commercial, legal, operational)
    - Whether mandatory or optional
    - Priority level
    """
    try:
        logger.info("Extracting structured requirements via LLM...")
        
        parsed_questions = state.get("parsed_questions", [])
        if not parsed_questions:
            logger.warning("No parsed questions to extract requirements from")
            return {
                "requirements": [],
                # Preserve critical state
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
        
        # Build prompt for LLM to extract requirements
        questions_text = "\n".join([f"{i+1}. {q['raw_text']}" for i, q in enumerate(parsed_questions)])
        
        prompt = f"""Analyze the following RFI/RFP questions and extract structured requirements.

QUESTIONS:
{questions_text}

For each question, provide a JSON object with:
- id: "req_001", "req_002", etc.
- question: The full requirement text
- category: One of: technical, commercial, legal, operational, other
- mandatory: true/false (is this a must-have?)
- compliance_critical: true/false (does this determine overall compliance?)
- priority: 1-5 (1=highest)

Return valid JSON array. Example:
[
  {{"id": "req_001", "question": "...", "category": "technical", "mandatory": true, "compliance_critical": true, "priority": 1}},
  {{"id": "req_002", "question": "...", "category": "commercial", "mandatory": false, "compliance_critical": false, "priority": 3}}
]"""
        
        # Get LLM instance from state (using centralized _get_llm if available)
        # For now, we'll extract it from metadata - the parent agent should have set it
        llm_factory = state.get("_llm_factory")  # Parent agent provides this
        if not llm_factory:
            logger.warning("No LLM factory provided - using fallback response")
            requirements = [
                ProposalRequirement(
                    requirement_id=f"req_{i+1:03d}",
                    question=q["raw_text"],
                    mandatory=True,
                    category="other"
                )
                for i, q in enumerate(parsed_questions[:10])  # Limit to 10 for now
            ]
        else:
            llm = llm_factory(temperature=0.3, state=state)
            messages = [
                SystemMessage(content="You are a requirements analyst. Extract and categorize RFI/RFP requirements."),
                HumanMessage(content=prompt)
            ]
            
            response = await llm.ainvoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON from response
            try:
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if json_match:
                    parsed_reqs = json.loads(json_match.group(0))
                    requirements = [
                        ProposalRequirement(**req) for req in parsed_reqs
                    ]
                else:
                    raise ValueError("No JSON array found in response")
            except Exception as parse_error:
                logger.warning(f"Failed to parse LLM response: {parse_error}, using fallback")
                requirements = [
                    ProposalRequirement(
                        requirement_id=f"req_{i+1:03d}",
                        question=q["raw_text"],
                        mandatory=True,
                        category="other"
                    )
                    for i, q in enumerate(parsed_questions)
                ]
        
        logger.info(f"Extracted {len(requirements)} requirements")
        
        return {
            "requirements": requirements,
            "requirement_count": len(requirements),
            # Preserve critical state
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }
    
    except Exception as e:
        logger.error(f"Failed to extract requirements: {e}")
        return {
            "requirements": [],
            "error": str(e),
            # Preserve critical state
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }


async def build_requirement_index_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build index mapping requirements to proposed proposal sections
    
    Uses LLM to determine which proposal sections should address each requirement
    """
    try:
        logger.info("Building requirement-to-section index...")
        
        requirements = state.get("requirements", [])
        if not requirements:
            return {
                "requirement_index": {},
                # Preserve critical state
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
        
        # Build requirement_id → section mapping
        # Start with common sections for proposals
        default_sections = [
            "executive_summary",
            "company_overview",
            "understanding_of_requirement",
            "proposed_solution",
            "implementation_approach",
            "timeline",
            "team_qualifications",
            "pricing",
            "terms_and_conditions"
        ]
        
        requirement_index = {}
        
        # Map each requirement to most appropriate section
        for req in requirements:
            if isinstance(req, dict):
                req_id = req.get("id", req.get("requirement_id", ""))
                question = req.get("question", "")
            else:
                req_id = req.requirement_id
                question = req.question
            
            # Simple categorization based on question keywords
            question_lower = question.lower()
            
            if any(kw in question_lower for kw in ["company", "background", "experience", "about"]):
                section = "company_overview"
            elif any(kw in question_lower for kw in ["price", "cost", "budget", "fee", "rate"]):
                section = "pricing"
            elif any(kw in question_lower for kw in ["timeline", "schedule", "duration", "when"]):
                section = "timeline"
            elif any(kw in question_lower for kw in ["team", "staff", "resource", "person", "qualification"]):
                section = "team_qualifications"
            elif any(kw in question_lower for kw in ["understand", "approach", "solution", "how"]):
                section = "proposed_solution"
            else:
                section = "understanding_of_requirement"
            
            requirement_index[req_id] = section
        
        logger.info(f"Built index for {len(requirement_index)} requirements")
        
        return {
            "requirement_index": requirement_index,
            "sections_needed": list(set(requirement_index.values())),
            # Preserve critical state
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }
    
    except Exception as e:
        logger.error(f"Failed to build requirement index: {e}")
        return {
            "requirement_index": {},
            "error": str(e),
            # Preserve critical state
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }


# ============================================
# Subgraph Builder
# ============================================

def build_proposal_requirement_analyzer_subgraph(
    checkpointer,
    llm_factory=None,
    get_datetime_context=None
) -> StateGraph:
    """
    Build requirement analyzer subgraph
    
    Args:
        checkpointer: LangGraph checkpointer for state persistence
        llm_factory: Function to get LLM instance
        get_datetime_context: Function to get datetime context
    
    Returns:
        Compiled StateGraph for requirement analysis
    """
    workflow = StateGraph(dict)
    
    # Add nodes
    workflow.add_node("parse_rfp_structure", parse_rfp_structure_node)
    workflow.add_node("extract_requirements", extract_requirements_node)
    workflow.add_node("build_requirement_index", build_requirement_index_node)
    
    # Set entry point
    workflow.set_entry_point("parse_rfp_structure")
    
    # Define edges
    workflow.add_edge("parse_rfp_structure", "extract_requirements")
    workflow.add_edge("extract_requirements", "build_requirement_index")
    workflow.add_edge("build_requirement_index", END)
    
    return workflow.compile(checkpointer=checkpointer)
