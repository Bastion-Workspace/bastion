"""
Proposal Section Generator Subgraph

Generates proposal sections by combining:
- Specific requirements to address
- Company knowledge and facts
- Style guide/template
- Customer context

Produces formatted proposal sections ready for inclusion in final document.
"""

import logging
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage

from orchestrator.models.proposal_models import ProposalSection

logger = logging.getLogger(__name__)


# ============================================
# Node Functions
# ============================================

async def prepare_section_context_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare context for section generation
    
    Gathers relevant requirements, company knowledge, and style guide
    for the current section being generated
    """
    try:
        logger.info("Preparing section generation context...")
        
        sections_needed = state.get("sections_needed", [])
        requested_sections = state.get("requested_sections", [])
        requirement_index = state.get("requirement_index", {})
        company_knowledge = state.get("company_knowledge", "")
        pr_style_content = state.get("pr_style_content", "")
        
        # Filter sections_needed based on user request
        # If user requested specific sections, only generate those
        # If requested_sections is empty, generate all sections_needed
        if requested_sections:
            # Only generate requested sections that are in sections_needed
            sections_to_generate = [
                sec for sec in requested_sections 
                if sec in sections_needed or not sections_needed  # If sections_needed empty, allow any requested
            ]
            if not sections_to_generate and sections_needed:
                # User requested sections that aren't in sections_needed - log warning but try anyway
                logger.warning(f"Requested sections {requested_sections} not in sections_needed {sections_needed}, generating requested anyway")
                sections_to_generate = requested_sections
        else:
            # No specific request - generate all sections_needed
            sections_to_generate = sections_needed
        
        if not sections_to_generate:
            logger.warning("No sections to generate")
            return {
                "section_context_ready": False,
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
        
        logger.info(f"Generating {len(sections_to_generate)} section(s): {sections_to_generate}")
        
        # Map requirements to sections for easy lookup during generation
        section_to_requirements = {}
        for req_id, section_name in requirement_index.items():
            if section_name not in section_to_requirements:
                section_to_requirements[section_name] = []
            section_to_requirements[section_name].append(req_id)
        
        logger.info(f"Context prepared for {len(sections_to_generate)} section(s)")
        
        return {
            "section_context_ready": True,
            "sections_to_generate": sections_to_generate,  # Pass filtered list
            "section_to_requirements": section_to_requirements,
            "section_definitions": state.get("section_definitions", {}),  # Preserve section definitions
            "company_knowledge_available": len(company_knowledge) > 0,
            "style_guide_available": len(pr_style_content) > 0,
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }
    
    except Exception as e:
        logger.error(f"Failed to prepare section context: {e}")
        return {
            "section_context_ready": False,
            "error": str(e),
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }


async def generate_section_content_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate proposal section content using LLM
    
    Combines section type, customer context, company knowledge, and requirements
    """
    try:
        logger.info("Generating proposal section content...")
        
        # Use filtered sections list (from prepare_section_context_node) or fallback to sections_needed
        sections_to_generate = state.get("sections_to_generate", state.get("sections_needed", []))
        requirement_index = state.get("requirement_index", {})
        company_knowledge = state.get("company_knowledge", "")
        pr_style_content = state.get("pr_style_content", "")
        customer_name = state.get("customer_name", "Valued Customer")
        proposal_type = state.get("proposal_type", "commercial_services")
        target_length = state.get("target_length_words", 8000)
        
        llm_factory = state.get("_llm_factory")
        if not llm_factory:
            logger.warning("No LLM factory provided")
            return {
                "sections": {},
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
        
        sections_generated = {}
        
        # Get section definitions from pr-outline (preferred) or use fallback templates
        section_definitions = state.get("section_definitions", {})
        
        # Fallback templates if pr-outline doesn't have section definitions
        fallback_templates = {
            "executive_summary": "A compelling 1-2 paragraph summary",
            "company_overview": "Background and key qualifications",
            "understanding_of_requirement": "Understanding of customer needs",
            "proposed_solution": "Specific solution approach",
            "implementation_approach": "Step-by-step implementation plan",
            "timeline": "Project schedule and milestones",
            "team_qualifications": "Team composition and qualifications",
            "pricing": "Pricing model and financial terms",
            "terms_and_conditions": "Terms and special conditions"
        }
        
        # Calculate target length per section
        target_per_section = target_length // len(sections_to_generate) if sections_to_generate else target_length
        
        for section_name in sections_to_generate:
            try:
                # Use section definition from pr-outline if available, otherwise fallback
                if section_name in section_definitions:
                    template = section_definitions[section_name]
                    logger.info(f"Using section definition from pr-outline for {section_name}")
                else:
                    template = fallback_templates.get(section_name, "A proposal section")
                    logger.debug(f"Using fallback template for {section_name}")
                
                section_reqs = []
                for req_id, sec_name in requirement_index.items():
                    if sec_name == section_name:
                        section_reqs.append(req_id)
                
                reqs_context = f"\nAddresses requirements: {', '.join(section_reqs)}" if section_reqs else ""
                
                # Build section guidance (from pr-outline if available)
                section_guidance = template
                if section_name in section_definitions:
                    section_guidance = f"{template}\n\n(This guidance comes from the pr-outline document)"
                
                prompt = f"""Generate a professional proposal section.

SECTION: {section_name.replace('_', ' ').title()}
SECTION GUIDANCE (from pr-outline): {section_guidance}
CUSTOMER: {customer_name}
PROPOSAL TYPE: {proposal_type}{reqs_context}

STYLE GUIDE:
{pr_style_content if pr_style_content else "Use professional, clear business language"}

COMPANY INFORMATION:
{company_knowledge if company_knowledge else "No additional company information provided"}

Generate markdown content. Target: {target_per_section} words.
Make it compelling and specific to the customer.
Follow the section guidance from pr-outline closely.
"""
                
                llm = llm_factory(temperature=0.6, state=state)
                messages = [
                    SystemMessage(content="You are an expert proposal writer. Create compelling, professional proposal sections."),
                    HumanMessage(content=prompt)
                ]
                
                response = await llm.ainvoke(messages)
                content = response.content if hasattr(response, 'content') else str(response)
                
                word_count = len(content.split())
                sections_generated[section_name] = ProposalSection(
                    section_name=section_name,
                    content=content,
                    word_count=word_count,
                    requirement_ids=section_reqs,
                    coverage_percentage=100.0 if section_reqs else 50.0,
                    quality_score=0.8,
                    style_applied=len(pr_style_content) > 0
                )
                
                logger.info(f"Generated {section_name}: {word_count} words")
            
            except Exception as section_error:
                logger.warning(f"Failed to generate section {section_name}: {section_error}")
                continue
        
        logger.info(f"Generated {len(sections_generated)} proposal sections")
        
        # Merge with existing sections (preserve sections not being regenerated)
        existing_sections = state.get("sections", {})
        if existing_sections:
            # Only keep existing sections that weren't just regenerated
            for section_name, section in existing_sections.items():
                if section_name not in sections_generated:
                    sections_generated[section_name] = section
                    logger.debug(f"Preserved existing section: {section_name}")
        
        return {
            "sections": sections_generated,
            "sections_count": len(sections_generated),
            "section_definitions": state.get("section_definitions", {}),  # Preserve section definitions
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }
    
    except Exception as e:
        logger.error(f"Failed to generate section content: {e}")
        return {
            "sections": {},
            "error": str(e),
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }


async def validate_section_quality_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate generated sections for quality and completeness
    """
    try:
        logger.info("Validating generated sections...")
        
        sections = state.get("sections", {})
        target_length = state.get("target_length_words", 8000)
        
        if not sections:
            return {
                "sections_valid": False,
                "validation_issues": ["No sections generated"],
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
        
        total_words = 0
        issues = []
        
        for section_name, section in sections.items():
            if isinstance(section, ProposalSection):
                total_words += section.word_count
                
                if section.word_count < 100:
                    issues.append(f"{section_name}: Only {section.word_count} words (recommended: 200+)")
            else:
                issues.append(f"{section_name}: Invalid section format")
        
        if total_words < target_length * 0.7:
            issues.append(f"Total proposal too short: {total_words} words (target: {target_length})")
        
        is_valid = len(issues) == 0
        
        logger.info(f"Validation complete: {total_words} words, {len(issues)} issues")
        
        return {
            "sections_valid": is_valid,
            "validation_issues": issues,
            "total_word_count": total_words,
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }
    
    except Exception as e:
        logger.error(f"Failed to validate sections: {e}")
        return {
            "sections_valid": False,
            "error": str(e),
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }


def build_proposal_section_generator_subgraph(
    checkpointer,
    llm_factory=None,
    get_datetime_context=None
) -> StateGraph:
    """
    Build section generator subgraph
    
    Args:
        checkpointer: LangGraph checkpointer for state persistence
        llm_factory: Function to get LLM instance
        get_datetime_context: Function to get datetime context
    
    Returns:
        Compiled StateGraph for section generation
    """
    workflow = StateGraph(dict)
    
    workflow.add_node("prepare_section_context", prepare_section_context_node)
    workflow.add_node("generate_section_content", generate_section_content_node)
    workflow.add_node("validate_section_quality", validate_section_quality_node)
    
    workflow.set_entry_point("prepare_section_context")
    
    workflow.add_edge("prepare_section_context", "generate_section_content")
    workflow.add_edge("generate_section_content", "validate_section_quality")
    workflow.add_edge("validate_section_quality", END)
    
    return workflow.compile(checkpointer=checkpointer)
