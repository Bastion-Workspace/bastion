"""
Proposal Generation Agent

LangGraph-based agent for generating customer proposals from RFI/RFP documents.
Orchestrates 4 subgraphs: requirement analyzer, section generator, compliance validator, and editor operations.

Supports:
- Generation mode: Create new proposals from RFI/RFP + company knowledge + style guide
- Editing mode: Update existing proposals with targeted operations
- Compliance validation: Build requirement-to-section mapping
"""

import logging
from typing import Dict, Any, List, Optional, TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage

from orchestrator.agents.base_agent import BaseAgent
from orchestrator.subgraphs.proposal import (
    build_proposal_requirement_analyzer_subgraph,
    build_proposal_section_generator_subgraph,
    build_proposal_compliance_validator_subgraph,
    build_proposal_editor_operations_subgraph
)

logger = logging.getLogger(__name__)


class ProposalState(TypedDict):
    """State for proposal generation workflow"""
    query: str
    user_id: str
    metadata: Dict[str, Any]
    messages: List[Any]
    shared_memory: Dict[str, Any]
    
    customer_name: Optional[str]
    proposal_type: Optional[str]
    
    # Reference document tracking (for checkpoint retention)
    reference_document_ids: Dict[str, Optional[str]]  # {"pr_outline": "doc_id"} - proposal references pr-outline
    
    # pr-outline content and cascaded references
    pr_outline_content: str
    section_definitions: Dict[str, str]  # Section definitions from pr-outline (section_name -> guidance)
    
    # Cascaded references from pr-outline frontmatter
    pr_req_content: str
    pr_style_content: str
    company_knowledge: str
    
    requirements: List[Any]
    requirement_index: Dict[str, str]
    sections_needed: List[str]
    requested_sections: List[str]  # User-specified sections to generate (empty = all)
    sections: Dict[str, Any]
    
    compliance_matrix: List[Any]
    missing_requirements: List[str]
    completeness_score: float
    
    editor_content: str
    editing_mode: bool
    editor_operations: List[Dict[str, Any]]
    
    # User intent flags (for conditional workflow routing)
    user_wants_regeneration: bool  # True if user explicitly wants to regenerate sections
    skip_requirement_analysis: bool  # True if requirements already exist and references unchanged
    skip_section_generation: bool  # True if sections exist and user is iterating (not regenerating)
    
    response: Dict[str, Any]
    task_status: str
    error: str


class ProposalGenerationAgent(BaseAgent):
    """
    Proposal Generation Agent
    
    Orchestrates proposal generation workflow using reusable subgraphs.
    Handles generation mode (new proposals) and editing mode (updates to existing proposals).
    """
    
    def __init__(self):
        super().__init__("proposal_generation_agent")
        self._grpc_client = None
        logger.debug("Proposal Generation Agent ready!")
    
    def _detect_user_intent(self, query: str) -> Dict[str, bool]:
        """
        Detect user intent from query to determine if they want regeneration vs iteration
        
        Returns:
            Dict with flags: user_wants_regeneration, user_wants_iteration
        """
        query_lower = query.lower().strip()
        
        # Explicit regeneration keywords
        regeneration_keywords = [
            "regenerate", "recreate", "start over", "generate new",
            "generate proposal", "create proposal", "write proposal",
            "generate from scratch", "rebuild", "redo"
        ]
        
        # Iteration/editing keywords
        iteration_keywords = [
            "update", "edit", "improve", "refine", "strengthen",
            "add more", "expand", "modify", "change", "adjust",
            "enhance", "polish", "revise", "tweak"
        ]
        
        user_wants_regeneration = any(kw in query_lower for kw in regeneration_keywords)
        user_wants_iteration = any(kw in query_lower for kw in iteration_keywords)
        
        # If both detected, regeneration takes precedence (user is explicit)
        if user_wants_regeneration and user_wants_iteration:
            user_wants_iteration = False
        
        return {
            "user_wants_regeneration": user_wants_regeneration,
            "user_wants_iteration": user_wants_iteration
        }
    
    def _detect_requested_sections(self, query: str) -> List[str]:
        """
        Parse user query to detect which specific sections they want to generate
        
        Maps natural language section names to internal section identifiers.
        
        Examples:
        - "Craft the introduction" → ["executive_summary"]
        - "Write the pricing section" → ["pricing"]
        - "Create the timeline" → ["timeline"]
        - "Generate introduction and pricing" → ["executive_summary", "pricing"]
        
        Returns:
            List of section names to generate, or empty list if all sections should be generated
        """
        query_lower = query.lower().strip()
        
        # Section name mappings (natural language → internal name)
        section_mappings = {
            # Introduction/opening sections
            "introduction": "executive_summary",
            "intro": "executive_summary",
            "opening": "executive_summary",
            "summary": "executive_summary",
            "executive summary": "executive_summary",
            "executive_summary": "executive_summary",
            
            # Company/background sections
            "company": "company_overview",
            "company overview": "company_overview",
            "company_overview": "company_overview",
            "about us": "company_overview",
            "background": "company_overview",
            "qualifications": "company_overview",
            "experience": "company_overview",
            
            # Understanding/requirements
            "understanding": "understanding_of_requirement",
            "understanding of requirement": "understanding_of_requirement",
            "understanding_of_requirement": "understanding_of_requirement",
            "requirements": "understanding_of_requirement",
            "needs": "understanding_of_requirement",
            
            # Solution sections
            "solution": "proposed_solution",
            "proposed solution": "proposed_solution",
            "proposed_solution": "proposed_solution",
            "approach": "proposed_solution",
            "methodology": "proposed_solution",
            
            # Implementation
            "implementation": "implementation_approach",
            "implementation approach": "implementation_approach",
            "implementation_approach": "implementation_approach",
            "plan": "implementation_approach",
            "process": "implementation_approach",
            "method": "implementation_approach",
            
            # Timeline/schedule
            "timeline": "timeline",
            "schedule": "timeline",
            "timing": "timeline",
            "milestones": "timeline",
            "deliverables": "timeline",
            
            # Team/people
            "team": "team_qualifications",
            "team qualifications": "team_qualifications",
            "team_qualifications": "team_qualifications",
            "staff": "team_qualifications",
            "people": "team_qualifications",
            "personnel": "team_qualifications",
            "resources": "team_qualifications",
            
            # Pricing/commercial
            "pricing": "pricing",
            "price": "pricing",
            "cost": "pricing",
            "budget": "pricing",
            "financial": "pricing",
            "commercial": "pricing",
            "fees": "pricing",
            
            # Terms/legal
            "terms": "terms_and_conditions",
            "terms and conditions": "terms_and_conditions",
            "terms_and_conditions": "terms_and_conditions",
            "conditions": "terms_and_conditions",
            "legal": "terms_and_conditions",
            "contract": "terms_and_conditions"
        }
        
        # Detect section requests in query
        requested_sections = []
        
        # Check for explicit section mentions
        for natural_name, internal_name in section_mappings.items():
            # Look for patterns like "craft the introduction", "write pricing", "create timeline"
            patterns = [
                f"craft the {natural_name}",
                f"write the {natural_name}",
                f"create the {natural_name}",
                f"generate the {natural_name}",
                f"draft the {natural_name}",
                f"the {natural_name}",
                f"{natural_name} section",
                f"section {natural_name}"
            ]
            
            if any(pattern in query_lower for pattern in patterns):
                if internal_name not in requested_sections:
                    requested_sections.append(internal_name)
        
        # Also check for direct section name mentions (e.g., "executive_summary")
        for internal_name in set(section_mappings.values()):
            if internal_name.replace("_", " ") in query_lower or internal_name in query_lower:
                if internal_name not in requested_sections:
                    requested_sections.append(internal_name)
        
        # If no specific sections requested, return empty list (generate all)
        if not requested_sections:
            logger.debug("No specific sections requested - will generate all sections")
        else:
            logger.info(f"Detected requested sections: {requested_sections}")
        
        return requested_sections
    
    def _extract_section_definitions(self, pr_outline_content: str) -> Dict[str, str]:
        """
        Extract section definitions from pr-outline document
        
        Looks for markdown sections like:
        ## Executive Summary
        [guidance text]
        
        Returns:
            Dict mapping section_name (snake_case) to guidance text
        """
        import re
        
        section_definitions = {}
        
        if not pr_outline_content:
            return section_definitions
        
        # Strip frontmatter
        from orchestrator.utils.frontmatter_utils import strip_frontmatter_block
        body_content = strip_frontmatter_block(pr_outline_content)
        
        # Find all level 2 headers (## Section Name)
        pattern = r'^##\s+(.+?)$'
        lines = body_content.split('\n')
        
        current_section = None
        current_content = []
        
        for line in lines:
            match = re.match(pattern, line)
            if match:
                # Save previous section if exists
                if current_section and current_content:
                    section_name = current_section.lower().replace(' ', '_').replace('-', '_')
                    section_definitions[section_name] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = match.group(1).strip()
                current_content = []
            elif current_section:
                # Continue collecting content for current section
                current_content.append(line)
        
        # Save last section
        if current_section and current_content:
            section_name = current_section.lower().replace(' ', '_').replace('-', '_')
            section_definitions[section_name] = '\n'.join(current_content).strip()
        
        logger.info(f"Extracted section definitions: {list(section_definitions.keys())}")
        return section_definitions
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """Build LangGraph workflow for proposal generation"""
        workflow = StateGraph(ProposalState)
        
        # Build subgraphs
        requirement_analyzer = build_proposal_requirement_analyzer_subgraph(
            checkpointer,
            llm_factory=self._get_llm,
            get_datetime_context=self._get_datetime_context
        )
        section_generator = build_proposal_section_generator_subgraph(
            checkpointer,
            llm_factory=self._get_llm,
            get_datetime_context=self._get_datetime_context
        )
        compliance_validator = build_proposal_compliance_validator_subgraph(
            checkpointer,
            llm_factory=self._get_llm,
            get_datetime_context=self._get_datetime_context
        )
        editor_operations = build_proposal_editor_operations_subgraph(
            checkpointer,
            llm_factory=self._get_llm,
            get_datetime_context=self._get_datetime_context
        )
        
        # Add nodes
        workflow.add_node("prepare_context", self._prepare_context_node)
        workflow.add_node("requirement_analyzer", requirement_analyzer)
        workflow.add_node("section_generator", section_generator)
        workflow.add_node("compliance_validator", compliance_validator)
        workflow.add_node("editor_operations", editor_operations)
        workflow.add_node("format_response", self._format_response_node)
        
        # Entry point
        workflow.set_entry_point("prepare_context")
        
        # Conditional edge: skip requirement_analyzer if requirements already exist and references unchanged
        workflow.add_conditional_edges(
            "prepare_context",
            lambda state: "skip_requirement_analysis" if state.get("skip_requirement_analysis") else "requirement_analyzer",
            {
                "skip_requirement_analysis": "check_section_generation",
                "requirement_analyzer": "requirement_analyzer"
            }
        )
        
        # Edge from requirement_analyzer to check_section_generation
        workflow.add_edge("requirement_analyzer", "check_section_generation")
        
        # Add a pass-through node to check if we should skip section generation
        workflow.add_node("check_section_generation", self._check_section_generation_node)
        
        # Conditional edge: skip section_generator if sections exist and user is iterating (not regenerating)
        workflow.add_conditional_edges(
            "check_section_generation",
            lambda state: "skip_section_generation" if state.get("skip_section_generation") else "section_generator",
            {
                "skip_section_generation": "compliance_validator",
                "section_generator": "section_generator"
            }
        )
        
        # Edge from section_generator to compliance_validator
        workflow.add_edge("section_generator", "compliance_validator")
        
        # Conditional edge: if editing mode, go to editor_operations; else to format_response
        workflow.add_conditional_edges(
            "compliance_validator",
            lambda state: "editor_operations" if state.get("editing_mode") else "format_response",
            {
                "editor_operations": "editor_operations",
                "format_response": "format_response"
            }
        )
        
        workflow.add_edge("editor_operations", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow.compile(checkpointer=checkpointer)
    
    async def _prepare_context_node(self, state: ProposalState) -> Dict[str, Any]:
        """
        Prepare context: load references and detect proposal type
        
        Uses retained content from checkpoint if references haven't changed.
        """
        try:
            logger.info("Preparing proposal generation context...")
            
            messages = state.get("messages", [])
            shared_memory = state.get("shared_memory", {})
            metadata = state.get("metadata", {})
            active_editor = shared_memory.get("active_editor", {})
            
            # Get pr-outline document ID from proposal frontmatter (like manuscript references outline)
            frontmatter = active_editor.get("frontmatter", {})
            editor_content = active_editor.get("content", "")
            
            # Proposal references pr-outline (cascading pattern like fiction)
            pr_outline_doc_id = frontmatter.get("pr_outline") or frontmatter.get("pr_outline_document_id") or metadata.get("pr_outline_document_id")
            
            # Get current reference document IDs (for checkpoint comparison)
            current_refs = {
                "pr_outline": pr_outline_doc_id
            }
            
            # Check if references changed (compare with existing state)
            existing_refs = state.get("reference_document_ids", {})
            references_changed = (
                current_refs.get("pr_outline") != existing_refs.get("pr_outline")
            )
            
            # Use retained content if references unchanged, otherwise load fresh
            pr_outline_content = state.get("pr_outline_content", "")
            pr_req_content = state.get("pr_req_content", "")
            pr_style_content = state.get("pr_style_content", "")
            company_knowledge = state.get("company_knowledge", "")
            section_definitions = state.get("section_definitions", {})
            
            # Load pr-outline if references changed or content missing
            if references_changed or not pr_outline_content:
                if pr_outline_doc_id:
                    try:
                        from orchestrator.tools.document_tools import get_document_content_tool
                        pr_outline_content = await get_document_content_tool(pr_outline_doc_id, state.get("user_id", "system"))
                        logger.info("Loaded pr-outline document from storage")
                    except Exception as e:
                        logger.warning(f"Failed to load pr-outline document: {e}")
                        pr_outline_content = ""
                else:
                    logger.warning("No pr-outline document ID found - proposal should reference pr-outline in frontmatter")
                    pr_outline_content = ""
            else:
                logger.info("Using retained pr-outline content from checkpoint")
            
            # Extract section definitions from pr-outline if we have it
            if pr_outline_content and (references_changed or not section_definitions):
                section_definitions = self._extract_section_definitions(pr_outline_content)
                logger.info(f"Extracted {len(section_definitions)} section definitions from pr-outline")
            
            # Load cascaded references from pr-outline frontmatter (like fiction cascades from outline)
            if pr_outline_content and (references_changed or not pr_req_content or not pr_style_content or not company_knowledge):
                try:
                    from orchestrator.tools.reference_file_loader import load_referenced_files
                    from orchestrator.utils.frontmatter_utils import parse_frontmatter
                    
                    # Parse pr-outline frontmatter to get its references
                    outline_frontmatter, _ = parse_frontmatter(pr_outline_content)
                    
                    # Load cascaded references using reference_file_loader
                    # pr-outline frontmatter has: pr_req_document_id, pr_style_document_id, company_knowledge_id
                    cascade_config = {
                        "pr_outline": {
                            "pr_req": ["pr_req_document_id", "pr_req"],
                            "pr_style": ["pr_style_document_id", "pr_style"],
                            "company": ["company_knowledge_id", "company_knowledge"]
                        }
                    }
                    
                    # Create temporary active_editor for pr-outline to load its references
                    outline_editor = {
                        "content": pr_outline_content,
                        "frontmatter": outline_frontmatter,
                        "document_id": pr_outline_doc_id
                    }
                    
                    result = await load_referenced_files(
                        active_editor=outline_editor,
                        user_id=state.get("user_id", "system"),
                        reference_config={"pr_outline": ["pr_outline"]},  # Not used, but required
                        doc_type_filter="pr-outline",
                        cascade_config=cascade_config
                    )
                    
                    loaded_files = result.get("loaded_files", {})
                    
                    # Extract cascaded content
                    if loaded_files.get("pr_req") and len(loaded_files["pr_req"]) > 0:
                        pr_req_content = loaded_files["pr_req"][0].get("content", "")
                        logger.info("Loaded pr-req via cascade from pr-outline")
                    
                    if loaded_files.get("pr_style") and len(loaded_files["pr_style"]) > 0:
                        pr_style_content = loaded_files["pr_style"][0].get("content", "")
                        logger.info("Loaded pr-style via cascade from pr-outline")
                    
                    if loaded_files.get("company") and len(loaded_files["company"]) > 0:
                        company_knowledge = loaded_files["company"][0].get("content", "")
                        logger.info("Loaded company knowledge via cascade from pr-outline")
                    
                except Exception as e:
                    logger.warning(f"Failed to load cascaded references from pr-outline: {e}")
                    # Fallback: try direct loading from pr-outline frontmatter
                    if pr_outline_content:
                        try:
                            from orchestrator.utils.frontmatter_utils import parse_frontmatter
                            from orchestrator.tools.document_tools import get_document_content_tool
                            
                            outline_frontmatter, _ = parse_frontmatter(pr_outline_content)
                            
                            # Direct load from pr-outline frontmatter
                            pr_req_doc_id = outline_frontmatter.get("pr_req_document_id") or outline_frontmatter.get("pr_req")
                            pr_style_doc_id = outline_frontmatter.get("pr_style_document_id") or outline_frontmatter.get("pr_style")
                            company_knowledge_id = outline_frontmatter.get("company_knowledge_id") or outline_frontmatter.get("company_knowledge")
                            
                            if pr_req_doc_id:
                                pr_req_content = await get_document_content_tool(pr_req_doc_id, state.get("user_id", "system"))
                            if pr_style_doc_id:
                                pr_style_content = await get_document_content_tool(pr_style_doc_id, state.get("user_id", "system"))
                            if company_knowledge_id:
                                company_knowledge = await get_document_content_tool(company_knowledge_id, state.get("user_id", "system"))
                            
                            logger.info("Loaded references directly from pr-outline frontmatter (fallback)")
                        except Exception as fallback_error:
                            logger.warning(f"Fallback reference loading also failed: {fallback_error}")
            else:
                logger.info("Using retained cascaded references from checkpoint")
            
            # Determine editing mode
            editing_mode = len(editor_content.strip()) > 0
            
            customer_name = frontmatter.get("customer_name") or metadata.get("customer_name", "Valued Customer")
            proposal_type = frontmatter.get("proposal_type") or metadata.get("proposal_type", "commercial_services")
            
            # Check if we should skip requirement analysis (requirements exist and references unchanged)
            existing_requirements = state.get("requirements", [])
            existing_requirement_index = state.get("requirement_index", {})
            skip_requirement_analysis = (
                not references_changed and
                len(existing_requirements) > 0 and
                existing_requirement_index
            )
            
            # Derive sections_needed from section_definitions (pr-outline) or requirement_index
            sections_needed = state.get("sections_needed", [])
            if not sections_needed:
                # Prefer section definitions from pr-outline (source of truth)
                if section_definitions:
                    sections_needed = list(section_definitions.keys())
                    logger.info(f"Derived sections_needed from pr-outline section definitions: {sections_needed}")
                elif skip_requirement_analysis and existing_requirement_index:
                    # Fallback: extract from requirement_index
                    sections_needed = list(set(existing_requirement_index.values()))
                    logger.info(f"Derived sections_needed from requirement_index: {sections_needed}")
            
            if skip_requirement_analysis:
                logger.info(f"✅ Skipping requirement analysis - {len(existing_requirements)} requirements retained from checkpoint")
            
            logger.info(f"Context prepared: {customer_name}, {proposal_type}, editing_mode={editing_mode}, references_changed={references_changed}")
            
            return {
                "reference_document_ids": current_refs,
                "pr_outline_content": pr_outline_content,
                "section_definitions": section_definitions,
                "pr_req_content": pr_req_content,
                "pr_style_content": pr_style_content,
                "company_knowledge": company_knowledge,
                "customer_name": customer_name,
                "proposal_type": proposal_type,
                "editing_mode": editing_mode,
                "editor_content": editor_content,
                "skip_requirement_analysis": skip_requirement_analysis,
                "sections_needed": sections_needed,
                "_llm_factory": self._get_llm,
                # Preserve critical state
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                # Preserve proposal-specific state (important when skipping requirement_analyzer)
                "requirements": state.get("requirements", []),
                "requirement_index": state.get("requirement_index", {}),
                "requested_sections": state.get("requested_sections", []),
                "sections": state.get("sections", {}),
                "compliance_matrix": state.get("compliance_matrix", []),
                "missing_requirements": state.get("missing_requirements", []),
                "completeness_score": state.get("completeness_score", 0.0),
                "user_wants_regeneration": state.get("user_wants_regeneration", False)
            }
        
        except Exception as e:
            logger.error(f"Failed to prepare context: {e}")
            return {
                "error": str(e),
                "task_status": "error",
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    async def _check_section_generation_node(self, state: ProposalState) -> Dict[str, Any]:
        """
        Check if we should skip section generation
        
        Skip if:
        - All requested sections already exist
        - User is iterating (not regenerating)
        - Requirements exist (so we can validate compliance)
        - User didn't request specific new sections
        """
        try:
            existing_sections = state.get("sections", {})
            user_wants_regeneration = state.get("user_wants_regeneration", False)
            requirements_exist = len(state.get("requirements", [])) > 0
            requested_sections = state.get("requested_sections", [])
            
            # If user requested specific sections, check if they all exist
            if requested_sections:
                # Check if all requested sections already exist
                all_requested_exist = all(
                    sec in existing_sections for sec in requested_sections
                )
                
                # Skip only if all requested sections exist and user is iterating (not regenerating)
                skip_section_generation = (
                    all_requested_exist and
                    not user_wants_regeneration and
                    requirements_exist
                )
                
                if skip_section_generation:
                    logger.info(f"✅ Skipping section generation - all requested sections ({requested_sections}) already exist")
                else:
                    missing_sections = [sec for sec in requested_sections if sec not in existing_sections]
                    if missing_sections:
                        logger.info(f"Generating requested sections: {missing_sections}")
                    else:
                        logger.info(f"Regenerating requested sections: {requested_sections}")
            else:
                # No specific request - skip if all sections exist and user is iterating
                skip_section_generation = (
                    len(existing_sections) > 0 and
                    not user_wants_regeneration and
                    requirements_exist
                )
                
                if skip_section_generation:
                    logger.info(f"✅ Skipping section generation - {len(existing_sections)} sections retained, user is iterating")
                else:
                    if user_wants_regeneration:
                        logger.info("User wants regeneration - will regenerate sections")
                    elif not requirements_exist:
                        logger.info("No requirements available - will generate sections after requirement analysis")
                    else:
                        logger.info("Generating all sections")
            
            return {
                "skip_section_generation": skip_section_generation,
                # Preserve critical state
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                # Preserve proposal-specific state
                "requirements": state.get("requirements", []),
                "requirement_index": state.get("requirement_index", {}),
                "sections_needed": state.get("sections_needed", []),
                "requested_sections": state.get("requested_sections", []),
                "section_definitions": state.get("section_definitions", {}),
                "sections": state.get("sections", {}),
                "reference_document_ids": state.get("reference_document_ids", {}),
                "pr_outline_content": state.get("pr_outline_content", ""),
                "pr_req_content": state.get("pr_req_content", ""),
                "pr_style_content": state.get("pr_style_content", ""),
                "company_knowledge": state.get("company_knowledge", ""),
                "customer_name": state.get("customer_name"),
                "proposal_type": state.get("proposal_type"),
                "editing_mode": state.get("editing_mode", False),
                "editor_content": state.get("editor_content", ""),
                "user_wants_regeneration": state.get("user_wants_regeneration", False)
            }
        
        except Exception as e:
            logger.error(f"Failed to check section generation: {e}")
            return {
                "skip_section_generation": False,
                "error": str(e),
                # Preserve critical state
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    async def _format_response_node(self, state: ProposalState) -> Dict[str, Any]:
        """Format final response with proposal content and compliance report"""
        try:
            logger.info("Formatting proposal generation response...")
            
            sections = state.get("sections", {})
            compliance_matrix = state.get("compliance_matrix", [])
            completeness_score = state.get("completeness_score", 0.0)
            missing_requirements = state.get("missing_requirements", [])
            editor_operations = state.get("editor_operations", [])
            editing_mode = state.get("editing_mode", False)
            
            # Build proposal content from sections
            proposal_content = "# Proposal\n\n"
            
            if sections:
                for section_name, section in sections.items():
                    if hasattr(section, 'content'):
                        content = section.content
                    else:
                        content = str(section)
                    
                    section_title = section_name.replace('_', ' ').title()
                    proposal_content += f"## {section_title}\n\n{content}\n\n"
            
            # Build compliance report
            compliance_report = f"""
## Compliance Report

**Completeness Score:** {completeness_score:.1f}%
**Status:** {"Complete" if completeness_score >= 95 else "Mostly Complete" if completeness_score >= 75 else "Incomplete"}

**Requirements Coverage:**
- Addressed: {len([e for e in compliance_matrix if hasattr(e, 'status') and e.status == 'addressed'])} requirements
- Partial: {len([e for e in compliance_matrix if hasattr(e, 'status') and e.status == 'partial'])} requirements
- Missing: {len(missing_requirements)} requirements

**Critical Gaps:**
{chr(10).join([f"- {req_id}" for req_id in missing_requirements[:5]]) if missing_requirements else "None"}
"""
            
            # Build response
            response_text = proposal_content + compliance_report if not editing_mode else f"Prepared {len(editor_operations)} operations for proposal update"
            
            result = {
                "messages": [AIMessage(content=response_text)],
                "agent_results": {
                    "agent_type": "proposal_generation_agent",
                    "is_complete": True,
                    "proposal_content": proposal_content,
                    "completeness_score": completeness_score,
                    "compliance_matrix": compliance_matrix,
                    "missing_requirements": missing_requirements,
                    "editing_mode": editing_mode,
                    "editor_operations": editor_operations if editing_mode else None
                },
                "is_complete": True
            }
            
            logger.info("✅ Proposal generation complete")
            
            return {
                "response": result,
                "task_status": "complete",
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
        
        except Exception as e:
            logger.error(f"Failed to format response: {e}")
            return {
                "response": self._create_error_result(f"Failed to format response: {str(e)}"),
                "task_status": "error",
                "error": str(e),
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    async def process(
        self,
        query: str,
        metadata: Dict[str, Any] = None,
        messages: List[Any] = None
    ) -> Dict[str, Any]:
        """
        Process proposal generation request
        
        Implements checkpoint retention for efficiency:
        - Retains requirements, sections, and compliance data if references unchanged
        - Skips expensive LLM calls when iterating on existing proposal
        - Automatically starts fresh if chat thread is erased (conversation_id changes)
        
        Args:
            query: User query string
            metadata: Optional metadata dictionary
            messages: Optional conversation history
        
        Returns:
            Dictionary with proposal response and compliance details
        
        Note:
            If conversation_id changes (e.g., user starts new chat thread), checkpoint
            is lost and agent starts fresh. This is expected behavior - each conversation
            thread has its own checkpoint state.
        """
        try:
            logger.info(f"Proposal Generation Agent: Starting: {query[:100]}...")
            
            metadata = metadata or {}
            user_id = metadata.get("user_id", "system")
            shared_memory = metadata.get("shared_memory", {}) or {}
            
            # Prepare messages
            new_messages = self._prepare_messages_with_query(messages, query)
            
            # Get workflow and checkpoint config
            # Note: checkpoint is tied to conversation_id via thread_id
            # If conversation_id changes (new thread), checkpoint is lost and we start fresh
            workflow = await self._get_workflow()
            config = self._get_checkpoint_config(metadata)
            
            # Load and merge checkpointed messages
            conversation_messages = await self._load_and_merge_checkpoint_messages(
                workflow, config, new_messages
            )
            
            # Load existing state from checkpoint if available
            # If checkpoint doesn't exist (new thread), existing_state will be empty dict
            checkpoint_state = await workflow.aget_state(config)
            existing_state = {}
            existing_shared_memory = {}
            if checkpoint_state and checkpoint_state.values:
                existing_state = checkpoint_state.values
                existing_shared_memory = existing_state.get("shared_memory", {})
            
            shared_memory_merged = existing_shared_memory.copy()
            shared_memory_merged.update(shared_memory)
            
            # Get current reference document ID from frontmatter/metadata (proposal references pr-outline)
            active_editor = shared_memory.get("active_editor", {})
            frontmatter = active_editor.get("frontmatter", {})
            pr_outline_doc_id = frontmatter.get("pr_outline") or frontmatter.get("pr_outline_document_id") or metadata.get("pr_outline_document_id")
            
            current_refs = {
                "pr_outline": pr_outline_doc_id
            }
            
            # Check if references changed
            existing_refs = existing_state.get("reference_document_ids", {})
            references_changed = (
                current_refs.get("pr_outline") != existing_refs.get("pr_outline")
            )
            
            # Detect user intent
            user_intent = self._detect_user_intent(query)
            user_wants_regeneration = user_intent["user_wants_regeneration"]
            
            # Detect which specific sections user wants to generate
            requested_sections = self._detect_requested_sections(query)
            
            # Retain state if references unchanged, otherwise start fresh
            # Requirements: retain if RFI/RFP unchanged
            if not references_changed and existing_state.get("requirements") and existing_state.get("requirement_index"):
                retained_requirements = existing_state.get("requirements", [])
                retained_requirement_index = existing_state.get("requirement_index", {})
                retained_sections_needed = existing_state.get("sections_needed", [])
                logger.info(f"✅ Retaining {len(retained_requirements)} requirements from checkpoint")
            else:
                retained_requirements = []
                retained_requirement_index = {}
                retained_sections_needed = []
                if references_changed:
                    logger.info("References changed - will re-extract requirements")
            
            # Reference document content: retain if references unchanged
            # pr-outline and its cascaded references
            if not references_changed:
                retained_pr_outline_content = existing_state.get("pr_outline_content", "")
                retained_section_definitions = existing_state.get("section_definitions", {})
                retained_pr_req_content = existing_state.get("pr_req_content", "")
                retained_pr_style_content = existing_state.get("pr_style_content", "")
                retained_company_knowledge = existing_state.get("company_knowledge", "")
            else:
                retained_pr_outline_content = ""
                retained_section_definitions = {}
                retained_pr_req_content = ""
                retained_pr_style_content = ""
                retained_company_knowledge = ""
            
            # Sections: retain if user is iterating (not regenerating) and references unchanged
            if not user_wants_regeneration and not references_changed and existing_state.get("sections"):
                retained_sections = existing_state.get("sections", {})
                logger.info(f"✅ Retaining {len(retained_sections)} sections from checkpoint (user is iterating)")
            else:
                retained_sections = {}
                if user_wants_regeneration:
                    logger.info("User wants regeneration - will regenerate sections")
            
            # Compliance data: retain if sections unchanged
            if retained_sections and existing_state.get("compliance_matrix"):
                retained_compliance_matrix = existing_state.get("compliance_matrix", [])
                retained_missing_requirements = existing_state.get("missing_requirements", [])
                retained_completeness_score = existing_state.get("completeness_score", 0.0)
            else:
                retained_compliance_matrix = []
                retained_missing_requirements = []
                retained_completeness_score = 0.0
            
            # Customer name and proposal type: always from current frontmatter (may have changed)
            customer_name = frontmatter.get("customer_name") or metadata.get("customer_name", "Valued Customer")
            proposal_type = frontmatter.get("proposal_type") or metadata.get("proposal_type", "commercial_services")
            
            # Build initial state with retained values
            initial_state: ProposalState = {
                "query": query,
                "user_id": user_id,
                "metadata": metadata,
                "messages": conversation_messages,
                "shared_memory": shared_memory_merged,
                "customer_name": customer_name,
                "proposal_type": proposal_type,
                "reference_document_ids": current_refs,
                "pr_outline_content": retained_pr_outline_content,
                "section_definitions": retained_section_definitions,
                "pr_req_content": retained_pr_req_content,
                "pr_style_content": retained_pr_style_content,
                "company_knowledge": retained_company_knowledge,
                "requirements": retained_requirements,
                "requirement_index": retained_requirement_index,
                "sections_needed": retained_sections_needed,
                "requested_sections": requested_sections,
                "sections": retained_sections,
                "compliance_matrix": retained_compliance_matrix,
                "missing_requirements": retained_missing_requirements,
                "completeness_score": retained_completeness_score,
                "editor_content": active_editor.get("content", ""),
                "editing_mode": len(active_editor.get("content", "").strip()) > 0,
                "editor_operations": [],
                "user_wants_regeneration": user_wants_regeneration,
                "skip_requirement_analysis": len(retained_requirements) > 0 and not references_changed,
                "skip_section_generation": False,  # Will be set by _check_section_generation_node
                "response": {},
                "task_status": "",
                "error": ""
            }
            
            # Invoke workflow
            final_state = await workflow.ainvoke(initial_state, config=config)
            
            # Extract response
            response = final_state.get("response", {
                "messages": [AIMessage(content="Proposal generation failed")],
                "agent_results": {
                    "agent_type": "proposal_generation_agent",
                    "is_complete": False
                },
                "is_complete": False
            })
            
            return response
        
        except Exception as e:
            logger.error(f"Proposal Generation Agent ERROR: {e}")
            return self._create_error_result(f"Proposal generation failed: {str(e)}")
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result"""
        logger.error(f"Proposal Generation Agent error: {error_message}")
        return {
            "messages": [AIMessage(content=f"Proposal generation failed: {error_message}")],
            "agent_results": {
                "agent_type": "proposal_generation_agent",
                "success": False,
                "error": error_message,
                "is_complete": True
            },
            "is_complete": True
        }


_proposal_generation_agent_instance = None


def get_proposal_generation_agent() -> ProposalGenerationAgent:
    """Get global proposal generation agent instance"""
    global _proposal_generation_agent_instance
    if _proposal_generation_agent_instance is None:
        _proposal_generation_agent_instance = ProposalGenerationAgent()
    return _proposal_generation_agent_instance
