"""
Electronics Agent for LLM Orchestrator
Circuit design, embedded programming, and component selection agent
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional, Tuple, TypedDict
from datetime import datetime

from langgraph.graph import StateGraph, END
from .base_agent import BaseAgent, TaskStatus
from config.settings import settings
from orchestrator.tools.document_tools import (
    get_document_content_tool,
    search_documents_structured
)
from orchestrator.tools.web_tools import search_web_structured
from orchestrator.tools.file_creation_tools import (
    create_user_file_tool,
    create_user_folder_tool
)
from orchestrator.tools.document_editing_tools import (
    update_document_metadata_tool,
    update_document_content_tool,
    propose_document_edit_tool
)
from orchestrator.tools.document_tools import get_document_content_tool
from orchestrator.tools.project_content_tools import (
    save_or_update_project_content,
    determine_content_target,
    enrich_documents_with_metadata,
    check_if_new_file_needed,
    create_new_project_file,
    should_update_existing_section,
    propose_section_update,
    append_project_content,
    _is_placeholder_content
)
from orchestrator.tools.project_structure_tools import (
    execute_project_structure_plan
)
from orchestrator.tools.reference_file_loader import load_referenced_files
from orchestrator.utils.project_utils import sanitize_filename

# Import Pydantic models for structured outputs
from orchestrator.models.electronics_models import (
    FileRoutingPlan, FileRouteItem, ContentStructure, UnifiedContentPlan,
    QueryTypeAnalysis, ProjectPlanAnalysis, SearchNeedAnalysis,
    FollowUpAnalysis, ContentConflictAnalysis, ResponseQualityAnalysis,
    IncrementalUpdateAnalysis, ComponentCompatibilityAnalysis,
    ProjectDecision, DocumentationMaintenancePlan, DocumentationMaintenanceItem,
    DocumentationVerificationResult, DocumentationInconsistency
)

logger = logging.getLogger(__name__)


class ElectronicsState(TypedDict):
    """Simplified state for electronics agent LangGraph workflow"""
    query: str
    user_id: str
    metadata: Dict[str, Any]
    messages: List[Any]
    query_type: str
    query_intent: str  # "informational" or "action" - determined by LLM
    search_needed: bool  # Unified search flag (replaces should_search, has_explicit_search, web_search_needed)
    documents: List[Dict[str, Any]]
    segments: List[Dict[str, Any]]  # NEW: Relevant segments found within documents
    referenced_context: Dict[str, Any]  # Loaded referenced files from active editor
    information_needs: Optional[Dict[str, Any]]  # NEW: Analysis of what information is needed
    search_queries: List[Dict[str, Any]]  # NEW: Project-aware search queries
    search_quality_assessment: Optional[Dict[str, Any]]  # NEW: Quality assessment of search results
    search_retry_count: int  # Counter to prevent infinite re-search loops
    project_plan_action: Optional[str]  # "create", "open", "plan", or None
    project_structure_plan: Optional[Dict[str, Any]]  # LLM-generated plan for project files and structure
    response: Dict[str, Any]
    save_plan: Optional[Dict[str, Any]]  # Unified content extraction + routing plan (replaces content_structure + file_routing_plan)
    web_search_needed: bool  # Whether web search is needed
    web_search_explicit: bool  # Whether user explicitly requested web search
    # Web search permission removed - automatic web search enabled
    project_decisions: List[Dict[str, Any]]  # Track all project decisions made during conversation
    documentation_maintenance_plan: Optional[Dict[str, Any]]  # Plan for documentation maintenance operations
    documentation_verification_result: Optional[Dict[str, Any]]  # Result of documentation consistency verification
    pending_save_plan: Optional[Dict[str, Any]]  # Save plan waiting for user approval
    editing_mode: bool  # Whether project plan has content (editing mode vs generation mode)
    editor_operations: List[Dict[str, Any]]  # Editor operations for inline editing (project plan only)
    manuscript_edit: Optional[Dict[str, Any]]  # Manuscript edit metadata
    plan_edits: Optional[List[Dict[str, Any]]]  # Edits targeting project plan (for operation resolution)
    task_status: str
    error: str


class ElectronicsAgent(BaseAgent):
    """
    Electronics Intelligence Agent
    Provides circuit design, embedded programming assistance, and component selection
    """

    def __init__(self):
        super().__init__("electronics_agent")
        # Initialize node modules
        from orchestrator.agents.electronics_nodes.electronics_search_nodes import ElectronicsSearchNodes
        from orchestrator.agents.electronics_nodes.electronics_content_nodes import ElectronicsContentNodes
        from orchestrator.agents.electronics_nodes.electronics_save_nodes import ElectronicsSaveNodes
        from orchestrator.agents.electronics_nodes.electronics_decision_nodes import ElectronicsDecisionNodes
        from orchestrator.agents.electronics_nodes.electronics_project_plan_nodes import ElectronicsProjectPlanNodes
        from orchestrator.agents.electronics_nodes.electronics_maintenance_nodes import ElectronicsMaintenanceNodes
        self.search_nodes = ElectronicsSearchNodes(self)
        self.content_nodes = ElectronicsContentNodes(self)
        self.save_nodes = ElectronicsSaveNodes(self)
        self.decision_nodes = ElectronicsDecisionNodes(self)
        self.project_plan_nodes = ElectronicsProjectPlanNodes(self)
        self.maintenance_nodes = ElectronicsMaintenanceNodes(self)
        logger.info("🔌 Electronics Agent ready for circuit design and embedded programming!")
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """
        Simplified LangGraph workflow for electronics agent.
        
        Core flow: intent -> context -> search -> response -> save
        Reduced from 15 nodes to 8 nodes for better performance and maintainability.
        """
        workflow = StateGraph(ElectronicsState)
        
        # Core nodes (multi-phased search approach)
        workflow.add_node("analyze_intent", self._analyze_intent_node)  # Merged: detect_query_type + detect_follow_up + check_project_plan
        workflow.add_node("plan_execution_strategy", self._plan_execution_strategy_node)  # Strategic planning with documentation maintenance
        workflow.add_node("load_project_context", self._load_project_context_node)  # Merged: load_referenced_context + score_relevance + load_selective
        workflow.add_node("analyze_information_needs", self.search_nodes.analyze_information_needs_node)  # NEW: Analyze what information project needs
        workflow.add_node("generate_project_aware_queries", self.search_nodes.generate_project_aware_queries_node)  # NEW: Generate targeted queries using project context
        workflow.add_node("search_content", self.search_nodes.search_content_node)  # Refactored: Semantic segment search
        workflow.add_node("assess_result_quality", self.search_nodes.assess_result_quality_node)  # NEW: Evaluate if results answer query
        # Web search is automatic - no permission request needed
        workflow.add_node("perform_web_search", self._perform_web_search_node)  # Actual web search
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("extract_decisions", self.decision_nodes.extract_decisions_node)  # Extract project decisions from conversation
        workflow.add_node("update_project_plan", self.project_plan_nodes.update_project_plan_node)  # Proactively update project plan with summaries and high-level info
        workflow.add_node("verify_documentation", self.decision_nodes.verify_documentation_node)  # Verify documentation consistency
        workflow.add_node("execute_maintenance", self.maintenance_nodes.execute_maintenance_node)  # Execute documentation maintenance operations
        workflow.add_node("extract_and_route_content", self.content_nodes.extract_and_route_content_node)  # Merged: extract_content_structure + route_and_save_content
        workflow.add_node("save_content", self.save_nodes.save_content_node)
        workflow.add_node("resolve_plan_operations", self._resolve_plan_operations_node)  # NEW: Resolve plan edits to editor operations
        workflow.add_node("format_response", self._format_response_node)  # NEW: Format response with editor operations
        
        # Entry point
        workflow.set_entry_point("analyze_intent")
        
        # Intent analysis routes to execution planning, context loading, or direct response
        workflow.add_conditional_edges(
            "analyze_intent",
            self._route_from_intent,
            {
                "plan_execution": "plan_execution_strategy",
                "load_context": "load_project_context",
                "save": "extract_and_route_content",
                "maintenance": "execute_maintenance",
                "generate": "generate_response"
            }
        )
        
        # Execution planning routes to context loading
        workflow.add_conditional_edges(
            "plan_execution_strategy",
            self._route_from_execution_plan,
            {
                "load_context": "load_project_context"
            }
        )
        
        # Context loading routes to information needs analysis or response
        workflow.add_conditional_edges(
            "load_project_context",
            self._route_from_context,
            {
                "analyze_needs": "analyze_information_needs",
                "generate": "generate_response"
            }
        )
        
        # Information needs analysis routes to query generation
        workflow.add_edge("analyze_information_needs", "generate_project_aware_queries")
        
        # Query generation routes to search
        workflow.add_edge("generate_project_aware_queries", "search_content")
        
        # Search routes to quality assessment
        workflow.add_edge("search_content", "assess_result_quality")
        
        # Quality assessment routes to web search, re-search, or response generation
        workflow.add_conditional_edges(
            "assess_result_quality",
            self._route_from_quality_assessment,
            {
                # Web search is automatic - no permission request
                "perform_web_search": "perform_web_search",
                "re_search": "generate_project_aware_queries",  # Re-generate queries with feedback
                "generate": "generate_response"
            }
        )
        
        # Note: Search routing now handled by assess_result_quality node
        
        # Permission routing removed - web search is automatic
        
        # Web search goes to response generation
        workflow.add_edge("perform_web_search", "generate_response")
        
        # Response generation routes to decision extraction or save or end
        workflow.add_conditional_edges(
            "generate_response",
            self._route_from_response,
            {
                "extract_decisions": "extract_decisions",
                "save": "extract_and_route_content",
                "end": END
            }
        )
        
        # Decision extraction routes to project plan update, verification, or save
        workflow.add_conditional_edges(
            "extract_decisions",
            self._route_from_decisions,
            {
                "update_project_plan": "update_project_plan",
                "verify": "verify_documentation",
                "save": "extract_and_route_content",
                "end": END
            }
        )
        
        # Project plan update routes to verification or save
        workflow.add_conditional_edges(
            "update_project_plan",
            self._route_from_project_plan_update,
            {
                "verify": "verify_documentation",
                "save": "extract_and_route_content",
                "end": END
            }
        )
        
        # Verification routes to maintenance or save
        workflow.add_conditional_edges(
            "verify_documentation",
            self._route_from_verification,
            {
                "maintenance": "execute_maintenance",
                "save": "extract_and_route_content",
                "end": END
            }
        )
        
        # Maintenance execution goes to save
        workflow.add_edge("execute_maintenance", "extract_and_route_content")
        
        # Extract and route goes to save
        workflow.add_edge("extract_and_route_content", "save_content")
        
        # Save content routes conditionally: if editing mode with plan edits, resolve operations; otherwise format response
        workflow.add_conditional_edges(
            "save_content",
            self._route_from_save,
            {
                "resolve_operations": "resolve_plan_operations",
                "format": "format_response",
                "end": END
            }
        )
        
        # Resolve operations goes to format response
        workflow.add_edge("resolve_plan_operations", "format_response")
        
        # Format response goes to end
        workflow.add_edge("format_response", END)
        
        # No HITL interrupts needed - web search is automatic
        return workflow.compile(checkpointer=checkpointer)
    
    def _route_from_intent(self, state: ElectronicsState) -> str:
        """Route from intent analysis - determine if we need strategic planning"""
        query_type = state.get("query_type", "general")
        query = state.get("query", "")
        query_lower = query.lower()
        
        # Fast path: Simple conversational queries skip search pipeline
        if self._is_simple_query(query):
            logger.info("🔌 Simple conversational query detected - skipping search pipeline")
            return "generate"
        
        # Check if this is an approval response to a pending operation
        is_approval = self._is_approval_response(query)
        
        if is_approval:
            # Check if there's a pending save plan from previous turn (stored in state)
            pending_save_plan = state.get("pending_save_plan")
            if pending_save_plan:
                logger.info("🔌 User approved pending operation - restoring save plan and resuming")
                # Restore the save plan to state and route directly to save
                return "save"
            
            # Check if there's a pending maintenance plan
            maintenance_plan = state.get("documentation_maintenance_plan", {})
            maintenance_items = maintenance_plan.get("maintenance_items", [])
            if maintenance_items:
                logger.info("🔌 User approved pending maintenance - resuming maintenance execution")
                return "maintenance"
            
            # Check previous response to see if it was asking for permission
            response = state.get("response", {})
            response_text = response.get("response", "") if isinstance(response, dict) else str(response)
            if "would you like" in response_text.lower() or "update" in response_text.lower():
                # Previous response was asking permission - check if we have a save_plan ready
                save_plan = state.get("save_plan")
                if save_plan and save_plan.get("routing"):
                    logger.info("🔌 User approved - proceeding with save operation")
                    return "save"
        
        # Use strategic planning for complex queries that might need documentation maintenance
        needs_planning = any(keyword in query for keyword in [
            "replace", "change", "update", "switch", "instead of", "remove", "delete",
            "component", "design", "architecture", "specification"
        ]) or query_type in ["component_selection", "circuit_design"]
        
        if needs_planning:
            logger.info("🔌 Complex query detected - routing to execution planning")
            return "plan_execution"
        
        return "load_context"
    
    def _route_from_execution_plan(self, state: ElectronicsState) -> str:
        """Route from execution planning - always load context (project creation handled by UI)"""
        return "load_context"
    
    def _route_from_project_plan_update(self, state: ElectronicsState) -> str:
        """Route from project plan update node"""
        # After updating project plan, check if verification is needed
        project_decisions = state.get("project_decisions", [])
        if project_decisions:
            return "verify"
        return "save"
    
    def _route_from_decisions(self, state: ElectronicsState) -> str:
        """Route from decision extraction - update project plan if relevant, then verify or save"""
        project_decisions = state.get("project_decisions", [])
        metadata = state.get("metadata", {})
        active_editor = metadata.get("active_editor") or metadata.get("shared_memory", {}).get("active_editor", {})
        has_project_plan = bool(active_editor.get("document_id"))
        
        # Update project plan if we have decisions or if project plan exists and conversation has meaningful content
        if has_project_plan:
            query = state.get("query", "")
            response = state.get("response", {})
            response_text = response.get("response", "") if isinstance(response, dict) else str(response)
            messages = state.get("messages", [])
            
            # Update project plan if:
            # 1. We have decisions to summarize
            # 2. OR we have substantial conversation content (especially early in project)
            # 3. OR query suggests high-level project information
            should_update = (
                project_decisions or
                (len(response_text) > 300 and len(messages) <= 10) or  # Early in project with substantial content
                any(keyword in query.lower() for keyword in ["project", "overview", "summary", "plan", "architecture", "design", "system"])
            )
            
            if should_update:
                logger.info(f"🔌 Routing to update_project_plan (decisions: {len(project_decisions)}, early_project: {len(messages) <= 10})")
                return "update_project_plan"
        
        # If we have decisions but no project plan, verify documentation
        if project_decisions:
            logger.info(f"🔌 {len(project_decisions)} decision(s) extracted - routing to verification")
            return "verify"
        
        # Check if we should save even without decisions
        response = state.get("response", {})
        response_text = response.get("response", "") if isinstance(response, dict) else str(response)
        if len(response_text) > 200:
            return "save"
        
        return "end"
    
    def _route_from_verification(self, state: ElectronicsState) -> str:
        """Route from verification - execute maintenance if inconsistencies found"""
        verification_result = state.get("documentation_verification_result", {})
        inconsistencies = verification_result.get("inconsistencies", [])
        maintenance_plan = state.get("documentation_maintenance_plan", {})
        maintenance_items = maintenance_plan.get("maintenance_items", [])
        
        if inconsistencies or maintenance_items:
            logger.info(f"🔌 Documentation issues found - routing to maintenance")
            return "maintenance"
        
        # Check if we should save
        response = state.get("response", {})
        response_text = response.get("response", "") if isinstance(response, dict) else str(response)
        if len(response_text) > 200:
            return "save"
        
        return "end"
    
    
    def _route_from_context(self, state: ElectronicsState) -> str:
        """Route based on whether search is needed after loading context"""
        if state.get("search_needed", False):
            # Check if information needs were already analyzed in intent node
            if state.get("information_needs_analyzed", False):
                logger.info("🔌 Information needs already analyzed - skipping to query generation")
                return "generate_project_aware_queries"
            else:
                logger.info("🔌 Routing to analyze_information_needs")
                return "analyze_needs"
        else:
            logger.info("🔌 Routing directly to generate_response")
            return "generate"
    
    def _route_from_quality_assessment(self, state: ElectronicsState) -> str:
        """Route based on result quality assessment with context-aware thresholds"""
        quality_result = state.get("search_quality_assessment", {})
        quality_score = quality_result.get("quality_score", 0.0)
        needs_web_search = quality_result.get("needs_web_search", False)
        should_re_search = quality_result.get("should_re_search", False)
        web_search_explicit = state.get("web_search_explicit", False)
        search_retry_count = state.get("search_retry_count", 0)
        max_search_retries = 3  # Maximum number of re-search attempts
        
        # Check for verification keywords in query
        query = state.get("query", "").lower()
        verification_keywords = [
            "double check", "verify", "confirm", "check if", "is this correct",
            "is this right", "validate", "fact check", "cross reference"
        ]
        is_verification_query = any(keyword in query for keyword in verification_keywords)
        
        # Context-aware threshold: Simple queries need lower threshold
        # Simple queries are short and don't require extensive verification
        is_simple_query = self._is_simple_query(query) or len(query.split()) <= 5
        quality_threshold = 0.65 if is_simple_query else 0.75  # Lower threshold for simple queries
        
        # Prevent infinite re-search loops
        if should_re_search and quality_score < 0.5:
            if search_retry_count >= max_search_retries:
                logger.warning(f"🔌 Max search retries ({max_search_retries}) reached (score: {quality_score:.2f}) - routing to web search")
                return "perform_web_search"
            logger.info(f"🔌 Low quality results (score: {quality_score:.2f}, retry {search_retry_count + 1}/{max_search_retries}) - re-searching with refined queries")
            return "re_search"
        elif needs_web_search or web_search_explicit:
            # Web search is automatic - no permission required
            logger.info("🔌 Web search needed - routing directly to web search")
            return "perform_web_search"
        elif is_verification_query and quality_score < 0.7:
            # For verification queries, use web search if quality is below threshold
            logger.info(f"🔌 Verification query with moderate quality (score: {quality_score:.2f}) - routing to web search")
            return "perform_web_search"
        elif quality_score >= quality_threshold:
            logger.info(f"🔌 Quality results (score: {quality_score:.2f}, threshold: {quality_threshold:.2f}) - routing to response generation")
            return "generate"
        else:
            # Quality below threshold - consider web search for complex queries, proceed for simple
            if not is_simple_query and quality_score < 0.65:
                logger.info(f"🔌 Moderate quality for complex query (score: {quality_score:.2f}) - routing to web search")
                return "perform_web_search"
            else:
                logger.info(f"🔌 Moderate quality results (score: {quality_score:.2f}) - routing to response generation")
                return "generate"
    
    def _route_from_save(self, state: ElectronicsState) -> str:
        """Route from save_content: resolve operations if editing mode with plan edits, otherwise format response"""
        editing_mode = state.get("editing_mode", False)
        plan_edits = state.get("plan_edits")
        
        if editing_mode and plan_edits and len(plan_edits) > 0:
            return "resolve_operations"
        elif editing_mode or plan_edits:
            # Has operations or editing mode, format response
            return "format"
        else:
            # No operations, end
            return "end"
    
    def _route_from_response(self, state: ElectronicsState) -> str:
        """Route based on whether content should be saved or decisions extracted"""
        # Check if response has substantial content that should be saved
        response = state.get("response", {})
        response_text = response.get("response", "") if isinstance(response, dict) else str(response)
        
        # Check if this is a project creation or update request
        query = state.get("query", "").lower()
        project_plan_action = state.get("project_plan_action")
        
        # Check for change indicators that suggest corrections/revisions
        change_indicators = [
            'instead of', 'replace', 'switch from', 'change', 'actually',
            'wrong', 'should be', 'not anymore', 'remove', 'delete',
            'update', 'correct', 'no longer',
            # Wrong path indicators
            'not that way', 'different approach', 'actually no', "that won't work",
            'that doesn\'t work', 'not going that route', 'wrong direction',
            # Conceptual changes
            'changed my mind', 'better to', 'prefer', 'rather',
            'not using', 'abandon', 'scrap', 'forget about',
            # Correction phrases
            'correction', 'fix', 'mistake', 'error', 'incorrect',
            'not correct', 'that\'s wrong', 'that is wrong',
            # Direction changes
            'going with', 'switching to', 'moving to', 'changing to',
            'revised', 'revision', 'updated approach'
        ]
        has_corrections = any(indicator in query for indicator in change_indicators)
        
        # Check if query involves decisions (component selection, design choices, etc.)
        involves_decisions = any(keyword in query for keyword in [
            "use", "select", "choose", "decide", "replace", "change", "switch",
            "instead of", "component", "design", "architecture"
        ]) or state.get("query_type", "") in ["component_selection", "circuit_design"]
        
        # Use LLM's query_intent decision (set during analyze_intent_node)
        query_intent = state.get("query_intent", "action")  # Default to action for safety
        
        # Should save if:
        # 1. Project was just created (project_plan_action == "create")
        # 2. LLM classified query as "action" (not "informational")
        # 3. Response has substantial content (>200 chars) for action queries
        should_save = False
        
        if project_plan_action == "create":
            should_save = True
            logger.info("🔌 New project created - will save content")
        elif query_intent == "informational":
            should_save = False
            logger.info("🔌 LLM classified as informational query - will not save content")
        elif query_intent == "action" and len(response_text) > 200:
            should_save = True
            logger.info(f"🔌 LLM classified as action query with substantial content ({len(response_text)} chars) - will save")
        elif query_intent == "action":
            should_save = True
            logger.info("🔌 LLM classified as action query - will save content")
        
        # Route to extract_decisions (which leads to verification) when:
        # 1. Corrections are detected (to ensure cleanup happens)
        # 2. Query involves decision-making
        if should_save and (has_corrections or involves_decisions):
            if has_corrections:
                logger.info("🔌 Corrections detected - routing to extract_decisions for verification")
            else:
                logger.info("🔌 Query involves decisions - routing to extract_decisions")
            return "extract_decisions"
        elif should_save:
            logger.info("🔌 Routing to extract_and_route_content")
            return "save"
        else:
            logger.info("🔌 No content to save - ending workflow")
            return "end"
    
    # Legacy routing functions removed - no longer used in workflow
    
    # ============================================
    # NEW SIMPLIFIED NODE IMPLEMENTATIONS
    # ============================================
    
    async def _analyze_intent_node(self, state: ElectronicsState) -> Dict[str, Any]:
        """
        Merged node: detect_query_type + detect_follow_up + check_project_plan
        Analyzes query intent, detects follow-ups, and determines project plan needs.
        Also checks if query is related to the active project.
        """
        query = state.get("query", "")
        user_id = state.get("user_id", "")
        metadata = state.get("metadata", {})
        messages = state.get("messages", [])
        
        # Early exit optimization: If we have a pending operation, skip LLM call
        pending_save_plan = state.get("pending_save_plan")
        maintenance_plan = state.get("documentation_maintenance_plan", {})
        maintenance_items = maintenance_plan.get("maintenance_items", [])
        
        is_approval = self._is_approval_response(query)
        if is_approval and (pending_save_plan or maintenance_items):
            # Skip LLM call - we already know what to do
            shared_memory = metadata.get("shared_memory", {})
            active_editor = metadata.get("active_editor") or shared_memory.get("active_editor", {})
            has_project_plan = bool(active_editor and active_editor.get("frontmatter", {}).get("type") == "electronics")
            return {
                "query_type": "general",
                "search_needed": False,
                "project_plan_action": None,
                "query_is_project_related": has_project_plan
            }
        
        try:
            # Use LLM to analyze query type, follow-up status, and project plan needs
            fast_model = self._get_fast_model(state)
            llm = self._get_llm(temperature=0.1, model=fast_model, state=state)
            
            # Check if project plan is open
            shared_memory = metadata.get("shared_memory", {})
            active_editor = metadata.get("active_editor") or shared_memory.get("active_editor", {})
            has_project_plan = bool(active_editor and active_editor.get("frontmatter", {}).get("type") == "electronics")
            
            # Get project title/description for relevance check
            project_title = ""
            project_description = ""
            project_components = []
            project_protocols = []
            if active_editor and has_project_plan:
                frontmatter = active_editor.get("frontmatter", {})
                project_title = frontmatter.get("title", "")
                project_description = frontmatter.get("description", "")
                project_components = frontmatter.get("components", [])[:10]  # Top 10
                project_protocols = frontmatter.get("protocols", [])[:10]  # Top 10
            
            # Optimization: If we have project context, combine intent + information needs analysis
            has_project_context = bool(project_components or project_protocols)
            
            if has_project_context and has_project_plan:
                # Combined analysis: intent + information needs in one call
                context_parts = []
                if project_components:
                    context_parts.append(f"- Components: {', '.join(str(c) for c in project_components)}")
                if project_protocols:
                    context_parts.append(f"- Protocols: {', '.join(str(p) for p in project_protocols)}")
                context_string = '\n'.join(context_parts) if context_parts else "None specified"
                
                prompt = f"""Analyze this electronics query and determine what information is needed:

**QUERY**: {query}

**PROJECT CONTEXT**:
- Project title: {project_title}
- Project description: {project_description[:200] if project_description else "N/A"}
{context_string}
- Previous messages: {len(messages) - 1} messages in conversation

**ANALYSIS NEEDED**:
1. **query_type**: circuit_design|embedded_code|component_selection|calculation|troubleshooting|general
2. **is_follow_up**: Is this a follow-up to previous conversation?
3. **project_plan_action**: "create" if new project, null otherwise
4. **search_needed**: Does this query need semantic search?
5. **information_gaps**: What specific information gaps exist? (list of strings)
6. **relevant_components**: Which project components are relevant? (list from project context)
7. **relevant_protocols**: Which project protocols are relevant? (list from project context)
8. **content_type**: "new"|"update"|"both" - Is this for new content or updating existing?
9. **detail_level**: "overview"|"detailed"|"implementation" - What level of detail is needed?
10. **query_intent**: "informational"|"action" - Is this asking for information/status/explanation (informational) or requesting changes/additions/creation (action)?

Examples:
- "What's the project status?" → informational
- "Explain how the circuit works" → informational
- "Show me the component list" → informational
- "Add a voltage regulator" → action
- "Update the power requirements" → action
- "Create a new schematic" → action

Return ONLY valid JSON:
{{
  "query_type": "circuit_design",
  "is_follow_up": false,
  "project_plan_action": null,
  "search_needed": true,
  "information_gaps": ["gap1", "gap2"],
  "relevant_components": ["component1"],
  "relevant_protocols": ["protocol1"],
  "content_type": "new",
  "detail_level": "detailed",
  "query_intent": "action",
  "reasoning": "Brief explanation"
}}"""
            else:
                # Standard intent analysis only (no project context available)
                prompt = f"""Analyze this electronics query:

**QUERY**: {query}

**CONTEXT**:
- Has project plan open: {has_project_plan}
- Project title: {project_title}
- Project description: {project_description[:200] if project_description else "N/A"}
- Previous messages: {len(messages) - 1} messages in conversation

**ANALYSIS NEEDED**:
1. **query_type**: circuit_design|embedded_code|component_selection|calculation|troubleshooting|general
2. **is_follow_up**: Is this a follow-up to previous conversation?
3. **project_plan_action**: "create" if new project, null otherwise
4. **search_needed**: Does this query need semantic search?
5. **query_intent**: "informational"|"action" - Is this asking for information/status/explanation (informational) or requesting changes/additions/creation (action)?

Examples:
- "What's the project status?" → informational
- "Explain how the circuit works" → informational
- "Show me the component list" → informational
- "Add a voltage regulator" → action
- "Update the power requirements" → action
- "Create a new schematic" → action

**NOTE**: All queries routed to electronics_agent are project-oriented by definition.

Return ONLY valid JSON:
{{
  "query_type": "circuit_design",
  "is_follow_up": false,
  "project_plan_action": null,
  "search_needed": true,
  "query_intent": "action",
  "reasoning": "Brief explanation"
}}"""
            
            # Build structured output schema based on whether we have project context
            if has_project_context and has_project_plan:
                schema = {
                    "type": "object",
                    "properties": {
                        "query_type": {"type": "string"},
                        "is_follow_up": {"type": "boolean"},
                        "project_plan_action": {"type": ["string", "null"]},
                        "search_needed": {"type": "boolean"},
                        "information_gaps": {"type": "array", "items": {"type": "string"}},
                        "relevant_components": {"type": "array", "items": {"type": "string"}},
                        "relevant_protocols": {"type": "array", "items": {"type": "string"}},
                        "content_type": {"type": "string"},
                        "detail_level": {"type": "string"},
                        "query_intent": {"type": "string"},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["query_type", "is_follow_up", "project_plan_action", "search_needed", "query_intent"]
                }
            else:
                schema = {
                    "type": "object",
                    "properties": {
                        "query_type": {"type": "string"},
                        "is_follow_up": {"type": "boolean"},
                        "project_plan_action": {"type": ["string", "null"]},
                        "search_needed": {"type": "boolean"},
                        "query_intent": {"type": "string"},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["query_type", "is_follow_up", "project_plan_action", "search_needed", "query_intent"]
                }
            
            try:
                structured_llm = llm.with_structured_output(schema)
                result = await structured_llm.ainvoke([{"role": "user", "content": prompt}])
                result_dict = result if isinstance(result, dict) else result.dict() if hasattr(result, 'dict') else result.model_dump()
            except Exception:
                # Fallback to manual parsing
                response = await llm.ainvoke([{"role": "user", "content": prompt}])
                content = response.content if hasattr(response, 'content') else str(response)
                result_dict = self._extract_json_from_response(content) or {}
            
            # ALL queries routed to electronics_agent are project-oriented by definition
            # If we have an active electronics editor, the query is definitely project-related
            query_is_project_related = has_project_plan
            
            # Build return dict with optional information needs data
            return_dict = {
                "query_type": result_dict.get("query_type", "general"),
                "search_needed": result_dict.get("search_needed", False),
                "project_plan_action": result_dict.get("project_plan_action"),
                "query_is_project_related": query_is_project_related,
                "query_intent": result_dict.get("query_intent", "action")  # Default to action for safety
            }
            
            # If we did combined analysis, include information needs to skip separate node
            if has_project_context and has_project_plan and result_dict.get("information_gaps") is not None:
                return_dict["information_needs"] = {
                    "information_gaps": result_dict.get("information_gaps", []),
                    "relevant_components": result_dict.get("relevant_components", []),
                    "relevant_protocols": result_dict.get("relevant_protocols", []),
                    "content_type": result_dict.get("content_type", "new"),
                    "detail_level": result_dict.get("detail_level", "detailed"),
                    "related_sections": [],
                    "search_focus": result_dict.get("reasoning", query)
                }
                return_dict["information_needs_analyzed"] = True  # Flag to skip separate node
            
            return return_dict
            
        except Exception as e:
            logger.error(f"❌ Intent analysis failed: {e}")
            # Fallback: if project plan is open, query is project-related
            shared_memory = metadata.get("shared_memory", {})
            active_editor = metadata.get("active_editor") or shared_memory.get("active_editor", {})
            has_project_plan = bool(active_editor and active_editor.get("frontmatter", {}).get("type") == "electronics")
            
            return {
                "query_type": "general",
                "search_needed": False,
                "project_plan_action": None,
                "query_is_project_related": has_project_plan
            }
    
    async def _load_project_context_node(self, state: ElectronicsState) -> Dict[str, Any]:
        """
        Merged node: load_referenced_context + score_file_relevance + load_selective_content
        Loads project context efficiently with lazy loading.
        Only loads if query is project-related and editor_preference is not "ignore".
        """
        try:
            user_id = state.get("user_id", "")
            metadata = state.get("metadata", {})
            shared_memory = metadata.get("shared_memory", {})
            
            # Check editor_preference - if "ignore", skip project context
            editor_preference = shared_memory.get("editor_preference", "prefer")
            if editor_preference == "ignore":
                logger.info(f"🔌 Skipping project context - editor_preference is 'ignore'")
                return {
                    "referenced_context": {}
                }
            
            # ALL queries routed to electronics_agent are project-oriented by definition
            # Always load context if we have an active electronics editor
            
            # Check both metadata.active_editor and shared_memory.active_editor
            active_editor = metadata.get("active_editor") or shared_memory.get("active_editor", {})
            
            # Only load if we have an active editor with electronics type
            if not active_editor or active_editor.get("frontmatter", {}).get("type", "").lower() != "electronics":
                logger.info(f"🔌 Skipping project context - no electronics project plan open")
                return {
                    "referenced_context": {}
                }
            
            # Load referenced files from frontmatter
            from orchestrator.tools.reference_file_loader import load_referenced_files
            
            # Electronics reference configuration
            reference_config = {
                "components": ["components", "component", "component_docs"],
                "protocols": ["protocols", "protocol", "protocol_docs"],
                "schematics": ["schematics", "schematic", "schematic_docs"],
                "specifications": ["specifications", "spec", "specs", "specification"],
                "other": ["references", "reference", "docs", "documents", "related", "files"]
            }
            
            result = await load_referenced_files(
                active_editor=active_editor,
                user_id=user_id,
                reference_config=reference_config,
                doc_type_filter="electronics"
            )
            
            referenced_context = result.get("loaded_files", {})
            
            logger.info(f"🔌 Loaded {len(referenced_context)} referenced files")
            
            # Detect editing mode: check if project plan has content after frontmatter
            editor_content = active_editor.get("content", "")
            editing_mode = False
            if editor_content:
                frontmatter_end = self._get_frontmatter_end(editor_content)
                has_content_after_frontmatter = len(editor_content) > frontmatter_end + 10
                editing_mode = has_content_after_frontmatter
            
            logger.info(f"🔌 Electronics agent mode: {'EDITING' if editing_mode else 'GENERATION'}")
            
            return {
                "referenced_context": referenced_context,
                "editing_mode": editing_mode,
                "editor_operations": [],
                "plan_edits": None
            }
            
        except Exception as e:
            logger.error(f"❌ Context loading failed: {e}")
            return {
                "referenced_context": {}
            }
    
    def _get_frontmatter_end(self, content: str) -> int:
        """Find frontmatter end position"""
        import re
        match = re.match(r'^---\s*\n[\s\S]*?\n---\s*\n', content)
        return match.end() if match else 0
    
    def _find_section_in_content(self, content: str, section_name: str) -> tuple:
        """Find section boundaries in markdown content. Returns (start, end) or (0, 0) if not found."""
        import re
        
        # Try to find section by heading (## Section Name or # Section Name)
        patterns = [
            rf'^##\s+{re.escape(section_name)}\s*$',  # Exact match with ##
            rf'^#\s+{re.escape(section_name)}\s*$',   # Exact match with #
            rf'^##\s+.*{re.escape(section_name)}.*$',  # Contains section name with ##
            rf'^#\s+.*{re.escape(section_name)}.*$',  # Contains section name with #
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
            if match:
                section_start = match.start()
                # Find the end of this section (next heading of same or higher level, or end of document)
                section_end = len(content)
                
                # Look for next heading at same or higher level
                next_heading_pattern = re.compile(r'^#{1,2}\s+', re.MULTILINE)
                next_match = next_heading_pattern.search(content, match.end())
                if next_match:
                    section_end = next_match.start()
                
                return (section_start, section_end)
        
        return (0, 0)
    
    def _convert_section_edit_to_operation(self, edit: Dict[str, Any], content: str, frontmatter_end: int) -> Optional[Dict[str, Any]]:
        """Convert a section-based edit to a position-based editor operation"""
        section = edit.get("section", "")
        action = edit.get("action", "append")
        edit_content = edit.get("content", "")
        
        if not section:
            return None
        
        # Find section boundaries
        section_start, section_end = self._find_section_in_content(content, section)
        
        if section_start == 0 and section_end == 0:
            # Section not found - skip
            logger.warning(f"Section '{section}' not found in content - skipping")
            return None
        
        # Build operation based on action
        if action == "replace":
            # Replace entire section
            original_text = content[section_start:section_end].strip()
            return {
                "op_type": "replace_range",
                "start": section_start,
                "end": section_end,
                "text": edit_content,
                "original_text": original_text[:200] if len(original_text) > 200 else original_text,  # First 200 chars for matching
                "left_context": content[max(0, section_start-50):section_start],
                "right_context": content[section_end:min(len(content), section_end+50)]
            }
        elif action == "remove":
            # Delete section
            original_text = content[section_start:section_end].strip()
            return {
                "op_type": "delete_range",
                "start": section_start,
                "end": section_end,
                "text": "",
                "original_text": original_text[:200] if len(original_text) > 200 else original_text,
                "left_context": content[max(0, section_start-50):section_start],
                "right_context": content[section_end:min(len(content), section_end+50)]
            }
        elif action == "append":
            # Insert after section
            # Find end of section (after last newline)
            section_end_line = content.rfind("\n", section_start, section_end)
            if section_end_line == -1:
                anchor_pos = section_end
            else:
                anchor_pos = section_end_line + 1
            
            anchor_text = content[max(0, anchor_pos-100):anchor_pos].strip()
            return {
                "op_type": "insert_after_heading",
                "start": anchor_pos,
                "end": anchor_pos,
                "text": edit_content,
                "anchor_text": anchor_text[-50:] if len(anchor_text) > 50 else anchor_text,  # Last 50 chars for matching
                "left_context": content[max(0, anchor_pos-50):anchor_pos],
                "right_context": content[anchor_pos:min(len(content), anchor_pos+50)]
            }
        
        return None
    
    def _resolve_operation_simple(
        self,
        content: str,
        op_dict: Dict[str, Any],
        frontmatter_end: int = 0
    ) -> Tuple[int, int, str, float]:
        """Resolve operation to exact positions using progressive search (copied from fiction agent)"""
        from typing import Tuple
        import re
        
        op_type = op_dict.get("op_type", "replace_range")
        original_text = op_dict.get("original_text")
        anchor_text = op_dict.get("anchor_text")
        occurrence_index = op_dict.get("occurrence_index", 0)
        text = op_dict.get("text", "")
        
        # Strategy 1: Exact match with original_text
        if original_text and op_type in ("replace_range", "delete_range"):
            count = 0
            search_from = 0
            while True:
                pos = content.find(original_text, search_from)
                if pos == -1:
                    break
                if count == occurrence_index:
                    end_pos = pos + len(original_text)
                    # Guard frontmatter
                    pos = max(pos, frontmatter_end)
                    end_pos = max(end_pos, pos)
                    return pos, end_pos, text, 1.0
                count += 1
                search_from = pos + 1
        
        # Strategy 2: Anchor text for insert_after_heading
        if anchor_text and op_type == "insert_after_heading":
            pos = content.find(anchor_text)
            if pos != -1:
                # Find end of line/paragraph
                end_pos = content.find("\n", pos)
                if end_pos == -1:
                    end_pos = len(content)
                else:
                    end_pos += 1
                # Guard frontmatter
                end_pos = max(end_pos, frontmatter_end)
                return end_pos, end_pos, text, 0.9
        
        # Fallback: use approximate positions from op_dict
        start = op_dict.get("start", 0)
        end = op_dict.get("end", 0)
        
        # Guard frontmatter
        start = max(start, frontmatter_end)
        end = max(end, start)
        
        return start, end, text, 0.5
    
    async def _resolve_plan_operations_node(self, state: ElectronicsState) -> Dict[str, Any]:
        """Resolve plan edits to position-based editor operations"""
        try:
            logger.info("📝 Resolving plan edits to editor operations...")
            
            plan_edits = state.get("plan_edits", [])
            if not plan_edits:
                return {
                    "editor_operations": [],
                    "task_status": "complete"
                }
            
            # Get editor content
            metadata = state.get("metadata", {})
            active_editor = metadata.get("active_editor") or metadata.get("shared_memory", {}).get("active_editor", {})
            editor_content = active_editor.get("content", "")
            
            if not editor_content:
                logger.warning("No editor content available for operation resolution")
                return {
                    "editor_operations": [],
                    "task_status": "error",
                    "error": "No editor content available"
                }
            
            frontmatter_end = self._get_frontmatter_end(editor_content)
            editor_operations = []
            
            for edit in plan_edits:
                # Convert section edit to operation
                operation = self._convert_section_edit_to_operation(edit, editor_content, frontmatter_end)
                if not operation:
                    continue
                
                # Resolve exact positions
                try:
                    resolved_start, resolved_end, resolved_text, resolved_confidence = self._resolve_operation_simple(
                        editor_content,
                        operation,
                        frontmatter_end=frontmatter_end
                    )
                    
                    # Calculate pre_hash for optimistic concurrency
                    import hashlib
                    pre_slice = editor_content[resolved_start:resolved_end]
                    pre_hash = hashlib.sha256(pre_slice.encode()).hexdigest()[:16]
                    
                    # Build final operation
                    final_op = {
                        "op_type": operation["op_type"],
                        "from": resolved_start,
                        "to": resolved_end,
                        "text": resolved_text,
                        "pre_hash": pre_hash
                    }
                    
                    editor_operations.append(final_op)
                    logger.info(f"✅ Resolved {operation['op_type']} [{resolved_start}:{resolved_end}] confidence={resolved_confidence:.2f}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to resolve operation: {e}")
                    continue
            
            logger.info(f"✅ Resolved {len(editor_operations)} plan operations")
            
            return {
                "editor_operations": editor_operations,
                "task_status": "complete"
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to resolve plan operations: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "editor_operations": [],
                "task_status": "error",
                "error": str(e)
            }
    
    async def _format_response_node(self, state: ElectronicsState) -> Dict[str, Any]:
        """Format final response with editor operations if in editing mode"""
        try:
            editing_mode = state.get("editing_mode", False)
            editor_operations = state.get("editor_operations", [])
            saved_files = state.get("saved_files", [])
            
            # Get existing response from generate_response node
            response = state.get("response", {})
            response_text = response.get("response", "") if isinstance(response, dict) else str(response)
            
            if editing_mode and editor_operations:
                # Editing mode: include operations in response
                response_text = "I've generated edits for the project plan. Review and accept the changes below.\n\n" + response_text
                
                response = {
                    "response": response_text,
                    "editor_operations": editor_operations,
                    "manuscript_edit": {
                        "target_filename": state.get("metadata", {}).get("active_editor", {}).get("filename", "project_plan.md"),
                        "scope": "section",
                        "summary": f"Updated {len(editor_operations)} section(s) in project plan"
                    },
                    "saved_files": saved_files,
                    "task_status": "complete"
                }
            else:
                # Generation mode or no operations: standard response
                if saved_files:
                    files_list = ", ".join(saved_files)
                    response_text = f"I've updated the project files: {files_list}\n\n" + response_text
                
                response = {
                    "response": response_text,
                    "saved_files": saved_files,
                    "task_status": "complete"
                }
            
            return {
                "response": response,
                "task_status": "complete"
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to format response: {e}")
            return {
                "response": {"response": "Error formatting response", "task_status": "error"},
                "task_status": "error",
                "error": str(e)
            }

    def _build_electronics_prompt(self) -> str:
        """Build system prompt for electronics queries"""
        return """You are an Electronics Engineering Agent - an expert in circuit design and embedded systems!

**BEST PRACTICES FOR ELECTRONICS PROJECT MANAGEMENT**:

**PROJECT STRUCTURE HIERARCHY**:
- **PROJECT PLAN = SOURCE OF TRUTH**: The project plan (project_plan.md) is the authoritative document containing:
  - Overarching system requirements and specifications
  - High-level system architecture and block diagrams
  - System design overview and processes
  - Project goals, scope, and constraints
  - Integration points and system-level decisions
  - References to detailed files (components, protocols, schematics, etc.)
  
- **REFERENCED FILES = SPECIFIC DETAILS**: Referenced files (component_list.md, protocol files, schematics, etc.) contain:
  - Specific component specifications and selections
  - Detailed protocol implementations
  - Circuit schematics and wiring diagrams
  - Detailed technical specifications
  - Code implementations and firmware
  - Specific aspects relevant to the project plan

**ROUTING RULES**:
- **System-level/overarching content** -> Project Plan (architecture, requirements, processes, high-level design)
- **Specific detailed content** -> Referenced Files (components, protocols, schematics, code, detailed specs)
- **Unclear/fallback** -> Project Plan (when in doubt, use the source of truth)

1. **ALWAYS USE EXISTING PROJECT FILES**: When you have referenced project files (components, protocols, schematics, specifications), ALWAYS prefer updating those existing files over suggesting new ones. Only suggest a new file if the content truly does not fit in any existing file.

2. **ALWAYS UPDATE FILES BASED ON USER INPUT**: When the user provides information, feedback, or asks questions, automatically update the appropriate project files with that information. Do not wait for explicit permission - intelligently route content to the right files based on the hierarchy above.
   - **DETECT REVISIONS**: When user says 'instead of X, use Y' or 'change X to Y', this means REPLACE existing content, not add new content
   - **UPDATE EXISTING SECTIONS**: If a section already exists with similar content, UPDATE it rather than creating duplicates
   - **COMPONENT REPLACEMENTS**: When components are replaced, update the existing component section, do not create new sections

3. **STAY CURRENT WITH PROJECT FILES**: You always have access to the latest content from referenced project files AND the project plan. Use this information to provide context-aware responses and ensure you're building on what's already documented. The project plan provides the big picture; referenced files provide the details.

4. **INTELLIGENT CONTENT ROUTING**: Route content to the most appropriate file based on:
   - **Content scope**: System-level/overarching -> project plan; Specific details -> referenced files
   - Content type (components -> component_list.md, protocols -> protocol files, etc.)
   - File titles and descriptions
   - Existing file structure
   - User's project organization

5. **CONSERVATIVE FILE CREATION**: Only suggest creating new files when:
   - Content is very substantial (>1500 chars)
   - Content is about a distinct, specific topic
   - NO existing file of that type exists OR existing files are a very poor match (<20%)
   - User explicitly requests a new file

6. **PROACTIVE UPDATES**: When you provide information, calculations, recommendations, or answers, automatically save/update the appropriate project files. The user shouldn't have to ask you to save information - you should do it intelligently, respecting the project plan as source of truth and referenced files for specific details.

**MISSION**: Provide practical electronics design assistance, embedded programming guidance, and component selection for DIY electronics projects.

**HOW YOU SEARCH**: You search the user's electronics knowledge base using precise technical queries.
- Documents are tagged with 'electronics', 'circuit', 'embedded', 'arduino', 'esp32', 'raspberry_pi', etc.
- You find and synthesize information from the user's technical documentation and design notes
- Focus on what's actually in their knowledge base

**CAPABILITIES**:
1. **Circuit Design**: Schematic guidance, component selection, power supply design
2. **Embedded Programming**: Code generation for Arduino, ESP32, Raspberry Pi, STM32
3. **Component Selection**: Microcontrollers, sensors, power components, connectors
4. **Troubleshooting**: Circuit debugging, signal integrity, power issues
5. **Calculations**: Resistor values, voltage dividers, power dissipation, timing
6. **Project Management**: Create project plans, organize project files and folders in user's My Documents section

**STRUCTURED OUTPUT REQUIRED**:
You MUST respond with valid JSON matching this exact schema:

{
  "task_status": "complete",
  "response": "Your formatted natural language response with electronics information",
  "design_type": "circuit|code|component_selection|troubleshooting|calculation",
  "components": [
    {
      "name": "Component Name",
      "type": "microcontroller|sensor|resistor|capacitor|power_supply|connector",
      "value": "Value/Specification",
      "purpose": "What it does in the circuit",
      "alternatives": ["Alternative 1", "Alternative 2"]
    }
  ],
  "code_snippets": [
    {
      "platform": "arduino|esp32|raspberry_pi|stm32",
      "language": "cpp|python|micropython",
      "purpose": "What the code does",
      "code": "The actual code snippet"
    }
  ],
  "calculations": [
    {
      "type": "resistor_divider|power_dissipation|capacitor_value|timing",
      "formula": "Mathematical formula used",
      "result": "Calculated value with units",
      "explanation": "How to apply this calculation"
    }
  ],
  "recommendations": ["Design tip 1", "Safety note 2", "Best practice 3"],
  "confidence": 0.85
}

**RESPONSE GUIDELINES**:
- Use a technical, practical tone like an experienced electronics engineer
- Include specific component values, pin connections, and safety considerations
- Provide complete, runnable code with comments
- Show calculations with formulas and units
- Include wiring diagrams in ASCII art when helpful
- Highlight safety considerations and power ratings
- Suggest alternative components when appropriate
- **BE PROACTIVE**: Ask clarifying questions, suggest next steps, propose concrete actions based on what you know
- **ENGAGE WITH THEIR PROJECT**: Reference specific details from their project, offer to help with specific aspects, identify challenges they should consider

**IMPORTANT**:
- **PROJECT CONTEXT**: Agent assumes an electronics project plan is always open (gated by intent classifier)
- **SIMPLE TECHNICAL QUERIES**: For queries like "calculate resistor", "design a circuit", "troubleshoot", etc.:
  * Provide technical help directly, updating project files as needed
- **FILE CREATION**: You can create files and folders in the user's Documents section. Use create_user_file and create_user_folder tools to organize project files. Default to "Projects" or "Projects/Electronics" folder unless user specifies otherwise.
- **DOCUMENT EDITING**: 
  * Use update_document_metadata to set appropriate titles and frontmatter types (e.g., "electronics", "component", "schematic", "protocol")
  * Use update_document_content_tool to add or update content in project files (append=True to add content, append=False to replace)
  * When helping with project work, you can add component specs, code examples, schematics, etc. directly to the appropriate project files
  * This helps organize and categorize project files properly and keeps project documentation up-to-date
- **PROJECT CONTEXT FIRST**: Documents marked "(CURRENTLY OPEN IN EDITOR - PROJECT CONTEXT)" are the user's active project work and should be prioritized over reference materials
- **REFERENCED PROJECT FILES**: Documents marked "(REFERENCED FROM PROJECT - ...)" are files referenced in the active editor's frontmatter (components, protocols, schematics, specifications). These are part of the project structure and should be understood in relation to the main project document. Understand how these documents relate to each other and the overall project architecture.
- **REFERENCE DATA SECONDARY**: Documents marked "(LIBRARY REFERENCE)" are general electronics documentation for supplemental information
- **CONTEXT-AWARE RESPONSES**: When project context is available, tailor solutions to the user's specific project requirements and constraints. Use referenced files to understand component specifications, protocol requirements, and design constraints.
- **PROJECT STRUCTURE UNDERSTANDING**: When multiple referenced files are provided, understand their relationships - component specs inform design choices, protocols define communication requirements, schematics show circuit connections, etc.
- **GENERAL KNOWLEDGE FALLBACK**: When no user documents are available, provide guidance based on standard electronics principles
- **WEB SEARCH INTEGRATION**: When web search results are available, use them to supplement project-specific and reference knowledge
- **PRACTICAL SOLUTIONS**: Focus on implementable solutions with real component specifications appropriate to the user's project context
- **TRANSPARENCY**: Clearly indicate when responses are based on project context vs general knowledge
"""

    # Legacy query type detection methods removed - functionality now in _analyze_intent_node
    # _analyze_existing_sections_for_updates moved to electronics_content_nodes module
    # _remove_old_component_references_from_content moved to electronics_save_nodes module (duplicate removed)
    # _cleanup_old_component_references_OLD removed - functionality in electronics_save_nodes module

    def _should_add_timestamp(self, content_type: str, new_content: str, existing_content: str, section: str) -> bool:
        """
        Decide whether to add a timestamp when updating content.

        Timestamps are useful for tracking changes but can clutter content.
        Add timestamps for:
        - Specifications and requirements (these evolve)
        - Code and implementations (version tracking)
        - Measurements and test results (when they change)
        Skip timestamps for:
        - Placeholder content replacement
        - Minor corrections/typos
        - Initial content population
        """
        # Always skip timestamps for placeholder replacement
        if _is_placeholder_content(existing_content):
            return False

        # Add timestamps for content that benefits from version tracking
        timestamp_content_types = [
            "current_state",  # System state changes over time
            "calculations",   # Formulas and values may be updated
            "code",          # Code implementations evolve
            "components"     # Component specs may change
        ]

        if content_type in timestamp_content_types:
            return True

        # Add timestamps for sections that typically evolve
        timestamp_sections = [
            "requirements", "specifications", "design", "implementation",
            "testing", "results", "measurements", "performance"
        ]

        section_lower = section.lower()
        if any(ts_section in section_lower for ts_section in timestamp_sections):
            return True

        # Skip timestamps for:
        # - Very short updates (likely corrections)
        # - Content that looks like initial population
        if len(new_content.strip()) < 100:
            return False

        if any(phrase in new_content.lower() for phrase in ["initial", "first draft", "placeholder", "tbd"]):
            return False

        # Default: add timestamp for safety
        return True

    # Legacy search methods removed - functionality now in search_nodes module

    async def _generate_response(
        self, query: str, documents: List[Dict[str, Any]], query_type: str, web_search_needed: bool = False, metadata: Dict[str, Any] = None, referenced_context: Dict[str, Any] = None, has_project_plan: bool = False, project_plan_needed: bool = False, messages: List[Any] = None
    ) -> Dict[str, Any]:
        """Generate LLM response based on search results"""
        try:
            # Check if web search permission is available
            web_permission_available = metadata and metadata.get("shared_memory", {}).get("web_search_permission")
            
            # Handle project plan creation success message
            project_plan_action = metadata.get("project_plan_action") if metadata else None
            if project_plan_action == "created":
                project_plan_location = metadata.get("project_plan_location", "My Documents")
                project_plan_filename = metadata.get("project_plan_filename", "project plan")
                logger.info("🔌 Project plan was created - generating success message")
                
                success_message = f"""Great! I've created your electronics project structure:

**Project Location**: {project_plan_location}
**Main Project File**: {project_plan_filename}

The project files have been created and are ready for you to open. Please open the project plan file in your editor to continue working on your project.

**Your original query**: {query}

I'm ready to help you with:
- Component specifications and selection
- Circuit design and schematics  
- Embedded code generation
- Protocol documentation
- Project organization and structure

Once you open the project plan, I can provide more specific assistance based on your project context!"""
                
                return {
                    "task_status": "complete",
                    "response": success_message,
                    "design_type": query_type,
                    "components": [],
                    "code_snippets": [],
                    "calculations": [],
                    "recommendations": ["Open the project plan file to continue working on your project"],
                    "confidence": 0.9
                }
            
            # Handle project plan prompt if needed (but NOT if we're in the process of creating)
            project_plan_action = metadata.get("project_plan_action") if metadata else None
            if project_plan_needed and not has_project_plan and project_plan_action != "create":
                logger.info("🔌 Project plan needed - generating prompt for user")
                
                prompt = f"""I notice you're asking about an electronics project, but I don't see a project plan open in your editor.

To help you effectively, I can work with your project plan document. Would you like to:

1. **Open an existing project plan** - If you have a project plan document already, please open it in your editor
2. **Create a new project plan** - I can create a new project plan document for you

If you'd like me to create a new project plan, please tell me:
- What would you like to name the project? (e.g., "Smart Home Controller", "Weather Station")
- Where should I create it? (e.g., "Projects/Electronics", "My Documents", or a specific folder path)

**Example**: "Create a new project called Smart Sensor Network in Projects/Electronics"

Once you have a project plan open (or tell me to create one), I can help you with:
- Component specifications and selection
- Circuit design and schematics
- Embedded code generation
- Protocol documentation
- Project organization and structure

**Your original query**: {query}

Please let me know how you'd like to proceed!"""
                
                return {
                    "task_status": "permission_required",
                    "response": prompt,
                    "design_type": query_type,
                    "components": [],
                    "code_snippets": [],
                    "calculations": [],
                    "recommendations": ["Please open an existing project plan or ask me to create a new one"],
                    "confidence": 0.9,
                    "project_plan_needed": True,
                    "project_plan_prompt": True
                }

            # Handle case with no local documents AND no referenced files - provide general electronics guidance
            # Check if we have referenced_context (loaded from frontmatter) - that counts as local content!
            has_referenced_files = referenced_context and any(
                isinstance(docs, list) and len(docs) > 0 
                for docs in referenced_context.values()
            )
            
            if not documents and not has_referenced_files:
                logger.info("🔌 No local documents or referenced files found - providing general electronics guidance")

                # Check if there's active editor content that might be relevant
                has_active_editor = metadata and metadata.get("shared_memory", {}).get("active_editor")
                if has_active_editor:
                    active_editor = metadata["shared_memory"]["active_editor"]
                    editor_content = active_editor.get("content", "")
                    frontmatter = active_editor.get("frontmatter", {})
                    doc_type = frontmatter.get("type", "").lower()
                    
                    if editor_content.strip():
                        if doc_type == "electronics":
                            logger.info(f"🔌 Active editor IS electronics-related ({len(editor_content)} chars, type={doc_type}) - but no referenced files loaded")
                        else:
                            logger.info(f"🔌 Active editor detected but not electronics-related ({len(editor_content)} chars, type={doc_type})")
                        # Note: This means the active editor exists but wasn't detected as electronics content
                        # The search phase already handled this case

                # Build task instruction based on query type
                if query_type == "circuit_design":
                    task_instruction = "Provide general circuit design guidance for the user's electronics project."
                    design_type = "circuit_design"
                elif query_type == "embedded_code":
                    task_instruction = "Provide embedded programming guidance and code examples for the user's project."
                    design_type = "embedded_code"
                elif query_type == "component_selection":
                    task_instruction = "Provide component selection guidance for electronics projects."
                    design_type = "component_selection"
                elif query_type == "calculation":
                    task_instruction = "Perform the requested electronics calculations and provide explanations."
                    design_type = "calculation"
                elif query_type == "troubleshooting":
                    task_instruction = "Provide general troubleshooting guidance for electronics issues."
                    design_type = "troubleshooting"
                else:
                    task_instruction = "Provide helpful electronics design and programming guidance."
                    design_type = "general"

                # Add web search context if needed
                web_context = ""
                if web_search_needed and web_permission_available:
                    web_context = "\n**WEB SEARCH AVAILABLE**: I can search online for additional technical details if needed."
                elif web_search_needed and not web_permission_available:
                    web_context = "\n**NOTE**: Web search could provide additional technical details, but I only searched your local documents to respect privacy preferences."

                prompt = f"""{task_instruction}

**USER QUERY**: {query}

**GENERAL ELECTRONICS GUIDANCE**:
While you don't have specific electronics documents in your library yet, I can provide general guidance based on standard electronics principles and best practices.{web_context}

**BE PROACTIVE AND ENGAGING**:
- Based on what the user has told you, ask clarifying questions to better understand their needs
- Suggest specific next steps or areas to explore
- Propose concrete actions they could take (e.g., "We should determine the voltage requirements for your control system", "Let's create a component list for the keyboard scanning circuit")
- Identify potential challenges or considerations they should think about
- Offer to help with specific aspects of their project (e.g., "I can help you design the keyboard scanning matrix", "Would you like me to create a schematic for the control interface?")
- Be conversational and helpful - don't just provide generic information

**INSTRUCTIONS**:
- Provide practical, implementable electronics solutions
- Include component recommendations with specifications
- Show circuit diagrams using ASCII art when helpful
- Provide code examples for embedded programming
- Include safety considerations and best practices
- Ask questions to clarify requirements
- Suggest specific next steps based on what you know
- Structure your JSON response according to the schema provided in the system prompt
"""

                # Build system prompt
                system_prompt = self._build_electronics_prompt()

                # Extract conversation history for context
                conversation_history = []
                if messages:
                    conversation_history = self._extract_conversation_history(messages, limit=10)

                # Build messages with conversation history
                llm_messages = self._build_messages(system_prompt, prompt, conversation_history)

                logger.info("🤖 Calling LLM for general electronics guidance")

                # Get LLM response - construct state dict from metadata for model selection
                state_dict = {"metadata": metadata} if metadata else None
                llm = self._get_llm(temperature=0.3, state=state_dict)  # Lower temperature for technical accuracy
                response = await llm.ainvoke(llm_messages)

                content = response.content if hasattr(response, 'content') else str(response)
                logger.info(f"✅ Got LLM response: {len(content)} chars")

                # Parse JSON response
                parsed_response = self._parse_json_response(content)

                # Ensure task_status is set
                if "task_status" not in parsed_response:
                    parsed_response["task_status"] = "complete"

                # Add metadata
                parsed_response["web_search_needed"] = web_search_needed
                parsed_response["web_permission_available"] = web_permission_available
                parsed_response["design_type"] = design_type
                parsed_response["used_general_knowledge"] = True  # Indicate this was general guidance

                return parsed_response

            # Separate and prioritize content types
            project_context_docs = [d for d in documents if d.get("source") == "active_editor"]
            reference_docs = [d for d in documents if d.get("source") != "active_editor"]

            context_parts = []
            doc_counter = 1

            # ALL queries routed to electronics_agent are project-oriented by definition
            # Always use project context unless editor_preference is "ignore"
            shared_memory = metadata.get("shared_memory", {}) if metadata else {}
            editor_preference = shared_memory.get("editor_preference", "prefer")
            
            should_use_project_context = (editor_preference != "ignore")
            
            # === PROJECT CONTEXT (ACTIVE EDITOR) - HIGHEST PRIORITY ===
            # Only include if query is project-related and editor_preference is not "ignore"
            if should_use_project_context:
                # First, try to get active editor from documents list (if search was performed)
                if project_context_docs:
                    logger.info(f"🔌 📋 Including {len(project_context_docs)} project context document(s) from active editor (from search results)")
                    for doc in project_context_docs[:2]:  # Limit to 2 project docs
                        content = doc.get("content", "")  # Full content - these are technical docs, not novels
                        title = doc.get("title", f"Project Document {doc_counter}")
                        source_info = " (CURRENTLY OPEN IN EDITOR - PROJECT CONTEXT)"
                        context_parts.append(f"### {title}{source_info}:\n{content}\n")
                        doc_counter += 1
                # If not in documents list, get directly from metadata (when search was skipped)
                elif metadata and metadata.get("shared_memory", {}).get("active_editor"):
                    active_editor = metadata["shared_memory"]["active_editor"]
                    editor_content = active_editor.get("content", "")
                    frontmatter = active_editor.get("frontmatter", {})
                    doc_type = frontmatter.get("type", "").lower()
                    filename = active_editor.get("filename", "untitled.md")
                    
                    if editor_content.strip() and doc_type == "electronics":
                        logger.info(f"🔌 📋 Including active editor content directly from metadata ({len(editor_content)} chars, type={doc_type})")
                        title = frontmatter.get("title", filename.replace(".md", "").replace("_", " ").title())
                        content = editor_content  # Full content - these are technical docs, not novels
                        source_info = " (CURRENTLY OPEN IN EDITOR - PROJECT CONTEXT)"
                        context_parts.append(f"### {title}{source_info}:\n{content}\n")
                        doc_counter += 1
            else:
                if editor_preference == "ignore":
                    logger.info(f"🔌 Skipping project context - editor_preference is 'ignore'")

            # === REFERENCED FILES FROM PROJECT (LOADED FROM FRONTMATTER) - HIGH PRIORITY ===
            # Only include if query is project-related and editor_preference is not "ignore"
            if should_use_project_context and referenced_context:
                total_referenced = sum(len(docs) for docs in referenced_context.values() if isinstance(docs, list))
                if total_referenced > 0:
                    logger.info(f"🔌 📄 Including {total_referenced} referenced file(s) from project frontmatter")
                    
                    # Include referenced files by category with clear labels
                    category_labels = {
                        "components": "COMPONENT SPECIFICATIONS",
                        "protocols": "PROTOCOL DOCUMENTATION",
                        "schematics": "SCHEMATIC DOCUMENTATION",
                        "specifications": "TECHNICAL SPECIFICATIONS",
                        "other": "PROJECT REFERENCES"
                    }
                    
                    for category, docs in referenced_context.items():
                        if isinstance(docs, list) and docs:
                            category_label = category_labels.get(category, "PROJECT REFERENCES")
                            for doc in docs[:3]:  # Limit to 3 per category
                                content = doc.get("content", "")  # Full content - these are technical docs, not novels
                                title = doc.get("title", f"Referenced {category.title()} {doc_counter}")
                                source_info = f" (REFERENCED FROM PROJECT - {category_label})"
                                context_parts.append(f"### {title}{source_info}:\n{content}\n")
                                doc_counter += 1

            # === REFERENCE DATA (LIBRARY DOCS) - SECONDARY ===
            if reference_docs:
                logger.info(f"🔌 📚 Including {len(reference_docs)} reference document(s) from library")
                for doc in reference_docs[:3]:  # Limit to 3 reference docs
                    content = doc.get("content", "")  # Full content - these are technical docs, not novels
                    title = doc.get("title", f"Reference Document {doc_counter}")
                    source_info = " (LIBRARY REFERENCE)"
                    context_parts.append(f"### {title}{source_info}:\n{content}\n")
                    doc_counter += 1

            context = "\n".join(context_parts)

            # Add content source summary for transparency
            content_sources = []
            if project_context_docs:
                content_sources.append(f"{len(project_context_docs)} project document(s) from active editor")
            elif metadata and metadata.get("shared_memory", {}).get("active_editor"):
                active_editor = metadata["shared_memory"]["active_editor"]
                if active_editor.get("content", "").strip() and active_editor.get("frontmatter", {}).get("type", "").lower() == "electronics":
                    content_sources.append("1 project document from active editor")
            if referenced_context:
                total_referenced = sum(len(docs) for docs in referenced_context.values() if isinstance(docs, list))
                if total_referenced > 0:
                    content_sources.append(f"{total_referenced} referenced file(s) from project frontmatter")
            if reference_docs:
                content_sources.append(f"{len(reference_docs)} reference document(s) from library")

            content_summary = f"Content sources: {', '.join(content_sources)}" if content_sources else "Using general electronics knowledge"
            logger.info(f"🔌 Context built: {content_summary}")

            # Build task instruction based on query type and content sources
            content_type_note = ""
            if project_context_docs or (referenced_context and sum(len(docs) for docs in referenced_context.values() if isinstance(docs, list)) > 0):
                content_type_note = " **PRIORITIZE PROJECT CONTEXT**: The user has project-specific content open in their editor, including referenced component specifications, protocols, schematics, and other project documentation. Use this current project context as the primary reference for their specific needs, understanding how the various documents relate to each other in the project structure. Then supplement with reference documentation as needed."

            if query_type == "circuit_design":
                task_instruction = f"Provide circuit design guidance based on the user's request.{content_type_note} Focus on their current project context first, then reference general electronics knowledge."
            elif query_type == "embedded_code":
                task_instruction = f"Generate embedded code for the specified platform and application.{content_type_note} Adapt code examples to their current project requirements and constraints."
            elif query_type == "component_selection":
                task_instruction = f"Recommend components for the user's electronics project.{content_type_note} Consider their existing project setup and requirements from the active editor."
            elif query_type == "calculation":
                task_instruction = f"Perform the requested electronics calculations with formulas and explanations.{content_type_note} Use values and constraints from their current project if available."
            elif query_type == "troubleshooting":
                task_instruction = f"Help troubleshoot the electronics issue.{content_type_note} Analyze their current project setup for potential issues and solutions."
            else:
                task_instruction = f"Provide electronics assistance based on the user's request.{content_type_note} Focus on their immediate project needs."

            # Add web search context
            web_context = ""
            if web_search_needed and web_permission_available:
                web_context = "\n**WEB SEARCH AVAILABLE**: Additional web results were included for comprehensive information."
            elif web_search_needed and not web_permission_available:
                web_context = "\n**LIMITED TO LOCAL CONTENT**: Additional web search could provide more comprehensive information, but only local documents were searched to respect privacy preferences."

            prompt = f"""{task_instruction}

**USER QUERY**: {query}

**AVAILABLE ELECTRONICS CONTENT**:
{context}{web_context}

**BE PROACTIVE AND ENGAGING**:
- Based on what the user has told you and the available project context, ask clarifying questions to better understand their needs
- Suggest specific next steps or areas to explore based on their current project state
- Propose concrete actions they could take (e.g., "We should determine the voltage requirements for your control system", "Let's create a component list for the keyboard scanning circuit")
- Identify potential challenges or considerations they should think about
- Offer to help with specific aspects of their project (e.g., "I can help you design the keyboard scanning matrix", "Would you like me to create a schematic for the control interface?")
- Reference specific details from their project documents when making suggestions
- Be conversational and helpful - engage with their actual project, not just provide generic information

**INSTRUCTIONS**:
- Base your response on the documents provided above
- Use a technical, practical tone like an experienced electronics engineer
- Include specific component values, calculations, and safety considerations
- For code generation, provide complete, runnable examples with comments
- For circuit design, include ASCII art schematics when helpful
- **CRITICAL**: DO NOT claim you have "updated files" or "saved information" in your response text. The system will automatically update the appropriate project files after you generate your response. Just provide the information - don't mention file updates.
- Show calculations with formulas and units
- Format response in markdown for readability
- Ask questions to clarify requirements based on what you know
- Suggest specific next steps based on their project context
- Structure your JSON response according to the schema provided in the system prompt
- If web search was needed but not available, acknowledge this limitation in your response
"""

            # Build system prompt
            system_prompt = self._build_electronics_prompt()

            # Extract conversation history for context
            conversation_history = []
            if messages:
                conversation_history = self._extract_conversation_history(messages, limit=10)

            # Build messages with conversation history
            llm_messages = self._build_messages(system_prompt, prompt, conversation_history)

            logger.info(f"🤖 Calling LLM for electronics response")

            # Get LLM response - construct state dict from metadata for model selection
            state_dict = {"metadata": metadata} if metadata else None
            llm = self._get_llm(temperature=0.3, state=state_dict)  # Lower temperature for technical accuracy
            response = await llm.ainvoke(llm_messages)

            content = response.content if hasattr(response, 'content') else str(response)
            logger.info(f"✅ Got LLM response: {len(content)} chars")

            # Parse JSON response
            parsed_response = self._parse_json_response(content)

            # Ensure task_status is set
            if "task_status" not in parsed_response:
                parsed_response["task_status"] = "complete"

            # Add web search metadata
            parsed_response["web_search_needed"] = web_search_needed
            parsed_response["web_permission_available"] = web_permission_available

            return parsed_response

        except Exception as e:
            logger.error(f"❌ Response generation failed: {e}")
            return self._create_error_response(str(e))

    # ===== Project Content Management =====
    # Project content management helpers now use orchestrator.tools.project_content_tools
    # Methods removed - use imported functions directly:
    # - save_or_update_project_content()
    # - determine_content_target()
    # - enrich_documents_with_metadata()
    # - check_if_new_file_needed()
    # - create_new_project_file()
    # - should_update_existing_section()
    # - propose_section_update()
    # - append_project_content()

    # ===== LangGraph Node Functions =====
    
    # Legacy node methods removed - functionality now in _analyze_intent_node or _load_project_context_node
    # Old lazy-loading workflow (_load_referenced_context_node, _score_file_relevance_node, _load_selective_content_node) removed
    # Search need detection now handled in _analyze_intent_node
    
    # _request_web_search_permission_node removed - web search is automatic
    
    async def _perform_web_search_node(self, state: ElectronicsState) -> Dict[str, Any]:
        """Node: Perform web search and add results to documents"""
        try:
            query = state["query"]
            query_type = state["query_type"]
            user_id = state["user_id"]
            metadata = state.get("metadata", {})
            existing_documents = state.get("documents", [])
            web_search_explicit = state.get("web_search_explicit", False)
            
            logger.info("🔌 Performing web search")
            
            # Web search is always permitted when deemed necessary
            
            # Build search queries
            from orchestrator.tools.enhancement_tools import expand_query_tool
            expansion_result = await expand_query_tool(query, num_variations=3)
            expanded_queries = expansion_result.get('expanded_queries', [query])
            
            query_lower = query.lower()
            explicit_web_search_keywords = [
                "search the web", "search online", "look up online", "search internet",
                "find online", "web search", "search for", "look up", "google",
                "search web for", "search online for", "find on the web"
            ]
            
            # For explicit requests, use cleaned query
            # For automatic searches, use expanded queries with electronics context
            if web_search_explicit:
                cleaned_query = query_lower
                for keyword in explicit_web_search_keywords:
                    cleaned_query = cleaned_query.replace(keyword, "").strip()
                web_search_queries = [cleaned_query if len(cleaned_query) > 10 else query]
                logger.info(f"🔌 Using explicit web search query: {web_search_queries[0]}")
            else:
                web_search_queries = [f"{sq} electronics components circuits" for sq in expanded_queries[:2]]
            
            # Perform web search
            web_documents = []
            try:
                for search_query in web_search_queries:
                    logger.info(f"🔌 Executing web search: {search_query[:100]}")
                    web_search_result = await search_web_structured(
                        query=search_query,
                        max_results=5
                    )
                    
                    if web_search_result and len(web_search_result) > 0:
                        logger.info(f"✅ Found {len(web_search_result)} web results")
                        
                        # Add web results as documents
                        for i, result in enumerate(web_search_result[:3]):  # Limit to 3 web results
                            if isinstance(result, dict):
                                web_doc = {
                                    "document_id": f"web_{i}",
                                    "title": result.get('title', f'Web Result {i+1}'),
                                    "filename": result.get('url', ''),
                                    "content": result.get('snippet', '') or result.get('content', ''),
                                    "tags": ["web", "electronics"],
                                    "category": "web_search",
                                    "source": "web"
                                }
                                web_documents.append(web_doc)
                        break  # Got results, no need to try more queries
                        
            except Exception as e:
                logger.error(f"❌ Web search failed: {e}")
                import traceback
                logger.error(f"❌ Web search traceback: {traceback.format_exc()}")
            
            # Combine existing documents with web results
            all_documents = existing_documents + web_documents
            
            logger.info(f"✅ Web search complete: added {len(web_documents)} web results to {len(existing_documents)} local documents")
            
            return {
                "documents": all_documents,
                "web_search_needed": False  # Web search completed
            }
            
        except Exception as e:
            logger.error(f"❌ Web search failed: {e}")
            return {
                "documents": state.get("documents", []),
                "web_search_needed": False,
                "error": str(e)
            }
    
    # Legacy _route_from_search functions removed - web search routing now handled by _route_from_quality_assessment
    
    # ============================================
    # DOCUMENTATION MAINTENANCE NODES
    # ============================================
    
    async def _plan_execution_strategy_node(self, state: ElectronicsState) -> Dict[str, Any]:
        """
        Strategic planning node that analyzes query and creates execution plan including documentation maintenance.
        
        Uses LLM to:
        1. Analyze information needs
        2. Identify dependencies
        3. Plan documentation maintenance operations (removals, updates, verifications)
        4. Optimize execution workflow
        """
        try:
            query = state.get("query", "")
            user_id = state.get("user_id", "")
            metadata = state.get("metadata", {})
            referenced_context = state.get("referenced_context", {})
            project_decisions = state.get("project_decisions", [])
            
            # Get available project files
            shared_memory = metadata.get("shared_memory", {})
            active_editor = shared_memory.get("active_editor", {})
            frontmatter = active_editor.get("frontmatter", {})
            
            available_files = []
            if frontmatter:
                for key in ["components", "protocols", "schematics", "specifications", "code"]:
                    refs = frontmatter.get(key, [])
                    if isinstance(refs, list):
                        available_files.extend([f"- {ref}" for ref in refs])
            
            # Build prompt for LLM to create execution plan
            fast_model = self._get_fast_model(state)
            llm = self._get_llm(temperature=0.1, model=fast_model, state=state)
            
            prompt = f"""You are a strategic planning expert for electronics project documentation. Analyze the query and create an execution plan.

**QUERY**: {query}

**AVAILABLE PROJECT FILES**:
{chr(10).join(available_files) if available_files else "None specified"}

**RECENT DECISIONS**:
{json.dumps(project_decisions[-5:], indent=2) if project_decisions else "None"}

**TASK**: Create a strategic execution plan that includes:
1. Information needs analysis (what information is needed to answer the query)
2. Dependency detection (what must be done before other steps)
3. Documentation maintenance planning (what sections need removal/update/verification)
4. Workflow optimization (optimal order of operations)

**DOCUMENTATION MAINTENANCE ANALYSIS**:
Analyze if the query involves:
- Component replacements (old component sections should be removed)
- Design changes (outdated design sections should be updated)
- Decision finalization (alternative evaluation sections should be removed)
- Specification updates (conflicting specifications should be resolved)

**INSTRUCTIONS**:
1. Identify if documentation maintenance is needed based on the query and recent decisions
2. For each maintenance item, specify: file, section, action (remove/update/verify/archive), reason, confidence
3. Only include maintenance items with confidence > 0.7
4. Consider dependencies between operations
5. Return structured plan with maintenance operations

**OUTPUT FORMAT**: Return ONLY valid JSON matching this schema:
{{
  "information_needs": ["need1", "need2"],
  "dependencies": [
    {{"step": "step1", "depends_on": ["step2"]}}
  ],
  "maintenance_plan": {{
    "maintenance_items": [
      {{
        "file": "./component_specs.md",
        "section": "AQV252G PhotoMOS Relay",
        "action": "remove",
        "reason": "Component replaced with ADW221S in recent decision",
        "confidence": 0.95,
        "related_decision_id": "dec_001"
      }}
    ],
    "priority": "high|medium|low",
    "reasoning": "Explanation of maintenance plan"
  }},
  "workflow_optimization": {{
    "recommended_order": ["step1", "step2"],
    "can_parallelize": ["step3", "step4"]
  }}
}}

Return ONLY the JSON object, no markdown, no code blocks."""
            
            try:
                # Try structured output with Pydantic model
                structured_llm = llm.with_structured_output({
                    "type": "object",
                    "properties": {
                        "information_needs": {"type": "array", "items": {"type": "string"}},
                        "dependencies": {"type": "array"},
                        "maintenance_plan": {"type": "object"},
                        "workflow_optimization": {"type": "object"}
                    }
                })
                result = await structured_llm.ainvoke([{"role": "user", "content": prompt}])
                plan_dict = result if isinstance(result, dict) else result.dict() if hasattr(result, 'dict') else result.model_dump()
            except Exception:
                # Fallback to manual parsing
                response = await llm.ainvoke([{"role": "user", "content": prompt}])
                content = response.content if hasattr(response, 'content') else str(response)
                plan_dict = self._extract_json_from_response(content) or {}
            
            # Validate and structure maintenance plan
            maintenance_plan_dict = plan_dict.get("maintenance_plan", {})
            maintenance_items = maintenance_plan_dict.get("maintenance_items", [])
            
            # Filter by confidence threshold
            filtered_items = [item for item in maintenance_items if item.get("confidence", 0) > 0.7]
            
            if filtered_items:
                try:
                    # Validate with Pydantic
                    maintenance_plan = DocumentationMaintenancePlan(
                        maintenance_items=[DocumentationMaintenanceItem(**item) for item in filtered_items],
                        priority=maintenance_plan_dict.get("priority", "medium"),
                        reasoning=maintenance_plan_dict.get("reasoning", "")
                    )
                    maintenance_plan_dict = maintenance_plan.dict() if hasattr(maintenance_plan, 'dict') else maintenance_plan.model_dump()
                    logger.info(f"✅ Created maintenance plan with {len(filtered_items)} items")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to validate maintenance plan: {e}")
                    maintenance_plan_dict = {"maintenance_items": filtered_items, "priority": "medium", "reasoning": ""}
            else:
                maintenance_plan_dict = {"maintenance_items": [], "priority": "low", "reasoning": "No maintenance needed"}
            
            return {
                "documentation_maintenance_plan": maintenance_plan_dict
            }
            
        except Exception as e:
            logger.error(f"❌ Execution planning failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "documentation_maintenance_plan": {"maintenance_items": [], "priority": "low", "reasoning": "Planning failed"}
            }
    
    # _extract_decisions_node moved to electronics_decision_nodes module
    # This method is now in ElectronicsDecisionNodes.extract_decisions_node
    
    # _verify_documentation_node moved to electronics_decision_nodes module
    # This method is now in ElectronicsDecisionNodes.verify_documentation_node
    
    # _execute_maintenance_node moved to electronics_maintenance_nodes module
    # This method is now in ElectronicsMaintenanceNodes.execute_maintenance_node

    # _update_project_plan_node moved to electronics_project_plan_nodes module
    # This method is now in ElectronicsProjectPlanNodes.update_project_plan_node
    
    async def _generate_response_node(self, state: ElectronicsState) -> Dict[str, Any]:
        """Node: Generate final response"""
        try:
            query = state["query"]
            query_type = state["query_type"]
            documents = state.get("documents", [])
            referenced_context = state.get("referenced_context", {})
            has_project_plan = state.get("has_project_plan", False)
            project_plan_needed = state.get("project_plan_needed", False)
            project_plan_action = state.get("project_plan_action")
            project_plan_location = state.get("project_plan_location")
            project_plan_filename = state.get("project_plan_filename")
            web_search_needed = state.get("web_search_needed", False)
            metadata = state.get("metadata", {})
            user_id = state.get("user_id", "system")
            
            # Get document_id from active editor if project plan is open
            project_plan_document_id = None
            if has_project_plan:
                shared_memory = metadata.get("shared_memory", {})
                active_editor = shared_memory.get("active_editor", {})
                # Try to get real document_id from active editor
                project_plan_document_id = active_editor.get("document_id")
                
                # Also check documents list - the project plan might be in there
                if (not project_plan_document_id or project_plan_document_id == "active_editor") and documents:
                    for doc in documents:
                        if doc.get("source") == "active_editor" and doc.get("document_id") and doc.get("document_id") != "active_editor":
                            project_plan_document_id = doc.get("document_id")
                            logger.info(f"🔌 Found project plan document_id from documents list: {project_plan_document_id}")
                            break
                
                # Fallback: try to get from file_path or filename and search
                if not project_plan_document_id or project_plan_document_id == "active_editor":
                    filename = active_editor.get("filename", "") or active_editor.get("file_path", "")
                    if filename:
                        logger.info(f"🔌 Project plan open but no document_id - searching by filename: {filename}")
                        try:
                            # Search for document by filename
                            search_result = await search_documents_structured(
                                query=filename,
                                limit=5,
                                user_id=user_id
                            )
                            if search_result.get("total_count", 0) > 0:
                                # Find exact match by filename
                                for result in search_result.get("results", []):
                                    if result.get("filename", "").endswith(filename) or filename in result.get("filename", ""):
                                        project_plan_document_id = result.get("document_id")
                                        logger.info(f"🔌 Found project plan document_id via search: {project_plan_document_id}")
                                        break
                        except Exception as e:
                            logger.warning(f"⚠️ Could not search for document by filename: {e}")
            
            # Pass project plan info and query relevance to metadata for response generation
            metadata = metadata.copy() if metadata else {}
            if project_plan_action:
                metadata["project_plan_action"] = project_plan_action
                if project_plan_location:
                    metadata["project_plan_location"] = project_plan_location
                if project_plan_filename:
                    metadata["project_plan_filename"] = project_plan_filename
            
            # ALL queries routed to electronics_agent are project-oriented by definition
            # ALWAYS reload referenced files before generating response to ensure we have latest content
            # This ensures the agent is always up-to-date on what's in the project files
            # Only skip if editor_preference is "ignore"
            shared_memory = metadata.get("shared_memory", {})
            editor_preference = shared_memory.get("editor_preference", "prefer")
            should_reload_files = (
                editor_preference != "ignore" and
                metadata.get("shared_memory", {}).get("active_editor")
            )
            
            if should_reload_files:
                logger.info("🔌 Reloading referenced files to ensure latest content before generating response")
                try:
                    shared_memory = metadata.get("shared_memory", {})
                    active_editor = shared_memory.get("active_editor", {})
                    
                    # Electronics reference configuration
                    reference_config = {
                        "components": ["components", "component", "component_docs"],
                        "protocols": ["protocols", "protocol", "protocol_docs"],
                        "schematics": ["schematics", "schematic", "schematic_docs"],
                        "specifications": ["specifications", "spec", "specs", "specification"],
                        "other": ["references", "reference", "docs", "documents", "related", "files"]
                    }
                    
                    # Reload referenced files with latest content
                    reload_result = await load_referenced_files(
                        active_editor=active_editor,
                        user_id=user_id,
                        reference_config=reference_config,
                        doc_type_filter="electronics",
                        cascade_config=None
                    )
                    
                    # Update referenced_context with fresh content
                    loaded_files = reload_result.get("loaded_files", {})
                    referenced_context = {
                        "components": loaded_files.get("components", []),
                        "protocols": loaded_files.get("protocols", []),
                        "schematics": loaded_files.get("schematics", []),
                        "specifications": loaded_files.get("specifications", []),
                        "other": loaded_files.get("other", [])
                    }
                    logger.info(f"🔌 Reloaded {sum(len(docs) for docs in referenced_context.values() if isinstance(docs, list))} referenced file(s) with latest content")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to reload referenced files: {e} - using existing referenced_context")
            else:
                if editor_preference == "ignore":
                    logger.info(f"🔌 Skipping file reload - editor_preference is 'ignore'")
            
            logger.info("🔌 Generating electronics response")
            
            # Get messages from state for conversation history
            state_messages = state.get("messages", [])
            
            result = await self._generate_response(
                query, documents, query_type, web_search_needed, metadata, referenced_context,
                has_project_plan, project_plan_needed, messages=state_messages
            )
            
            # Store project_plan_document_id in state for later nodes
            # CRITICAL: Also update referenced_context in state so subsequent nodes have the latest file content
            state_updates = {
                "response": result,
                "task_status": result.get("task_status", "complete"),
                "referenced_context": referenced_context  # Update state with reloaded files
            }
            
            # Also check documents list for project plan if we don't have document_id yet
            if has_project_plan and (not project_plan_document_id or project_plan_document_id == "active_editor"):
                # Look for project plan in documents list
                for doc in documents:
                    doc_type = doc.get("metadata", {}).get("type", "").lower() if isinstance(doc.get("metadata"), dict) else ""
                    if doc_type == "electronics" and doc.get("source") == "active_editor":
                        # Try to get real document_id - might need to search
                        potential_id = doc.get("document_id")
                        if potential_id and potential_id != "active_editor":
                            project_plan_document_id = potential_id
                            logger.info(f"🔌 Found project plan document_id from documents: {project_plan_document_id}")
                            break
            
            # Store project plan document_id in metadata for content routing nodes
            if project_plan_document_id and project_plan_document_id != "active_editor":
                if metadata:
                    metadata = metadata.copy()
                else:
                    metadata = {}
                metadata["project_plan_document_id"] = project_plan_document_id
                state_updates["metadata"] = metadata
            
            # Add assistant response to messages for checkpoint persistence
            response_text = result.get("response", "")
            if isinstance(response_text, dict):
                response_text = response_text.get("response", "")
            if response_text:
                state = self._add_assistant_response_to_messages(state, str(response_text))
                state_updates["messages"] = state.get("messages", [])
            
            return state_updates
        except Exception as e:
            logger.error(f"❌ Response generation failed: {e}")
            error_response = self._create_error_response(str(e))
            return {
                "response": error_response,
                "task_status": "error",
                "error": str(e)
            }
    
    # Legacy _check_content_saving_needed_node removed - saving decision now handled in _route_from_response
    
    def _convert_json_to_markdown(self, content: str, content_type: Optional[str] = None) -> str:
        """
        Convert any JSON arrays/objects in content to markdown format.
        Delegates to shared utility function for consistency across agents.
        
        Args:
            content: Content that may contain JSON
            content_type: Type of content ("code", "components", "calculations", "general", etc.)
        """
        from orchestrator.utils.content_formatting_utils import convert_json_to_markdown
        return convert_json_to_markdown(content, content_type=content_type)
    
    def _extract_json_from_response(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from LLM response, handling markdown code blocks and extra text.
        
        Tries multiple strategies:
        1. Direct JSON parsing
        2. Extract from markdown code blocks (```json ... ```)
        3. Find JSON object in text using regex
        """
        if not content or not content.strip():
            logger.warning("⚠️ Empty content received for JSON extraction")
            return None
        
        # Strategy 1: Try direct JSON parsing
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract from markdown code blocks
        json_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        matches = re.findall(json_block_pattern, content, re.DOTALL)
        if matches:
            try:
                return json.loads(matches[0])
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Find JSON object using regex
        json_object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_object_pattern, content, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # Strategy 4: Try to find JSON after common prefixes
        for prefix in ["Here's the JSON:", "JSON:", "Response:", "```"]:
            if prefix in content:
                idx = content.find(prefix) + len(prefix)
                remaining = content[idx:].strip()
                # Remove leading/trailing backticks if present
                remaining = remaining.strip('`').strip()
                try:
                    return json.loads(remaining)
                except json.JSONDecodeError:
                    continue
        
        logger.warning(f"⚠️ Could not extract JSON from response. First 200 chars: {content[:200]}")
        return None
    
    # Legacy unused enhancement nodes removed - functionality moved to node modules:
    # _extract_content_structure_node -> moved to electronics_content_nodes.extract_and_route_content_node
    # _detect_follow_up_node -> functionality in _analyze_intent_node
    # _validate_response_quality_node -> functionality in search_nodes.assess_result_quality_node
    # _detect_incremental_update_node -> functionality in content_nodes
    # _detect_content_conflicts_node -> functionality in content_nodes
    # _check_component_compatibility_node -> functionality in content_nodes
    
    async def process(self, query: str, metadata: Dict[str, Any] = None, messages: List[Any] = None) -> Dict[str, Any]:
        """Process electronics query using LangGraph workflow"""
        try:
            logger.info(f"🔌 Electronics agent processing: {query[:100]}...")

            # Extract user_id from metadata
            metadata = metadata or {}
            user_id = metadata.get("user_id", "system")

            # Prepare new messages (current query)
            new_messages = self._prepare_messages_with_query(messages, query)
            
            # Get workflow to access checkpoint
            workflow = await self._get_workflow()
            config = self._get_checkpoint_config(metadata)
            
            # Load and merge checkpointed messages to preserve conversation history
            conversation_messages = await self._load_and_merge_checkpoint_messages(
                workflow, config, new_messages
            )
            
            # Check if this is an approval response - if so, check for pending operations
            is_approval = self._is_approval_response(query)
            
            pending_save_plan = None
            if is_approval:
                # Try to restore pending_save_plan from checkpoint
                pending_save_plan = await self._restore_pending_operation_from_checkpoint(
                    workflow, config, "pending_save_plan"
                )
                
                # Fallback: check if save_plan exists but wasn't executed
                if not pending_save_plan:
                    try:
                        checkpoint_state = await workflow.aget_state(config)
                        if checkpoint_state and checkpoint_state.values:
                            save_plan = checkpoint_state.values.get("save_plan")
                            if save_plan and save_plan.get("routing"):
                                pending_save_plan = save_plan
                                logger.info("🔌 Found pending save_plan in checkpoint - will restore on approval")
                    except Exception as e:
                        logger.debug(f"Could not load checkpoint state: {e}")
            
            # Initialize state for LangGraph workflow
            initial_state: ElectronicsState = {
                "query": query,
                "user_id": user_id,
                "pending_save_plan": pending_save_plan,  # Restore pending operation if found
                "metadata": metadata,  # Includes user_chat_model and user_fast_model
                "messages": conversation_messages,
                "query_type": "",
                "should_search": False,
                "has_explicit_search": False,
                "documents": [],
                "segments": [],
                "referenced_context": {
                    "components": [],
                    "protocols": [],
                    "schematics": [],
                    "specifications": [],
                    "other": []
                },
                "information_needs": None,
                "search_queries": [],
                "search_quality_assessment": None,
                "search_retry_count": 0,
                "has_project_plan": False,
                "project_plan_needed": False,
                "project_plan_action": None,
                "project_plan_location": None,
                "project_structure_plan": None,
                "web_search_needed": False,
                "web_search_explicit": False,
                "web_search_permission_granted": False,
                "response": {},
                "should_save_content": False,
                "is_explicit_save_request": False,
                "content_structure": None,
                "file_routing_plan": None,
                "is_follow_up": False,
                "needs_previous_context": False,
                "has_content_conflicts": False,
                "content_conflicts": None,
                "response_quality_score": None,
                "needs_response_refinement": False,
                "is_incremental_update": False,
                "component_compatibility_issues": None,
                "editing_mode": False,
                "editor_operations": [],
                "manuscript_edit": None,
                "plan_edits": None,
                "task_status": "",
                "error": ""
            }
            
            # Get workflow (lazy initialization with checkpointer)
            workflow = await self._get_workflow()
            
            # Get checkpoint config (handles thread_id from conversation_id/user_id)
            config = self._get_checkpoint_config(metadata)
            
            # Run LangGraph workflow with checkpointing
            result_state = await workflow.ainvoke(initial_state, config=config)
            
            # Extract final response
            response = result_state.get("response", {})
            task_status = result_state.get("task_status", "complete")
            
            # Define allowed user-facing keys (filter out internal state)
            allowed_keys = {
                "response", "task_status", "taskstatus",  # Response text and status
                "components", "code_snippets", "calculations", "recommendations",  # Structured content
                "design_type", "confidence",  # Metadata
                "saved_files", "files_updated",  # File operation results
                "editor_operations", "manuscript_edit"  # Inline editing operations
            }
            
            # Extract editor_operations and manuscript_edit from state (for inline editing)
            editor_operations = result_state.get("editor_operations", [])
            manuscript_edit = result_state.get("manuscript_edit")
            
            # Ensure response is a dict with "response" key containing text
            if isinstance(response, dict):
                # Extract response text (handle both "response" and "message" keys for compatibility)
                response_text = response.get("response") or response.get("message", "")
                if not response_text or not isinstance(response_text, str):
                    # Try to convert to string if it's not already
                    response_text = str(response_text) if response_text else "Electronics design assistance complete"
                
                # Check if response_text is JSON that should be converted to readable format
                if response_text.strip().startswith("{") or response_text.strip().startswith("["):
                    try:
                        import json
                        parsed_json = json.loads(response_text)
                        # Convert JSON to readable markdown format
                        response_text = self._convert_json_to_markdown(str(parsed_json))
                    except (json.JSONDecodeError, ValueError):
                        # Not valid JSON, use as-is
                        pass
                
                # Append cleanup summary if available (from save_content_node)
                cleanup_summary = result_state.get("cleanup_summary")
                if cleanup_summary:
                    response_text = f"{response_text}\n\n{cleanup_summary}"
                
                # Build final response with only allowed keys
                final_response = {
                    "response": response_text,
                    "task_status": task_status
                }
                
                # Add editor_operations and manuscript_edit if present
                if editor_operations:
                    final_response["editor_operations"] = editor_operations
                if manuscript_edit:
                    final_response["manuscript_edit"] = manuscript_edit
                
                # Add allowed keys from response dict (normalize taskstatus -> task_status)
                for key, value in response.items():
                    if key == "response":
                        continue  # Already handled above
                    elif key == "taskstatus":
                        # Normalize to task_status
                        if "task_status" not in final_response:
                            final_response["task_status"] = value
                    elif key.lower() in {k.lower() for k in allowed_keys}:
                        # Include if it's an allowed key (case-insensitive)
                        final_response[key] = value
                    # Filter out internal state keys like 'pistons', 'websearchneeded', etc.
            else:
                # Fallback: wrap non-dict response
                final_response = {
                    "response": str(response) if response else "Electronics design assistance complete",
                    "task_status": task_status
                }
            
            logger.info(f"✅ Electronics agent completed: {task_status}")
            return final_response

        except Exception as e:
            logger.error(f"❌ Electronics agent failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._create_error_response(str(e))


# Factory function for lazy loading
_electronics_agent_instance = None

def get_electronics_agent() -> ElectronicsAgent:
    """Get or create electronics agent instance"""
    global _electronics_agent_instance
    if _electronics_agent_instance is None:
        _electronics_agent_instance = ElectronicsAgent()
    return _electronics_agent_instance
