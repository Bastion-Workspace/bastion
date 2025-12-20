"""
General Project Agent for LLM Orchestrator
Handles general project planning, management, and documentation for non-electronics projects
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional, TypedDict, Tuple

from langgraph.graph import StateGraph, END
from .base_agent import BaseAgent, TaskStatus
from orchestrator.utils.editor_operation_resolver import resolve_editor_operation
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
from orchestrator.models.general_project_models import (
    GeneralProjectUnifiedContentPlan, GeneralProjectFileRouteItem,
    GeneralProjectDecision, GeneralProjectDocumentationInconsistency,
    GeneralProjectDocumentationVerificationResult,
    GeneralProjectDocumentationMaintenancePlan
)

logger = logging.getLogger(__name__)


class GeneralProjectState(TypedDict):
    """State for general project agent LangGraph workflow"""
    query: str
    user_id: str
    metadata: Dict[str, Any]
    messages: List[Any]
    query_type: str
    query_intent: str  # "informational" or "action" - determined by LLM
    search_needed: bool
    documents: List[Dict[str, Any]]
    segments: List[Dict[str, Any]]
    referenced_context: Dict[str, Any]
    information_needs: Optional[Dict[str, Any]]
    search_queries: List[Dict[str, Any]]
    search_quality_assessment: Optional[Dict[str, Any]]
    search_retry_count: int  # Counter to prevent infinite re-search loops
    project_plan_action: Optional[str]
    project_structure_plan: Optional[Dict[str, Any]]
    response: Dict[str, Any]
    save_plan: Optional[Dict[str, Any]]
    web_search_needed: bool
    web_search_explicit: bool
    project_decisions: List[Dict[str, Any]]
    documentation_maintenance_plan: Optional[Dict[str, Any]]
    documentation_verification_result: Optional[Dict[str, Any]]
    pending_save_plan: Optional[Dict[str, Any]]
    task_status: str
    error: str


class GeneralProjectAgent(BaseAgent):
    """
    General Project Intelligence Agent
    Provides project planning, management, and documentation assistance for non-electronics projects
    """

    def __init__(self):
        super().__init__("general_project_agent")
        # Initialize node modules
        from orchestrator.agents.general_project_nodes.general_project_search_nodes import GeneralProjectSearchNodes
        from orchestrator.agents.general_project_nodes.general_project_content_nodes import GeneralProjectContentNodes
        from orchestrator.agents.general_project_nodes.general_project_save_nodes import GeneralProjectSaveNodes
        from orchestrator.agents.general_project_nodes.general_project_decision_nodes import GeneralProjectDecisionNodes
        from orchestrator.agents.general_project_nodes.general_project_plan_nodes import GeneralProjectPlanNodes
        from orchestrator.agents.general_project_nodes.general_project_maintenance_nodes import GeneralProjectMaintenanceNodes
        self.search_nodes = GeneralProjectSearchNodes(self)
        self.content_nodes = GeneralProjectContentNodes(self)
        self.save_nodes = GeneralProjectSaveNodes(self)
        self.decision_nodes = GeneralProjectDecisionNodes(self)
        self.project_plan_nodes = GeneralProjectPlanNodes(self)
        self.maintenance_nodes = GeneralProjectMaintenanceNodes(self)
        logger.info("General Project Agent ready for project planning and management!")
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """
        LangGraph workflow for general project agent.
        
        Core flow: intent -> context -> search -> response -> decisions -> project plan update -> save
        """
        workflow = StateGraph(GeneralProjectState)
        
        # Core nodes
        workflow.add_node("analyze_intent", self._analyze_intent_node)
        workflow.add_node("plan_execution_strategy", self._plan_execution_strategy_node)
        workflow.add_node("load_project_context", self._load_project_context_node)
        workflow.add_node("analyze_information_needs", self.search_nodes.analyze_information_needs_node)
        workflow.add_node("generate_project_aware_queries", self.search_nodes.generate_project_aware_queries_node)
        workflow.add_node("search_content", self.search_nodes.search_content_node)
        workflow.add_node("assess_result_quality", self.search_nodes.assess_result_quality_node)
        workflow.add_node("perform_web_search", self._perform_web_search_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("extract_decisions", self.decision_nodes.extract_decisions_node)
        workflow.add_node("update_project_plan", self.project_plan_nodes.update_project_plan_node)
        workflow.add_node("verify_documentation", self.decision_nodes.verify_documentation_node)
        workflow.add_node("execute_maintenance", self.maintenance_nodes.execute_maintenance_node)
        workflow.add_node("extract_and_route_content", self.content_nodes.extract_and_route_content_node)
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
        
        # Context loading routes to information needs analysis, query generation, or response
        workflow.add_conditional_edges(
            "load_project_context",
            self._route_from_context,
            {
                "analyze_needs": "analyze_information_needs",
                "generate_project_aware_queries": "generate_project_aware_queries",
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
                "perform_web_search": "perform_web_search",
                "re_search": "generate_project_aware_queries",
                "generate": "generate_response"
            }
        )
        
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
        
        return workflow.compile(checkpointer=checkpointer)
    
    def _route_from_intent(self, state: GeneralProjectState) -> str:
        """Route from intent analysis - determine if we need strategic planning"""
        query_type = state.get("query_type", "general")
        query = state.get("query", "")
        query_lower = query.lower()
        
        # Fast path: Simple conversational queries skip search pipeline
        if self._is_simple_query(query):
            logger.info("Simple conversational query detected - skipping search pipeline")
            return "generate"
        
        # Check if this is an approval response to a pending operation
        is_approval = self._is_approval_response(query)
        
        if is_approval:
            # Check if there's a pending save plan from previous turn
            pending_save_plan = state.get("pending_save_plan")
            if pending_save_plan:
                logger.info("User approved pending operation - restoring save plan and resuming")
                return "save"
            
            # Check if there's a pending maintenance plan
            maintenance_plan = state.get("documentation_maintenance_plan") or {}
            maintenance_items = maintenance_plan.get("maintenance_items", [])
            if maintenance_items:
                logger.info("User approved pending maintenance - resuming maintenance execution")
                return "maintenance"
            
            # Check previous response to see if it was asking for permission
            response = state.get("response", {})
            response_text = response.get("response", "") if isinstance(response, dict) else str(response)
            if "would you like" in response_text.lower() or "update" in response_text.lower():
                save_plan = state.get("save_plan")
                if save_plan and save_plan.get("routing"):
                    logger.info("User approved - proceeding with save operation")
                    return "save"
        
        # Use strategic planning for complex queries that might need documentation maintenance
        needs_planning = any(keyword in query for keyword in [
            "replace", "change", "update", "switch", "instead of", "remove", "delete",
            "design", "architecture", "specification", "requirement", "approach"
        ]) or query_type in ["design", "planning", "requirements"]
        
        if needs_planning:
            logger.info("Complex query detected - routing to execution planning")
            return "plan_execution"
        
        return "load_context"
    
    def _route_from_save(self, state: GeneralProjectState) -> str:
        """Route from save_content: resolve operations if editing mode with plan edits, otherwise format response"""
        editing_mode = state.get("editing_mode", False)
        plan_edits = state.get("plan_edits")
        
        if editing_mode and plan_edits and len(plan_edits) > 0:
            return "resolve_operations"
        elif editing_mode or plan_edits:
            return "format"
        else:
            return "end"
    
    def _route_from_execution_plan(self, state: GeneralProjectState) -> str:
        """Route from execution planning - always load context"""
        return "load_context"
    
    def _route_from_project_plan_update(self, state: GeneralProjectState) -> str:
        """Route from project plan update node"""
        project_decisions = state.get("project_decisions", [])
        if project_decisions:
            return "verify"
        return "save"
    
    def _route_from_decisions(self, state: GeneralProjectState) -> str:
        """Route from decision extraction - update project plan if relevant, then verify or save"""
        project_decisions = state.get("project_decisions", [])
        metadata = state.get("metadata", {})
        active_editor = metadata.get("active_editor") or metadata.get("shared_memory", {}).get("active_editor", {})
        has_project_plan = bool(active_editor.get("document_id"))
        
        if has_project_plan:
            query = state.get("query", "")
            response = state.get("response", {})
            response_text = response.get("response", "") if isinstance(response, dict) else str(response)
            messages = state.get("messages", [])
            
            should_update = (
                project_decisions or
                (len(response_text) > 300 and len(messages) <= 10) or
                any(keyword in query.lower() for keyword in ["project", "overview", "summary", "plan", "architecture", "design", "system"])
            )
            
            if should_update:
                logger.info(f"Routing to update_project_plan (decisions: {len(project_decisions)}, early_project: {len(messages) <= 10})")
                return "update_project_plan"
        
        if project_decisions:
            logger.info(f"{len(project_decisions)} decision(s) extracted - routing to verification")
            return "verify"
        
        response = state.get("response", {})
        response_text = response.get("response", "") if isinstance(response, dict) else str(response)
        if len(response_text) > 200:
            return "save"
        
        return "end"
    
    def _route_from_verification(self, state: GeneralProjectState) -> str:
        """Route from verification - execute maintenance if inconsistencies found"""
        verification_result = state.get("documentation_verification_result") or {}
        inconsistencies = verification_result.get("inconsistencies", [])
        maintenance_plan = state.get("documentation_maintenance_plan") or {}
        maintenance_items = maintenance_plan.get("maintenance_items", [])
        
        if inconsistencies or maintenance_items:
            logger.info(f"Documentation issues found - routing to maintenance")
            return "maintenance"
        
        response = state.get("response", {})
        response_text = response.get("response", "") if isinstance(response, dict) else str(response)
        if len(response_text) > 200:
            return "save"
        
        return "end"
    
    def _route_from_context(self, state: GeneralProjectState) -> str:
        """Route based on whether search is needed after loading context"""
        if state.get("search_needed", False):
            # Check if information needs were already analyzed in intent node
            if state.get("information_needs_analyzed", False):
                logger.info("Information needs already analyzed - skipping to query generation")
                return "generate_project_aware_queries"
            else:
                logger.info("Routing to analyze_information_needs")
                return "analyze_needs"
        else:
            logger.info("Routing directly to generate_response")
            return "generate"
    
    def _route_from_quality_assessment(self, state: GeneralProjectState) -> str:
        """Route based on result quality assessment with context-aware thresholds"""
        quality_result = state.get("search_quality_assessment", {})
        quality_score = quality_result.get("quality_score", 0.0)
        needs_web_search = quality_result.get("needs_web_search", False)
        should_re_search = quality_result.get("should_re_search", False)
        web_search_explicit = state.get("web_search_explicit", False)
        
        query = state.get("query", "")
        query_lower = query.lower()
        verification_keywords = [
            "double check", "verify", "confirm", "check if", "is this correct",
            "is this right", "validate", "fact check", "cross reference"
        ]
        is_verification_query = any(keyword in query_lower for keyword in verification_keywords)
        
        # Context-aware threshold: Simple queries need lower threshold
        # Simple queries are short and don't require extensive verification
        is_simple_query = self._is_simple_query(query) or len(query.split()) <= 5
        quality_threshold = 0.65 if is_simple_query else 0.75  # Lower threshold for simple queries
        
        search_retry_count = state.get("search_retry_count", 0)
        max_search_retries = 3  # Maximum number of re-search attempts
        
        # Prevent infinite re-search loops - check retry count for ANY should_re_search case
        if should_re_search:
            if search_retry_count >= max_search_retries:
                logger.warning(f"Max search retries ({max_search_retries}) reached (score: {quality_score:.2f}, should_re_search=True) - routing to web search")
                return "perform_web_search"
            if quality_score < 0.5:
                logger.info(f"Low quality results (score: {quality_score:.2f}, retry {search_retry_count + 1}/{max_search_retries}) - re-searching with refined queries")
            else:
                logger.info(f"Re-search requested (score: {quality_score:.2f}, retry {search_retry_count + 1}/{max_search_retries}) - re-searching with refined queries")
            return "re_search"
        elif needs_web_search or web_search_explicit:
            logger.info("Web search needed - routing directly to web search")
            return "perform_web_search"
        elif is_verification_query and quality_score < 0.7:
            logger.info(f"Verification query with moderate quality (score: {quality_score:.2f}) - routing to web search")
            return "perform_web_search"
        elif quality_score >= quality_threshold:
            logger.info(f"Quality results (score: {quality_score:.2f}, threshold: {quality_threshold:.2f}) - routing to response generation")
            return "generate"
        else:
            # Quality below threshold - consider web search for complex queries, proceed for simple
            if not is_simple_query and quality_score < 0.65:
                logger.info(f"Moderate quality for complex query (score: {quality_score:.2f}) - routing to web search")
                return "perform_web_search"
            else:
                logger.info(f"Moderate quality results (score: {quality_score:.2f}) - routing to response generation")
                return "generate"
    
    def _route_from_response(self, state: GeneralProjectState) -> str:
        """Route based on whether content should be saved or decisions extracted"""
        response = state.get("response", {})
        response_text = response.get("response", "") if isinstance(response, dict) else str(response)
        
        query = state.get("query", "").lower()
        project_plan_action = state.get("project_plan_action")
        
        involves_decisions = any(keyword in query for keyword in [
            "use", "select", "choose", "decide", "replace", "change", "switch",
            "instead of", "design", "architecture", "approach", "requirement"
        ]) or state.get("query_type", "") in ["design", "planning", "requirements"]
        
        # Use LLM's query_intent decision (set during analyze_intent_node)
        query_intent = state.get("query_intent", "action")  # Default to action for safety
        
        should_save = False
        
        if project_plan_action == "create":
            should_save = True
            logger.info("New project created - will save content")
        elif query_intent == "informational":
            should_save = False
            logger.info("LLM classified as informational query - will not save content")
        elif query_intent == "action" and len(response_text) > 200:
            should_save = True
            logger.info(f"LLM classified as action query with substantial content ({len(response_text)} chars) - will save")
        elif query_intent == "action":
            should_save = True
            logger.info("LLM classified as action query - will save content")
        
        if should_save and involves_decisions:
            logger.info("Query involves decisions - routing to extract_decisions")
            return "extract_decisions"
        elif should_save:
            logger.info("Routing to extract_and_route_content")
            return "save"
        else:
            logger.info("No content to save - ending workflow")
            return "end"
    
    def _is_approval_response(self, query: str) -> bool:
        """Check if query is an approval response"""
        approval_keywords = ["yes", "y", "ok", "okay", "proceed", "approved", "go ahead", "sure", "do it"]
        return any(keyword in query.lower() for keyword in approval_keywords)
    
    async def _analyze_intent_node(self, state: GeneralProjectState) -> Dict[str, Any]:
        """
        Analyze query intent, detect follow-ups, and determine project plan needs.
        """
        query = state.get("query", "")
        user_id = state.get("user_id", "")
        metadata = state.get("metadata", {})
        messages = state.get("messages", [])
        
        # Early exit optimization: If we have a pending operation, skip LLM call
        pending_save_plan = state.get("pending_save_plan")
        maintenance_plan = state.get("documentation_maintenance_plan") or {}
        maintenance_items = maintenance_plan.get("maintenance_items", []) if maintenance_plan else []
        
        is_approval = self._is_approval_response(query)
        if is_approval and (pending_save_plan or maintenance_items):
            # Skip LLM call - we already know what to do
            shared_memory = metadata.get("shared_memory", {})
            active_editor = metadata.get("active_editor") or shared_memory.get("active_editor", {})
            has_project_plan = bool(active_editor and active_editor.get("frontmatter", {}).get("type") == "project")
            return {
                "query_type": "general",
                "search_needed": False,
                "project_plan_action": None,
                "query_is_project_related": has_project_plan
            }
        
        try:
            fast_model = self._get_fast_model(state)
            llm = self._get_llm(temperature=0.1, model=fast_model, state=state)
            
            shared_memory = metadata.get("shared_memory", {})
            active_editor = metadata.get("active_editor") or shared_memory.get("active_editor", {})
            has_project_plan = bool(active_editor and active_editor.get("frontmatter", {}).get("type") == "project")
            
            project_title = ""
            project_description = ""
            project_specifications = []
            project_design = []
            if active_editor and has_project_plan:
                frontmatter = active_editor.get("frontmatter", {})
                project_title = frontmatter.get("title", "")
                project_description = frontmatter.get("description", "")
                project_specifications = frontmatter.get("specifications", [])[:10]  # Top 10
                project_design = frontmatter.get("design", [])[:10]  # Top 10
            
            # Optimization: If we have project context, combine intent + information needs analysis
            has_project_context = bool(project_specifications or project_design)
            
            if has_project_context and has_project_plan:
                # Combined analysis: intent + information needs in one call
                context_parts = []
                if project_specifications:
                    context_parts.append(f"- Specifications: {', '.join(str(s) for s in project_specifications)}")
                if project_design:
                    context_parts.append(f"- Design: {', '.join(str(d) for d in project_design)}")
                context_string = '\n'.join(context_parts) if context_parts else "None specified"
                
                prompt = f"""Analyze this general project query and determine what information is needed:

**QUERY**: {query}

**PROJECT CONTEXT**:
- Project title: {project_title}
- Project description: {project_description[:200] if project_description else "N/A"}
{context_string}
- Previous messages: {len(messages) - 1} messages in conversation

**ANALYSIS NEEDED**:
1. **query_type**: design|planning|requirements|research|general
2. **is_follow_up**: Is this a follow-up to previous conversation?
3. **project_plan_action**: "create" if new project, null otherwise
4. **search_needed**: Does this query need semantic search?
5. **information_gaps**: What specific information gaps exist? (list of strings)
6. **relevant_specifications**: Which project specifications are relevant? (list from project context)
7. **relevant_design**: Which project design elements are relevant? (list from project context)
8. **content_type**: "new"|"update"|"both" - Is this for new content or updating existing?
9. **detail_level**: "overview"|"detailed"|"implementation" - What level of detail is needed?
10. **query_intent**: "informational"|"action" - Is this asking for information/status/explanation (informational) or requesting changes/additions/creation (action)?

Examples:
- "What's the project status?" → informational
- "Show me the timeline" → informational
- "Explain the requirements" → informational
- "Add a new milestone" → action
- "Update the budget" → action  
- "Create a task breakdown" → action

Return ONLY valid JSON:
{{
  "query_type": "design",
  "is_follow_up": false,
  "project_plan_action": null,
  "search_needed": true,
  "information_gaps": ["gap1", "gap2"],
  "relevant_specifications": ["spec1"],
  "relevant_design": ["design1"],
  "content_type": "new",
  "detail_level": "detailed",
  "query_intent": "action",
  "reasoning": "Brief explanation"
}}"""
            else:
                # Standard intent analysis only (no project context available)
                prompt = f"""Analyze this general project query:

**QUERY**: {query}

**CONTEXT**:
- Has project plan open: {has_project_plan}
- Project title: {project_title}
- Project description: {project_description[:200] if project_description else "N/A"}
- Previous messages: {len(messages) - 1} messages in conversation

**ANALYSIS NEEDED**:
1. **query_type**: design|planning|requirements|research|general
2. **is_follow_up**: Is this a follow-up to previous conversation?
3. **project_plan_action**: "create" if new project, null otherwise
4. **search_needed**: Does this query need semantic search?
5. **query_intent**: "informational"|"action" - Is this asking for information/status/explanation (informational) or requesting changes/additions/creation (action)?

Examples:
- "What's the project status?" → informational
- "Show me the timeline" → informational
- "Explain the requirements" → informational
- "Add a new milestone" → action
- "Update the budget" → action
- "Create a task breakdown" → action

**NOTE**: All queries routed to general_project_agent are project-oriented by definition.

Return ONLY valid JSON:
{{
  "query_type": "design",
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
                        "relevant_specifications": {"type": "array", "items": {"type": "string"}},
                        "relevant_design": {"type": "array", "items": {"type": "string"}},
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
                response = await llm.ainvoke([{"role": "user", "content": prompt}])
                content = response.content if hasattr(response, 'content') else str(response)
                result_dict = self._extract_json_from_response(content) or {}
            
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
                    "relevant_specifications": result_dict.get("relevant_specifications", []),
                    "relevant_design": result_dict.get("relevant_design", []),
                    "content_type": result_dict.get("content_type", "new"),
                    "detail_level": result_dict.get("detail_level", "detailed"),
                    "related_sections": [],
                    "search_focus": result_dict.get("reasoning", query)
                }
                return_dict["information_needs_analyzed"] = True  # Flag to skip separate node
            
            return return_dict
            
        except Exception as e:
            logger.error(f"Intent analysis failed: {e}")
            shared_memory = metadata.get("shared_memory", {})
            active_editor = metadata.get("active_editor") or shared_memory.get("active_editor", {})
            has_project_plan = bool(active_editor and active_editor.get("frontmatter", {}).get("type") == "project")
            
            return {
                "query_type": "general",
                "search_needed": False,
                "project_plan_action": None,
                "query_is_project_related": has_project_plan
            }
    
    async def _plan_execution_strategy_node(self, state: GeneralProjectState) -> Dict[str, Any]:
        """
        Plan execution strategy for complex queries that might need documentation maintenance.
        """
        try:
            query = state.get("query", "")
            query_type = state.get("query_type", "general")
            
            # For general projects, execution planning is simpler - just check if maintenance might be needed
            needs_maintenance = any(keyword in query.lower() for keyword in [
                "replace", "change", "update", "switch", "instead of", "remove", "delete",
                "correct", "fix", "wrong"
            ])
            
            if needs_maintenance:
                logger.info("Query may require documentation maintenance")
                return {
                    "documentation_maintenance_plan": {
                        "maintenance_items": [],
                        "priority": "medium",
                        "reasoning": "Query indicates potential documentation updates needed"
                    }
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Execution planning failed: {e}")
            return {}
    
    async def _load_project_context_node(self, state: GeneralProjectState) -> Dict[str, Any]:
        """
        Load project context efficiently with lazy loading.
        Only loads if query is project-related and editor_preference is not "ignore".
        """
        try:
            user_id = state.get("user_id", "")
            metadata = state.get("metadata", {})
            shared_memory = metadata.get("shared_memory", {})
            
            editor_preference = shared_memory.get("editor_preference", "prefer")
            if editor_preference == "ignore":
                logger.info(f"Skipping project context - editor_preference is 'ignore'")
                return {
                    "referenced_context": {}
                }
            
            active_editor = metadata.get("active_editor") or shared_memory.get("active_editor", {})
            
            if not active_editor or active_editor.get("frontmatter", {}).get("type", "").lower() != "project":
                logger.info(f"Skipping project context - no project plan open")
                return {
                    "referenced_context": {}
                }
            
            # Load referenced files from frontmatter
            from orchestrator.tools.reference_file_loader import load_referenced_files
            
            # General project reference configuration
            reference_config = {
                "specifications": ["specifications", "spec", "specs", "specification"],
                "design": ["design", "design_docs", "architecture"],
                "tasks": ["tasks", "task", "todo", "checklist"],
                "notes": ["notes", "note", "documentation", "docs"],
                "other": ["references", "reference", "files", "related", "documents"]
            }
            
            result = await load_referenced_files(
                active_editor=active_editor,
                user_id=user_id,
                reference_config=reference_config,
                doc_type_filter="project"
            )
            
            referenced_context = result.get("loaded_files", {})
            
            logger.info(f"Loaded {len(referenced_context)} referenced files")
            
            # Detect editing mode: check if project plan has content after frontmatter
            editor_content = active_editor.get("content", "")
            editing_mode = False
            if editor_content:
                frontmatter_end = self._get_frontmatter_end(editor_content)
                has_content_after_frontmatter = len(editor_content) > frontmatter_end + 10
                editing_mode = has_content_after_frontmatter
            
            logger.info(f"General project agent mode: {'EDITING' if editing_mode else 'GENERATION'}")
            
            return {
                "referenced_context": referenced_context,
                "editing_mode": editing_mode,
                "editor_operations": [],
                "plan_edits": None
            }
            
        except Exception as e:
            logger.error(f"Context loading failed: {e}")
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
                "original_text": original_text[:200] if len(original_text) > 200 else original_text,
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
                "anchor_text": anchor_text[-50:] if len(anchor_text) > 50 else anchor_text,
                "left_context": content[max(0, anchor_pos-50):anchor_pos],
                "right_context": content[anchor_pos:min(len(content), anchor_pos+50)]
            }
        
        return None
    
    # Removed: _resolve_operation_simple - now using centralized resolver from orchestrator.utils.editor_operation_resolver
    
    async def _resolve_plan_operations_node(self, state: GeneralProjectState) -> Dict[str, Any]:
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
            
            # Check if file is empty (only frontmatter)
            body_only = editor_content[frontmatter_end:] if frontmatter_end < len(editor_content) else ""
            is_empty_file = not body_only.strip()
            
            editor_operations = []
            
            for edit in plan_edits:
                operation = self._convert_section_edit_to_operation(edit, editor_content, frontmatter_end)
                if not operation:
                    continue
                
                try:
                    # Use centralized resolver
                    resolved_start, resolved_end, resolved_text, resolved_confidence = resolve_editor_operation(
                        content=editor_content,
                        op_dict=operation,
                        selection=None,
                        frontmatter_end=frontmatter_end,
                        cursor_offset=None
                    )
                    
                    # Special handling for empty files: ensure operations insert after frontmatter
                    if is_empty_file and resolved_start < frontmatter_end:
                        resolved_start = frontmatter_end
                        resolved_end = frontmatter_end
                        resolved_confidence = 0.7
                        logger.info(f"Empty file detected - adjusting operation to insert after frontmatter at {frontmatter_end}")
                    
                    import hashlib
                    pre_slice = editor_content[resolved_start:resolved_end]
                    pre_hash = hashlib.sha256(pre_slice.encode()).hexdigest()[:16]
                    
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
    
    async def _format_response_node(self, state: GeneralProjectState) -> Dict[str, Any]:
        """Format final response with editor operations if in editing mode"""
        try:
            editing_mode = state.get("editing_mode", False)
            editor_operations = state.get("editor_operations", [])
            saved_files = state.get("saved_files", [])
            
            response = state.get("response", {})
            response_text = response.get("response", "") if isinstance(response, dict) else str(response)
            
            if editing_mode and editor_operations:
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
    
    async def _perform_web_search_node(self, state: GeneralProjectState) -> Dict[str, Any]:
        """Perform web search when needed"""
        try:
            query = state.get("query", "")
            search_queries = state.get("search_queries", [])
            existing_documents = state.get("documents", [])
            
            # Extract queries from search_queries
            query_list = []
            for sq in search_queries[:3]:
                if isinstance(sq, dict):
                    query_list.append(sq.get("query", ""))
                else:
                    query_list.append(str(sq))
            query_list = [q for q in query_list if q.strip()]
            
            if not query_list:
                query_list = [query]
            
            logger.info(f"Performing web search with {len(query_list)} queries")
            
            # Perform web search - loop through queries and call search_web_structured for each
            web_documents = []
            try:
                for search_query in query_list:
                    logger.info(f"Executing web search: {search_query[:100]}")
                    web_search_result = await search_web_structured(
                        query=search_query,
                        max_results=5
                    )
                    
                    if web_search_result and len(web_search_result) > 0:
                        logger.info(f"Found {len(web_search_result)} web results")
                        
                        # Add web results as documents
                        for i, result in enumerate(web_search_result[:3]):  # Limit to 3 web results per query
                            if isinstance(result, dict):
                                web_doc = {
                                    "document_id": f"web_{len(web_documents)}",
                                    "title": result.get('title', f'Web Result {len(web_documents)+1}'),
                                    "filename": result.get('url', ''),
                                    "content": result.get('snippet', '') or result.get('content', ''),
                                    "tags": ["web", "general_project"],
                                    "category": "web_search",
                                    "source": "web"
                                }
                                web_documents.append(web_doc)
                        break  # Got results, no need to try more queries
                        
            except Exception as e:
                logger.error(f"Web search failed: {e}")
            
            # Combine existing documents with web documents
            all_documents = existing_documents + web_documents
            logger.info(f"Web search returned {len(web_documents)} new documents (total: {len(all_documents)})")
            
            return {
                "documents": all_documents,
                "web_search_needed": False
            }
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return {
                "documents": state.get("documents", []),
                "web_search_needed": False
            }
    
    async def _generate_response_node(self, state: GeneralProjectState) -> Dict[str, Any]:
        """Generate LLM response based on search results and project context"""
        try:
            query = state.get("query", "")
            documents = state.get("documents", [])
            query_type = state.get("query_type", "general")
            metadata = state.get("metadata", {})
            referenced_context = state.get("referenced_context", {})
            segments = state.get("segments", [])
            
            # Build system prompt
            system_prompt = self._build_general_project_prompt()
            
            # Build context from documents and referenced files
            context_parts = []
            
            if segments:
                for seg in segments[:10]:
                    context_parts.append(f"**From {seg.get('filename', 'document')}**:\n{seg.get('content', '')[:500]}")
            
            if documents:
                for doc in documents[:10]:
                    title = doc.get("title", "Document")
                    content = doc.get("content", "")
                    source = doc.get("source", "unknown")
                    context_parts.append(f"**{title}** ({source}):\n{content[:500]}")
            
            context_text = "\n\n".join(context_parts) if context_parts else "No specific project context available."
            
            # Build user prompt
            user_prompt = f"""**USER QUERY**: {query}

**PROJECT CONTEXT**:
{context_text}

**TASK**: Provide helpful, practical guidance for this general project query. Be proactive, ask clarifying questions, and suggest concrete next steps.

**INSTRUCTIONS**:
- Provide practical, implementable solutions
- Reference project context when available
- Ask questions to clarify requirements
- Suggest specific next steps
- Be conversational and helpful
- Structure your response as natural language (not JSON)"""
            
            # Build messages with conversation history using standardized helper
            state_messages = state.get("messages", [])
            llm_messages = self._build_conversational_agent_messages(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                messages_list=state_messages,
                look_back_limit=10
            )
            
            logger.info("Calling LLM for general project response")
            
            # Get LLM response
            llm = self._get_llm(temperature=0.7, state=state)
            response = await llm.ainvoke(llm_messages)
            
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "response": {
                    "task_status": "complete",
                    "response": response_text,
                    "query_type": query_type,
                    "confidence": 0.8
                }
            }
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return {
                "response": {
                    "task_status": "error",
                    "response": f"I encountered an error while generating a response: {str(e)}",
                    "error": str(e)
                }
            }
    
    def _build_general_project_prompt(self) -> str:
        """Build system prompt for general project queries"""
        return """You are a General Project Management Agent - an expert in project planning, design, and documentation for a wide variety of projects (HVAC, landscaping, gardening, home improvement, etc.).

**BEST PRACTICES FOR GENERAL PROJECT MANAGEMENT**:

**PROJECT STRUCTURE HIERARCHY**:
- **PROJECT PLAN = SOURCE OF TRUTH**: The project plan (project_plan.md) is the authoritative document containing:
  - Project goals, scope, and requirements
  - High-level design and approach
  - Project timeline and milestones
  - Key decisions and tradeoffs
  - References to detailed files (specifications, design docs, tasks, notes, etc.)
  
- **REFERENCED FILES = SPECIFIC DETAILS**: Referenced files contain:
  - Detailed specifications
  - Design documents and diagrams
  - Task lists and checklists
  - Project notes and documentation
  - Specific aspects relevant to the project plan

**ROUTING RULES**:
- **System-level/overarching content** -> Project Plan (goals, requirements, high-level design, decisions)
- **Specific detailed content** -> Referenced Files (specifications, design details, tasks, notes)
- **Unclear/fallback** -> Project Plan (when in doubt, use the source of truth)

1. **ALWAYS USE EXISTING PROJECT FILES**: When you have referenced project files, ALWAYS prefer updating those existing files over suggesting new ones.

2. **ALWAYS UPDATE FILES BASED ON USER INPUT**: When the user provides information, automatically update the appropriate project files.

3. **STAY CURRENT WITH PROJECT FILES**: Use project context to provide context-aware responses.

4. **INTELLIGENT CONTENT ROUTING**: Route content to the most appropriate file based on content scope and type.

5. **PROACTIVE UPDATES**: When you provide information, automatically save/update the appropriate project files.

**MISSION**: Provide practical project planning, design, and management assistance for general projects.

**CAPABILITIES**:
1. **Project Planning**: Requirements gathering, scope definition, timeline planning
2. **Design Assistance**: Design guidance, approach recommendations, tradeoff analysis
3. **Documentation**: Organize project information, maintain project files
4. **Research**: Find relevant information for project planning and design
5. **Task Management**: Help organize tasks, checklists, and project milestones

**RESPONSE GUIDELINES**:
- Use a practical, helpful tone
- Include specific recommendations and next steps
- Ask clarifying questions to better understand requirements
- Reference project context when available
- Be proactive in suggesting concrete actions
- Structure responses naturally and conversationally"""

    
    def _extract_json_from_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response"""
        import json
        import re
        
        json_text = content.strip()
        if '```json' in json_text:
            match = re.search(r'```json\s*\n(.*?)\n```', json_text, re.DOTALL)
            if match:
                json_text = match.group(1).strip()
        elif '```' in json_text:
            match = re.search(r'```\s*\n(.*?)\n```', json_text, re.DOTALL)
            if match:
                json_text = match.group(1).strip()
        
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            return None
    
    async def process(self, query: str, metadata: Dict[str, Any] = None, messages: List[Any] = None) -> Dict[str, Any]:
        """Process general project query using LangGraph workflow"""
        try:
            logger.info(f"General project agent processing: {query[:100]}...")

            # Extract user_id from metadata
            metadata = metadata or {}
            user_id = metadata.get("user_id", "system")

            # Add current user query to messages for checkpoint persistence
            conversation_messages = self._prepare_messages_with_query(messages, query)
            
            # Get workflow to access checkpoint
            workflow = await self._get_workflow()
            config = self._get_checkpoint_config(metadata)
            
            # Check if this is an approval response - if so, check for pending operations
            is_approval = self._is_approval_response(query)
            
            pending_save_plan = None
            if is_approval:
                # Try to restore pending_save_plan from checkpoint
                pending_save_plan = await self._restore_pending_operation_from_checkpoint(
                    workflow, config, "pending_save_plan"
                )
            
            # Load and merge checkpointed messages to preserve conversation history
            conversation_messages = await self._load_and_merge_checkpoint_messages(
                workflow, config, conversation_messages
            )
            
            # Load shared_memory from checkpoint if available
            checkpoint_state = await workflow.aget_state(config)
            existing_shared_memory = {}
            if checkpoint_state and checkpoint_state.values:
                existing_shared_memory = checkpoint_state.values.get("shared_memory", {})
            
            # Merge with any shared_memory from metadata
            shared_memory = metadata.get("shared_memory", {}) or {}
            shared_memory.update(existing_shared_memory)
            metadata["shared_memory"] = shared_memory
            
            # Initialize state for LangGraph workflow
            initial_state: GeneralProjectState = {
                "query": query,
                "user_id": user_id,
                "metadata": metadata,
                "messages": conversation_messages,
                "query_type": "",
                "search_needed": False,
                "documents": [],
                "segments": [],
                "referenced_context": {},
                "information_needs": None,
                "search_queries": [],
                "search_quality_assessment": None,
                "search_retry_count": 0,
                "project_plan_action": None,
                "project_structure_plan": None,
                "response": {},
                "save_plan": None,
                "web_search_needed": False,
                "web_search_explicit": False,
                "project_decisions": [],
                "documentation_maintenance_plan": None,
                "documentation_verification_result": None,
                "pending_save_plan": pending_save_plan,
                "editing_mode": False,
                "editor_operations": [],
                "manuscript_edit": None,
                "plan_edits": None,
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
                logger.error(f"General project agent failed: {error_msg}")
                return self._create_error_response(error_msg)
            
            # Build final response dict
            if isinstance(response, dict):
                final_response = {
                    "response": response.get("response", "General project assistance complete"),
                    "task_status": task_status
                }
                
                # Extract editor_operations and manuscript_edit from state (for inline editing)
                editor_operations = result_state.get("editor_operations", [])
                manuscript_edit = result_state.get("manuscript_edit")
                
                # Add editor_operations and manuscript_edit if present
                if editor_operations:
                    final_response["editor_operations"] = editor_operations
                if manuscript_edit:
                    final_response["manuscript_edit"] = manuscript_edit
                
                # Add allowed keys from response dict
                allowed_keys = {"query_type", "confidence", "design_type", "components", "code_snippets", "calculations", "recommendations", "editor_operations", "manuscript_edit", "saved_files"}
                for key, value in response.items():
                    if key == "response":
                        continue
                    elif key.lower() in {k.lower() for k in allowed_keys}:
                        final_response[key] = value
            else:
                final_response = {
                    "response": str(response) if response else "General project assistance complete",
                    "task_status": task_status
                }
            
            logger.info(f"General project agent completed: {task_status}")
            return final_response

        except Exception as e:
            logger.error(f"General project agent failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._create_error_response(str(e))


# Singleton pattern for agent instance
_general_project_agent_instance = None

def get_general_project_agent() -> GeneralProjectAgent:
    """Get or create general project agent instance"""
    global _general_project_agent_instance
    if _general_project_agent_instance is None:
        _general_project_agent_instance = GeneralProjectAgent()
    return _general_project_agent_instance

