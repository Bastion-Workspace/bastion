"""
Technical Hyperspace Agent
Handles system design, topology modeling, and failure simulation
"""

import logging
import json
from typing import Dict, Any, List, Optional, TypedDict

from langgraph.graph import StateGraph, END
from .base_agent import BaseAgent, TaskStatus
from orchestrator.models.editor_models import EditorOperation
from orchestrator.utils.editor_operation_resolver import resolve_editor_operation
from orchestrator.tools.system_modeling_tools import (
    design_system_component_tool,
    simulate_system_failure_tool,
    get_system_topology_tool
)
from orchestrator.tools.visualization_tools import create_chart_tool
from orchestrator.tools.document_editing_tools import propose_document_edit_tool, update_document_content_tool
from orchestrator.utils.frontmatter_utils import add_to_frontmatter_list
from orchestrator.tools.document_tools import get_document_content_tool
from orchestrator.tools.file_creation_tools import create_user_file_tool

logger = logging.getLogger(__name__)


class TechnicalHyperspaceState(TypedDict):
    """State for Technical Hyperspace agent LangGraph workflow"""
    query: str
    user_id: str
    metadata: Dict[str, Any]
    messages: List[Any]
    shared_memory: Dict[str, Any]
    intent: str  # "design" or "simulate" or "analyze"
    component_definitions: List[Dict[str, Any]]  # Components to design
    simulation_request: Optional[Dict[str, Any]]  # Simulation parameters
    topology_data: Optional[Dict[str, Any]]  # Current system topology
    simulation_results: Optional[Dict[str, Any]]  # Simulation results
    diagram_result: Optional[Dict[str, Any]]  # Diagram generation result
    chart_result: Optional[Dict[str, Any]]  # Chart generation result
    system_gaps: List[Dict[str, Any]]  # Missing data or ambiguous logic
    pending_questions: List[str]  # Questions for the user
    editor_operations: List[Dict[str, Any]]  # Editor operations for inline editing
    save_error: str  # Error message when editor is missing but components need to be saved
    response: Dict[str, Any]
    task_status: str
    error: str


class TechnicalHyperspaceAgent(BaseAgent):
    """
    Technical Hyperspace Agent
    Provides system design assistance and deterministic failure simulation
    """
    
    def __init__(self):
        super().__init__("technical_hyperspace_agent")
        logger.info("Technical Hyperspace Agent initialized")
    
    async def process(self, query: str, metadata: Dict[str, Any] = None, messages: List[Any] = None) -> Dict[str, Any]:
        """Standard process method for Technical Hyperspace agent with elite history management"""
        try:
            # Get workflow
            workflow = await self._get_workflow()
            
            # Extract user_id
            metadata = metadata or {}
            user_id = metadata.get("user_id", "system")
            
            # Get checkpoint config
            config = self._get_checkpoint_config(metadata)
            
            # Prepare messages
            new_messages = self._prepare_messages_with_query(messages, query)
            conversation_messages = await self._load_and_merge_checkpoint_messages(workflow, config, new_messages)
            
            # Load shared memory from checkpoint if available
            checkpoint_state = await workflow.aget_state(config)
            existing_shared_memory = {}
            if checkpoint_state and checkpoint_state.values:
                existing_shared_memory = checkpoint_state.values.get("shared_memory", {})
            
            # Merge with any shared_memory from metadata
            shared_memory = metadata.get("shared_memory", {}) or {}
            shared_memory.update(existing_shared_memory)
            
            # Initialize state
            initial_state: TechnicalHyperspaceState = {
                "query": query,
                "user_id": user_id,
                "metadata": metadata,
                "messages": conversation_messages,
                "shared_memory": shared_memory,
                "intent": "analyze",
                "component_definitions": [],
                "simulation_request": None,
                "topology_data": None,
                "simulation_results": None,
                "diagram_result": None,
                "chart_result": None,
                "system_gaps": [],
                "pending_questions": [],
                "editor_operations": [],
                "save_error": "",
                "response": {},
                "task_status": "",
                "error": ""
            }
            
            # Run workflow with checkpointing
            result_state = await workflow.ainvoke(initial_state, config=config)
            
            # Extract response
            response = result_state.get("response", {})
            task_status = result_state.get("task_status", "complete")
            
            if task_status == "error":
                error_msg = result_state.get("error", "Unknown error")
                logger.error(f"❌ Technical Hyperspace Agent failed: {error_msg}")
                return self._create_error_response(error_msg)
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Technical Hyperspace Agent failed: {e}")
            return self._create_error_response(str(e))

    def _get_diagramming_subgraph(self, checkpointer):
        """Get or build diagramming subgraph"""
        if not hasattr(self, '_diagramming_subgraph') or self._diagramming_subgraph is None:
            from orchestrator.subgraphs import build_diagramming_subgraph
            self._diagramming_subgraph = build_diagramming_subgraph(checkpointer)
        return self._diagramming_subgraph
    
    async def _call_diagramming_subgraph_node(self, state: TechnicalHyperspaceState) -> Dict[str, Any]:
        """Call diagramming subgraph to generate failure path diagrams"""
        try:
            logger.info("Calling diagramming subgraph for failure visualization")
            
            workflow = await self._get_workflow()
            checkpointer = workflow.checkpointer
            diagramming_sg = self._get_diagramming_subgraph(checkpointer)
            
            query = state.get("query", "")
            metadata = state.get("metadata", {})
            messages = state.get("messages", [])
            shared_memory = state.get("shared_memory", {})
            simulation_results = state.get("simulation_results", {})
            
            # Build diagram request from simulation results
            diagram_query = f"Create a Mermaid diagram showing the failure cascade paths: {json.dumps(simulation_results.get('failure_paths', []))}"
            
            diagram_state = {
                "query": diagram_query,
                "messages": messages,
                "metadata": metadata,
                "shared_memory": shared_memory,
                "user_id": state.get("user_id", "system"),
                "project_context": {
                    "simulation_results": simulation_results,
                    "topology_data": state.get("topology_data", {})
                }
            }
            
            config = self._get_checkpoint_config(metadata)
            result = await diagramming_sg.ainvoke(diagram_state, config)
            
            return {
                "diagram_result": result.get("diagram_result", {}),
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
            
        except Exception as e:
            logger.error(f"Diagramming subgraph failed: {e}")
            return {
                "diagram_result": {"success": False, "error": str(e)},
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """Build LangGraph workflow for Technical Hyperspace agent"""
        workflow = StateGraph(TechnicalHyperspaceState)
        
        # Core nodes
        workflow.add_node("analyze_intent", self._analyze_intent_node)
        workflow.add_node("inspect_system_gaps", self._inspect_system_gaps_node)
        workflow.add_node("design_components", self._design_components_node)
        workflow.add_node("save_to_editor", self._save_to_editor_node)  # Save component definitions to editor
        workflow.add_node("get_topology", self._get_topology_node)
        workflow.add_node("simulate_failure", self._simulate_failure_node)
        workflow.add_node("generate_diagrams", self._generate_diagrams_node)
        workflow.add_node("generate_charts", self._generate_charts_node)
        workflow.add_node("format_response", self._format_response_node)
        
        # Set entry point
        workflow.set_entry_point("analyze_intent")
        
        # Define edges with conditional routing
        workflow.add_conditional_edges(
            "analyze_intent",
            self._route_from_intent,
            {
                "design": "inspect_system_gaps",
                "simulate": "get_topology",
                "analyze": "get_topology",
                "format": "format_response"
            }
        )
        
        # Conditional edge for gaps
        workflow.add_conditional_edges(
            "inspect_system_gaps",
            lambda state: "ask_user" if state.get("pending_questions") else "continue",
            {
                "ask_user": "format_response",
                "continue": "design_components"
            }
        )
        
        workflow.add_edge("design_components", "save_to_editor")
        workflow.add_edge("save_to_editor", "format_response")
        workflow.add_edge("get_topology", "simulate_failure")
        workflow.add_edge("simulate_failure", "generate_diagrams")
        workflow.add_edge("generate_diagrams", "generate_charts")
        workflow.add_edge("generate_charts", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow.compile(checkpointer=checkpointer)
    
    def _route_from_intent(self, state: TechnicalHyperspaceState) -> str:
        """Route based on intent"""
        intent = state.get("intent", "format")
        if intent == "design":
            return "design"
        elif intent == "simulate":
            return "simulate"
        elif intent == "analyze":
            return "analyze"
        else:
            return "format"
    
    async def _analyze_intent_node(self, state: TechnicalHyperspaceState) -> Dict[str, Any]:
        """Analyze user intent: design, simulate, or analyze"""
        try:
            query = state.get("query", "")
            messages = state.get("messages", [])
            
            # Check if this is an answer to a previous question
            last_message = messages[-2] if len(messages) >= 2 else None
            is_answer = False
            if last_message and hasattr(last_message, 'content') and "?" in str(last_message.content):
                is_answer = True
            
            llm = self._get_llm(temperature=0.3, state=state)
            
            system_prompt = """You are a Technical Hyperspace assistant. Analyze the user's query to determine intent:
- "design": User wants to define/create system components (e.g., "Build a cardboard engine")
- "simulate": User wants to simulate component failures
- "analyze": User wants to analyze existing topology

STRATEGIC DESIGN INSTRUCTION:
For "design" intent, be PROACTIVE! If the user provides a high-level concept (e.g., "2-cylinder engine"), use your knowledge to break it down into logical sub-components (cylinders, pistons, crank, etc.). Derive reasonable properties for the materials mentioned (e.g., cardboard's structural limits).

Respond with JSON: {"intent": "design|simulate|analyze", "component_definitions": [...], "simulation_request": {...}}
For design: extract or DERIVE component definitions with id, type, requires, provides, redundancy_group, criticality, dependency_logic, m_of_n_threshold. Use metadata for material properties and constraints."""
            
            prompt = f"{system_prompt}\n\nUser query: {query}"
            
            response = await llm.ainvoke([{"role": "system", "content": system_prompt}, {"role": "user", "content": query}])
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                intent = parsed.get("intent", "format")
                component_definitions = parsed.get("component_definitions", [])
                simulation_request = parsed.get("simulation_request")
            else:
                # Fallback: simple keyword matching
                query_lower = query.lower()
                if "design" in query_lower or "define" in query_lower or "create" in query_lower or is_answer:
                    intent = "design"
                    component_definitions = []
                    simulation_request = None
                elif "simulate" in query_lower or "failure" in query_lower or "fail" in query_lower:
                    intent = "simulate"
                    component_definitions = []
                    simulation_request = {"failed_component_ids": [], "failure_modes": ["random"]}
                else:
                    intent = "analyze"
                    component_definitions = []
                    simulation_request = None
            
            return {
                "intent": intent,
                "component_definitions": component_definitions,
                "simulation_request": simulation_request,
                "system_gaps": [],
                "pending_questions": [],
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
            
        except Exception as e:
            logger.error(f"Intent analysis failed: {e}")
            return {
                "intent": "format",
                "component_definitions": [],
                "simulation_request": None,
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    async def _inspect_system_gaps_node(self, state: TechnicalHyperspaceState) -> Dict[str, Any]:
        """Inspect component definitions and proactively fill in derived engineering defaults"""
        try:
            component_definitions = state.get("component_definitions", [])
            if not component_definitions:
                return {
                    "system_gaps": [],
                    "pending_questions": [],
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", "")
                }
            
            llm = self._get_llm(temperature=0.0, state=state)
            
            inspector_prompt = f"""You are a Technical Hyperspace Strategic Inspector. Analyze these component definitions.
Instead of just reporting gaps, use your engineering knowledge to PROPOSE and DERIVE reasonable defaults for missing properties based on materials and system context.

STRATEGIC DERIVATION RULES:
1. **Material Logic**: If material is "cardboard" or "duct tape", derive moisture sensitivity, heat tolerance, and structural degradation thresholds automatically.
2. **System Defaults**: For mechanical systems like "engines", assume WEIGHTED_INTEGRITY for complex assemblies and suggest operational ranges (RPM, Temp, Pressure) if not provided.
3. **Redundancy**: Propose redundancy logic (like M_OF_N) for critical components like spark plugs or valves if it makes engineering sense.
4. **Bust the Questions**: Only ask a question if there is a "Tactical Ambiguity" that is critical and cannot be reasonably inferred from your knowledge base.

Components: {json.dumps(component_definitions)}

Respond with JSON:
{{
  "updated_definitions": [... updated definitions with derived values added ...],
  "gaps": [{{ "component_id": "...", "gap_type": "...", "description": "..." }}],
  "questions": ["Only critical questions that cannot be derived"]
}}"""
            
            response = await llm.ainvoke([{"role": "user", "content": inspector_prompt}])
            content = response.content if hasattr(response, 'content') else str(response)
            
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            gaps = []
            questions = []
            updated_definitions = component_definitions
            
            if json_match:
                parsed = json.loads(json_match.group())
                updated_definitions = parsed.get("updated_definitions", component_definitions)
                gaps = parsed.get("gaps", [])
                questions = parsed.get("questions", [])
            
            return {
                "component_definitions": updated_definitions,
                "system_gaps": gaps,
                "pending_questions": questions,
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
            
        except Exception as e:
            logger.error(f"Gap inspection failed: {e}")
            return {
                "system_gaps": [],
                "pending_questions": [],
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }

    async def _design_components_node(self, state: TechnicalHyperspaceState) -> Dict[str, Any]:
        """Design system components"""
        try:
            component_definitions = state.get("component_definitions", [])
            user_id = state.get("user_id", "system")
            
            if not component_definitions:
                # Try to extract from query using LLM
                query = state.get("query", "")
                llm = self._get_llm(temperature=0.3, state=state)
                
                extract_prompt = f"""Extract system component definitions from this query. Return JSON array:
[{{"component_id": "...", "component_type": "...", "requires": [...], "provides": [...], "redundancy_group": "...", "criticality": 1-5, "dependency_logic": "AND|OR|MAJORITY|WEIGHTED_INTEGRITY", "m_of_n_threshold": 0, "dependency_weights": {{ "dep_id": 0.5 }}, "integrity_threshold": 0.7}}]

Query: {query}"""
                
                response = await llm.ainvoke([{"role": "user", "content": extract_prompt}])
                content = response.content if hasattr(response, 'content') else str(response)
                
                import re
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    component_definitions = json.loads(json_match.group())
            
            # Design each component
            results = []
            for comp_def in component_definitions:
                result = await design_system_component_tool(
                    component_id=comp_def.get("component_id", ""),
                    component_type=comp_def.get("component_type", "component"),
                    requires=comp_def.get("requires", []),
                    provides=comp_def.get("provides", []),
                    redundancy_group=comp_def.get("redundancy_group"),
                    criticality=comp_def.get("criticality", 3),
                    metadata=comp_def.get("metadata", {}),
                    dependency_logic=comp_def.get("dependency_logic", "AND"),
                    m_of_n_threshold=comp_def.get("m_of_n_threshold", 0),
                    dependency_weights=comp_def.get("dependency_weights", {}),
                    integrity_threshold=comp_def.get("integrity_threshold", 0.5),
                    user_id=user_id
                )
                results.append(result)
            
            return {
                "component_definitions": component_definitions,
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "editor_operations": []
            }
            
        except Exception as e:
            logger.error(f"Component design failed: {e}")
            return {
                "component_definitions": [],
                "error": str(e),
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "editor_operations": []
            }
    
    async def _get_topology_node(self, state: TechnicalHyperspaceState) -> Dict[str, Any]:
        """Get current system topology"""
        try:
            user_id = state.get("user_id", "system")
            
            result = await get_system_topology_tool(user_id=user_id)
            
            return {
                "topology_data": result,
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
            
        except Exception as e:
            logger.error(f"Get topology failed: {e}")
            return {
                "topology_data": {},
                "error": str(e),
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    async def _simulate_failure_node(self, state: TechnicalHyperspaceState) -> Dict[str, Any]:
        """Simulate system failure"""
        try:
            simulation_request = state.get("simulation_request")
            user_id = state.get("user_id", "system")
            
            if not simulation_request:
                # Extract from query
                query = state.get("query", "")
                llm = self._get_llm(temperature=0.3, state=state)
                
                extract_prompt = f"""Extract simulation parameters from this query. Return JSON:
{{"failed_component_ids": [...], "failure_modes": [...], "simulation_type": "cascade|monte_carlo", "monte_carlo_iterations": 1000}}

Query: {query}"""
                
                response = await llm.ainvoke([{"role": "user", "content": extract_prompt}])
                content = response.content if hasattr(response, 'content') else str(response)
                
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    simulation_request = json.loads(json_match.group())
                else:
                    simulation_request = {"failed_component_ids": [], "failure_modes": ["random"], "simulation_type": "cascade"}
            
            result = await simulate_system_failure_tool(
                failed_component_ids=simulation_request.get("failed_component_ids", []),
                failure_modes=simulation_request.get("failure_modes", ["random"]),
                simulation_type=simulation_request.get("simulation_type", "cascade"),
                monte_carlo_iterations=simulation_request.get("monte_carlo_iterations"),
                failure_parameters=simulation_request.get("failure_parameters", {}),
                user_id=user_id
            )
            
            return {
                "simulation_results": result,
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
            
        except Exception as e:
            logger.error(f"Failure simulation failed: {e}")
            return {
                "simulation_results": {"success": False, "error": str(e)},
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    async def _generate_diagrams_node(self, state: TechnicalHyperspaceState) -> Dict[str, Any]:
        """Generate failure path diagrams"""
        try:
            simulation_results = state.get("simulation_results", {})
            
            if not simulation_results or not simulation_results.get("success"):
                return {
                    "diagram_result": {"success": False, "message": "No simulation results to diagram"},
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", "")
                }
            
            return await self._call_diagramming_subgraph_node(state)
            
        except Exception as e:
            logger.error(f"Diagram generation failed: {e}")
            return {
                "diagram_result": {"success": False, "error": str(e)},
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    async def _generate_charts_node(self, state: TechnicalHyperspaceState) -> Dict[str, Any]:
        """Generate health metrics charts"""
        try:
            simulation_results = state.get("simulation_results", {})
            
            if not simulation_results or not simulation_results.get("success"):
                return {
                    "chart_result": {"success": False, "message": "No simulation results to chart"},
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", "")
                }
            
            health_metrics = simulation_results.get("health_metrics", {})
            
            # Create bar chart for component states
            chart_data = {
                "operational": health_metrics.get("operational_components", 0),
                "degraded": health_metrics.get("degraded_components", 0),
                "failed": health_metrics.get("failed_components", 0)
            }
            
            result = await create_chart_tool(
                chart_type="bar",
                data_json=json.dumps(chart_data),
                title="System Health After Failure Simulation",
                x_label="Component State",
                y_label="Count",
                interactive=True
            )
            
            return {
                "chart_result": result,
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
            
        except Exception as e:
            logger.error(f"Chart generation failed: {e}")
            return {
                "chart_result": {"success": False, "error": str(e)},
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    async def _save_to_editor_node(self, state: TechnicalHyperspaceState) -> Dict[str, Any]:
        """Save component definitions to active editor using elite Editor Operation Resolver"""
        try:
            metadata = state.get("metadata", {})
            shared_memory = state.get("shared_memory", {})
            active_editor = metadata.get("active_editor") or shared_memory.get("active_editor", {})
            
            # Check if active editor is a systems document
            frontmatter = active_editor.get("frontmatter", {})
            doc_type = str(frontmatter.get("type", "")).strip().lower()
            
            component_definitions = state.get("component_definitions", [])
            
            # ✅ CRITICAL: Base state to preserve in all return paths
            preserved_state = {
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "component_definitions": component_definitions,
                "intent": state.get("intent", "analyze")
            }
            
            # Accept both "system" (singular) and "systems" (plural) for compatibility
            if not active_editor or doc_type not in ["system", "systems"]:
                # No active editor or wrong type - provide user-friendly error if we have components to save
                if component_definitions:
                    error_message = "I've designed the system components, but I need a systems document open to save them. Please open a document with `type: system` or `type: systems` in the frontmatter, then I can save the component definitions."
                    logger.info("No active systems editor - cannot save component definitions")
                    return {
                        **preserved_state,
                        "editor_operations": [],
                        "save_error": error_message
                    }
                else:
                    return {
                        **preserved_state,
                        "editor_operations": []
                    }
            
            if not component_definitions:
                return {
                    **preserved_state,
                    "editor_operations": []
                }
            
            content = active_editor.get("content", "")
            
            # Format component definitions as markdown
            components_markdown = "## System Components\n\n"
            for comp in component_definitions:
                components_markdown += f"### {comp.get('component_id', 'Unknown')}\n\n"
                components_markdown += f"- **Type**: {comp.get('component_type', 'component')}\n"
                components_markdown += f"- **Criticality**: {comp.get('criticality', 3)}/5\n"
                if comp.get('requires'):
                    components_markdown += f"- **Requires**: {', '.join(comp.get('requires', []))}\n"
                if comp.get('provides'):
                    components_markdown += f"- **Provides**: {', '.join(comp.get('provides', []))}\n"
                if comp.get('dependency_logic'):
                    components_markdown += f"- **Dependency Logic**: {comp.get('dependency_logic')}\n"
                if comp.get('integrity_threshold'):
                    components_markdown += f"- **Integrity Threshold**: {comp.get('integrity_threshold')}\n"
                if comp.get('metadata'):
                    components_markdown += f"- **Metadata**: {json.dumps(comp.get('metadata', {}), indent=2)}\n"
                components_markdown += "\n"
            
            # Use elite resolve_editor_operation instead of regex!
            # If "## System Components" exists, replace it. Otherwise, append.
            if "## System Components" in content:
                # Find the existing section
                import re
                section_pattern = r'(## System Components\n.*?)(?=\n## |\Z)'
                match = re.search(section_pattern, content, flags=re.DOTALL)
                if match:
                    original_text = match.group(1)
                    op = EditorOperation(
                        op_type="replace_range",
                        original_text=original_text,
                        text=components_markdown.strip() + "\n\n"
                    )
                else:
                    op = EditorOperation(
                        op_type="insert_after_heading",
                        anchor_text="## System Components",
                        text="\n" + components_markdown.strip() + "\n"
                    )
            else:
                # Append to end of document
                op = EditorOperation(
                    op_type="insert_after",
                    anchor_text=content[-50:] if len(content) > 50 else content,
                    text="\n\n" + components_markdown.strip() + "\n"
                )
            
            # Resolve the operation to get exact positioning
            resolved_ops = resolve_editor_operation(op, content)
            logger.info(f"✅ Resolved {len(resolved_ops)} editor operation(s) for Technical Hyperspace")
            
            return {
                **preserved_state,
                "editor_operations": resolved_ops
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to save to editor: {e}")
            return {
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "editor_operations": [],
                "error": str(e)
            }
    
    async def _format_response_node(self, state: TechnicalHyperspaceState) -> Dict[str, Any]:
        """Format final response with editor operations if available"""
        try:
            intent = state.get("intent", "format")
            simulation_results = state.get("simulation_results", {})
            diagram_result = state.get("diagram_result", {})
            chart_result = state.get("chart_result", {})
            pending_questions = state.get("pending_questions", [])
            editor_operations = state.get("editor_operations", [])
            save_error = state.get("save_error", "")  # Error from save_to_editor if editor missing
            
            response_text = ""
            
            # Check for save error (missing editor when components need to be saved)
            if save_error:
                response_text = save_error
                if intent == "design":
                    response_text += "\n\nThe component definitions are ready and will be saved once you open a systems document."
            elif pending_questions:
                response_text = "I've analyzed your system model, but I need some additional information before we proceed:\n\n"
                for q in pending_questions:
                    response_text += f"- {q}\n"
                response_text += "\nPlease provide these details so we can ensure the model is complete."
            elif intent == "design":
                response_text = "System components designed successfully. Components added to topology."
            elif intent == "simulate" or intent == "analyze":
                if simulation_results and simulation_results.get("success"):
                    health = simulation_results.get("health_metrics", {})
                    response_text = f"""Failure Simulation Results:

System Health Score: {health.get('system_health_score', 0.0):.2%}
Operational Components: {health.get('operational_components', 0)}/{health.get('total_components', 0)}
Degraded Components: {health.get('degraded_components', 0)}
Failed Components: {health.get('failed_components', 0)}

Failure Paths: {len(simulation_results.get('failure_paths', []))} paths identified
Critical Vulnerabilities: {len(health.get('critical_vulnerabilities', []))}

Structural Insights:
"""
                    # Add integrity-based insights
                    llm = self._get_llm(temperature=0.3, state=state)
                    insight_prompt = f"""Interpret these system modeling results. Explain failures in terms of structural integrity loss and logic gates.
Simulation Results: {json.dumps(simulation_results)}
Health Metrics: {json.dumps(health)}"""
                    
                    insight_response = await llm.ainvoke([{"role": "user", "content": insight_prompt}])
                    response_text += insight_response.content if hasattr(insight_response, 'content') else str(insight_response)
                else:
                    err = simulation_results.get('error', 'Unknown error') if simulation_results else 'No simulation results'
                    response_text = f"Simulation failed: {err}"
            
            # If we have editor_operations, include them in response
            metadata = state.get("metadata", {})
            shared_memory = state.get("shared_memory", {})
            active_editor = metadata.get("active_editor") or shared_memory.get("active_editor", {})
            
            response = {
                "response": response_text,
                "simulation_results": simulation_results,
                "diagram_result": diagram_result,
                "chart_result": chart_result,
                "pending_questions": pending_questions,
                "task_status": TaskStatus.COMPLETE.value if not pending_questions else TaskStatus.INCOMPLETE.value
            }
            
            # Add editor operations if available
            if editor_operations and active_editor:
                target_filename = active_editor.get("filename", "system_model.md")
                response["editor_operations"] = editor_operations
                response["manuscript_edit"] = {
                    "target_filename": target_filename,
                    "scope": "section",
                    "summary": f"Updated system components in {target_filename}"
                }
                response_text = "I've updated the system components in your document. Review and accept the changes below.\n\n" + response_text
                response["response"] = response_text
            
            return {
                "response": response,
                "task_status": TaskStatus.COMPLETE.value if not pending_questions else TaskStatus.INCOMPLETE.value,
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "editor_operations": editor_operations
            }
            
        except Exception as e:
            logger.error(f"Response formatting failed: {e}")
            return {
                "response": {"error": str(e)},
                "task_status": TaskStatus.ERROR.value,
                "error": str(e),
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }


_technical_hyperspace_agent_instance = None

def get_technical_hyperspace_agent() -> TechnicalHyperspaceAgent:
    """Get singleton instance of Technical Hyperspace Agent"""
    global _technical_hyperspace_agent_instance
    if _technical_hyperspace_agent_instance is None:
        _technical_hyperspace_agent_instance = TechnicalHyperspaceAgent()
    return _technical_hyperspace_agent_instance
