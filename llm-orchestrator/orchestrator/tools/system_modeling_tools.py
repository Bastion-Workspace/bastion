"""
Technical Hyperspace Tools - gRPC client wrappers for system topology and failure simulation
"""

import logging
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field

from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)


# ── I/O models (minimal for system modeling tools) ──────────────────────────

class DesignSystemComponentInputs(BaseModel):
    """Required inputs for design_system_component_tool."""
    component_id: str = Field(description="Unique component identifier")
    component_type: str = Field(description="Type of component e.g. pump, valve, sensor")


class DesignSystemComponentOutputs(BaseModel):
    """Outputs for design_system_component_tool."""
    success: bool = Field(description="Whether the component was added")
    component_id: str = Field(description="Component ID")
    message: Optional[str] = Field(default=None, description="Status message")
    topology_json: str = Field(description="Updated topology JSON")
    error: Optional[str] = Field(default=None, description="Error if failed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class SimulateSystemFailureInputs(BaseModel):
    """Required inputs for simulate_system_failure_tool."""
    failed_component_ids: List[str] = Field(description="Components to fail initially")


class SimulateSystemFailureOutputs(BaseModel):
    """Outputs for simulate_system_failure_tool."""
    success: bool = Field(description="Whether simulation succeeded")
    simulation_id: str = Field(description="Simulation ID")
    component_states: List[Dict[str, Any]] = Field(default_factory=list, description="Component states")
    failure_paths: List[Dict[str, Any]] = Field(default_factory=list, description="Failure paths")
    health_metrics: Dict[str, Any] = Field(default_factory=dict, description="Health metrics")
    topology_json: str = Field(description="Topology JSON")
    error: Optional[str] = Field(default=None, description="Error if failed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class GetSystemTopologyInputs(BaseModel):
    """Inputs for get_system_topology_tool (optional system_name)."""
    system_name: Optional[str] = Field(default=None, description="Optional system name filter")


class GetSystemTopologyOutputs(BaseModel):
    """Outputs for get_system_topology_tool."""
    success: bool = Field(description="Whether topology was retrieved")
    topology_json: str = Field(description="Topology JSON")
    component_count: int = Field(description="Number of components")
    edge_count: int = Field(description="Number of edges")
    redundancy_groups: List[str] = Field(default_factory=list, description="Redundancy group names")
    error: Optional[str] = Field(default=None, description="Error if failed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def design_system_component_tool(
    component_id: str,
    component_type: str,
    requires: List[str] = None,
    provides: List[str] = None,
    redundancy_group: Optional[str] = None,
    criticality: int = 3,
    metadata: Dict[str, str] = None,
    dependency_logic: str = "AND",
    m_of_n_threshold: int = 0,
    dependency_weights: Dict[str, float] = None,
    integrity_threshold: float = 0.5,
    user_id: str = "system"
) -> Dict[str, Any]:
    """
    Design/add a system component to the topology
    
    Args:
        component_id: Unique component identifier
        component_type: Type of component (e.g., "pump", "valve", "sensor")
        requires: List of component IDs this component depends on
        provides: List of functionalities this component enables
        redundancy_group: Optional redundancy group name
        criticality: Criticality level 1-5 (5 = most critical)
        metadata: Additional component properties
        dependency_logic: Logic for dependencies ("AND", "OR", "MAJORITY", "M_OF_N", "WEIGHTED_INTEGRITY")
        m_of_n_threshold: Threshold for M_OF_N logic
        dependency_weights: Weights for each dependency (for WEIGHTED_INTEGRITY)
        integrity_threshold: Threshold for WEIGHTED_INTEGRITY logic
        user_id: User ID for access control
        
    Returns:
        Dict with success status and updated topology JSON
    """
    try:
        from orchestrator.backend_tool_client import get_backend_tool_client
        
        client = await get_backend_tool_client()
        await client.connect()
        
        from protos import tool_service_pb2
        
        request = tool_service_pb2.DesignSystemComponentRequest(
            user_id=user_id,
            component_id=str(component_id),
            component_type=str(component_type),
            requires=[str(r) for r in (requires or [])],
            provides=[str(p) for p in (provides or [])],
            criticality=int(criticality),
            metadata={str(k): str(v) for k, v in (metadata or {}).items()},
            dependency_logic=str(dependency_logic),
            m_of_n_threshold=int(m_of_n_threshold),
            dependency_weights={str(k): float(v) for k, v in (dependency_weights or {}).items()},
            integrity_threshold=float(integrity_threshold)
        )
        
        if redundancy_group:
            request.redundancy_group = redundancy_group
        
        response = await client._stub.DesignSystemComponent(request)
        formatted = f"Component {response.component_id} designed successfully." if response.success else (response.error or "Failed")
        return {
            "success": response.success,
            "component_id": response.component_id,
            "message": response.message,
            "topology_json": response.topology_json,
            "error": response.error if response.HasField("error") else None,
            "formatted": formatted
        }
        
    except Exception as e:
        logger.error(f"Failed to design system component: {e}")
        err = str(e)
        return {
            "success": False,
            "component_id": component_id,
            "message": f"Failed to design component: {err}",
            "error": err,
            "topology_json": "{}",
            "formatted": f"Failed to design component: {err}"
        }


async def simulate_system_failure_tool(
    failed_component_ids: List[str],
    failure_modes: List[str] = None,
    simulation_type: str = "cascade",
    monte_carlo_iterations: Optional[int] = None,
    failure_parameters: Dict[str, str] = None,
    user_id: str = "system"
) -> Dict[str, Any]:
    """
    Simulate system failure with deterministic cascade propagation
    
    Args:
        failed_component_ids: Components to fail initially
        failure_modes: Types of failures ("sabotage", "maintenance_neglect", "wear", "random")
        simulation_type: "single", "cascade", or "monte_carlo"
        monte_carlo_iterations: Number of iterations for Monte Carlo simulation
        failure_parameters: Additional simulation parameters
        user_id: User ID for access control
        
    Returns:
        Dict with simulation results including component states, failure paths, and health metrics
    """
    try:
        from orchestrator.backend_tool_client import get_backend_tool_client
        
        client = await get_backend_tool_client()
        await client.connect()
        
        from protos import tool_service_pb2
        
        request = tool_service_pb2.SimulateSystemFailureRequest(
            user_id=user_id,
            failed_component_ids=failed_component_ids,
            failure_modes=failure_modes or ["random"],
            simulation_type=simulation_type,
            failure_parameters=failure_parameters or {}
        )
        
        if monte_carlo_iterations:
            request.monte_carlo_iterations = monte_carlo_iterations
        
        response = await client._stub.SimulateSystemFailure(request)
        
        if not response.success:
            err = response.error if response.HasField("error") else "Unknown error"
            return {
                "success": False,
                "simulation_id": response.simulation_id,
                "error": err,
                "topology_json": response.topology_json,
                "component_states": [],
                "failure_paths": [],
                "health_metrics": {},
                "formatted": f"Simulation failed: {err}"
            }
        
        # Convert component states
        component_states = []
        for state in response.component_states:
            component_states.append({
                "component_id": state.component_id,
                "state": state.state,
                "failed_dependencies": list(state.failed_dependencies),
                "failure_probability": state.failure_probability,
                "metadata": dict(state.metadata)
            })
        
        # Convert failure paths
        failure_paths = []
        for path in response.failure_paths:
            failure_paths.append({
                "source_component_id": path.source_component_id,
                "affected_component_ids": list(path.affected_component_ids),
                "failure_type": path.failure_type,
                "path_length": path.path_length
            })
        
        # Convert health metrics
        health = response.health_metrics
        health_metrics = {
            "total_components": health.total_components,
            "operational_components": health.operational_components,
            "degraded_components": health.degraded_components,
            "failed_components": health.failed_components,
            "system_health_score": health.system_health_score,
            "critical_vulnerabilities": list(health.critical_vulnerabilities),
            "redundancy_groups_at_risk": list(health.redundancy_groups_at_risk)
        }
        
        formatted = f"Simulation {response.simulation_id}: {health_metrics.get('operational_components', 0)} operational, {health_metrics.get('failed_components', 0)} failed."
        return {
            "success": True,
            "simulation_id": response.simulation_id,
            "component_states": component_states,
            "failure_paths": failure_paths,
            "health_metrics": health_metrics,
            "topology_json": response.topology_json,
            "formatted": formatted
        }
        
    except Exception as e:
        logger.error(f"Failed to simulate system failure: {e}")
        err = str(e)
        return {
            "success": False,
            "simulation_id": "",
            "error": err,
            "topology_json": "{}",
            "component_states": [],
            "failure_paths": [],
            "health_metrics": {},
            "formatted": f"Simulation failed: {err}"
        }


async def get_system_topology_tool(
    system_name: Optional[str] = None,
    user_id: str = "system"
) -> Dict[str, Any]:
    """
    Get system topology as JSON
    
    Args:
        system_name: Optional system name filter
        user_id: User ID for access control
        
    Returns:
        Dict with topology JSON and metadata
    """
    try:
        from orchestrator.backend_tool_client import get_backend_tool_client
        
        client = await get_backend_tool_client()
        await client.connect()
        
        from protos import tool_service_pb2
        
        request = tool_service_pb2.GetSystemTopologyRequest(
            user_id=user_id
        )
        
        if system_name:
            request.system_name = system_name
        
        response = await client._stub.GetSystemTopology(request)
        formatted = f"Topology: {response.component_count} components, {response.edge_count} edges." if response.success else (response.error or "Failed")
        return {
            "success": response.success,
            "topology_json": response.topology_json,
            "component_count": response.component_count,
            "edge_count": response.edge_count,
            "redundancy_groups": list(response.redundancy_groups),
            "error": response.error if response.HasField("error") else None,
            "formatted": formatted
        }
        
    except Exception as e:
        logger.error(f"Failed to get system topology: {e}")
        err = str(e)
        return {
            "success": False,
            "error": err,
            "topology_json": "{}",
            "component_count": 0,
            "edge_count": 0,
            "redundancy_groups": [],
            "formatted": f"Failed to get topology: {err}"
        }


register_action(
    name="design_system_component",
    category="system_modeling",
    description="Design/add a system component to the topology",
    inputs_model=DesignSystemComponentInputs,
    outputs_model=DesignSystemComponentOutputs,
    tool_function=design_system_component_tool,
)
register_action(
    name="simulate_system_failure",
    category="system_modeling",
    description="Simulate system failure with cascade propagation",
    inputs_model=SimulateSystemFailureInputs,
    outputs_model=SimulateSystemFailureOutputs,
    tool_function=simulate_system_failure_tool,
)
register_action(
    name="get_system_topology",
    category="system_modeling",
    description="Get system topology as JSON",
    inputs_model=GetSystemTopologyInputs,
    outputs_model=GetSystemTopologyOutputs,
    tool_function=get_system_topology_tool,
)
