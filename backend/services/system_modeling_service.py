"""
Technical Hyperspace - Deterministic system topology and failure simulation
Uses NetworkX for graph processing and failure propagation
"""

import logging
import json
import uuid
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict, deque

import networkx as nx

logger = logging.getLogger(__name__)


class SystemModelingService:
    """
    Deterministic system modeling service using NetworkX
    
    Provides:
    - Component design and topology management
    - Failure simulation with cascade propagation
    - Monte Carlo stress testing
    - Redundancy group analysis
    """
    
    def __init__(self):
        """Initialize system modeling service"""
        # Store topologies per user (in-memory for now, could be persisted)
        self._user_topologies: Dict[str, nx.DiGraph] = {}
        self._user_components: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        self._redundancy_groups: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        logger.info("System Modeling Service initialized")
    
    def _get_user_topology(self, user_id: str) -> nx.DiGraph:
        """Get or create topology graph for user"""
        if user_id not in self._user_topologies:
            self._user_topologies[user_id] = nx.DiGraph()
        return self._user_topologies[user_id]
    
    def design_component(
        self,
        user_id: str,
        component_id: str,
        component_type: str,
        requires: List[str],
        provides: List[str],
        redundancy_group: Optional[str] = None,
        criticality: int = 3,
        metadata: Dict[str, str] = None,
        dependency_logic: str = "AND",
        m_of_n_threshold: int = 0,
        dependency_weights: Dict[str, float] = None,
        integrity_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Design/add a system component to the topology
        
        Args:
            user_id: User ID
            component_id: Unique component identifier
            component_type: Type of component (e.g., "pump", "valve", "sensor")
            requires: List of component IDs this component depends on
            provides: List of functionalities this component enables (for other components)
            redundancy_group: Optional redundancy group name
            criticality: Criticality level 1-5 (5 = most critical)
            metadata: Additional component properties
            dependency_logic: Logic for dependencies ("AND", "OR", "MAJORITY", "M_OF_N", "WEIGHTED_INTEGRITY")
            m_of_n_threshold: Threshold for M_OF_N logic
            dependency_weights: Weights for each dependency
            integrity_threshold: Threshold for WEIGHTED_INTEGRITY logic
            
        Returns:
            Dict with success status and updated topology JSON
        """
        try:
            graph = self._get_user_topology(user_id)
            metadata = metadata or {}
            dependency_weights = dependency_weights or {}
            
            # Add component node with attributes
            graph.add_node(
                component_id,
                component_type=component_type,
                criticality=criticality,
                redundancy_group=redundancy_group,
                state="operational",
                dependency_logic=dependency_logic,
                m_of_n_threshold=m_of_n_threshold,
                dependency_weights=dependency_weights,
                integrity_threshold=integrity_threshold,
                current_integrity=1.0,
                metadata=metadata
            )
            
            # Add dependency edges (requires -> this component)
            for required_id in requires:
                if required_id not in graph:
                    # Create placeholder for missing dependency
                    graph.add_node(required_id, component_type="unknown", state="unknown")
                graph.add_edge(required_id, component_id, dependency_type="requires")
            
            # Add provision edges (this component -> provides)
            # Note: "provides" are functionalities, not component IDs
            # We store them as node attributes for now
            if provides:
                graph.nodes[component_id]["provides"] = provides
            
            # Track redundancy groups
            if redundancy_group:
                self._redundancy_groups[user_id][redundancy_group].add(component_id)
            
            # Store component details
            self._user_components[user_id][component_id] = {
                "component_type": component_type,
                "requires": requires,
                "provides": provides,
                "redundancy_group": redundancy_group,
                "criticality": criticality,
                "metadata": metadata
            }
            
            # Serialize topology to JSON
            topology_json = self._serialize_topology(graph)
            
            logger.info(f"Designed component {component_id} for user {user_id}")
            
            return {
                "success": True,
                "component_id": component_id,
                "message": f"Component {component_id} added to topology",
                "topology_json": topology_json
            }
            
        except Exception as e:
            logger.error(f"Failed to design component: {e}")
            return {
                "success": False,
                "component_id": component_id,
                "message": f"Failed to add component: {str(e)}",
                "error": str(e),
                "topology_json": "{}"
            }
    
    def simulate_failure(
        self,
        user_id: str,
        failed_component_ids: List[str],
        failure_modes: List[str],
        simulation_type: str = "cascade",
        monte_carlo_iterations: Optional[int] = None,
        failure_parameters: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """
        Simulate system failure with deterministic cascade propagation
        
        Args:
            user_id: User ID
            failed_component_ids: Components to fail initially
            failure_modes: Types of failures ("sabotage", "maintenance_neglect", "wear", "random")
            simulation_type: "single", "cascade", or "monte_carlo"
            monte_carlo_iterations: Number of iterations for Monte Carlo simulation
            failure_parameters: Additional simulation parameters
            
        Returns:
            Dict with simulation results including component states, failure paths, and health metrics
        """
        try:
            graph = self._get_user_topology(user_id)
            
            if len(graph.nodes) == 0:
                return {
                    "success": False,
                    "simulation_id": str(uuid.uuid4()),
                    "error": "No components in topology",
                    "component_states": [],
                    "failure_paths": [],
                    "health_metrics": {},
                    "topology_json": "{}"
                }
            
            simulation_id = str(uuid.uuid4())
            failure_parameters = failure_parameters or {}
            
            # Reset all components to operational
            for node in graph.nodes:
                graph.nodes[node]["state"] = "operational"
                graph.nodes[node]["failed_dependencies"] = []
            
            # Apply initial failures
            failed_set = set(failed_component_ids)
            for component_id in failed_component_ids:
                if component_id in graph:
                    graph.nodes[component_id]["state"] = "failed"
                    graph.nodes[component_id]["failure_mode"] = failure_modes[0] if failure_modes else "random"
            
            # Propagate failures based on simulation type
            if simulation_type == "cascade":
                self._propagate_cascade_failures(graph, failed_set)
            elif simulation_type == "monte_carlo" and monte_carlo_iterations:
                return self._monte_carlo_simulation(
                    graph, failed_component_ids, failure_modes, monte_carlo_iterations, failure_parameters
                )
            
            # Analyze redundancy groups
            self._analyze_redundancy_groups(user_id, graph)
            
            # Build component states
            component_states = []
            for node_id in graph.nodes:
                node_data = graph.nodes[node_id]
                component_states.append({
                    "component_id": node_id,
                    "state": node_data.get("state", "operational"),
                    "failed_dependencies": node_data.get("failed_dependencies", []),
                    "failure_probability": node_data.get("failure_probability", 0.0),
                    "metadata": node_data.get("metadata", {})
                })
            
            # Build failure paths
            failure_paths = self._trace_failure_paths(graph, failed_component_ids)
            
            # Calculate health metrics
            health_metrics = self._calculate_health_metrics(graph, user_id)
            
            # Serialize updated topology
            topology_json = self._serialize_topology(graph)
            
            logger.info(f"Simulated failure for user {user_id}: {len(failed_component_ids)} initial failures, {health_metrics['failed_components']} total failures")
            
            return {
                "success": True,
                "simulation_id": simulation_id,
                "component_states": component_states,
                "failure_paths": failure_paths,
                "health_metrics": health_metrics,
                "topology_json": topology_json
            }
            
        except Exception as e:
            logger.error(f"Failed to simulate failure: {e}")
            return {
                "success": False,
                "simulation_id": str(uuid.uuid4()),
                "error": str(e),
                "component_states": [],
                "failure_paths": [],
                "health_metrics": {},
                "topology_json": "{}"
            }
    
    def _propagate_cascade_failures(self, graph: nx.DiGraph, initial_failed: Set[str]):
        """Propagate failures through dependency graph based on logic gates"""
        queue = deque(initial_failed)
        visited = set(initial_failed)
        
        while queue:
            failed_id = queue.popleft()
            
            # Find all components that depend on this failed component
            for dependent_id in graph.successors(failed_id):
                if dependent_id in visited:
                    continue
                
                node_data = graph.nodes[dependent_id]
                logic = node_data.get("dependency_logic", "AND")
                m_of_n = node_data.get("m_of_n_threshold", 0)
                
                predecessors = list(graph.predecessors(dependent_id))
                if not predecessors:
                    continue
                
                failed_preds = [p for p in predecessors if graph.nodes[p]["state"] == "failed"]
                num_failed = len(failed_preds)
                total_preds = len(predecessors)
                
                is_failed = False
                current_integrity = 1.0
                
                if logic == "OR":
                    # If ANY critical dependency fails, the component fails
                    is_failed = num_failed > 0
                elif logic == "MAJORITY":
                    # If > 50% fail
                    is_failed = num_failed > (total_preds / 2)
                elif logic == "M_OF_N":
                    # If at least M fail
                    is_failed = num_failed >= m_of_n
                elif logic == "WEIGHTED_INTEGRITY":
                    # Calculate current integrity based on weights of operational dependencies
                    weights = node_data.get("dependency_weights", {})
                    threshold = node_data.get("integrity_threshold", 0.5)
                    
                    # Contribution of failed dependencies
                    integrity_loss = 0.0
                    for p in failed_preds:
                        integrity_loss += weights.get(p, 1.0 / total_preds)
                    
                    current_integrity = 1.0 - integrity_loss
                    is_failed = current_integrity < threshold
                    graph.nodes[dependent_id]["current_integrity"] = current_integrity
                else: # Default: AND
                    # All must fail
                    is_failed = num_failed == total_preds
                
                if is_failed:
                    graph.nodes[dependent_id]["state"] = "failed"
                    graph.nodes[dependent_id]["failed_dependencies"] = failed_preds
                    visited.add(dependent_id)
                    queue.append(dependent_id)
                elif num_failed > 0:
                    # Mark as degraded if some dependencies failed but logic gate not tripped
                    graph.nodes[dependent_id]["state"] = "degraded"
                    graph.nodes[dependent_id]["failed_dependencies"] = failed_preds
                    graph.nodes[dependent_id]["current_integrity"] = current_integrity
    
    def _monte_carlo_simulation(
        self,
        graph: nx.DiGraph,
        initial_failures: List[str],
        failure_modes: List[str],
        iterations: int,
        failure_parameters: Dict[str, str]
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation for probabilistic failure analysis"""
        import random
        
        failure_counts = defaultdict(int)
        total_failures = 0
        
        for _ in range(iterations):
            # Reset graph
            for node in graph.nodes:
                graph.nodes[node]["state"] = "operational"
            
            # Apply initial failures with probability
            initial_failed = set()
            for component_id in initial_failures:
                if component_id in graph:
                    # Use failure probability from parameters or default
                    prob = float(failure_parameters.get(f"{component_id}_prob", "0.5"))
                    if random.random() < prob:
                        graph.nodes[component_id]["state"] = "failed"
                        initial_failed.add(component_id)
            
            # Propagate cascade
            self._propagate_cascade_failures(graph, initial_failed)
            
            # Count failures
            for node_id in graph.nodes:
                if graph.nodes[node_id]["state"] == "failed":
                    failure_counts[node_id] += 1
                    total_failures += 1
        
        # Calculate failure probabilities
        for node_id in graph.nodes:
            graph.nodes[node_id]["failure_probability"] = failure_counts[node_id] / iterations
        
        # Build results (use average state across iterations)
        component_states = []
        for node_id in graph.nodes:
            prob = graph.nodes[node_id]["failure_probability"]
            state = "failed" if prob > 0.5 else ("degraded" if prob > 0.1 else "operational")
            component_states.append({
                "component_id": node_id,
                "state": state,
                "failed_dependencies": [],
                "failure_probability": prob,
                "metadata": graph.nodes[node_id].get("metadata", {})
            })
        
        # Build failure paths (most common paths)
        failure_paths = self._trace_failure_paths(graph, initial_failures)
        
        # Calculate health metrics
        health_metrics = {
            "total_components": len(graph.nodes),
            "operational_components": sum(1 for s in component_states if s["state"] == "operational"),
            "degraded_components": sum(1 for s in component_states if s["state"] == "degraded"),
            "failed_components": sum(1 for s in component_states if s["state"] == "failed"),
            "system_health_score": 1.0 - (total_failures / (iterations * len(graph.nodes))),
            "critical_vulnerabilities": [],
            "redundancy_groups_at_risk": []
        }
        
        topology_json = self._serialize_topology(graph)
        
        return {
            "success": True,
            "simulation_id": str(uuid.uuid4()),
            "component_states": component_states,
            "failure_paths": failure_paths,
            "health_metrics": health_metrics,
            "topology_json": topology_json
        }
    
    def _analyze_redundancy_groups(self, user_id: str, graph: nx.DiGraph):
        """Analyze redundancy groups for N+1 failure scenarios"""
        for group_name, component_set in self._redundancy_groups[user_id].items():
            operational_count = sum(
                1 for comp_id in component_set
                if comp_id in graph and graph.nodes[comp_id].get("state") == "operational"
            )
            total_count = len(component_set)
            
            # Mark group as at risk if N-1 or fewer operational
            if operational_count <= total_count - 1:
                for comp_id in component_set:
                    if comp_id in graph:
                        graph.nodes[comp_id]["redundancy_at_risk"] = True
    
    def _trace_failure_paths(self, graph: nx.DiGraph, initial_failures: List[str]) -> List[Dict[str, Any]]:
        """Trace paths of cascading failures"""
        paths = []
        
        for initial_id in initial_failures:
            if initial_id not in graph:
                continue
            
            # BFS to find all affected components
            queue = deque([(initial_id, [initial_id])])
            visited = {initial_id}
            
            while queue:
                current_id, path = queue.popleft()
                
                # Find dependents
                for dependent_id in graph.successors(current_id):
                    if dependent_id in visited:
                        continue
                    
                    if graph.nodes[dependent_id].get("state") in ["failed", "degraded"]:
                        new_path = path + [dependent_id]
                        visited.add(dependent_id)
                        queue.append((dependent_id, new_path))
                        
                        # Record failure path
                        failure_type = "cascade" if len(new_path) > 2 else "direct"
                        paths.append({
                            "source_component_id": initial_id,
                            "affected_component_ids": new_path[1:],
                            "failure_type": failure_type,
                            "path_length": len(new_path) - 1
                        })
        
        return paths
    
    def _calculate_health_metrics(
        self,
        graph: nx.DiGraph,
        user_id: str
    ) -> Dict[str, Any]:
        """Calculate overall system health metrics"""
        total = len(graph.nodes)
        operational = sum(1 for n in graph.nodes if graph.nodes[n].get("state") == "operational")
        degraded = sum(1 for n in graph.nodes if graph.nodes[n].get("state") == "degraded")
        failed = sum(1 for n in graph.nodes if graph.nodes[n].get("state") == "failed")
        
        health_score = operational / total if total > 0 else 0.0
        
        # Find single points of failure (components with no redundancy)
        critical_vulnerabilities = []
        for node_id in graph.nodes:
            node_data = graph.nodes[node_id]
            redundancy_group = node_data.get("redundancy_group")
            if not redundancy_group:
                # Check if this component has dependents
                if len(list(graph.successors(node_id))) > 0:
                    critical_vulnerabilities.append(node_id)
        
        # Find redundancy groups at risk
        redundancy_groups_at_risk = []
        for group_name, component_set in self._redundancy_groups[user_id].items():
            operational_count = sum(
                1 for comp_id in component_set
                if comp_id in graph and graph.nodes[comp_id].get("state") == "operational"
            )
            if operational_count <= len(component_set) - 1:
                redundancy_groups_at_risk.append(group_name)
        
        return {
            "total_components": total,
            "operational_components": operational,
            "degraded_components": degraded,
            "failed_components": failed,
            "system_health_score": health_score,
            "critical_vulnerabilities": critical_vulnerabilities,
            "redundancy_groups_at_risk": redundancy_groups_at_risk
        }
    
    def get_topology(self, user_id: str, system_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get system topology as JSON
        
        Args:
            user_id: User ID
            system_name: Optional system name filter (not used in current implementation)
            
        Returns:
            Dict with topology JSON and metadata
        """
        try:
            graph = self._get_user_topology(user_id)
            
            topology_json = self._serialize_topology(graph)
            
            # Get redundancy groups
            redundancy_groups = list(self._redundancy_groups[user_id].keys())
            
            return {
                "success": True,
                "topology_json": topology_json,
                "component_count": len(graph.nodes),
                "edge_count": len(graph.edges),
                "redundancy_groups": redundancy_groups
            }
            
        except Exception as e:
            logger.error(f"Failed to get topology: {e}")
            return {
                "success": False,
                "error": str(e),
                "topology_json": "{}",
                "component_count": 0,
                "edge_count": 0,
                "redundancy_groups": []
            }
    
    def _serialize_topology(self, graph: nx.DiGraph) -> str:
        """Serialize NetworkX graph to JSON"""
        try:
            # Use node_link_data format for NetworkX compatibility
            data = nx.node_link_data(graph)
            return json.dumps(data, indent=2)
        except Exception as e:
            logger.error(f"Failed to serialize topology: {e}")
            return "{}"
