"""Analysis domain operations for gRPC (charts, text metrics, system modeling) — no protobuf."""

import json
import uuid
from typing import Any, Dict, List, Optional


async def create_chart_operation(
    *,
    chart_type: str,
    data_json: str,
    title: str,
    x_label: str,
    y_label: str,
    interactive: bool,
    color_scheme: str,
    width: int,
    height: int,
    include_static: bool,
) -> Dict[str, Any]:
    """Parse chart data JSON and invoke visualization service."""
    from services.visualization_service import create_chart

    try:
        data = json.loads(data_json)
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Invalid JSON data: {e}"}

    return await create_chart(
        chart_type=chart_type,
        data=data,
        title=title,
        x_label=x_label,
        y_label=y_label,
        interactive=interactive,
        color_scheme=color_scheme or "plotly",
        width=width if width > 0 else 800,
        height=height if height > 0 else 600,
        include_static=include_static,
    )


def analyze_text_operation(*, content: str, include_advanced: bool) -> Dict[str, Any]:
    """Run FileAnalysisService and return metrics + metadata_json string."""
    from tools_service.services.file_analysis_service import FileAnalysisService

    analysis_service = FileAnalysisService()
    metrics = analysis_service.analyze_text(
        content=content,
        include_advanced=include_advanced,
    )
    metadata_json = json.dumps({"analysis_timestamp": None})
    return {"metrics": metrics, "metadata_json": metadata_json}


def design_system_component_operation(
    *,
    user_id: str,
    component_id: str,
    component_type: str,
    requires: List[str],
    provides: List[str],
    redundancy_group: Optional[str],
    criticality: str,
    metadata: Dict[str, Any],
    dependency_logic: str,
    m_of_n_threshold: int,
    dependency_weights: Dict[str, float],
    integrity_threshold: float,
) -> Dict[str, Any]:
    from services.system_modeling_service import SystemModelingService

    service = SystemModelingService()
    return service.design_component(
        user_id=user_id,
        component_id=component_id,
        component_type=component_type,
        requires=requires,
        provides=provides,
        redundancy_group=redundancy_group,
        criticality=criticality,
        metadata=metadata,
        dependency_logic=dependency_logic or "AND",
        m_of_n_threshold=m_of_n_threshold,
        dependency_weights=dependency_weights,
        integrity_threshold=integrity_threshold if integrity_threshold > 0 else 0.5,
    )


def simulate_system_failure_operation(
    *,
    user_id: str,
    failed_component_ids: List[str],
    failure_modes: List[str],
    simulation_type: str,
    monte_carlo_iterations: Optional[int],
    failure_parameters: Dict[str, Any],
) -> Dict[str, Any]:
    from services.system_modeling_service import SystemModelingService

    service = SystemModelingService()
    return service.simulate_failure(
        user_id=user_id,
        failed_component_ids=failed_component_ids,
        failure_modes=failure_modes,
        simulation_type=simulation_type or "cascade",
        monte_carlo_iterations=monte_carlo_iterations,
        failure_parameters=failure_parameters,
    )


def get_system_topology_operation(
    *, user_id: str, system_name: Optional[str]
) -> Dict[str, Any]:
    from services.system_modeling_service import SystemModelingService

    service = SystemModelingService()
    return service.get_topology(user_id=user_id, system_name=system_name)


def new_simulation_id() -> str:
    return str(uuid.uuid4())
