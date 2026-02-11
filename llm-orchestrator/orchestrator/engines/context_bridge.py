"""
Context Bridge - accumulates results across plan steps for injection into subsequent steps.

Used by PlanEngine to pass prior step outputs (response text, structured data) into
dependent steps via shared_memory.
"""

from typing import Any, Dict, List

from orchestrator.engines.plan_models import PlanStep


def _extract_response_text(result: Dict[str, Any]) -> str:
    """Extract response text from engine result (handles nested response dict)."""
    raw = result.get("response", "")
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        return (raw.get("response") or raw.get("message") or "") or ""
    return str(raw) if raw else ""


class ContextBridge:
    """Accumulates results across plan steps for injection into subsequent steps."""

    def __init__(self) -> None:
        self.step_results: Dict[int, Dict[str, Any]] = {}

    def store_result(self, step_id: int, result: Dict[str, Any]) -> None:
        """Store a step's result for use by dependent steps."""
        self.step_results[step_id] = {
            "response_text": _extract_response_text(result),
            "structured_data": result.get("structured_data", {}),
            "skill_name": result.get("agent_type", ""),
        }

    def build_context_for_step(self, step: PlanStep) -> Dict[str, Any]:
        """Build shared_memory injection for a step from its dependencies."""
        context: Dict[str, Any] = {}
        for dep_id in step.depends_on:
            dep_result = self.step_results.get(dep_id, {})
            for key in step.context_keys:
                if key in dep_result:
                    context[f"step_{dep_id}_{key}"] = dep_result[key]
            context[f"prior_step_{dep_id}_response"] = dep_result.get("response_text", "")
        return context
