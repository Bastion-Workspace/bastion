"""
Plan Engine - executes multi-step compound plans with context bridging.

Runs steps in dependency order (parallel within a level), injects prior step
results into shared_memory, and streams ChatChunks to the client.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional

from protos import orchestrator_pb2

from orchestrator.engines.context_bridge import ContextBridge
from orchestrator.engines.plan_models import ExecutionPlan, PlanStep
from orchestrator.engines.unified_dispatch import get_unified_dispatcher

logger = logging.getLogger(__name__)


def _build_dependency_levels(steps: List[PlanStep]) -> List[List[PlanStep]]:
    """Group steps by dependency level. Level 0 = no deps, level 1 = deps only on 0, etc."""
    step_ids = {s.step_id for s in steps}
    levels: List[List[PlanStep]] = []
    remaining = list(steps)
    while remaining:
        level: List[PlanStep] = []
        completed_ids = {s.step_id for level_steps in levels for s in level_steps}
        for s in remaining[:]:
            if all(dep in completed_ids for dep in s.depends_on):
                level.append(s)
                remaining.remove(s)
        if not level:
            logger.warning("Plan has circular or missing dependencies, treating remaining as level")
            level = remaining
            remaining = []
        levels.append(level)
    return levels


async def _run_step_collect_chunks(
    step: PlanStep,
    metadata: Dict[str, Any],
    messages: List[Any],
    cancellation_token: Optional[Any],
) -> tuple[List[Any], str]:
    """Run a single step via dispatcher; return (list of chunks, aggregated content text)."""
    dispatcher = get_unified_dispatcher()
    chunks: List[Any] = []
    content_parts: List[str] = []
    try:
        async for chunk in dispatcher.dispatch(
            step.skill_name,
            step.sub_query,
            metadata,
            messages,
            cancellation_token,
        ):
            chunks.append(chunk)
            if getattr(chunk, "type", None) == "content" and getattr(chunk, "message", None):
                content_parts.append(chunk.message)
    except Exception as e:
        logger.exception("Plan step %s failed: %s", step.step_id, e)
        chunks.append(
            orchestrator_pb2.ChatChunk(
                type="content",
                message=f"Step failed: {e}",
                timestamp=datetime.now().isoformat(),
                agent_name=f"{step.skill_name}_agent",
            )
        )
        content_parts.append(f"Step failed: {e}")
    content_text = "\n\n".join(content_parts) if content_parts else ""
    return chunks, content_text


class PlanEngine:
    """Executes a compound ExecutionPlan by running steps in order, with context bridging."""

    def __init__(self) -> None:
        self._dispatcher = None

    def _get_dispatcher(self):
        if self._dispatcher is None:
            self._dispatcher = get_unified_dispatcher()
        return self._dispatcher

    async def execute_plan(
        self,
        plan: ExecutionPlan,
        query: str,
        metadata: Dict[str, Any],
        messages: List[Any],
        cancellation_token: Optional[Any] = None,
    ) -> AsyncIterator[Any]:
        """
        Execute a multi-step plan: run steps in dependency order, inject context,
        yield ChatChunk stream (status, content, editor_operations, complete).
        """
        steps = plan.steps
        if not steps:
            yield orchestrator_pb2.ChatChunk(
                type="content",
                message="No steps in plan.",
                timestamp=datetime.now().isoformat(),
                agent_name="system",
            )
            yield orchestrator_pb2.ChatChunk(
                type="complete",
                message="Complete",
                timestamp=datetime.now().isoformat(),
                agent_name="system",
            )
            return

        n_steps = len(steps)
        yield orchestrator_pb2.ChatChunk(
            type="status",
            message=f"Running multi-step plan ({n_steps} steps)...",
            timestamp=datetime.now().isoformat(),
            agent_name="system",
        )
        context_bridge = ContextBridge()
        dispatcher = self._get_dispatcher()
        levels = _build_dependency_levels(steps)

        for level_steps in levels:
            tasks = []
            for step in level_steps:
                logger.info(
                    "Compound plan step %s: skill=%s sub_query=%s",
                    step.step_id,
                    step.skill_name,
                    (step.sub_query or "")[:80],
                )
                step_metadata = dict(metadata)
                shared = dict(step_metadata.get("shared_memory") or {})
                shared.update(context_bridge.build_context_for_step(step))
                step_metadata["shared_memory"] = shared
                tasks.append(
                    _run_step_collect_chunks(step, step_metadata, messages, cancellation_token)
                )
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for step, result in zip(level_steps, results):
                if isinstance(result, Exception):
                    logger.exception("Plan step %s raised: %s", step.step_id, result)
                    yield orchestrator_pb2.ChatChunk(
                        type="content",
                        message=f"Step {step.step_id} failed: {result}",
                        timestamp=datetime.now().isoformat(),
                        agent_name=f"{step.skill_name}_agent",
                    )
                    context_bridge.store_result(
                        step.step_id,
                        {"response": str(result), "agent_type": step.skill_name},
                    )
                    continue
                chunks, content_text = result
                context_bridge.store_result(
                    step.step_id,
                    {"response": content_text, "agent_type": step.skill_name},
                )
                for chunk in chunks:
                    yield chunk

        yield orchestrator_pb2.ChatChunk(
            type="complete",
            message="Complete",
            timestamp=datetime.now().isoformat(),
            agent_name="system",
        )
