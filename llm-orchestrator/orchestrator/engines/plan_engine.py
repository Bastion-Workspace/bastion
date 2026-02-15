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
from orchestrator.engines.fragment_registry import invoke_fragment
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


async def _run_fragment_step_collect_chunks(
    step: PlanStep,
    metadata: Dict[str, Any],
    messages: List[Any],
    prior_context: Dict[str, Any],
) -> tuple[List[Any], str]:
    """Run a fragment step via invoke_fragment; return (list of chunks, content text)."""
    try:
        result = await invoke_fragment(
            step.fragment_name,
            step.sub_query or "",
            metadata,
            messages,
            prior_context,
        )
        content_text = result.get("response", "")
        chunk = orchestrator_pb2.ChatChunk(
            type="content",
            message=content_text,
            timestamp=datetime.now().isoformat(),
            agent_name=f"{step.fragment_name}_fragment",
        )
        chunks: List[Any] = [chunk]

        # If the fragment produced structured images, include them in a step-level complete chunk.
        # The frontend expects images in ChatChunk.metadata["images"] as a JSON string.
        structured_images = result.get("structured_images")
        if structured_images:
            import json

            step_metadata = {"images": json.dumps(structured_images)}
            chunks.append(
                orchestrator_pb2.ChatChunk(
                    type="complete",
                    message="Complete",
                    timestamp=datetime.now().isoformat(),
                    agent_name=f"{step.fragment_name}_fragment",
                    metadata=step_metadata,
                )
            )

        return chunks, content_text
    except Exception as e:
        logger.exception("Fragment step %s failed: %s", step.step_id, e)
        chunk = orchestrator_pb2.ChatChunk(
            type="content",
            message=f"Step failed: {e}",
            timestamp=datetime.now().isoformat(),
            agent_name=f"{step.fragment_name}_fragment",
        )
        return [chunk], f"Step failed: {e}"


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
        accumulated_images: List[Dict[str, Any]] = []

        for level_steps in levels:
            tasks = []
            for step in level_steps:
                step_metadata = dict(metadata)
                prior_ctx = context_bridge.build_context_for_step(step)
                shared = dict(step_metadata.get("shared_memory") or {})
                shared.update(prior_ctx)
                step_metadata["shared_memory"] = shared
                step_metadata["tool_packs"] = step.tool_packs
                logger.info(
                    "Compound plan step %s: skill=%s fragment=%s sub_query=%s",
                    step.step_id,
                    step.skill_name or "(none)",
                    step.fragment_name or "(none)",
                    (step.sub_query or "")[:80],
                )
                if step.fragment_name:
                    tasks.append(
                        _run_fragment_step_collect_chunks(
                            step, step_metadata, messages, prior_ctx
                        )
                    )
                else:
                    tasks.append(
                        _run_step_collect_chunks(
                            step, step_metadata, messages, cancellation_token
                        )
                    )
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for step, result in zip(level_steps, results):
                agent_label = step.fragment_name or step.skill_name
                if isinstance(result, Exception):
                    logger.exception("Plan step %s raised: %s", step.step_id, result)
                    yield orchestrator_pb2.ChatChunk(
                        type="content",
                        message=f"Step {step.step_id} failed: {result}",
                        timestamp=datetime.now().isoformat(),
                        agent_name=f"{agent_label}_agent",
                    )
                    context_bridge.store_result(
                        step.step_id,
                        {"response": str(result), "agent_type": agent_label},
                    )
                    continue
                chunks, content_text = result
                context_bridge.store_result(
                    step.step_id,
                    {"response": content_text, "agent_type": agent_label},
                )
                for chunk in chunks:
                    # Collect structured images from any step-level complete chunks so we can
                    # include them in the final plan-level complete chunk (frontend expects
                    # images on the last complete chunk).
                    try:
                        if (
                            getattr(chunk, "metadata", None)
                            and isinstance(chunk.metadata, dict)
                            and chunk.metadata.get("images")
                        ):
                            import json

                            parsed = json.loads(chunk.metadata.get("images", "[]"))
                            if isinstance(parsed, list):
                                accumulated_images.extend(parsed)
                    except Exception:
                        # Best-effort accumulation; never break streaming.
                        pass
                    yield chunk

        final_metadata = None
        if accumulated_images:
            import json

            final_metadata = {"images": json.dumps(accumulated_images)}

        yield orchestrator_pb2.ChatChunk(
            type="complete",
            message="Complete",
            timestamp=datetime.now().isoformat(),
            agent_name="system",
            metadata=final_metadata,
        )
