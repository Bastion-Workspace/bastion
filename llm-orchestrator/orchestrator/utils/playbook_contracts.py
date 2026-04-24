"""
Canonical playbook step types and deep-agent phase types.

Keep in sync with backend/services/playbook_contracts.py (same literal sets).
Used by orchestrator tools, validators, and tests.
"""

from typing import FrozenSet

# Playbook step_type values accepted by Agent Factory + pipeline.
VALID_PLAYBOOK_STEP_TYPES: FrozenSet[str] = frozenset(
    {
        "tool",
        "llm_task",
        "llm_agent",
        "approval",
        "loop",
        "parallel",
        "branch",
        "deep_agent",
        "browser_authenticate",
    }
)

# deep_agent phases[].type values supported by the deep agent executor.
VALID_DEEP_AGENT_PHASE_TYPES: FrozenSet[str] = frozenset(
    {
        "reason",
        "act",
        "search",
        "evaluate",
        "synthesize",
        "refine",
        "rerank",
    }
)

# phase_results[phase_name] keys commonly referenced as {phaseName.field} in prompts.
DEEP_AGENT_SISTER_PHASE_OUTPUT_FIELDS: FrozenSet[str] = frozenset(
    {"output", "feedback", "score", "pass"}
)
