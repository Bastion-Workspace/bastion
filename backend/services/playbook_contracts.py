"""
Canonical playbook step types and deep-agent phase types.

Keep in sync with llm-orchestrator/orchestrator/utils/playbook_contracts.py (same literal sets).
"""

from typing import FrozenSet

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

DEEP_AGENT_SISTER_PHASE_OUTPUT_FIELDS: FrozenSet[str] = frozenset(
    {"output", "feedback", "score", "pass"}
)
