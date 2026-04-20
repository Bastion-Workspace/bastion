"""
Orchestrator utility helpers for thread isolation (LangGraph checkpointing).
"""

from typing import Optional


def normalize_thread_id(
    user_id: str, conversation_id: str, branch_suffix: Optional[str] = None
) -> str:
    """Produce a namespaced thread_id ensuring per-user isolation.

    Format: "{user_id}:{conversation_id}". If conversation_id already appears namespaced, use as base.
    When branch_suffix is set, append ":branch_{suffix}" for forked transcript checkpoints.
    """
    if not user_id or not conversation_id:
        raise ValueError("normalize_thread_id requires both user_id and conversation_id")
    if ":" in conversation_id and conversation_id.startswith(f"{user_id}:"):
        base = conversation_id
    else:
        base = f"{user_id}:{conversation_id}"
    if branch_suffix:
        return f"{base}:branch_{branch_suffix}"
    return base
