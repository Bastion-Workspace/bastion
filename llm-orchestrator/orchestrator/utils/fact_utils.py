"""
Canonical formatter for user facts in prompts. Used by custom_agent_runner
when consuming facts from GetUserFacts (or metadata). Keeps formatting consistent with backend.
"""
from datetime import datetime, timezone
from typing import Any, List


def format_user_facts_for_prompt(facts: List[dict]) -> str:
    """Format facts for LLM system prompt: filter expired, sort by confidence DESC, return 'USER FACTS:\\n- key: value'."""
    if not facts:
        return ""
    now = datetime.now(timezone.utc)
    valid = []
    for f in facts:
        exp = f.get("expires_at")
        if exp is not None:
            if hasattr(exp, "tzinfo") and exp.tzinfo is None:
                try:
                    exp = exp.replace(tzinfo=timezone.utc)
                except TypeError:
                    valid.append(f)
                    continue
            elif isinstance(exp, str):
                try:
                    exp = datetime.fromisoformat(exp.replace("Z", "+00:00"))
                except ValueError:
                    valid.append(f)
                    continue
            if exp < now:
                continue
        valid.append(f)
    if not valid:
        return ""
    sorted_facts = sorted(valid, key=lambda x: (-(x.get("confidence") or 1.0), x.get("fact_key", "")))
    lines = [f"- {f.get('fact_key', '')}: {f.get('value', '')}" for f in sorted_facts]
    return "USER FACTS:\n" + "\n".join(lines)
