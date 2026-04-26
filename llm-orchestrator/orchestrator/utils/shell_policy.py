"""Evaluate per-user shell command policy rules (allow / deny / require_approval)."""

import fnmatch
from typing import Any, Dict, List, Optional, Tuple


def evaluate_shell_policy(
    rules: List[Dict[str, Any]],
    command: str,
    workspace_id: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """
    First matching rule wins (rules should be sorted by priority ascending).

    Returns:
        (action, label) where action is allow | deny | require_approval.
    """
    cmd = (command or "").strip()
    if not cmd:
        return "allow", None
    parts = cmd.split()
    first_token = parts[0].rsplit("/", 1)[-1] if parts else ""

    for rule in rules or []:
        if not isinstance(rule, dict):
            continue
        scope = rule.get("scope_workspace_id")
        if scope and str(scope).strip() and workspace_id and str(scope).strip() != str(workspace_id).strip():
            continue
        pattern = (rule.get("pattern") or "").strip()
        if not pattern:
            continue
        mode = str(rule.get("match_mode") or "prefix").strip().lower()
        action = str(rule.get("action") or "allow").strip().lower()
        if action not in ("allow", "deny", "require_approval"):
            action = "allow"

        matched = False
        if mode == "prefix":
            matched = first_token == pattern
        elif mode == "contains":
            matched = pattern in cmd
        elif mode == "glob":
            matched = fnmatch.fnmatch(cmd, pattern)
        else:
            matched = first_token == pattern

        if matched:
            return action, rule.get("label")

    return "allow", None
