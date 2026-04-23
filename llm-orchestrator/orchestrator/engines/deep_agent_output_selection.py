"""
Pure selection logic for deep_agent step formatted output (no LangGraph / LangChain imports).

Used by deep_agent_executor and unit tests.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple


def resolve_deep_agent_formatted_output(
    phases: List[Dict[str, Any]],
    phase_results: Dict[str, Any],
    resolve_fn: Callable[[str, Dict[str, Any]], str],
    output_phase: Optional[str] = None,
    output_template: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Choose formatted/raw text for a deep_agent step.

    Precedence: output_template > output_phase > smart default (reverse walk, skip evaluate)
    > legacy fallback (reverse walk, include evaluate).

    resolve_fn(template, phase_results) must match deep_agent phase resolution (merged into playbook_state).
    """
    pr = phase_results or {}
    tmpl = (output_template or "").strip()
    if tmpl:
        text = resolve_fn(tmpl, pr)
        return (text, text)

    op = (output_phase or "").strip()
    if op:
        row = pr.get(op) or {}
        if isinstance(row, dict):
            out = row.get("output")
            if out is not None and str(out).strip():
                s = str(out)
                return (s, s)

    phase_names_list = [(p.get("name") or "").strip() for p in phases if (p.get("name") or "").strip()]
    phase_type_by_name: Dict[str, str] = {}
    for p in phases:
        n = (p.get("name") or "").strip()
        if n:
            phase_type_by_name[n] = str(p.get("type") or "").strip().lower()

    def _first_output(names: List[str], skip_evaluate: bool) -> Tuple[str, str]:
        for name in names:
            if skip_evaluate and phase_type_by_name.get(name) == "evaluate":
                continue
            row = pr.get(name) or {}
            if isinstance(row, dict):
                val = row.get("output")
                if val is not None and str(val).strip():
                    s = str(val)
                    return (s, s)
        return ("", "")

    formatted, raw = _first_output(list(reversed(phase_names_list)), skip_evaluate=True)
    if formatted:
        return (formatted, raw)
    formatted, raw = _first_output(list(reversed(phase_names_list)), skip_evaluate=False)
    return (formatted, raw)
