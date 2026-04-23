"""Tests for deep_agent step formatted output selection (evaluate skip, output_phase, output_template)."""

import re
import unittest

from orchestrator.engines.deep_agent_output_selection import resolve_deep_agent_formatted_output


def _simple_brace_resolve(template: str, phase_results: dict) -> str:
    """Minimal {phase.field} resolver for unit tests (no LangChain / pipeline imports)."""

    def repl(match: re.Match[str]) -> str:
        ref = match.group(1).strip()
        parts = ref.split(".")
        if len(parts) < 2:
            return match.group(0)
        obj: object = phase_results.get(parts[0])
        for part in parts[1:]:
            if not isinstance(obj, dict):
                return match.group(0)
            obj = obj.get(part)
        if obj is None:
            return ""
        return str(obj)

    return re.sub(r"\{([^}]+)\}", repl, template)


class ResolveDeepAgentFormattedOutputTests(unittest.TestCase):
    def test_skips_evaluate_in_default_fallback(self) -> None:
        phases = [
            {"name": "synth", "type": "synthesize"},
            {"name": "qc", "type": "evaluate"},
        ]
        phase_results = {
            "synth": {"output": "THE CRITIQUE"},
            "qc": {"output": '{"score": 1}', "feedback": "ok"},
        }
        formatted, raw = resolve_deep_agent_formatted_output(phases, phase_results, _simple_brace_resolve)
        self.assertEqual(formatted, "THE CRITIQUE")
        self.assertEqual(raw, "THE CRITIQUE")

    def test_honors_output_phase(self) -> None:
        phases = [
            {"name": "plan", "type": "reason"},
            {"name": "draft", "type": "synthesize"},
            {"name": "qc", "type": "evaluate"},
        ]
        phase_results = {
            "plan": {"output": "p"},
            "draft": {"output": "DRAFT BODY"},
            "qc": {"output": '{"pass":true}'},
        }
        formatted, _ = resolve_deep_agent_formatted_output(
            phases, phase_results, _simple_brace_resolve, output_phase="draft"
        )
        self.assertEqual(formatted, "DRAFT BODY")

    def test_output_phase_falls_through_when_phase_missing_or_empty(self) -> None:
        phases = [
            {"name": "a", "type": "reason"},
            {"name": "b", "type": "synthesize"},
        ]
        phase_results = {"a": {"output": ""}, "b": {"output": "BOK"}}
        formatted, _ = resolve_deep_agent_formatted_output(
            phases, phase_results, _simple_brace_resolve, output_phase="ghost"
        )
        self.assertEqual(formatted, "BOK")

        formatted2, _ = resolve_deep_agent_formatted_output(
            phases, phase_results, _simple_brace_resolve, output_phase="a"
        )
        self.assertEqual(formatted2, "BOK")

    def test_output_template_concat(self) -> None:
        phases = [
            {"name": "a", "type": "reason"},
            {"name": "b", "type": "reason"},
        ]
        phase_results = {"a": {"output": "A"}, "b": {"output": "B"}}
        formatted, _ = resolve_deep_agent_formatted_output(
            phases,
            phase_results,
            _simple_brace_resolve,
            output_template="X: {a.output} | Y: {b.output}",
        )
        self.assertEqual(formatted, "X: A | Y: B")

    def test_template_beats_output_phase(self) -> None:
        phases = [
            {"name": "draft", "type": "synthesize"},
            {"name": "qc", "type": "evaluate"},
        ]
        phase_results = {
            "draft": {"output": "DRAFT"},
            "qc": {"output": "JSON"},
        }
        formatted, _ = resolve_deep_agent_formatted_output(
            phases,
            phase_results,
            _simple_brace_resolve,
            output_phase="qc",
            output_template="ONLY {draft.output}",
        )
        self.assertEqual(formatted, "ONLY DRAFT")

    def test_legacy_fallback_when_only_evaluate_has_output(self) -> None:
        phases = [{"name": "qc", "type": "evaluate"}]
        phase_results = {"qc": {"output": '{"pass": true}'}}
        formatted, _ = resolve_deep_agent_formatted_output(phases, phase_results, _simple_brace_resolve)
        self.assertEqual(formatted, '{"pass": true}')


if __name__ == "__main__":
    unittest.main()
