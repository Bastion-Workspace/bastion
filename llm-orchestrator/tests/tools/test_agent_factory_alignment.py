"""Regression tests for Agent Factory wiring validation and deep act tool scoping.

Requires full orchestrator dependencies (pydantic, langchain-core, etc.). Skipped in minimal envs.
"""

import unittest


def _has_pydantic() -> bool:
    try:
        import pydantic  # noqa: F401

        return True
    except ImportError:
        return False


def _has_langchain_core() -> bool:
    try:
        import langchain_core  # noqa: F401

        return True
    except ImportError:
        return False


if _has_pydantic():
    from orchestrator.tools.agent_factory_tools import _validate_playbook_steps, _validate_playbook_wiring
else:
    _validate_playbook_wiring = None  # type: ignore[misc, assignment]
    _validate_playbook_steps = None  # type: ignore[misc, assignment]

if _has_langchain_core():
    from orchestrator.engines.deep_agent_executor import _run_act_node
else:
    _run_act_node = None  # type: ignore[misc, assignment]


@unittest.skipUnless(_has_pydantic(), "pydantic required (orchestrator tools import)")
class PlaybookWiringValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.actions = {
            "_noop": {
                "name": "_noop",
                "input_fields": [],
                "output_fields": [{"name": "formatted", "type": "text"}],
            },
        }

    def test_browser_authenticate_step_type_no_wiring_error(self) -> None:
        steps = [
            {
                "name": "Login",
                "step_type": "browser_authenticate",
                "output_key": "browser_auth",
                "inputs": {},
            },
        ]
        issues = _validate_playbook_wiring(steps, self.actions)
        step_type_issues = [i for i in issues if i.get("input_key") == "step_type"]
        self.assertEqual(step_type_issues, [])

    def test_deep_phase_prompt_unknown_phase_ref(self) -> None:
        steps = [
            {
                "name": "Deep",
                "step_type": "deep_agent",
                "output_key": "deep_out",
                "inputs": {},
                "phases": [
                    {
                        "name": "plan",
                        "type": "reason",
                        "prompt": "Use {ghost_phase.output} here.",
                    },
                ],
            },
        ]
        issues = _validate_playbook_wiring(steps, self.actions)
        self.assertTrue(any("ghost_phase" in (i.get("message") or "") for i in issues))

    def test_deep_phase_prompt_bad_sister_field(self) -> None:
        steps = [
            {
                "name": "Deep",
                "step_type": "deep_agent",
                "output_key": "deep_out",
                "inputs": {},
                "phases": [
                    {"name": "a", "type": "reason", "prompt": "x"},
                    {"name": "b", "type": "synthesize", "prompt": "See {a.typo_field}"},
                ],
            },
        ]
        issues = _validate_playbook_wiring(steps, self.actions)
        self.assertTrue(any("typo_field" in (i.get("message") or "") for i in issues))

    def test_deep_output_template_valid_sister_ref(self) -> None:
        steps = [
            {
                "name": "Deep",
                "step_type": "deep_agent",
                "output_key": "deep_out",
                "inputs": {},
                "phases": [
                    {"name": "draft", "type": "synthesize", "prompt": "Write"},
                ],
                "output_template": "Final: {draft.output}",
            },
        ]
        issues = _validate_playbook_wiring(steps, self.actions)
        self.assertEqual(issues, [])

    def test_validate_playbook_steps_accepts_rerank_phase(self) -> None:
        steps = [
            {
                "name": "Deep",
                "step_type": "deep_agent",
                "output_key": "deep_out",
                "phases": [{"name": "rr", "type": "rerank"}],
            },
        ]
        warnings = _validate_playbook_steps(steps)
        self.assertFalse(any("invalid type" in w for w in warnings))


@unittest.skipUnless(_has_pydantic() and _has_langchain_core(), "pydantic + langchain-core required")
class DeepActToolScopeTests(unittest.IsolatedAsyncioTestCase):
    async def test_act_honors_phase_available_tools_subset(self) -> None:
        captured: dict = {}

        async def fake_exec(fake_step, *_a, **_kw):
            captured["available_tools"] = list(fake_step.get("available_tools") or [])
            return {"formatted": "ok", "_token_usage": {"input_tokens": 1, "output_tokens": 1}}

        phase = {"name": "toolphase", "type": "act", "prompt": "hi", "available_tools": ["a", "b"]}
        state = {
            "phase_results": {},
            "phase_trace": [],
            "playbook_state": {},
            "inputs": {},
            "_token_usage": {"input_tokens": 0, "output_tokens": 0},
        }
        tools_map = {"a": (None, None), "b": (None, None), "c": (None, None)}
        await _run_act_node(
            phase,
            state,
            None,
            tools_map,
            lambda t, _ns: t,
            None,
            "user-1",
            None,
            fake_exec,
            step_palette_tools=["a", "b", "c"],
            parent_step_for_policy=None,
        )
        self.assertEqual(sorted(captured["available_tools"]), ["a", "b"])

    async def test_act_inherits_step_palette_when_no_phase_tools(self) -> None:
        captured: dict = {}

        async def fake_exec(fake_step, *_a, **_kw):
            captured["available_tools"] = list(fake_step.get("available_tools") or [])
            return {"formatted": "ok", "_token_usage": {"input_tokens": 0, "output_tokens": 0}}

        phase = {"name": "act1", "type": "act", "prompt": "go"}
        state = {
            "phase_results": {},
            "phase_trace": [],
            "playbook_state": {},
            "inputs": {},
            "_token_usage": {"input_tokens": 0, "output_tokens": 0},
        }
        tools_map = {"x": (None, None), "y": (None, None)}
        await _run_act_node(
            phase,
            state,
            None,
            tools_map,
            lambda t, _ns: t,
            None,
            "u",
            None,
            fake_exec,
            step_palette_tools=["x"],
            parent_step_for_policy=None,
        )
        self.assertEqual(captured["available_tools"], ["x"])


if __name__ == "__main__":
    unittest.main()
